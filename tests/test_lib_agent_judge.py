from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import (  # noqa: E402
    _JUDGE_SYSTEM_MSG,
    _extract_grok_cli_text,
    call_judge_api,
)


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": '{"total": 1.0}'}}]}
        ).encode("utf-8")


class GrokJudgeTests(unittest.TestCase):
    def test_extract_grok_cli_text_plain(self) -> None:
        raw = '{"scores": {"accuracy": 0.8}, "total": 0.8, "notes": "ok"}'
        self.assertEqual(_extract_grok_cli_text(raw), raw)

    def test_extract_grok_cli_text_json_envelope(self) -> None:
        payload = {
            "text": '{"total": 0.5, "scores": {"x": 0.5}, "notes": "mid"}',
            "usage": {"input_tokens": 10},
        }
        self.assertEqual(
            _extract_grok_cli_text(json.dumps(payload)),
            payload["text"],
        )

    def test_extract_grok_cli_text_prefers_structured_output(self) -> None:
        payload = {
            "text": "ignored",
            "structuredOutput": {"total": 1.0, "scores": {"ok": 1.0}, "notes": "n"},
        }
        extracted = _extract_grok_cli_text(json.dumps(payload))
        self.assertEqual(
            json.loads(extracted),
            {"total": 1.0, "scores": {"ok": 1.0}, "notes": "n"},
        )

    def test_call_judge_api_grok_runs_cli_with_expected_flags(self) -> None:
        captured: dict = {}

        def fake_run(cmd, capture_output, text, timeout, check):
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            prompt_path = cmd[cmd.index("--prompt-file") + 1]
            captured["prompt"] = Path(prompt_path).read_text(encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                stdout='{"scores": {"accuracy": 1.0}, "total": 1.0, "notes": "good"}',
                stderr="",
            )

        with patch("lib_agent.subprocess.run", side_effect=fake_run):
            result = call_judge_api(
                prompt="grade this transcript",
                model="grok:grok-4.5",
                timeout_seconds=42.0,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["text"],
            '{"scores": {"accuracy": 1.0}, "total": 1.0, "notes": "good"}',
        )
        self.assertEqual(captured["timeout"], 42.0)
        self.assertEqual(captured["prompt"], "grade this transcript")
        cmd = captured["cmd"]
        self.assertEqual(cmd[0], "grok")
        self.assertIn("--prompt-file", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "plain")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertIn("--no-subagents", cmd)
        self.assertIn("--disable-web-search", cmd)
        self.assertEqual(cmd[cmd.index("--system-prompt-override") + 1], _JUDGE_SYSTEM_MSG)
        self.assertIn("--verbatim", cmd)
        self.assertIn("--no-memory", cmd)
        self.assertIn("--no-plan", cmd)
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "1")
        self.assertEqual(cmd[cmd.index("--model") + 1], "grok-4.5")
        # Temp prompt file should be cleaned up after the call
        prompt_path = cmd[cmd.index("--prompt-file") + 1]
        self.assertFalse(Path(prompt_path).exists())

    def test_call_judge_api_grok_missing_cli(self) -> None:
        with patch("lib_agent.subprocess.run", side_effect=FileNotFoundError):
            result = call_judge_api(prompt="grade this", model="grok")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "grok CLI not found")

    def test_call_judge_api_grok_timeout(self) -> None:
        with patch(
            "lib_agent.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="grok", timeout=1),
        ):
            result = call_judge_api(prompt="grade this", model="grok")

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["error"], "grok timed out")

    def test_call_judge_api_grok_non_zero_exit(self) -> None:
        with patch(
            "lib_agent.subprocess.run",
            return_value=SimpleNamespace(
                returncode=2,
                stdout="",
                stderr="not logged in",
            ),
        ):
            result = call_judge_api(prompt="grade this", model="grok")

        self.assertEqual(result["status"], "error")
        self.assertIn("grok exit 2", result["error"])
        self.assertIn("not logged in", result["error"])

    def test_call_judge_api_grok_dispatch_does_not_fall_back_to_openrouter(self) -> None:
        with patch(
            "lib_agent._judge_via_grok_cli",
            return_value={"status": "success", "text": "ok"},
        ) as grok_cli, patch(
            "lib_agent._judge_via_openrouter",
            return_value={"status": "error", "text": "", "error": "should not call"},
        ) as openrouter:
            result = call_judge_api(prompt="grade this", model="grok")

        self.assertEqual(result, {"status": "success", "text": "ok"})
        grok_cli.assert_called_once()
        openrouter.assert_not_called()


class KiloJudgeTests(unittest.TestCase):
    def test_call_judge_api_kilo_requires_kilo_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = call_judge_api(
                prompt="grade this",
                model="kilo/anthropic/claude-sonnet-4-5",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["error"], "KILO_API_KEY not set")

    def test_call_judge_api_kilo_posts_to_gateway_with_bare_model(self) -> None:
        captured_request = None

        def fake_urlopen(req, timeout):
            nonlocal captured_request
            captured_request = req
            self.assertEqual(timeout, 12.5)
            return _FakeResponse()

        with patch.dict(os.environ, {"KILO_API_KEY": "test-key"}, clear=True), patch(
            "lib_agent.request.urlopen", side_effect=fake_urlopen
        ):
            result = call_judge_api(
                prompt="grade this",
                model="kilo/anthropic/claude-sonnet-4-5",
                timeout_seconds=12.5,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["text"], '{"total": 1.0}')
        self.assertIsNotNone(captured_request)
        self.assertEqual(
            captured_request.full_url,
            "https://api.kilo.ai/api/gateway/chat/completions",
        )
        self.assertEqual(captured_request.get_method(), "POST")
        self.assertEqual(captured_request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(captured_request.headers["Content-type"], "application/json")

        payload = json.loads(captured_request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "anthropic/claude-sonnet-4-5")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_completion_tokens"], 2048)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "grade this"})

    def test_call_judge_api_kilo_dispatch_does_not_fall_back_to_openrouter(self) -> None:
        with patch.dict(os.environ, {"KILO_API_KEY": "test-key"}, clear=True), patch(
            "lib_agent._judge_via_openai_compat",
            return_value={"status": "success", "text": "ok"},
        ) as compat:
            result = call_judge_api(
                prompt="grade this",
                model="kilo/openai/gpt-4o",
                timeout_seconds=30,
            )

        self.assertEqual(result, {"status": "success", "text": "ok"})
        compat.assert_called_once_with(
            "grade this",
            "openai/gpt-4o",
            "https://api.kilo.ai/api/gateway/chat/completions",
            "test-key",
            30,
        )


if __name__ == "__main__":
    unittest.main()
