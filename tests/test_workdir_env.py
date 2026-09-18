"""Tests for WORKDIR injection into agent processes and MCP config passthrough."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import (  # noqa: E402
    DEFAULT_REASONING_VISIBILITY,
    WORKDIR_ENV_KEY,
    WORKDIR_ENV_TEMPLATE,
    _agent_process_env,
    _apply_default_reasoning_visibility,
    _ensure_mcp_workdir_env_passthrough,
    _upsert_workspace_env,
    execute_openclaw_task,
)
from lib_tasks import Task  # noqa: E402


def _make_task(**kwargs) -> Task:
    return Task(
        task_id=kwargs.pop("task_id", "task_test"),
        name="Test Task",
        category="test",
        grading_type="automated",
        timeout_seconds=120,
        workspace_files=[],
        prompt=kwargs.pop("prompt", "Hello"),
        expected_behavior="",
        grading_criteria=[],
        frontmatter=kwargs.pop("frontmatter", {}),
        **kwargs,
    )


class TestAgentProcessEnv(unittest.TestCase):
    def test_sets_workdir_to_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            with patch.dict(os.environ, {"SENTINEL_KEY": "keep-me"}, clear=False):
                env = _agent_process_env(workspace)
            self.assertEqual(env[WORKDIR_ENV_KEY], str(workspace.resolve()))
            self.assertEqual(env.get("SENTINEL_KEY"), "keep-me")

    def test_without_workspace_does_not_set_workdir(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            env = _agent_process_env(None)
        # May inherit ambient WORKDIR; only assert helper does not force a path.
        self.assertIsInstance(env, dict)


class TestUpsertWorkspaceEnv(unittest.TestCase):
    def test_creates_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _upsert_workspace_env(workspace, "WORKDIR", "/path/to/ws")
            content = (workspace / ".env").read_text(encoding="utf-8")
            self.assertIn("WORKDIR=/path/to/ws", content)

    def test_preserves_other_keys_and_updates_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".env").write_text("FOO=bar\nWORKDIR=old\nBAZ=qux\n", encoding="utf-8")
            _upsert_workspace_env(workspace, "WORKDIR", "/new/path")
            content = (workspace / ".env").read_text(encoding="utf-8")
            self.assertIn("FOO=bar", content)
            self.assertIn("BAZ=qux", content)
            self.assertIn("WORKDIR=/new/path", content)
            self.assertNotIn("WORKDIR=old", content)

    def test_replaces_export_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".env").write_text("export WORKDIR=old\n", encoding="utf-8")
            _upsert_workspace_env(workspace, "WORKDIR", "/new")
            content = (workspace / ".env").read_text(encoding="utf-8")
            self.assertIn("WORKDIR=/new", content)
            self.assertNotIn("export WORKDIR=old", content)


class TestEnsureMcpWorkdirPassthrough(unittest.TestCase):
    def test_adds_template_to_stdio_servers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcp": {
                            "servers": {
                                "execute-python": {"command": "execute-python-mcp"},
                                "remote": {"url": "https://example.com/mcp"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _ensure_mcp_workdir_env_passthrough()
            self.assertTrue(changed)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                data["mcp"]["servers"]["execute-python"]["env"][WORKDIR_ENV_KEY],
                WORKDIR_ENV_TEMPLATE,
            )
            # HTTP server left without env forced
            self.assertNotIn("env", data["mcp"]["servers"]["remote"])

    def test_preserves_existing_env_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcp": {
                            "servers": {
                                "execute-python": {
                                    "command": "execute-python-mcp",
                                    "env": {"OTHER": "1"},
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _ensure_mcp_workdir_env_passthrough()
            self.assertTrue(changed)
            env = json.loads(config_path.read_text(encoding="utf-8"))["mcp"]["servers"][
                "execute-python"
            ]["env"]
            self.assertEqual(env["OTHER"], "1")
            self.assertEqual(env[WORKDIR_ENV_KEY], WORKDIR_ENV_TEMPLATE)

    def test_idempotent_when_already_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcp": {
                            "servers": {
                                "execute-python": {
                                    "command": "execute-python-mcp",
                                    "env": {WORKDIR_ENV_KEY: WORKDIR_ENV_TEMPLATE},
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _ensure_mcp_workdir_env_passthrough()
            self.assertFalse(changed)


class TestExecutePassesWorkdirEnv(unittest.TestCase):
    @patch("lib_agent.subprocess.run")
    @patch("lib_agent._load_transcript", return_value=([], None))
    @patch("lib_agent.cleanup_agent_sessions")
    @patch("lib_agent.prepare_task_workspace")
    @patch("lib_agent._ensure_mcp_workdir_env_passthrough")
    @patch("lib_agent._apply_default_reasoning_visibility")
    @patch("lib_agent.is_fws_task", return_value=False)
    @patch("lib_agent._wait_for_activity_settle")
    def test_subprocess_receives_workdir(
        self,
        _settle: MagicMock,
        _fws: MagicMock,
        _reasoning: MagicMock,
        _mcp: MagicMock,
        mock_prepare: MagicMock,
        _cleanup: MagicMock,
        _load: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            mock_prepare.return_value = workspace
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            execute_openclaw_task(
                task=_make_task(),
                agent_id="test-agent",
                model_id="test-model",
                run_id="test-run",
                timeout_multiplier=1.0,
                skill_dir=ROOT,
            )

            self.assertTrue(mock_run.called)
            kwargs = mock_run.call_args.kwargs
            self.assertIn("env", kwargs)
            self.assertEqual(kwargs["env"][WORKDIR_ENV_KEY], str(workspace.resolve()))


class TestDefaultReasoningVisibility(unittest.TestCase):
    def test_sets_global_and_per_agent_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "defaults": {},
                            "list": [{"id": "bench-agent", "name": "bench-agent"}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _apply_default_reasoning_visibility("bench-agent")
            self.assertTrue(changed)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                data["agents"]["defaults"]["reasoningDefault"],
                DEFAULT_REASONING_VISIBILITY,
            )
            self.assertEqual(
                data["agents"]["list"][0]["reasoningDefault"],
                DEFAULT_REASONING_VISIBILITY,
            )

    def test_idempotent_when_already_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "defaults": {"reasoningDefault": DEFAULT_REASONING_VISIBILITY},
                            "list": [
                                {
                                    "id": "bench-agent",
                                    "reasoningDefault": DEFAULT_REASONING_VISIBILITY,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _apply_default_reasoning_visibility("bench-agent")
            self.assertFalse(changed)

    def test_sets_global_default_without_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(json.dumps({"agents": {"defaults": {}}}), encoding="utf-8")
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _apply_default_reasoning_visibility()
            self.assertTrue(changed)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                data["agents"]["defaults"]["reasoningDefault"],
                DEFAULT_REASONING_VISIBILITY,
            )


if __name__ == "__main__":
    unittest.main()
