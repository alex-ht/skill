from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_pi import (  # noqa: E402
    build_pi_command,
    extract_pi_session_id,
    extract_pi_usage,
    map_pi_thinking,
    parse_pi_jsonl,
    pi_events_to_transcript,
)


class PiCommandTests(unittest.TestCase):
    def test_build_pi_command_includes_json_print_and_session(self) -> None:
        cmd = build_pi_command(
            prompt="List files",
            model_id="openrouter/anthropic/claude-sonnet-4",
            session_id="sess-1",
            session_dir=Path("/tmp/pi-sessions"),
            thinking_level="adaptive",
            api_key="sk-test",
        )
        self.assertEqual(cmd[0], "pi")
        self.assertIn("-p", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "json")
        self.assertIn("--approve", cmd)
        self.assertEqual(cmd[cmd.index("--session-id") + 1], "sess-1")
        self.assertEqual(cmd[cmd.index("--session-dir") + 1], "/tmp/pi-sessions")
        self.assertEqual(cmd[cmd.index("--model") + 1], "openrouter/anthropic/claude-sonnet-4")
        self.assertEqual(cmd[cmd.index("--thinking") + 1], "max")
        self.assertEqual(cmd[cmd.index("--api-key") + 1], "sk-test")
        self.assertEqual(cmd[-1], "List files")

    def test_map_pi_thinking_passthrough(self) -> None:
        self.assertEqual(map_pi_thinking("high"), "high")
        self.assertIsNone(map_pi_thinking(None))


class PiTranscriptTests(unittest.TestCase):
    def test_parse_pi_jsonl_skips_banners(self) -> None:
        raw = "\n".join(
            [
                "Starting pi...",
                '{"type":"session","id":"abc","cwd":"/tmp"}',
                "not json",
                '{"type":"agent_start"}',
            ]
        )
        events = parse_pi_jsonl(raw)
        self.assertEqual([e["type"] for e in events], ["session", "agent_start"])
        self.assertEqual(extract_pi_session_id(events), "abc")

    def test_pi_events_to_transcript_maps_tools_and_text(self) -> None:
        events = [
            {"type": "session", "id": "abc"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will write the file."},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "write",
                            "arguments": {"path": "out.txt", "content": "hi"},
                        },
                    ],
                    "usage": {
                        "input": 10,
                        "output": 4,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 14,
                        "cost": {"total": 0.01},
                    },
                },
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "call-1",
                "toolName": "write",
                "result": {"content": [{"type": "text", "text": "wrote out.txt"}]},
                "isError": False,
            },
        ]
        transcript = pi_events_to_transcript(events, "Write out.txt")
        self.assertEqual(transcript[0]["message"]["role"], "user")
        self.assertEqual(transcript[0]["message"]["content"], ["Write out.txt"])
        assistant = transcript[1]["message"]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0]["type"], "text")
        self.assertEqual(assistant["content"][1]["type"], "toolCall")
        self.assertEqual(assistant["content"][1]["name"], "write")
        tool_result = transcript[2]["message"]
        self.assertEqual(tool_result["role"], "toolResult")
        self.assertEqual(tool_result["toolName"], "write")

        usage = extract_pi_usage(transcript)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 4)
        self.assertEqual(usage["request_count"], 1)
        self.assertAlmostEqual(usage["cost_usd"], 0.01)

    def test_duplicate_user_prompt_from_session_replay_is_skipped(self) -> None:
        events = [
            {
                "type": "message_end",
                "message": {"role": "user", "content": "Write out.txt"},
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {},
                },
            },
        ]
        transcript = pi_events_to_transcript(events, "Write out.txt")
        roles = [entry["message"]["role"] for entry in transcript]
        self.assertEqual(roles, ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
