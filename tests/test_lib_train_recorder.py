from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_train_recorder import (  # noqa: E402
    TrainingRecorder,
    extract_transcript_assistant_messages,
    normalize_anthropic_request_body,
    normalize_openai_request_body,
    supports_recording_api,
)


class TrainRecorderNormalizationTests(unittest.TestCase):
    def test_supports_recording_api(self) -> None:
        self.assertTrue(supports_recording_api("openai-completions"))
        self.assertTrue(supports_recording_api("anthropic-messages"))
        self.assertFalse(supports_recording_api("openai-responses"))
        self.assertFalse(supports_recording_api(None))

    def test_normalize_openai_request_body_preserves_tool_arguments_as_string(self) -> None:
        messages, tools = normalize_openai_request_body(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": {"query": "hdfs fsck", "top_k": 3},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "result text",
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "search_docs"}}],
            }
        )

        self.assertEqual(messages[0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(messages[1], {"role": "user", "content": "hi"})
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], '{"query":"hdfs fsck","top_k":3}')
        self.assertEqual(messages[3], {"role": "tool", "tool_call_id": "call_1", "content": "result text"})
        self.assertEqual(tools, [{"type": "function", "function": {"name": "search_docs"}}])

    def test_normalize_anthropic_request_body_maps_tool_use_and_result(self) -> None:
        messages, tools = normalize_anthropic_request_body(
            {
                "system": "system prompt",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "find logs"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Checking."},
                            {"type": "tool_use", "id": "toolu_1", "name": "grep_logs", "input": {"path": "/tmp/app.log"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                            {"type": "text", "text": "continue"},
                        ],
                    },
                ],
                "tools": [{"name": "grep_logs", "input_schema": {"type": "object"}}],
            }
        )

        self.assertEqual(messages[0], {"role": "system", "content": "system prompt"})
        self.assertEqual(messages[1], {"role": "user", "content": "find logs"})
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Checking.")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], '{"path":"/tmp/app.log"}')
        self.assertEqual(messages[3], {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"})
        self.assertEqual(messages[4], {"role": "user", "content": "continue"})
        self.assertEqual(tools, [{"name": "grep_logs", "input_schema": {"type": "object"}}])

    def test_extract_transcript_assistant_messages_normalizes_tool_calls(self) -> None:
        transcript = [
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I can help."},
                        {"type": "toolCall", "id": "call_2", "name": "run_cmd", "arguments": {"cmd": "ls"}},
                    ],
                },
            },
        ]

        assistant_messages = extract_transcript_assistant_messages(transcript)

        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["content"], "I can help.")
        self.assertEqual(assistant_messages[0]["tool_calls"][0]["function"]["arguments"], '{"cmd":"ls"}')


class TrainRecorderFinalizeTests(unittest.TestCase):
    def test_finalize_execution_writes_aligned_prefix_with_task_and_run_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "train.jsonl"
            recorder = TrainingRecorder(output_path)
            recorder.start()
            try:
                recorder.register_execution(
                    benchmark_run_id="0013",
                    task_id="task_log_hdfs_connections",
                    run_index=1,
                    api="openai-completions",
                    upstream_base_url="https://example.test/v1",
                )
                recorder._captures[("0013", "task_log_hdfs_connections", 1)] = [
                    {
                        "messages": [
                            {"role": "system", "content": "system prompt"},
                            {"role": "user", "content": "first"},
                        ],
                        "tools": [{"type": "function", "function": {"name": "search_docs"}}],
                    },
                    {
                        "messages": [
                            {"role": "system", "content": "system prompt"},
                            {"role": "user", "content": "second"},
                        ]
                    },
                ]

                transcript = [
                    {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "reply 1"}]}} ,
                    {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "reply 2"}]}} ,
                    {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "extra reply"}]}} ,
                ]

                rows_written = recorder.finalize_execution(
                    benchmark_run_id="0013",
                    task_id="task_log_hdfs_connections",
                    run_index=1,
                    transcript=transcript,
                )
            finally:
                recorder.stop()

            self.assertEqual(rows_written, 2)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertEqual(first["task_id"], "task_log_hdfs_connections")
            self.assertEqual(first["run_index"], 1)
            self.assertEqual(first["messages"][-1], {"role": "assistant", "content": "reply 1"})
            self.assertIn("tools", first)
            self.assertEqual(second["messages"][-1], {"role": "assistant", "content": "reply 2"})
            self.assertNotIn("tools", second)


if __name__ == "__main__":
    unittest.main()
