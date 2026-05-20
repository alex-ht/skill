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
)


class TrainRecorderTests(unittest.TestCase):
    def test_normalize_openai_request_body_preserves_tools_and_stringifies_arguments(self) -> None:
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Read README.md"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": {"path": "README.md"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "# README",
                },
            ],
            "tools": [{"type": "function", "function": {"name": "read"}}],
        }

        messages, tools = normalize_openai_request_body(body)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], '{"path":"README.md"}')
        self.assertEqual(tools[0]["function"]["name"], "read")

    def test_normalize_anthropic_request_body_converts_tool_results_and_system(self) -> None:
        body = {
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Find the summary."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will inspect the file."},
                        {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "README.md"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "# README"}
                    ],
                },
            ],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        }

        messages, tools = normalize_anthropic_request_body(body)

        self.assertEqual(messages[0], {"role": "system", "content": "You are helpful."})
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(messages[3], {"role": "tool", "tool_call_id": "toolu_1", "content": "# README"})
        self.assertEqual(tools[0]["name"], "read")

    def test_finalize_execution_writes_merged_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "train.jsonl"
            recorder = TrainingRecorder(output_path)
            recorder._captures[("0007", "task_files", 1)] = [  # type: ignore[attr-defined]
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Read README.md"},
                    ],
                    "tools": [{"type": "function", "function": {"name": "read"}}],
                }
            ]
            transcript = [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Done."},
                            {
                                "type": "toolCall",
                                "id": "call_1",
                                "name": "read",
                                "arguments": {"path": "README.md"},
                            },
                        ],
                    },
                }
            ]

            rows_written = recorder.finalize_execution(
                benchmark_run_id="0007",
                task_id="task_files",
                run_index=1,
                transcript=transcript,
            )

            self.assertEqual(rows_written, 1)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["task_id"], "task_files")
            self.assertEqual(row["run_index"], 1)
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertEqual(row["messages"][-1]["tool_calls"][0]["function"]["arguments"], '{"path":"README.md"}')
            self.assertEqual(row["tools"][0]["function"]["name"], "read")

    def test_extract_transcript_assistant_messages_ignores_non_assistant_events(self) -> None:
        transcript = [
            {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]}} ,
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}} ,
        ]

        messages = extract_transcript_assistant_messages(transcript)

        self.assertEqual(messages, [{"role": "assistant", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
