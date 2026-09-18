from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_grading import (  # noqa: E402
    MAX_JUDGE_WORKSPACE_CHARS,
    _combine_grades,
    _compute_cache_key,
    _grade_llm_judge,
    _is_context_length_error,
    _normalize_judge_response,
    _parse_judge_response,
    _read_workspace_files,
    _shrink_judge_inputs,
    _summarize_transcript,
    GradeResult,
    clear_judge_cache,
)
from lib_tasks import Task  # noqa: E402


class JudgeNormalizationTests(unittest.TestCase):
    def test_normalize_judge_response_averages_summed_total_when_breakdown_is_unit_scale(
        self,
    ) -> None:
        parsed = {
            "scores": {
                "coverage": 0.75,
                "synthesis": 0.75,
                "structure": 0.75,
                "tone": 0.8,
                "conciseness": 0.8,
            },
            "total": 3.85,
            "notes": "Summed by mistake",
        }

        normalized = _normalize_judge_response(parsed)

        self.assertAlmostEqual(normalized["total"], 0.77)

    def test_hybrid_score_uses_normalized_judge_total(self) -> None:
        auto = GradeResult(
            task_id="task_email_triage",
            score=0.7062937062937062,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="",
        )
        judge = GradeResult(
            task_id="task_email_triage",
            score=0.87,
            max_score=1.0,
            grading_type="llm_judge",
            breakdown={},
            notes="",
        )

        class _Task:
            task_id = "task_email_triage"
            grading_weights = {"automated": 0.4, "llm_judge": 0.6}

        combined = _combine_grades(_Task(), auto, judge)

        self.assertAlmostEqual(combined.score, 0.8045174825174824)

    def test_parse_judge_response_prefers_latest_assistant_json_over_embedded_tool_json(
        self,
    ) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                'Tool: web_search({"count": 10, "query": "WWDC 2025"})\n'
                                'Result: {"query": "WWDC 2025", "count": 10}'
                            ),
                        }
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "NO_REPLY"}],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"scores": {"accuracy": 0.75, "completeness": 1.0}, '
                                '"total": 0.875, "notes": "Final judgment"}'
                            ),
                        }
                    ],
                },
            },
        ]

        parsed = _parse_judge_response(transcript)

        self.assertEqual(parsed["scores"]["accuracy"], 0.75)
        self.assertEqual(parsed["scores"]["completeness"], 1.0)
        self.assertEqual(parsed["total"], 0.875)

    def test_parse_judge_response_ignores_waiting_messages_before_final_json(self) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Waiting for remaining parts (6-7)."}
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '{"scores": {"clarity": 0.75, "accuracy": 0.85}, '
                                '"total": 0.8, "notes": "Looks good"}'
                            ),
                        }
                    ],
                },
            },
        ]

        parsed = _parse_judge_response(transcript)

        self.assertEqual(parsed["scores"]["clarity"], 0.75)
        self.assertEqual(parsed["total"], 0.8)


class WorkspaceFilesForJudgeTests(unittest.TestCase):
    def test_read_workspace_files_preserves_full_text_file_content(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            long_content = "A" * 3000 + "TAIL_MARKER"
            (workspace / "report.md").write_text(long_content, encoding="utf-8")

            content = _read_workspace_files(str(workspace))

        self.assertIn("### File: report.md", content)
        self.assertIn("TAIL_MARKER", content)
        self.assertIn(long_content, content)

    def test_compute_cache_key_changes_when_workspace_content_changes(self) -> None:
        first_key = _compute_cache_key(
            "task_report",
            "same transcript",
            "same rubric",
            "same model",
            "workspace version one",
        )
        second_key = _compute_cache_key(
            "task_report",
            "same transcript",
            "same rubric",
            "same model",
            "workspace version two",
        )

        self.assertNotEqual(first_key, second_key)

    def test_read_workspace_files_caps_total_content(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            file_body = "X" * 3500
            for index in range(20):
                (workspace / f"dump_{index:02d}.txt").write_text(file_body, encoding="utf-8")

            content = _read_workspace_files(str(workspace))

        self.assertLessEqual(len(content), MAX_JUDGE_WORKSPACE_CHARS + 200)
        self.assertIn("omitted for judge context limit", content)


def _make_judge_task() -> Task:
    return Task(
        task_id="task_browser_automation",
        name="Browser",
        category="coding",
        grading_type="llm_judge",
        timeout_seconds=60,
        workspace_files=[],
        prompt="Automate the browser",
        expected_behavior="Complete the workflow",
        grading_criteria=["quality"],
        llm_judge_rubric="Score quality from 0 to 1",
    )


def _assistant_transcript(text: str = "done") -> list[dict]:
    return [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        }
    ]


CONTEXT_OVERFLOW_ERROR = (
    'HTTP 400: {"error": {"code": "context_length_exceeded", '
    '"message": "Input tokens exceed the configured limit of 272000 tokens. '
    'Your messages resulted in 1878242 tokens."}}'
)
SUCCESS_JUDGE_TEXT = '{"scores": {"quality": 0.8}, "total": 0.8, "notes": "ok"}'


class JudgeFallbackContextLengthTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_judge_cache()

    def tearDown(self) -> None:
        clear_judge_cache()

    def test_is_context_length_error_detects_azure_payload(self) -> None:
        self.assertTrue(_is_context_length_error(CONTEXT_OVERFLOW_ERROR))
        self.assertFalse(_is_context_length_error("HTTP 500: internal server error"))

    def test_shrink_judge_inputs_uses_reported_token_limit(self) -> None:
        transcript = "T" * 80_000
        workspace = "W" * 20_000
        shrunk_t, shrunk_w = _shrink_judge_inputs(
            transcript, workspace, CONTEXT_OVERFLOW_ERROR
        )
        self.assertLess(len(shrunk_t) + len(shrunk_w), len(transcript) + len(workspace))
        self.assertIn("truncated for judge context limit", shrunk_t)

    def test_summarize_transcript_truncates_huge_user_messages(self) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": ["U" * 50_000],
                },
            }
        ]
        summary = _summarize_transcript(transcript)
        self.assertLess(len(summary), 5_000)
        self.assertIn("[truncated]", summary)

    def test_context_length_error_skips_retries_and_uses_fallback_model(self) -> None:
        calls: list[str] = []

        def fake_call_judge_api(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gpt-5.4-mini":
                return {"status": "error", "text": "", "error": CONTEXT_OVERFLOW_ERROR}
            return {"status": "success", "text": SUCCESS_JUDGE_TEXT}

        with (
            patch("lib_grading.call_judge_api", side_effect=fake_call_judge_api),
            patch("lib_grading.time.sleep") as sleep_mock,
        ):
            result = _grade_llm_judge(
                task=_make_judge_task(),
                execution_result={"transcript": _assistant_transcript(), "status": "success"},
                judge_model="gpt-5.4-mini",
                judge_fallback_model="gpt-5.4",
                judge_agent_prefix="bench-judge",
                judge_timeout_seconds=30,
                judge_backend="api",
            )

        self.assertEqual(calls, ["gpt-5.4-mini", "gpt-5.4"])
        sleep_mock.assert_not_called()
        self.assertAlmostEqual(result.score, 0.8)

    def test_context_length_error_shrinks_prompt_when_fallback_also_overflows(self) -> None:
        prompt_lengths: list[int] = []

        def fake_call_judge_api(**kwargs):
            prompt_lengths.append(len(kwargs["prompt"]))
            if len(prompt_lengths) == 1:
                self.assertEqual(kwargs["model"], "gpt-5.4-mini")
                return {"status": "error", "text": "", "error": CONTEXT_OVERFLOW_ERROR}
            if len(prompt_lengths) == 2:
                self.assertEqual(kwargs["model"], "gpt-5.4")
                return {
                    "status": "error",
                    "text": "",
                    "error": (
                        'HTTP 400: {"error": {"code": "context_length_exceeded", '
                        '"message": "Input tokens exceed the configured limit of 922000 tokens. '
                        'Your messages resulted in 1878242 tokens."}}'
                    ),
                }
            self.assertEqual(kwargs["model"], "gpt-5.4")
            return {"status": "success", "text": SUCCESS_JUDGE_TEXT}

        huge_transcript = _assistant_transcript("A" * 20_000)
        with (
            patch("lib_grading.call_judge_api", side_effect=fake_call_judge_api),
            patch("lib_grading.time.sleep"),
        ):
            result = _grade_llm_judge(
                task=_make_judge_task(),
                execution_result={"transcript": huge_transcript, "status": "success"},
                judge_model="gpt-5.4-mini",
                judge_fallback_model="gpt-5.4",
                judge_agent_prefix="bench-judge",
                judge_timeout_seconds=30,
                judge_backend="api",
            )

        self.assertEqual(len(prompt_lengths), 3)
        self.assertLess(prompt_lengths[2], prompt_lengths[1])
        self.assertAlmostEqual(result.score, 0.8)

    def test_non_context_errors_still_retry_before_fallback(self) -> None:
        calls: list[str] = []

        def fake_call_judge_api(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gpt-5.4-mini":
                return {"status": "error", "text": "", "error": "HTTP 500: boom"}
            return {"status": "success", "text": SUCCESS_JUDGE_TEXT}

        with (
            patch("lib_grading.call_judge_api", side_effect=fake_call_judge_api),
            patch("lib_grading.time.sleep") as sleep_mock,
        ):
            result = _grade_llm_judge(
                task=_make_judge_task(),
                execution_result={"transcript": _assistant_transcript(), "status": "success"},
                judge_model="gpt-5.4-mini",
                judge_fallback_model="gpt-5.4",
                judge_agent_prefix="bench-judge",
                judge_timeout_seconds=30,
                judge_backend="api",
            )

        self.assertEqual(calls, ["gpt-5.4-mini", "gpt-5.4-mini", "gpt-5.4"])
        self.assertEqual(sleep_mock.call_count, 1)
        self.assertAlmostEqual(result.score, 0.8)


if __name__ == "__main__":
    unittest.main()
