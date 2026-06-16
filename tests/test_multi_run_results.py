"""Tests for multi-run benchmark result aggregation."""
import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib_grading import GradeResult  # noqa: E402
from lib_trend import _aggregate_task_scores, _mean_score_for_task_runs  # noqa: E402
from lib_upload import _build_payload  # noqa: E402


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark", SCRIPTS / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestMultiRunResultHelpers(TestCase):
    def test_single_run_grading_wraps_one_grade(self):
        benchmark = _load_benchmark_module()
        grade = GradeResult(
            task_id="task_a",
            score=0.8,
            max_score=1.0,
            grading_type="automated",
            breakdown={"check": 1.0},
            notes="ok",
        )
        block = benchmark._single_run_grading(grade)
        self.assertEqual(len(block["runs"]), 1)
        self.assertEqual(block["mean"], 0.8)
        self.assertEqual(block["std"], 0.0)

    def test_grading_for_result_prefers_per_run_grade(self):
        benchmark = _load_benchmark_module()
        grade = GradeResult(
            task_id="task_a",
            score=0.7,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="run 2",
        )
        result = {
            "task_id": "task_a",
            "_grade": grade,
        }
        aggregated = {
            "task_a": {
                "runs": [{"score": 0.8}, {"score": 0.6}],
                "mean": 0.7,
                "std": 0.1,
                "min": 0.6,
                "max": 0.8,
            }
        }
        block = benchmark._grading_for_result(result, aggregated)
        self.assertEqual(len(block["runs"]), 1)
        self.assertEqual(block["runs"][0]["notes"], "run 2")
        self.assertEqual(block["mean"], 0.7)

    def test_category_scores_count_unique_tasks_only(self):
        benchmark = _load_benchmark_module()

        class Task:
            def __init__(self, task_id: str, category: str):
                self.task_id = task_id
                self.category = category

        grades_by_task_id = {
            "task_a": {"mean": 0.8},
            "task_b": {"mean": 0.6},
        }
        tasks_by_id = {
            "task_a": Task("task_a", "coding"),
            "task_b": Task("task_b", "coding"),
        }
        scores = benchmark._compute_category_scores(grades_by_task_id, tasks_by_id)
        self.assertEqual(scores["CODING"]["task_count"], 2)
        self.assertAlmostEqual(scores["CODING"]["score"], 1.4)
        self.assertAlmostEqual(scores["CODING"]["max_score"], 2.0)

    def test_efficiency_summary_aggregates_usage_per_task(self):
        benchmark = _load_benchmark_module()
        task_entries = [
            {
                "task_id": "task_a",
                "execution_time": 10.0,
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "cost_usd": 0.0, "request_count": 1},
            },
            {
                "task_id": "task_a",
                "execution_time": 12.0,
                "usage": {"input_tokens": 150, "output_tokens": 30, "total_tokens": 180, "cost_usd": 0.0, "request_count": 1},
            },
            {
                "task_id": "task_b",
                "execution_time": 5.0,
                "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60, "cost_usd": 0.0, "request_count": 1},
            },
        ]
        grades_by_task_id = {
            "task_a": {"mean": 0.75},
            "task_b": {"mean": 0.5},
        }
        summary = benchmark._compute_efficiency_summary(task_entries, grades_by_task_id)
        self.assertEqual(summary["total_tokens"], 360)
        self.assertEqual(summary["tokens_per_task"], 180.0)
        self.assertEqual(len(summary["per_task"]), 2)
        by_id = {row["task_id"]: row for row in summary["per_task"]}
        self.assertEqual(by_id["task_a"]["total_tokens"], 300)
        self.assertEqual(by_id["task_b"]["total_tokens"], 60)


class TestLegacyAndPerRunAggregation(TestCase):
    def _legacy_multi_run_tasks(self):
        shared_runs = [
            {"score": 0.8, "max_score": 1.0, "grading_type": "automated", "breakdown": {}, "notes": "r1"},
            {"score": 0.6, "max_score": 1.0, "grading_type": "automated", "breakdown": {}, "notes": "r2"},
        ]
        grading = {
            "runs": shared_runs,
            "mean": 0.7,
            "std": 0.1,
            "min": 0.6,
            "max": 0.8,
        }
        return [
            {
                "task_id": "task_a",
                "workspace": "/tmp/run_001",
                "grading": grading,
                "usage": {"input_tokens": 10, "output_tokens": 1, "request_count": 1, "cost_usd": 0.0},
                "execution_time": 1.0,
                "timed_out": False,
                "frontmatter": {},
            },
            {
                "task_id": "task_a",
                "workspace": "/tmp/run_002",
                "grading": grading,
                "usage": {"input_tokens": 12, "output_tokens": 2, "request_count": 1, "cost_usd": 0.0},
                "execution_time": 2.0,
                "timed_out": False,
                "frontmatter": {},
            },
        ]

    def _per_run_tasks(self):
        return [
            {
                "task_id": "task_a",
                "run_index": 1,
                "workspace": "/tmp/run_001",
                "grading": {
                    "runs": [{"score": 0.8, "max_score": 1.0, "grading_type": "automated", "breakdown": {}, "notes": "r1"}],
                    "mean": 0.8,
                    "std": 0.0,
                    "min": 0.8,
                    "max": 0.8,
                },
                "usage": {"input_tokens": 10, "output_tokens": 1, "request_count": 1, "cost_usd": 0.0},
                "execution_time": 1.0,
                "timed_out": False,
                "frontmatter": {},
            },
            {
                "task_id": "task_a",
                "run_index": 2,
                "workspace": "/tmp/run_002",
                "grading": {
                    "runs": [{"score": 0.6, "max_score": 1.0, "grading_type": "automated", "breakdown": {}, "notes": "r2"}],
                    "mean": 0.6,
                    "std": 0.0,
                    "min": 0.6,
                    "max": 0.6,
                },
                "usage": {"input_tokens": 12, "output_tokens": 2, "request_count": 1, "cost_usd": 0.0},
                "execution_time": 2.0,
                "timed_out": False,
                "frontmatter": {},
            },
        ]

    def test_trend_handles_legacy_duplicated_grading(self):
        score_pct, task_count = _aggregate_task_scores(self._legacy_multi_run_tasks())
        self.assertEqual(task_count, 1)
        self.assertAlmostEqual(score_pct, 70.0)

    def test_trend_handles_per_run_grading(self):
        score_pct, task_count = _aggregate_task_scores(self._per_run_tasks())
        self.assertEqual(task_count, 1)
        self.assertAlmostEqual(score_pct, 70.0)

    def test_upload_groups_runs_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "0001_model.json"
            payload_source = {
                "model": "test/model",
                "benchmark_version": "2.0.0",
                "run_id": "0001",
                "timestamp": 1.0,
                "suite": "task_a",
                "runs_per_task": 2,
                "tasks": self._per_run_tasks(),
            }
            results_path.write_text(json.dumps(payload_source), encoding="utf-8")
            payload = _build_payload(results_path)
            self.assertEqual(len(payload["tasks"]), 1)
            self.assertAlmostEqual(payload["tasks"][0]["score"], 0.7)
            self.assertAlmostEqual(payload["total_score"], 0.7)
            self.assertAlmostEqual(payload["tasks"][0]["execution_time_seconds"], 3.0)

    def test_mean_score_for_task_runs_matches_statistics_mean(self):
        per_run = self._per_run_tasks()
        self.assertAlmostEqual(
            _mean_score_for_task_runs(per_run),
            statistics.mean([0.8, 0.6]),
        )