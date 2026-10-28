"""Workspace prep must not revive OpenClaw's seeded first-run BOOTSTRAP.md."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import prepare_task_workspace  # noqa: E402
from lib_tasks import Task  # noqa: E402


def _task() -> Task:
    return Task(
        task_id="task_sanity",
        name="Sanity",
        category="test",
        grading_type="automated",
        timeout_seconds=60,
        workspace_files=[],
        prompt="ping",
        expected_behavior="",
        grading_criteria=[],
    )


class TestPrepareTaskWorkspaceBootstrap(unittest.TestCase):
    def test_does_not_restore_seeded_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            main = home / ".openclaw" / "workspace"
            main.mkdir(parents=True)
            (main / "AGENTS.md").write_text("from-main\n", encoding="utf-8")

            workspace = root / "agent-ws"
            workspace.mkdir()
            (workspace / "AGENTS.md").write_text("seeded-agents\n", encoding="utf-8")
            (workspace / "SOUL.md").write_text("seeded-soul\n", encoding="utf-8")
            (workspace / "BOOTSTRAP.md").write_text("you just woke up\n", encoding="utf-8")

            with patch("lib_agent.Path.home", return_value=home):
                with patch("lib_agent._get_agent_workspace", return_value=workspace):
                    result = prepare_task_workspace(root / "skill", "0029-sanity-run1", _task(), "agent")

            self.assertEqual(result, workspace)
            self.assertEqual((workspace / "AGENTS.md").read_text(encoding="utf-8"), "from-main\n")
            self.assertEqual((workspace / "SOUL.md").read_text(encoding="utf-8"), "seeded-soul\n")
            self.assertFalse((workspace / "BOOTSTRAP.md").exists())

    def test_copies_bootstrap_when_main_workspace_has_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            main = home / ".openclaw" / "workspace"
            main.mkdir(parents=True)
            (main / "BOOTSTRAP.md").write_text("operator-kept\n", encoding="utf-8")

            workspace = root / "agent-ws"
            workspace.mkdir()
            (workspace / "BOOTSTRAP.md").write_text("seeded\n", encoding="utf-8")

            with patch("lib_agent.Path.home", return_value=home):
                with patch("lib_agent._get_agent_workspace", return_value=workspace):
                    prepare_task_workspace(root / "skill", "0029-sanity-run1", _task(), "agent")

            text = (workspace / "BOOTSTRAP.md").read_text(encoding="utf-8")
            self.assertEqual(text, "operator-kept\n")
