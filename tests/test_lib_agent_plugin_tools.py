"""Tests for plugin tool alsoAllow wiring on auto-created agents."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import (  # noqa: E402
    DEFAULT_ALLOWED_PLUGIN_IDS,
    _apply_plugin_tool_allowances,
    _enabled_plugin_ids,
)


class EnabledPluginIdsTests(unittest.TestCase):
    def test_always_includes_task_guard_and_tavily(self) -> None:
        self.assertEqual(
            _enabled_plugin_ids({}),
            list(DEFAULT_ALLOWED_PLUGIN_IDS),
        )
        self.assertIn("task-guard", DEFAULT_ALLOWED_PLUGIN_IDS)
        self.assertIn("tavily", DEFAULT_ALLOWED_PLUGIN_IDS)

    def test_includes_other_enabled_plugins(self) -> None:
        config = {
            "plugins": {
                "entries": {
                    "tavily": {"enabled": True},
                    "task-guard": {"enabled": True},
                    "browser": {"enabled": True},
                    "diffs": {"enabled": False},
                }
            }
        }
        ids = _enabled_plugin_ids(config)
        self.assertEqual(ids[:2], ["task-guard", "tavily"])
        self.assertIn("browser", ids)
        self.assertNotIn("diffs", ids)


class ApplyPluginToolAllowancesTests(unittest.TestCase):
    def test_sets_global_and_per_agent_also_allow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "tools": {"profile": "coding"},
                        "plugins": {
                            "entries": {
                                "tavily": {"enabled": True},
                                "task-guard": {"enabled": True},
                            }
                        },
                        "agents": {
                            "list": [
                                {
                                    "id": "bench-agent",
                                    "tools": {"deny": ["process", "sessions_spawn"]},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _apply_plugin_tool_allowances("bench-agent")
            self.assertTrue(changed)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["tools"]["alsoAllow"], ["task-guard", "tavily"])
            self.assertEqual(
                data["agents"]["list"][0]["tools"]["alsoAllow"],
                ["task-guard", "tavily"],
            )
            self.assertEqual(
                data["agents"]["list"][0]["tools"]["deny"],
                ["process", "sessions_spawn"],
            )

    def test_merges_without_dropping_existing_also_allow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "tools": {"alsoAllow": ["web_search"]},
                        "agents": {
                            "list": [
                                {
                                    "id": "bench-agent",
                                    "tools": {"alsoAllow": ["task_plan"]},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _apply_plugin_tool_allowances("bench-agent")
            self.assertTrue(changed)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["tools"]["alsoAllow"], ["web_search", "task-guard", "tavily"])
            self.assertEqual(
                data["agents"]["list"][0]["tools"]["alsoAllow"],
                ["task_plan", "task-guard", "tavily"],
            )

    def test_idempotent_when_already_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            already = list(DEFAULT_ALLOWED_PLUGIN_IDS)
            config_path.write_text(
                json.dumps(
                    {
                        "tools": {"alsoAllow": already},
                        "agents": {
                            "list": [
                                {"id": "bench-agent", "tools": {"alsoAllow": already}}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                changed = _apply_plugin_tool_allowances("bench-agent")
            self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
