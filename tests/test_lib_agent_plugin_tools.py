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
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_DENIED_TOOLS,
    _apply_default_tool_denials,
    _apply_plugin_tool_allowances,
    _enabled_plugin_ids,
)


class EnabledPluginIdsTests(unittest.TestCase):
    def test_always_includes_task_guard(self) -> None:
        self.assertEqual(
            _enabled_plugin_ids({}),
            list(DEFAULT_ALLOWED_PLUGIN_IDS),
        )
        self.assertIn("task-guard", DEFAULT_ALLOWED_PLUGIN_IDS)
        self.assertNotIn("tavily", DEFAULT_ALLOWED_PLUGIN_IDS)

    def test_includes_other_enabled_plugins_but_not_tavily(self) -> None:
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
        self.assertEqual(ids[0], "task-guard")
        self.assertIn("browser", ids)
        self.assertNotIn("tavily", ids)
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
            self.assertEqual(
                data["tools"]["alsoAllow"],
                ["task-guard", "web_search", "web_fetch"],
            )
            self.assertEqual(
                data["agents"]["list"][0]["tools"]["alsoAllow"],
                ["task-guard", "web_search", "web_fetch"],
            )
            self.assertEqual(
                data["agents"]["list"][0]["tools"]["deny"],
                ["process", "sessions_spawn"],
            )
            self.assertFalse(data["plugins"]["entries"]["tavily"]["enabled"])

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
            self.assertEqual(
                data["tools"]["alsoAllow"],
                ["web_search", "task-guard", "web_fetch"],
            )
            self.assertEqual(
                data["agents"]["list"][0]["tools"]["alsoAllow"],
                ["task_plan", "task-guard", "web_search", "web_fetch"],
            )
            self.assertFalse(data["plugins"]["entries"]["tavily"]["enabled"])

    def test_builtin_web_tools_are_allowlisted(self) -> None:
        self.assertNotIn("web_search", DEFAULT_DENIED_TOOLS)
        self.assertNotIn("web_fetch", DEFAULT_DENIED_TOOLS)
        self.assertIn("web_search", DEFAULT_ALLOWED_TOOLS)
        self.assertIn("web_fetch", DEFAULT_ALLOWED_TOOLS)

    def test_allowlists_builtin_web_after_main_tools_copy(self) -> None:
        main_tools = {
            "deny": ["update_plan", "web_fetch"],
            "alsoAllow": ["task_mark", "task_plan", "tavily_extract", "tavily_search"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "tools": {"profile": "coding"},
                        "agents": {
                            "list": [
                                {"id": "main", "tools": json.loads(json.dumps(main_tools))},
                                {
                                    "id": "bench-agent",
                                    "tools": json.loads(json.dumps(main_tools)),
                                },
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("lib_agent._openclaw_config_path", return_value=config_path):
                _apply_default_tool_denials("bench-agent")
                changed = _apply_plugin_tool_allowances("bench-agent")
            self.assertTrue(changed)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            main_entry = data["agents"]["list"][0]
            bench_tools = data["agents"]["list"][1]["tools"]
            self.assertEqual(main_entry["tools"], main_tools)
            self.assertEqual(
                bench_tools["deny"],
                [
                    "update_plan",
                    "process",
                    "sessions_spawn",
                    "tavily",
                    "tavily_search",
                    "tavily_extract",
                ],
            )
            self.assertEqual(
                bench_tools["alsoAllow"],
                [
                    "task_mark",
                    "task_plan",
                    "task-guard",
                    "web_search",
                    "web_fetch",
                ],
            )
            self.assertFalse(data["plugins"]["entries"]["tavily"]["enabled"])
            for tool_name in DEFAULT_DENIED_TOOLS:
                self.assertIn(tool_name, bench_tools["deny"])
            for tool_name in DEFAULT_ALLOWED_TOOLS:
                self.assertNotIn(tool_name, bench_tools["deny"])
                self.assertIn(tool_name, bench_tools["alsoAllow"])

    def test_idempotent_when_already_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "openclaw.json"
            already = list(DEFAULT_ALLOWED_PLUGIN_IDS) + list(DEFAULT_ALLOWED_TOOLS)
            config_path.write_text(
                json.dumps(
                    {
                        "tools": {"alsoAllow": already},
                        "plugins": {"entries": {"tavily": {"enabled": False}}},
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
