"""Tests for --record-train provider resolution, including bundled OpenRouter."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_agent import (  # noqa: E402
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    normalize_openrouter_model_id,
    resolve_recording_provider,
)


class OpenRouterRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.missing = Path("/tmp/pinchbench-recording-missing")
        self.env = {OPENROUTER_API_KEY_ENV: "present"}

    def test_normalize_slash_free_suffix(self) -> None:
        self.assertEqual(normalize_openrouter_model_id("auto"), "openrouter/auto")
        self.assertEqual(
            normalize_openrouter_model_id("anthropic/claude-sonnet-4"),
            "anthropic/claude-sonnet-4",
        )

    def test_openrouter_auto_uses_env_marker_and_chat_completions(self) -> None:
        resolution = resolve_recording_provider(
            "openrouter/auto",
            {},
            env=self.env,
            auth_profiles_path=self.missing,
            cache_path=self.missing,
        )
        self.assertIsNone(resolution.skip_reason)
        assert resolution.provider is not None
        self.assertEqual(resolution.model_id, "openrouter/auto")
        self.assertEqual(resolution.provider["api"], "openai-completions")
        self.assertEqual(resolution.provider["baseUrl"], OPENROUTER_BASE_URL)
        self.assertEqual(resolution.provider["apiKey"], OPENROUTER_API_KEY_ENV)
        model = resolution.provider["models"][0]
        self.assertEqual(model["id"], "openrouter/auto")
        self.assertEqual(model["contextWindow"], 200_000)
        self.assertFalse(model["reasoning"])
        self.assertEqual(model["compat"], {"thinkingFormat": "openrouter"})

    def test_openrouter_vendor_model_keeps_slashed_id(self) -> None:
        resolution = resolve_recording_provider(
            "openrouter/anthropic/claude-sonnet-4",
            {},
            env=self.env,
            auth_profiles_path=self.missing,
            cache_path=self.missing,
        )
        self.assertEqual(resolution.model_id, "anthropic/claude-sonnet-4")
        assert resolution.provider is not None
        self.assertEqual(resolution.provider["models"][0]["id"], "anthropic/claude-sonnet-4")

    def test_profile_key_used_when_env_is_unset(self) -> None:
        profile = Path("/tmp/pinchbench-openrouter-auth-profiles.json")
        profile.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "openrouter:default": {
                            "type": "api_key",
                            "provider": "openrouter",
                            "key": "sk-or-test",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.addCleanup(profile.unlink, missing_ok=True)
        resolution = resolve_recording_provider(
            "openrouter/auto",
            {},
            env={},
            auth_profiles_path=profile,
            cache_path=self.missing,
        )
        assert resolution.provider is not None
        self.assertEqual(resolution.provider["apiKey"], "sk-or-test")

    def test_profile_key_wins_over_environment(self) -> None:
        profile = Path("/tmp/pinchbench-openrouter-auth-profiles-both.json")
        profile.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "openrouter:default": {
                            "type": "api_key",
                            "provider": "openrouter",
                            "key": "sk-or-profile",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.addCleanup(profile.unlink, missing_ok=True)
        resolution = resolve_recording_provider(
            "openrouter/auto",
            {},
            env=self.env,
            auth_profiles_path=profile,
            cache_path=self.missing,
        )
        assert resolution.provider is not None
        self.assertEqual(resolution.provider["apiKey"], "sk-or-profile")

    def test_missing_credentials_skips_recording(self) -> None:
        resolution = resolve_recording_provider(
            "openrouter/auto",
            {},
            env={},
            auth_profiles_path=self.missing,
            cache_path=self.missing,
        )
        self.assertIsNone(resolution.provider)
        self.assertEqual(resolution.skip_reason, "OpenRouter credentials are not available")
        self.assertEqual(resolution.model_id, "openrouter/auto")

    def test_configured_provider_is_unchanged(self) -> None:
        custom = {
            "baseUrl": "https://example.internal/v1",
            "api": "openai-completions",
            "apiKey": "SK_DUMMY",
            "models": [{"id": "gemma-4-31b"}],
        }
        resolution = resolve_recording_provider(
            "custom/gemma-4-31b",
            {"custom": custom},
            env={},
            auth_profiles_path=self.missing,
            cache_path=self.missing,
        )
        self.assertIsNone(resolution.skip_reason)
        self.assertEqual(resolution.model_id, "gemma-4-31b")
        self.assertIs(resolution.provider, custom)

    def test_cache_supplies_context_and_reasoning(self) -> None:
        cache = Path("/tmp/pinchbench-openrouter-models.json")
        cache.write_text(
            json.dumps(
                {
                    "version": 3,
                    "models": {
                        "openrouter/auto": {
                            "name": "Auto Router",
                            "input": ["text", "image"],
                            "reasoning": True,
                            "contextWindow": 2_000_000,
                            "maxTokens": 8192,
                            "cost": {"input": 0, "output": 0},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.addCleanup(cache.unlink, missing_ok=True)
        resolution = resolve_recording_provider(
            "openrouter/auto",
            {},
            env=self.env,
            auth_profiles_path=self.missing,
            cache_path=cache,
        )
        assert resolution.provider is not None
        model = resolution.provider["models"][0]
        self.assertEqual(model["name"], "Auto Router")
        self.assertTrue(model["reasoning"])
        self.assertEqual(model["input"], ["text", "image"])
        self.assertEqual(model["contextWindow"], 2_000_000)
        self.assertEqual(model["cost"], {"input": 0, "output": 0})

    def test_unknown_provider_is_skipped(self) -> None:
        resolution = resolve_recording_provider(
            "missing/model",
            {},
            env=self.env,
            auth_profiles_path=self.missing,
            cache_path=self.missing,
        )
        self.assertIsNone(resolution.provider)
        self.assertIn("not configured", resolution.skip_reason or "")
