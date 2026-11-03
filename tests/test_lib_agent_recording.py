"""Tests for --record-train provider resolution, including bundled OpenRouter."""
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

from lib_agent import (  # noqa: E402
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    install_recording_provider_auth,
    normalize_openrouter_model_id,
    resolve_recording_provider,
)


class OpenRouterRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.missing = Path(tempfile.gettempdir()) / "pinchbench-recording-missing"
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
        profile = Path(tempfile.gettempdir()) / "pinchbench-openrouter-auth-profiles.json"
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
        profile = Path(tempfile.gettempdir()) / "pinchbench-openrouter-auth-profiles-both.json"
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
        cache = Path(tempfile.gettempdir()) / "pinchbench-openrouter-models.json"
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


class RecordingProviderAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pinchbench-rec-auth-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.source = self.tmp / "main-auth-profiles.json"
        self.store = self.tmp / "agent" / "auth-profiles.json"
        self.source.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "sglang:other": {
                            "type": "api_key",
                            "provider": "sglang",
                            "key": "other-key",
                        },
                        "sglang:default": {
                            "type": "api_key",
                            "provider": "sglang",
                            "key": "SK_DUMMY",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_unset_env_marker_copies_default_profile(self) -> None:
        provider = {
            "baseUrl": "http://127.0.0.1:9/v1",
            "api": "openai-completions",
            "apiKey": "SGLANG_API_KEY",
        }
        status = install_recording_provider_auth(
            recording_provider_id="trainrec-bench",
            source_provider_name="sglang",
            provider=provider,
            auth_store_path=self.store,
            source_auth_profiles_path=self.source,
            env={},
        )
        self.assertEqual(status, "profile")
        self.assertEqual(provider["apiKey"], "SGLANG_API_KEY")
        saved = json.loads(self.store.read_text(encoding="utf-8"))
        profile = saved["profiles"]["trainrec-bench:default"]
        self.assertEqual(profile["provider"], "trainrec-bench")
        self.assertEqual(profile["type"], "api_key")
        self.assertEqual(profile["key"], "SK_DUMMY")

    def test_set_env_marker_does_not_write_a_profile(self) -> None:
        provider = {"apiKey": "SGLANG_API_KEY", "baseUrl": "http://127.0.0.1:9/v1"}
        status = install_recording_provider_auth(
            recording_provider_id="trainrec-bench",
            source_provider_name="sglang",
            provider=provider,
            auth_store_path=self.store,
            source_auth_profiles_path=self.source,
            env={"SGLANG_API_KEY": "from-env"},
        )
        self.assertEqual(status, "env")
        self.assertFalse(self.store.exists())

    def test_literal_key_is_left_alone(self) -> None:
        provider = {"apiKey": "sk-live", "baseUrl": "https://example.internal/v1"}
        status = install_recording_provider_auth(
            recording_provider_id="trainrec-bench",
            source_provider_name="custom",
            provider=provider,
            auth_store_path=self.store,
            source_auth_profiles_path=self.source,
            env={},
        )
        self.assertEqual(status, "literal")
        self.assertEqual(provider["apiKey"], "sk-live")
        self.assertFalse(self.store.exists())

    def test_local_provider_without_credentials_drops_api_key(self) -> None:
        provider = {"apiKey": "SGLANG_API_KEY", "baseUrl": "http://127.0.0.1:30000/v1"}
        status = install_recording_provider_auth(
            recording_provider_id="trainrec-bench",
            source_provider_name="sglang",
            provider=provider,
            auth_store_path=self.store,
            source_auth_profiles_path=self.tmp / "missing.json",
            env={},
        )
        self.assertEqual(status, "local")
        self.assertNotIn("apiKey", provider)

    def test_remote_provider_without_credentials_stays_missing(self) -> None:
        provider = {"apiKey": "OPENROUTER_API_KEY", "baseUrl": "https://openrouter.ai/api/v1"}
        status = install_recording_provider_auth(
            recording_provider_id="trainrec-bench",
            source_provider_name="openrouter",
            provider=provider,
            auth_store_path=self.store,
            source_auth_profiles_path=self.tmp / "missing.json",
            env={},
        )
        self.assertEqual(status, "missing")
        self.assertEqual(provider["apiKey"], "OPENROUTER_API_KEY")
