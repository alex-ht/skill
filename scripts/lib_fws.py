"""
fws lifecycle management for GWS-based tasks.

Starts/stops the fws server and configures environment variables
so that gws CLI commands are redirected to the local mock.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from lib_paths import pinchbench_workdir

logger = logging.getLogger("pinchbench")

# Environment variables that fws sets
MOCK_GWS_ENV_KEYS = [
    "GOOGLE_WORKSPACE_CLI_CONFIG_DIR",
    "GOOGLE_WORKSPACE_CLI_TOKEN",
    "HTTPS_PROXY",
    "SSL_CERT_FILE",
    "GH_TOKEN",  # `fws server env` exports GH_TOKEN=fake; gh refuses to run without a token
]

# Ports are configurable so two jobs sharing a node do not collide on 4100/4101.
FWS_PORT = int(os.environ.get("PINCHBENCH_FWS_PORT", "4100"))
FWS_PROXY_PORT = int(os.environ.get("PINCHBENCH_FWS_PROXY_PORT", str(FWS_PORT + 1)))


def fws_data_dir() -> Path:
    """fws state (config, certs, server.json, log). Kept out of ~/.local/share
    so a read-only / shared HOME works; override with FWS_DATA_DIR."""
    explicit = (os.environ.get("FWS_DATA_DIR") or "").strip()
    return Path(explicit) if explicit else pinchbench_workdir() / "fws"


def _fws_subprocess_env() -> dict:
    env = os.environ.copy()
    env["FWS_DATA_DIR"] = str(fws_data_dir())
    return env


def is_fws_task(frontmatter: dict) -> bool:
    """Check if a task requires fws (category is gws/github or prerequisites include fws)."""
    if frontmatter.get("category") in ("gws", "github"):
        return True
    prereqs = frontmatter.get("prerequisites", [])
    return any("fws" in str(p) for p in prereqs)


def fws_available() -> bool:
    """Check if fws CLI is available."""
    return shutil.which("fws") is not None


def start_fws() -> dict:
    """Start the fws server and set environment variables.

    Returns a dict of the original env var values (for restoration).
    """
    logger.info("🔧 Starting fws server...")

    data_dir = fws_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    env = _fws_subprocess_env()

    # Stop any existing server (only the one owned by this data dir)
    subprocess.run(["fws", "server", "stop"], capture_output=True, check=False, env=env)
    time.sleep(0.3)

    # Start server
    result = subprocess.run(
        ["fws", "server", "start", "-p", str(FWS_PORT), "--proxy-port", str(FWS_PROXY_PORT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )

    # fws 0.4.x exits 0 even when the health check fails ("Failed to start
    # server." on stderr), so check stdout too instead of silently continuing
    # with a dead mock.
    if result.returncode != 0 or "fws server started" not in result.stdout:
        detail = (result.stderr or result.stdout or "").strip()
        log_file = data_dir / "server.log"
        logger.error("Failed to start fws (data dir %s): %s", data_dir, detail[-2000:])
        raise RuntimeError(f"fws server start failed (see {log_file}): {detail[-500:]}")

    # Parse the env vars from output
    env_vars = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and any(key in line for key in MOCK_GWS_ENV_KEYS):
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip()

    if not env_vars:
        # Fallback: same values `fws server env` would print
        env_vars = {
            "GOOGLE_WORKSPACE_CLI_CONFIG_DIR": str(data_dir / "config"),
            "GOOGLE_WORKSPACE_CLI_TOKEN": "fake",
            "HTTPS_PROXY": f"http://localhost:{FWS_PROXY_PORT}",
            "SSL_CERT_FILE": str(data_dir / "certs" / "ca-bundle.crt"),
            "GH_TOKEN": "fake",
        }
    # Child tools must find the same fws instance (e.g. `fws` called by the agent).
    env_vars.setdefault("FWS_DATA_DIR", str(data_dir))

    # Save original values and set new ones
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    logger.info("✅ fws server started on :%d/:%d (data %s), env configured",
                FWS_PORT, FWS_PROXY_PORT, data_dir)
    return original_env


def stop_fws(original_env: dict) -> None:
    """Stop the fws server and restore original environment variables."""
    logger.info("🔧 Stopping fws server...")

    subprocess.run(
        ["fws", "server", "stop"], capture_output=True, check=False, env=_fws_subprocess_env()
    )

    # Restore original env
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    logger.info("✅ fws server stopped, env restored")
