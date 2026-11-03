"""
Central place for PinchBench scratch / state paths.

Everything that used to be hard-coded under /tmp/pinchbench or ~/.openclaw is
resolved here so a single env var (PINCHBENCH_WORKDIR) relocates it. This
matters on SLURM + enroot/Pyxis, where /tmp can be small, shared between jobs
or wiped, the image is read-only and /root is not writable.

Resolution order for the work dir (first hit wins):
  1. $PINCHBENCH_WORKDIR
  2. $SLURM_TMPDIR/pinchbench                      (job-local scratch, if the site sets it)
  3. ${SLURM_SUBMIT_DIR:-$PWD}/.pinchbench-work/job-$SLURM_JOB_ID   (inside a SLURM job)
  4. ${XDG_CACHE_HOME:-~/.cache}/pinchbench/work   (plain local run)

scripts/pinchbench_env.sh implements the same order for the shell side and
exports PINCHBENCH_WORKDIR, so normally (1) is what Python sees.

Layout under the work dir:
  runs/     benchmark run root (was /tmp/pinchbench); per-task workspaces
  judge/    judge agent workspaces + private grading keys (was /tmp/pinchbench/judge)
  tmp/      TMPDIR for every child process (grading snapshots, judge prompts, ...)

OpenClaw state dir: $OPENCLAW_STATE_DIR if set (OpenClaw itself honours it),
else ~/.openclaw — identical to OpenClaw's own default, so Python and the
openclaw CLI always look at the same directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = (os.environ.get(name) or "").strip()
    return Path(value).expanduser() if value else None


def pinchbench_workdir() -> Path:
    explicit = _env_path("PINCHBENCH_WORKDIR")
    if explicit is not None:
        return explicit
    slurm_tmp = _env_path("SLURM_TMPDIR")
    if slurm_tmp is not None:
        return slurm_tmp / "pinchbench"
    job_id = (os.environ.get("SLURM_JOB_ID") or "").strip()
    if job_id:
        base = _env_path("SLURM_SUBMIT_DIR") or Path.cwd()
        return base / ".pinchbench-work" / f"job-{job_id}"
    cache = _env_path("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return cache / "pinchbench" / "work"


def run_root() -> Path:
    """Root for benchmark runs (replaces /tmp/pinchbench)."""
    return pinchbench_workdir() / "runs"


def judge_root() -> Path:
    """Root for judge agent workspaces (replaces /tmp/pinchbench/judge)."""
    return pinchbench_workdir() / "judge"


def openclaw_state_dir() -> Path:
    """OpenClaw state dir, matching the openclaw CLI's own resolution."""
    override = _env_path("OPENCLAW_STATE_DIR")
    if override is not None:
        return override
    return Path.home() / ".openclaw"


def openclaw_config_path() -> Path:
    override = _env_path("OPENCLAW_CONFIG_PATH")
    if override is not None:
        return override
    return openclaw_state_dir() / "openclaw.json"


def ensure_tmpdir() -> Path:
    """Point TMPDIR (and Python's tempfile) at <workdir>/tmp unless TMPDIR is
    already set to a writable dir. Called once at benchmark start-up."""
    import tempfile

    current = _env_path("TMPDIR")
    if current is not None and current.is_dir() and os.access(current, os.W_OK | os.X_OK):
        target = current
    else:
        target = pinchbench_workdir() / "tmp"
        target.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(target)
    tempfile.tempdir = str(target)
    return target
