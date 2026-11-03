#!/usr/bin/env bash
# Convenience wrapper for running PinchBench
# Usage: ./scripts/run.sh --model anthropic/claude-sonnet-4
#
# All scratch/state/cache paths go under $PINCHBENCH_WORKDIR (see
# scripts/pinchbench_env.sh). Inside the pinchbench image (PINCHBENCH_IMAGE=1)
# the benchmark runs on the image's system python3, which already has pyyaml,
# pytest and every task dependency (pandas, openpyxl, playwright, ...). With
# `uv run` the agent's `python3` would resolve to uv's throw-away script env
# (pyyaml + pytest only), and uv would need network + a writable cache.
# Force a choice with PINCHBENCH_PYTHON=system|uv.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pinchbench_env.sh
source "$SCRIPT_DIR/pinchbench_env.sh"
cd "$SCRIPT_DIR/.."

mode="${PINCHBENCH_PYTHON:-auto}"
if [[ "$mode" == "auto" ]]; then
  if [[ "${PINCHBENCH_IMAGE:-0}" == "1" ]] && python3 -c 'import yaml, pytest' 2>/dev/null; then
    mode=system
  else
    mode=uv
  fi
fi
if [[ "$mode" == "system" ]]; then
  exec python3 scripts/benchmark.py "$@"
fi
exec uv run scripts/benchmark.py "$@"
