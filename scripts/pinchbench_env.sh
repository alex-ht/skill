#!/usr/bin/env bash
# shellcheck shell=bash
# Source this file (do not execute it) to put every PinchBench / vLLM / OpenClaw
# temp, work, cache and log path under ONE directory: $PINCHBENCH_WORKDIR.
#
# Why: on SLURM + enroot/Pyxis the image is read-only, the job runs as the
# submitting user (not root, /root unwritable), HOME may be the shared host
# home, and /tmp can be small, shared between jobs, or wiped.
#
#   PINCHBENCH_WORKDIR   default (first that is writable):
#                          $SLURM_TMPDIR/pinchbench
#                          ${SLURM_SUBMIT_DIR:-$PWD}/.pinchbench-work/job-$SLURM_JOB_ID
#                          ${XDG_CACHE_HOME:-$HOME/.cache}/pinchbench/work   (no SLURM)
#   PINCHBENCH_CACHE_DIR   default $PINCHBENCH_WORKDIR/cache. Point it at a
#                          persistent shared dir to reuse vLLM/torch.compile/
#                          Triton caches across jobs (faster vLLM start-up).
#   PINCHBENCH_LOG_DIR     default $PINCHBENCH_WORKDIR/logs (vLLM log -> ./vllm)
#   PINCHBENCH_ISOLATE_OPENCLAW=1
#                          use $PINCHBENCH_WORKDIR/openclaw as OPENCLAW_STATE_DIR,
#                          seeded once from ~/.openclaw (config, workspace,
#                          skills, plugins; old bench/judge agents skipped).
#                          Automatic when ~/.openclaw is not writable.
#   PINCHBENCH_FWS_PORT    fws mock port (default 4100; proxy = port+1)
#
# Variables that are already set to a writable location are left alone, so
# anything exported by the sbatch script wins.

# Only once per process tree.
if [[ "${PINCHBENCH_ENV_DONE:-}" == "1" && -n "${PINCHBENCH_WORKDIR:-}" && -w "${PINCHBENCH_WORKDIR:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_pb_log() { echo "[pinchbench_env] $*" >&2; }

# _pb_writable_dir DIR -> 0 if DIR exists (or can be created) and is writable
_pb_writable_dir() {
  [[ -n "$1" ]] || return 1
  mkdir -p "$1" 2>/dev/null || return 1
  [[ -d "$1" && -w "$1" && -x "$1" ]]
}

# _pb_dir VAR DEFAULT -> keep $VAR if it is a writable dir, else export DEFAULT
_pb_dir() {
  local var="$1" def="$2" cur="${!1:-}"
  if [[ -n "$cur" ]] && _pb_writable_dir "$cur"; then
    export "${var?}"
    return 0
  fi
  [[ -n "$cur" ]] && _pb_log "$var=$cur is not writable -> $def"
  mkdir -p "$def"
  export "$var=$def"
}

# ---------------------------------------------------------------- work dir
_pb_candidates=()
[[ -n "${PINCHBENCH_WORKDIR:-}" ]] && _pb_candidates+=("$PINCHBENCH_WORKDIR")
[[ -n "${SLURM_TMPDIR:-}" ]] && _pb_candidates+=("$SLURM_TMPDIR/pinchbench")
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _pb_candidates+=("${SLURM_SUBMIT_DIR:-$PWD}/.pinchbench-work/job-${SLURM_JOB_ID}")
fi
_pb_candidates+=("${XDG_CACHE_HOME:-${HOME:-/nonexistent}/.cache}/pinchbench/work")
_pb_candidates+=("$PWD/.pinchbench-work/local-$(id -u)")
_pb_w=""
for _pb_c in "${_pb_candidates[@]}"; do
  if _pb_writable_dir "$_pb_c"; then _pb_w="$_pb_c"; break; fi
  _pb_log "work dir candidate not writable: $_pb_c"
done
if [[ -z "$_pb_w" ]]; then
  _pb_log "ERROR: no writable PINCHBENCH_WORKDIR candidate (${_pb_candidates[*]})"
  return 1 2>/dev/null || exit 1
fi
PINCHBENCH_WORKDIR="$(cd "$_pb_w" && pwd -P)"
export PINCHBENCH_WORKDIR
_pb_cache="${PINCHBENCH_CACHE_DIR:-$PINCHBENCH_WORKDIR/cache}"
_pb_writable_dir "$_pb_cache" || _pb_cache="$PINCHBENCH_WORKDIR/cache"
mkdir -p "$_pb_cache"
export PINCHBENCH_CACHE_DIR="$_pb_cache"
_pb_dir PINCHBENCH_LOG_DIR "$PINCHBENCH_WORKDIR/logs"

# ---------------------------------------------------------------- temp
mkdir -p "$PINCHBENCH_WORKDIR/tmp"
export TMPDIR="$PINCHBENCH_WORKDIR/tmp" TMP="$PINCHBENCH_WORKDIR/tmp" TEMP="$PINCHBENCH_WORKDIR/tmp"

# ---------------------------------------------------------------- HOME
# enroot keeps the host HOME (fine) but plain `docker run --user` / some Pyxis
# setups leave HOME=/root or unset, which this user cannot write.
PINCHBENCH_ORIG_HOME="${PINCHBENCH_ORIG_HOME:-${HOME:-}}"
export PINCHBENCH_ORIG_HOME
if [[ -z "${HOME:-}" || ! -d "$HOME" || ! -w "$HOME" ]]; then
  _pb_log "HOME='${HOME:-}' is not writable -> $PINCHBENCH_WORKDIR/home"
  mkdir -p "$PINCHBENCH_WORKDIR/home"
  export HOME="$PINCHBENCH_WORKDIR/home"
fi

# ---------------------------------------------------------------- OpenClaw state
if [[ -z "${OPENCLAW_STATE_DIR:-}" ]]; then
  # Check the ORIGINAL home: if HOME was swapped above, its ~/.openclaw
  # (config, AGENTS.md, skills) must still be carried over via seeding.
  _pb_src="${PINCHBENCH_ORIG_HOME:-$HOME}/.openclaw"
  if [[ "${PINCHBENCH_ISOLATE_OPENCLAW:-0}" == "1" ]] || ! _pb_writable_dir "$_pb_src"; then
    export OPENCLAW_STATE_DIR="$PINCHBENCH_WORKDIR/openclaw"
    mkdir -p "$OPENCLAW_STATE_DIR"
    if [[ ! -e "$OPENCLAW_STATE_DIR/openclaw.json" && -r "$_pb_src/openclaw.json" ]]; then
      _pb_log "seeding OPENCLAW_STATE_DIR=$OPENCLAW_STATE_DIR from $_pb_src"
      (cd "$_pb_src" && tar -cf - --exclude='./agents/bench-*' --exclude='./agents/*judge*' \
          --exclude='./logs' --exclude='./sessions' --exclude='*.lock' . 2>/dev/null) \
        | (cd "$OPENCLAW_STATE_DIR" && tar -xf -) \
        || _pb_log "WARNING: seeding OpenClaw state from $_pb_src was incomplete"
      # the source may be read-only (e.g. a shared profile); the copy must be ours
      chmod -R u+rwX "$OPENCLAW_STATE_DIR" 2>/dev/null || true
    fi
  fi
else
  _pb_dir OPENCLAW_STATE_DIR "$PINCHBENCH_WORKDIR/openclaw"
fi

# ---------------------------------------------------------------- fws / gh
_pb_dir FWS_DATA_DIR "$PINCHBENCH_WORKDIR/fws"
export PINCHBENCH_FWS_PORT="${PINCHBENCH_FWS_PORT:-4100}"
# Isolated gh config: the benchmark must never pick up the user's real gh login.
_pb_dir GH_CONFIG_DIR "$PINCHBENCH_WORKDIR/gh"
export GH_NO_UPDATE_NOTIFIER=1 GH_PROMPT_DISABLED=1

# ---------------------------------------------------------------- caches
_pb_dir XDG_CACHE_HOME          "$PINCHBENCH_CACHE_DIR/xdg"
_pb_dir VLLM_CACHE_ROOT         "$PINCHBENCH_CACHE_DIR/vllm"
_pb_dir TORCHINDUCTOR_CACHE_DIR "$PINCHBENCH_CACHE_DIR/torchinductor"
_pb_dir TRITON_CACHE_DIR        "$PINCHBENCH_CACHE_DIR/triton"
_pb_dir TORCH_EXTENSIONS_DIR    "$PINCHBENCH_CACHE_DIR/torch_extensions"
_pb_dir CUDA_CACHE_PATH         "$PINCHBENCH_CACHE_DIR/nv"
_pb_dir HF_HOME                 "$PINCHBENCH_CACHE_DIR/huggingface"
_pb_dir MPLCONFIGDIR            "$PINCHBENCH_CACHE_DIR/matplotlib"
_pb_dir NPM_CONFIG_CACHE        "$PINCHBENCH_CACHE_DIR/npm"
_pb_dir UV_CACHE_DIR            "$PINCHBENCH_CACHE_DIR/uv"   # image default /opt/uv/cache is read-only under enroot
export NPM_CONFIG_UPDATE_NOTIFIER=false NO_UPDATE_NOTIFIER=1

# vLLM ZMQ IPC sockets: unix-socket paths must be short (<108 bytes) and must
# not live on a network FS. Prefer node-local /dev/shm.
_pb_ipc="${PINCHBENCH_IPC_DIR:-/dev/shm/pb-$(id -u)-${SLURM_JOB_ID:-$$}}"
if _pb_writable_dir "$_pb_ipc" && (( ${#_pb_ipc} < 60 )); then
  export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-$_pb_ipc}"
else
  export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-$TMPDIR}"
fi

# Playwright browsers are baked into the image (read-only is fine).
if [[ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" && -d /opt/ms-playwright ]]; then
  export PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
fi

export PINCHBENCH_ENV_DONE=1
_pb_log "PINCHBENCH_WORKDIR=$PINCHBENCH_WORKDIR cache=$PINCHBENCH_CACHE_DIR logs=$PINCHBENCH_LOG_DIR HOME=$HOME OPENCLAW_STATE_DIR=${OPENCLAW_STATE_DIR:-$HOME/.openclaw}"
unset _pb_candidates _pb_c _pb_w _pb_cache _pb_src _pb_ipc
