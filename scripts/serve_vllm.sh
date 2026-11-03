#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on a LOCAL model in the background,
# wait until /health is OK, then run the given command (e.g. PinchBench).
# The server is stopped when the command exits.
#
# Usage:
#   serve_vllm.sh                      # serve only (stay in foreground)
#   serve_vllm.sh ./scripts/run.sh --model gemma-4-26b-a4b-it \
#       --base-url http://127.0.0.1:8000/v1 --api-key EMPTY --no-upload
#
# Env (defaults):
#   MODEL_PATH=/models/gemma-4-26B-A4B-it   local HF-format dir (mounted)
#   SERVED_MODEL_NAME=gemma-4-26b-a4b-it    must equal PinchBench --model (no "/")
#   TP=8  DP=1                              tensor / data parallel size
#   VLLM_HOST=127.0.0.1  VLLM_PORT=8000
#   MAX_MODEL_LEN=200000  GPU_MEM_UTIL=0.90  MAX_NUM_SEQS=(vLLM default)
#   KV_CACHE_DTYPE=auto                     e.g. fp8
#   ENABLE_THINKING=1                       Gemma-4 thinking on by default
#   CHAT_TEMPLATE=$VLLM_GEMMA4_CHAT_TEMPLATE
#   VLLM_API_KEY=                           if set, server requires this key
#   VLLM_LOG_DIR=$PINCHBENCH_LOG_DIR/vllm  HEALTH_TIMEOUT=1800 (s)
#   PINCHBENCH_WORKDIR=...                  all temp/cache/state paths, see pinchbench_env.sh
#   VLLM_EXTRA_ARGS=""                      appended verbatim to `vllm serve`
set -euo pipefail

# Temp / cache / state / log paths: one root ($PINCHBENCH_WORKDIR), never a
# hard-coded /tmp; works as non-root with a read-only image.
_pb_self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _pb_env in "$_pb_self_dir/pinchbench_env.sh" /usr/local/bin/pinchbench_env.sh; do
  # shellcheck disable=SC1090
  if [[ -f "$_pb_env" ]]; then source "$_pb_env"; break; fi
done

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-26b-a4b-it}"
TP="${TP:-8}"
DP="${DP:-1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${VLLM_GEMMA4_CHAT_TEMPLATE:-/vllm-workspace/examples/tool_chat_template_gemma4.jinja}}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${PINCHBENCH_LOG_DIR:-${TMPDIR:-/tmp}}/vllm}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[serve_vllm] MODEL_PATH '$MODEL_PATH' is not a directory (mount it, e.g. --container-mounts=/path/on/host:/models:ro)" >&2
  exit 1
fi

# Local model only: never reach out to the Hugging Face Hub.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# PinchBench's custom provider reads ${OPENAI_API_KEY}; vLLM ignores it unless VLLM_API_KEY is set.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${VLLM_API_KEY:-EMPTY}}"

mkdir -p "$VLLM_LOG_DIR"
LOG_FILE="$VLLM_LOG_DIR/vllm-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S).log"

args=(
  serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$VLLM_HOST" --port "$VLLM_PORT"
  --tensor-parallel-size "$TP"
  --data-parallel-size "$DP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --enable-prefix-caching
  --async-scheduling
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
)
[[ -f "$CHAT_TEMPLATE" ]] && args+=(--chat-template "$CHAT_TEMPLATE")
[[ "$ENABLE_THINKING" == "1" ]] && args+=(--default-chat-template-kwargs '{"enable_thinking": true}')
[[ -n "${MAX_NUM_SEQS:-}" ]] && args+=(--max-num-seqs "$MAX_NUM_SEQS")
[[ -n "${VLLM_API_KEY:-}" ]] && args+=(--api-key "$VLLM_API_KEY")
# shellcheck disable=SC2206
[[ -n "$VLLM_EXTRA_ARGS" ]] && args+=($VLLM_EXTRA_ARGS)

echo "[serve_vllm] vllm ${args[*]}"
echo "[serve_vllm] log: $LOG_FILE"
nohup vllm "${args[@]}" >"$LOG_FILE" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve_vllm] stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

BASE="http://127.0.0.1:${VLLM_PORT}"
[[ "$VLLM_HOST" != "0.0.0.0" && "$VLLM_HOST" != "127.0.0.1" ]] && BASE="http://${VLLM_HOST}:${VLLM_PORT}"

echo "[serve_vllm] waiting for $BASE/health (timeout ${HEALTH_TIMEOUT}s)..."
start=$(date +%s)
until curl -sf "$BASE/health" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve_vllm] vLLM exited during startup; last log lines:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if (( $(date +%s) - start > HEALTH_TIMEOUT )); then
    echo "[serve_vllm] timed out waiting for /health; last log lines:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 5
done
echo "[serve_vllm] vLLM ready after $(( $(date +%s) - start ))s"
curl -sf -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" "$BASE/v1/models" || true
echo
export VLLM_BASE_URL="$BASE/v1"
echo "[serve_vllm] OpenAI base URL: $VLLM_BASE_URL  model: $SERVED_MODEL_NAME"

if [[ $# -eq 0 ]]; then
  echo "[serve_vllm] no command given; serving until vLLM exits (Ctrl-C / scancel to stop)"
  wait "$VLLM_PID"
else
  set +e
  "$@"
  rc=$?
  set -e
  exit "$rc"
fi
