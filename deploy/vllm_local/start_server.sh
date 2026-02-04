#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID is empty. Set it in deploy/vllm_local/config.env" >&2
  exit 1
fi

MODEL_PATH="${MODEL_DIR:-$MODEL_ID}"
SERVED_NAME="${SERVED_MODEL_NAME:-}"

mkdir -p "${LOG_DIR}"

ARGS=("vllm" "serve" "$MODEL_PATH" "--host" "$HOST" "--port" "$PORT" "--dtype" "$DTYPE")

if [[ -n "${SERVED_NAME}" ]]; then
  ARGS+=("--served-model-name" "$SERVED_NAME")
fi
if [[ -n "${API_KEY}" ]]; then
  ARGS+=("--api-key" "$API_KEY")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
fi
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a EXTRA <<< "$EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

if [[ "${1:-}" == "--fg" ]]; then
  echo "Starting vLLM server (foreground)..."
  exec "${ARGS[@]}"
fi

LOG_FILE="${LOG_DIR}/vllm_server.log"
nohup "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "$SERVER_PID" > "$PID_FILE"

ACCESS_HOST="$HOST"
if [[ "$ACCESS_HOST" == "0.0.0.0" ]]; then
  ACCESS_HOST="127.0.0.1"
fi

echo "vLLM server started (PID: $SERVER_PID)"
echo "Logs: $LOG_FILE"
echo "Access URL: http://${ACCESS_HOST}:${PORT}/v1"
