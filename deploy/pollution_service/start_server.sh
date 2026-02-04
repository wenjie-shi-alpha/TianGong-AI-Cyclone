#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

mkdir -p "$LOG_DIR"

if [[ -n "${API_KEY}" ]]; then
  export API_KEY
fi

ARGS=("uvicorn" "app:app" "--host" "$HOST" "--port" "$PORT")

if [[ "${1:-}" == "--fg" ]]; then
  echo "Starting pollution service (foreground)..."
  exec "${ARGS[@]}"
fi

LOG_FILE="${LOG_DIR}/service.log"
nohup "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$PID_FILE"

ACCESS_HOST="$HOST"
if [[ "$ACCESS_HOST" == "0.0.0.0" ]]; then
  ACCESS_HOST="127.0.0.1"
fi

echo "Service started (PID: $PID)"
echo "Logs: $LOG_FILE"
echo "Access URL: http://${ACCESS_HOST}:${PORT}"
