#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "PID file not found: $PID_FILE" >&2
  exit 1
fi

PID=$(cat "$PID_FILE")
if [[ -z "$PID" ]]; then
  echo "PID file is empty." >&2
  exit 1
fi

if kill "$PID" >/dev/null 2>&1; then
  echo "Stopped vLLM server (PID: $PID)"
  rm -f "$PID_FILE"
else
  echo "Failed to stop PID $PID (already stopped?)" >&2
fi
