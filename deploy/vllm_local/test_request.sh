#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

ACCESS_HOST="$HOST"
if [[ "$ACCESS_HOST" == "0.0.0.0" ]]; then
  ACCESS_HOST="127.0.0.1"
fi

MODEL_FOR_REQUEST="$MODEL_ID"
if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  MODEL_FOR_REQUEST="$SERVED_MODEL_NAME"
fi

URL="http://${ACCESS_HOST}:${PORT}/v1/chat/completions"

PAYLOAD=$(cat <<JSON
{
  "model": "${MODEL_FOR_REQUEST}",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a one-sentence summary of air pollution dispersion."}
  ],
  "temperature": 0.2
}
JSON
)

if [[ -n "${API_KEY}" ]]; then
  curl -sS -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
    -d "$PAYLOAD" "$URL"
else
  curl -sS -H "Content-Type: application/json" -d "$PAYLOAD" "$URL"
fi
