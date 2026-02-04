#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

ACCESS_HOST="$HOST"
if [[ "$ACCESS_HOST" == "0.0.0.0" ]]; then
  ACCESS_HOST="127.0.0.1"
fi

URL="http://${ACCESS_HOST}:${PORT}/high-pressure"

PAYLOAD=$(cat <<JSON
{
  "lat": [20, 21, 22],
  "lon": [110, 111, 112],
  "center_lat": 21,
  "center_lon": 111,
  "z500": [
    [5880, 5890, 5885],
    [5895, 5920, 5890],
    [5882, 5892, 5887]
  ]
}
JSON
)

if [[ -n "${API_KEY}" ]]; then
  curl -sS -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
    -d "$PAYLOAD" "$URL"
else
  curl -sS -H "Content-Type: application/json" -d "$PAYLOAD" "$URL"
fi
