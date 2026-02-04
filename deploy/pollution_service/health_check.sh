#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

ACCESS_HOST="$HOST"
if [[ "$ACCESS_HOST" == "0.0.0.0" ]]; then
  ACCESS_HOST="127.0.0.1"
fi

URL="http://${ACCESS_HOST}:${PORT}/health"

if [[ -n "${API_KEY}" ]]; then
  curl -sS -H "Authorization: Bearer ${API_KEY}" "$URL"
else
  curl -sS "$URL"
fi
