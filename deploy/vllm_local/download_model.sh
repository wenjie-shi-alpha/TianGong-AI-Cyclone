#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

if [[ "${MODEL_SOURCE}" != "modelscope" ]]; then
  echo "MODEL_SOURCE must be 'modelscope' for this folder." >&2
  exit 1
fi

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID is empty. Set it in deploy/vllm_local/config.env" >&2
  exit 1
fi

python3 - <<'PY'
import os
import sys
from pathlib import Path

model_id = os.environ.get("MODEL_ID")
model_dir = os.environ.get("MODEL_DIR")
cache_dir = os.environ.get("MS_CACHE_DIR")

try:
    from modelscope.hub.snapshot_download import snapshot_download
except Exception:
    print("modelscope is not installed. Run: pip install modelscope", file=sys.stderr)
    raise SystemExit(1)

kwargs = {"model_id": model_id}
if model_dir:
    kwargs["local_dir"] = model_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
elif cache_dir:
    kwargs["cache_dir"] = cache_dir

path = snapshot_download(**kwargs)
print(f"Model downloaded to: {path}")
PY
