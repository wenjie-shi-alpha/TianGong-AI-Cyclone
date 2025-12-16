#!/bin/bash
#SBATCH --job-name=cyclone_track_extract
#SBATCH --output=cyclone_track_extract_%j.out
#SBATCH --error=cyclone_track_extract_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@stanford.edu

set -euo pipefail

# === 环境（参考 build_wildfire_hazard.slurm）===
# Stanford HPC 常见：module load mamba，然后用 mamba run 进入环境
module load mamba

echo "=== Job started on $(hostname) at $(date) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
command -v mamba >/dev/null 2>&1 && echo "mamba: $(command -v mamba)" || true

# 多 NC/并发下载时可能触发 “too many open files”
ulimit -n 4096 2>/dev/null || true

# === 目录结构适配 ===
# 你的目录结构（截图）：slurm 脚本放在父目录，项目在 TianGong-AI-Cyclone/ 内。
# 这里自动根据脚本位置推断 PROJECT_DIR；也支持 sbatch --export=PROJECT_DIR=... 手工覆盖。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/src" && -f "$SCRIPT_DIR/requirements.txt" ]]; then
    PROJECT_DIR="$SCRIPT_DIR"
  elif [[ -d "$SCRIPT_DIR/TianGong-AI-Cyclone/src" && -f "$SCRIPT_DIR/TianGong-AI-Cyclone/requirements.txt" ]]; then
    PROJECT_DIR="$SCRIPT_DIR/TianGong-AI-Cyclone"
  else
    PROJECT_DIR="$HOME/TianGong-AI-Cyclone"
  fi
fi

if [[ ! -d "$PROJECT_DIR/src" ]]; then
  echo "❌ 未找到项目目录（缺少 src/）：PROJECT_DIR=$PROJECT_DIR"
  echo "   你可以通过 sbatch --export=PROJECT_DIR=/path/to/TianGong-AI-Cyclone 指定。"
  exit 2
fi

# 输出位置与 colab.ipynb 一致：默认写入 <PROJECT_DIR>/colab_outputs
# 注意：environment_extractor 的输出目录是相对 CWD 的固定名称：
#   data/nc_files, track_single, final_single_output
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/colab_outputs}"

# mamba/conda 环境名（需包含 requirements.txt 里的依赖）
ENV_NAME="${ENV_NAME:-tianGong_cyclone}"

# 输入：NC URL 列表 CSV（含 s3_url）
NC_URLS_CSV="${NC_URLS_CSV:-$PROJECT_DIR/output/nc_file_urls_new.csv}"

# 输入：初始点 CSV（matched_cyclone_tracks.csv）
INITIALS_CSV="${INITIALS_CSV:-$PROJECT_DIR/input/matched_cyclone_tracks.csv}"

# 并行进程数（默认=分配 CPU）
PROCESSES="${PROCESSES:-${SLURM_CPUS_PER_TASK:-16}}"

# 仅跑前 N 个（空或 0 表示全跑）
LIMIT="${LIMIT:-0}"

# 行为开关：1=开启，0=关闭
AUTO_TRACK="${AUTO_TRACK:-1}"      # 无轨迹则自动追踪（推荐 1）
CONCISE_LOG="${CONCISE_LOG:-1}"    # 精简日志（推荐 1）
KEEP_NC="${KEEP_NC:-0}"            # 保留下载的 NC（调试用）

# 避免多进程 + BLAS 线程过度抢占
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT"
cd "$OUTPUT_ROOT"

mkdir -p inputs logs
if [[ ! -f "$NC_URLS_CSV" ]]; then
  echo "❌ NC_URLS_CSV 不存在: $NC_URLS_CSV"
  exit 2
fi
if [[ ! -f "$INITIALS_CSV" ]]; then
  echo "❌ INITIALS_CSV 不存在: $INITIALS_CSV"
  exit 2
fi
cp -f "$NC_URLS_CSV" "inputs/$(basename "$NC_URLS_CSV")"
cp -f "$INITIALS_CSV" "inputs/$(basename "$INITIALS_CSV")"

NC_URLS_LOCAL="inputs/$(basename "$NC_URLS_CSV")"
INITIALS_LOCAL="inputs/$(basename "$INITIALS_CSV")"

echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "ENV_NAME=$ENV_NAME"
echo "NC_URLS_LOCAL=$NC_URLS_LOCAL"
echo "INITIALS_LOCAL=$INITIALS_LOCAL"
echo "PROCESSES=$PROCESSES"
echo "LIMIT=$LIMIT"
echo "AUTO_TRACK=$AUTO_TRACK CONCISE_LOG=$CONCISE_LOG KEEP_NC=$KEEP_NC"

echo "=== Preflight: 统计将处理/跳过的条目 ==="
TODO_COUNT="$(
mamba run -n "$ENV_NAME" python3 - "$NC_URLS_LOCAL" "$LIMIT" <<'PY'
from __future__ import annotations

from pathlib import Path
import os
import sys

csv_path = Path(sys.argv[1])
limit = int(sys.argv[2] or "0")
final_dir = Path("final_single_output")

try:
    import pandas as pd
except Exception as e:
    print(f"❌ pandas 不可用: {e}", file=sys.stderr)
    raise

try:
    from environment_extractor.pipeline import _index_existing_json  # type: ignore
except Exception as e:
    print(f"❌ 无法导入 environment_extractor（检查环境/依赖）: {e}", file=sys.stderr)
    raise

if not csv_path.exists():
    raise FileNotFoundError(f"CSV 不存在: {csv_path.resolve()}")

df = pd.read_csv(csv_path)
if "s3_url" not in df.columns:
    raise ValueError("CSV 缺少 s3_url 列")

if limit > 0:
    df = df.head(limit)

index = _index_existing_json(final_dir) if final_dir.exists() else {}
stems = df["s3_url"].map(lambda u: Path(str(u)).stem)
done_mask = stems.map(lambda s: bool(index.get(s)))
done = int(done_mask.sum())
todo = int((~done_mask).sum())

print(f"CSV: {csv_path} | entries={len(df)} | done={done} | todo={todo}", file=sys.stderr)
if todo == 0:
    print("⏹️ 全部条目已有 final_single_output JSON，将跳过本次任务。", file=sys.stderr)
else:
    print("示例待处理 NC stems:", ", ".join(stems[~done_mask].astype(str).head(5).tolist()), file=sys.stderr)

print(todo)
PY
)"

if [[ "${TODO_COUNT}" == "0" ]]; then
  echo "=== Preflight 结论：todo=0，直接结束（避免重复处理）==="
  echo "=== Job finished at $(date) ==="
  exit 0
fi

cmd=(python3 -u "$PROJECT_DIR/src/extractSyst.py"
  --csv "$NC_URLS_LOCAL"
  --initials "$INITIALS_LOCAL"
  --processes "$PROCESSES"
)

if [[ "${LIMIT}" != "0" && -n "${LIMIT}" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ "$AUTO_TRACK" == "1" ]]; then
  cmd+=(--auto)
fi
if [[ "$CONCISE_LOG" == "1" ]]; then
  cmd+=(--concise-log)
fi
if [[ "$KEEP_NC" == "1" ]]; then
  cmd+=(--keep-nc)
fi

echo "=== Running command ==="
printf '  %q' mamba run -n "$ENV_NAME" "${cmd[@]}"
echo

# 同时写入日志文件，便于 tail -f
set +e
mamba run -n "$ENV_NAME" "${cmd[@]}" 2>&1 | tee -a "logs/run_${SLURM_JOB_ID}.log"
status=${PIPESTATUS[0]}
set -e

echo "=== Pipeline exit code: $status ==="
echo "=== Job finished at $(date) ==="
exit "$status"
