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
# 有些集群提供 `mamba` module；也有集群只提供 `conda/miniconda/anaconda`。
# 这里尽量加载常见 module，但不会因为缺少某个 module 而直接失败。
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck source=/usr/share/Modules/init/bash
    source /usr/share/Modules/init/bash
  fi
fi

if command -v module >/dev/null 2>&1; then
  for mod in mamba micromamba miniconda3 anaconda3 miniforge3 mambaforge; do
    if module --ignore_cache load "$mod" >/dev/null 2>&1 || module load "$mod" >/dev/null 2>&1; then
      echo "Loaded module: $mod"
      break
    fi
  done
fi

echo "=== Job started on $(hostname) at $(date) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
echo "SCRIPT_FILE=${BASH_SOURCE[0]}"
command -v sha1sum >/dev/null 2>&1 && echo "SCRIPT_SHA1=$(sha1sum "${BASH_SOURCE[0]}" | awk '{print $1}')" || true
command -v mamba >/dev/null 2>&1 && echo "mamba: $(command -v mamba)" || true
command -v conda >/dev/null 2>&1 && echo "conda: $(command -v conda)" || true
command -v micromamba >/dev/null 2>&1 && echo "micromamba: $(command -v micromamba)" || true

# 多 NC/并发下载时可能触发 “too many open files”
ulimit -n 4096 2>/dev/null || true

# === 目录结构适配 ===
# Open OnDemand/Job Composer 会把脚本复制到 OOD 的 job 目录执行（如 ~/ondemand/.../myjobs/...）。
# 因此不能假设脚本所在目录就是项目根目录。这里按一组常见候选路径自动定位项目根目录；
# 也支持在提交时显式设置 PROJECT_DIR（推荐）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_PROJECT_DIR="${PROJECT_DIR:-}"
if [[ -n "$USER_PROJECT_DIR" && -d "$USER_PROJECT_DIR" ]]; then
  USER_PROJECT_DIR="$(cd "$USER_PROJECT_DIR" && pwd)"
fi

PROJECT_DIR=""
declare -a CANDIDATES=()
if [[ -n "$USER_PROJECT_DIR" ]]; then
  CANDIDATES+=("$USER_PROJECT_DIR")
fi
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  CANDIDATES+=("${SLURM_SUBMIT_DIR}" "${SLURM_SUBMIT_DIR}/TianGong-AI-Cyclone")
fi
CANDIDATES+=(
  "$SCRIPT_DIR"
  "$SCRIPT_DIR/TianGong-AI-Cyclone"
  "$PWD"
  "$PWD/TianGong-AI-Cyclone"
  "$HOME/TianGong-AI-Cyclone"
  "/scratch/users/${USER}/TianGong-AI-Cyclone"
  "/scratch/${USER}/TianGong-AI-Cyclone"
)

for cand in "${CANDIDATES[@]}"; do
  [[ -z "$cand" ]] && continue
  if [[ -d "$cand/src" && -f "$cand/requirements.txt" ]]; then
    PROJECT_DIR="$cand"
    break
  fi
done

if [[ -z "$PROJECT_DIR" || ! -d "$PROJECT_DIR/src" ]]; then
  echo "❌ 未找到项目目录（需要包含 src/ 和 requirements.txt）。"
  echo "   SCRIPT_DIR=$SCRIPT_DIR"
  echo "   PWD=$PWD"
  echo "   SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
  echo "   尝试过的候选路径："
  for cand in "${CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -d "$cand" ]]; then
      echo "     - $cand (exists)"
    else
      echo "     - $cand"
    fi
  done
  echo "   解决：在提交时设置 PROJECT_DIR，例如："
  echo "     export PROJECT_DIR=/scratch/users/${USER}/TianGong-AI-Cyclone"
  echo "     # 或 sbatch --export=PROJECT_DIR=/scratch/users/${USER}/TianGong-AI-Cyclone run_cyclone_track_extract.slurm"
  exit 2
fi

# 输出位置与 colab.ipynb 一致：默认写入 <PROJECT_DIR>/colab_outputs
# 注意：environment_extractor 的输出目录是相对 CWD 的固定名称：
#   data/nc_files, track_single, final_single_output
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/colab_outputs}"

# mamba/conda 环境名（需包含 requirements.txt 里的依赖）
ENV_NAME="${ENV_NAME:-tianGong_cyclone}"

# 选择环境运行器：
# 你当前报错的 prefix 位于 ~/.local/share/mamba/envs/，这通常是 micromamba 的默认根目录；
# 即使你在 ~/miniforge3 里用 conda 创建过环境，mamba/micromamba 仍可能因为 root prefix 不一致而找不到。
# 为了避免这个坑，这里默认直接用 `conda run`（最符合你“能 conda activate”的现实情况）。
ENV_RUNNER=()

# 可选：强制指定 runner（二选一：conda/mamba/micromamba）
FORCE_RUNNER="${FORCE_RUNNER:-}"
if [[ -n "$FORCE_RUNNER" ]]; then
  case "$FORCE_RUNNER" in
    conda) ENV_RUNNER=(conda run -n "$ENV_NAME") ;;
    mamba) ENV_RUNNER=(mamba run -n "$ENV_NAME") ;;
    micromamba) ENV_RUNNER=(micromamba run -n "$ENV_NAME") ;;
    *)
      echo "❌ FORCE_RUNNER=$FORCE_RUNNER 不支持（仅 conda/mamba/micromamba）"
      exit 2
      ;;
  esac
elif command -v conda >/dev/null 2>&1; then
  ENV_RUNNER=(conda run -n "$ENV_NAME")
elif command -v mamba >/dev/null 2>&1; then
  ENV_RUNNER=(mamba run -n "$ENV_NAME")
elif command -v micromamba >/dev/null 2>&1; then
  ENV_RUNNER=(micromamba run -n "$ENV_NAME")
else
  echo "❌ 未找到 conda/mamba/micromamba，无法进入 Python 环境：$ENV_NAME"
  echo "   解决：module load (mini)conda/anaconda/miniforge，或把 conda 加入 PATH。"
  exit 2
fi

echo "ENV_RUNNER=${ENV_RUNNER[*]}"

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
"${ENV_RUNNER[@]}" python3 - "$NC_URLS_LOCAL" "$LIMIT" <<'PY'
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
printf '  %q' "${ENV_RUNNER[@]}" "${cmd[@]}"
echo

# 同时写入日志文件，便于 tail -f
set +e
"${ENV_RUNNER[@]}" "${cmd[@]}" 2>&1 | tee -a "logs/run_${SLURM_JOB_ID}.log"
status=${PIPESTATUS[0]}
set -e

echo "=== Pipeline exit code: $status ==="
echo "=== Job finished at $(date) ==="
exit "$status"
