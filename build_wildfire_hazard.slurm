#!/bin/bash
#SBATCH --job-name=wildfire_CHN
#SBATCH --output=wildfire_CHN_%j.out
#SBATCH --error=wildfire_CHN_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ziqianx@stanford.edu

# === 环境 ===
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

WORKDIR=/scratch/users/ziqianx/wildfire_CHN_output
mkdir -p $WORKDIR
cd $WORKDIR

DATA_PATH=/scratch/users/ziqianx/fire_archive_M-C61_680419.csv

# === 使用 mamba run 运行 Python 脚本 ===
ENV_NAME="${ENV_NAME:-climada_env}"
ENV_RUNNER=()
if command -v mamba >/dev/null 2>&1; then
  ENV_RUNNER=(mamba run -n "$ENV_NAME")
elif command -v conda >/dev/null 2>&1; then
  ENV_RUNNER=(conda run -n "$ENV_NAME")
elif command -v micromamba >/dev/null 2>&1; then
  ENV_RUNNER=(micromamba run -n "$ENV_NAME")
else
  echo "❌ 未找到 mamba/conda/micromamba，无法进入 Python 环境：$ENV_NAME"
  exit 2
fi

"${ENV_RUNNER[@]}" python3 <<'PYCODE'

import pandas as pd
import numpy as np
from climada_petals.hazard import WildFire

print("=== Loading FIRMS data... ===")
firms = pd.read_csv(
    '/scratch/users/ziqianx/fire_archive_M-C61_680419.csv',
    low_memory=False
)

# --- 1) 基础清洗：确保关键列存在并可解析 ---
# MODIS C6.1: confidence 为 0–100；VIIRS 有时是 'low/nominal/high'，这里统一成数值阈值
if firms['confidence'].dtype == object:
    conf_map = {'low': 10, 'nominal': 50, 'high': 100}
    firms['confidence'] = firms['confidence'].str.lower().map(conf_map).fillna(0)

# 解析日期；CLIMADA 依赖日期来聚类到火季
if 'acq_date' not in firms.columns:
    raise ValueError("FIRMS 数据缺少 'acq_date' 列，无法识别火季。")
firms['acq_date'] = pd.to_datetime(firms['acq_date'], errors='coerce')
firms = firms.dropna(subset=['acq_date'])

print(f"Loaded {len(firms):,} records from FIRMS data.")


# --- 2) 过滤置信度、空间子集（中国大陆/港澳台可按需调整）、时间范围 ---
firms = firms[firms['confidence'] >= 80]

# 年份限制：与输出文件名一致
YEAR_START, YEAR_END = 2001, 2025
firms = firms[(firms['acq_date'].dt.year >= YEAR_START) & (firms['acq_date'].dt.year <= YEAR_END)]
firms["acq_date"] = firms["acq_date"].astype(str)


print(f"After filtering: {len(firms):,} points in China ({YEAR_START}-{YEAR_END}) with high confidence.")

if firms.empty:
    raise ValueError("筛选后数据为空：请检查时间范围/空间范围/置信度阈值。")
    
# --- 3) 生成历史火季（关键修复：用类方法 + 指定半球 + 年份范围）---
# 注意：WildFire.from_hist_fire_seasons_FIRMS 是 classmethod，应直接在类上调用
# 半球：China 属于北半球（NHS），影响火季起止的归属
print("=== Building historical fire seasons (CLIMADA WildFire) ===")
wf = WildFire.from_hist_fire_seasons_FIRMS(
    firms,
    centr_res_factor=1/2.0
)

print(f"Identified {len(getattr(wf, 'hist_fire_seasons', []))} season raw groups (if kept).")
print(f"Generated {len(wf.event_id)} wildfire seasons.")
print("Event names (years):", wf.event_name[:10], "..." if len(wf.event_name) > 10 else "")    

# --- 4) 保存输出 ---
output_file = 'WildFire_CHN_2020_2025_3.h5'
wf.write_hdf5(output_file)
print(f"✅ WildFire hazard object saved to {output_file}")
PYCODE

echo "=== Job finished at $(date) ==="
