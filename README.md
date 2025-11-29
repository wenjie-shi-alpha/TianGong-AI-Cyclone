
# TianGong AI Cyclone

## src 目录模块说明

- `environment_extractor/`：一体化的热带气旋环境分析流水线，涵盖命令行入口、下载与追踪编排（`cli.py`、`pipeline.py`）、形状分析工具（`shape_analysis.py`）以及对外部依赖的封装（`deps.py`、`workflow_utils.py`）。
- `initial_tracker/`：重构后的初始点追踪内核，负责数据批处理与坐标换算（`batching.py`、`geo.py`）、异常处理（`exceptions.py`）以及核心的逐时追踪逻辑（`tracker.py`、`workflow.py`）。
- `extractSyst.py`：兼容历史用法的入口脚本，转调 `environment_extractor` 完成“下载→追踪→环境分析”的批处理流程，并处理缺失依赖提示。
- `initialTracker.py`：为旧版脚本提供的薄封装，暴露与早期实现一致的命令行接口，内部直接调用 `initial_tracker` 包的组件。
- `process.py`：对 NOAA OAR MLWP 公共 S3 桶的匿名下载工具，提供 `download_from_noaa` 函数以缓存或临时保存指定 NetCDF 文件。
- `generate_nc_urls.py`：根据轨迹 CSV 中的时间戳，从多个模式前缀下枚举 S3 目录，生成可下载的 NetCDF 文件列表及元数据（CSV 输出）。
- `list_all_nc_files.py`：遍历指定模式前缀的全部 NetCDF 对象，支持按年份过滤，并将结果写入 `output/all_nc_files.csv` 供批量分析或审计使用。

## Google Earth Engine 版 GFS 处理脚本

- `src/gee_gfs_pipeline.py` 直接在 Google Earth Engine (GEE) 上处理 `NOAA/GFS0P25` 历史预报集：追踪逻辑沿用 `initial_tracker` 的最低压差捕获策略，环境量提取参考了 `environment_extractor` 的环平均分析。
- 重量运算全部由 GEE 完成（通过 `reduceRegion` / `pixelLonLat` 等服务器端算子）；本地只接收每个时次的统计值，原始格点不会下载，符合“算法在云端执行、结果再拉回”的要求。
- 依赖：`earthengine-api`（已写入 `requirements.txt`）。首次运行需在 shell 中执行 `earthengine authenticate` 或让脚本自动触发交互授权。
- 典型用法：
  ```bash
  python3 src/gee_gfs_pipeline.py \
    --initials-csv input/matched_cyclone_tracks.csv \
    --storm-id 2018243N15262 \
    --start-time 2018-09-12T00:00Z \
    --temporal-span-hours 144 \
    --output-dir colab_outputs/gee_pipeline
  ```
  输出会按照台风 ID 存放在 `colab_outputs/gee_pipeline/<storm_id>/`，同目录下同时保存 `*_gee_track.csv` 与 `*_gee_track.json`，内容包括逐时位置、海平面气压、核心风速、外围压差以及水汽/温度/湿度等环境要素摘要。
- 常用参数：
  - `--analysis-only`：仅使用 `forecast_hour==0` 的分析场；若需要完整预报序列，可去掉该标志并通过 `--max-forecast-hour` 控制可接受的最大小时数。
  - `--max-steps` 与 `--temporal-span-hours` 控制追踪长度；`--spatial-pad-deg` 用于设定检索区域。
- 运行后即可直接下载（或同步到云端目录）这些已经分析好的 CSV/JSON，整个过程不会把原始 GFS NetCDF/格点拉回本地。

## Env Preparing

Setup `venv`:

```bash

sudo apt-get install python3.12-dev
sudo apt-get install nvidia-cuda-toolkit

python3.12 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
python.exe -m pip install --upgrade pip

pip install --upgrade pip

pip install --upgrade pip
pip install -r requirements.txt

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --upgrade

pip install -r requirements_freeze.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip freeze > requirements_freeze.txt


aws s3 ls --no-sign-request --region us-east-1 s3://noaa-oar-mlwp-data/

# 使用默认数据集 (western_pacific_typhoons_superfast.csv)
python3 src/extractSyst.py --csv output/nc_file_urls.csv --limit 10 --processes 15 --concise-log --auto
python3 src/extractSyst.py --csv output/nc_file_urls.csv --limit 1 --auto --no-clean

nohup python3 src/extractSyst.py --csv output/nc_file_urls.csv --auto --concise-log --processes 15 > run.log 2>&1 &

# 使用新数据集 (matched_cyclone_tracks.csv)
python3 src/extractSyst.py --csv output/nc_file_urls_new.csv --initials input/matched_cyclone_tracks.csv --limit 10 --processes 15 --concise-log --auto

nohup python3 src/extractSyst.py --csv output/nc_file_urls_new.csv --initials input/matched_cyclone_tracks.csv --auto --concise-log --processes 15 > run_matched.log 2>&1 &

```

Auto lint:
```bash
black .
```

## Run with PM2

```bash

npm i -g pm2

pm2 start ecosystem.config.json

pm2 start ecosystem.quatro.json

pm2 restart all

pm2 status

pm2 restart unstructured-gunicorn
pm2 stop unstructured-gunicorn
pm2 delete unstructured-gunicorn

pm2 logs unstructured-gunicorn
```

## Processing & Skip Logic (extractSyst)

The script `src/extractSyst.py` has built‑in logic to avoid repeating expensive work. There is currently **no `--force` flag**; recomputation is achieved by removing existing outputs. Behavior summary:

1. Batch iteration: When you run for multiple NetCDF files (from `--csv` + `--limit` or a directory), the script loops through candidates and processes each independently.
2. Output skip: If one or more JSON analysis files already exist in `final_output/` for an NC file (pattern: `<ncstem>_TC_Analysis_*.json`, non‑empty >10 bytes), that NC file is skipped and the loop continues to the next one (it does NOT exit early).
3. Internal double check: Inside the analysis function there is a second safeguard that exits early if it detects that all expected JSON outputs for that file already exist.
4. Track files: The script attempts to match an existing track CSV in `track_output/` using a forecast tag extracted from the NC filename. If absent and `--auto` is supplied, it will generate the track on the fly.
5. Cleaning NC files: By default NC files may be removed after successful processing unless you specify `--no-clean` (or `--keep-nc` depending on current options—use the retention flag if you want to preserve downloads).

### Recomputing a file deliberately
Because there is no `--force` option, to re-run analysis for a specific NetCDF file:

```bash
rm final_output/<ncstem>_TC_Analysis_*.json
python3 src/extractSyst.py --nc data/nc_files/<file>.nc --auto
```

If you want to redo a batch, remove (or move) the corresponding JSON outputs first:

```bash
mkdir -p backup_outputs
mv final_output/AURO_v100_GFS_20250610*_TC_Analysis_*.json backup_outputs/
python3 src/extractSyst.py --csv output/nc_file_urls.csv --limit 500 --auto
```

### Practical tips
- Use smaller `--limit` for quick smoke tests while building.
- Keep an eye on `run.log` (if using nohup) to confirm skip vs processed counts.
- To conserve space but allow recomputation later, archive outputs instead of deleting them.

### Example log messages
You will see lines like:
```
Skipping AURO_v100_GFS_2025061000_f000_f240_06: existing final_output JSON detected.
Processed 37 files (skipped 112 already complete).
```
These confirm the skip logic is functioning.

## Logging Modes

- 默认模式会打印完整的流水线细节，配合 `--processes` 使用时，每个子任务还会在终端输出进度。
- 传入 `--concise-log` 可切换到精简模式，只保留必要的摘要统计；处理流程仍会在失败时输出错误信息。
- 当启用多进程(`--processes > 1`)时，每个 NC 文件的详细日志会写入 `final_single_output/logs/<nc文件名>.log`；若启用 `--concise-log`，则不再生成这些详细日志以减少写入开销。
- 示例：`python3 src/extractSyst.py --csv output/nc_file_urls.csv --processes 4 --auto --concise-log`

## 生成预报样本
```bash
python3 src/generate_forecast_dataset.py --limit 3 --samples-per-forecast 3
```
