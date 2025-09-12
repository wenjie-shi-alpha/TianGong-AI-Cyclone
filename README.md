
# TianGong AI Cyclone

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

python3 src/extractSyst.py --csv output/nc_file_urls.csv --limit 1 --auto
python3 src/extractSyst.py --csv output/nc_file_urls.csv --limit 1 --auto --no-clean

nohup python3 src/extractSyst.py --csv output/nc_file_urls.csv --limit 500 --auto > run.log 2>&1 &

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

