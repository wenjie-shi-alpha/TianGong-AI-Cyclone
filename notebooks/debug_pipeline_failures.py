# %% [markdown]
# # EC 双源管线失败排查（Jupyter 脚本）
#
# 用途：
# 1. 读取 `pipeline_state_aws.json` / `pipeline_state_tigge.json`
# 2. 做静态产物扫描（不重跑）
# 3. 做环境预检（依赖、凭据、关键导入）
# 4. 重放少量失败样本并抓取真实 `error + traceback`
# 5. 输出按错误聚类的汇总，便于后续针对性修复
#
# 建议先保持默认参数跑一轮，把输出粘贴给我再做下一步修改。

# %% 配置
import os
import sys
import json
import time
import traceback
import importlib
from pathlib import Path
from datetime import datetime

import pandas as pd
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

PROJECT_ROOT = Path("/home/jovyan/TianGong-AI-Cyclone") if Path("/home/jovyan/TianGong-AI-Cyclone").exists() else Path.cwd()
os.chdir(PROJECT_ROOT)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RAW_DIR = PROJECT_ROOT / "data" / "ecmwf_raw"
NC_DIR = PROJECT_ROOT / "data" / "ecmwf_nc"
TRACK_DIR = PROJECT_ROOT / "track_output_cds"
ENV_DIR = PROJECT_ROOT / "output" / "forecast_environment"
SUMMARY_DIR = PROJECT_ROOT / "output" / "cds_notebook_summary"
for p in [RAW_DIR, NC_DIR, TRACK_DIR, ENV_DIR, SUMMARY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

INITIALS_CSV = PROJECT_ROOT / "input" / "matched_cyclone_tracks_2021onwards.csv"
MATCH_TOL_HOURS = 6
STEPS = list(range(0, 241, 6))

# 重放配置（先小样本定位根因）
REPLAY_LIMIT_PER_SOURCE = {
    "aws": 8,
    "tigge": 8,
}
REPLAY_ONLY_KEYS = {
    "aws": [],      # 例如: ["2021-02-14_12"]
    "tigge": [],    # 为空表示自动从 failed 列表截取前 N 个
}
KEEP_FAILED_ARTIFACTS_DURING_REPLAY = True
REPLAY_ORDER = ["tigge", "aws"]  # 优先看 TIGGE（当前是全失败）

STATE_CANDIDATES = {
    "aws": [
        PROJECT_ROOT / "pipeline_state_aws.json",
        PROJECT_ROOT / "output" / "cds_notebook_state" / "pipeline_state_aws.json",
    ],
    "tigge": [
        PROJECT_ROOT / "pipeline_state_tigge.json",
        PROJECT_ROOT / "output" / "cds_notebook_state" / "pipeline_state_tigge.json",
    ],
}

print(f"PROJECT_ROOT={PROJECT_ROOT}")
print(f"INITIALS_CSV exists={INITIALS_CSV.exists()} -> {INITIALS_CSV}")


# %% 工具函数
def pick_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_state(path: Path):
    if path is None or (not path.exists()):
        return {"done": [], "failed": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "done": list(data.get("done", [])),
        "failed": list(data.get("failed", [])),
    }


def parse_key(key: str):
    date_str, hh = key.rsplit("_", 1)
    return date_str, int(hh)


def stem_for(source_tag: str, key: str):
    date_str, hh = parse_key(key)
    ymd = date_str.replace("-", "")
    if source_tag == "aws":
        return f"ecmwf_{ymd}_{hh:02d}"
    return f"tigge_ecmwf_{ymd}_{hh:02d}"


def raw_path_for(source_tag: str, key: str):
    stem = stem_for(source_tag, key)
    if source_tag == "aws":
        return RAW_DIR / f"{stem}.grib2"
    return RAW_DIR / f"{stem}.grib"


def nc_path_for(source_tag: str, key: str):
    stem = stem_for(source_tag, key)
    return NC_DIR / f"{stem}.nc"


def index_track_and_env():
    track_map = {}
    env_map = {}

    for f in TRACK_DIR.glob("track_*_*.csv"):
        name = f.name
        parts = name[:-4].split("_")
        # track_{storm_id}_{stem}.csv -> 从末尾 3 段拼 stem: (ecmwf|tigge_ecmwf)_YYYYMMDD_HH
        if len(parts) < 4:
            continue
        if parts[-3] == "ecmwf":
            stem = "_".join(parts[-3:])
        elif parts[-4] == "tigge" and parts[-3] == "ecmwf":
            stem = "_".join(parts[-4:])
        else:
            continue
        track_map[stem] = track_map.get(stem, 0) + 1

    for f in ENV_DIR.glob("*_TC_Analysis_*.json"):
        name = f.name
        if "_TC_Analysis_" not in name:
            continue
        stem = name.split("_TC_Analysis_", 1)[0]
        env_map[stem] = env_map.get(stem, 0) + 1

    return track_map, env_map


def artifact_snapshot(source_tag: str, key: str, track_map=None, env_map=None):
    stem = stem_for(source_tag, key)
    raw_path = raw_path_for(source_tag, key)
    nc_path = nc_path_for(source_tag, key)

    if track_map is None or env_map is None:
        track_map, env_map = index_track_and_env()

    return {
        "stem": stem,
        "raw_path": str(raw_path),
        "raw_exists": raw_path.exists(),
        "raw_size_mb": round(raw_path.stat().st_size / (1024 * 1024), 3) if raw_path.exists() else 0.0,
        "nc_path": str(nc_path),
        "nc_exists": nc_path.exists(),
        "nc_size_mb": round(nc_path.stat().st_size / (1024 * 1024), 3) if nc_path.exists() else 0.0,
        "track_count": int(track_map.get(stem, 0)),
        "env_count": int(env_map.get(stem, 0)),
    }


def short_tb(tb: str, max_lines: int = 6):
    if not tb:
        return ""
    lines = tb.strip().splitlines()
    return "\n".join(lines[:max_lines])


def guess_stage(error_text: str, snap_after: dict):
    e = (error_text or "").lower()

    # 明确错误优先
    if "missing ~/.ecmwfapirc" in e or "ecmwfapirc" in e:
        return "tigge_auth_config"
    if "tigge pressure-level retrieve failed" in e or "empty tigge response" in e:
        return "tigge_download_or_request"
    if "no available s3 prefix" in e or "no required step available" in e:
        return "aws_data_unavailable_or_s3"
    if "cannot parse required vars" in e:
        return "grib_to_nc_parse"
    if "workererror" in e:
        return "worker_runtime"

    # 产物启发式
    if snap_after["track_count"] > 0 and snap_after["env_count"] == 0:
        return "environment_extraction"
    if snap_after["track_count"] > 0 and snap_after["env_count"] > 0 and error_text:
        return "partial_env_then_fail"
    if snap_after["nc_exists"] and snap_after["track_count"] == 0:
        return "tracking"
    if snap_after["raw_exists"] and (not snap_after["nc_exists"]):
        return "grib_to_nc_conversion"
    if (not snap_after["raw_exists"]) and (not snap_after["nc_exists"]):
        return "download_or_early_failure"

    return "unknown"


# %% 读取状态文件并汇总
state_paths = {
    tag: pick_existing(candidates)
    for tag, candidates in STATE_CANDIDATES.items()
}

states = {
    tag: load_state(path)
    for tag, path in state_paths.items()
}

for tag in ["aws", "tigge"]:
    p = state_paths[tag]
    s = states[tag]
    print(f"[{tag}] state_file={p}")
    print(f"  done={len(s['done'])}, failed={len(s['failed'])}")
    print(f"  failed head={s['failed'][:5]}")


# %% 静态扫描（不重跑）
track_map, env_map = index_track_and_env()
rows = []
for source_tag in ["aws", "tigge"]:
    for key in states[source_tag]["failed"]:
        snap = artifact_snapshot(source_tag, key, track_map=track_map, env_map=env_map)
        rows.append({
            "source": source_tag,
            "key": key,
            **snap,
        })

static_df = pd.DataFrame(rows)
print(f"static scan rows = {len(static_df)}")
if not static_df.empty:
    display(
        static_df.assign(
            raw=static_df["raw_exists"].astype(int),
            nc=static_df["nc_exists"].astype(int),
            has_track=(static_df["track_count"] > 0).astype(int),
            has_env=(static_df["env_count"] > 0).astype(int),
        )
        .groupby(["source", "raw", "nc", "has_track", "has_env"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )


# %% 环境预检（依赖、凭据、关键导入）
dependency_status = []
for mod_name in ["boto3", "botocore", "xarray", "cfgrib", "ecmwfapi", "numpy", "pandas"]:
    try:
        importlib.import_module(mod_name)
        dependency_status.append((mod_name, "OK", ""))
    except Exception as exc:
        dependency_status.append((mod_name, "FAIL", f"{type(exc).__name__}: {exc}"))

precheck_df = pd.DataFrame(dependency_status, columns=["module", "status", "detail"])
display(precheck_df)

ecmwfapi_cfg = Path.home() / ".ecmwfapirc"
cdsapi_cfg = Path.home() / ".cdsapirc"
print(f"~/.ecmwfapirc exists={ecmwfapi_cfg.exists()} path={ecmwfapi_cfg}")
print(f"~/.cdsapirc exists={cdsapi_cfg.exists()} path={cdsapi_cfg}")

try:
    import ec_dual_pipeline as dual
    print(f"ec_dual_pipeline import OK: {dual.__file__}")
except Exception as exc:
    print(f"ec_dual_pipeline import FAIL: {type(exc).__name__}: {exc}")


# %% 重放失败样本并抓取真实错误
import ec_dual_pipeline as dual


def build_payload(source_tag: str):
    source = dual.AWS_SOURCE if source_tag == "aws" else dual.TIGGE_SOURCE
    return {
        "project_root": str(PROJECT_ROOT),
        "source": source,
        "steps": list(STEPS),
        "match_tol_hours": int(MATCH_TOL_HOURS),
        "keep_failed_artifacts": bool(KEEP_FAILED_ARTIFACTS_DURING_REPLAY),
        "initials_csv": str(INITIALS_CSV),
    }


def choose_keys(source_tag: str):
    manual = REPLAY_ONLY_KEYS.get(source_tag, [])
    if manual:
        return manual
    failed = states[source_tag]["failed"]
    limit = int(REPLAY_LIMIT_PER_SOURCE.get(source_tag, 0))
    return failed[:limit] if limit > 0 else []


replay_rows = []
for source_tag in REPLAY_ORDER:
    keys = choose_keys(source_tag)
    if not keys:
        print(f"[{source_tag}] 没有待重放样本（请检查 REPLAY_LIMIT_PER_SOURCE / REPLAY_ONLY_KEYS）")
        continue

    payload = build_payload(source_tag)
    print(f"\n[{source_tag}] 开始重放 {len(keys)} 个失败样本...")

    for idx, key in enumerate(keys, start=1):
        date_str, run_hour = parse_key(key)
        track_map, env_map = index_track_and_env()
        snap_before = artifact_snapshot(source_tag, key, track_map=track_map, env_map=env_map)

        t0 = time.time()
        try:
            res = dual._run_single_job(date_str, run_hour, payload)
        except Exception as exc:
            res = {
                "key": key,
                "ok": False,
                "tracks": 0,
                "env_files": 0,
                "error": f"UnhandledException: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        elapsed = round(time.time() - t0, 2)

        track_map, env_map = index_track_and_env()
        snap_after = artifact_snapshot(source_tag, key, track_map=track_map, env_map=env_map)

        error_text = res.get("error")
        tb_text = res.get("traceback", "")
        row = {
            "source": source_tag,
            "key": key,
            "ok": bool(res.get("ok")),
            "elapsed_s": elapsed,
            "error": error_text,
            "traceback_head": short_tb(tb_text, max_lines=8),
            "tracks_reported": int(res.get("tracks", 0) or 0),
            "env_reported": int(res.get("env_files", 0) or 0),
            "stage_guess": guess_stage(error_text, snap_after),
            "before_raw": bool(snap_before["raw_exists"]),
            "before_nc": bool(snap_before["nc_exists"]),
            "before_track_count": int(snap_before["track_count"]),
            "before_env_count": int(snap_before["env_count"]),
            "after_raw": bool(snap_after["raw_exists"]),
            "after_nc": bool(snap_after["nc_exists"]),
            "after_track_count": int(snap_after["track_count"]),
            "after_env_count": int(snap_after["env_count"]),
            "raw_size_mb": float(snap_after["raw_size_mb"]),
            "nc_size_mb": float(snap_after["nc_size_mb"]),
        }
        replay_rows.append(row)

        err_preview = (error_text or "")[:140]
        print(
            f"[{source_tag}] {idx:02d}/{len(keys)} {key} | ok={row['ok']} | "
            f"stage={row['stage_guess']} | elapsed={elapsed}s | err={err_preview}"
        )

replay_df = pd.DataFrame(replay_rows)
print(f"\nreplay rows = {len(replay_df)}")
if not replay_df.empty:
    display(replay_df[["source", "key", "ok", "stage_guess", "error", "elapsed_s"]])


# %% 重放结果聚合 + 落盘
if replay_df.empty:
    print("没有重放结果可汇总。")
else:
    def normalize_error_text(x):
        if not x:
            return ""
        txt = str(x).strip()
        if ": " in txt:
            # 保留前两段，避免每条细节完全不同导致无法聚类
            parts = txt.split(": ")
            return ": ".join(parts[:2])
        return txt

    replay_df = replay_df.copy()
    replay_df["error_group"] = replay_df["error"].apply(normalize_error_text)

    print("\n按 source + stage 聚合：")
    display(
        replay_df.groupby(["source", "stage_guess"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )

    print("\n按 source + error_group 聚合（前 30）：")
    display(
        replay_df.groupby(["source", "error_group"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = SUMMARY_DIR / f"failure_replay_{ts}.csv"
    out_json = SUMMARY_DIR / f"failure_replay_{ts}.json"

    replay_df.to_csv(out_csv, index=False, encoding="utf-8")
    replay_df.to_json(out_json, orient="records", force_ascii=False, indent=2)
    print(f"\n已保存: {out_csv}")
    print(f"已保存: {out_json}")

    print("\n建议粘贴给我的最小信息：")
    print("1) 环境预检表（module/status/detail）")
    print("2) source+stage 聚合表")
    print("3) source+error_group 聚合表")
    print("4) replay_df 中前 10 行（含 error + traceback_head）")
