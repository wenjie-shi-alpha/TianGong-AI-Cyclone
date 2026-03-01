#!/usr/bin/env python3
"""One-click EC dual-source pipeline.

Execution order is configurable and defaults to:
1) AWS open-data ECMWF forecasts
2) TIGGE ECMWF forecasts

Each source runs the same closed loop:
request/download -> track -> environment extraction -> cleanup GRIB/NC.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


AWS_SOURCE = "aws_open_data"
TIGGE_SOURCE = "tigge_webapi"
S3_BUCKET = "ecmwf-forecasts"
REQUIRED_PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 200]
TIGGE_SURFACE_PARAM_CODES = "151/165/166/34/167"  # msl, 10u, 10v, sst, 2t
TIGGE_PRESSURE_PARAM_CODES = "129/130/131/132"  # z, t, u, v
TIGGE_VERTICAL_VELOCITY_PARAM_CODES = "135"  # w (optional)


@dataclass
class PipelineConfig:
    project_root: Path
    source: str
    start_date: str
    end_date: str
    run_hours: list[int]
    steps: list[int]
    initials_csv: Path
    match_tol_hours: int
    workers: int
    mp_context: str
    cleanup_intermediates_before_run: bool
    keep_failed_artifacts: bool
    ignore_state: bool
    max_jobs: int | None

    @property
    def source_tag(self) -> str:
        return "aws" if self.source == AWS_SOURCE else "tigge"

    @property
    def state_file(self) -> Path:
        return self.project_root / "output" / "cds_notebook_state" / f"pipeline_state_{self.source_tag}.json"

    def to_worker_payload(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "source": self.source,
            "steps": list(self.steps),
            "match_tol_hours": int(self.match_tol_hours),
            "keep_failed_artifacts": bool(self.keep_failed_artifacts),
            "initials_csv": str(self.initials_csv),
        }


def _ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw_dir": root / "data" / "ecmwf_raw",
        "nc_dir": root / "data" / "ecmwf_nc",
        "track_dir": root / "track_output_cds",
        "env_dir": root / "output" / "forecast_environment",
        "summary_dir": root / "output" / "cds_notebook_summary",
        "state_dir": root / "output" / "cds_notebook_state",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_points_with_dt(csv_path: Path) -> pd.DataFrame:
    all_points = pd.read_csv(csv_path)
    if "dt" not in all_points.columns:
        if "datetime" not in all_points.columns:
            raise KeyError("datetime")
        all_points["dt"] = pd.to_datetime(all_points["datetime"], errors="coerce")
    else:
        all_points["dt"] = pd.to_datetime(all_points["dt"], errors="coerce")

    all_points = all_points.dropna(subset=["dt"]).copy()
    for col in ["max_wind_usa", "min_pressure_usa"]:
        if col in all_points.columns:
            all_points[col] = pd.to_numeric(all_points[col], errors="coerce")
    return all_points


def _build_active_jobs_from_initials(
    initials_csv: Path,
    start_ymd: str,
    end_ymd: str,
    run_hours: Iterable[int],
    tol_hours: int,
) -> list[tuple[str, int]]:
    """Keep only init times that have any nearby observed point."""
    import numpy as np

    all_points = _load_points_with_dt(initials_csv)
    dt_ns = np.sort(all_points["dt"].values.astype("datetime64[ns]").astype("int64"))
    if dt_ns.size == 0:
        return []

    s = pd.Timestamp(f"{start_ymd} 00:00:00")
    e = pd.Timestamp(f"{end_ymd} 23:59:59")

    candidate_inits: list[pd.Timestamp] = []
    cur = s.normalize()
    while cur <= e.normalize():
        for h in run_hours:
            init_t = cur + pd.Timedelta(hours=int(h))
            if s <= init_t <= e:
                candidate_inits.append(init_t)
        cur += pd.Timedelta(days=1)

    tol_ns = int(pd.Timedelta(hours=tol_hours).value)
    jobs: list[tuple[str, int]] = []
    for init_t in candidate_inits:
        x = int(init_t.value)
        idx = np.searchsorted(dt_ns, x)
        ok = False
        if idx < dt_ns.size and abs(int(dt_ns[idx]) - x) <= tol_ns:
            ok = True
        elif idx > 0 and abs(int(dt_ns[idx - 1]) - x) <= tol_ns:
            ok = True
        if ok:
            jobs.append((init_t.strftime("%Y-%m-%d"), int(init_t.hour)))
    return jobs


def _cleanup_file(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def _cleanup_intermediates(raw_dir: Path, nc_dir: Path, source: str) -> None:
    if source == AWS_SOURCE:
        gribs = list(raw_dir.glob("ecmwf_*.grib2"))
        ncs = list(nc_dir.glob("ecmwf_*.nc"))
        part_dirs = [p for p in raw_dir.glob("ecmwf_*_parts") if p.is_dir()]
    else:
        gribs = list(raw_dir.glob("tigge_ecmwf_*.grib")) + list(raw_dir.glob("tigge_ecmwf_*.grib2"))
        ncs = list(nc_dir.glob("tigge_ecmwf_*.nc"))
        part_dirs = []

    for p in gribs:
        _cleanup_file(p)
    for p in ncs:
        _cleanup_file(p)
    for d in part_dirs:
        shutil.rmtree(d, ignore_errors=True)

    print(
        f"[{source}] startup cleanup: grib={len(gribs)}, nc={len(ncs)}, parts={len(part_dirs)}"
    )


def _to_step_expr(steps: list[int]) -> str:
    if len(steps) == 1:
        return str(steps[0])
    delta = steps[1] - steps[0]
    return f"{steps[0]}/to/{steps[-1]}/by/{delta}"


def _subset_pressure_levels(ds: "Any", wanted_levels: list[int]) -> "Any":
    for level_dim in ["isobaricInhPa", "level", "pressure"]:
        if level_dim not in ds.coords:
            continue
        try:
            raw_vals = np.asarray(ds[level_dim].values, dtype=float)
        except Exception:
            continue

        if raw_vals.size == 0:
            continue

        # normalize unit to hPa for matching; if values look like Pa then divide by 100
        vals_hpa = raw_vals / 100.0 if np.nanmax(raw_vals) > 2000 else raw_vals
        keep_native: list[float] = []
        for target in wanted_levels:
            idx = int(np.nanargmin(np.abs(vals_hpa - float(target))))
            if abs(float(vals_hpa[idx]) - float(target)) <= 5.0:
                keep_native.append(float(raw_vals[idx]))

        if not keep_native:
            continue

        unique_keep = sorted(set(keep_native))
        try:
            return ds.sel({level_dim: unique_keep})
        except Exception:
            continue
    return ds


def _normalize_forecast_time(ds: "Any") -> "Any":
    if "step" in ds.dims:
        if "valid_time" in ds.coords:
            valid_times = pd.DatetimeIndex(ds["valid_time"].values)
        else:
            base_raw = ds["time"].values
            base = pd.Timestamp(base_raw.ravel()[0] if hasattr(base_raw, "ravel") else base_raw)
            valid_times = pd.DatetimeIndex([base + pd.Timedelta(s) for s in ds["step"].values])

        ds = ds.drop_vars("time", errors="ignore")
        if "valid_time" not in ds.coords:
            ds = ds.assign_coords(valid_time=("step", valid_times))
        ds = ds.swap_dims({"step": "valid_time"})
        ds = ds.drop_vars("step", errors="ignore")
        ds = ds.rename({"valid_time": "time"})
    elif "time" not in ds.dims:
        ds = ds.expand_dims("time")
    return ds


def _convert_grib_to_nc(grib_path: Path, nc_path: Path) -> None:
    import xarray as xr

    datasets = []
    filters: list[tuple[dict[str, str], list[int] | None]] = [
        ({"shortName": "msl"}, None),
        ({"shortName": "10u"}, None),
        ({"shortName": "10v"}, None),
        ({"shortName": "sst"}, None),
        ({"shortName": "2t"}, None),
        ({"shortName": "u", "typeOfLevel": "isobaricInhPa"}, REQUIRED_PRESSURE_LEVELS),
        ({"shortName": "v", "typeOfLevel": "isobaricInhPa"}, REQUIRED_PRESSURE_LEVELS),
        ({"shortName": "z", "typeOfLevel": "isobaricInhPa"}, REQUIRED_PRESSURE_LEVELS),
        ({"shortName": "t", "typeOfLevel": "isobaricInhPa"}, REQUIRED_PRESSURE_LEVELS),
        ({"shortName": "w", "typeOfLevel": "isobaricInhPa"}, [700]),
    ]

    for filter_keys, wanted_levels in filters:
        try:
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
            )
            if wanted_levels is not None:
                ds = _subset_pressure_levels(ds, wanted_levels)
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise RuntimeError(f"cannot parse required vars from {grib_path.name}")

    ds_merged = datasets[0]
    for ds in datasets[1:]:
        ds_merged = ds_merged.merge(ds, compat="override", join="outer")

    rename_map = {}
    if "u10" in ds_merged.data_vars:
        rename_map["u10"] = "10u"
    if "v10" in ds_merged.data_vars:
        rename_map["v10"] = "10v"
    if rename_map:
        ds_merged = ds_merged.rename(rename_map)

    ds_merged = _normalize_forecast_time(ds_merged)
    if "time" in ds_merged.coords:
        ds_merged = ds_merged.sortby("time")

    # ECMWF z is usually geopotential (m^2 s^-2); convert to gpm for consistency with thresholds.
    if "z" in ds_merged.data_vars:
        z_da = ds_merged["z"]
        units = str(z_da.attrs.get("units", "")).lower()
        try:
            sample = float(np.nanmedian(np.abs(z_da.values)))
        except Exception:
            sample = np.nan
        looks_like_geopotential = ("m**2" in units and "s**-2" in units) or ("m^2" in units and "s^-2" in units)
        if looks_like_geopotential or (np.isfinite(sample) and sample > 15000):
            ds_merged["z"] = z_da / 9.80665
            ds_merged["z"].attrs.update(
                {
                    "units": "gpm",
                    "original_units": units or "unknown",
                    "note": "converted from geopotential by dividing 9.80665",
                }
            )

    ds_merged.to_netcdf(nc_path)
    vars_short = ",".join(sorted(str(v) for v in ds_merged.data_vars))
    logging.info("converted %s -> %s | vars=%s", grib_path.name, nc_path.name, vars_short)
    for ds in datasets:
        ds.close()
    ds_merged.close()


def _make_unsigned_s3_client() -> Any:
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )


def _discover_oper_prefix_and_steps(date_str: str, run_hour: int) -> tuple[str, list[int]]:
    ymd = date_str.replace("-", "")
    hh = f"{run_hour:02d}"
    prefixes = [
        f"{ymd}/{hh}z/ifs/0p4-beta/oper",
        f"{ymd}/{hh}z/0p4-beta/oper",
    ]

    client = _make_unsigned_s3_client()
    for prefix in prefixes:
        steps: set[int] = set()
        paginator = client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{prefix}/"):
                for obj in page.get("Contents", []):
                    name = obj["Key"].split("/")[-1]
                    match = re.search(r"-(\d+)h-oper-fc\.grib2$", name)
                    if match:
                        steps.add(int(match.group(1)))
        except Exception:
            continue
        if steps:
            return prefix, sorted(steps)

    raise RuntimeError(f"no available S3 prefix for {date_str}_{run_hour:02d}")


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=3, max=20), reraise=True)
def _download_one_job_aws(date_str: str, run_hour: int, grib_path: Path, steps: list[int]) -> None:
    ymd = date_str.replace("-", "")
    ymdhms = f"{ymd}{run_hour:02d}0000"
    prefix, available_steps = _discover_oper_prefix_and_steps(date_str, run_hour)
    avail = set(available_steps)
    req_steps = [s for s in steps if s in avail]
    if not req_steps:
        raise RuntimeError(f"no required step available for {date_str}_{run_hour:02d}")

    missing = [s for s in steps if s not in avail]
    if missing:
        logging.warning("%s_%02dZ missing %d steps, skipping", date_str, run_hour, len(missing))

    tmp_dir = grib_path.parent / f"{grib_path.stem}_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    client = _make_unsigned_s3_client()
    part_files: list[Path] = []
    try:
        for step in req_steps:
            name = f"{ymdhms}-{step}h-oper-fc.grib2"
            key = f"{prefix}/{name}"
            part = tmp_dir / f"{step:03d}h.grib2"
            client.download_file(S3_BUCKET, key, str(part))
            part_files.append(part)

        with grib_path.open("wb") as fw:
            for part in part_files:
                with part.open("rb") as fr:
                    shutil.copyfileobj(fr, fw)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=3, max=20), reraise=True)
def _download_one_job_tigge(date_str: str, run_hour: int, nc_path: Path, grib_path: Path, steps: list[int]) -> None:
    from ecmwfapi import ECMWFDataServer

    if not (Path.home() / ".ecmwfapirc").exists():
        raise RuntimeError("missing ~/.ecmwfapirc for TIGGE")

    request_common = {
        "class": "ti",
        "dataset": "tigge",
        "expver": "prod",
        "origin": "ecmf",
        "type": "cf",
        "grid": "0.5/0.5",
        "date": date_str,
        "time": f"{run_hour:02d}",
        "step": _to_step_expr(steps),
    }

    server = ECMWFDataServer()
    tmp_sfc = grib_path.with_suffix(".tmp.sfc.grib")
    tmp_pl = grib_path.with_suffix(".tmp.pl.grib")
    tmp_w = grib_path.with_suffix(".tmp.w.grib")
    tmp_combined = grib_path.with_suffix(".tmp.grib")

    def _retrieve(request: dict[str, str], target: Path) -> None:
        req = {**request, "target": str(target)}
        server.retrieve(req)
        if not target.exists() or target.stat().st_size == 0:
            raise RuntimeError(f"empty TIGGE response: {target.name}")

    pl_level_options = [
        REQUIRED_PRESSURE_LEVELS,
        [850, 700, 500, 300, 200],
        [850, 700, 500, 200],
    ]

    try:
        _retrieve(
            {
                **request_common,
                "levtype": "sfc",
                "param": TIGGE_SURFACE_PARAM_CODES,
            },
            tmp_sfc,
        )

        last_pl_error: Exception | None = None
        pl_ok = False
        for levels in pl_level_options:
            _cleanup_file(tmp_pl)
            try:
                _retrieve(
                    {
                        **request_common,
                        "levtype": "pl",
                        "levelist": "/".join(str(v) for v in levels),
                        "param": TIGGE_PRESSURE_PARAM_CODES,
                    },
                    tmp_pl,
                )
                pl_ok = True
                break
            except Exception as exc:
                last_pl_error = exc
                continue

        if not pl_ok:
            raise RuntimeError(f"TIGGE pressure-level retrieve failed: {last_pl_error}")

        _cleanup_file(tmp_w)
        try:
            _retrieve(
                {
                    **request_common,
                    "levtype": "pl",
                    "levelist": "700",
                    "param": TIGGE_VERTICAL_VELOCITY_PARAM_CODES,
                },
                tmp_w,
            )
        except Exception as exc:
            logging.warning("optional TIGGE w@700 retrieve failed for %s_%02d: %s", date_str, run_hour, exc)
            _cleanup_file(tmp_w)

        with tmp_combined.open("wb") as fw:
            for src in [tmp_sfc, tmp_pl, tmp_w]:
                if not src.exists() or src.stat().st_size == 0:
                    continue
                with src.open("rb") as fr:
                    shutil.copyfileobj(fr, fw)

        if not tmp_combined.exists() or tmp_combined.stat().st_size == 0:
            raise RuntimeError("empty merged grib from TIGGE")

        tmp_combined.replace(grib_path)
        _convert_grib_to_nc(grib_path, nc_path)
    finally:
        _cleanup_file(tmp_sfc)
        _cleanup_file(tmp_pl)
        _cleanup_file(tmp_w)
        _cleanup_file(tmp_combined)


def _run_single_job(date_str: str, run_hour: int, payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["project_root"])
    source = str(payload["source"])
    steps = [int(v) for v in payload["steps"]]
    match_tol_hours = int(payload["match_tol_hours"])
    keep_failed = bool(payload["keep_failed_artifacts"])
    initials_csv = Path(payload["initials_csv"])

    os.chdir(root)
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    dirs = _ensure_dirs(root)
    raw_dir = dirs["raw_dir"]
    nc_dir = dirs["nc_dir"]
    track_dir = dirs["track_dir"]
    env_dir = dirs["env_dir"]

    key = f"{date_str}_{run_hour:02d}"
    if source == AWS_SOURCE:
        grib_path = raw_dir / f"ecmwf_{date_str.replace('-', '')}_{run_hour:02d}.grib2"
        nc_path = nc_dir / f"ecmwf_{date_str.replace('-', '')}_{run_hour:02d}.nc"
    else:
        grib_path = raw_dir / f"tigge_ecmwf_{date_str.replace('-', '')}_{run_hour:02d}.grib"
        nc_path = nc_dir / f"tigge_ecmwf_{date_str.replace('-', '')}_{run_hour:02d}.nc"

    try:
        if source == AWS_SOURCE:
            if not grib_path.exists() or grib_path.stat().st_size == 0:
                _download_one_job_aws(date_str, run_hour, grib_path, steps)
            if not nc_path.exists() or nc_path.stat().st_size == 0:
                _convert_grib_to_nc(grib_path, nc_path)
        else:
            if not nc_path.exists() or nc_path.stat().st_size == 0:
                _download_one_job_tigge(date_str, run_hour, nc_path, grib_path, steps)

        from initial_tracker.workflow import track_file_with_initials
        from environment_extractor.extractor import TCEnvironmentalSystemsExtractor

        all_points = _load_points_with_dt(initials_csv)
        written = track_file_with_initials(
            nc_path=nc_path,
            all_points=all_points,
            output_dir=track_dir,
            max_storms=None,
            time_window_hours=match_tol_hours,
        )

        track_files = [Path(p) for p in (written or []) if p]
        env_created = 0
        for track_csv in track_files:
            if not track_csv.exists():
                continue

            # Force refresh per storm output to avoid reusing stale cached JSON from older logic.
            try:
                track_df = pd.read_csv(track_csv, usecols=["particle"])
                particle_ids = sorted(set(str(v) for v in track_df["particle"].dropna().tolist()))
            except Exception:
                particle_ids = []
            for particle_id in particle_ids:
                _cleanup_file(env_dir / f"{nc_path.stem}_TC_Analysis_{particle_id}.json")

            pre = {p.name for p in env_dir.glob("*.json")}
            with TCEnvironmentalSystemsExtractor(str(nc_path), str(track_csv)) as extractor:
                extractor.analyze_and_export_as_json(output_dir=str(env_dir))
            post = {p.name for p in env_dir.glob("*.json")}
            env_created += max(0, len(post - pre))

        _cleanup_file(nc_path)
        _cleanup_file(grib_path)
        return {"key": key, "ok": True, "tracks": len(track_files), "env_files": env_created, "error": None}
    except Exception as exc:
        if not keep_failed:
            _cleanup_file(nc_path)
            _cleanup_file(grib_path)
        return {
            "key": key,
            "ok": False,
            "tracks": 0,
            "env_files": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def _load_state(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return set(payload.get("done", [])), set(payload.get("failed", []))
    except Exception:
        return set(), set()


def _save_state(path: Path, done_set: set[str], failed_set: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"done": sorted(done_set), "failed": sorted(failed_set)},
            fh,
            ensure_ascii=False,
            indent=2,
        )


def run_source(cfg: PipelineConfig) -> dict[str, Any]:
    _ensure_dirs(cfg.project_root)
    if cfg.ignore_state:
        done_set, failed_set = set(), set()
        print(f"[{cfg.source}] ignoring existing state file: {cfg.state_file}")
    else:
        done_set, failed_set = _load_state(cfg.state_file)

    if cfg.cleanup_intermediates_before_run:
        dirs = _ensure_dirs(cfg.project_root)
        _cleanup_intermediates(dirs["raw_dir"], dirs["nc_dir"], cfg.source)

    all_jobs = _build_active_jobs_from_initials(
        initials_csv=cfg.initials_csv,
        start_ymd=cfg.start_date,
        end_ymd=cfg.end_date,
        run_hours=cfg.run_hours,
        tol_hours=cfg.match_tol_hours,
    )
    pending = [j for j in all_jobs if f"{j[0]}_{j[1]:02d}" not in done_set]
    if cfg.max_jobs is not None:
        pending = pending[: cfg.max_jobs]

    print(
        f"\n[{cfg.source}] range={cfg.start_date}->{cfg.end_date}, "
        f"jobs={len(all_jobs)}, pending={len(pending)}, done={len(done_set)}"
    )

    if not pending:
        return {
            "source": cfg.source,
            "jobs_total": len(all_jobs),
            "jobs_ran": 0,
            "done": len(done_set),
            "failed": len(failed_set),
        }

    payload = cfg.to_worker_payload()
    jobs_ran = 0
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=get_context(cfg.mp_context)) as ex:
        futs = {ex.submit(_run_single_job, d, h, payload): (d, h) for d, h in pending}
        for fut in as_completed(futs):
            d, h = futs[fut]
            key = f"{d}_{h:02d}"
            jobs_ran += 1
            try:
                res = fut.result(timeout=14400)
            except Exception as exc:
                res = {"ok": False, "error": f"WorkerError: {exc}", "tracks": 0, "env_files": 0}

            if res.get("ok"):
                done_set.add(key)
                failed_set.discard(key)
                print(f"[{cfg.source}] OK  {key}: tracks={res.get('tracks', 0)}, env={res.get('env_files', 0)}")
            else:
                failed_set.add(key)
                print(f"[{cfg.source}] ERR {key}: {res.get('error')}")
            _save_state(cfg.state_file, done_set, failed_set)

    return {
        "source": cfg.source,
        "jobs_total": len(all_jobs),
        "jobs_ran": jobs_ran,
        "done": len(done_set),
        "failed": len(failed_set),
        "state_file": str(cfg.state_file),
    }


def _parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("empty int list")
    return values


def _parse_sources(raw: str) -> list[str]:
    out = []
    for token in [x.strip().lower() for x in raw.split(",") if x.strip()]:
        if token in {"aws", "aws_open_data"}:
            out.append(AWS_SOURCE)
        elif token in {"tigge", "tigge_webapi"}:
            out.append(TIGGE_SOURCE)
        else:
            raise ValueError(f"unsupported source token: {token}")
    if not out:
        raise ValueError("no source provided")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="EC dual-source pipeline (AWS then TIGGE)")
    parser.add_argument("--sources", default="aws,tigge", help="comma list: aws,tigge")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--run-hours", default="0,12")
    parser.add_argument("--steps", default=",".join(str(s) for s in range(0, 241, 6)))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--match-tol-hours", type=int, default=6)
    parser.add_argument("--initials-csv", default=str(PROJECT_ROOT / "input" / "matched_cyclone_tracks_2021onwards.csv"))
    parser.add_argument("--mp-context", default="fork")
    parser.add_argument("--cleanup-intermediates-before-run", dest="cleanup", action="store_true")
    parser.add_argument("--no-cleanup-intermediates-before-run", dest="cleanup", action="store_false")
    parser.set_defaults(cleanup=True)
    parser.add_argument("--keep-failed-artifacts", action="store_true")
    parser.add_argument("--ignore-state", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    sources = _parse_sources(args.sources)
    run_hours = _parse_int_list(args.run_hours)
    steps = _parse_int_list(args.steps)
    initials_csv = Path(args.initials_csv)
    if not initials_csv.exists():
        raise FileNotFoundError(f"initials csv not found: {initials_csv}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    summaries = []
    for source in sources:
        cfg = PipelineConfig(
            project_root=PROJECT_ROOT,
            source=source,
            start_date=args.start_date,
            end_date=args.end_date,
            run_hours=run_hours,
            steps=steps,
            initials_csv=initials_csv,
            match_tol_hours=int(args.match_tol_hours),
            workers=max(1, int(args.workers)),
            mp_context=args.mp_context,
            cleanup_intermediates_before_run=bool(args.cleanup),
            keep_failed_artifacts=bool(args.keep_failed_artifacts),
            ignore_state=bool(args.ignore_state),
            max_jobs=args.max_jobs,
        )
        summaries.append(run_source(cfg))

    print("\n==== dual-source summary ====")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
