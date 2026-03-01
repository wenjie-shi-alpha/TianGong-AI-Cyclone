#!/usr/bin/env python3
"""Minimal EC smoke test: request -> track -> environment extraction.

This script is intended for quick validation on a real runtime
(e.g. CDS JupyterLab / AWS VM). It downloads a tiny sample, runs tracking,
then verifies that outputs are organized by real storm IDs (particle/time_idx).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class SmokePaths:
    root: Path
    raw_dir: Path
    nc_dir: Path
    track_dir: Path
    env_dir: Path
    summary_file: Path


def _parse_steps(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("steps is empty")
    return sorted(set(values))


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            f"Missing dependency `{name}`. Install requirements and retry."
        ) from exc


def _ensure_home_credentials() -> None:
    for fname in (".cdsapirc", ".ecmwfapirc"):
        src = PROJECT_ROOT / fname
        dst = Path.home() / fname
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(src, dst)
            except Exception as exc:  # pragma: no cover - runtime env dependent
                print(f"warning: failed to copy {src} -> {dst}: {exc}")


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


def _convert_grib_to_nc(grib_path: Path, nc_path: Path) -> None:
    xr = _safe_import("xarray")
    pd_mod = _safe_import("pandas")

    datasets = []
    for short_name in ["msl", "10u", "10v", "sst"]:
        try:
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {"shortName": short_name},
                    "indexpath": "",
                },
            )
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise RuntimeError(f"Cannot parse vars from GRIB: {grib_path.name}")

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

    if "step" in ds_merged.dims:
        if "valid_time" in ds_merged.coords:
            valid_times = pd_mod.DatetimeIndex(ds_merged.valid_time.values)
        else:
            base = pd_mod.Timestamp(ds_merged.time.values)
            valid_times = pd_mod.DatetimeIndex(
                [base + pd_mod.Timedelta(s) for s in ds_merged.step.values]
            )
        ds_merged = ds_merged.drop_vars("time", errors="ignore")
        if "valid_time" not in ds_merged.coords:
            ds_merged = ds_merged.assign_coords(valid_time=("step", valid_times))
        ds_merged = ds_merged.swap_dims({"step": "valid_time"})
        ds_merged = ds_merged.drop_vars("step", errors="ignore")
        ds_merged = ds_merged.rename({"valid_time": "time"})
    elif "time" not in ds_merged.dims:
        ds_merged = ds_merged.expand_dims("time")

    ds_merged.to_netcdf(nc_path)
    for ds in datasets:
        ds.close()
    ds_merged.close()


def _download_tigge_sample(date_str: str, run_hour: int, steps: Iterable[int], nc_path: Path, grib_path: Path) -> None:
    _ensure_home_credentials()
    if not (Path.home() / ".ecmwfapirc").exists():
        raise RuntimeError("Missing ~/.ecmwfapirc for TIGGE WebAPI")

    ecmwfapi = _safe_import("ecmwfapi")
    step_list = sorted(set(int(s) for s in steps))
    if len(step_list) < 2:
        step_expr = str(step_list[0])
    else:
        step_expr = f"{step_list[0]}/to/{step_list[-1]}/by/{step_list[1]-step_list[0]}"

    request_common = {
        "class": "ti",
        "dataset": "tigge",
        "expver": "prod",
        "origin": "ecmf",
        "type": "cf",
        "levtype": "sfc",
        "grid": "0.5/0.5",
        "param": "151/165/166/34",  # msl / 10u / 10v / sst
        "date": date_str,
        "time": f"{run_hour:02d}",
        "step": step_expr,
    }

    server = ecmwfapi.ECMWFDataServer()
    tmp_nc = nc_path.with_suffix(".tmp.nc")
    tmp_grib = grib_path.with_suffix(".tmp.grib")

    try:
        req_nc = {**request_common, "format": "netcdf", "target": str(tmp_nc)}
        server.retrieve(req_nc)
        if not tmp_nc.exists() or tmp_nc.stat().st_size == 0:
            raise RuntimeError("TIGGE returned empty netcdf")
        tmp_nc.replace(nc_path)
        return
    except Exception:
        if tmp_nc.exists():
            tmp_nc.unlink()

    req_grib = {**request_common, "target": str(tmp_grib)}
    server.retrieve(req_grib)
    if not tmp_grib.exists() or tmp_grib.stat().st_size == 0:
        raise RuntimeError("TIGGE returned empty grib")
    tmp_grib.replace(grib_path)
    _convert_grib_to_nc(grib_path, nc_path)


def _download_aws_sample(date_str: str, run_hour: int, steps: Iterable[int], grib_path: Path) -> None:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    ymd = date_str.replace("-", "")
    hh = f"{run_hour:02d}"
    ymdhms = f"{ymd}{hh}0000"
    prefixes = [
        f"{ymd}/{hh}z/ifs/0p4-beta/oper",
        f"{ymd}/{hh}z/0p4-beta/oper",
    ]

    client = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )
    bucket = "ecmwf-forecasts"

    tmp_dir = grib_path.parent / f"{grib_path.stem}_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    probe_errors: list[str] = []
    try:
        for step in sorted(set(int(s) for s in steps)):
            name = f"{ymdhms}-{step}h-oper-fc.grib2"
            hit_key = None
            for prefix in prefixes:
                key = f"{prefix}/{name}"
                try:
                    client.head_object(Bucket=bucket, Key=key)
                    hit_key = key
                    break
                except Exception as exc:
                    if len(probe_errors) < 6:
                        probe_errors.append(f"{key}: {type(exc).__name__}: {exc}")
                    continue
            if hit_key is None:
                continue
            local_part = tmp_dir / f"{step:03d}h.grib2"
            client.download_file(bucket, hit_key, str(local_part))
            downloaded.append(local_part)

        if not downloaded:
            details = "; ".join(probe_errors[:3]) if probe_errors else "no probe error captured"
            raise RuntimeError(f"No AWS open-data GRIB step downloaded ({details})")

        with grib_path.open("wb") as fw:
            for part in downloaded:
                with part.open("rb") as fr:
                    shutil.copyfileobj(fr, fw)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_tracking_and_extraction(
    nc_path: Path,
    initials_csv: Path,
    paths: SmokePaths,
    max_storms: int | None,
    time_window_hours: int,
) -> dict:
    from initial_tracker.workflow import track_file_with_initials
    from environment_extractor.extractor import TCEnvironmentalSystemsExtractor

    all_points = _load_points_with_dt(initials_csv)
    written = track_file_with_initials(
        nc_path=nc_path,
        all_points=all_points,
        output_dir=paths.track_dir,
        max_storms=max_storms,
        time_window_hours=time_window_hours,
    )
    track_files = [Path(p) for p in (written or []) if Path(p).exists()]

    track_checks = []
    env_created = 0
    for track_csv in track_files:
        track_df = pd.read_csv(track_csv)
        track_checks.append(
            {
                "file": track_csv.name,
                "rows": len(track_df),
                "has_particle": "particle" in track_df.columns,
                "has_time_idx": "time_idx" in track_df.columns,
                "particles": sorted(track_df["particle"].dropna().astype(str).unique().tolist())
                if "particle" in track_df.columns
                else [],
            }
        )
        pre = {p.name for p in paths.env_dir.glob("*.json")}
        with TCEnvironmentalSystemsExtractor(str(nc_path), str(track_csv)) as extractor:
            extractor.analyze_and_export_as_json(output_dir=str(paths.env_dir))
        post = {p.name for p in paths.env_dir.glob("*.json")}
        env_created += max(0, len(post - pre))

    return {
        "track_files": [p.name for p in track_files],
        "track_checks": track_checks,
        "env_files_created": env_created,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small EC request/tracking/extraction smoke test"
    )
    parser.add_argument("--source", choices=["tigge", "aws"], default="tigge")
    parser.add_argument("--date", default="2024-07-28")
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--steps", default="0,6,12,18,24")
    parser.add_argument(
        "--initials",
        default=str(PROJECT_ROOT / "input" / "matched_cyclone_tracks_2021onwards.csv"),
    )
    parser.add_argument("--max-storms", type=int, default=3)
    parser.add_argument("--time-window-hours", type=int, default=6)
    parser.add_argument(
        "--workdir",
        default=str(PROJECT_ROOT / "output" / "ec_smoke_debug"),
    )
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()

    steps = _parse_steps(args.steps)
    root = Path(args.workdir)
    paths = SmokePaths(
        root=root,
        raw_dir=root / "raw",
        nc_dir=root / "nc",
        track_dir=root / "track",
        env_dir=root / "env",
        summary_file=root / "summary.json",
    )
    for p in [paths.root, paths.raw_dir, paths.nc_dir, paths.track_dir, paths.env_dir]:
        p.mkdir(parents=True, exist_ok=True)

    stamp = f"{args.date.replace('-', '')}_{args.hour:02d}"
    raw_suffix = "grib2" if args.source == "aws" else "grib"
    grib_path = paths.raw_dir / f"{args.source}_ec_{stamp}.{raw_suffix}"
    nc_path = paths.nc_dir / f"{args.source}_ec_{stamp}.nc"

    summary = {
        "source": args.source,
        "date": args.date,
        "hour": args.hour,
        "steps": steps,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "nc_path": str(nc_path),
        "grib_path": str(grib_path),
    }

    try:
        if args.source == "tigge":
            _download_tigge_sample(args.date, args.hour, steps, nc_path, grib_path)
        else:
            _download_aws_sample(args.date, args.hour, steps, grib_path)
            _convert_grib_to_nc(grib_path, nc_path)

        summary["download_ok"] = nc_path.exists() and nc_path.stat().st_size > 0
        run_summary = _run_tracking_and_extraction(
            nc_path=nc_path,
            initials_csv=Path(args.initials),
            paths=paths,
            max_storms=args.max_storms,
            time_window_hours=args.time_window_hours,
        )
        summary.update(run_summary)
        summary["ok"] = True
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if not args.keep_intermediate:
            if nc_path.exists():
                nc_path.unlink()
            if grib_path.exists():
                grib_path.unlink()

        with paths.summary_file.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
