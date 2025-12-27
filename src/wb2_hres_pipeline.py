#!/usr/bin/env python3
"""WeatherBench2 HRES downloader + cyclone tracking + environment extraction.

This script ports the core logic from `colab.ipynb` to a Slurm/HPC-friendly CLI:
- Read WeatherBench2 HRES forecast Zarr (gs://weatherbench2/..., public, anon)
- Apply variable renaming / unit conversions used in the Colab notebook
- For each storm and each forecast init (00Z/12Z), run the project's tracker
- Persist a minimal NetCDF subset locally (this is the "download" step)
- Run `TCEnvironmentalSystemsExtractor` on the subset and track CSV
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor
from initialTracker import NoEyeException, Tracker, _DsAdapter, _build_batch_from_ds_fast
from initialTracker import _load_all_points, _select_initials_for_time


DEFAULT_DATASET_URL = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def _lat_slice(coord: np.ndarray, lower: float, upper: float) -> slice:
    if coord[0] > coord[-1]:
        return slice(upper, lower)
    return slice(lower, upper)


def _nearest_time_idx(times: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    if len(times) == 0:
        return 0
    try:
        deltas = (times - target).to_numpy(dtype="timedelta64[ns]")
        return int(np.argmin(np.abs(deltas)))
    except Exception:
        return 0


def _ensure_output_layout(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "output_root": output_root,
        "nc_dir": output_root / "data" / "nc_files",
        "track_dir": output_root / "track_single",
        "final_dir": output_root / "final_single_output",
        "logs_dir": output_root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _infer_netcdf_engine() -> str:
    try:
        import netCDF4  # noqa: F401

        return "netcdf4"
    except Exception:
        return "scipy"


def _open_hres_zarr(
    dataset_url: str,
    *,
    storage_options: dict,
    chunks: dict,
    consolidated: bool,
    cache_dir: Path | None,
) -> xr.Dataset:
    if dataset_url.startswith("gs://") and cache_dir is not None:
        try:
            import fsspec
        except Exception as exc:
            raise RuntimeError(
                "启用 --cache-dir 需要安装 fsspec；请在环境中安装 requirements.txt 里的新增依赖。"
            ) from exc

        gcs_path = dataset_url[5:]
        fs = fsspec.filesystem(
            "simplecache",
            target_protocol="gs",
            target_options=storage_options,
            cache_storage=str(cache_dir),
            same_names=True,
        )
        mapper = fs.get_mapper(gcs_path)
        return xr.open_zarr(
            mapper,
            consolidated=consolidated,
            chunks=chunks,
            decode_timedelta=True,
        )

    return xr.open_zarr(
        dataset_url,
        consolidated=consolidated,
        storage_options=storage_options if dataset_url.startswith("gs://") else None,
        chunks=chunks,
        decode_timedelta=True,
    )


def _adapt_variables(ds_raw: xr.Dataset) -> xr.Dataset:
    rename_map = {
        "mean_sea_level_pressure": "msl",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "temperature": "t",
        "specific_humidity": "q",
        "geopotential": "z",
        "land_sea_mask": "lsm",
        "2m_temperature": "t2m",
    }
    present = {src: dst for src, dst in rename_map.items() if src in ds_raw}
    if not present:
        raise ValueError("Zarr 数据集中未找到任何预期变量（检查 dataset_url 是否为 HRES 预报集）")
    ds = ds_raw[list(present.keys())].rename(present)

    # geopotential (m^2/s^2) -> geopotential height (m)
    if "z" in ds:
        ds = ds.assign(z=ds["z"] / 9.80665)

    if "lsm" not in ds:
        lat_dim = "latitude" if "latitude" in ds.coords else "lat"
        lon_dim = "longitude" if "longitude" in ds.coords else "lon"
        ds["lsm"] = xr.DataArray(
            np.zeros((len(ds[lat_dim]), len(ds[lon_dim])), dtype=np.float32),
            coords={lat_dim: ds[lat_dim], lon_dim: ds[lon_dim]},
            dims=[lat_dim, lon_dim],
            name="lsm",
        )
    return ds


def _prepare_single_forecast_dataset(ds_single: xr.Dataset) -> xr.Dataset:
    if "time" in ds_single.coords:
        ds = ds_single.rename({"time": "init_time"})
    else:
        ds = ds_single

    if "prediction_timedelta" not in ds.coords:
        if "init_time" in ds.coords:
            return ds.rename({"init_time": "time"})
        return ds

    pt = ds["prediction_timedelta"]
    hours = pt.values.astype("timedelta64[ns]").astype(float) / (3600 * 1e9)
    keep_mask = np.abs(hours % 6) < 0.01
    ds = ds.isel(prediction_timedelta=keep_mask)

    init_ts = pd.Timestamp(ds["init_time"].values)
    lead_offsets = ds["prediction_timedelta"].values
    if not np.issubdtype(lead_offsets.dtype, np.timedelta64):
        lead_offsets = pd.to_timedelta(lead_offsets)
    valid_times = init_ts + lead_offsets

    ds = ds.assign_coords(valid_time=("prediction_timedelta", valid_times))
    ds = ds.swap_dims({"prediction_timedelta": "valid_time"})
    ds = ds.rename({"valid_time": "time"})
    return ds


def _persist_subset_to_netcdf(ds_subset: xr.Dataset, nc_path: Path, engine: str) -> None:
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds_loaded = ds_subset.load()
    encoding: dict[str, dict[str, object]] = {}
    for name, da in ds_loaded.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            encoding[name] = {"dtype": "float32", "zlib": False}
    ds_loaded.to_netcdf(nc_path, engine=engine, encoding=encoding)


def _expected_json_paths(final_dir: Path, nc_stem: str, particles: Iterable[str]) -> list[Path]:
    return [final_dir / f"{nc_stem}_TC_Analysis_{p}.json" for p in particles]


def _has_complete_json(final_dir: Path, nc_stem: str, particles: Iterable[str]) -> bool:
    for path in _expected_json_paths(final_dir, nc_stem, particles):
        try:
            if not path.exists() or path.stat().st_size <= 10:
                return False
        except OSError:
            return False
    return True


@dataclass(frozen=True)
class _StormResult:
    storm_id: str
    ok: bool
    message: str


def _process_single_storm(
    *,
    storm_id: str,
    dataset_url: str,
    tracks_csv: Path,
    initials_csv: Path,
    output_root: Path,
    cache_dir: Path | None,
    consolidated: bool,
    max_forecasts_per_storm: int,
    max_steps: int,
    pad_days: int,
    pad_deg: float,
    track_pad_deg: float,
    time_pad_steps: int,
    only_00_12z: bool,
    enable_detailed_shape_analysis: bool,
    keep_nc: bool,
    concise_log: bool,
) -> _StormResult:
    t0 = time.time()
    layout = _ensure_output_layout(output_root)
    nc_dir = layout["nc_dir"]
    track_dir = layout["track_dir"]
    final_dir = layout["final_dir"]

    def log(msg: str) -> None:
        if not concise_log:
            print(msg, flush=True)

    try:
        df_tracks = pd.read_csv(tracks_csv)
        if "dt" not in df_tracks.columns:
            df_tracks["dt"] = pd.to_datetime(df_tracks["datetime"], errors="coerce")
        storm_rows = df_tracks.loc[df_tracks["storm_id"] == storm_id].dropna(subset=["dt"])
        if storm_rows.empty:
            return _StormResult(storm_id, False, "storm_id not found in tracks")

        storm_start_dt = pd.Timestamp(storm_rows["dt"].min())
        start_dt = storm_start_dt - pd.Timedelta(days=pad_days)
        end_dt = pd.Timestamp(storm_rows["dt"].max()) + pd.Timedelta(days=pad_days)

        lons = storm_rows["longitude"].astype(float).to_numpy()
        lons_360 = np.where(lons < 0, lons + 360, lons)
        lon_min = max(0.0, float(np.nanmin(lons_360)) - pad_deg)
        lon_max = min(360.0, float(np.nanmax(lons_360)) + pad_deg)
        lat_min = max(-90.0, float(storm_rows["latitude"].astype(float).min()) - pad_deg)
        lat_max = min(90.0, float(storm_rows["latitude"].astype(float).max()) + pad_deg)

        chunks = {"time": 1, "prediction_timedelta": -1, "level": -1, "latitude": 181, "longitude": 361}
        storage_options = {"token": "anon"}
        ds_raw = _open_hres_zarr(
            dataset_url,
            storage_options=storage_options,
            chunks=chunks,
            consolidated=consolidated,
            cache_dir=cache_dir,
        )
        ds_adapted = _adapt_variables(ds_raw)

        lat_name = "latitude" if "latitude" in ds_adapted.coords else "lat"
        lon_name = "longitude" if "longitude" in ds_adapted.coords else "lon"

        lat_slice = _lat_slice(ds_adapted[lat_name].values, lat_min, lat_max)
        lon_slice = slice(lon_min, lon_max) if lon_min <= lon_max else slice(None)

        ds_focus = ds_adapted.sel(
            {
                lat_name: lat_slice,
                lon_name: lon_slice,
                "time": slice(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")),
            }
        )
        if "time" not in ds_focus.coords:
            ds_focus.close()
            ds_raw.close()
            return _StormResult(storm_id, False, "no time coord after subsetting")

        init_times = pd.to_datetime(ds_focus["time"].values)
        if len(init_times) == 0:
            ds_focus.close()
            ds_raw.close()
            return _StormResult(storm_id, False, "no init times")

        all_points = _load_all_points(initials_csv)
        success = 0
        attempted = 0
        for init_time in init_times:
            ts = pd.Timestamp(init_time)
            if only_00_12z and ts.hour not in (0, 12):
                continue

            attempted += 1
            init_str = ts.strftime("%Y%m%d_%H%M")
            forecast_unique_id = f"{storm_id}_{init_str}"
            nc_stem = forecast_unique_id

            if _has_complete_json(final_dir, nc_stem, [storm_id]):
                log(f"⏭️  Skip existing JSON: {forecast_unique_id}")
                continue

            if max_forecasts_per_storm > 0 and success >= max_forecasts_per_storm:
                break

            ds_single = None
            ds_run = None
            try:
                ds_single = ds_focus.sel(time=init_time)
                ds_run = _prepare_single_forecast_dataset(ds_single)

                adapter = _DsAdapter.build(ds_run)
                times_idx = pd.DatetimeIndex(adapter.times)
                if len(times_idx) == 0:
                    continue

                tracking_start_time = max(ts, storm_start_dt)
                start_idx = _nearest_time_idx(times_idx, tracking_start_time)

                init_candidates = _select_initials_for_time(all_points, times_idx[start_idx], tol_hours=6)
                init_candidates = init_candidates.loc[init_candidates["storm_id"] == storm_id]
                if init_candidates.empty:
                    continue

                row = init_candidates.iloc[0]
                init_lat = float(row["init_lat"])
                init_lon = float(row["init_lon"])
                if float(adapter.lons.min()) >= 0 and init_lon < 0:
                    init_lon = init_lon % 360

                init_wind = float(row["max_wind_usa"]) if pd.notna(row.get("max_wind_usa")) else None
                init_msl = float(row["min_pressure_usa"]) * 100.0 if pd.notna(row.get("min_pressure_usa")) else None

                tracker = Tracker(
                    init_lat=init_lat,
                    init_lon=init_lon,
                    init_time=times_idx[start_idx].to_pydatetime(),
                    init_msl=init_msl,
                    init_wind=init_wind,
                )

                last_idx = min(len(adapter.times), start_idx + max_steps)
                for time_idx in range(start_idx, last_idx):
                    batch = _build_batch_from_ds_fast(adapter, time_idx)
                    try:
                        tracker.step(batch)
                    except NoEyeException:
                        if time_idx == start_idx:
                            tracker = None  # type: ignore[assignment]
                            break
                        continue
                    if tracker.dissipated:
                        break

                if tracker is None:
                    continue

                track_df = tracker.results().copy()
                if track_df.empty:
                    continue
                track_df["storm_id"] = storm_id
                track_df["particle"] = storm_id
                track_df["time"] = pd.to_datetime(track_df["time"], errors="coerce")

                ds_times = pd.to_datetime(ds_run["time"].values) if "time" in ds_run.coords else pd.DatetimeIndex([])
                ds_times_idx = pd.DatetimeIndex(ds_times)
                track_df["time_idx"] = track_df["time"].apply(
                    lambda t: _nearest_time_idx(ds_times_idx, pd.Timestamp(t)) if pd.notna(t) else 0
                )

                track_path = track_dir / f"track_{forecast_unique_id}.csv"
                track_df.to_csv(track_path, index=False)

                lat_vals = track_df["lat"].astype(float)
                lon_vals = track_df["lon"].astype(float)
                lon_grid = ds_run[lon_name].values
                if float(np.nanmin(lon_grid)) >= 0:
                    lon_vals = lon_vals % 360

                lat_min_t = max(float(lat_vals.min()) - track_pad_deg, float(ds_run[lat_name].values.min()))
                lat_max_t = min(float(lat_vals.max()) + track_pad_deg, float(ds_run[lat_name].values.max()))
                lon_min_t = max(float(lon_vals.min()) - track_pad_deg, float(np.nanmin(lon_grid)))
                lon_max_t = min(float(lon_vals.max()) + track_pad_deg, float(np.nanmax(lon_grid)))

                lat_s_t = _lat_slice(ds_run[lat_name].values, lat_min_t, lat_max_t)
                lon_s_t = slice(lon_min_t, lon_max_t)

                times_track = pd.to_datetime(track_df["time"])
                t_min = pd.Timestamp(times_track.min()) - pd.Timedelta(hours=6 * time_pad_steps)
                t_max = pd.Timestamp(times_track.max()) + pd.Timedelta(hours=6 * time_pad_steps)
                local_ds = ds_run.sel({lat_name: lat_s_t, lon_name: lon_s_t, "time": slice(t_min, t_max)})

                nc_path = nc_dir / f"{forecast_unique_id}.nc"
                engine = _infer_netcdf_engine()
                _persist_subset_to_netcdf(local_ds, nc_path, engine=engine)

                with TCEnvironmentalSystemsExtractor(
                    str(nc_path),
                    str(track_path),
                    enable_detailed_shape_analysis=enable_detailed_shape_analysis,
                ) as extractor:
                    extractor.analyze_and_export_as_json(output_dir=str(final_dir))

                if not keep_nc:
                    try:
                        nc_path.unlink()
                    except FileNotFoundError:
                        pass

                success += 1
                del local_ds
                gc.collect()
            except Exception:
                continue
            finally:
                if ds_run is not None:
                    try:
                        ds_run.close()
                    except Exception:
                        pass
                if ds_single is not None:
                    try:
                        ds_single.close()
                    except Exception:
                        pass
                gc.collect()

        ds_focus.close()
        ds_raw.close()
        elapsed = time.time() - t0
        if success > 0:
            return _StormResult(storm_id, True, f"processed={success} attempted={attempted} elapsed_s={elapsed:.1f}")
        return _StormResult(storm_id, False, f"no forecasts processed (attempted={attempted})")
    except Exception as exc:
        return _StormResult(storm_id, False, f"error: {exc}")


def _iter_storm_ids(df_tracks: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    df = df_tracks.copy()
    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    mask = (df["dt"] >= start) & (df["dt"] <= end)
    return sorted({str(s) for s in df.loc[mask, "storm_id"].dropna().unique().tolist()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WB2 HRES 下载 + 追踪 + 环境场提取（Slurm/HPC）")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--tracks-csv", default=str(Path("input") / "matched_cyclone_tracks.csv"))
    parser.add_argument("--initials-csv", default="", help="默认等于 --tracks-csv")
    parser.add_argument("--output-root", default=str(Path("colab_outputs") / "hres_pipeline"))
    parser.add_argument("--cache-dir", default="", help="使用 fsspec simplecache 先下载需要的 zarr chunks")
    parser.add_argument("--consolidated", default="1", help="Zarr 是否使用 consolidated metadata（默认 1）")

    parser.add_argument("--storm-id", default="", help="只处理单个 storm_id")
    parser.add_argument("--limit-storms", type=int, default=0, help="最多处理多少个 storm（0 表示全量）")
    parser.add_argument("--processes", type=int, default=1, help="并行处理 storm 的进程数")

    parser.add_argument("--max-forecasts-per-storm", type=int, default=0, help="每个 storm 最多处理多少个 init（0=不限）")
    parser.add_argument("--max-steps", type=int, default=40, help="追踪步数上限（6h/步）")
    parser.add_argument("--pad-days", type=int, default=1)
    parser.add_argument("--pad-deg", type=float, default=10.0)
    parser.add_argument("--track-pad-deg", type=float, default=8.0)
    parser.add_argument("--time-pad-steps", type=int, default=1)

    parser.add_argument("--only-00-12z", default="1")
    parser.add_argument("--keep-nc", default="0")
    parser.add_argument("--concise-log", default="0")
    parser.add_argument("--enable-detailed-shape-analysis", default="0")
    args = parser.parse_args(argv)

    dataset_url = str(args.dataset_url)
    tracks_csv = Path(args.tracks_csv).resolve()
    initials_csv = (Path(args.initials_csv).resolve() if args.initials_csv else tracks_csv)
    output_root = Path(args.output_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    if not tracks_csv.exists():
        print(f"❌ tracks_csv 不存在: {tracks_csv}", file=sys.stderr)
        return 2
    if not initials_csv.exists():
        print(f"❌ initials_csv 不存在: {initials_csv}", file=sys.stderr)
        return 2

    consolidated = _parse_bool(args.consolidated)
    only_00_12z = _parse_bool(args.only_00_12z)
    keep_nc = _parse_bool(args.keep_nc)
    concise_log = _parse_bool(args.concise_log)
    enable_detailed_shape_analysis = _parse_bool(args.enable_detailed_shape_analysis)

    df_tracks = pd.read_csv(tracks_csv)
    hres_start = pd.Timestamp("2016-01-01")
    hres_end = pd.Timestamp("2022-12-31")
    storm_ids = []
    if args.storm_id:
        storm_ids = [str(args.storm_id)]
    else:
        storm_ids = _iter_storm_ids(df_tracks, hres_start, hres_end)
        if args.limit_storms and args.limit_storms > 0:
            storm_ids = storm_ids[: args.limit_storms]

    if not storm_ids:
        print("❌ 未找到需要处理的 storm_id", file=sys.stderr)
        return 2

    layout = _ensure_output_layout(output_root)
    (layout["logs_dir"] / "cmd.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    print(f"dataset_url={dataset_url}")
    print(f"tracks_csv={tracks_csv}")
    print(f"initials_csv={initials_csv}")
    print(f"output_root={output_root}")
    if cache_dir is not None:
        print(f"cache_dir={cache_dir}")
    print(f"storms={len(storm_ids)} processes={args.processes}")

    worker_kwargs = dict(
        dataset_url=dataset_url,
        tracks_csv=tracks_csv,
        initials_csv=initials_csv,
        output_root=output_root,
        cache_dir=cache_dir,
        consolidated=consolidated,
        max_forecasts_per_storm=int(args.max_forecasts_per_storm),
        max_steps=int(args.max_steps),
        pad_days=int(args.pad_days),
        pad_deg=float(args.pad_deg),
        track_pad_deg=float(args.track_pad_deg),
        time_pad_steps=int(args.time_pad_steps),
        only_00_12z=only_00_12z,
        enable_detailed_shape_analysis=enable_detailed_shape_analysis,
        keep_nc=keep_nc,
        concise_log=concise_log,
    )

    start = time.time()
    results: list[_StormResult] = []

    processes = max(1, int(args.processes))
    if processes == 1:
        for sid in storm_ids:
            results.append(_process_single_storm(storm_id=sid, **worker_kwargs))
    else:
        import multiprocessing as mp

        with mp.Pool(processes=processes) as pool:
            asyncs = [
                pool.apply_async(_process_single_storm, kwds={"storm_id": sid, **worker_kwargs})
                for sid in storm_ids
            ]
            for a in asyncs:
                results.append(a.get())

    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    elapsed = time.time() - start
    print(f"done storms={len(results)} ok={len(ok)} bad={len(bad)} elapsed_s={elapsed:.1f}")
    if bad:
        print("failed storms (first 20):")
        for r in bad[:20]:
            print(f"  {r.storm_id}: {r.message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
