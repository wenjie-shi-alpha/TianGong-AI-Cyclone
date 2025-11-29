"""High-level tracking workflow that bridges datasets and the Tracker class."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .batching import _Metadata, _SimpleBatch
from .dataset_adapter import _DsAdapter
from .exceptions import NoEyeException
from .initials import _select_initials_for_time
from .tracker import Tracker

logger = logging.getLogger(__name__)


def _inside_domain(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> bool:
    lon_mod = lon % 360
    return (lats.min() - 1 <= lat <= lats.max() + 1) and (
        lons.min() - 1 <= lon_mod <= lons.max() + 1
    )


def track_file_with_initials(
    nc_path: Path,
    all_points: pd.DataFrame,
    output_dir: Path,
    max_storms: Optional[int] = None,
    time_window_hours: int = 6,
) -> List[Path]:
    """Track all storms within a NetCDF file and export results as CSV files."""
    ds = xr.open_dataset(nc_path)
    adapter = _DsAdapter.build(ds)
    lats = adapter.lats
    lons = adapter.lons
    times = adapter.times
    if len(times) == 0:
        ds.close()
        raise ValueError(f"文件无时间维度: {nc_path}")

    t0 = pd.Timestamp(times[0])
    initials = _select_initials_for_time(all_points, t0, tol_hours=time_window_hours)
    if initials.empty:
        logger.info(
            "%s: 在 ±%s 小时内未匹配到任何气旋初始点, 跳过",
            nc_path.name,
            time_window_hours,
        )
        ds.close()
        return []

    written: List[Path] = []
    count = 0
    time_cache: Dict[int, Dict[str, np.ndarray]] = {}

    for _, row in initials.iterrows():
        if max_storms is not None and count >= max_storms:
            break

        storm_id = str(row["storm_id"]) if "storm_id" in row else f"storm_{count}"
        init_lat = float(row["init_lat"]) if "init_lat" in row else float(row.get("latitude"))  # type: ignore[arg-type]
        init_lon = float(row["init_lon"]) if "init_lon" in row else float(row.get("longitude"))  # type: ignore[arg-type]

        if not _inside_domain(init_lat, init_lon, lats, lons):
            logger.debug("%s: 初始点 (%.2f, %.2f) 超出网格范围, 跳过", storm_id, init_lat, init_lon)
            continue

        def _safe_float(val: object) -> float | None:
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        init_wind = _safe_float(row.get("max_wind_usa"))
        init_msl = _safe_float(row.get("min_pressure_usa"))
        if init_msl is not None:
            init_msl *= 100.0  # catalogue provides hPa, tracker persists Pascals

        tracker = Tracker(
            init_lat=init_lat,
            init_lon=init_lon,
            init_time=times[0],
            init_msl=init_msl,
            init_wind=init_wind,
        )

        for time_idx in range(len(times)):
            cache = time_cache.get(time_idx)
            if cache is None:
                msl_2d = adapter.msl_at(time_idx)
                u10_2d = adapter.u10_at(time_idx)
                v10_2d = adapter.v10_at(time_idx)
                z2d = adapter.z_near700_at(time_idx)
                cache = {"msl": msl_2d, "10u": u10_2d, "10v": v10_2d}
                if z2d is not None:
                    cache["z"] = z2d
                time_cache[time_idx] = cache

            surf_vars = {
                "msl": cache["msl"][np.newaxis, np.newaxis, ...],
                "10u": cache["10u"][np.newaxis, np.newaxis, ...],
                "10v": cache["10v"][np.newaxis, np.newaxis, ...],
            }
            atmos_vars: Dict[str, np.ndarray] = {}
            if "z" in cache:
                atmos_vars["z"] = cache["z"][np.newaxis, np.newaxis, np.newaxis, ...]
            metadata = _Metadata(
                lat=lats,
                lon=lons,
                time=[times[time_idx]],
                atmos_levels=[adapter.z_level_near_700 or 700],
            )
            static_vars = {"lsm": adapter.lsm}
            batch = _SimpleBatch(atmos_vars=atmos_vars, surf_vars=surf_vars, static_vars=static_vars, metadata=metadata)

            try:
                tracker.step(batch)
            except NoEyeException as exc:
                if time_idx == 0:
                    logger.info("%s: 首步失败, 跳过 (%s)", storm_id, exc)
                    tracker = None  # type: ignore[assignment]
                    break
                logger.info("%s: 非首步异常, 忽略该时次并继续 (t=%s) -> %s", storm_id, times[time_idx], exc)
                continue

            if tracker.dissipated:
                diss_time = tracker.dissipated_time or times[time_idx]
                logger.info(
                    "%s: 在 %s 判定气旋消亡, 原因: %s",
                    storm_id,
                    diss_time,
                    tracker.dissipation_reason or "未提供原因",
                )
                break

        if tracker is None:
            continue

        out_df = tracker.results()
        if len(out_df) <= 1:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = nc_path.stem
        out_name = f"track_{storm_id}_{stem}.csv"
        out_path = output_dir / out_name
        out_df.to_csv(out_path, index=False)
        written.append(out_path)
        count += 1

    ds.close()
    return written


__all__ = ["track_file_with_initials"]
