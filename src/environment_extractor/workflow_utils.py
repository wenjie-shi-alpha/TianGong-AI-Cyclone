"""Shared helpers for the environment extraction workflow.

These utilities were previously defined inside ``trackTC`` but are now
collocated with the environment extractor to avoid the heavy tracking
dependencies in the workflow entry points.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import boto3
from botocore import UNSIGNED
from botocore.config import Config

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_\-]+")
_FORECAST_PATTERN = re.compile(r"(f\d{3}_f\d{3}_\d{2})")


def sanitize_filename(text: str) -> str:
    """Replace characters that are unsafe for filesystem paths."""

    return _SANITIZE_PATTERN.sub("_", text)


def extract_forecast_tag(name: str) -> str:
    """Return the forecast tag contained in ``name`` or ``track`` as a fallback."""

    match = _FORECAST_PATTERN.search(Path(name).stem)
    return match.group(1) if match else "track"


def download_s3_public(s3_url: str, target_path: Path) -> None:
    """Download a public S3 object without credentials."""

    if not s3_url.startswith("s3://"):
        raise ValueError(f"无效S3 URL: {s3_url}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = s3_url[5:].split("/", 1)
    client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    client.download_file(bucket, key, str(target_path))


if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd


def combine_initial_tracker_outputs(
    per_storm_csvs: Iterable[Path],
    nc_path: Path,
) -> "pd.DataFrame | None":
    """Merge per-storm CSV outputs from ``initialTracker``."""

    import numpy as np
    import pandas as pd
    import xarray as xr

    per_storm_paths = [Path(p) for p in per_storm_csvs if Path(p).exists()]
    if not per_storm_paths:
        return None

    try:
        with xr.open_dataset(nc_path) as ds:
            ds_times = pd.to_datetime(ds.time.values) if "time" in ds.coords else []
    except Exception:
        ds_times = []

    def nearest_time_idx(ts: pd.Timestamp) -> int:
        if len(ds_times) == 0 or pd.isna(ts):
            return 0
        try:
            return int(np.argmin(np.abs(ds_times - ts)))
        except Exception:
            return 0

    nc_stem = Path(nc_path).stem
    parts = []
    for csv_path in per_storm_paths:
        try:
            df_piece = pd.read_csv(csv_path)
        except Exception:
            continue
        particle_id = _infer_particle_id(csv_path, nc_stem)
        df_piece["particle"] = particle_id
        if "time" in df_piece.columns:
            df_piece["time"] = pd.to_datetime(df_piece["time"], errors="coerce")
            df_piece["time_idx"] = df_piece["time"].apply(nearest_time_idx)
        else:
            df_piece["time_idx"] = np.arange(len(df_piece))
        parts.append(df_piece)

    if not parts:
        return None

    return pd.concat(parts, ignore_index=True)


def _infer_particle_id(csv_path: Path, nc_stem: str) -> str:
    """Best-effort extraction of the particle ID from the per-storm filename."""

    base = csv_path.stem
    pattern = re.compile(r"track_(.+?)_" + re.escape(nc_stem) + r"$")
    match = pattern.match(base)
    if match:
        return match.group(1)
    return base.replace("track_", "")
