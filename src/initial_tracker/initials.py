"""Utilities for loading and filtering cyclone initial positions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_all_points(csv_path: Path) -> pd.DataFrame:
    """Read the cyclone catalogue and normalise time information."""
    df = pd.read_csv(csv_path)
    required = {"storm_id", "datetime", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 缺少必要列: {required - set(df.columns)}")
    df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    return df


def _load_initial_points(csv_path: Path) -> pd.DataFrame:
    """Backward-compatible alias used by downstream code."""
    return _load_all_points(csv_path)


def _select_initials_for_time(
    df_all: pd.DataFrame,
    target_time: pd.Timestamp,
    tol_hours: int = 6,
) -> pd.DataFrame:
    """Select the best matching initial point for each storm near a target time."""
    if df_all.empty:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])
    delta = pd.Timedelta(hours=tol_hours)
    sub = df_all.loc[(df_all["dt"] >= target_time - delta) & (df_all["dt"] <= target_time + delta)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["storm_id", "init_time", "init_lat", "init_lon"])
    sub["time_diff"] = (sub["dt"] - target_time).abs()
    idx = sub.groupby("storm_id")["time_diff"].idxmin()
    pick = sub.loc[idx].copy()
    pick = pick.rename(columns={"latitude": "init_lat", "longitude": "init_lon"})
    pick["init_time"] = pick["dt"].values
    return pick[["storm_id", "init_time", "init_lat", "init_lon"]].reset_index(drop=True)


__all__ = ["_load_all_points", "_load_initial_points", "_select_initials_for_time"]
