"""Utilities for adapting raw xarray datasets into tracker-compatible batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .batching import _Metadata, _SimpleBatch


def _to_0360(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from [-180, 180] to [0, 360], keeping existing [0, 360] unchanged."""
    lon = np.asarray(lon)
    if lon.min() < 0:
        lon = (lon + 360.0) % 360.0
    return lon


def _safe_get(ds: xr.Dataset, names: Iterable[str]) -> Optional[xr.DataArray]:
    for name in names:
        if name in ds.data_vars or name in ds.coords:
            return ds[name]
    return None


@dataclass
class _DsAdapter:
    """Thin wrapper around an xarray dataset that caches common metadata."""

    ds: xr.Dataset
    lat_name: str
    lon_name: str
    lats: np.ndarray
    lons: np.ndarray
    times: List[pd.Timestamp]
    msl_name: str
    u10_name: Optional[str]
    v10_name: Optional[str]
    z_name: Optional[str]
    z_level_dim: Optional[str]
    z_levels_hpa: Optional[np.ndarray]
    z_idx_near_700: Optional[int]
    z_level_near_700: Optional[int]
    lsm: np.ndarray

    @staticmethod
    def build(ds: xr.Dataset) -> "_DsAdapter":
        lat_da = _safe_get(ds, ["latitude", "lat"])
        lon_da = _safe_get(ds, ["longitude", "lon"])
        if lat_da is None or lon_da is None:
            raise ValueError("Dataset 缺少经纬度坐标 (lat/lon 或 latitude/longitude)")
        lat_name = str(lat_da.name)
        lon_name = str(lon_da.name)
        lats = lat_da.values
        lons = _to_0360(lon_da.values)

        if "time" in ds.coords:
            times = list(pd.to_datetime(ds.time.values))
        else:
            times = [pd.Timestamp.now()]

        msl_da = _safe_get(ds, ["msl", "mslp"])
        if msl_da is None:
            raise ValueError("Dataset 缺少海平面气压 (msl/mslp)")
        msl_name = str(msl_da.name)

        u10_da = _safe_get(ds, ["u10", "10u"])
        v10_da = _safe_get(ds, ["v10", "10v"])
        u10_name = str(u10_da.name) if u10_da is not None else None
        v10_name = str(v10_da.name) if v10_da is not None else None

        z_da = _safe_get(ds, ["z", "gh", "geopotential", "geopotential_height"])
        z_name = str(z_da.name) if z_da is not None else None
        z_level_dim: Optional[str] = None
        z_levels_hpa: Optional[np.ndarray] = None
        z_idx_near_700: Optional[int] = None
        z_level_near_700: Optional[int] = None

        if z_da is not None:
            for dim in ["level", "isobaricInhPa", "pressure", "isobaricInPa"]:
                if dim in z_da.dims:
                    z_level_dim = dim
                    break
            if z_level_dim is None:
                z_levels_hpa = np.array([700], dtype=int)
                z_idx_near_700 = 0
                z_level_near_700 = 700
            else:
                levels_vals = z_da[z_level_dim].values
                if levels_vals.max() > 2000:
                    levels_vals = (levels_vals / 100.0).astype(float)
                levels_hpa = np.array([int(round(v)) for v in levels_vals.tolist()])
                z_levels_hpa = levels_hpa
                z_idx_near_700 = int(np.argmin(np.abs(levels_hpa.astype(float) - 700.0)))
                z_level_near_700 = int(levels_hpa[z_idx_near_700])

        lsm_da = _safe_get(ds, ["lsm", "land_sea_mask", "landmask"])
        if lsm_da is not None:
            lsm = lsm_da.values
            if lsm.ndim == 3:
                lsm = lsm[0]
        else:
            sample = ds[msl_name].isel(time=0).values
            if sample.ndim == 3:
                sample = sample[0]
            lsm = np.zeros_like(sample)

        return _DsAdapter(
            ds=ds,
            lat_name=lat_name,
            lon_name=lon_name,
            lats=lats,
            lons=lons,
            times=times,
            msl_name=msl_name,
            u10_name=u10_name,
            v10_name=v10_name,
            z_name=z_name,
            z_level_dim=z_level_dim,
            z_levels_hpa=z_levels_hpa,
            z_idx_near_700=z_idx_near_700,
            z_level_near_700=z_level_near_700,
            lsm=lsm,
        )

    def msl_at(self, time_idx: int) -> np.ndarray:
        arr = self.ds[self.msl_name].isel(time=time_idx).values
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def u10_at(self, time_idx: int) -> np.ndarray:
        if self.u10_name is None:
            return np.zeros_like(self.msl_at(time_idx))
        arr = self.ds[self.u10_name].isel(time=time_idx).values
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def v10_at(self, time_idx: int) -> np.ndarray:
        if self.v10_name is None:
            return np.zeros_like(self.msl_at(time_idx))
        arr = self.ds[self.v10_name].isel(time=time_idx).values
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def z_near700_at(self, time_idx: int) -> Optional[np.ndarray]:
        if self.z_name is None:
            return None
        if self.z_level_dim is None:
            z2d = self.ds[self.z_name].isel(time=time_idx).values
            if z2d.ndim == 3:
                z2d = z2d[0]
        else:
            assert self.z_idx_near_700 is not None
            z_sel = self.ds[self.z_name].isel(time=time_idx, **{self.z_level_dim: self.z_idx_near_700})
            z2d_da = z_sel.transpose(self.lat_name, self.lon_name)
            z2d = z2d_da.values
        return z2d


def _build_batch_from_ds_fast(adapter: _DsAdapter, time_idx: int) -> _SimpleBatch:
    lats = adapter.lats
    lons = adapter.lons
    time_val = adapter.times[time_idx]

    msl_2d = adapter.msl_at(time_idx)
    u10_2d = adapter.u10_at(time_idx)
    v10_2d = adapter.v10_at(time_idx)

    surf_vars = {
        "msl": msl_2d[np.newaxis, np.newaxis, ...],
        "10u": u10_2d[np.newaxis, np.newaxis, ...],
        "10v": v10_2d[np.newaxis, np.newaxis, ...],
    }

    atmos_vars: Dict[str, np.ndarray] = {}
    atmos_levels = [adapter.z_level_near_700 or 700]
    z2d = adapter.z_near700_at(time_idx)
    if z2d is not None:
        atmos_vars["z"] = z2d[np.newaxis, np.newaxis, np.newaxis, ...]

    static_vars = {"lsm": adapter.lsm}
    metadata = _Metadata(lat=lats, lon=lons, time=[time_val], atmos_levels=atmos_levels)
    return _SimpleBatch(atmos_vars=atmos_vars, surf_vars=surf_vars, static_vars=static_vars, metadata=metadata)


def _build_batch_from_ds(ds: xr.Dataset, time_idx: int) -> _SimpleBatch:
    lat_da = _safe_get(ds, ["latitude", "lat"])
    lon_da = _safe_get(ds, ["longitude", "lon"])
    if lat_da is None or lon_da is None:
        raise ValueError("Dataset 缺少经纬度坐标 (lat/lon 或 latitude/longitude)")
    lats = lat_da.values
    lons = _to_0360(lon_da.values)

    msl_da = _safe_get(ds, ["msl", "mslp"])
    if msl_da is None:
        raise ValueError("Dataset 缺少海平面气压 (msl/mslp)")
    msl_2d = msl_da.isel(time=time_idx).values
    if msl_2d.ndim == 3:
        msl_2d = msl_2d[0]

    u10_da = _safe_get(ds, ["u10", "10u"])
    v10_da = _safe_get(ds, ["v10", "10v"])
    if u10_da is None or v10_da is None:
        u10_2d = np.zeros_like(msl_2d)
        v10_2d = np.zeros_like(msl_2d)
    else:
        u10_2d = u10_da.isel(time=time_idx).values
        v10_2d = v10_da.isel(time=time_idx).values
        if u10_2d.ndim == 3:
            u10_2d = u10_2d[0]
        if v10_2d.ndim == 3:
            v10_2d = v10_2d[0]

    z_da = _safe_get(ds, ["z", "gh", "geopotential", "geopotential_height"])
    atmos_levels: List[int] = []
    z_3d: Optional[np.ndarray] = None
    if z_da is not None:
        level_dim = None
        for dim in ["level", "isobaricInhPa", "pressure", "isobaricInPa"]:
            if dim in z_da.dims:
                level_dim = dim
                break
        if level_dim is None:
            z_2d = z_da.isel(time=time_idx).values
            if z_2d.ndim == 3:
                z_2d = z_2d[0]
            z_3d = z_2d[np.newaxis, ...]
            atmos_levels = [700]
        else:
            levels_vals = z_da[level_dim].values
            if levels_vals.max() > 2000:
                levels_vals = (levels_vals / 100.0).astype(float)
            atmos_levels = [int(round(v)) for v in levels_vals.tolist()]
            z_sel = z_da.isel(time=time_idx)
            order = ("latitude" if "latitude" in z_sel.dims else "lat", "longitude" if "longitude" in z_sel.dims else "lon")
            z_3d = z_sel.transpose(..., *order).values  # type: ignore[arg-type]
            if z_3d.ndim == 4:
                z_3d = z_3d[0]
            if z_3d.shape[0] != len(atmos_levels):
                if z_3d.shape[-3] == len(atmos_levels):
                    z_3d = np.moveaxis(z_3d, -3, 0)

    lsm_da = _safe_get(ds, ["lsm", "land_sea_mask", "landmask"])
    if lsm_da is not None:
        lsm = lsm_da.values
        if lsm.ndim == 3:
            lsm = lsm[0]
    else:
        lsm = np.zeros_like(msl_2d)

    time_val = pd.to_datetime(ds.time.values[time_idx]) if "time" in ds.coords else pd.Timestamp.now()
    metadata = _Metadata(lat=lats, lon=lons, time=[time_val], atmos_levels=atmos_levels or [700])

    surf_vars = {
        "msl": msl_2d[np.newaxis, np.newaxis, ...],
        "10u": u10_2d[np.newaxis, np.newaxis, ...],
        "10v": v10_2d[np.newaxis, np.newaxis, ...],
    }
    atmos_vars: Dict[str, np.ndarray] = {}
    if z_3d is not None:
        atmos_vars["z"] = z_3d[np.newaxis, np.newaxis, ...]

    static_vars = {"lsm": lsm}
    return _SimpleBatch(atmos_vars=atmos_vars, surf_vars=surf_vars, static_vars=static_vars, metadata=metadata)


__all__ = [
    "_DsAdapter",
    "_build_batch_from_ds_fast",
    "_build_batch_from_ds",
    "_safe_get",
    "_to_0360",
]
