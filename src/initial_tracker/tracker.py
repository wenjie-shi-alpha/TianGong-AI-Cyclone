"""Core tracking logic operating on successive meteorological batches."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from .batching import _SimpleBatch
from .exceptions import NoEyeException
from .geo import get_box, get_closest_min, extrapolate

logger = logging.getLogger(__name__)


class Tracker:
    """Simple tropical cyclone tracker based on surface pressure minima."""

    def __init__(
        self,
        init_lat: float,
        init_lon: float,
        init_time: datetime,
        init_msl: float | None = None,
        init_wind: float | None = None,
    ) -> None:
        self.tracked_times: List[datetime] = [init_time]
        self.tracked_lats: List[float] = [init_lat]
        self.tracked_lons: List[float] = [init_lon]
        init_msl_val = float(init_msl) if init_msl is not None else np.nan
        init_wind_val = float(init_wind) if init_wind is not None else np.nan
        if not np.isfinite(init_msl_val):
            init_msl_val = np.nan
        if not np.isfinite(init_wind_val):
            init_wind_val = np.nan
        self.tracked_msls: List[float] = [init_msl_val]
        self.tracked_winds: List[float] = [init_wind_val]
        self.fails: int = 0
        self.last_success_time: Optional[datetime] = init_time
        self.dissipated: bool = False
        self.dissipated_time: Optional[datetime] = None
        self.dissipation_reason: Optional[str] = None
        self.peak_pressure_drop_hpa: float = 0.0
        self.peak_wind: float = 0.0

    def results(self) -> pd.DataFrame:
        """Assemble the current track as a DataFrame."""
        return pd.DataFrame(
            {
                "time": self.tracked_times,
                "lat": self.tracked_lats,
                "lon": self.tracked_lons,
                "msl": self.tracked_msls,
                "wind": self.tracked_winds,
            }
        )

    def step(self, batch: _SimpleBatch) -> None:
        """Advance the tracker by one time step using the provided batch."""
        if len(batch.metadata.time) != 1:
            raise RuntimeError("Predictions don't have batch size one.")

        if self.dissipated:
            logger.debug("Tracker already dissipated; skipping step at %s", batch.metadata.time[0])
            return

        batch = batch.to("cpu")

        z700 = None
        if "z" in batch.atmos_vars and len(batch.metadata.atmos_levels) > 0:
            levels = np.array(batch.metadata.atmos_levels)
            try:
                levels_float = levels.astype(float)
                idx = int(np.argmin(np.abs(levels_float - 700)))
                z700 = batch.atmos_vars["z"][0, 0, idx]
            except Exception:
                z700 = None

        msl = batch.surf_vars["msl"][0, 0]
        u10 = batch.surf_vars["10u"][0, 0]
        v10 = batch.surf_vars["10v"][0, 0]
        wind = np.sqrt(u10 * u10 + v10 * v10)
        lsm = batch.static_vars["lsm"]
        lats = np.array(batch.metadata.lat)
        lons = np.array(batch.metadata.lon)
        time = batch.metadata.time[0]

        lat, lon = extrapolate(self.tracked_lats, self.tracked_lons)
        lat = max(min(lat, 90), -90)
        lon = lon % 360

        def is_clear(lat_val: float, lon_val: float, delta: float) -> bool:
            _, _, lsm_box = get_box(
                lsm,
                lats,
                lons,
                lat_val - delta,
                lat_val + delta,
                lon_val - delta,
                lon_val + delta,
            )
            return lsm_box.max() < 0.5

        snap = False
        for delta in [5, 4, 3, 2, 1.5]:
            try:
                if is_clear(lat, lon, delta):
                    lat, lon = get_closest_min(msl, lats, lons, lat, lon, delta_lat=delta, delta_lon=delta)
                    snap = True
                    break
            except NoEyeException:
                pass

        if not snap and z700 is not None:
            try:
                lat, lon = get_closest_min(
                    z700,
                    lats,
                    lons,
                    lat,
                    lon,
                    delta_lat=5,
                    delta_lon=5,
                )
                snap = True
                for delta in [5, 4, 3, 2, 1.5]:
                    try:
                        if is_clear(lat, lon, delta):
                            lat, lon = get_closest_min(
                                msl,
                                lats,
                                lons,
                                lat,
                                lon,
                                delta_lat=delta,
                                delta_lon=delta,
                            )
                            break
                    except NoEyeException:
                        pass
            except NoEyeException:
                pass

        if snap:
            self.fails = 0
            self.last_success_time = time
        else:
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info("Failed at time %s. Extrapolating in a silly way.", time)
            else:
                raise NoEyeException("Completely failed at the first step.")

        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)

        _, _, msl_crop = get_box(msl, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)
        _, _, wind_crop = get_box(wind, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)
        min_msl = float(np.nanmin(msl_crop))
        max_wind = float(np.nanmax(wind_crop))
        self.tracked_msls.append(min_msl)
        self.tracked_winds.append(max_wind)

        pressure_drop_hpa = self._compute_pressure_drop_hpa(msl, lats, lons, lat, lon, min_msl)
        if pressure_drop_hpa is not None:
            self.peak_pressure_drop_hpa = max(self.peak_pressure_drop_hpa, pressure_drop_hpa)

        if np.isfinite(max_wind):
            self.peak_wind = max(self.peak_wind, max_wind)

        self._check_dissipation(time, snap, pressure_drop_hpa, max_wind)


    def _compute_pressure_drop_hpa(
        self,
        msl: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        lat: float,
        lon: float,
        center_pressure: float,
    ) -> Optional[float]:
        if not np.isfinite(center_pressure):
            return None

        lat_box, lon_box, msl_box = get_box(msl, lats, lons, lat - 7.0, lat + 7.0, lon - 7.0, lon + 7.0)
        if msl_box.size == 0:
            return None

        lat_grid, lon_grid = np.meshgrid(lat_box, lon_box, indexing="ij")
        lat0 = np.deg2rad(lat)
        lon0 = np.deg2rad(lon % 360)
        lat_grid_rad = np.deg2rad(lat_grid)
        lon_grid_rad = np.deg2rad(lon_grid % 360)
        dlat = lat_grid_rad - lat0
        dlon = lon_grid_rad - lon0
        dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat_grid_rad) * np.sin(dlon / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        angle = 2.0 * np.arcsin(np.sqrt(a))
        distance_deg = np.rad2deg(angle)

        annulus_mask = (distance_deg >= 5.0) & (distance_deg <= 7.0)
        if not np.any(annulus_mask):
            annulus_mask = distance_deg >= 5.0
            if not np.any(annulus_mask):
                return None

        periphery_values = msl_box[annulus_mask]
        periphery_values = periphery_values[np.isfinite(periphery_values)]
        if periphery_values.size == 0:
            return None

        periphery_pressure = float(np.mean(periphery_values))
        if not np.isfinite(periphery_pressure):
            return None

        scale = 100.0 if max(abs(periphery_pressure), abs(center_pressure)) > 2000 else 1.0
        drop = (periphery_pressure - center_pressure) / scale
        return float(max(drop, 0.0))

    def _check_dissipation(
        self,
        time: datetime,
        snap: bool,
        pressure_drop_hpa: Optional[float],
        max_wind: float,
    ) -> None:
        if self.dissipated:
            return

        if not snap and self.fails >= 3 and self.last_success_time is not None:
            duration = time - self.last_success_time
            if duration >= timedelta(hours=18):
                self._mark_dissipated(time, "连续追踪失败18小时")
                return

        if pressure_drop_hpa is not None:
            if pressure_drop_hpa < 1.0:
                self._mark_dissipated(time, "中心-外围压差低于1.0 hPa")
                return
            if self.peak_pressure_drop_hpa > 0 and pressure_drop_hpa < 0.25 * self.peak_pressure_drop_hpa:
                self._mark_dissipated(time, "结构强度降至峰值的25%以下 (压差)")
                return
        elif np.isfinite(max_wind) and self.peak_wind > 0:
            if max_wind < 0.25 * self.peak_wind:
                self._mark_dissipated(time, "结构强度降至峰值的25%以下 (10米风)")

    def _mark_dissipated(self, time: datetime, reason: str) -> None:
        if self.dissipated:
            return
        self.dissipated = True
        self.dissipated_time = time
        self.dissipation_reason = reason
        logger.info("Cyclone dissipated at %s: %s", time, reason)

__all__ = ["Tracker"]
