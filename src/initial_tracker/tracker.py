"""Core tracking logic operating on successive meteorological batches."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from .batching import _SimpleBatch
from .exceptions import NoEyeException
from .geo import (
    bilinear_interpolate,
    get_box,
    get_closest_min,
    extrapolate,
    havdist,
    snap_to_grid,
)

logger = logging.getLogger(__name__)

# Optional relaxed mode for weaker systems (e.g., EP/ATL lows). Enable by
# setting environment variable RELAXED_TRACKING=1 when running the pipeline.
_RELAXED = os.getenv("RELAXED_TRACKING", "0").lower() in {"1", "true", "yes"}


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
        # Normalize longitude to [0, 360) to avoid wrap-induced extrapolation jumps
        norm_init_lon = (init_lon + 360.0) % 360.0
        self.tracked_lons: List[float] = [norm_init_lon]
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
        self._init_snapped: bool = False
        self._fix_lats: List[float] = [self.tracked_lats[0]]
        self._fix_lons: List[float] = [self.tracked_lons[0]]
        self._stationary_steps: int = 0

        # Relaxed thresholds extend tolerance for weak or messy systems.
        self._max_consecutive_fails = 10 if not _RELAXED else 12
        self._max_fail_hours = 24 if not _RELAXED else 36
        self._min_drop_hpa = 0.5 if not _RELAXED else 0.3
        self._min_drop_fraction = 0.2 if not _RELAXED else 0.1
        self._max_step_km = 350.0 if not _RELAXED else 450.0
        self._stationary_distance_km = 60.0
        self._max_stationary_steps = 6 if not _RELAXED else 8
        self._stationary_intensity_fraction = 0.35 if not _RELAXED else 0.25

    def results(self) -> pd.DataFrame:
        """Assemble the current track as a DataFrame."""
        df = pd.DataFrame(
            {
                "time": self.tracked_times,
                "lat": self.tracked_lats,
                "lon": self.tracked_lons,
                "msl": self.tracked_msls,
                "wind": self.tracked_winds,
            }
        )

        def _dlon(a: float, b: float) -> float:
            diff = abs(a - b)
            return min(diff, 360.0 - diff)

        def _compute_stationary(frame: pd.DataFrame) -> list[bool]:
            length = len(frame)
            flags = [False] * length
            if length < 2:
                return flags
            lat_vals = frame["lat"].to_numpy(dtype=float)
            lon_vals = frame["lon"].to_numpy(dtype=float)
            wind_vals = frame["wind"].to_numpy(dtype=float)
            for idx in range(1, length):
                still = abs(lat_vals[idx] - lat_vals[idx - 1]) < 0.05
                still &= _dlon(lon_vals[idx], lon_vals[idx - 1]) < 0.05
                wind_ok = (not np.isfinite(wind_vals[idx])) or wind_vals[idx] < 12.0
                flags[idx] = bool(still and wind_ok)
            return flags

        # Prune trailing flat segments (little to no motion and weak winds) to avoid artificial plateaus.
        if len(df) >= 5:
            stationary = _compute_stationary(df)
            cutoff = len(df)
            while cutoff > 3:
                if stationary[cutoff - 1]:
                    cutoff -= 1
                else:
                    break
            if cutoff < len(df):
                df = df.iloc[:cutoff].reset_index(drop=True)

        return df

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

        # Constrain the initial catalogue position to the model domain once we know it.
        if not self._init_snapped:
            snapped_lat, snapped_lon = snap_to_grid(self.tracked_lats[0], self.tracked_lons[0], lats, lons)
            self.tracked_lats[0] = snapped_lat
            self.tracked_lons[0] = snapped_lon
            self._fix_lats[0] = snapped_lat
            self._fix_lons[0] = snapped_lon
            self._init_snapped = True

        history_lats = self._fix_lats if self._fix_lats else self.tracked_lats
        history_lons = self._fix_lons if self._fix_lons else self.tracked_lons
        guess_lat, guess_lon = extrapolate(history_lats, history_lons)
        guess_lat = max(min(guess_lat, 90), -90)
        guess_lon = guess_lon % 360
        lat, lon = guess_lat, guess_lon

        def can_search(
            lat_val: float,
            lon_val: float,
            delta: float,
            allow_empty: bool,
            max_land_fraction: float,
        ) -> bool:
            _, _, lsm_box = get_box(
                lsm,
                lats,
                lons,
                lat_val - delta,
                lat_val + delta,
                lon_val - delta,
                lon_val + delta,
            )
            if lsm_box.size == 0:
                return allow_empty
            lsm_vals = np.asarray(lsm_box, dtype=float)
            if not np.isfinite(lsm_vals).any():
                return True
            lsm_vals = np.clip(np.nan_to_num(lsm_vals, nan=1.0), 0.0, 1.0)
            land_fraction = float(np.mean(lsm_vals >= 0.8))
            sea_fraction = float(np.mean(lsm_vals <= 0.2))
            if sea_fraction <= 0 and not allow_empty:
                return False
            return land_fraction <= max_land_fraction or sea_fraction >= 0.25

        def build_search_radii() -> list[float]:
            base = [5.0, 4.0, 3.0, 2.0, 1.5]
            max_radius = min(5.0 + self.fails, 8.0)
            extras: list[float] = []
            current = max_radius
            while current > 5.0 + 1e-6:
                extras.append(current)
                current -= 1.0
            ordered = extras + base
            seen: set[float] = set()
            radii: list[float] = []
            for val in ordered:
                capped = max(1.5, min(val, max_radius))
                key = round(capped, 2)
                if key not in seen:
                    radii.append(capped)
                    seen.add(key)
            return radii

        snap = False
        search_radii = build_search_radii()
        for delta in search_radii:
            try:
                if can_search(lat, lon, delta, allow_empty=False, max_land_fraction=0.6):
                    lat, lon = get_closest_min(msl, lats, lons, lat, lon, delta_lat=delta, delta_lon=delta)
                    snap = True
                    break
            except NoEyeException:
                pass

        if not snap:
            for delta in search_radii:
                try:
                    if can_search(lat, lon, delta, allow_empty=True, max_land_fraction=0.95):
                        lat, lon = get_closest_min(msl, lats, lons, lat, lon, delta_lat=delta, delta_lon=delta)
                        snap = True
                        break
                except NoEyeException:
                    pass

        if not snap and z700 is not None:
            try:
                z_delta = search_radii[0] if search_radii else 5.0
                lat, lon = get_closest_min(
                    z700,
                    lats,
                    lons,
                    lat,
                    lon,
                    delta_lat=z_delta,
                    delta_lon=z_delta,
                )
                snap = True
                for delta in search_radii:
                    try:
                        if can_search(lat, lon, delta, allow_empty=False, max_land_fraction=0.6):
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
                if snap:
                    for delta in search_radii:
                        try:
                            if can_search(lat, lon, delta, allow_empty=True, max_land_fraction=0.95):
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

        movement_km = 0.0
        if snap and self._fix_lats:
            prev_lat = self._fix_lats[-1]
            prev_lon = self._fix_lons[-1]
            step_distance = havdist(prev_lat, prev_lon, lat, lon)
            if step_distance > self._max_step_km:
                logger.debug(
                    "Discarded jump of %.1f km (> %.1f km limit) between fixes", step_distance, self._max_step_km
                )
                snap = False
            else:
                movement_km = step_distance

        if snap:
            self.fails = 0
            self.last_success_time = time
        else:
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info("Failed at time %s. Extrapolating in a silly way.", time)
            else:
                raise NoEyeException("Completely failed at the first step.")

            if self.fails >= self._max_consecutive_fails:
                self._mark_dissipated(time, f"连续{self.fails}次未找到中心，终止追踪")
                return

            # Use the last known good position to avoid runaway drift between steps.
            lat = self._fix_lats[-1]
            lon = self._fix_lons[-1]

        # Keep the coordinate inside the native domain while preserving sub-grid offsets.
        lat, lon = snap_to_grid(lat, lon, lats, lons)

        if snap:
            self._fix_lats.append(lat)
            self._fix_lons.append(lon)
            if len(self._fix_lats) > 16:
                del self._fix_lats[:-16]
                del self._fix_lons[:-16]
        else:
            movement_km = 0.0

        if movement_km < self._stationary_distance_km:
            self._stationary_steps += 1
        else:
            self._stationary_steps = 0

        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)

        _, _, msl_crop = get_box(msl, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)
        _, _, wind_crop = get_box(wind, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)

        def _safe_min(arr: np.ndarray, fallback: float) -> float:
            if arr.size == 0 or not np.isfinite(arr).any():
                return float(fallback) if np.isfinite(fallback) else float("nan")
            return float(np.nanmin(arr))

        def _safe_max(arr: np.ndarray, fallback: float) -> float:
            if arr.size == 0 or not np.isfinite(arr).any():
                return float(fallback) if np.isfinite(fallback) else float("nan")
            return float(np.nanmax(arr))

        if msl.ndim >= 2:
            center_msl = bilinear_interpolate(msl, lats, lons, lat, lon)
        else:
            center_msl = float(msl)
        if wind.ndim >= 2:
            center_wind = bilinear_interpolate(wind, lats, lons, lat, lon)
        else:
            center_wind = float(wind)

        if not np.isfinite(center_msl):
            center_msl = _safe_min(msl_crop, np.nan)
        if not np.isfinite(center_wind):
            center_wind = _safe_max(wind_crop, np.nan)

        grid_min_msl = _safe_min(msl_crop, center_msl)
        grid_max_wind = _safe_max(wind_crop, center_wind)

        if np.isfinite(center_msl) and np.isfinite(grid_min_msl):
            min_msl = float(min(center_msl, grid_min_msl))
        else:
            min_msl = float(grid_min_msl)

        if np.isfinite(center_wind) and np.isfinite(grid_max_wind):
            max_wind = float(max(center_wind, grid_max_wind))
        else:
            max_wind = float(grid_max_wind)

        # Backfill the initial catalogue row so the first line is on-grid and has intensity.
        if np.isnan(self.tracked_msls[0]):
            self.tracked_msls[0] = min_msl
        if np.isnan(self.tracked_winds[0]):
            self.tracked_winds[0] = max_wind

        self.tracked_msls.append(min_msl)
        self.tracked_winds.append(max_wind)

        pressure_drop_hpa = self._compute_pressure_drop_hpa(msl, lats, lons, lat, lon, min_msl)
        if pressure_drop_hpa is not None:
            self.peak_pressure_drop_hpa = max(self.peak_pressure_drop_hpa, pressure_drop_hpa)

        if np.isfinite(max_wind):
            self.peak_wind = max(self.peak_wind, max_wind)

        self._check_dissipation(time, snap, pressure_drop_hpa, max_wind)
        self._check_stationary_quit(time, pressure_drop_hpa, max_wind)


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

        life_len = len(self.tracked_times)
        # Do not declare dissipation in the first few steps; early fields can be noisy.
        if life_len < 4:
            return

        if (not snap) and self.fails >= self._max_consecutive_fails and self.last_success_time is not None:
            duration = time - self.last_success_time
            if duration >= timedelta(hours=self._max_fail_hours):
                self._mark_dissipated(time, f"连续追踪失败{self._max_fail_hours}小时")
                return

        # Require a minimal number of successful samples before evaluating structure-based dissipation.
        if life_len < 10:
            return

        if pressure_drop_hpa is not None:
            if pressure_drop_hpa < self._min_drop_hpa:
                self._mark_dissipated(time, f"中心-外围压差低于{self._min_drop_hpa:.1f} hPa")
                return
        elif np.isfinite(max_wind) and self.peak_wind > 0:
            if max_wind < self._min_drop_fraction * self.peak_wind:
                self._mark_dissipated(time, "结构强度降至峰值的极小比例 (10米风)")

    def _mark_dissipated(self, time: datetime, reason: str) -> None:
        if self.dissipated:
            return
        self.dissipated = True
        self.dissipated_time = time
        self.dissipation_reason = reason
        logger.info("Cyclone dissipated at %s: %s", time, reason)
        print(f"[TRACKER] Dissipated at {time} because: {reason}")

    def _check_stationary_quit(
        self,
        time: datetime,
        pressure_drop_hpa: Optional[float],
        max_wind: float,
    ) -> None:
        if self.dissipated:
            return
        if len(self.tracked_times) < self._max_stationary_steps + 2:
            return
        if self._stationary_steps < self._max_stationary_steps:
            return

        ratio = self._current_intensity_ratio(pressure_drop_hpa, max_wind)
        effective_ratio = ratio if ratio is not None else 0.0
        if effective_ratio >= self._stationary_intensity_fraction:
            # System remains strong relative to its peak — treat as meandering instead of dissipated.
            self._stationary_steps = max(1, self._max_stationary_steps // 2)
            logger.debug(
                "Stationary but intensity still at %.0f%% of peak, continuing track",
                effective_ratio * 100.0,
            )
            return

        ratio_desc = "未知"
        if ratio is not None:
            ratio_desc = f"{max(min(ratio, 1.0), 0.0) * 100:.0f}%"

        self._mark_dissipated(
            time,
            f"连续{self._stationary_steps}次几乎未移动且强度降至峰值的{ratio_desc}",
        )

    def _current_intensity_ratio(
        self,
        pressure_drop_hpa: Optional[float],
        max_wind: float,
    ) -> Optional[float]:
        """Return the current/peak intensity ratio using drop or wind metrics."""
        ratios: list[float] = []
        if pressure_drop_hpa is not None and self.peak_pressure_drop_hpa > 0:
            ratios.append(max(pressure_drop_hpa, 0.0) / self.peak_pressure_drop_hpa)
        if np.isfinite(max_wind) and self.peak_wind > 0:
            ratios.append(max_wind / self.peak_wind)
        if not ratios:
            return None
        return float(np.clip(max(ratios), 0.0, 2.0))

__all__ = ["Tracker"]
