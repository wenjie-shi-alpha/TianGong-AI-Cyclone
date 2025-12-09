"""
Robust Tracker implementation designed for East Pacific / Weak Cyclones.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .batching import _SimpleBatch
from .exceptions import NoEyeException
from .geo import bilinear_interpolate, get_box, get_closest_min, extrapolate, snap_to_grid

logger = logging.getLogger(__name__)

_RELAXED = os.getenv("RELAXED_TRACKING", "0").lower() in {"1", "true", "yes"}

class RobustTracker:
    """
    A robust tropical cyclone tracker designed for the Eastern Pacific.
    
    Key differences from standard Tracker:
    1.  **Dynamic Search Radius**: Starts with a small radius (2.0 deg) to avoid drifting to nearby systems.
        Only expands if no center is found, but strictly checks structure.
    2.  **Structure Check (Closed Low)**: Verifies that a candidate minimum is actually a "low" 
        by comparing it to the surrounding environment. This prevents tracking flat fields or noise.
    3.  **Relaxed Dissipation**: Does not penalize for lack of "warm core" (which is often weak in EP).
        Allows for more consecutive "lost" steps before giving up, to handle temporary disorganization.
    """

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
        self.tracked_lons: List[float] = [(init_lon + 360.0) % 360.0]
        self.tracked_msls: List[float] = [init_msl if init_msl else np.nan]
        self.tracked_winds: List[float] = [init_wind if init_wind else np.nan]
        self._init_snapped: bool = False
        
        self.fails: int = 0
        self.dissipated: bool = False
        self.dissipated_time: Optional[datetime] = None
        self.dissipation_reason: Optional[str] = None
        
        # Configuration tuned for weak systems; relaxed mode widens limits further.
        # Search radii: prioritize local (2.0), then expand slightly.
        self.search_radii = [2.0, 3.0, 4.0]

        # Gradient threshold: center must be lower than surroundings. In relaxed
        # mode we halve the requirement to avoid rejecting shallow minima.
        self.gradient_threshold = 50.0 if not _RELAXED else 25.0

        # Max consecutive fails before declaring dissipation. Relaxed mode allows
        # more missed detections to keep tracks alive across reorganisations.
        self.max_consecutive_fails = 4 if not _RELAXED else 8

    def results(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time": self.tracked_times,
            "lat": self.tracked_lats,
            "lon": self.tracked_lons,
            "msl": self.tracked_msls,
            "wind": self.tracked_winds,
        })

    def step(self, batch: _SimpleBatch) -> None:
        if self.dissipated:
            return

        # Extract data
        msl = batch.surf_vars["msl"][0, 0]
        u10 = batch.surf_vars["10u"][0, 0]
        v10 = batch.surf_vars["10v"][0, 0]
        wind = np.sqrt(u10**2 + v10**2)
        lats = np.array(batch.metadata.lat)
        lons = np.array(batch.metadata.lon)
        time = batch.metadata.time[0]

        if not self._init_snapped:
            snap_lat, snap_lon = snap_to_grid(self.tracked_lats[0], self.tracked_lons[0], lats, lons)
            self.tracked_lats[0] = snap_lat
            self.tracked_lons[0] = snap_lon
            self._init_snapped = True
        
        # 1. Extrapolate next position
        guess_lat, guess_lon = extrapolate(self.tracked_lats, self.tracked_lons)
        guess_lat = max(min(guess_lat, 90), -90)
        guess_lon = guess_lon % 360

        # 2. Search for center
        found_lat, found_lon = None, None
        
        for r in self.search_radii:
            try:
                cand_lat, cand_lon = get_closest_min(
                    msl, lats, lons, guess_lat, guess_lon, delta_lat=r, delta_lon=r
                )

                # In relaxed mode, accept the first viable minimum without a strict
                # structure check to keep weak systems alive.
                if _RELAXED or self._check_structure(msl, lats, lons, cand_lat, cand_lon):
                    found_lat, found_lon = cand_lat, cand_lon
                    break
            except NoEyeException:
                continue

        # 3. Update State
        if found_lat is not None:
            self.fails = 0
            lat, lon = snap_to_grid(found_lat, found_lon, lats, lons)
        else:
            self.fails += 1
            if self.fails >= self.max_consecutive_fails:
                self._mark_dissipated(time, f"Lost track for {self.fails} steps")
                return

            # Hold position near the last valid fix to avoid runaway drift and still sample fields.
            lat, lon = snap_to_grid(self.tracked_lats[-1], self.tracked_lons[-1], lats, lons)

        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)

        _, _, msl_box = get_box(msl, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)
        _, _, wind_box = get_box(wind, lats, lons, lat - 1.5, lat + 1.5, lon - 1.5, lon + 1.5)

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
            center_msl = _safe_min(msl_box, np.nan)
        if not np.isfinite(center_wind):
            center_wind = _safe_max(wind_box, np.nan)

        grid_min_msl = _safe_min(msl_box, center_msl)
        grid_max_wind = _safe_max(wind_box, center_wind)

        if np.isfinite(center_msl) and np.isfinite(grid_min_msl):
            min_msl = float(min(center_msl, grid_min_msl))
        else:
            min_msl = float(grid_min_msl)

        if np.isfinite(center_wind) and np.isfinite(grid_max_wind):
            max_wind = float(max(center_wind, grid_max_wind))
        else:
            max_wind = float(grid_max_wind)

        if np.isnan(self.tracked_msls[0]):
            self.tracked_msls[0] = min_msl
        if np.isnan(self.tracked_winds[0]):
            self.tracked_winds[0] = max_wind

        self.tracked_msls.append(min_msl)
        self.tracked_winds.append(max_wind)

    def _check_structure(self, msl, lats, lons, lat, lon):
        """
        Check if the found minimum is a 'closed low' using TempestExtremes-like criteria.
        Criteria: The minimum value on a circle (or box perimeter) of radius R must be 
        greater than the center value by at least 'threshold'.
        """
        # TempestExtremes typically uses 4 degrees (great circle distance)
        # Here we approximate with a box/annulus for efficiency on the grid
        radius = 4.0
        
        # Get a box that covers the radius
        lats_box, lons_box, box = get_box(msl, lats, lons, lat-radius-1, lat+radius+1, lon-radius-1, lon+radius+1)
        
        if box.size == 0: return False
        
        # Calculate distances from center for all points in box
        # We need 2D arrays of lat/lon
        lat_grid, lon_grid = np.meshgrid(lats_box, lons_box, indexing='ij')
        
        # Simple Euclidean distance approximation for speed (deg)
        # For more accuracy near poles, use haversine, but for tropics this is fine
        dists = np.sqrt((lat_grid - lat)**2 + (lon_grid - lon)**2)
        
        # Define the "perimeter" as an annulus between R-0.5 and R+0.5
        # Or just check everything > R. 
        # TempestExtremes checks "closed contour within distance d".
        # A robust check: min(Perimeter) - Center > Threshold
        
        mask_perimeter = (dists >= radius - 0.5) & (dists <= radius + 0.5)
        
        if not np.any(mask_perimeter):
            # If grid is too coarse or box too small (shouldn't happen with get_box margin)
            return False
            
        perimeter_vals = box[mask_perimeter]
        min_perimeter = np.min(perimeter_vals)
        
        # Center value (from the tracker's candidate)
        # We re-extract it from the box to be sure, or pass it in.
        # Let's trust the candidate lat/lon is the minimum in its local neighborhood.
        # But to be safe, let's take the min within a small radius (0.5) of the candidate
        mask_center = dists <= 0.5
        if np.any(mask_center):
            center_val = np.min(box[mask_center])
        else:
            # Fallback if grid is very coarse
            center_val = float(np.min(box))

        # Check gradient relative to configured threshold (Pa)
        return (min_perimeter - center_val) > self.gradient_threshold

    def _mark_dissipated(self, time, reason):
        self.dissipated = True
        self.dissipated_time = time
        self.dissipation_reason = reason
