"""
Robust Tracker implementation designed for East Pacific / Weak Cyclones.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .batching import _SimpleBatch
from .exceptions import NoEyeException
from .geo import get_box, get_closest_min, extrapolate

logger = logging.getLogger(__name__)

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
        self.tracked_lons: List[float] = [init_lon]
        self.tracked_msls: List[float] = [init_msl if init_msl else np.nan]
        self.tracked_winds: List[float] = [init_wind if init_wind else np.nan]
        
        self.fails: int = 0
        self.dissipated: bool = False
        self.dissipated_time: Optional[datetime] = None
        self.dissipation_reason: Optional[str] = None
        
        # Configuration
        # Search radii: Prioritize local (2.0), then expand slightly. 
        # Avoid 5.0 deg which causes large jumps.
        self.search_radii = [2.0, 3.0, 4.0] 
        
        # Gradient threshold: Center must be at least this much lower than surroundings (Pascals)
        # 50 Pa = 0.5 hPa. Weak but non-zero.
        self.gradient_threshold = 50.0 
        
        # Max consecutive fails before declaring dissipation
        # 4 steps * 6 hours = 24 hours
        self.max_consecutive_fails = 4

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
        
        # 1. Extrapolate next position
        guess_lat, guess_lon = extrapolate(self.tracked_lats, self.tracked_lons)
        guess_lat = max(min(guess_lat, 90), -90)
        guess_lon = guess_lon % 360

        # 2. Search for center
        found_lat, found_lon = None, None
        
        for r in self.search_radii:
            try:
                # Use existing helper to find local min
                cand_lat, cand_lon = get_closest_min(
                    msl, lats, lons, guess_lat, guess_lon, delta_lat=r, delta_lon=r
                )
                
                # Check if this minimum has structure (is a closed low)
                if self._check_structure(msl, lats, lons, cand_lat, cand_lon):
                    found_lat, found_lon = cand_lat, cand_lon
                    break # Found a good one, stop searching
            except NoEyeException:
                continue

        # 3. Update State
        if found_lat is not None:
            self.fails = 0
            self.tracked_times.append(time)
            self.tracked_lats.append(found_lat)
            self.tracked_lons.append(found_lon)
            
            # Get values at center
            _, _, msl_box = get_box(msl, lats, lons, found_lat-1.5, found_lat+1.5, found_lon-1.5, found_lon+1.5)
            _, _, wind_box = get_box(wind, lats, lons, found_lat-1.5, found_lat+1.5, found_lon-1.5, found_lon+1.5)
            self.tracked_msls.append(float(np.nanmin(msl_box)) if msl_box.size > 0 else np.nan)
            self.tracked_winds.append(float(np.nanmax(wind_box)) if wind_box.size > 0 else np.nan)
            
        else:
            self.fails += 1
            if self.fails > self.max_consecutive_fails:
                self._mark_dissipated(time, f"Lost track for {self.fails} steps")
            else:
                # Extrapolate blindly to keep the track alive
                self.tracked_times.append(time)
                self.tracked_lats.append(guess_lat)
                self.tracked_lons.append(guess_lon)
                self.tracked_msls.append(np.nan)
                self.tracked_winds.append(np.nan)

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

        # Check gradient
        # Threshold: 2 hPa = 200 Pa
        return (min_perimeter - center_val) > 200.0

    def _mark_dissipated(self, time, reason):
        self.dissipated = True
        self.dissipated_time = time
        self.dissipation_reason = reason
