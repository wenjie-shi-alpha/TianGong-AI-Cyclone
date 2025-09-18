"""Core tracking logic operating on successive meteorological batches."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from .batching import _SimpleBatch
from .exceptions import NoEyeException
from .geo import get_box, get_closest_min, extrapolate

logger = logging.getLogger(__name__)


class Tracker:
    """Simple tropical cyclone tracker based on surface pressure minima."""

    def __init__(self, init_lat: float, init_lon: float, init_time: datetime) -> None:
        self.tracked_times: List[datetime] = [init_time]
        self.tracked_lats: List[float] = [init_lat]
        self.tracked_lons: List[float] = [init_lon]
        self.tracked_msls: List[float] = [np.nan]
        self.tracked_winds: List[float] = [np.nan]
        self.fails: int = 0

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

        if not snap:
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


__all__ = ["Tracker"]
