"""Geographical helpers for cyclone tracking."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

from .exceptions import NoEyeException


def get_box(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a sub-domain around the given latitude/longitude window."""
    lat_mask = (lat_min <= lats) & (lats <= lat_max)
    box = variable[..., lat_mask, :]
    lats_sel = lats[lat_mask]

    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        lon_mask = (lon_min <= lons) & (lons <= lon_max)
        box = box[..., lon_mask]
        lons_sel = lons[lon_mask]
    else:
        lon_mask1 = lon_min <= lons
        lon_mask2 = lons <= lon_max
        box = np.concatenate((box[..., lon_mask1], box[..., lon_mask2]), axis=-1)
        lons_sel = np.concatenate((lons[lon_mask1], lons[lon_mask2]))

    return lats_sel, lons_sel, box


def havdist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the haversine distance between two coordinates in kilometres."""
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1, lon2 = np.deg2rad(lon1), np.deg2rad(lon2)
    rad_earth_km = 6371
    inner = 1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))


def get_closest_min(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta_lat: float = 5,
    delta_lon: float = 5,
    minimum_cap_size: int = 8,
) -> Tuple[float, float]:
    """Locate the closest local minimum around the given coordinate."""
    lats_box, lons_box, box = get_box(
        variable,
        lats,
        lons,
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )

    box = gaussian_filter(box, sigma=1)
    local_minima = minimum_filter(box, size=(minimum_cap_size, minimum_cap_size)) == box

    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0

    if local_minima.sum() == 0:
        raise NoEyeException()

    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats_box[list(lat_inds)], lons_box[list(lon_inds)], lat, lon)
    idx = int(np.argmin(dists))
    return float(lats_box[lat_inds[idx]]), float(lons_box[lon_inds[idx]])


def extrapolate(lats: list[float], lons: list[float]) -> Tuple[float, float]:
    """Linearly extrapolate the next position using up to the last eight points."""
    if len(lats) == 0:
        raise ValueError("Cannot extrapolate from empty lists.")
    if len(lats) == 1:
        return lats[0], lons[0]
    lats_recent = lats[-8:]
    lons_recent = lons[-8:]
    n = len(lats_recent)
    fit = np.polyfit(np.arange(n), np.stack((lats_recent, lons_recent), axis=-1), 1)
    lat_pred, lon_pred = np.polyval(fit, n)
    return float(lat_pred), float(lon_pred)


def compute_relative_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta: float = 2.0,
) -> float:
    """
    Compute relative vorticity at a specific location.
    
    Args:
        u: U wind component (2D array)
        v: V wind component (2D array)
        lats: Latitude array
        lons: Longitude array
        lat: Center latitude
        lon: Center longitude
        delta: Box size in degrees
        
    Returns:
        Relative vorticity in s^-1
    """
    lats_box, lons_box, u_box = get_box(u, lats, lons, lat - delta, lat + delta, lon - delta, lon + delta)
    _, _, v_box = get_box(v, lats, lons, lat - delta, lat + delta, lon - delta, lon + delta)
    
    if u_box.size == 0 or v_box.size == 0:
        return 0.0
    
    # Convert to meters
    lat_spacing = np.mean(np.diff(lats_box))
    lon_spacing = np.mean(np.diff(lons_box))
    
    dy = lat_spacing * 111000  # meters per degree latitude
    dx = lon_spacing * 111000 * np.cos(np.deg2rad(lat))  # meters per degree longitude
    
    # Compute vorticity: dv/dx - du/dy
    dv_dx = np.gradient(v_box, axis=1) / dx
    du_dy = np.gradient(u_box, axis=0) / dy
    
    vorticity = dv_dx - du_dy
    
    # Return average vorticity in the box
    return float(np.nanmean(vorticity))


def check_warm_core(
    t_upper: np.ndarray,
    t_lower: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta: float = 2.0,
) -> Tuple[bool, float]:
    """
    Check for warm core structure by comparing temperature anomalies.
    
    Args:
        t_upper: Temperature at upper level (e.g., 200 hPa) in Kelvin
        t_lower: Temperature at lower level (e.g., 850 hPa) in Kelvin
        lats: Latitude array
        lons: Longitude array
        lat: Center latitude
        lon: Center longitude
        delta: Box size in degrees for analysis
        
    Returns:
        Tuple of (has_warm_core, core_strength)
        - has_warm_core: True if warm core is detected
        - core_strength: Temperature anomaly in the core relative to surroundings (K)
    """
    # Get temperature in the center region
    lats_center, lons_center, t_upper_center = get_box(
        t_upper, lats, lons, lat - delta/2, lat + delta/2, lon - delta/2, lon + delta/2
    )
    _, _, t_lower_center = get_box(
        t_lower, lats, lons, lat - delta/2, lat + delta/2, lon - delta/2, lon + delta/2
    )
    
    if t_upper_center.size == 0 or t_lower_center.size == 0:
        return False, 0.0
    
    # Get temperature in the surrounding annulus
    lats_outer, lons_outer, t_upper_outer = get_box(
        t_upper, lats, lons, lat - delta*1.5, lat + delta*1.5, lon - delta*1.5, lon + delta*1.5
    )
    _, _, t_lower_outer = get_box(
        t_lower, lats, lons, lat - delta*1.5, lat + delta*1.5, lon - delta*1.5, lon + delta*1.5
    )
    
    if t_upper_outer.size == 0 or t_lower_outer.size == 0:
        return False, 0.0
    
    # Compute temperature anomaly (200-850 hPa difference)
    # In tropical cyclones, upper levels are warmer relative to surroundings
    center_anomaly = np.nanmean(t_upper_center - t_lower_center)
    outer_anomaly = np.nanmean(t_upper_outer - t_lower_outer)
    
    # Warm core: center should have higher upper-level temperatures
    # (smaller lapse rate or temperature inversion)
    core_strength = float(center_anomaly - outer_anomaly)
    
    # Threshold: at least 1K warmer in the core
    has_warm_core = core_strength > 1.0
    
    return has_warm_core, core_strength


__all__ = ["get_box", "havdist", "get_closest_min", "extrapolate", "compute_relative_vorticity", "check_warm_core"]
