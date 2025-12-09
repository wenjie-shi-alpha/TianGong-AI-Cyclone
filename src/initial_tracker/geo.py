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

    raw_box = np.asarray(box, dtype=float)
    filtered_box = gaussian_filter(raw_box, sigma=1)
    local_minima = minimum_filter(filtered_box, size=(minimum_cap_size, minimum_cap_size)) == filtered_box

    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0

    if local_minima.sum() == 0:
        raise NoEyeException()

    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats_box[list(lat_inds)], lons_box[list(lon_inds)], lat, lon)
    idx = int(np.argmin(dists))
    best_lat_idx = int(lat_inds[idx])
    best_lon_idx = int(lon_inds[idx])
    return _refine_quadratic_minimum(
        raw_box,
        lats_box,
        lons_box,
        best_lat_idx,
        best_lon_idx,
    )


def extrapolate(lats: list[float], lons: list[float]) -> Tuple[float, float]:
    """Linearly extrapolate the next position using up to the last eight points.

    Longitudes are unwrapped to avoid spurious jumps across the 0/360 boundary.
    """
    if len(lats) == 0:
        raise ValueError("Cannot extrapolate from empty lists.")
    if len(lats) == 1:
        return lats[0], lons[0]

    lats_recent = np.asarray(lats[-8:], dtype=float)
    lons_recent = np.asarray(lons[-8:], dtype=float)

    # Unwrap longitudes in radians, then fit in degrees to keep continuity
    lons_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lons_recent)))

    n = len(lats_recent)
    x = np.arange(n, dtype=float)
    lat_fit = np.polyfit(x, lats_recent, 1)
    lon_fit = np.polyfit(x, lons_unwrapped, 1)

    lat_pred = np.polyval(lat_fit, n)
    lon_pred = np.polyval(lon_fit, n)

    # Wrap longitude back to [0, 360)
    lon_pred = (lon_pred + 360.0) % 360.0
    return float(lat_pred), float(lon_pred)


def _refine_quadratic_minimum(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_idx: int,
    lon_idx: int,
) -> Tuple[float, float]:
    """Use a local quadratic fit to estimate a sub-grid minimum location."""
    if field.ndim != 2:
        return float(lats[lat_idx]), float((lons[lon_idx] + 360.0) % 360.0)

    lat_count, lon_count = field.shape
    if lat_count < 3 or lon_count < 3:
        return float(lats[lat_idx]), float((lons[lon_idx] + 360.0) % 360.0)

    lat_slice = slice(max(lat_idx - 1, 0), min(lat_idx + 2, lat_count))
    lon_slice = slice(max(lon_idx - 1, 0), min(lon_idx + 2, lon_count))
    if (lat_slice.stop - lat_slice.start) < 3 or (lon_slice.stop - lon_slice.start) < 3:
        return float(lats[lat_idx]), float((lons[lon_idx] + 360.0) % 360.0)

    patch = field[lat_slice, lon_slice]
    lat_vals = lats[lat_slice]
    lon_vals = lons[lon_slice]

    center_lat = float(lats[lat_idx])
    center_lon = float((lons[lon_idx] + 360.0) % 360.0)

    lat_offsets = lat_vals - center_lat
    lon_offsets = _wrap_longitude_diffs(lon_vals, center_lon)

    lat_grid, lon_grid = np.meshgrid(lat_offsets, lon_offsets, indexing="ij")
    A = np.column_stack(
        [
            np.ones(patch.size, dtype=float),
            lon_grid.ravel(),
            lat_grid.ravel(),
            (lon_grid * lat_grid).ravel(),
            (lon_grid ** 2).ravel(),
            (lat_grid ** 2).ravel(),
        ]
    )

    try:
        coeffs, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    except np.linalg.LinAlgError:
        return center_lat, center_lon

    a0, ax, ay, axy, axx, ayy = coeffs
    H = np.array([[2.0 * axx, axy], [axy, 2.0 * ayy]], dtype=float)
    b = np.array([-ax, -ay], dtype=float)

    try:
        delta_lon, delta_lat = np.linalg.solve(H, b)
    except np.linalg.LinAlgError:
        return center_lat, center_lon

    det = np.linalg.det(H)
    if det <= 0 or H[0, 0] <= 0 or H[1, 1] <= 0:
        return center_lat, center_lon

    lon_limit = np.max(np.abs(lon_offsets)) + 1e-6
    lat_limit = np.max(np.abs(lat_offsets)) + 1e-6
    if abs(delta_lon) > lon_limit or abs(delta_lat) > lat_limit:
        return center_lat, center_lon

    refined_lat = float(center_lat + delta_lat)
    refined_lon = float((center_lon + delta_lon + 720.0) % 360.0)
    return refined_lat, refined_lon


def _wrap_longitude_diffs(values: np.ndarray, center_lon: float) -> np.ndarray:
    """Return longitudinal offsets relative to the provided center."""
    vals = np.asarray(values, dtype=float)
    center = float((center_lon + 360.0) % 360.0)
    offsets = (vals + 360.0) % 360.0 - center
    offsets = (offsets + 540.0) % 360.0 - 180.0
    return offsets


def _wrap_longitude_to_domain(lon: float, lon_array: np.ndarray) -> float:
    """Map longitude into the same domain as the model grid."""
    lon_arr = np.asarray(lon_array, dtype=float)
    lon_min = float(np.nanmin(lon_arr))
    lon_max = float(np.nanmax(lon_arr))
    span = lon_max - lon_min
    if span >= 350.0:
        return ((lon - lon_min) % 360.0) + lon_min
    return lon


def _fractional_index(coords: np.ndarray, value: float) -> float:
    """Return the fractional index position of value along a monotonic axis."""
    arr = np.asarray(coords, dtype=float)
    n = arr.size
    if n <= 1:
        return 0.0

    if arr[0] < arr[-1]:
        idx = np.interp(value, arr, np.arange(n, dtype=float))
    else:
        idx = np.interp(value, arr[::-1], np.arange(n - 1, -1, -1, dtype=float))
    return float(np.clip(idx, 0.0, n - 1.0))


def bilinear_interpolate(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    """Sample a 2D field at an arbitrary coordinate using bilinear interpolation."""
    data = np.asarray(field, dtype=float)
    if data.ndim != 2:
        raise ValueError("bilinear_interpolate expects a 2-D field")
    lat_idx = _fractional_index(lats, lat)
    lon_value = _wrap_longitude_to_domain(lon, lons)
    lon_idx = _fractional_index(lons, lon_value)

    i0 = int(np.floor(lat_idx))
    j0 = int(np.floor(lon_idx))
    i1 = min(i0 + 1, data.shape[0] - 1)
    j1 = min(j0 + 1, data.shape[1] - 1)

    ty = lat_idx - i0
    tx = lon_idx - j0

    v00 = data[i0, j0]
    v10 = data[i1, j0]
    v01 = data[i0, j1]
    v11 = data[i1, j1]

    top = (1.0 - tx) * v00 + tx * v01
    bottom = (1.0 - tx) * v10 + tx * v11
    return float((1.0 - ty) * top + ty * bottom)


def snap_to_grid(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float]:
    """Clamp a lat/lon pair inside the domain without forcing it onto discrete grid nodes."""
    lat_arr = np.asarray(lats, dtype=float)
    lat_min = float(np.nanmin(lat_arr))
    lat_max = float(np.nanmax(lat_arr))
    lat_val = float(np.clip(lat, lat_min, lat_max))

    lon_val = float((lon + 360.0) % 360.0)
    return lat_val, lon_val


__all__ = ["get_box", "havdist", "get_closest_min", "extrapolate", "snap_to_grid", "bilinear_interpolate"]
