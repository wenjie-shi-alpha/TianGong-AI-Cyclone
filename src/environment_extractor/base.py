"""Common initialization and utility helpers for the TC environment extractor."""

from __future__ import annotations

import math
from datetime import datetime  # noqa: F401  # retained for compatibility with older modules
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from .deps import (
    approximate_polygon,
    center_of_mass,
    find_contours,
    find_objects,
    label,
)
from .shape_analysis import WeatherSystemShapeAnalyzer


class BaseExtractor:
    """
    提供热带气旋环境场提取所需的通用初始化与工具函数。
    """

    def __init__(self, forecast_data_path, tc_tracks_path, enable_detailed_shape_analysis=True):
        """
        初始化环境场提取器
        
        Args:
            forecast_data_path: 预报数据文件路径
            tc_tracks_path: 台风路径文件路径
            enable_detailed_shape_analysis: 是否启用详细形状分析
                True (默认): 完整分析，包含面积、周长、分形维数等（与原实现一致）
                False: 快速模式，跳过昂贵计算，性能提升约 60-80%
        """
        # 保存配置
        self.enable_detailed_shape_analysis = enable_detailed_shape_analysis
        
        # 保持原有初始化逻辑不变；为避免大文件一次性载入内存，优先按需分块读取
        open_kwargs: dict[str, object] = {"cache": False}
        try:
            import dask  # noqa: F401

            open_kwargs["chunks"] = "auto"
        except ImportError:
            # dask 非必选，缺失时回退为常规打开方式
            pass

        self.ds = xr.open_dataset(forecast_data_path, **open_kwargs)
        try:
            p = Path(forecast_data_path)
            self.nc_filename = p.name
            self.nc_stem = p.stem
        except Exception:
            self.nc_filename = "data"
            self.nc_stem = "data"

        self.lat = self.ds.latitude.values if "latitude" in self.ds.coords else self.ds.lat.values
        self.lon = self.ds.longitude.values if "longitude" in self.ds.coords else self.ds.lon.values
        self.lon_180 = np.where(self.lon > 180, self.lon - 360, self.lon)
        self.lat_spacing = np.abs(np.diff(self.lat).mean())
        self.lon_spacing = np.abs(np.diff(self.lon).mean())

        self._coslat = np.cos(np.deg2rad(self.lat))
        self._coslat_safe = np.where(np.abs(self._coslat) < 1e-6, np.nan, self._coslat)

        def _raw_gradients(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # 直接计算梯度，避免跨时间步缓存大量大数组导致内存膨胀
            gy = np.gradient(arr, axis=0)
            gx = np.gradient(arr, axis=1)
            return gy, gx

        self._raw_gradients: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = _raw_gradients

        def _loc_idx(lat_val: float, lon_val: float) -> tuple[int, int]:
            return (np.abs(self.lat - lat_val).argmin(), np.abs(self.lon - lon_val).argmin())

        self._loc_idx: Callable[[float, float], tuple[int, int]] = _loc_idx

        # 🚀 优化：传递形状分析配置
        self.shape_analyzer = WeatherSystemShapeAnalyzer(
            self.lat, self.lon, 
            enable_detailed_analysis=enable_detailed_shape_analysis
        )

        self.tc_tracks = pd.read_csv(tc_tracks_path)
        has_time_column = "time" in self.tc_tracks.columns
        if has_time_column:
            self.tc_tracks["time"] = pd.to_datetime(
                self.tc_tracks["time"], errors="coerce", utc=True
            )
        else:
            self.tc_tracks["time"] = pd.NaT

        ds_times = (
            pd.to_datetime(self.ds.time.values, utc=True)
            if "time" in self.ds.coords
            else pd.DatetimeIndex([])
        )
        self._align_track_times_with_dataset(ds_times, has_time_column)

        print(f"📊 加载{len(self.tc_tracks)}个热带气旋路径点")
        print(
            f"🌍 区域范围: {self.lat.min():.1f}°-{self.lat.max():.1f}°N, "
            f"{self.lon.min():.1f}°-{self.lon.max():.1f}°E"
        )
        if enable_detailed_shape_analysis:
            print("🔍 增强形状分析功能已启用（完整模式）")
        else:
            print("⚡ 形状分析快速模式已启用（跳过昂贵计算，性能提升 60-80%）")

    # --- 资源管理 ---

    def close(self) -> None:
        dataset = getattr(self, "ds", None)
        if dataset is not None:
            try:
                dataset.close()
            except Exception:
                pass
            finally:
                self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.close()

    # --- 几何与掩膜函数 ---

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_rad = np.deg2rad(lat1)
        lat2_rad = np.deg2rad(lat2)
        lon1_rad = np.deg2rad(lon1)
        lon2_rad = np.deg2rad(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        return R * c

    def _create_circular_mask_haversine(self, center_lat, center_lon, radius_km):
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        lon_normalized = lon_grid.copy()
        lon_diff = lon_grid - center_lon
        lon_normalized = np.where(lon_diff > 180, lon_grid - 360, lon_grid)
        lon_normalized = np.where(lon_diff < -180, lon_grid + 360, lon_normalized)
        distances = self._haversine_distance(lat_grid, lon_normalized, center_lat, center_lon)
        return distances <= radius_km

    def _normalize_longitude(self, lon_array, center_lon):
        lon_normalized = lon_array.copy()
        lon_diff = lon_array - center_lon
        lon_normalized = np.where(lon_diff > 180, lon_array - 360, lon_array)
        lon_normalized = np.where(lon_diff < -180, lon_array + 360, lon_normalized)
        return lon_normalized

    def _create_region_mask(self, center_lat, center_lon, radius_deg):
        lat_mask = (self.lat >= center_lat - radius_deg) & (self.lat <= center_lat + radius_deg)
        lon_mask = (self.lon >= center_lon - radius_deg) & (self.lon <= center_lon + radius_deg)
        return np.outer(lat_mask, lon_mask)

    def _align_track_times_with_dataset(self, ds_times, has_time_column: bool) -> None:
        """Align track rows to the dataset time axis and ensure ``time_idx`` is usable."""

        if has_time_column and len(ds_times) > 0:
            ds_values = ds_times.view("int64")
            if len(ds_values) > 1:
                diffs = np.diff(ds_values)
                approx_step = pd.to_timedelta(np.median(np.abs(diffs)), unit="ns")
            else:
                approx_step = pd.Timedelta(hours=6)
            tolerance = approx_step / 2 + pd.Timedelta(minutes=5)
            tolerance = max(tolerance, pd.Timedelta(minutes=15))

            def nearest_idx(ts: pd.Timestamp) -> int | None:
                if pd.isna(ts):
                    return None
                deltas = np.abs(ds_values - ts.value)
                idx = int(np.argmin(deltas))
                delta_ns = abs(ds_values[idx] - ts.value)
                if pd.to_timedelta(delta_ns, unit="ns") > tolerance:
                    return None
                return idx

            idx_series = self.tc_tracks["time"].apply(nearest_idx)
            invalid = idx_series.isna()
            if invalid.any():
                print(
                    f"⚠️ 有 {invalid.sum()} 个轨迹时间点未能在 NC 时间轴 ±{tolerance} 内匹配，已忽略"
                )
            self.tc_tracks = self.tc_tracks.loc[~invalid].copy()
            if self.tc_tracks.empty:
                raise ValueError("轨迹文件中的时间点无法与NC时间轴匹配")
            self.tc_tracks["time_idx"] = idx_series.loc[~invalid].astype(int).to_numpy()
        elif "time_idx" in self.tc_tracks.columns:
            self.tc_tracks["time_idx"] = (
                pd.to_numeric(self.tc_tracks["time_idx"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            self.tc_tracks["time_idx"] = np.arange(len(self.tc_tracks))

    # --- 数据访存 ---

    def _get_data_at_level(self, var_name, level_hPa, time_idx):
        if var_name not in self.ds.data_vars:
            return None
        var_data = self.ds[var_name]
        level_dim = next(
            (dim for dim in ["level", "isobaricInhPa", "pressure"] if dim in var_data.dims),
            None,
        )
        if level_dim is None:
            return var_data.isel(time=time_idx).values if "time" in var_data.dims else var_data.values
        levels = self.ds[level_dim].values
        level_idx = np.abs(levels - level_hPa).argmin()
        return var_data.isel(time=time_idx, **{level_dim: level_idx}).values

    def _get_sst_field(self, time_idx):
        for var_name in ["sst", "ts"]:
            if var_name in self.ds.data_vars:
                sst_data = self.ds[var_name].isel(time=time_idx).values
                return sst_data - 273.15 if np.nanmean(sst_data) > 200 else sst_data

        for var_name in ["t2", "t2m"]:
            if var_name in self.ds.data_vars:
                t2_data = self.ds[var_name].isel(time=time_idx).values
                sst_approx = t2_data - 273.15 if np.nanmean(t2_data) > 200 else t2_data
                print(f"⚠️  使用{var_name}作为海表温度近似")
                return sst_approx

        return None

    # --- 形状与等值线辅助 ---

    def _get_contour_coords(self, data_field, level, center_lon, max_points=100):
        try:
            contours = find_contours(data_field, level)
            if not contours:
                return None
            main_contour = sorted(contours, key=len, reverse=True)[0]
            contour_lon = self.lon[main_contour[:, 1].astype(int)]
            contour_lat = self.lat[main_contour[:, 0].astype(int)]
            step = max(1, len(main_contour) // max_points)
            return [
                [round(lon, 2), round(lat, 2)]
                for lon, lat in zip(contour_lon[::step], contour_lat[::step])
            ]
        except Exception:
            return None

    def _get_contour_coords_local(self, data_field, level, lat_array, lon_array, center_lon, max_points=100):
        try:
            contours = find_contours(data_field, level)
            if not contours:
                return None
            main_contour = sorted(contours, key=len, reverse=True)[0]
            contour_indices_lat = np.clip(main_contour[:, 0].astype(int), 0, len(lat_array) - 1)
            contour_indices_lon = np.clip(main_contour[:, 1].astype(int), 0, len(lon_array) - 1)
            contour_lon = lon_array[contour_indices_lon]
            contour_lat = lat_array[contour_indices_lat]
            contour_lon_normalized = self._normalize_longitude(contour_lon, center_lon)
            step = max(1, len(main_contour) // max_points)
            coords = []
            for lon, lat in zip(contour_lon_normalized[::step], contour_lat[::step]):
                lon_val = lon + 360 if lon < 0 else lon
                coords.append([round(float(lon_val), 2), round(float(lat), 2)])
            return coords
        except Exception:
            return None

    def _get_enhanced_shape_info(self, data_field, threshold, system_type, center_lat, center_lon):
        """获取增强的形状信息（简化版，仅包含边界坐标）."""
        try:
            shape_analysis = self.shape_analyzer.analyze_system_shape(
                data_field, threshold, system_type, center_lat, center_lon
            )
            if shape_analysis:
                basic_info = {
                    "description": shape_analysis.get("description", ""),
                    "detailed_analysis": shape_analysis,
                }
                # 新的简化结构：直接包含边界坐标和多边形特征
                if "boundary_coordinates" in shape_analysis:
                    basic_info["coordinate_info"] = {
                        "main_contour_coords": shape_analysis.get("boundary_coordinates", []),
                        "polygon_features": shape_analysis.get("polygon_features", {}),
                    }
                return basic_info
        except Exception as exc:  # noqa: F841
            print(f"形状分析失败: {exc}")
        return None

    def _get_system_coordinates_local(
        self, data_field, threshold, system_type, center_lat, center_lon, max_points=20
    ):
        try:
            valid_mask = np.isfinite(data_field)
            if not np.any(valid_mask):
                return None
            if system_type == "high":
                mask = valid_mask & (data_field >= threshold)
            else:
                mask = valid_mask & (data_field <= threshold)
            if not np.any(mask):
                return None
            labeled_mask, num_features = label(mask)
            if num_features == 0:
                return None
            flat_labels = labeled_mask.ravel()
            counts = np.bincount(flat_labels)[1 : num_features + 1]
            if counts.size == 0:
                return None
            main_label = int(np.argmax(counts) + 1)
            main_region = labeled_mask == main_label
            contours = find_contours(main_region.astype(float), 0.5)
            if not contours:
                return None
            main_contour = max(contours, key=len)
            epsilon = max(len(main_contour) * 0.01, 1)
            simplified = approximate_polygon(main_contour, tolerance=epsilon)
            if len(simplified) > max_points:
                step = max(1, len(simplified) // max_points)
                simplified = simplified[::step]
            geo_coords = []
            for point in simplified:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                geo_coords.append([round(self.lon[lon_idx], 3), round(self.lat[lat_idx], 3)])
            if geo_coords:
                lons = [coord[0] for coord in geo_coords]
                lats = [coord[1] for coord in geo_coords]
                lons_normalized = []
                for lon in lons:
                    diff = lon - center_lon
                    if diff > 180:
                        lon -= 360
                    elif diff < -180:
                        lon += 360
                    lons_normalized.append(lon)
                lon_min = round(min(lons_normalized), 3)
                lon_max = round(max(lons_normalized), 3)
                lat_min = round(min(lats), 3)
                lat_max = round(max(lats), 3)
                if lon_min < 0:
                    lon_min += 360
                if lon_max < 0:
                    lon_max += 360
                span_lon = round(lon_max - lon_min, 3)
                span_lat = round(lat_max - lat_min, 3)
                return {
                    "vertices": geo_coords,
                    "total_points": len(geo_coords),
                    "lon_range": [lon_min, lon_max],
                    "lat_range": [lat_min, lat_max],
                    "span_deg": [span_lon, span_lat],
                }
            return None
        except Exception:
            return None

    def _get_system_coordinates(self, data_field, threshold, system_type, max_points=20):
        try:
            mask = data_field >= threshold if system_type == "high" else data_field <= threshold
            if not np.any(mask):
                return None
            labeled_mask, num_features = label(mask)
            if num_features == 0:
                return None
            flat_labels = labeled_mask.ravel()
            counts = np.bincount(flat_labels)[1 : num_features + 1]
            if counts.size == 0:
                return None
            main_label = int(np.argmax(counts) + 1)
            main_region = labeled_mask == main_label
            contours = find_contours(main_region.astype(float), 0.5)
            if not contours:
                return None
            main_contour = max(contours, key=len)
            epsilon = len(main_contour) * 0.01
            simplified = approximate_polygon(main_contour, tolerance=epsilon)
            if len(simplified) > max_points:
                step = len(simplified) // max_points
                simplified = simplified[::step]
            geo_coords = []
            for point in simplified:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                geo_coords.append([round(self.lon[lon_idx], 3), round(self.lat[lat_idx], 3)])
            if geo_coords:
                lons = [coord[0] for coord in geo_coords]
                lats = [coord[1] for coord in geo_coords]
                extent = {
                    "boundaries": [
                        round(min(lons), 3),
                        round(min(lats), 3),
                        round(max(lons), 3),
                        round(max(lats), 3),
                    ],
                    "center": [round(np.mean(lons), 3), round(np.mean(lats), 3)],
                    "span": [
                        round(max(lons) - min(lons), 3),
                        round(max(lats) - min(lats), 3),
                    ],
                }
                return {
                    "vertices": geo_coords,
                    "vertex_count": len(geo_coords),
                    "extent": extent,
                    "span_deg": [extent["span"][0], extent["span"][1]],
                }
            return None
        except Exception as exc:  # noqa: F841
            print(f"坐标提取失败: {exc}")
            return None

    # --- 描述生成与基础几何 ---

    def _generate_coordinate_description(self, coords_info, system_name="系统"):
        if not coords_info:
            return ""
        try:
            description_parts = []
            if "extent" in coords_info:
                extent = coords_info["extent"]
                boundaries = extent["boundaries"]
                description_parts.append(
                    f"{system_name}主体位于{boundaries[0]:.1f}°E-{boundaries[2]:.1f}°E，"
                    f"{boundaries[1]:.1f}°N-{boundaries[3]:.1f}°N"
                )
            if "vertices" in coords_info and coords_info.get("vertex_count", 0) > 0:
                vertex_count = coords_info["vertex_count"]
                description_parts.append(f"由{vertex_count}个关键顶点构成的多边形形状")
            if "span_deg" in coords_info:
                lon_span, lat_span = coords_info["span_deg"]
                lat_km = lat_span * 111
                center_lat = coords_info.get("extent", {}).get("center", [0, 30])[1]
                lon_km = lon_span * 111 * np.cos(np.radians(center_lat))
                description_parts.append(
                    f"纬向跨度约{lat_km:.0f}km，经向跨度约{lon_km:.0f}km"
                )
            return "，".join(description_parts) + "。" if description_parts else ""
        except Exception:
            return ""

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        dLon = math.radians(lon2 - lon1)
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        y = math.sin(dLon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dLon)
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
        return bearing, self._bearing_to_desc(bearing)[1]

    def _bearing_to_desc(self, bearing):
        dirs = [
            "北",
            "东北偏北",
            "东北",
            "东北偏东",
            "东",
            "东南偏东",
            "东南",
            "东南偏南",
            "南",
            "西南偏南",
            "西南",
            "西南偏西",
            "西",
            "西北偏西",
            "西北",
            "西北偏北",
        ]
        wind_dirs = [
            "偏北风",
            "东北偏北风",
            "东北风",
            "东北偏东风",
            "偏东风",
            "东南偏东风",
            "东南风",
            "东南偏南风",
            "偏南风",
            "西南偏南风",
            "西南风",
            "西南偏西风",
            "偏西风",
            "西北偏西风",
            "西北风",
            "西北偏北风",
        ]
        index = round(bearing / 22.5) % 16
        return wind_dirs[index], f"{dirs[index]}方向"

    def _get_vector_coords(self, lat, lon, u, v, scale=0.1):
        end_lat = lat + v * scale * 0.009
        end_lon = lon + u * scale * 0.009 / math.cos(math.radians(lat))
        return {
            "start": {"lat": round(lat, 2), "lon": round(lon, 2)},
            "end": {"lat": round(end_lat, 2), "lon": round(end_lon, 2)},
        }

    # --- 系统识别与强度 ---

    def _identify_pressure_system(self, data_field, tc_lat, tc_lon, system_type, threshold):
        if system_type == "high":
            mask = data_field > threshold
        else:
            mask = data_field < threshold
        if not np.any(mask):
            return None
        labeled_array, num_features = label(mask)
        if num_features == 0:
            return None
        objects_slices = find_objects(labeled_array)
        min_dist, closest_feature_idx = float("inf"), -1
        tc_lat_idx, tc_lon_idx = (
            np.abs(self.lat - tc_lat).argmin(),
            np.abs(self.lon - tc_lon).argmin(),
        )
        for i, slc in enumerate(objects_slices):
            center_y = (slc[0].start + slc[0].stop) / 2
            center_x = (slc[1].start + slc[1].stop) / 2
            dist = np.sqrt((center_y - tc_lat_idx) ** 2 + (center_x - tc_lon_idx) ** 2)
            if dist < min_dist:
                min_dist, closest_feature_idx = dist, i
        if closest_feature_idx == -1:
            return None
        target_slc = objects_slices[closest_feature_idx]
        target_mask = labeled_array == (closest_feature_idx + 1)
        com_y, com_x = center_of_mass(target_mask)
        pos_lat = self.lat[int(com_y)]
        pos_lon = self.lon[int(com_x)]
        intensity_val = (
            np.max(data_field[target_mask]) if system_type == "high" else np.min(data_field[target_mask])
        )
        lat_min, lat_max = self.lat[target_slc[0].start], self.lat[target_slc[0].stop - 1]
        lon_min, lon_max = self.lon[target_slc[1].start], self.lon[target_slc[1].stop - 1]
        return {
            "position": {
                "center_of_mass": {"lat": round(pos_lat.item(), 2), "lon": round(pos_lon.item(), 2)}
            },
            "intensity": {"value": round(intensity_val.item(), 1), "unit": "gpm"},
            "shape": {},
        }

    # --- 面积计算 ---

    def _calculate_polygon_area_km2(self, coords):
        if not coords or len(coords) < 3:
            return 0.0
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        x_m = (lons - center_lon) * 111000 * np.cos(np.radians(center_lat))
        y_m = (lats - center_lat) * 111000
        area_m2 = 0.5 * abs(
            sum(x_m[i] * y_m[i + 1] - x_m[i + 1] * y_m[i] for i in range(len(x_m) - 1))
        )
        area_km2 = area_m2 / 1e6
        return round(float(area_km2), 1)


__all__ = ["BaseExtractor"]
