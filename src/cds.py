#!/usr/bin/env python3
"""
CDS服务器环境气象系统提取器
基于ERA5数据和台风路径文件，提取关键天气系统
专为CDS服务器环境优化
"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import json
import cdsapi
import os
import sys
from pathlib import Path
import warnings
import concurrent.futures
import gc
import math

_OPTIONAL_DEPS_ERROR = "错误：需要scipy和scikit-image库。请运行 'pip install scipy scikit-image' 进行安装。"

try:  # noqa: SIM105 - dependency guard mirrors original environment_extractor behaviour
    from scipy.ndimage import center_of_mass, find_objects, label
    from skimage.measure import approximate_polygon, find_contours, regionprops
    from skimage.morphology import convex_hull_image
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(_OPTIONAL_DEPS_ERROR) from exc

warnings.filterwarnings('ignore')


def _running_in_notebook() -> bool:
    """Return True when executed inside a Jupyter/IPython kernel."""
    return 'ipykernel' in sys.modules


class WeatherSystemShapeAnalyzer:
    """本地实现的气象系统形状分析器。"""

    def __init__(self, lat_grid, lon_grid):
        self.lat = np.asarray(lat_grid)
        self.lon = np.asarray(lon_grid)
        self.lat_spacing = float(np.abs(np.diff(self.lat).mean())) if self.lat.size > 1 else 1.0
        self.lon_spacing = float(np.abs(np.diff(self.lon).mean())) if self.lon.size > 1 else 1.0

    def analyze_system_shape(
        self, data_field, threshold, system_type="high", center_lat=None, center_lon=None
    ):
        """全面分析气象系统的形状特征."""
        try:
            mask = data_field >= threshold if system_type == "high" else data_field <= threshold
            if not np.any(mask):
                return None

            labeled_mask, num_features = label(mask)
            if num_features == 0:
                return None

            main_region = self._select_main_system(
                labeled_mask, num_features, center_lat, center_lon
            )
            if main_region is None:
                return None

            basic_features = self._calculate_basic_features(
                main_region, data_field, threshold, system_type
            )
            complexity_features = self._calculate_complexity_features(main_region)
            orientation_features = self._calculate_orientation_features(main_region)
            contour_features = self._extract_contour_features(data_field, threshold, system_type)
            multiscale_features = self._calculate_multiscale_features(
                data_field, threshold, system_type
            )

            return {
                "basic_geometry": basic_features,
                "shape_complexity": complexity_features,
                "orientation": orientation_features,
                "contour_analysis": contour_features,
                "multiscale_features": multiscale_features,
            }

        except Exception as exc:  # pragma: no cover - defensive parity with legacy
            print(f"形状分析失败: {exc}")
            return None

    def _select_main_system(self, labeled_mask, num_features, center_lat, center_lon):
        if center_lat is None or center_lon is None:
            flat_labels = labeled_mask.ravel()
            counts = np.bincount(flat_labels)[1 : num_features + 1]
            if counts.size == 0:
                return None
            main_label = int(np.argmax(counts) + 1)
        else:
            center_lat_idx = np.abs(self.lat - center_lat).argmin()
            center_lon_idx = np.abs(self.lon - center_lon).argmin()

            min_dist = float("inf")
            main_label = 1

            for i in range(1, num_features + 1):
                region_mask = labeled_mask == i
                com_y, com_x = center_of_mass(region_mask)
                dist = np.sqrt((com_y - center_lat_idx) ** 2 + (com_x - center_lon_idx) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    main_label = i

        return labeled_mask == main_label

    def _calculate_basic_features(self, region_mask, data_field, threshold, system_type):
        props_list = regionprops(region_mask.astype(int), intensity_image=data_field)
        if not props_list:
            return None
        props = props_list[0]

        area_pixels = props.area
        lat_factor = self.lat_spacing * 111
        lon_factor = self.lon_spacing * 111 * np.cos(np.deg2rad(np.mean(self.lat)))
        area_km2 = area_pixels * lat_factor * lon_factor

        perimeter_pixels = props.perimeter
        perimeter_km = perimeter_pixels * np.sqrt(lat_factor**2 + lon_factor**2)

        compactness = 4 * np.pi * area_km2 / (perimeter_km**2) if perimeter_km > 0 else 0
        shape_index = perimeter_km / (2 * np.sqrt(np.pi * area_km2)) if area_km2 > 0 else 0

        major_axis_length = props.major_axis_length * lat_factor
        minor_axis_length = props.minor_axis_length * lat_factor
        aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 1
        eccentricity = props.eccentricity

        intensity_values = data_field[region_mask]
        if system_type == "high":
            max_intensity = np.max(intensity_values)
            intensity_range = max_intensity - threshold
        else:
            min_intensity = np.min(intensity_values)
            intensity_range = threshold - min_intensity

        return {
            "area_km2": round(area_km2, 1),
            "perimeter_km": round(perimeter_km, 1),
            "compactness": round(compactness, 3),
            "shape_index": round(shape_index, 3),
            "aspect_ratio": round(aspect_ratio, 2),
            "eccentricity": round(eccentricity, 3),
            "major_axis_km": round(major_axis_length, 1),
            "minor_axis_km": round(minor_axis_length, 1),
            "intensity_range": round(intensity_range, 1),
            "description": self._describe_basic_shape(compactness, aspect_ratio, eccentricity),
        }

    def _calculate_complexity_features(self, region_mask):
        convex_hull = convex_hull_image(region_mask)
        convex_area = np.sum(convex_hull)
        actual_area = np.sum(region_mask)

        solidity = actual_area / convex_area if convex_area > 0 else 0

        contours = find_contours(region_mask.astype(float), 0.5)
        if contours:
            main_contour = max(contours, key=len)
            epsilon = 0.02 * len(main_contour)
            approx_contour = approximate_polygon(main_contour, tolerance=epsilon)
            boundary_complexity = (
                len(main_contour) / len(approx_contour) if len(approx_contour) > 0 else 1
            )
        else:
            boundary_complexity = 1

        fractal_dimension = self._estimate_fractal_dimension(region_mask)

        return {
            "solidity": round(solidity, 3),
            "boundary_complexity": round(boundary_complexity, 2),
            "fractal_dimension": round(fractal_dimension, 3),
            "description": self._describe_complexity(solidity, boundary_complexity),
        }

    def _calculate_orientation_features(self, region_mask):
        props_list = regionprops(region_mask.astype(int))
        if not props_list:
            return {
                "orientation_deg": 0.0,
                "direction_type": "方向不明确",
                "description": "区域形状不足以确定主轴方向",
            }

        props = props_list[0]
        orientation_rad = props.orientation
        orientation_deg = float(np.degrees(orientation_rad))

        if orientation_deg < 0:
            orientation_deg += 180

        if 0 <= orientation_deg < 22.5 or 157.5 <= orientation_deg <= 180:
            direction_desc = "南北向延伸"
        elif 22.5 <= orientation_deg < 67.5:
            direction_desc = "东北-西南向延伸"
        elif 67.5 <= orientation_deg < 112.5:
            direction_desc = "东西向延伸"
        else:
            direction_desc = "西北-东南向延伸"

        return {
            "orientation_deg": round(orientation_deg, 1),
            "direction_type": direction_desc,
            "description": f"系统主轴呈{direction_desc}，方向角为{orientation_deg:.1f}°",
        }

    def _extract_contour_features(self, data_field, threshold, system_type):
        try:
            contours = find_contours(data_field, threshold)
            if not contours:
                return None

            main_contour = max(contours, key=len)

            contour_lats = self.lat[main_contour[:, 0].astype(int)]
            contour_lons = self.lon[main_contour[:, 1].astype(int)]

            contour_length_km = 0
            for i in range(1, len(contour_lats)):
                dist = self._haversine_distance(
                    contour_lats[i - 1], contour_lons[i - 1], contour_lats[i], contour_lons[i]
                )
                contour_length_km += dist

            step = max(1, len(main_contour) // 50)
            simplified_contour = [
                [round(lon, 2), round(lat, 2)]
                for lat, lon in zip(contour_lats[::step], contour_lons[::step])
            ]

            polygon_features = self._extract_polygon_coordinates(main_contour)

            return {
                "contour_length_km": round(contour_length_km, 1),
                "contour_points": len(main_contour),
                "simplified_coordinates": simplified_contour,
                "polygon_features": polygon_features,
                "description": f"主等值线长度{contour_length_km:.0f}km，包含{len(main_contour)}个数据点",
            }
        except Exception:  # pragma: no cover - parity with legacy fallback
            return None

    def _extract_polygon_coordinates(self, contour):
        try:
            epsilon = 0.02 * len(contour)
            approx_polygon = approximate_polygon(contour, tolerance=epsilon)

            polygon_coords = []
            for point in approx_polygon:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                polygon_coords.append([round(float(self.lon[lon_idx]), 2), round(float(self.lat[lat_idx]), 2)])

            if polygon_coords:
                lons = [coord[0] for coord in polygon_coords]
                lats = [coord[1] for coord in polygon_coords]
                bbox = [
                    round(float(min(lons)), 2),
                    round(float(min(lats)), 2),
                    round(float(max(lons)), 2),
                    round(float(max(lats)), 2),
                ]

                center = [round(float(np.mean(lons)), 2), round(float(np.mean(lats)), 2)]

                cardinal_points = {
                    "N": [lons[lats.index(max(lats))], max(lats)],
                    "S": [lons[lats.index(min(lats))], min(lats)],
                    "E": [max(lons), lats[lons.index(max(lons))]],
                    "W": [min(lons), lats[lons.index(min(lons))]],
                }

                return {
                    "polygon": polygon_coords,
                    "vertices": len(polygon_coords),
                    "bbox": bbox,
                    "center": center,
                    "cardinal_points": cardinal_points,
                    "span": [
                        round(bbox[2] - bbox[0], 2),
                        round(bbox[3] - bbox[1], 2),
                    ],
                }

            return None
        except Exception:  # pragma: no cover - parity
            return None

    def _calculate_multiscale_features(self, data_field, threshold, system_type):
        features = {}

        if system_type == "high":
            thresholds = [threshold, threshold + 20, threshold + 40]
            threshold_names = ["外边界", "中等强度", "强中心"]
        else:
            thresholds = [threshold, threshold - 20, threshold - 40]
            threshold_names = ["外边界", "中等强度", "强中心"]

        for thresh, name in zip(thresholds, threshold_names):
            mask = data_field >= thresh if system_type == "high" else data_field <= thresh

            if np.any(mask):
                area_pixels = np.sum(mask)
                lat_factor = self.lat_spacing * 111
                lon_factor = self.lon_spacing * 111 * np.cos(np.deg2rad(np.mean(self.lat)))
                area_km2 = area_pixels * lat_factor * lon_factor
                features[f"area_{name}_km2"] = round(area_km2, 1)
            else:
                features[f"area_{name}_km2"] = 0

        outer_area = features.get("area_外边界_km2", 0)
        if outer_area > 0:
            features["core_ratio"] = round(
                features.get("area_强中心_km2", 0) / outer_area, 3
            )
            features["middle_ratio"] = round(
                features.get("area_中等强度_km2", 0) / outer_area, 3
            )

        return features

    def _describe_basic_shape(self, compactness, aspect_ratio, eccentricity):
        if compactness > 0.7:
            shape_desc = "近圆形"
        elif compactness > 0.4:
            shape_desc = "较规则"
        else:
            shape_desc = "不规则"

        if aspect_ratio > 3:
            elongation_desc = "高度拉长"
        elif aspect_ratio > 2:
            elongation_desc = "明显拉长"
        elif aspect_ratio > 1.5:
            elongation_desc = "略微拉长"
        else:
            elongation_desc = "较为圆润"

        return f"{shape_desc}的{elongation_desc}系统"

    def _describe_complexity(self, solidity, boundary_complexity):
        if solidity > 0.9:
            complexity_desc = "边界平滑"
        elif solidity > 0.7:
            complexity_desc = "边界较规则"
        else:
            complexity_desc = "边界复杂"

        if boundary_complexity > 2:
            detail_desc = "具有精细结构"
        elif boundary_complexity > 1.5:
            detail_desc = "具有一定细节"
        else:
            detail_desc = "结构相对简单"

        return f"{complexity_desc}，{detail_desc}"

    def _estimate_fractal_dimension(self, region_mask):
        try:
            sizes = [2, 4, 8, 16]
            counts = []

            for size in sizes:
                h, w = region_mask.shape
                count = 0
                for i in range(0, h, size):
                    for j in range(0, w, size):
                        box = region_mask[i : min(i + size, h), j : min(j + size, w)]
                        if np.any(box):
                            count += 1
                counts.append(count)

            if len(counts) > 1 and all(c > 0 for c in counts):
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                fractal_dim = -coeffs[0]
                return max(1.0, min(2.0, float(fractal_dim)))
            return 1.5
        except Exception:  # pragma: no cover - parity
            return 1.5

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

class CDSEnvironmentExtractor:
    """
    CDS服务器环境气象系统提取器
    """

    def __init__(self, tracks_file, output_dir="./cds_output", cleanup_intermediate=True, max_workers=None, dask_chunks_env="CDS_XR_CHUNKS"):
        """
        初始化提取器

        Args:
            tracks_file: 台风路径CSV文件路径
            output_dir: 输出目录
            cleanup_intermediate: 是否在分析完成后清理中间ERA5数据文件
            max_workers: 并行处理的最大工作线程数（None=自动，1=禁用并行）
            dask_chunks_env: 从环境变量读取xarray分块设置的键名（例如 "time:1,latitude:200,longitude:200"）
        """
        self.tracks_file = tracks_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cleanup_intermediate = cleanup_intermediate
        self.max_workers = max_workers
        self.dask_chunks_env = dask_chunks_env

        # CDS API客户端
        self._check_cdsapi_config()
        self.cds_client = cdsapi.Client()

        # 加载台风路径数据
        self.load_tracks_data()

        # 下载文件记录，便于后续清理
        self._downloaded_files = []
        self._grad_cache = {}

        print("✅ CDS环境提取器初始化完成")

    def _check_cdsapi_config(self):
        """检查CDS API配置是否可用，并给出提示（在CDS JupyterLab中尤为重要）"""
        try:
            test_client = cdsapi.Client()
            print("🛠️ CDS API客户端创建成功")
            return True
        except Exception as e:
            print(f"⚠️ CDS API配置验证失败: {e}")
            print("请确保在CDS JupyterLab环境中运行，或正确配置CDS API凭据")
            return False

    def load_tracks_data(self):
        """加载台风路径数据"""
        try:
            self.tracks_df = pd.read_csv(self.tracks_file)
            self.tracks_df['datetime'] = pd.to_datetime(self.tracks_df['datetime'])

            # 重命名列以匹配标准格式
            column_mapping = {
                'latitude': 'lat',
                'longitude': 'lon',
                'datetime': 'time',
                'storm_id': 'particle'
            }

            self.tracks_df = self.tracks_df.rename(columns=column_mapping)

            # 添加time_idx列
            self.tracks_df['time_idx'] = range(len(self.tracks_df))

            print(f"📊 加载了 {len(self.tracks_df)} 个路径点")
            print(f"🌀 台风ID: {self.tracks_df['particle'].unique()}")

        except Exception as e:
            message = f"❌ 加载路径数据失败: {e}"
            print(message)
            raise RuntimeError(message) from e

    def download_era5_data(self, start_date, end_date):
        """
        从CDS下载ERA5数据（无区域裁剪，默认全局范围）

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        output_file = self.output_dir / f"era5_single_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"

        if output_file.exists():
            print(f"📁 ERA5数据已存在: {output_file}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        print(f"📥 下载ERA5数据: {start_date} 到 {end_date}")

        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            self.cds_client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'mean_sea_level_pressure', '10m_u_component_of_wind',
                        '10m_v_component_of_wind', '2m_temperature',
                        'sea_surface_temperature', 'total_column_water_vapour'
                    ],
                    'year': sorted(list(set(date_range.year))),
                    'month': sorted(list(set(date_range.month))),
                    'day': sorted(list(set(date_range.day))),
                    'time': [
                        '00:00', '06:00', '12:00', '18:00'
                    ],
                },
                str(output_file)
            )

            print(f"✅ ERA5数据下载完成: {output_file}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        except Exception as e:
            print(f"❌ ERA5数据下载失败: {e}")
            return None

    def download_era5_pressure_data(self, start_date, end_date, levels=("850","500","200")):
        """从CDS下载ERA5等压面数据（无区域裁剪，默认全局范围）"""
        output_file = self.output_dir / f"era5_pressure_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"

        if output_file.exists():
            print(f"📁 ERA5等压面数据已存在: {output_file}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        print(f"📥 下载ERA5等压面数据: {start_date} 到 {end_date}")

        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            self.cds_client.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'u_component_of_wind', 'v_component_of_wind',
                        'geopotential', 'temperature', 'relative_humidity'
                    ],
                    'pressure_level': list(levels),
                    'year': sorted(list(set(date_range.year))),
                    'month': sorted(list(set(date_range.month))),
                    'day': sorted(list(set(date_range.day))),
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                },
                str(output_file)
            )

            print(f"✅ ERA5等压面数据下载完成: {output_file}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        except Exception as e:
            print(f"❌ ERA5等压面数据下载失败: {e}")
            return None

    def _parse_chunks_from_env(self):
        """从环境变量解析xarray分块设置"""
        chunks_str = os.environ.get(self.dask_chunks_env, "").strip()
        if not chunks_str:
            return None
        chunks = {}
        try:
            for part in chunks_str.split(','):
                k, v = part.split(':')
                chunks[k.strip()] = int(v.strip())
            return chunks
        except Exception:
            print(f"⚠️ 无法解析 {self.dask_chunks_env} 环境变量的分块设置: '{chunks_str}'")
            return None

    def load_era5_data(self, single_file, pressure_file=None):
        """加载ERA5数据文件"""
        try:
            chunks = self._parse_chunks_from_env()
            open_kwargs = {"chunks": chunks} if chunks else {}

            ds_single = xr.open_dataset(single_file, **open_kwargs)
            
            if pressure_file and Path(pressure_file).exists():
                ds_pressure = xr.open_dataset(pressure_file, **open_kwargs)
                self.ds = xr.merge([ds_single, ds_pressure])
            else:
                self.ds = ds_single

            print(f"📊 ERA5数据加载完成: {dict(self.ds.dims)}")
            if 'latitude' in self.ds and 'longitude' in self.ds:
                print(f"🌍 坐标范围: lat {self.ds.latitude.min().values:.1f}°-{self.ds.latitude.max().values:.1f}°, "
                      f"lon {self.ds.longitude.min().values:.1f}°-{self.ds.longitude.max().values:.1f}°")
            self._initialize_coordinate_metadata()
            return True
        except Exception as e:
            print(f"❌ 加载ERA5数据失败: {e}")
            return False

    def _initialize_coordinate_metadata(self):
        """初始化经纬度、时间等元数据，便于后续与高级提取逻辑保持一致"""
        # 纬度坐标
        lat_coord = next((name for name in ("latitude", "lat") if name in self.ds.coords), None)
        lon_coord = next((name for name in ("longitude", "lon") if name in self.ds.coords), None)
        if lat_coord is None or lon_coord is None:
            raise ValueError("数据集中缺少纬度或经度坐标")

        self._lat_name = lat_coord
        self._lon_name = lon_coord
        self.latitudes = np.asarray(self.ds[self._lat_name].values)
        self.longitudes = np.asarray(self.ds[self._lon_name].values)

        # 处理经度到 [0, 360) 区间的标准化形式，便于距离判断
        self._lon_normalized = self._normalize_lon(self.longitudes)

        # 维度间距（度），如只有单点则使用1避免除零
        if self.latitudes.size > 1:
            self.lat_spacing = float(np.abs(np.diff(self.latitudes).mean()))
        else:
            self.lat_spacing = 1.0

        if self.longitudes.size > 1:
            sorted_unique_lon = np.sort(np.unique(self.longitudes))
            diffs = np.abs(np.diff(sorted_unique_lon))
            self.lon_spacing = float(diffs[diffs > 0].mean()) if np.any(diffs > 0) else 1.0
        else:
            self.lon_spacing = 1.0

        cos_lat = np.cos(np.deg2rad(self.latitudes))
        self._coslat = cos_lat
        self._coslat_safe = np.where(np.abs(cos_lat) < 1e-6, np.nan, cos_lat)

        # 时间轴
        time_dim = None
        time_coord_name = None
        if 'time' in self.ds.dims:
            time_dim = 'time'
        elif 'valid_time' in self.ds.dims:
            time_dim = 'valid_time'

        if time_dim is None:
            for candidate in ("time", "valid_time"):
                if candidate in self.ds.coords:
                    dims = self.ds[candidate].dims
                    if dims:
                        time_dim = dims[0]
                        time_coord_name = candidate
                        break

        if time_dim is None:
            raise ValueError("数据集中缺少时间维度")

        if time_coord_name is None:
            time_coord_name = time_dim

        self._time_dim = time_dim
        self._time_coord_name = time_coord_name
        time_coord = self.ds[time_coord_name]
        time_values = pd.to_datetime(time_coord.values)
        self._time_values = np.asarray(time_values)

        # 建立与高级形状分析一致的辅助属性
        self.lat = self.latitudes
        self.lon = self.longitudes
        self._grad_cache = {}

        self.shape_analyzer = WeatherSystemShapeAnalyzer(self.lat, self.lon)

    def extract_environmental_systems(self, time_point, tc_lat, tc_lon):
        """提取指定时间点的环境系统，输出格式与environment_extractor保持一致"""
        systems = []
        try:
            time_idx, era5_time = self._get_time_index(time_point)
            print(f"🔍 处理时间点: {time_point} (ERA5时间: {era5_time})")

            ds_at_time = self._dataset_at_index(time_idx)

            system_extractors = [
                lambda: self.extract_steering_system(time_idx, tc_lat, tc_lon),
                lambda: self.extract_vertical_wind_shear(time_idx, tc_lat, tc_lon),
                lambda: self.extract_ocean_heat_content(time_idx, tc_lat, tc_lon),
                lambda: self.extract_upper_level_divergence(time_idx, tc_lat, tc_lon),
                lambda: self.extract_intertropical_convergence_zone(time_idx, tc_lat, tc_lon),
                lambda: self.extract_westerly_trough(time_idx, tc_lat, tc_lon),
                lambda: self.extract_frontal_system(time_idx, tc_lat, tc_lon),
                lambda: self.extract_monsoon_trough(time_idx, tc_lat, tc_lon),
                lambda: self.extract_low_level_flow(ds_at_time, tc_lat, tc_lon),
                lambda: self.extract_atmospheric_stability(ds_at_time, tc_lat, tc_lon),
            ]

            for extractor in system_extractors:
                system_obj = extractor()
                if system_obj:
                    systems.append(system_obj)

        except Exception as e:
            print(f"⚠️ 提取环境系统失败: {e}")
            systems.append({"system_name": "ExtractionError", "description": str(e)})
        return systems

    def _normalize_lon(self, lon_values):
        arr = np.asarray(lon_values, dtype=np.float64)
        return np.mod(arr + 360.0, 360.0)

    def _loc_idx(self, lat_val: float, lon_val: float):
        if not hasattr(self, "latitudes") or not hasattr(self, "longitudes"):
            raise RuntimeError("坐标元数据尚未初始化，无法定位网格索引")
        lat_idx = int(np.abs(self.latitudes - lat_val).argmin())
        lon_idx = int(np.abs(self.longitudes - lon_val).argmin())
        return lat_idx, lon_idx

    def _raw_gradients(self, arr: np.ndarray):
        cache = getattr(self, "_grad_cache", None)
        if cache is None:
            cache = {}
            self._grad_cache = cache
        key = id(arr)
        if key in cache:
            return cache[key]

        gy = np.gradient(arr, axis=0)
        gx = np.gradient(arr, axis=1)
        cache[key] = (gy, gx)
        return gy, gx

    def _lon_distance(self, lon_values, center_lon):
        lon_norm = self._normalize_lon(lon_values)
        center = float(self._normalize_lon(center_lon))
        diff = np.abs(lon_norm - center)
        return np.minimum(diff, 360.0 - diff)

    def _get_time_index(self, target_time):
        if not hasattr(self, "_time_values"):
            raise ValueError("尚未初始化时间轴信息")
        target_ts = pd.Timestamp(target_time)
        target_np = np.datetime64(target_ts.to_datetime64())
        diffs = np.abs(self._time_values - target_np).astype('timedelta64[s]').astype(np.int64)
        idx = int(diffs.argmin())
        era5_time = pd.Timestamp(self._time_values[idx]).to_pydatetime()
        return idx, era5_time

    def _dataset_at_index(self, time_idx):
        selector = {self._time_dim: time_idx} if self._time_dim in self.ds.dims else {}
        ds_at_time = self.ds.isel(**selector)
        if 'expver' in ds_at_time.dims:
            ds_at_time = ds_at_time.isel(expver=0)
        return ds_at_time.squeeze()

    def _get_field_at_time(self, var_name, time_idx):
        if var_name not in self.ds.data_vars:
            return None
        data = self.ds[var_name]
        indexers = {}
        if self._time_dim in data.dims:
            indexers[self._time_dim] = time_idx
        field = data.isel(**indexers)
        if 'expver' in field.dims:
            field = field.isel(expver=0)
        return field.squeeze()

    def _get_data_at_level(self, var_name, level_hpa, time_idx):
        if var_name not in self.ds.data_vars:
            return None
        data = self.ds[var_name]
        indexers = {}
        if self._time_dim in data.dims:
            indexers[self._time_dim] = time_idx
        # ERA5 pressure层在不同产品中的维度命名不尽相同，这里做一次统一识别
        level_dim_candidates = (
            "level",
            "isobaricInhPa",
            "pressure",
            "pressure_level",
            "plev",
            "lev",
        )
        level_dim = next((dim for dim in level_dim_candidates if dim in data.dims), None)
        if level_dim is None:
            field = data.isel(**indexers)
        else:
            if level_dim in data.coords:
                level_values = data[level_dim].values
            elif level_dim in self.ds.coords:
                level_values = self.ds[level_dim].values
            else:
                level_values = np.arange(data.sizes[level_dim])
            try:
                numeric_levels = np.asarray(level_values, dtype=float)
            except (TypeError, ValueError):
                numeric_levels = np.asarray([float(str(v)) for v in level_values])
            level_idx = int(np.abs(numeric_levels - float(level_hpa)).argmin())
            indexers[level_dim] = level_idx
            field = data.isel(**indexers)
        if 'expver' in field.dims:
            field = field.isel(expver=0)
        result = field.squeeze()
        return result.values if hasattr(result, 'values') else np.asarray(result)

    def _get_sst_field(self, time_idx):
        # 优先使用海表温度，如无则使用地表温度近似
        for var_name in ("sst", "ts"):
            field = self._get_field_at_time(var_name, time_idx)
            if field is not None:
                values = field.values if hasattr(field, "values") else np.asarray(field)
                if np.nanmean(values) > 200:
                    values = values - 273.15
                return values

        for var_name in ("t2", "t2m"):
            field = self._get_field_at_time(var_name, time_idx)
            if field is not None:
                values = field.values if hasattr(field, "values") else np.asarray(field)
                if np.nanmean(values) > 200:
                    values = values - 273.15
                print(f"⚠️ 使用{var_name}作为海表温度近似")
                return values
        return None

    def _create_region_mask(self, center_lat, center_lon, radius_deg):
        lat_mask = (self.lat >= center_lat - radius_deg) & (self.lat <= center_lat + radius_deg)
        lon_mask = (self.lon >= center_lon - radius_deg) & (self.lon <= center_lon + radius_deg)
        return np.outer(lat_mask, lon_mask)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        return float(self._haversine_distance(lat1, lon1, lat2, lon2))

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        dlon = math.radians(lon2 - lon1)
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
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
        end_lon = lon + u * scale * 0.009 / max(math.cos(math.radians(lat)), 1e-6)
        return {
            "start": {"lat": round(lat, 2), "lon": round(lon, 2)},
            "end": {"lat": round(end_lat, 2), "lon": round(end_lon, 2)},
        }

    def _calculate_steering_flow(self, z500, tc_lat, tc_lon):
        gy, gx = self._raw_gradients(z500)
        dy = gy / (self.lat_spacing * 111000.0)
        dx = gx / (self.lon_spacing * 111000.0 * self._coslat_safe[:, np.newaxis])
        lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
        u_steering = -dx[lat_idx, lon_idx] / (9.8 * 1e-5)
        v_steering = dy[lat_idx, lon_idx] / (9.8 * 1e-5)
        speed = float(np.sqrt(u_steering**2 + v_steering**2))
        direction = (float(np.degrees(np.arctan2(u_steering, v_steering))) + 180.0) % 360.0
        return speed, direction, float(u_steering), float(v_steering)

    def _get_contour_coords(self, data_field, level, max_points=100):
        try:
            contours = find_contours(data_field, level)
            if not contours:
                return None
            main_contour = sorted(contours, key=len, reverse=True)[0]
            contour_lon = self.lon[main_contour[:, 1].astype(int)]
            contour_lat = self.lat[main_contour[:, 0].astype(int)]
            step = max(1, len(main_contour) // max_points)
            return [
                [round(float(lon), 2), round(float(lat), 2)]
                for lon, lat in zip(contour_lon[::step], contour_lat[::step])
            ]
        except Exception:
            return None

    def _get_enhanced_shape_info(self, data_field, threshold, system_type, center_lat, center_lon):
        """获取增强的形状信息（简化版，仅包含边界坐标）."""
        try:
            shape_analysis = self.shape_analyzer.analyze_system_shape(
                data_field, threshold, system_type, center_lat, center_lon
            )
            if not shape_analysis:
                return None

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
        except Exception as exc:
            print(f"形状分析失败: {exc}")
            return None

    def _get_system_coordinates(self, data_field, threshold, system_type, max_points=20):
        try:
            mask = data_field >= threshold if system_type == "high" else data_field <= threshold
            if not np.any(mask):
                return None

            labeled_mask, num_features = label(mask)
            if num_features == 0:
                return None

            counts = np.bincount(labeled_mask.ravel())[1 : num_features + 1]
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
                step = max(1, len(simplified) // max_points)
                simplified = simplified[::step]

            geo_coords = []
            for point in simplified:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                geo_coords.append([
                    round(float(self.lon[lon_idx]), 3),
                    round(float(self.lat[lat_idx]), 3),
                ])

            if not geo_coords:
                return None

            lons = [coord[0] for coord in geo_coords]
            lats = [coord[1] for coord in geo_coords]
            extent = {
                "boundaries": [
                    round(float(min(lons)), 3),
                    round(float(min(lats)), 3),
                    round(float(max(lons)), 3),
                    round(float(max(lats)), 3),
                ],
                "center": [
                    round(float(np.mean(lons)), 3),
                    round(float(np.mean(lats)), 3),
                ],
                "span": [
                    round(float(max(lons) - min(lons)), 3),
                    round(float(max(lats) - min(lats)), 3),
                ],
            }

            return {
                "vertices": geo_coords,
                "vertex_count": len(geo_coords),
                "extent": extent,
                "span_deg": [extent["span"][0], extent["span"][1]],
            }
        except Exception as exc:
            print(f"坐标提取失败: {exc}")
            return None

    def _generate_coordinate_description(self, coords_info, system_name="系统"):
        if not coords_info:
            return ""

        try:
            parts = []
            extent = coords_info.get("extent")
            if extent and "boundaries" in extent:
                west, south, east, north = extent["boundaries"]
                parts.append(
                    f"{system_name}主体位于{west:.1f}°E-{east:.1f}°E，{south:.1f}°N-{north:.1f}°N"
                )

            if coords_info.get("vertex_count"):
                parts.append(f"由{coords_info['vertex_count']}个关键顶点构成的多边形形状")

            if "span_deg" in coords_info:
                lon_span, lat_span = coords_info["span_deg"]
                center_lat = extent.get("center", [0, 30])[1] if extent else 30
                lat_km = lat_span * 111.0
                lon_km = lon_span * 111.0 * math.cos(math.radians(center_lat))
                parts.append(f"纬向跨度约{lat_km:.0f}km，经向跨度约{lon_km:.0f}km")

            return "，".join(parts) + "。" if parts else ""
        except Exception:
            return ""

    def _identify_pressure_system(self, data_field, tc_lat, tc_lon, system_type, threshold):
        mask = data_field > threshold if system_type == "high" else data_field < threshold
        if not np.any(mask):
            return None

        labeled_array, num_features = label(mask)
        if num_features == 0:
            return None

        objects_slices = find_objects(labeled_array)
        tc_lat_idx, tc_lon_idx = self._loc_idx(tc_lat, tc_lon)
        min_dist = float("inf")
        closest_idx = -1
        for i, slc in enumerate(objects_slices):
            center_y = (slc[0].start + slc[0].stop) / 2
            center_x = (slc[1].start + slc[1].stop) / 2
            dist = math.hypot(center_y - tc_lat_idx, center_x - tc_lon_idx)
            if dist < min_dist:
                min_dist, closest_idx = dist, i

        if closest_idx == -1:
            return None

        target_mask = labeled_array == (closest_idx + 1)
        com_y, com_x = center_of_mass(target_mask)
        pos_lat = float(self.lat[int(com_y)])
        pos_lon = float(self.lon[int(com_x)])
        intensity_val = (
            float(np.max(data_field[target_mask]))
            if system_type == "high"
            else float(np.min(data_field[target_mask]))
        )

        return {
            "position": {
                "center_of_mass": {
                    "lat": round(pos_lat, 2),
                    "lon": round(pos_lon, 2),
                }
            },
            "intensity": {"value": round(intensity_val, 1), "unit": "gpm"},
            "shape": {},
        }

    def extract_steering_system(self, time_idx, tc_lat, tc_lon):
        """提取副热带高压及引导气流，匹配extractSyst结构"""
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            field_values = np.asarray(z500, dtype=float)
            if np.nanmean(field_values) > 10000:
                field_values = field_values / 9.80665

            threshold = 5880
            subtropical_high = self._identify_pressure_system(
                field_values, tc_lat, tc_lon, "high", threshold
            )
            if not subtropical_high:
                return None

            enhanced_shape = self._get_enhanced_shape_info(
                field_values, threshold, "high", tc_lat, tc_lon
            )

            steering_speed, steering_direction, u_steering, v_steering = self._calculate_steering_flow(
                field_values, tc_lat, tc_lon
            )

            intensity_val = subtropical_high["intensity"]["value"]
            if intensity_val > 5900:
                level = "强"
            elif intensity_val > 5880:
                level = "中等"
            else:
                level = "弱"
            subtropical_high["intensity"]["level"] = level

            if enhanced_shape:
                subtropical_high.setdefault("shape", {}).update(
                    {
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                        "area_km2": enhanced_shape["area_km2"],
                        "shape_type": enhanced_shape["shape_type"],
                        "orientation": enhanced_shape["orientation"],
                        "complexity": enhanced_shape["complexity"],
                    }
                )
                coord_info = enhanced_shape.get("coordinate_info")
                if coord_info:
                    subtropical_high["shape"]["coordinate_details"] = coord_info
                else:
                    subtropical_high.setdefault("shape", {})

            system_coords = self._get_system_coordinates(field_values, threshold, "high", max_points=15)
            if system_coords:
                subtropical_high["shape"]["coordinates"] = system_coords
                subtropical_high["shape"]["coordinate_description"] = self._generate_coordinate_description(
                    system_coords, "副热带高压"
                )

            contour_coords = self._get_contour_coords(field_values, threshold, max_points=120)
            if contour_coords:
                subtropical_high["shape"]["contour_5880gpm"] = contour_coords

            center = subtropical_high["position"].get("center_of_mass", {})
            bearing, rel_dir_text = self._calculate_bearing(
                tc_lat, tc_lon, center.get("lat", tc_lat), center.get("lon", tc_lon)
            )
            distance = self._calculate_distance(
                tc_lat, tc_lon, center.get("lat", tc_lat), center.get("lon", tc_lon)
            )

            desc = (
                f"一个强度为“{level}”的副热带高压系统位于台风的{rel_dir_text}，其主体形态稳定，"
                f"为台风提供了稳定的{steering_direction:.0f}°方向、速度为{steering_speed:.1f} m/s的引导气流。"
            )

            subtropical_high.update(
                {
                    "system_name": "SubtropicalHigh",
                    "description": desc,
                    "position": {
                        "description": "副热带高压中心",
                        "center_of_mass": {
                            "lat": round(center.get("lat", tc_lat), 2),
                            "lon": round(center.get("lon", tc_lon), 2),
                        },
                        "relative_to_tc": rel_dir_text,
                        "distance_km": round(distance, 1),
                        "bearing_deg": round(bearing, 1),
                    },
                    "properties": {
                        "influence": "主导台风未来路径",
                        "steering_flow": {
                            "speed_mps": round(steering_speed, 2),
                            "direction_deg": round(steering_direction, 1),
                            "vector_mps": {"u": round(u_steering, 2), "v": round(v_steering, 2)},
                        },
                    },
                }
            )
            return subtropical_high
        except Exception as exc:
            print(f"⚠️ 引导系统提取失败: {exc}")
            return None

    # 兼容旧名称
    def extract_subtropical_high(self, time_idx, ds_at_time, tc_lat, tc_lon):
        return self.extract_steering_system(time_idx, tc_lat, tc_lon)

    def extract_ocean_heat_content(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        """提取海洋热含量及暖水边界信息"""
        try:
            sst = self._get_sst_field(time_idx)
            if sst is None:
                return None

            region_mask = self._create_region_mask(tc_lat, tc_lon, radius_deg)
            if not np.any(region_mask):
                return None

            mean_temp = float(np.nanmean(np.where(region_mask, sst, np.nan)))
            if not np.isfinite(mean_temp):
                return None

            if mean_temp > 29:
                level, impact = "极高", "为爆发性增强提供顶级能量"
            elif mean_temp > 28:
                level, impact = "高", "非常有利于加强"
            elif mean_temp > 26.5:
                level, impact = "中等", "足以维持强度"
            else:
                level, impact = "低", "能量供应不足，将导致减弱"

            contour_26_5 = self._get_contour_coords(sst, 26.5)
            enhanced_shape = self._get_enhanced_shape_info(sst, 26.5, "high", tc_lat, tc_lon)

            desc = (
                f"台风下方海域的平均海表温度为{mean_temp:.1f}°C，海洋热含量等级为“{level}”，{impact}。"
            )

            shape_info = {
                "description": "26.5°C是台风发展的最低海温门槛，此线是生命线",
                "warm_water_boundary_26.5C": contour_26_5,
            }

            if enhanced_shape:
                shape_info.update(
                    {
                        "warm_water_area_km2": enhanced_shape["area_km2"],
                        "warm_region_shape": enhanced_shape["shape_type"],
                        "warm_region_orientation": enhanced_shape["orientation"],
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                    }
                )
                desc += (
                    f" 暖水区域面积约{enhanced_shape['area_km2']:.0f}km²，呈{enhanced_shape['shape_type']}，"
                    f"{enhanced_shape['orientation']}。"
                )

            return {
                "system_name": "OceanHeatContent",
                "description": desc,
                "position": {
                    "description": f"台风中心周围{radius_deg}度半径内的海域",
                    "lat": round(tc_lat, 2),
                    "lon": round(tc_lon, 2),
                },
                "intensity": {"value": round(mean_temp, 2), "unit": "°C", "level": level},
                "shape": shape_info,
                "properties": {"impact": impact},
            }
        except Exception as exc:
            print(f"⚠️ 海洋热含量提取失败: {exc}")
            return None

    # 兼容旧名称
    def extract_ocean_heat(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        return self.extract_ocean_heat_content(time_idx, tc_lat, tc_lon, radius_deg)

    def extract_upper_level_divergence(self, time_idx, tc_lat, tc_lon):
        try:
            u200 = self._get_data_at_level("u", 200, time_idx)
            v200 = self._get_data_at_level("v", 200, time_idx)
            if u200 is None or v200 is None:
                return None

            with np.errstate(divide="ignore", invalid="ignore"):
                gy_u, gx_u = self._raw_gradients(u200)
                gy_v, gx_v = self._raw_gradients(v200)
                du_dx = gx_u / (self.lon_spacing * 111000.0 * self._coslat_safe[:, np.newaxis])
                dv_dy = gy_v / (self.lat_spacing * 111000.0)
                divergence = du_dx + dv_dy

            if not np.any(np.isfinite(divergence)):
                return None

            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            div_val = divergence[lat_idx, lon_idx]
            if not np.isfinite(div_val):
                sub = divergence[max(0, lat_idx - 1) : lat_idx + 2, max(0, lon_idx - 1) : lon_idx + 2]
                finite = sub[np.isfinite(sub)]
                if finite.size == 0:
                    return None
                div_val = float(np.nanmean(finite))
            div_val = float(np.clip(div_val, -5e-4, 5e-4))
            div_scaled = div_val * 1e5

            if div_scaled > 5:
                level, impact = "强", "极其有利于台风发展和加强"
            elif div_scaled > 2:
                level, impact = "中等", "有利于台风维持和发展"
            elif div_scaled > -2:
                level, impact = "弱", "对台风发展影响较小"
            else:
                level, impact = "负值", "不利于台风发展"

            desc = (
                f"台风上方200hPa高度的散度值为{div_scaled:.1f}×10⁻⁵ s⁻¹，"
                f"高空辐散强度为'{level}'，{impact}。"
            )

            return {
                "system_name": "UpperLevelDivergence",
                "description": desc,
                "position": {
                    "description": "台风中心上方200hPa高度",
                    "lat": round(tc_lat, 2),
                    "lon": round(tc_lon, 2),
                },
                "intensity": {"value": round(div_scaled, 2), "unit": "×10⁻⁵ s⁻¹", "level": level},
                "shape": {"description": "高空辐散中心的空间分布"},
                "properties": {"impact": impact, "favorable_development": div_scaled > 0},
            }
        except Exception as exc:
            print(f"⚠️ 高空辐散提取失败: {exc}")
            return None

    def extract_intertropical_convergence_zone(self, time_idx, tc_lat, tc_lon):
        try:
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if u850 is None or v850 is None:
                return None

            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)
            du_dy = gy_u / (self.lat_spacing * 111000.0)
            dv_dx = gx_v / (self.lon_spacing * 111000.0 * self._coslat_safe[:, np.newaxis])
            vorticity = dv_dx - du_dy

            tropical_mask = (self.lat >= 0) & (self.lat <= 20)
            if not np.any(tropical_mask):
                return None

            tropical_vort = vorticity[tropical_mask, :]
            rel_idx = np.nanargmax(tropical_vort)
            lat_idx = rel_idx // tropical_vort.shape[1]
            itcz_lat = float(self.lat[tropical_mask][lat_idx])

            distance_deg = abs(tc_lat - itcz_lat)
            if distance_deg < 5:
                influence = "直接影响台风发展"
            elif distance_deg < 10:
                influence = "对台风路径有显著影响"
            else:
                influence = "对台风影响较小"

            desc = (
                f"热带辐合带当前位于约{itcz_lat:.1f}°N附近，与台风中心距离{distance_deg:.1f}度，{influence}。"
            )

            return {
                "system_name": "InterTropicalConvergenceZone",
                "description": desc,
                "position": {"description": "热带辐合带位置", "lat": round(itcz_lat, 1), "lon": "跨经度带"},
                "intensity": {"description": "基于850hPa涡度确定的活跃程度"},
                "shape": {"description": "东西向延伸的辐合带"},
                "properties": {
                    "distance_to_tc_deg": round(distance_deg, 1),
                    "influence": influence,
                },
            }
        except Exception as exc:
            print(f"⚠️ ITCZ 提取失败: {exc}")
            return None

    def extract_westerly_trough(self, time_idx, tc_lat, tc_lon):
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            field_values = np.asarray(z500, dtype=float)
            if np.nanmean(field_values) > 10000:
                field_values = field_values / 9.80665

            mid_lat_mask = (self.lat >= 20) & (self.lat <= 60)
            if not np.any(mid_lat_mask):
                return None

            z500_mid = field_values[mid_lat_mask, :]
            trough_threshold = np.percentile(z500_mid, 25)
            trough = self._identify_pressure_system(
                field_values, tc_lat, tc_lon, "low", trough_threshold
            )
            if not trough:
                return None

            center = trough["position"]["center_of_mass"]
            bearing, rel_desc = self._calculate_bearing(tc_lat, tc_lon, center["lat"], center["lon"])
            distance = self._calculate_distance(tc_lat, tc_lon, center["lat"], center["lon"])

            if distance < 1000:
                influence = "直接影响台风路径和强度"
            elif distance < 2000:
                influence = "对台风有间接影响"
            else:
                influence = "影响较小"

            coords = self._get_system_coordinates(field_values, trough_threshold, "low", max_points=12)
            shape_info = {"description": "南北向延伸的槽线系统"}
            if coords:
                shape_info.update(
                    {
                        "coordinates": coords,
                        "extent_desc": f"纬度跨度{coords['span_deg'][1]:.1f}°，经度跨度{coords['span_deg'][0]:.1f}°",
                    }
                )

            desc = (
                f"在台风{rel_desc}约{distance:.0f}公里处存在西风槽系统，{influence}。"
            )
            if coords:
                desc += (
                    f" 槽线主体跨越纬度{coords['span_deg'][1]:.1f}°，经度{coords['span_deg'][0]:.1f}°。"
                )

            return {
                "system_name": "WesterlyTrough",
                "description": desc,
                "position": trough["position"],
                "intensity": trough["intensity"],
                "shape": shape_info,
                "properties": {
                    "distance_to_tc_km": round(distance, 0),
                    "bearing_from_tc": round(bearing, 1),
                    "influence": influence,
                },
            }
        except Exception as exc:
            print(f"⚠️ 西风槽提取失败: {exc}")
            return None

    def extract_frontal_system(self, time_idx, tc_lat, tc_lon):
        try:
            t850 = self._get_data_at_level("t", 850, time_idx)
            t700 = self._get_data_at_level("t", 700, time_idx)
            if t850 is None or t700 is None:
                return None

            if np.nanmean(t850) > 200:
                t850 = t850 - 273.15
            if np.nanmean(t700) > 200:
                t700 = t700 - 273.15

            temp_gradient = np.sqrt(
                (np.gradient(t850, axis=0) / self.lat_spacing) ** 2
                + (np.gradient(t850, axis=1) / self.lon_spacing) ** 2
            )

            threshold = np.percentile(temp_gradient, 75)
            frontal_mask = temp_gradient > threshold
            if not np.any(frontal_mask):
                return None

            coords = self._get_system_coordinates(temp_gradient, threshold, "high", max_points=20)
            coord_desc = self._generate_coordinate_description(coords, "锋面系统") if coords else ""

            desc = (
                f"在台风附近存在显著温度梯度的锋面系统，沿锋区温差明显，可能影响台风路径。{coord_desc}"
            )

            return {
                "system_name": "FrontalSystem",
                "description": desc,
                "position": {"description": "锋面大致位置", "lat": tc_lat, "lon": tc_lon},
                "intensity": {"description": "基于温度梯度的锋面强度"},
                "shape": {"description": "沿温度梯度形成的锋区", "coordinates": coords},
                "properties": {"temperature_gradient": round(float(threshold), 2)},
            }
        except Exception as exc:
            print(f"⚠️ 锋面系统提取失败: {exc}")
            return None

    def extract_monsoon_trough(self, time_idx, tc_lat, tc_lon):
        try:
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if u850 is None or v850 is None:
                return None

            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)
            du_dy = gy_u / (self.lat_spacing * 111000.0)
            dv_dx = gx_v / (self.lon_spacing * 111000.0 * self._coslat_safe[:, np.newaxis])
            relative_vorticity = dv_dx - du_dy

            monsoon_threshold = np.nanpercentile(relative_vorticity, 85)
            if monsoon_threshold <= 0:
                return None

            monsoon_mask = relative_vorticity > monsoon_threshold
            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            search_radius = 30
            sub = monsoon_mask[
                max(0, lat_idx - search_radius) : lat_idx + search_radius,
                max(0, lon_idx - search_radius) : lon_idx + search_radius,
            ]
            if not np.any(sub):
                return None

            finite_vort = relative_vorticity[monsoon_mask]
            finite_vort = finite_vort[np.isfinite(finite_vort)]
            if finite_vort.size == 0:
                return None
            max_vorticity = float(np.clip(np.max(finite_vort), 0, 2e-3)) * 1e5

            if max_vorticity > 10:
                level, impact = "活跃", "为台风发展提供有利环境"
            elif max_vorticity > 5:
                level, impact = "中等", "对台风发展有一定支持"
            else:
                level, impact = "弱", "对台风影响有限"

            desc = (
                f"台风周围存在活跃程度为'{level}'的季风槽系统，最大相对涡度为{max_vorticity:.1f}×10⁻⁵ s⁻¹，"
                f"{impact}。"
            )

            return {
                "system_name": "MonsoonTrough",
                "description": desc,
                "position": {"description": "台风周围的季风槽区域", "lat": tc_lat, "lon": tc_lon},
                "intensity": {
                    "value": round(max_vorticity, 1),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": level,
                },
                "shape": {"description": "东西向延伸的低压槽"},
                "properties": {"impact": impact, "vorticity_support": max_vorticity > 5},
            }
        except Exception as exc:
            print(f"⚠️ 季风槽提取失败: {exc}")
            return None

    def extract_low_level_flow(self, ds_at_time, tc_lat, tc_lon):
        """提取低层(10m)风场，保持与主流程相同的结构"""
        try:
            if "u10" not in ds_at_time.data_vars or "v10" not in ds_at_time.data_vars:
                return None

            u10 = float(ds_at_time.u10.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values)
            v10 = float(ds_at_time.v10.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values)

            mean_speed = float(np.sqrt(u10**2 + v10**2))
            mean_direction = (np.degrees(np.arctan2(u10, v10)) + 360.0) % 360.0

            if mean_speed >= 15:
                wind_level = "强"
            elif mean_speed >= 10:
                wind_level = "中等"
            else:
                wind_level = "弱"

            wind_desc, dir_text = self._bearing_to_desc(mean_direction)
            desc = (
                f"近地层存在{wind_level}低层风场，风速约{mean_speed:.1f}m/s，"
                f"主导风向为{wind_desc} (约{mean_direction:.0f}°)。"
            )

            return {
                "system_name": "LowLevelFlow",
                "description": desc,
                "position": {"lat": round(tc_lat, 2), "lon": round(tc_lon, 2)},
                "intensity": {
                    "speed": round(mean_speed, 2),
                    "direction_deg": round(mean_direction, 1),
                    "unit": "m/s",
                    "level": wind_level,
                    "vector": {"u": round(u10, 2), "v": round(v10, 2)},
                },
                "properties": {"direction_text": dir_text},
            }
        except Exception as e:
            print(f"⚠️ 提取低层风场失败: {e}")
            return None

    def extract_atmospheric_stability(self, ds_at_time, tc_lat, tc_lon):
        """提取大气稳定性，提供与其他系统一致的数据结构"""
        try:
            if "t2m" not in ds_at_time.data_vars:
                return None

            t2m = ds_at_time.t2m
            if np.nanmean(t2m.values) > 200:
                t2m = t2m - 273.15
            point_t2m = float(t2m.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values)
            if point_t2m > 28:
                stability = "不稳定"
            elif point_t2m > 24:
                stability = "中等"
            else:
                stability = "较稳定"

            desc = f"近地表温度约{point_t2m:.1f}°C，低层大气{stability}。"

            return {
                "system_name": "AtmosphericStability",
                "description": desc,
                "position": {"lat": round(tc_lat, 2), "lon": round(tc_lon, 2)},
                "intensity": {"surface_temp": round(point_t2m, 2), "unit": "°C"},
                "properties": {"stability_level": stability},
            }
        except Exception as e:
            print(f"⚠️ 提取大气稳定性失败: {e}")
            return None

    def extract_vertical_wind_shear(self, time_idx, tc_lat, tc_lon):
        """提取垂直风切变，复用extractSyst语义"""
        try:
            u200 = self._get_data_at_level("u", 200, time_idx)
            v200 = self._get_data_at_level("v", 200, time_idx)
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if any(x is None for x in (u200, v200, u850, v850)):
                return None

            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            shear_u = float(u200[lat_idx, lon_idx] - u850[lat_idx, lon_idx])
            shear_v = float(v200[lat_idx, lon_idx] - v850[lat_idx, lon_idx])
            shear_mag = float(np.sqrt(shear_u**2 + shear_v**2))

            if shear_mag < 5:
                level, impact = "弱", "非常有利于发展"
            elif shear_mag < 10:
                level, impact = "中等", "基本有利发展"
            else:
                level, impact = "强", "显著抑制发展"

            direction_from = (np.degrees(np.arctan2(shear_u, shear_v)) + 180.0) % 360.0
            wind_desc, dir_text = self._bearing_to_desc(direction_from)

            desc = (
                f"台风核心区正受到来自{wind_desc}方向、强度为“{level}”的垂直风切变影响，"
                f"当前风切变环境对台风的发展{impact.split(' ')[-1]}。"
            )

            vector_coords = self._get_vector_coords(tc_lat, tc_lon, shear_u, shear_v)

            return {
                "system_name": "VerticalWindShear",
                "description": desc,
                "position": {
                    "description": "在台风中心点计算的200-850hPa风矢量差",
                    "lat": round(tc_lat, 2),
                    "lon": round(tc_lon, 2),
                },
                "intensity": {"value": round(shear_mag, 2), "unit": "m/s", "level": level},
                "shape": {"description": f"来自{wind_desc}的切变矢量", "vector_coordinates": vector_coords},
                "properties": {
                    "direction_from_deg": round(direction_from, 1),
                    "impact": impact,
                    "shear_vector_mps": {"u": round(shear_u, 2), "v": round(shear_v, 2)},
                },
            }
        except Exception as e:
            print(f"⚠️ 提取垂直风切变失败: {e}")
            return None

    # 兼容旧名称
    def extract_vertical_shear(self, time_idx, tc_lat, tc_lon):
        return self.extract_vertical_wind_shear(time_idx, tc_lat, tc_lon)
            
    # 兼容旧接口的占位符，保留名称以避免潜在外部调用
    def _find_nearest_grid(self, lat, lon):
        return self._loc_idx(lat, lon)

    def _detect_completed_months(self):
        """扫描输出目录，返回已经生成结果的月份集合"""
        completed_months = set()
        if not self.output_dir.exists():
            return completed_months

        prefix = "cds_environment_analysis_"

        def _parse_month(candidate):
            if not candidate:
                return None
            try:
                return pd.Period(candidate, freq='M')
            except Exception:
                pass
            try:
                ts = pd.to_datetime(candidate)
                return pd.Period(ts, freq='M')
            except Exception:
                return None

        for json_path in sorted(self.output_dir.glob("*.json")):
            month_candidates = []
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                month_meta = metadata.get("month_processed")
                if month_meta:
                    month_candidates.append(month_meta)
            except Exception as exc:
                print(f"⚠️ 无法解析已有结果 {json_path}: {exc}")

            stem = json_path.stem
            if stem.startswith(prefix):
                month_candidates.append(stem[len(prefix):])

            for candidate in month_candidates:
                period = _parse_month(candidate)
                if period is not None:
                    completed_months.add(str(period))
                    break

        if completed_months:
            print(f"📁 检测到已有 {len(completed_months)} 个已完成的月份: {sorted(completed_months)}")
        else:
            print("📁 未检测到已完成的月份，开始全量处理。")
        return completed_months

    def _process_track_point(self, args):
        """Process a single track row. Extracted as a top-level method for multiprocessing pickling support."""
        idx, track_point = args

        try:
            time_point = track_point['time']
            tc_lat = float(track_point['lat'])
            tc_lon = float(track_point['lon'])

            total_points = getattr(self, "_current_month_total_points", None)
            if total_points is None:
                total_points = len(self.tracks_df)

            point_idx = track_point.get('time_idx', idx)
            if point_idx is None or (isinstance(point_idx, float) and math.isnan(point_idx)):
                point_idx = idx
            try:
                point_idx = int(point_idx)
            except (TypeError, ValueError):
                point_idx = int(idx) if isinstance(idx, (int, np.integer)) else 0

            print(
                "🔄 处理路径点 {}/{}: {}".format(
                    point_idx + 1,
                    total_points,
                    time_point.strftime('%Y-%m-%d %H:%M') if hasattr(time_point, 'strftime') else str(time_point),
                )
            )

            systems = self.extract_environmental_systems(time_point, tc_lat, tc_lon)

            return {
                "time": time_point.isoformat() if hasattr(time_point, 'isoformat') else str(time_point),
                "time_idx": point_idx,
                "tc_position": {"lat": tc_lat, "lon": tc_lon},
                "tc_intensity": {
                    "max_wind": track_point.get('max_wind_wmo', None),
                    "min_pressure": track_point.get('min_pressure_wmo', None),
                },
                "environmental_systems": systems,
            }
        except Exception as exc:
            print(f"⚠️ 处理单个路径点失败: {exc}")
            raise

    def process_all_tracks(self):
        """
        按月下载、处理、保存和清理数据，支持并行计算。
        """
        # 按年月分组
        self.tracks_df['year_month'] = self.tracks_df['time'].dt.to_period('M')
        unique_months = sorted(self.tracks_df['year_month'].unique())
        print(f"🗓️ 找到 {len(unique_months)} 个需要处理的月份: {[str(m) for m in unique_months]}")

        saved_files = []
        completed_months = self._detect_completed_months()
        
        for month in unique_months:
            month_key = str(month)
            if month_key in completed_months:
                print(f"⏭️ {month_key} 的结果已存在，跳过该月份。")
                continue

            print(f"\n{'='*25} 开始处理月份: {month} {'='*25}")
            month_tracks_df = self.tracks_df[self.tracks_df['year_month'] == month]
            start_date = month_tracks_df['time'].min().strftime('%Y-%m-%d')
            end_date = month_tracks_df['time'].max().strftime('%Y-%m-%d')
            print(f"📅 该月时间范围: {start_date} 到 {end_date}，共 {len(month_tracks_df)} 个路径点")
            
            single_file = self.download_era5_data(start_date, end_date)
            pressure_file = self.download_era5_pressure_data(start_date, end_date)
            
            if not single_file:
                print(f"❌ 无法获取 {month} 的单层数据，跳过此月份")
                continue

            if not self.load_era5_data(single_file, pressure_file):
                print(f"❌ 无法加载 {month} 的数据，跳过此月份")
                if self.cleanup_intermediate: self._cleanup_intermediate_files([single_file, pressure_file])
                continue

            # 并行或串行处理当前月份的路径点
            self._current_month_total_points = len(month_tracks_df)
            iterable = list(month_tracks_df.iterrows())
            
            if self.max_workers and self.max_workers > 1:
                print(f"⚙️ 使用 {self.max_workers} 个进程并行处理...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    processed_this_month = list(executor.map(self._process_track_point, iterable))
            else:
                print("⚙️ 使用串行模式处理...")
                processed_this_month = [self._process_track_point(item) for item in iterable]

            if hasattr(self, "_current_month_total_points"):
                delattr(self, "_current_month_total_points")

            # *** 新增：为当前月份创建并保存结果 ***
            if processed_this_month:
                monthly_results = {
                    "metadata": {
                        "extraction_time": datetime.now().isoformat(),
                        "tracks_file": str(self.tracks_file),
                        "total_points_in_month": len(processed_this_month),
                        "month_processed": str(month),
                        "data_source": "ERA5_reanalysis",
                        "processing_mode": "CDS_server_monthly_save"
                    },
                    "environmental_analysis": sorted(processed_this_month, key=lambda x: x['time_idx'])
                }
                
                monthly_output_file = self.output_dir / f"cds_environment_analysis_{month}.json"
                saved_path = self.save_results(monthly_results, output_file=monthly_output_file)
                if saved_path:
                    saved_files.append(saved_path)
            
            # 清理当月下载的文件
            if self.cleanup_intermediate:
                print(f"🧹 正在清理 {month} 的中间文件...")
                self._cleanup_intermediate_files([single_file, pressure_file])

        print(f"\n✅ 所有月份处理完毕。")
        return saved_files

    def save_results(self, results, output_file=None):
        """保存结果到JSON文件"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"cds_environment_analysis_{timestamp}.json"
        try:
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                if isinstance(obj, np.ndarray):
                    return convert_numpy(obj.tolist())
                if isinstance(obj, (np.float32, np.float64, np.float16)):
                    val = float(obj)
                    return None if not math.isfinite(val) else val
                if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                    return int(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return obj

            def sanitize(obj):
                if isinstance(obj, dict):
                    return {k: sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [sanitize(v) for v in obj]
                if isinstance(obj, float):
                    if math.isinf(obj) or math.isnan(obj):
                        return None
                    return obj
                return obj

            serializable = sanitize(convert_numpy(results))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"💾 结果已保存到: {output_file}")
            print(f"📊 文件大小: {Path(output_file).stat().st_size / 1024:.1f} KB")
            return str(output_file)
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            return None

    def _cleanup_intermediate_files(self, files_to_delete):
        """关闭数据集并删除指定的ERA5临时文件以释放磁盘空间"""
        try:
            if hasattr(self, 'ds') and self.ds is not None:
                self.ds.close()
                self.ds = None
            gc.collect()

            removed_count = 0
            for f in files_to_delete:
                if f and Path(f).exists():
                    try:
                        Path(f).unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"⚠️ 无法删除文件 {f}: {e}")
            if removed_count > 0:
                print(f"🧹 成功清理 {removed_count} 个中间数据文件。")
        except Exception as e:
            print(f"⚠️ 清理中间文件时发生错误: {e}")


def run_extraction(
    tracks_file: str = 'western_pacific_typhoons_superfast.csv',
    output_dir: str | Path = './cds_output',
    *,
    max_points: int | None = None,
    cleanup_intermediate: bool = True,
    workers: int | None = None,
) -> list[str]:
    """Helper to execute the extractor programmatically (e.g., inside notebooks)."""

    extractor = CDSEnvironmentExtractor(
        tracks_file,
        output_dir,
        cleanup_intermediate=cleanup_intermediate,
        max_workers=workers,
    )

    if max_points:
        print(f"🧪 测试模式: 仅处理前 {max_points} 个路径点")
        extractor.tracks_df = extractor.tracks_df.head(max_points)

    saved_file_list = extractor.process_all_tracks()
    if not saved_file_list:
        raise RuntimeError("❌ 处理失败，没有生成任何结果文件。")

    print(f"\n✅ 处理完成！共生成 {len(saved_file_list)} 个月度结果文件:")
    for file_path in saved_file_list:
        print(f"  -> {file_path}")

    return saved_file_list


def main(cli_args: list[str] | None = None) -> list[str]:
    """主函数，可同时用于命令行与Jupyter环境。"""
    import argparse

    parser = argparse.ArgumentParser(description='CDS服务器环境气象系统提取器')
    parser.add_argument('--tracks', default='western_pacific_typhoons_superfast.csv', help='台风路径CSV文件路径')
    parser.add_argument('--output', default='./cds_output', help='输出目录')
    parser.add_argument('--max-points', type=int, default=None, help='最大处理路径点数（用于测试）')
    parser.add_argument('--no-clean', action='store_true', help='保留中间ERA5数据文件')
    parser.add_argument('--workers', type=int, default=4, help='并行线程数（1表示禁用并行）')

    if cli_args is None:
        cli_args = [] if _running_in_notebook() else sys.argv[1:]

    args = parser.parse_args(cli_args)

    print("🌀 CDS环境气象系统提取器")
    print("=" * 50)
    print(f"📁 路径文件: {args.tracks}")
    print(f"📂 输出目录: {args.output}")

    try:
        return run_extraction(
            tracks_file=args.tracks,
            output_dir=args.output,
            max_points=args.max_points,
            cleanup_intermediate=not args.no_clean,
            workers=args.workers,
        )
    except RuntimeError as err:
        if _running_in_notebook():
            raise
        print(err)
        raise


if __name__ == "__main__":
    try:
        main()
    except RuntimeError:
        sys.exit(1)