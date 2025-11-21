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

        print(f"📥 下载ERA5单层数据: {start_date} 到 {end_date}")

        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 提取所有唯一的年、月、日
            years = sorted(list(set(date_range.year.astype(str))))
            months = sorted(list(set(date_range.month.astype(str).str.zfill(2))))
            days = sorted(list(set(date_range.day.astype(str).str.zfill(2))))
            
            print(f"   年份: {years}")
            print(f"   月份: {months}")
            print(f"   请求天数: {len(days)} 天")
            
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
                    'year': years,
                    'month': months,
                    'day': days,
                    'time': [
                        '00:00', '06:00', '12:00', '18:00'
                    ],
                },
                str(output_file)
            )

            print(f"✅ ERA5单层数据下载完成: {output_file}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        except Exception as e:
            print(f"❌ ERA5单层数据下载失败: {e}")
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
            
            # 提取所有唯一的年、月、日
            years = sorted(list(set(date_range.year.astype(str))))
            months = sorted(list(set(date_range.month.astype(str).str.zfill(2))))
            days = sorted(list(set(date_range.day.astype(str).str.zfill(2))))
            
            print(f"   年份: {years}")
            print(f"   月份: {months}")
            print(f"   请求天数: {len(days)} 天")
            
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
                    'year': years,
                    'month': months,
                    'day': days,
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
                # 移除 LowLevelFlow 和 AtmosphericStability，与 environment_extractor 对齐
                # lambda: self.extract_low_level_flow(ds_at_time, tc_lat, tc_lon),
                # lambda: self.extract_atmospheric_stability(ds_at_time, tc_lat, tc_lon),
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
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return 6371.0 * c

    def _create_circular_mask_haversine(self, center_lat, center_lon, radius_km):
        """创建基于Haversine距离的圆形掩码"""
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        lon_normalized = lon_grid.copy()
        lon_diff = lon_grid - center_lon
        lon_normalized = np.where(lon_diff > 180, lon_grid - 360, lon_grid)
        lon_normalized = np.where(lon_diff < -180, lon_grid + 360, lon_normalized)
        distances = self._haversine_distance(lat_grid, lon_normalized, center_lat, center_lon)
        return distances <= radius_km

    def _normalize_longitude(self, lon_array, center_lon):
        """将经度标准化到中心经度附近的连续区间"""
        lon_normalized = lon_array.copy()
        lon_diff = lon_array - center_lon
        lon_normalized = np.where(lon_diff > 180, lon_array - 360, lon_array)
        lon_normalized = np.where(lon_diff < -180, lon_array + 360, lon_normalized)
        return lon_normalized

    def _adaptive_boundary_sampling(self, coords, target_points=50, method="auto"):
        """自适应边界采样，兼容多种策略。"""
        if len(coords) <= target_points:
            return coords

        if method == "auto":
            perimeter_deg = self._calculate_perimeter(coords)
            if perimeter_deg < 50:
                method = "curvature"
            else:
                method = "douglas_peucker"

        if method == "curvature":
            return self._curvature_adaptive_sampling(coords, target_points)
        if method == "perimeter":
            return self._perimeter_proportional_sampling(coords, target_points)
        if method == "douglas_peucker":
            return self._douglas_peucker_sampling(coords, target_points)

        step = max(1, len(coords) // target_points)
        return coords[::step]

    def _calculate_perimeter(self, coords):
        if len(coords) < 2:
            return 0.0
        coords_array = np.array(coords)
        next_coords = np.roll(coords_array, -1, axis=0)
        deltas = next_coords - coords_array
        distances = np.sqrt(np.sum(deltas**2, axis=1))
        return float(np.sum(distances))

    def _perimeter_proportional_sampling(self, coords, target_points):
        if len(coords) < 2:
            return coords

        cumulative = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            cumulative.append(cumulative[-1] + math.hypot(dx, dy))

        total = cumulative[-1]
        if total < 1e-10:
            return [coords[0]]

        target_distances = np.linspace(0, total, target_points, endpoint=False)
        cumulative_arr = np.array(cumulative)
        sampled_coords = []
        for dist in target_distances:
            idx = int(np.argmin(np.abs(cumulative_arr - dist)))
            sampled_coords.append(coords[idx])
        return sampled_coords

    def _douglas_peucker_sampling(self, coords, target_points):
        if len(coords) <= target_points:
            return coords

        sampled = coords.copy()
        while len(sampled) > target_points:
            min_importance = float("inf")
            remove_idx = -1
            for i in range(1, len(sampled) - 1):
                importance = self._point_to_line_distance(
                    sampled[i], sampled[i - 1], sampled[i + 1]
                )
                if importance < min_importance:
                    min_importance = importance
                    remove_idx = i
            if remove_idx > 0:
                sampled.pop(remove_idx)
            else:
                break
        return sampled

    def _point_to_line_distance(self, point, line_start, line_end):
        p = np.array(point, dtype=float)
        a = np.array(line_start, dtype=float)
        b = np.array(line_end, dtype=float)
        ab = b - a
        ap = p - a
        if np.linalg.norm(ab) < 1e-10:
            return float(np.linalg.norm(ap))
        t = float(np.dot(ap, ab) / np.dot(ab, ab))
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))

    def _curvature_adaptive_sampling(self, coords, target_points):
        if len(coords) < 3:
            return coords

        coords_array = np.array(coords, dtype=float)
        p_prev = np.roll(coords_array, 1, axis=0)
        p_curr = coords_array
        p_next = np.roll(coords_array, -1, axis=0)

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        v3 = p_next - p_prev

        cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        norm_v3 = np.linalg.norm(v3, axis=1)

        denom = norm_v1 * norm_v2 * norm_v3
        curvatures = np.where(denom > 1e-10, cross / denom, 0.0)

        if curvatures.max() > 1e-10:
            weights = 0.5 + (curvatures / curvatures.max())
        else:
            weights = np.ones_like(curvatures)

        cum_weights = np.cumsum(weights)
        cum_weights = cum_weights / cum_weights[-1]
        target_weights = np.linspace(0, 1, target_points, endpoint=False)

        sampled_indices = []
        for tw in target_weights:
            idx = int(np.argmin(np.abs(cum_weights - tw)))
            if idx not in sampled_indices:
                sampled_indices.append(idx)

        sampled_indices = sorted(sampled_indices)
        return [coords[i] for i in sampled_indices]

    def _calculate_boundary_metrics(self, coords, tc_lat, tc_lon, method_used):
        """计算边界度量指标"""
        if not coords or len(coords) < 3:
            return {}

        coords_array = np.array(coords)
        next_coords = np.roll(coords_array, -1, axis=0)
        deltas = next_coords - coords_array
        
        distances_deg = np.sqrt(np.sum(deltas**2, axis=1))
        perimeter_deg = float(np.sum(distances_deg))
        
        avg_lat = np.mean([c[1] for c in coords])
        perimeter_km = perimeter_deg * 111.0 * math.cos(math.radians(avg_lat))
        
        first = coords[0]
        last = coords[-1]
        closure_dist = math.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)
        is_closed = closure_dist < 1.0

        return {
            "total_points": len(coords),
            "perimeter_km": round(perimeter_km, 1),
            "is_closed": is_closed,
            "closure_distance_deg": round(closure_dist, 3),
            "extraction_method": method_used or "unknown"
        }

    def _calculate_polygon_area_km2(self, coords):
        """计算多边形面积（平方公里）"""
        if not coords or len(coords) < 3:
            return 0.0
        
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        x_m = (lons - center_lon) * 111000 * np.cos(np.radians(center_lat))
        y_m = (lats - center_lat) * 111000
        
        area_m2 = 0.5 * abs(sum(x_m[i] * y_m[i + 1] - x_m[i + 1] * y_m[i] for i in range(len(x_m) - 1)))
        area_km2 = area_m2 / 1e6
        return round(float(area_km2), 1)

    def _extract_ocean_boundary_features(self, coords, tc_lat, tc_lon, threshold):
        if not coords or len(coords) < 3:
            return {}

        coords_array = np.array(coords, dtype=float)
        lons = coords_array[:, 0]
        lats = coords_array[:, 1]

        north_idx = int(np.argmax(lats))
        south_idx = int(np.argmin(lats))
        east_idx = int(np.argmax(lons))
        west_idx = int(np.argmin(lons))

        extreme_points = {
            "northernmost": {"lon": float(lons[north_idx]), "lat": float(lats[north_idx])},
            "southernmost": {"lon": float(lons[south_idx]), "lat": float(lats[south_idx])},
            "easternmost": {"lon": float(lons[east_idx]), "lat": float(lats[east_idx])},
            "westernmost": {"lon": float(lons[west_idx]), "lat": float(lats[west_idx])},
        }

        distances = [
            self._haversine_distance(tc_lat, tc_lon, lat_val, lon_val)
            for lon_val, lat_val in coords
        ]
        nearest_idx = int(np.argmin(distances))
        farthest_idx = int(np.argmax(distances))

        tc_relative_points = {
            "nearest_to_tc": {
                "lon": float(lons[nearest_idx]),
                "lat": float(lats[nearest_idx]),
                "distance_km": round(float(distances[nearest_idx]), 1),
                "description": "台风到暖水区边界的最短距离",
            },
            "farthest_from_tc": {
                "lon": float(lons[farthest_idx]),
                "lat": float(lats[farthest_idx]),
                "distance_km": round(float(distances[farthest_idx]), 1),
                "description": "暖水区延伸的最远点",
            },
        }

        curvature_extremes = []
        warm_eddy_centers = []
        cold_intrusion_points = []

        if len(coords) >= 5:
            curvatures = []
            for i in range(len(coords)):
                prev_idx = (i - 2) % len(coords)
                next_idx = (i + 2) % len(coords)
                p1 = np.array([lons[prev_idx], lats[prev_idx]])
                p2 = np.array([lons[i], lats[i]])
                p3 = np.array([lons[next_idx], lats[next_idx]])
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                curvature = 0.0
                if a * b * c > 1e-10:
                    curvature = 4 * area / (a * b * c)
                curvatures.append(curvature)

            curvatures = np.array(curvatures)
            high_threshold = np.percentile(curvatures, 90)
            indices = np.where(curvatures > high_threshold)[0]

            avg_dist = float(np.mean(distances)) if distances else 0.0
            for idx in indices[:5]:
                point_info = {
                    "lon": float(lons[idx]),
                    "lat": float(lats[idx]),
                    "curvature": round(float(curvatures[idx]), 6),
                }
                dist_to_tc = distances[idx]
                if avg_dist > 0:
                    if dist_to_tc > avg_dist * 1.1:
                        warm_eddy_centers.append(
                            {**point_info, "type": "warm_eddy", "description": "暖水区向外延伸的暖涡"}
                        )
                    elif dist_to_tc < avg_dist * 0.9:
                        cold_intrusion_points.append(
                            {
                                **point_info,
                                "type": "cold_intrusion",
                                "description": "冷水向暖水区侵入",
                            }
                        )
                curvature_extremes.append(point_info)

        return {
            "extreme_points": extreme_points,
            "warm_eddy_centers": warm_eddy_centers[:3],
            "cold_intrusion_points": cold_intrusion_points[:3],
            "curvature_extremes": curvature_extremes[:5],
            "tc_relative_points": tc_relative_points,
            "threshold_used": threshold,
        }

    def _extract_closed_ocean_boundary_with_features(
        self,
        sst,
        tc_lat,
        tc_lon,
        threshold=26.5,
        lat_range=20.0,
        lon_range=40.0,
        target_points=50,
    ):
        try:
            from skimage.measure import label as sk_label, find_contours as sk_find_contours

            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2

            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)

            local_sst = sst[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]

            if local_sst.size == 0:
                print("⚠️ 局部区域无SST数据")
                return None

            boundary_coords = None
            method_used = None

            try:
                mask = (local_sst >= threshold).astype(int)
                labeled = sk_label(mask, connectivity=2)
                if labeled.max() == 0:
                    raise ValueError("未找到暖水连通区域")

                tc_lat_idx = np.argmin(np.abs(local_lat - tc_lat))
                tc_lon_idx = np.argmin(np.abs(local_lon - tc_lon))
                target_label = labeled[tc_lat_idx, tc_lon_idx]

                if target_label == 0:
                    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
                    if unique.size > 0:
                        target_label = unique[np.argmax(counts)]

                contours = sk_find_contours((labeled == target_label).astype(float), 0.5)
                if contours:
                    main_contour = sorted(contours, key=len, reverse=True)[0]
                    boundary_coords = main_contour
                    method_used = "connected_component_labeling"
                    print(f"✅ 方法1成功: 连通区域标注提取到{len(main_contour)}个点")
            except Exception as exc:
                print(f"⚠️ 连通区域方法失败: {exc}，尝试方法2")

            if boundary_coords is None:
                try:
                    print("🔄 方法2: 扩大区域到30°x60°")
                    expanded = self._extract_closed_ocean_boundary_with_features(
                        sst,
                        tc_lat,
                        tc_lon,
                        threshold,
                        lat_range=30.0,
                        lon_range=60.0,
                        target_points=target_points,
                    )
                    if expanded:
                        expanded["boundary_metrics"]["method_note"] = "使用扩大区域(30x60)"
                        return expanded
                except Exception as exc:
                    print(f"⚠️ 扩大区域方法失败: {exc}，尝试方法3")

            if boundary_coords is None:
                try:
                    print("🔄 方法3: 使用原始find_contours方法")
                    contours = sk_find_contours(local_sst, threshold)
                    if contours:
                        boundary_coords = sorted(contours, key=len, reverse=True)[0]
                        method_used = "direct_contour_extraction"
                        print(f"✅ 方法3成功: 提取到{len(boundary_coords)}个点")
                except Exception as exc:
                    print(f"⚠️ 所有方法均失败: {exc}")
                    return None

            if boundary_coords is None or len(boundary_coords) == 0:
                return None

            geo_coords = []
            for point in boundary_coords:
                lat_idx = int(np.clip(point[0], 0, len(local_lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(local_lon) - 1))
                lat_val = float(local_lat[lat_idx])
                lon_val = float(local_lon[lon_idx])
                lon_normalized = self._normalize_longitude(np.array([lon_val]), tc_lon)[0]
                if lon_normalized < 0:
                    lon_normalized += 360
                geo_coords.append([lon_normalized, lat_val])

            sampled_coords = self._adaptive_boundary_sampling(
                geo_coords, target_points=target_points, method="curvature"
            )

            if len(sampled_coords) > 2:
                first = sampled_coords[0]
                last = sampled_coords[-1]
                closure_dist = math.hypot(last[0] - first[0], last[1] - first[1])
                if closure_dist > 1.0:
                    sampled_coords.append(first)
                    print(f"🔒 边界闭合: 添加首点，闭合距离从{closure_dist:.2f}°降至0")

            features = self._extract_ocean_boundary_features(sampled_coords, tc_lat, tc_lon, threshold)
            metrics = self._calculate_boundary_metrics(sampled_coords, tc_lat, tc_lon, method_used)
            metrics["warm_water_area_approx_km2"] = self._calculate_polygon_area_km2(sampled_coords)

            return {
                "boundary_coordinates": sampled_coords,
                "boundary_features": features,
                "boundary_metrics": metrics,
            }

        except Exception as exc:
            print(f"⚠️ OceanHeat闭合边界提取完全失败: {exc}")
            import traceback as _traceback  # noqa: WPS433

            _traceback.print_exc()
            return None

    def _extract_boundary_features(self, coords, tc_lat, tc_lon, threshold):
        if not coords or len(coords) < 4:
            return {}

        coords_array = np.array(coords, dtype=float)
        lons = coords_array[:, 0]
        lats = coords_array[:, 1]

        north_idx = int(np.argmax(lats))
        south_idx = int(np.argmin(lats))
        east_idx = int(np.argmax(lons))
        west_idx = int(np.argmin(lons))

        extreme_points = {
            "north": {"lon": round(float(lons[north_idx]), 2), "lat": round(float(lats[north_idx]), 2)},
            "south": {"lon": round(float(lons[south_idx]), 2), "lat": round(float(lats[south_idx]), 2)},
            "east": {"lon": round(float(lons[east_idx]), 2), "lat": round(float(lats[east_idx]), 2)},
            "west": {"lon": round(float(lons[west_idx]), 2), "lat": round(float(lats[west_idx]), 2)},
        }

        distances = [
            self._haversine_distance(tc_lat, tc_lon, lat_val, lon_val)
            for lon_val, lat_val in coords
        ]
        nearest_idx = int(np.argmin(distances))
        farthest_idx = int(np.argmax(distances))

        tc_relative_points = {
            "nearest": {
                "lon": round(float(lons[nearest_idx]), 2),
                "lat": round(float(lats[nearest_idx]), 2),
                "distance_km": round(float(distances[nearest_idx]), 1),
            },
            "farthest": {
                "lon": round(float(lons[farthest_idx]), 2),
                "lat": round(float(lats[farthest_idx]), 2),
                "distance_km": round(float(distances[farthest_idx]), 1),
            },
        }

        curvature_analysis = []
        if len(coords) >= 5:
            for i in range(len(coords)):
                prev_idx = (i - 1) % len(coords)
                next_idx = (i + 1) % len(coords)
                p1 = np.array([lons[prev_idx], lats[prev_idx]])
                p2 = np.array([lons[i], lats[i]])
                p3 = np.array([lons[next_idx], lats[next_idx]])
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                curvature = 0.0
                if a * b * c > 1e-10:
                    curvature = 4 * area / (a * b * c)
                curvature_analysis.append(
                    {"index": i, "lon": round(float(lons[i]), 2), "lat": round(float(lats[i]), 2), "curvature": round(float(curvature), 6)}
                )

        return {
            "extreme_points": extreme_points,
            "tc_relative_points": tc_relative_points,
            "curvature_analysis": curvature_analysis[:10],
            "threshold": threshold,
        }

    def _extract_closed_boundary_with_features(
        self,
        data_field,
        tc_lat,
        tc_lon,
        threshold,
        lat_range=20.0,
        lon_range=40.0,
        target_points=50,
    ):
        try:
            from skimage.measure import label as sk_label, find_contours as sk_find_contours

            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2

            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)

            local_field = data_field[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]

            if local_field.size == 0:
                print("⚠️ 局部区域无数据")
                return None

            boundary_coords = None
            method_used = None

            try:
                mask = (local_field >= threshold).astype(int)
                labeled = sk_label(mask, connectivity=2)
                if labeled.max() == 0:
                    raise ValueError("未找到连通区域")

                tc_lat_idx = np.argmin(np.abs(local_lat - tc_lat))
                tc_lon_idx = np.argmin(np.abs(local_lon - tc_lon))
                target_label = labeled[tc_lat_idx, tc_lon_idx]

                if target_label == 0:
                    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
                    if unique.size > 0:
                        target_label = unique[np.argmax(counts)]

                contours = sk_find_contours((labeled == target_label).astype(float), 0.5)
                if contours:
                    main_contour = sorted(contours, key=len, reverse=True)[0]
                    boundary_coords = main_contour
                    method_used = "connected_component_labeling"
            except Exception as exc:
                print(f"⚠️ 连通区域方法失败: {exc}，尝试方法2")

            if boundary_coords is None:
                try:
                    expanded = self._extract_closed_boundary_with_features(
                        data_field,
                        tc_lat,
                        tc_lon,
                        threshold,
                        lat_range=30.0,
                        lon_range=60.0,
                        target_points=target_points,
                    )
                    if expanded:
                        expanded["boundary_metrics"]["method_note"] = "使用扩大区域(30x60)"
                        return expanded
                except Exception as exc:
                    print(f"⚠️ 扩大区域方法失败: {exc}，尝试方法3")

            if boundary_coords is None:
                try:
                    contours = sk_find_contours(local_field, threshold)
                    if contours:
                        boundary_coords = sorted(contours, key=len, reverse=True)[0]
                        method_used = "direct_contour_extraction"
                except Exception as exc:
                    print(f"⚠️ 所有方法均失败: {exc}")
                    return None

            if boundary_coords is None or len(boundary_coords) == 0:
                return None

            geo_coords = []
            for point in boundary_coords:
                lat_idx = int(np.clip(point[0], 0, len(local_lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(local_lon) - 1))
                lat_val = float(local_lat[lat_idx])
                lon_val = float(local_lon[lon_idx])
                lon_normalized = self._normalize_longitude(np.array([lon_val]), tc_lon)[0]
                if lon_normalized < 0:
                    lon_normalized += 360
                geo_coords.append([lon_normalized, lat_val])

            sampled_coords = self._adaptive_boundary_sampling(geo_coords, target_points=target_points)

            if len(sampled_coords) > 2:
                first = sampled_coords[0]
                last = sampled_coords[-1]
                closure_dist = math.hypot(last[0] - first[0], last[1] - first[1])
                if closure_dist > 1.0:
                    sampled_coords.append(first)

            features = self._extract_boundary_features(sampled_coords, tc_lat, tc_lon, threshold)
            metrics = self._calculate_boundary_metrics(sampled_coords, tc_lat, tc_lon, method_used)

            return {
                "boundary_coordinates": sampled_coords,
                "boundary_features": features,
                "boundary_metrics": metrics,
            }
        except Exception as exc:
            print(f"⚠️ 闭合边界提取完全失败: {exc}")
            import traceback as _traceback  # noqa: WPS433

            _traceback.print_exc()
            return None

    def _get_contour_coords_local(self, data_field, level, lat_array, lon_array, center_lon, max_points=100):
        try:
            contours = find_contours(data_field, level)
            if not contours:
                return None
            main_contour = sorted(contours, key=len, reverse=True)[0]
            contour_lat_idx = np.clip(main_contour[:, 0].astype(int), 0, len(lat_array) - 1)
            contour_lon_idx = np.clip(main_contour[:, 1].astype(int), 0, len(lon_array) - 1)
            contour_lon = lon_array[contour_lon_idx]
            contour_lat = lat_array[contour_lat_idx]
            contour_lon_normalized = self._normalize_longitude(contour_lon, center_lon)
            step = max(1, len(main_contour) // max_points)
            coords = []
            for lon_val, lat_val in zip(contour_lon_normalized[::step], contour_lat[::step]):
                if lon_val < 0:
                    lon_val += 360
                coords.append([round(float(lon_val), 2), round(float(lat_val), 2)])
            return coords
        except Exception:
            return None

    def _extract_local_boundary_coords(
        self, data_field, tc_lat, tc_lon, threshold=5880, radius_deg=20.0, max_points=50
    ):
        try:
            lat_min = max(tc_lat - radius_deg, self.lat.min())
            lat_max = min(tc_lat + radius_deg, self.lat.max())
            lon_min = tc_lon - radius_deg
            lon_max = tc_lon + radius_deg

            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)

            local_field = data_field[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]

            return self._get_contour_coords_local(
                local_field, threshold, local_lat, local_lon, tc_lon, max_points
            )
        except Exception as exc:
            print(f"⚠️ 局部边界提取失败: {exc}")
            return None

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

    def _identify_subtropical_high_regional(self, z500, tc_lat, tc_lon, time_idx):  # noqa: ARG002
        try:
            lat_range = 20.0
            lon_range = 40.0

            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2

            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)

            region_z500 = z500[np.ix_(lat_mask, lon_mask)]
            if region_z500.size == 0:
                return None

            z500_mean = np.nanmean(region_z500)
            threshold_percentile = np.nanpercentile(region_z500, 75)
            threshold_std = z500_mean + np.nanstd(region_z500)
            dynamic_threshold = min(threshold_percentile, threshold_std)
            dynamic_threshold = max(dynamic_threshold, 5860)

            high_mask = region_z500 > dynamic_threshold
            if not np.any(high_mask):
                return None

            labeled_array, num_features = label(high_mask)
            if num_features == 0:
                return None

            max_area = 0
            best_idx = -1
            for label_idx in range(1, num_features + 1):
                feature_mask = labeled_array == label_idx
                area = np.sum(feature_mask)
                if area > max_area:
                    max_area = area
                    best_idx = label_idx

            if best_idx == -1:
                return None

            target_mask = labeled_array == best_idx
            com_y, com_x = center_of_mass(target_mask)

            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            pos_lat = float(local_lat[int(com_y)])
            pos_lon = float(local_lon[int(com_x)])
            intensity_val = float(np.nanmax(region_z500[target_mask]))

            return {
                "position": {
                    "center_of_mass": {
                        "lat": round(pos_lat, 2),
                        "lon": round(pos_lon, 2),
                    }
                },
                "intensity": {"value": round(intensity_val, 1), "unit": "gpm"},
                "shape": {},
                "extraction_info": {
                    "method": "regional_processing",
                    "region_extent": {
                        "lat_range": [float(lat_min), float(lat_max)],
                        "lon_range": [float(lon_min), float(lon_max)],
                    },
                    "dynamic_threshold": round(float(dynamic_threshold), 1),
                },
            }
        except Exception as exc:
            print(f"⚠️ 区域化副高识别失败: {exc}")
            return None

    def _calculate_steering_flow_layered(self, time_idx, tc_lat, tc_lon, radius_deg=5.0):
        try:
            levels = [850, 700, 500, 300]
            weights = [0.3, 0.3, 0.2, 0.2]

            u_weighted = 0.0
            v_weighted = 0.0
            total_weight = 0.0

            region_mask = self._create_region_mask(tc_lat, tc_lon, radius_deg)

            for level, weight in zip(levels, weights):
                u_level = self._get_data_at_level("u", level, time_idx)
                v_level = self._get_data_at_level("v", level, time_idx)
                if u_level is None or v_level is None:
                    continue
                u_mean = np.nanmean(np.where(region_mask, u_level, np.nan))
                v_mean = np.nanmean(np.where(region_mask, v_level, np.nan))
                if not (np.isfinite(u_mean) and np.isfinite(v_mean)):
                    continue
                u_weighted += weight * u_mean
                v_weighted += weight * v_mean
                total_weight += weight

            if total_weight == 0:
                return None

            u_steering = u_weighted / total_weight
            v_steering = v_weighted / total_weight
            speed = float(np.sqrt(u_steering**2 + v_steering**2))
            direction = (float(np.degrees(np.arctan2(u_steering, v_steering))) + 180.0) % 360.0

            return {
                "speed": speed,
                "direction": direction,
                "u": float(u_steering),
                "v": float(v_steering),
                "method": "layer_averaged_wind_850-300hPa",
            }
        except Exception as exc:
            print(f"⚠️ 层平均引导气流计算失败: {exc}")
            return None

    def _extract_ridge_line(self, z500, tc_lat, tc_lon, threshold=5880):
        try:
            contours = find_contours(z500, threshold)
            if not contours:
                return None

            main_contour = sorted(contours, key=len, reverse=True)[0]
            contour_lat_idx = np.clip(main_contour[:, 0].astype(int), 0, len(self.lat) - 1)
            contour_lon_idx = np.clip(main_contour[:, 1].astype(int), 0, len(self.lon) - 1)
            contour_lats = self.lat[contour_lat_idx]
            contour_lons = self.lon[contour_lon_idx]

            contour_lons_normalized = self._normalize_longitude(contour_lons, tc_lon)
            east_idx = int(np.argmax(contour_lons_normalized))
            west_idx = int(np.argmin(contour_lons_normalized))

            east_lat = float(contour_lats[east_idx])
            east_lon = float(contour_lons[east_idx])
            west_lat = float(contour_lats[west_idx])
            west_lon = float(contour_lons[west_idx])

            _, east_rel = self._calculate_bearing(tc_lat, tc_lon, east_lat, east_lon)
            _, west_rel = self._calculate_bearing(tc_lat, tc_lon, west_lat, west_lon)

            return {
                "east_end": {
                    "latitude": round(east_lat, 2),
                    "longitude": round(east_lon, 2),
                    "relative_position": east_rel,
                },
                "west_end": {
                    "latitude": round(west_lat, 2),
                    "longitude": round(west_lon, 2),
                    "relative_position": west_rel,
                },
                "threshold_gpm": threshold,
                "description": f"{threshold}gpm 等值线从{west_rel}延伸至{east_rel}",
            }
        except Exception as exc:
            print(f"⚠️ 脊线提取失败: {exc}")
            return None

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
        """提取副热带高压及引导气流，匹配 extractSyst 结构。"""
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            field_values = np.asarray(z500, dtype=float)
            # 不进行单位转换，保持与environment_extractor一致（使用geopotential原始值）
            # if np.nanmean(field_values) > 10000:
            #     field_values = field_values / 9.80665

            subtropical_high_obj = self._identify_subtropical_high_regional(field_values, tc_lat, tc_lon, time_idx)
            if not subtropical_high_obj:
                # 阈值调整: 5880 gpm * 9.80665 ≈ 57651 (geopotential)
                subtropical_high_obj = self._identify_pressure_system(field_values, tc_lat, tc_lon, "high", 57651)
                if not subtropical_high_obj:
                    return None

            enhanced_shape = self._get_enhanced_shape_info(field_values, 57651, "high", tc_lat, tc_lon)

            steering_result = self._calculate_steering_flow_layered(time_idx, tc_lat, tc_lon)
            if not steering_result:
                speed, direction, u_steering, v_steering = self._calculate_steering_flow(field_values, tc_lat, tc_lon)
                steering_result = {
                    "speed": speed,
                    "direction": direction,
                    "u": u_steering,
                    "v": v_steering,
                    "method": "geostrophic_wind",
                }

            ridge_info = self._extract_ridge_line(field_values, tc_lat, tc_lon)

            intensity_val = subtropical_high_obj["intensity"]["value"]
            # 阈值调整: 原来是gpm，现在是geopotential（约乘以9.8）
            if intensity_val > 57800:  # ~5900 gpm
                level = "强"
            elif intensity_val > 57651:  # ~5880 gpm
                level = "中等"
            else:
                level = "弱"
            subtropical_high_obj["intensity"]["level"] = level

            if enhanced_shape:
                shape_section = subtropical_high_obj.setdefault("shape", {})
                shape_section.update(
                    {
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                        "shape_type": enhanced_shape.get("shape_type"),
                        "orientation": enhanced_shape.get("orientation"),
                        "complexity": enhanced_shape.get("complexity"),
                    }
                )
                coord_info = enhanced_shape.get("coordinate_info")
                if coord_info:
                    shape_section["coordinate_details"] = coord_info

            extraction_info = subtropical_high_obj.get("extraction_info", {})
            dynamic_threshold = extraction_info.get("dynamic_threshold", 57651)  # 调整为geopotential值
            boundary_result = self._extract_closed_boundary_with_features(
                field_values,
                tc_lat,
                tc_lon,
                threshold=dynamic_threshold,
                lat_range=20.0,
                lon_range=40.0,
                target_points=50,
            )

            if boundary_result:
                subtropical_high_obj["boundary_coordinates"] = boundary_result["boundary_coordinates"]
                subtropical_high_obj["boundary_features"] = boundary_result["boundary_features"]
                subtropical_high_obj["boundary_metrics"] = boundary_result["boundary_metrics"]
                print(
                    f"✅ 边界提取成功: {boundary_result['boundary_metrics']['total_points']}点, "
                    f"{'闭合' if boundary_result['boundary_metrics']['is_closed'] else '开放'}, "
                    f"方法: {boundary_result['boundary_metrics']['extraction_method']}"
                )
            else:
                print("⚠️ 新方法失败，使用旧方法提取边界")
                fallback_coords = self._extract_local_boundary_coords(
                    field_values, tc_lat, tc_lon, threshold=dynamic_threshold, radius_deg=20.0
                )
                if fallback_coords:
                    subtropical_high_obj["boundary_coordinates"] = fallback_coords
                    subtropical_high_obj["boundary_note"] = "使用旧方法（新方法失败）"

            high_pos = subtropical_high_obj["position"]["center_of_mass"]
            bearing, rel_pos_desc = self._calculate_bearing(tc_lat, tc_lon, high_pos["lat"], high_pos["lon"])
            distance = self._calculate_distance(tc_lat, tc_lon, high_pos["lat"], high_pos["lon"])

            steering_speed = steering_result["speed"]
            steering_direction = steering_result["direction"]
            u_steering = steering_result["u"]
            v_steering = steering_result["v"]

            desc = (
                f"一个强度为“{level}”的副热带高压系统位于台风的{rel_pos_desc}，"
                f"其主体形态稳定，为台风提供了稳定的{steering_direction:.0f}°方向、"
                f"速度为{steering_speed:.1f} m/s的引导气流。"
            )

            # 更新position，只保留与environment_extractor一致的字段
            subtropical_high_obj["position"]["relative_to_tc"] = rel_pos_desc
            # 移除 description, distance_km, bearing_deg 以与 environment_extractor 对齐
            
            subtropical_high_obj.update(
                {
                    "system_name": "SubtropicalHigh",
                    "description": desc,
                    "properties": {
                        "influence": "主导台风未来路径",
                        "steering_flow": {
                            "speed_mps": round(steering_speed, 2),
                            "direction_deg": round(steering_direction, 1),
                            "vector_mps": {"u": round(u_steering, 2), "v": round(v_steering, 2)},
                            "calculation_method": steering_result.get("method", "unknown"),
                        },
                    },
                }
            )

            if ridge_info:
                subtropical_high_obj["properties"]["ridge_line"] = ridge_info

            return subtropical_high_obj
        except Exception as exc:
            print(f"⚠️ 引导系统提取失败: {exc}")
            return None

    # 兼容旧名称
    def extract_subtropical_high(self, time_idx, ds_at_time, tc_lat, tc_lon):
        return self.extract_steering_system(time_idx, tc_lat, tc_lon)

    def extract_ocean_heat_content(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        """提取海洋热含量及暖水边界信息，向 extractSyst 对齐。"""
        try:
            sst = self._get_sst_field(time_idx)
            if sst is None:
                return None

            radius_km = radius_deg * 111.0
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            if not np.any(circular_mask):
                return None

            sst_mean = float(np.nanmean(np.where(circular_mask, sst, np.nan)))
            if not np.isfinite(sst_mean):
                return None

            if sst_mean > 29:
                level, impact = "极高", "为爆发性增强提供顶级能量"
            elif sst_mean > 28:
                level, impact = "高", "非常有利于加强"
            elif sst_mean > 26.5:
                level, impact = "中等", "足以维持强度"
            else:
                level, impact = "低", "能量供应不足，将导致减弱"

            desc = (
                f"台风下方海域的平均海表温度为{sst_mean:.1f}°C，海洋热含量等级为“{level}”，"
                f"{impact}。"
            )
            desc_base = desc.rstrip("。")
            extra_notes = []

            shape_info = {
                "description": "26.5°C是台风发展的最低海温门槛，此线是生命线",
                "boundary_type": "closed_contour_with_features",
                "extraction_radius_deg": radius_deg * 3,
            }

            boundary_result = self._extract_closed_ocean_boundary_with_features(
                sst,
                tc_lat,
                tc_lon,
                threshold=26.5,
                lat_range=radius_deg * 6,
                lon_range=radius_deg * 12,
                target_points=50,
            )

            if boundary_result:
                shape_info["warm_water_boundary_26.5C"] = boundary_result["boundary_coordinates"]
                shape_info["boundary_features"] = boundary_result["boundary_features"]
                shape_info["boundary_metrics"] = boundary_result["boundary_metrics"]

                metrics = boundary_result["boundary_metrics"]
                if "warm_water_area_approx_km2" in metrics:
                    area_val = metrics["warm_water_area_approx_km2"]
                    shape_info["warm_water_area_km2"] = area_val
                    extra_notes.append(f"暖水区域面积约{area_val:.0f}km²")

                if metrics.get("is_closed"):
                    extra_notes.append(
                        f"边界完整闭合（{metrics['total_points']}个采样点，"
                        f"周长{metrics['perimeter_km']:.0f}km）"
                    )

                tc_features = boundary_result["boundary_features"].get("tc_relative_points", {})
                if "nearest_to_tc" in tc_features:
                    nearest = tc_features["nearest_to_tc"]
                    extra_notes.append(f"台风距暖水区边界最近{nearest['distance_km']:.0f}km")

                warm_eddies = boundary_result["boundary_features"].get("warm_eddy_centers", [])
                if warm_eddies:
                    extra_notes.append(f"检测到{len(warm_eddies)}个暖涡特征")
            else:
                print("⚠️ 闭合边界提取失败，回退到旧方法")
                lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
                radius_points = max(1, int(radius_deg * 3 / max(self.lat_spacing, 1e-6)))
                lat_start = max(0, lat_idx - radius_points)
                lat_end = min(len(self.lat), lat_idx + radius_points + 1)
                lon_start = max(0, lon_idx - radius_points)
                lon_end = min(len(self.lon), lon_idx + radius_points + 1)

                sst_local = sst[lat_start:lat_end, lon_start:lon_end]
                local_lat = self.lat[lat_start:lat_end]
                local_lon = self.lon[lon_start:lon_end]

                contour_26_5 = self._get_contour_coords_local(
                    sst_local, 26.5, local_lat, local_lon, tc_lon
                )
                shape_info["warm_water_boundary_26.5C"] = contour_26_5
                shape_info["boundary_type"] = "fallback_local_region"

                enhanced_shape = self._get_enhanced_shape_info(sst, 26.5, "high", tc_lat, tc_lon)
                if enhanced_shape:
                    shape_info.update(
                        {
                            "warm_water_area_km2": enhanced_shape["area_km2"],
                            "warm_region_shape": enhanced_shape.get("shape_type"),
                            "warm_region_orientation": enhanced_shape.get("orientation"),
                            "detailed_analysis": enhanced_shape["detailed_analysis"],
                        }
                    )
                    area_note = f"暖水区域面积约{enhanced_shape['area_km2']:.0f}km²"
                    shape_type = enhanced_shape.get("shape_type")
                    orientation = enhanced_shape.get("orientation")
                    if shape_type:
                        area_note += f"，呈{shape_type}"
                    if orientation:
                        area_note += f"，{orientation}"
                    extra_notes.append(area_note)

            if extra_notes:
                desc = desc_base + "。" + "，".join(extra_notes) + "。"
            else:
                desc = desc_base + "。"

            return {
                "system_name": "OceanHeatContent",
                "description": desc,
                "position": {
                    "description": f"台风中心周围{radius_deg}度半径内的海域",
                    "lat": round(tc_lat, 2),
                    "lon": round(tc_lon, 2),
                },
                "intensity": {"value": round(sst_mean, 2), "unit": "°C", "level": level},
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
                a = 6371000.0
                gy_u, gx_u = self._raw_gradients(u200)
                coslat = self._coslat_safe[:, np.newaxis]
                v_coslat = v200 * coslat
                gy_v_coslat, gx_v_coslat = self._raw_gradients(v_coslat)
                dlambda = np.deg2rad(self.lon_spacing)
                dphi = np.deg2rad(self.lat_spacing)
                du_dlambda = gx_u / dlambda
                dv_coslat_dphi = gy_v_coslat / dphi
                divergence = (du_dlambda / (a * coslat) + dv_coslat_dphi / a)

            if not np.any(np.isfinite(divergence)):
                return None
            divergence = np.where(np.isfinite(divergence), divergence, np.nan)

            radius_km = 500
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            divergence_masked = np.where(circular_mask, divergence, np.nan)
            div_val_raw = float(np.nanmean(divergence_masked))
            if not np.isfinite(div_val_raw):
                return None

            if np.all(np.isnan(divergence_masked)):
                return None

            max_div_idx = np.nanargmax(divergence_masked)
            max_lat_idx, max_lon_idx = np.unravel_index(max_div_idx, divergence_masked.shape)
            max_div_lat = float(self.lat[max_lat_idx])
            max_div_lon = float(self.lon[max_lon_idx])
            max_div_value = float(np.clip(divergence[max_lat_idx, max_lon_idx], -5e-4, 5e-4))

            distance_to_max = self._haversine_distance(tc_lat, tc_lon, max_div_lat, max_div_lon)

            def _bearing(lat1, lon1, lat2, lon2):
                lat1_rad = np.deg2rad(lat1)
                lat2_rad = np.deg2rad(lat2)
                dlon_rad = np.deg2rad(lon2 - lon1)
                x = np.sin(dlon_rad) * np.cos(lat2_rad)
                y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
                bearing = np.rad2deg(np.arctan2(x, y))
                return (bearing + 360) % 360

            bearing = _bearing(tc_lat, tc_lon, max_div_lat, max_div_lon)
            direction_names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
            direction_idx = int((bearing + 22.5) // 45) % 8
            direction = direction_names[direction_idx]

            div_val_raw = float(np.clip(div_val_raw, -5e-4, 5e-4))
            div_value = div_val_raw * 1e5
            max_div_value_scaled = max_div_value * 1e5

            if div_value > 5:
                level, impact = "强", "极其有利于台风发展和加强"
            elif div_value > 2:
                level, impact = "中等", "有利于台风维持和发展"
            elif div_value > -2:
                level, impact = "弱", "对台风发展影响较小"
            else:
                level, impact = "负值", "不利于台风发展"

            offset_note = ""
            if distance_to_max > 100:
                offset_note = (
                    f"最大辐散中心位于台风中心{direction}方向约{distance_to_max:.0f}公里处，"
                    f"强度为{max_div_value_scaled:.1f}×10⁻⁵ s⁻¹，"
                )
                if distance_to_max > 200:
                    offset_note += "辐散中心明显偏移可能影响台风的对称结构。"
                else:
                    offset_note += "辐散中心略有偏移。"

            desc = (
                f"台风中心周围500公里范围内200hPa高度的平均散度值为{div_value:.1f}×10⁻⁵ s⁻¹，"
                f"高空辐散强度为“{level}”，{impact}。"
            )
            if offset_note:
                desc += offset_note

            return {
                "system_name": "UpperLevelDivergence",
                "description": desc,
                "position": {
                    "description": f"台风中心周围{radius_km}公里范围内200hPa高度",
                    "center_lat": round(tc_lat, 2),
                    "center_lon": round(tc_lon, 2),
                    "radius_km": radius_km,
                },
                "intensity": {
                    "average_value": round(div_value, 2),
                    "max_value": round(max_div_value_scaled, 2),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": level,
                },
                "divergence_center": {
                    "lat": round(max_div_lat, 2),
                    "lon": round(max_div_lon, 2),
                    "distance_to_tc_km": round(distance_to_max, 1),
                    "direction": direction,
                    "bearing_deg": round(bearing, 1),
                },
                "shape": {"description": "高空辐散中心的空间分布"},
                "properties": {
                    "impact": impact,
                    "favorable_development": div_value > 0,
                    "center_offset": distance_to_max > 100,
                },
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

            # 使用与environment_extractor相同的算法：计算辐散->辐合
            a = 6371000
            lat_rad = np.deg2rad(self.lat)
            lon_rad = np.deg2rad(self.lon)

            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)

            dlat = self.lat_spacing * np.pi / 180
            dlon = self.lon_spacing * np.pi / 180
            cos_lat = self._coslat_safe[:, np.newaxis]

            du_dlon = gx_u / dlon
            dv_dlat = gy_v / dlat

            divergence = (1 / (a * cos_lat)) * du_dlon + (1 / a) * (dv_dlat * cos_lat - v850 * np.sin(lat_rad)[:, np.newaxis])

            # 根据半球选择搜索范围
            if tc_lat >= 0:
                lat_min, lat_max = 5, 20
                hemisphere = "北半球"
            else:
                lat_min, lat_max = -20, -5
                hemisphere = "南半球"

            tropical_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if not np.any(tropical_mask):
                return None

            convergence = -divergence
            tropical_conv = convergence[tropical_mask, :]
            conv_by_lat = np.nanmean(tropical_conv, axis=1)
            if not np.any(np.isfinite(conv_by_lat)):
                return None

            max_conv_idx = np.nanargmax(conv_by_lat)
            itcz_lat = self.lat[tropical_mask][max_conv_idx]
            max_convergence = conv_by_lat[max_conv_idx] * 1e5

            # 查找经度范围
            lat_idx = self._loc_idx(itcz_lat, tc_lon)[0]
            conv_at_lat = convergence[lat_idx, :]

            conv_threshold = (
                np.nanpercentile(conv_at_lat[conv_at_lat > 0], 50) if np.any(conv_at_lat > 0) else 0
            )
            strong_conv_mask = conv_at_lat > conv_threshold

            lon_ranges = []
            in_range = False
            start_lon = None
            for i, is_strong in enumerate(strong_conv_mask):
                if is_strong and not in_range:
                    start_lon = self.lon[i]
                    in_range = True
                elif not is_strong and in_range:
                    lon_ranges.append((start_lon, self.lon[i - 1]))
                    in_range = False
            if in_range:
                lon_ranges.append((start_lon, self.lon[-1]))

            best_range = None
            min_dist = float("inf")
            for lon_start, lon_end in lon_ranges:
                if lon_start <= tc_lon <= lon_end:
                    best_range = (lon_start, lon_end)
                    break
                dist = min(abs(tc_lon - lon_start), abs(tc_lon - lon_end))
                if dist < min_dist:
                    min_dist = dist
                    best_range = (lon_start, lon_end)

            distance_km = self._haversine_distance(tc_lat, tc_lon, itcz_lat, tc_lon)
            distance_deg = abs(tc_lat - itcz_lat)

            if distance_km < 500:
                influence = "直接影响台风发展"
                impact_level = "强"
            elif distance_km < 1000:
                influence = "对台风路径有显著影响"
                impact_level = "中"
            else:
                influence = "对台风影响较小"
                impact_level = "弱"

            if max_convergence > 5:
                conv_level = "强"
                conv_desc = "辐合活跃，有利于对流发展"
            elif max_convergence > 2:
                conv_level = "中等"
                conv_desc = "辐合中等，对对流有一定支持"
            else:
                conv_level = "弱"
                conv_desc = "辐合较弱"

            lon_range_str = f"{best_range[0]:.1f}°E-{best_range[1]:.1f}°E" if best_range else "跨经度带"

            desc = (
                f"{hemisphere}热带辐合带位于约{itcz_lat:.1f}°{'N' if itcz_lat >= 0 else 'S'}附近，"
                f"经度范围{lon_range_str}，"
                f"辐合强度{max_convergence:.2f}×10⁻⁵ s⁻¹（{conv_level}）。"
                f"与台风中心距离{distance_km:.0f}公里（{distance_deg:.1f}度），{influence}。"
            )

            result = {
                "system_name": "InterTropicalConvergenceZone",
                "description": desc,
                "position": {
                    "description": "热带辐合带中心位置",
                    "lat": round(itcz_lat, 2),
                    "lon": tc_lon,
                    "lon_range": lon_range_str,
                },
                "intensity": {
                    "value": round(max_convergence, 2),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": conv_level,
                    "description": conv_desc,
                },
                "shape": {"description": "东西向延伸的辐合带", "type": "convergence_line"},
                "properties": {
                    "distance_to_tc_km": round(distance_km, 1),
                    "distance_to_tc_deg": round(distance_deg, 2),
                    "influence": influence,
                    "impact_level": impact_level,
                    "hemisphere": hemisphere,
                    "convergence_strength": conv_level,
                },
            }

            if best_range:
                sample_lons = [best_range[0], (best_range[0] + best_range[1]) / 2, best_range[1]]
                boundary_coords = [[lon, itcz_lat] for lon in sample_lons]
                result["boundary_coordinates"] = boundary_coords

            return result

        except Exception as exc:
            print(f"⚠️ ITCZ 提取失败: {exc}")
            return None

    def extract_westerly_trough(self, time_idx, tc_lat, tc_lon):
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            field_values = np.asarray(z500, dtype=float)
            # 不进行单位转换，保持与environment_extractor一致（使用geopotential原始值）
            # if np.nanmean(field_values) > 10000:
            #     field_values = field_values / 9.80665

            # 使用与environment_extractor相同的算法：基于高度距平
            z500_zonal_mean = np.nanmean(field_values, axis=1, keepdims=True)
            z500_anomaly = field_values - z500_zonal_mean

            mid_lat_mask = (self.lat >= 20) & (self.lat <= 60)
            if not np.any(mid_lat_mask):
                return None

            z500_anomaly_mid = z500_anomaly.copy()
            z500_anomaly_mid[~mid_lat_mask, :] = np.nan

            negative_anomaly = z500_anomaly_mid < 0
            if not np.any(negative_anomaly):
                return None

            neg_values = z500_anomaly_mid[negative_anomaly]
            if len(neg_values) == 0:
                return None

            trough_threshold_anomaly = np.percentile(neg_values, 25)

            # 局部搜索
            search_radius_deg = 30
            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            radius_points = int(search_radius_deg / self.lat_spacing)
            lat_start = max(0, lat_idx - radius_points)
            lat_end = min(len(self.lat), lat_idx + radius_points + 1)
            lon_start = max(0, lon_idx - radius_points)
            lon_end = min(len(self.lon), lon_idx + radius_points + 1)

            local_mask = np.zeros_like(z500_anomaly, dtype=bool)
            local_mask[lat_start:lat_end, lon_start:lon_end] = True
            local_mask = local_mask & mid_lat_mask[:, np.newaxis]

            trough_mask = (z500_anomaly < trough_threshold_anomaly) & local_mask
            if not np.any(trough_mask):
                return None

            # 提取槽轴和槽底
            trough_axis = []
            trough_lons = []
            trough_lats = []

            lon_indices = np.where(np.any(trough_mask, axis=0))[0]
            if len(lon_indices) < 2:
                return None

            for lon_idx_local in lon_indices:
                col = z500_anomaly[:, lon_idx_local]
                col_mask = trough_mask[:, lon_idx_local]
                if not np.any(col_mask):
                    continue
                masked_col = np.where(col_mask, col, np.nan)
                if not np.any(np.isfinite(masked_col)):
                    continue
                min_lat_idx = np.nanargmin(masked_col)
                trough_lats.append(float(self.lat[min_lat_idx]))
                trough_lons.append(float(self.lon[lon_idx_local]))
                trough_axis.append([float(self.lon[lon_idx_local]), float(self.lat[min_lat_idx])])

            if len(trough_axis) < 2:
                return None

            # 找槽底（高度距平最小点）
            min_anomaly_idx = np.nanargmin(z500_anomaly[trough_mask])
            trough_mask_indices = np.where(trough_mask)
            trough_bottom_lat_idx = trough_mask_indices[0][min_anomaly_idx]
            trough_bottom_lon_idx = trough_mask_indices[1][min_anomaly_idx]

            trough_bottom_lat = float(self.lat[trough_bottom_lat_idx])
            trough_bottom_lon = float(self.lon[trough_bottom_lon_idx])
            trough_bottom_anomaly = float(z500_anomaly[trough_bottom_lat_idx, trough_bottom_lon_idx])

            trough_center_lat = np.mean(trough_lats)
            trough_center_lon = np.mean(trough_lons)

            bearing, rel_pos_desc = self._calculate_bearing(tc_lat, tc_lon, trough_center_lat, trough_center_lon)
            distance = self._calculate_distance(tc_lat, tc_lon, trough_center_lat, trough_center_lon)

            distance_bottom = self._haversine_distance(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)
            bearing_bottom, _ = self._calculate_bearing(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)

            trough_intensity = abs(trough_bottom_anomaly)

            trough_intensity = abs(trough_bottom_anomaly)

            if trough_intensity > 1500:  # geopotential单位，约150 gpm
                strength = "强"
            elif trough_intensity > 800:  # 约80 gpm
                strength = "中等"
            else:
                strength = "弱"

            if distance < 1000:
                if distance_bottom < 500:
                    influence = "槽前西南气流直接影响台风路径和强度，可能促进台风向东北方向移动"
                else:
                    influence = "直接影响台风路径和强度"
            elif distance < 2000:
                influence = "对台风有间接影响，可能通过引导气流影响台风移动"
            else:
                influence = "影响较小"

            desc = (
                f"在台风{rel_pos_desc}约{distance:.0f}公里处存在{strength}西风槽系统，"
                f"槽底位于({trough_bottom_lat:.1f}°N, {trough_bottom_lon:.1f}°E)，"
                f"距台风中心{distance_bottom:.0f}公里。"
            )

            desc += f"槽轴呈南北向延伸，跨越{len(trough_axis)}个采样点。"
            desc += influence + "。"

            shape_info = {
                "description": "南北向延伸的槽线系统",
                "trough_axis": trough_axis,
                "trough_bottom": [trough_bottom_lon, trough_bottom_lat],
                "axis_extent": {
                    "lat_range": [min(trough_lats), max(trough_lats)],
                    "lon_range": [min(trough_lons), max(trough_lons)],
                    "lat_span_deg": max(trough_lats) - min(trough_lats),
                    "lon_span_deg": max(trough_lons) - min(trough_lons),
                },
            }

            return {
                "system_name": "WesterlyTrough",
                "description": desc,
                "position": {
                    "description": "槽的质心位置（槽轴平均）",
                    "center_of_mass": {
                        "lat": round(trough_center_lat, 2),
                        "lon": round(trough_center_lon, 2),
                    },
                    "trough_bottom": {
                        "lat": round(trough_bottom_lat, 2),
                        "lon": round(trough_bottom_lon, 2),
                        "description": "槽底（高度距平最小点）",
                    },
                },
                "intensity": {
                    "value": round(trough_intensity, 1),
                    "unit": "gpm",
                    "description": "500hPa高度距平绝对值",
                    "level": strength,
                    "z500_anomaly_at_bottom": round(trough_bottom_anomaly, 1),
                },
                "shape": shape_info,
                "properties": {
                    "distance_to_tc_km": round(distance, 0),
                    "distance_bottom_to_tc_km": round(distance_bottom, 0),
                    "bearing_from_tc": round(bearing, 1),
                    "bearing_bottom_from_tc": round(bearing_bottom, 1),
                    "azimuth": f"台风{rel_pos_desc}",
                    "influence": influence,
                },
            }
        except Exception as exc:
            print(f"⚠️ 西风槽提取失败: {exc}")
            return None

    def extract_frontal_system(self, time_idx, tc_lat, tc_lon):
        try:
            t850 = self._get_data_at_level("t", 850, time_idx)
            t500 = self._get_data_at_level("t", 500, time_idx)
            t1000 = self._get_data_at_level("t", 1000, time_idx)
            u925 = self._get_data_at_level("u", 925, time_idx)
            v925 = self._get_data_at_level("v", 925, time_idx)

            if t850 is None or t500 is None:
                return None

            # 温度单位转换
            if np.nanmean(t850) > 200:
                t850 = t850 - 273.15
            if np.nanmean(t500) > 200:
                t500 = t500 - 273.15
            if t1000 is not None and np.nanmean(t1000) > 200:
                t1000 = t1000 - 273.15

            # 计算厚度
            if t1000 is not None:
                thickness = t1000 - t500
            else:
                thickness = t850 - t500

            # 使用与environment_extractor相同的综合frontal_index算法
            with np.errstate(divide="ignore", invalid="ignore"):
                gy_thick, gx_thick = self._raw_gradients(thickness)
                dthick_dy = gy_thick / (self.lat_spacing * 111000)
                dthick_dx = gx_thick / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
                thickness_gradient = np.sqrt(dthick_dx**2 + dthick_dy**2)

            with np.errstate(divide="ignore", invalid="ignore"):
                gy_t, gx_t = self._raw_gradients(t850)
                dt_dy = gy_t / (self.lat_spacing * 111000)
                dt_dx = gx_t / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
                temp_gradient = np.sqrt(dt_dx**2 + dt_dy**2)

            with np.errstate(divide="ignore", invalid="ignore"):
                gy_tgrad, gx_tgrad = self._raw_gradients(temp_gradient)
                frontogenesis = np.sqrt(gy_tgrad**2 + gx_tgrad**2)

            wind_convergence = None
            if u925 is not None and v925 is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    gy_v, gx_u = self._raw_gradients(v925)[0], self._raw_gradients(u925)[1]
                    du_dx = gx_u / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
                    dv_dy = gy_v / (self.lat_spacing * 111000)
                    wind_convergence = -(du_dx + dv_dy)

            # 清理NaN
            for field in [thickness_gradient, temp_gradient, frontogenesis]:
                if field is not None:
                    field[~np.isfinite(field)] = np.nan
            if wind_convergence is not None:
                wind_convergence[~np.isfinite(wind_convergence)] = np.nan

            # 归一化
            def normalize_field(field):
                if field is None or not np.any(np.isfinite(field)):
                    return np.zeros_like(thickness_gradient)
                valid = field[np.isfinite(field)]
                if len(valid) == 0:
                    return np.zeros_like(field)
                p5, p95 = np.percentile(valid, [5, 95])
                if p95 <= p5:
                    return np.zeros_like(field)
                return np.clip((field - p5) / (p95 - p5), 0, 1)

            norm_thickness_grad = normalize_field(thickness_gradient)
            norm_temp_grad = normalize_field(temp_gradient)
            norm_frontogenesis = normalize_field(frontogenesis)
            norm_convergence = (
                normalize_field(wind_convergence) if wind_convergence is not None else np.zeros_like(norm_thickness_grad)
            )

            # 综合frontal_index
            frontal_index = (
                0.5 * norm_thickness_grad + 0.25 * norm_temp_grad + 0.15 * norm_frontogenesis + 0.10 * norm_convergence
            )

            # 局部搜索
            search_radius_km = 1000
            search_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, search_radius_km)
            frontal_index_local = np.where(search_mask, frontal_index, np.nan)

            if not np.any(np.isfinite(frontal_index_local)):
                return None

            valid_values = frontal_index_local[np.isfinite(frontal_index_local)]
            if len(valid_values) < 10:
                return None

            front_threshold = np.percentile(valid_values, 85)
            front_mask = frontal_index_local > front_threshold
            if not np.any(front_mask):
                return None

            # 找锋面位置
            max_idx = np.unravel_index(np.nanargmax(frontal_index_local), frontal_index_local.shape)
            front_lat = self.lat[max_idx[0]]
            front_lon = self.lon[max_idx[1]]

            front_temp_gradient = temp_gradient[max_idx]
            if not np.isfinite(front_temp_gradient) or front_temp_gradient <= 0:
                return None
            front_temp_gradient = float(np.clip(front_temp_gradient, 0, 5e-4))

            if front_temp_gradient > 3e-5:
                level = "强"
            elif front_temp_gradient > 1.5e-5:
                level = "中等"
            else:
                level = "弱"

            distance_to_tc = self._haversine_distance(tc_lat, tc_lon, front_lat, front_lon)

            # 判断锋面类型
            front_type = "准静止锋"
            if max_idx[0] > 0 and max_idx[0] < len(self.lat) - 1:
                t_north = t850[max_idx[0] - 1, max_idx[1]]
                t_south = t850[max_idx[0] + 1, max_idx[1]]
                if np.isfinite(t_north) and np.isfinite(t_south):
                    if t_south > t_north + 2:
                        front_type = "冷锋"
                    elif t_north > t_south + 2:
                        front_type = "暖锋"

            strength_1e5 = front_temp_gradient * 1e5
            desc = (
                f"台风周围{distance_to_tc:.0f}km处存在{front_type}，强度为'{level}'，"
                f"温度梯度达到{strength_1e5:.1f}×10⁻⁵ °C/m。"
                f"锋面位于{front_lat:.2f}°N, {front_lon:.2f}°E，"
                f"可能影响台风的移动路径和强度变化。"
            )

            return {
                "system_name": "FrontalSystem",
                "description": desc,
                "position": {
                    "description": f"锋面位置（距台风中心{distance_to_tc:.0f}km）",
                    "lat": float(front_lat),
                    "lon": float(front_lon),
                },
                "intensity": {
                    "value": round(strength_1e5, 2),
                    "unit": "×10⁻⁵ °C/m",
                    "level": level,
                    "frontal_index": round(float(np.nanmax(frontal_index_local)), 3),
                },
                "shape": {"description": f"线性的{front_type}带，基于厚度场梯度识别", "type": front_type},
                "properties": {
                    "impact": "影响台风路径和结构",
                    "distance_to_tc_km": round(float(distance_to_tc), 1),
                    "front_type": front_type,
                    "search_radius_km": search_radius_km,
                },
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

            # 使用与environment_extractor相同的算法
            if tc_lat >= 0:
                lat_min, lat_max = 5, 25
                hemisphere = "北半球"
            else:
                lat_min, lat_max = -25, -5
                hemisphere = "南半球"

            search_radius_km = 1500
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)

            # 创建距离mask
            distance_mask = np.zeros((len(self.lat), len(self.lon)), dtype=bool)
            for i, lat in enumerate(self.lat):
                if not lat_mask[i]:
                    continue
                for j, lon in enumerate(self.lon):
                    dist = self._haversine_distance(tc_lat, tc_lon, lat, lon)
                    if dist <= search_radius_km:
                        distance_mask[i, j] = True

            if not np.any(distance_mask):
                return None

            # 计算涡度
            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)
            du_dy = gy_u / (self.lat_spacing * 111000.0)
            dv_dx = gx_v / (self.lon_spacing * 111000.0 * self._coslat_safe[:, np.newaxis])
            relative_vorticity = dv_dx - du_dy
            with np.errstate(invalid="ignore"):
                relative_vorticity[~np.isfinite(relative_vorticity)] = np.nan

            # 南半球取反
            if hemisphere == "南半球":
                relative_vorticity = -relative_vorticity

            masked_vort = np.where(distance_mask, relative_vorticity, np.nan)
            if not np.any(np.isfinite(masked_vort)):
                return None

            vort_threshold = np.nanpercentile(masked_vort[masked_vort > 0], 75) if np.any(masked_vort > 0) else 0
            if vort_threshold <= 0:
                return None

            trough_mask = masked_vort > vort_threshold
            if not np.any(trough_mask):
                return None

            # 找槽底（最大涡度点）
            max_vort_idx = np.unravel_index(np.nanargmax(masked_vort), masked_vort.shape)
            trough_bottom_lat = self.lat[max_vort_idx[0]]
            trough_bottom_lon = self.lon[max_vort_idx[1]]
            max_vorticity = masked_vort[max_vort_idx] * 1e5

            # 沿槽底纬度找槽轴
            trough_lat_idx = max_vort_idx[0]
            vort_along_axis = masked_vort[trough_lat_idx, :]
            axis_threshold = vort_threshold * 0.7
            axis_mask = vort_along_axis > axis_threshold

            axis_lons = self.lon[axis_mask]
            if len(axis_lons) > 0:
                axis_lon_start = axis_lons[0]
                axis_lon_end = axis_lons[-1]
                axis_length_deg = axis_lon_end - axis_lon_start
                axis_length_km = axis_length_deg * 111 * np.cos(np.deg2rad(trough_bottom_lat))
            else:
                axis_lon_start = trough_bottom_lon
                axis_lon_end = trough_bottom_lon
                axis_length_km = 0

            # 低层风场分析
            u_at_trough = u850[trough_lat_idx, :]
            mean_u = np.nanmean(u_at_trough[axis_mask]) if np.any(axis_mask) else 0

            if mean_u > 2:
                wind_pattern = "西风为主"
                monsoon_confidence = "高"
            elif mean_u > 0:
                wind_pattern = "弱西风"
                monsoon_confidence = "中"
            else:
                wind_pattern = "东风分量"
                monsoon_confidence = "低"

            distance_to_trough = self._haversine_distance(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)
            bearing, direction = self._calculate_bearing(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)

            if distance_to_trough < 500:
                influence = "台风位于季风槽内或紧邻，受水汽输送直接影响"
                impact_level = "强"
            elif distance_to_trough < 1000:
                influence = "台风受季风槽环流影响，水汽条件较好"
                impact_level = "中"
            else:
                influence = "季风槽对台风影响有限"
                impact_level = "弱"

            if max_vorticity > 10:
                vort_level = "强"
                vort_desc = "季风槽活跃，有利于台风发展"
            elif max_vorticity > 5:
                vort_level = "中等"
                vort_desc = "季风槽中等强度"
            else:
                vort_level = "弱"
                vort_desc = "季风槽较弱"

            desc = (
                f"在台风{direction}约{distance_to_trough:.0f}公里处检测到{hemisphere}季风槽，"
                f"槽底位于{trough_bottom_lat:.1f}°{'N' if trough_bottom_lat >= 0 else 'S'}, "
                f"{trough_bottom_lon:.1f}°E，"
                f"槽轴长度约{axis_length_km:.0f}公里，"
                f"最大涡度{max_vorticity:.1f}×10⁻⁵ s⁻¹（{vort_level}），"
                f"低层{wind_pattern}。{influence}。"
            )

            result = {
                "system_name": "MonsoonTrough",
                "description": desc,
                "position": {
                    "description": "季风槽槽底位置",
                    "lat": round(trough_bottom_lat, 2),
                    "lon": round(trough_bottom_lon, 2),
                },
                "intensity": {
                    "value": round(max_vorticity, 2),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": vort_level,
                    "description": vort_desc,
                },
                "shape": {
                    "description": f"东西向延伸的低压槽，长度约{axis_length_km:.0f}公里",
                    "type": "trough_axis",
                    "axis_length_km": round(axis_length_km, 1),
                },
                "properties": {
                    "distance_to_tc_km": round(distance_to_trough, 1),
                    "direction_from_tc": direction,
                    "bearing": round(bearing, 1),
                    "influence": influence,
                    "impact_level": impact_level,
                    "hemisphere": hemisphere,
                    "vorticity_level": vort_level,
                    "zonal_wind_pattern": wind_pattern,
                    "monsoon_confidence": monsoon_confidence,
                    "axis_lon_range": f"{axis_lon_start:.1f}°E - {axis_lon_end:.1f}°E",
                },
            }

            if axis_length_km > 0:
                boundary_coords = [
                    [axis_lon_start, trough_bottom_lat],
                    [trough_bottom_lon, trough_bottom_lat],
                    [axis_lon_end, trough_bottom_lat],
                ]
                result["boundary_coordinates"] = boundary_coords

            return result

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

    def extract_vertical_wind_shear(self, time_idx, tc_lat, tc_lon, radius_km=500):
        """提取垂直风切变，复用 extractSyst 语义并使用圆域平均。"""
        try:
            u200 = self._get_data_at_level("u", 200, time_idx)
            v200 = self._get_data_at_level("v", 200, time_idx)
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if any(x is None for x in (u200, v200, u850, v850)):
                return None

            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            if not np.any(circular_mask):
                return None

            u200_mean = float(np.nanmean(np.where(circular_mask, u200, np.nan)))
            v200_mean = float(np.nanmean(np.where(circular_mask, v200, np.nan)))
            u850_mean = float(np.nanmean(np.where(circular_mask, u850, np.nan)))
            v850_mean = float(np.nanmean(np.where(circular_mask, v850, np.nan)))

            shear_u = u200_mean - u850_mean
            shear_v = v200_mean - v850_mean
            shear_mag = float(np.sqrt(shear_u**2 + shear_v**2))

            if shear_mag < 5:
                level, impact = "弱", "非常有利于发展"
            elif shear_mag < 10:
                level, impact = "中等", "基本有利发展"
            else:
                level, impact = "强", "显著抑制发展"

            direction_from = np.degrees(np.arctan2(-shear_u, -shear_v)) % 360.0
            wind_desc, dir_text = self._bearing_to_desc(direction_from)

            desc = (
                f"台风中心{radius_km}公里范围内的垂直风切变来自{wind_desc}方向，"
                f"强度为“{level}”（{shear_mag:.1f} m/s），"
                f"当前风切变环境对台风的发展{impact}。"
            )

            vector_coords = self._get_vector_coords(tc_lat, tc_lon, shear_u, shear_v)

            return {
                "system_name": "VerticalWindShear",
                "description": desc,
                "position": {
                    "description": f"台风中心{radius_km}km圆域平均的200-850hPa风矢量差",
                    "lat": round(tc_lat, 2),
                    "lon": round(tc_lon, 2),
                    "radius_km": radius_km,
                },
                "intensity": {"value": round(shear_mag, 2), "unit": "m/s", "level": level},
                "shape": {"description": f"一个从{wind_desc}指向的切变矢量", "vector_coordinates": vector_coords},
                "properties": {
                    "direction_from_deg": round(direction_from, 1),
                    "impact": impact,
                    "shear_vector_mps": {"u": round(shear_u, 2), "v": round(shear_v, 2)},
                    "calculation_method": f"面积平均于{radius_km}km圆域",
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

    def download_all_data(self):
        """
        第一步：按年份下载所有需要的ERA5数据
        """
        # 获取整个数据集的时间范围
        start_date = self.tracks_df['time'].min()
        end_date = self.tracks_df['time'].max()
        
        print(f"🗓️ 数据时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 共 {len(self.tracks_df)} 个路径点")
        
        # 按年份分组下载
        years = sorted(list(set(self.tracks_df['time'].dt.year)))
        print(f"📅 将按年份下载，共 {len(years)} 年: {years}")
        
        downloaded_files = {
            'single_files': [],
            'pressure_files': []
        }
        
        for year in years:
            print(f"\n{'='*25} 下载 {year} 年数据 {'='*25}")
            
            # 获取该年的数据范围
            year_data = self.tracks_df[self.tracks_df['time'].dt.year == year]
            year_start = year_data['time'].min().strftime('%Y-%m-%d')
            year_end = year_data['time'].max().strftime('%Y-%m-%d')
            
            print(f"   时间范围: {year_start} 到 {year_end}")
            print(f"   路径点数: {len(year_data)}")
            
            # 下载该年的单层数据
            single_file = self.download_era5_data(year_start, year_end)
            if single_file:
                downloaded_files['single_files'].append(single_file)
            else:
                print(f"⚠️ {year} 年单层数据下载失败")
            
            # 下载该年的等压面数据
            pressure_file = self.download_era5_pressure_data(year_start, year_end)
            if pressure_file:
                downloaded_files['pressure_files'].append(pressure_file)
            else:
                print(f"⚠️ {year} 年等压面数据下载失败")
        
        if not downloaded_files['single_files']:
            raise RuntimeError("❌ 没有成功下载任何单层数据")
        
        print(f"\n✅ 所有数据下载完成！")
        print(f"   单层数据文件: {len(downloaded_files['single_files'])} 个")
        print(f"   等压面数据文件: {len(downloaded_files['pressure_files'])} 个")
        
        return downloaded_files

    def process_downloaded_data(self, data_info):
        """
        第二步：按月处理已下载的数据（支持多个年份文件）
        """
        single_files = data_info['single_files']
        pressure_files = data_info['pressure_files']
        
        print(f"\n{'='*60}")
        print("合并并加载数据...")
        print(f"{'='*60}")
        
        # 合并所有年份的数据文件
        import xarray as xr
        
        # 加载并合并单层数据
        print(f"📥 加载 {len(single_files)} 个单层数据文件...")
        ds_single_list = [xr.open_dataset(f) for f in single_files]
        ds_single_merged = xr.concat(ds_single_list, dim='time') if len(ds_single_list) > 1 else ds_single_list[0]
        
        # 加载并合并等压面数据
        if pressure_files:
            print(f"📥 加载 {len(pressure_files)} 个等压面数据文件...")
            ds_pressure_list = [xr.open_dataset(f) for f in pressure_files]
            ds_pressure_merged = xr.concat(ds_pressure_list, dim='time') if len(ds_pressure_list) > 1 else ds_pressure_list[0]
            self.ds = xr.merge([ds_single_merged, ds_pressure_merged])
        else:
            self.ds = ds_single_merged
        
        print(f"📊 ERA5数据加载完成: {dict(self.ds.dims)}")
        self._initialize_coordinate_metadata()
        
        print(f"\n{'='*60}")
        
        print("开始处理数据（按月保存结果）")
        print(f"{'='*60}")
        
        # 按年月分组进行处理和保存
        self.tracks_df['year_month'] = self.tracks_df['time'].dt.to_period('M')
        unique_months = sorted(self.tracks_df['year_month'].unique())
        print(f"🗓️ 将处理 {len(unique_months)} 个月份: {[str(m) for m in unique_months]}")
        
        saved_files = []
        completed_months = self._detect_completed_months()
        
        for idx, month in enumerate(unique_months, 1):
            month_key = str(month)
            if month_key in completed_months:
                print(f"⏭️ {month_key} 的结果已存在，跳过该月份。")
                continue
            
            print(f"\n{'='*25} 处理月份 [{idx}/{len(unique_months)}]: {month} {'='*25}")
            month_tracks_df = self.tracks_df[self.tracks_df['year_month'] == month]
            print(f"📊 该月共 {len(month_tracks_df)} 个路径点")
            
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

            # 为当前月份创建并保存结果
            if processed_this_month:
                monthly_results = {
                    "metadata": {
                        "extraction_time": datetime.now().isoformat(),
                        "tracks_file": str(self.tracks_file),
                        "total_points_in_month": len(processed_this_month),
                        "month_processed": str(month),
                        "data_source": "ERA5_reanalysis",
                        "processing_mode": "CDS_server_single_download_batch_process"
                    },
                    "environmental_analysis": sorted(processed_this_month, key=lambda x: x['time_idx'])
                }
                
                monthly_output_file = self.output_dir / f"cds_environment_analysis_{month}.json"
                saved_path = self.save_results(monthly_results, output_file=monthly_output_file)
                if saved_path:
                    saved_files.append(saved_path)

        print(f"\n✅ 所有月份处理完毕。")
        
        # 清理下载的文件
        if self.cleanup_intermediate:
            print(f"\n🧹 正在清理下载的数据文件...")
            all_files = single_files + pressure_files
            self._cleanup_intermediate_files(all_files)
        
        return saved_files

    def download_month_data(self, year, month):
        """
        下载指定年月的ERA5数据
        
        Args:
            year: 年份
            month: 月份 (1-12)
        
        Returns:
            (single_file, pressure_file) 文件路径元组
        """
        # 构建该月的日期范围
        month_start = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            month_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
        
        start_date = month_start.strftime('%Y-%m-%d')
        end_date = month_end.strftime('%Y-%m-%d')
        
        print(f"📥 下载 {year}-{month:02d} 数据: {start_date} 到 {end_date}")
        
        # 下载地面层数据
        single_file = self.download_era5_data(start_date, end_date)
        if not single_file:
            print(f"❌ {year}-{month:02d} 地面层数据下载失败")
            return None, None
        
        # 下载压力层数据（单月数据不需要分批）
        pressure_file = self.download_era5_pressure_data_single_month(start_date, end_date)
        if not pressure_file:
            print(f"❌ {year}-{month:02d} 压力层数据下载失败")
            return None, None
        
        return single_file, pressure_file
    
    def download_era5_pressure_data_single_month(self, start_date, end_date, levels=("850","500","200")):
        """
        下载单月ERA5等压面数据（不需要分批，单月数据量小）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            levels: 压力层列表
        
        Returns:
            文件路径或None
        """
        output_file = self.output_dir / f"era5_pressure_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"

        # 检查文件是否已存在
        if output_file.exists():
            print(f"   ✅ 压力层数据已存在，跳过下载: {output_file.name}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 提取所有唯一的年、月、日
            years = sorted(list(set(date_range.year.astype(str))))
            months = sorted(list(set(date_range.month.astype(str).str.zfill(2))))
            days = sorted(list(set(date_range.day.astype(str).str.zfill(2))))
            
            print(f"   📥 请求压力层数据: 年={years}, 月={months}, {len(days)}天")
            
            # 单月数据不需要分批，直接请求
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
                    'year': years,
                    'month': months,
                    'day': days,
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                },
                str(output_file)
            )

            print(f"   ✅ 压力层数据下载完成: {output_file.name}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        except Exception as e:
            print(f"   ❌ 压力层数据下载失败: {e}")
            return None

    def process_month_data(self, month_period, month_tracks_df):
        """
        处理单个月份的数据：下载 -> 加载 -> 分析 -> 保存 -> 删除
        
        Args:
            month_period: 月份Period对象 (例如: 2006-03)
            month_tracks_df: 该月份的路径数据
        
        Returns:
            生成的结果文件路径或None
        """
        year = month_period.year
        month = month_period.month
        
        print(f"\n{'='*70}")
        print(f"处理月份: {month_period} ({year}年{month}月)")
        print(f"{'='*70}")
        print(f"📊 该月共 {len(month_tracks_df)} 个路径点")
        
        # 第一步：下载该月数据
        print(f"\n{'='*25} 步骤1: 下载数据 {'='*25}")
        single_file, pressure_file = self.download_month_data(year, month)
        
        if not single_file or not pressure_file:
            print(f"❌ {month_period} 数据下载失败，跳过该月")
            return None
        
        # 第二步：加载数据
        print(f"\n{'='*25} 步骤2: 加载数据 {'='*25}")
        
        try:
            import xarray as xr
            
            chunks = self._parse_chunks_from_env()
            open_kwargs = {"chunks": chunks} if chunks else {}
            
            ds_single = xr.open_dataset(single_file, **open_kwargs)
            ds_pressure = xr.open_dataset(pressure_file, **open_kwargs)
            self.ds = xr.merge([ds_single, ds_pressure])
            
            print(f"   ✅ 数据加载完成: {dict(self.ds.dims)}")
            self._initialize_coordinate_metadata()
            
        except Exception as e:
            print(f"   ❌ 加载数据失败: {e}")
            return None
        
        # 第三步：处理数据
        print(f"\n{'='*25} 步骤3: 分析数据 {'='*25}")
        
        self._current_month_total_points = len(month_tracks_df)
        iterable = list(month_tracks_df.iterrows())
        
        if self.max_workers and self.max_workers > 1:
            print(f"   ⚙️ 使用 {self.max_workers} 个进程并行处理...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                processed_results = list(executor.map(self._process_track_point, iterable))
        else:
            print(f"   ⚙️ 使用串行模式处理...")
            processed_results = [self._process_track_point(item) for item in iterable]
        
        if hasattr(self, "_current_month_total_points"):
            delattr(self, "_current_month_total_points")
        
        # 第四步：保存结果
        print(f"\n{'='*25} 步骤4: 保存结果 {'='*25}")
        
        saved_path = None
        if processed_results:
            monthly_results = {
                "metadata": {
                    "extraction_time": datetime.now().isoformat(),
                    "tracks_file": str(self.tracks_file),
                    "total_points_in_month": len(processed_results),
                    "month_processed": str(month_period),
                    "year": year,
                    "month": month,
                    "data_source": "ERA5_reanalysis",
                    "processing_mode": "CDS_server_month_by_month_process"
                },
                "environmental_analysis": sorted(processed_results, key=lambda x: x['time_idx'])
            }
            
            monthly_output_file = self.output_dir / f"cds_environment_analysis_{month_period}.json"
            saved_path = self.save_results(monthly_results, output_file=monthly_output_file)
            
            if saved_path:
                print(f"   ✅ 结果已保存: {Path(saved_path).name}")
        
        # 第五步：清理该月数据文件
        print(f"\n{'='*25} 步骤5: 清理数据 {'='*25}")
        
        # 关闭数据集
        if hasattr(self, 'ds') and self.ds is not None:
            self.ds.close()
            self.ds = None
        
        # 删除该月的数据文件
        if self.cleanup_intermediate:
            files_to_clean = [single_file, pressure_file]
            for file_path in files_to_clean:
                if file_path and Path(file_path).exists():
                    try:
                        file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
                        Path(file_path).unlink()
                        print(f"   🧹 已删除: {Path(file_path).name} ({file_size:.1f} MB)")
                    except Exception as e:
                        print(f"   ⚠️ 删除失败 {Path(file_path).name}: {e}")
        
        gc.collect()  # 强制垃圾回收
        
        print(f"✅ {month_period} 月份处理完成")
        
        return saved_path

    def process_all_tracks(self):
        """
        逐月处理工作流程：
        对于每个月份：
            1. 下载该月数据（地面层 + 压力层）
            2. 加载并分析该月数据
            3. 保存结果JSON文件
            4. 删除该月数据文件（释放磁盘空间）
            5. 进入下一个月
        
        优势：
        - 磁盘占用最小（同时只保存1个月的数据）
        - 内存占用最小（只加载1个月的数据）
        - 支持断点续传（已完成的月份会自动跳过）
        """
        # 获取整个数据集的时间范围
        start_date = self.tracks_df['time'].min()
        end_date = self.tracks_df['time'].max()
        
        print(f"{'='*70}")
        print(f"CDS 逐月数据处理流程")
        print(f"{'='*70}")
        print(f"🗓️ 数据时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 总路径点数: {len(self.tracks_df)}")
        print(f"💡 处理策略: 逐月下载->处理->删除，最小化磁盘占用")
        
        # 按月份分组
        self.tracks_df['year_month'] = self.tracks_df['time'].dt.to_period('M')
        unique_months = sorted(self.tracks_df['year_month'].unique())
        print(f"📅 将逐月处理，共 {len(unique_months)} 个月份")
        print(f"   月份列表: {', '.join([str(m) for m in unique_months[:5]])}{'...' if len(unique_months) > 5 else ''}")
        
        # 检测已完成的月份
        completed_months = self._detect_completed_months()
        
        all_saved_files = []
        
        for month_idx, month_period in enumerate(unique_months, 1):
            month_key = str(month_period)
            
            # 检查是否已完成
            if month_key in completed_months:
                print(f"\n{'='*70}")
                print(f"月份进度: [{month_idx}/{len(unique_months)}] - {month_period}")
                print(f"{'='*70}")
                print(f"⏭️ {month_key} 的结果已存在，跳过该月份")
                print(f"📊 总进度: 已完成 {month_idx}/{len(unique_months)} 月")
                continue
            
            print(f"\n{'='*70}")
            print(f"月份进度: [{month_idx}/{len(unique_months)}] - {month_period}")
            print(f"{'='*70}")
            
            # 获取该月的路径数据
            month_tracks_df = self.tracks_df[self.tracks_df['year_month'] == month_period].copy()
            
            # 处理该月数据
            saved_path = self.process_month_data(month_period, month_tracks_df)
            
            if saved_path:
                all_saved_files.append(saved_path)
            
            print(f"\n📊 总进度: 已完成 {month_idx}/{len(unique_months)} 月，共生成 {len(all_saved_files)} 个结果文件")
        
        print(f"\n{'='*70}")
        print(f"✅ 所有月份处理完毕！")
        print(f"{'='*70}")
        print(f"📁 共生成 {len(all_saved_files)} 个月度结果文件")
        
        return all_saved_files

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
    tracks_file: str = 'matched_cyclone_tracks_2021onwards.csv',
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
    parser.add_argument('--tracks', default='matched_cyclone_tracks_2021onwards.csv', help='台风路径CSV文件路径')
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