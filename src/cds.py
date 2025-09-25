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

warnings.filterwarnings('ignore')

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
            print(f"❌ 加载路径数据失败: {e}")
            sys.exit(1)

    def download_era5_data(self, start_date, end_date, area=None):
        """
        从CDS下载ERA5数据

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            area: 区域 [north, west, south, east]
        """
        if area is None:
            # 基于路径数据确定区域
            lat_min = self.tracks_df['lat'].min() - 10
            lat_max = self.tracks_df['lat'].max() + 10
            lon_min = self.tracks_df['lon'].min() - 10
            lon_max = self.tracks_df['lon'].max() + 10
            area = [lat_max, lon_min, lat_min, lon_max]

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
                    'area': area,
                },
                str(output_file)
            )

            print(f"✅ ERA5数据下载完成: {output_file}")
            self._downloaded_files.append(str(output_file))
            return str(output_file)

        except Exception as e:
            print(f"❌ ERA5数据下载失败: {e}")
            return None

    def download_era5_pressure_data(self, start_date, end_date, area=None, levels=("850","500","200")):
        """从CDS下载ERA5等压面数据"""
        if area is None:
            lat_min = self.tracks_df['lat'].min() - 10
            lat_max = self.tracks_df['lat'].max() + 10
            lon_min = self.tracks_df['lon'].min() - 10
            lon_max = self.tracks_df['lon'].max() + 10
            area = [lat_max, lon_min, lat_min, lon_max]

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
                    'area': area,
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

        # 保留辅助索引函数
        def _loc_idx(lat_val: float, lon_val: float):
            lat_idx = int(np.abs(self.latitudes - lat_val).argmin())
            lon_idx = int(np.abs(self.longitudes - lon_val).argmin())
            return lat_idx, lon_idx

        self._loc_idx = _loc_idx

    def extract_environmental_systems(self, time_point, tc_lat, tc_lon):
        """提取指定时间点的环境系统，输出格式与environment_extractor保持一致"""
        systems = []
        try:
            time_idx, era5_time = self._get_time_index(time_point)
            print(f"🔍 处理时间点: {time_point} (ERA5时间: {era5_time})")

            ds_at_time = self._dataset_at_index(time_idx)

            system_extractors = [
                lambda: self.extract_subtropical_high(time_idx, ds_at_time, tc_lat, tc_lon),
                lambda: self.extract_vertical_shear(time_idx, tc_lat, tc_lon),
                lambda: self.extract_ocean_heat(time_idx, tc_lat, tc_lon),
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
        level_dim = next((dim for dim in ("level", "isobaricInhPa", "pressure") if dim in data.dims), None)
        if level_dim is None:
            field = data.isel(**indexers)
        else:
            if level_dim in data.coords:
                level_values = data[level_dim].values
            elif level_dim in self.ds.coords:
                level_values = self.ds[level_dim].values
            else:
                level_values = np.arange(data.sizes[level_dim])
            level_idx = int(np.abs(level_values - level_hpa).argmin())
            indexers[level_dim] = level_idx
            field = data.isel(**indexers)
        if 'expver' in field.dims:
            field = field.isel(expver=0)
        result = field.squeeze()
        return result.values if hasattr(result, 'values') else np.asarray(result)

    def _get_sst_field(self, time_idx):
        for var_name in ("sst", "ts"):
            field = self._get_field_at_time(var_name, time_idx)
            if field is not None:
                values = field.values if hasattr(field, 'values') else np.asarray(field)
                if np.nanmean(values) > 200:
                    values = values - 273.15
                return values
        for var_name in ("t2m", "t2"):
            field = self._get_field_at_time(var_name, time_idx)
            if field is not None:
                values = field.values if hasattr(field, 'values') else np.asarray(field)
                if np.nanmean(values) > 200:
                    values = values - 273.15
                return values
        return None

    def _create_region_mask(self, center_lat, center_lon, radius_deg):
        lat_mask = (self.latitudes >= center_lat - radius_deg) & (self.latitudes <= center_lat + radius_deg)
        lon_mask = self._lon_distance(self.longitudes, center_lon) <= radius_deg
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
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        return bearing

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
        index = int(round(bearing / 22.5)) % 16
        return wind_dirs[index], f"{dirs[index]}方向"

    def _get_vector_coords(self, lat, lon, u, v, scale=0.1):
        factor = scale * 0.009
        end_lat = lat + v * factor
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) < 1e-6:
            cos_lat = 1e-6
        end_lon = lon + u * factor / cos_lat
        return {
            "start": {"lat": round(lat, 2), "lon": round(lon, 2)},
            "end": {"lat": round(end_lat, 2), "lon": round(end_lon, 2)},
        }

    def extract_subtropical_high(self, time_idx, ds_at_time, tc_lat, tc_lon):
        """提取副热带高压系统，与environment_extractor输出语义保持一致"""
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            field_source = "z500"
            if z500 is not None:
                field_values = np.asarray(z500, dtype=float)
                if np.nanmean(field_values) > 10000:
                    field_values = field_values / 9.80665  # 转换为gpm
                threshold = 5880
                unit = "gpm"
            else:
                field_source = "msl"
                msl = self._get_field_at_time("msl", time_idx)
                if msl is None:
                    return None
                field_values = (msl.values if hasattr(msl, "values") else np.asarray(msl)) / 100.0
                threshold = 1020
                unit = "hPa"

            if not np.any(np.isfinite(field_values)):
                return None

            mask = np.isfinite(field_values)
            if field_source == "z500":
                mask &= field_values >= threshold
            else:
                mask &= field_values >= threshold

            if np.any(mask):
                candidate_idx = np.argwhere(mask)
                candidate_lats = self.latitudes[candidate_idx[:, 0]]
                candidate_lons = self.longitudes[candidate_idx[:, 1]]
                distances = self._haversine_distance(candidate_lats, candidate_lons, tc_lat, tc_lon)
                if np.all(np.isnan(distances)):
                    target_idx = np.unravel_index(np.nanargmax(field_values), field_values.shape)
                else:
                    best = int(np.nanargmin(distances))
                    target_idx = tuple(candidate_idx[best])
            else:
                target_idx = np.unravel_index(np.nanargmax(field_values), field_values.shape)

            high_lat = float(self.latitudes[target_idx[0]])
            high_lon = float(self.longitudes[target_idx[1]])
            intensity_val = float(field_values[target_idx])

            if field_source == "z500":
                if intensity_val > 5900:
                    level = "强"
                elif intensity_val > 5880:
                    level = "中等"
                else:
                    level = "弱"
            else:
                if intensity_val >= 1025:
                    level = "强"
                elif intensity_val >= 1020:
                    level = "中等"
                else:
                    level = "弱"

            bearing = self._calculate_bearing(tc_lat, tc_lon, high_lat, high_lon)
            _, rel_dir_text = self._bearing_to_desc(bearing)
            distance = self._calculate_distance(tc_lat, tc_lon, high_lat, high_lon)

            # 引导气流估算
            gy, gx = np.gradient(field_values)
            denom_lat = self.lat_spacing * 111000.0 if self.lat_spacing else 1.0
            denom_lon = self.lon_spacing * 111000.0 if self.lon_spacing else 1.0
            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            coslat_val = self._coslat_safe[lat_idx] if np.isfinite(self._coslat_safe[lat_idx]) else math.cos(math.radians(tc_lat))
            dx_val = gx[lat_idx, lon_idx] / (denom_lon * coslat_val if coslat_val else denom_lon)
            dy_val = gy[lat_idx, lon_idx] / denom_lat
            u_steering = -dx_val / (9.8 * 1e-5)
            v_steering = dy_val / (9.8 * 1e-5)
            steering_speed = float(np.sqrt(u_steering**2 + v_steering**2))
            steering_direction = (np.degrees(np.arctan2(u_steering, v_steering)) + 180.0) % 360.0
            wind_desc, steering_dir_text = self._bearing_to_desc(steering_direction)

            description = (
                f"一个强度为“{level}”的副热带高压系统位于台风的{rel_dir_text}，"
                f"核心强度约为{intensity_val:.0f}{unit}，为台风提供来自{wind_desc}、方向"
                f"约{steering_direction:.0f}°、速度{steering_speed:.1f}m/s的引导气流。"
            )

            shape_info = {
                "description": "基于阈值识别的高压控制区",
                "field": field_source,
                "threshold": threshold,
            }

            return {
                "system_name": "SubtropicalHigh",
                "description": description,
                "position": {
                    "description": "副热带高压中心",
                    "center_of_mass": {"lat": round(high_lat, 2), "lon": round(high_lon, 2)},
                    "relative_to_tc": {
                        "direction": rel_dir_text,
                        "bearing_deg": round(bearing, 1),
                        "distance_km": round(distance, 1),
                    },
                },
                "intensity": {"value": round(intensity_val, 1), "unit": unit, "level": level},
                "shape": shape_info,
                "properties": {
                    "influence": "主导台风未来路径",
                    "steering_flow": {
                        "speed_mps": round(steering_speed, 2),
                        "direction_deg": round(steering_direction, 1),
                        "vector_mps": {"u": round(float(u_steering), 2), "v": round(float(v_steering), 2)},
                        "wind_desc": wind_desc,
                    },
                },
            }
        except Exception as e:
            print(f"⚠️ 提取副热带高压失败: {e}")
            return None

    def extract_ocean_heat(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        """提取海洋热含量，与environment_extractor的热力判断保持一致"""
        try:
            sst_values = self._get_sst_field(time_idx)
            if sst_values is None:
                return None

            region_mask = self._create_region_mask(tc_lat, tc_lon, radius_deg)
            if not np.any(region_mask):
                return None

            warm_region = np.where(region_mask, sst_values, np.nan)
            mean_temp = float(np.nanmean(warm_region))
            if not np.isfinite(mean_temp):
                return None

            if mean_temp > 29:
                level, impact = "极高", "为爆发性增强提供顶级能量"
            elif mean_temp > 28:
                level, impact = "高", "非常有利于加强"
            elif mean_temp > 26.5:
                level, impact = "中等", "足以维持强度"
            else:
                level, impact = "低", "能量供应不足"

            desc = (
                f"台风下方半径约{radius_deg}°的海域平均海表温度为{mean_temp:.1f}°C，"
                f"海洋热含量等级为“{level}”，{impact}。"
            )

            cell_lat_km = self.lat_spacing * 111.0 if self.lat_spacing else 0.0
            cell_lon_km = (self.lon_spacing * 111.0 * math.cos(math.radians(tc_lat))) if self.lon_spacing else 0.0
            approx_area = float(region_mask.astype(float).sum() * abs(cell_lat_km) * abs(cell_lon_km))

            shape_info = {
                "description": "台风附近暖水覆盖区",
                "radius_deg": radius_deg,
            }
            if approx_area > 0:
                shape_info["approx_area_km2"] = round(approx_area, 0)

            return {
                "system_name": "OceanHeatContent",
                "description": desc,
                "position": {
                    "description": f"台风中心周围{radius_deg}°半径内的海域",
                    "lat": round(tc_lat, 2),
                    "lon": round(tc_lon, 2),
                },
                "intensity": {"value": round(mean_temp, 2), "unit": "°C", "level": level},
                "shape": shape_info,
                "properties": {"impact": impact, "warm_water_support": mean_temp > 26.5},
            }
        except Exception as e:
            print(f"⚠️ 提取海洋热含量失败: {e}")
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

    def extract_vertical_shear(self, time_idx, tc_lat, tc_lon):
        """提取垂直风切变，复用与environment_extractor一致的阈值和描述"""
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
            
    # 兼容旧接口的占位符，保留名称以避免潜在外部调用
    def _find_nearest_grid(self, lat, lon):
        return self._loc_idx(lat, lon)

    def process_all_tracks(self):
        """
        按月下载、处理、保存和清理数据，支持并行计算。
        """
        # 按年月分组
        self.tracks_df['year_month'] = self.tracks_df['time'].dt.to_period('M')
        unique_months = sorted(self.tracks_df['year_month'].unique())
        print(f"🗓️ 找到 {len(unique_months)} 个需要处理的月份: {[str(m) for m in unique_months]}")

        saved_files = []
        
        for month in unique_months:
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

            # 定义用于并行处理的单点处理函数
            def _process_row(args):
                idx, track_point = args
                time_point = track_point['time']
                tc_lat, tc_lon = track_point['lat'], track_point['lon']
                print(f"🔄 处理路径点 {track_point['time_idx']+1}/{len(self.tracks_df)}: {time_point.strftime('%Y-%m-%d %H:%M')}")
                systems = self.extract_environmental_systems(time_point, tc_lat, tc_lon)
                return {
                    "time": time_point.isoformat(), "time_idx": int(track_point['time_idx']),
                    "tc_position": {"lat": tc_lat, "lon": tc_lon},
                    "tc_intensity": {"max_wind": track_point.get('max_wind_wmo', None), "min_pressure": track_point.get('min_pressure_wmo', None)},
                    "environmental_systems": systems
                }

            # 并行或串行处理当前月份的路径点
            iterable = list(month_tracks_df.iterrows())
            processed_this_month = []
            if self.max_workers is None or self.max_workers > 1:
                workers = self.max_workers or min(8, (os.cpu_count() or 1) + 4)
                print(f"⚙️ 使用 {workers} 个工作线程进行并行处理...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    processed_this_month = list(ex.map(_process_row, iterable))
            else:
                print("⚙️ 正在进行串行处理...")
                processed_this_month = [_process_row(item) for item in iterable]

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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
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


def main():
    """主函数"""
    import argparse
    is_jupyter = 'ipykernel' in sys.modules

    if is_jupyter:
        print("🌀 检测到 Jupyter 环境，使用默认参数运行")
        args = type('Args', (), {
            'tracks': 'western_pacific_typhoons_superfast.csv',
            'output': './cds_output',
            'max_points': None,
            'no_clean': False,
            'workers': None
        })()
    else:
        parser = argparse.ArgumentParser(description='CDS服务器环境气象系统提取器')
        parser.add_argument('--tracks', default='western_pacific_typhoons_superfast.csv', help='台风路径CSV文件路径')
        parser.add_argument('--output', default='./cds_output', help='输出目录')
        parser.add_argument('--max-points', type=int, default=None, help='最大处理路径点数（用于测试）')
        parser.add_argument('--no-clean', action='store_true', help='保留中间ERA5数据文件')
        parser.add_argument('--workers', type=int, default=None, help='并行线程数（默认自动，1表示禁用并行）')
        args = parser.parse_args()

    print("🌀 CDS环境气象系统提取器")
    print("=" * 50)
    print(f"📁 路径文件: {args.tracks}")
    print(f"📂 输出目录: {args.output}")

    extractor = CDSEnvironmentExtractor(args.tracks, args.output, cleanup_intermediate=not args.no_clean, max_workers=args.workers)

    if args.max_points:
        print(f"🧪 测试模式: 仅处理前 {args.max_points} 个路径点")
        extractor.tracks_df = extractor.tracks_df.head(args.max_points)

    # process_all_tracks现在处理所有事情，包括保存
    saved_file_list = extractor.process_all_tracks()

    if saved_file_list:
        print(f"\n✅ 处理完成！共生成 {len(saved_file_list)} 个月度结果文件:")
        for file_path in saved_file_list:
            print(f"  -> {file_path}")
    else:
        print("\n❌ 处理失败，没有生成任何结果文件。")
        sys.exit(1)


if __name__ == "__main__":
    main()
