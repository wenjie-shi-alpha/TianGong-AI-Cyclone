#!/usr/bin/env python3
"""
CDS服务器环境气象系统提取器
基于ERA5数据和台风路径文件，提取关键天气系统
专为CDS服务器环境优化
"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import cdsapi
import os
import sys
from pathlib import Path
import warnings
import concurrent.futures
import gc

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
            return True
        except Exception as e:
            print(f"❌ 加载ERA5数据失败: {e}")
            return False

    def extract_environmental_systems(self, time_point, tc_lat, tc_lon):
        """提取指定时间点的环境系统"""
        systems = {}
        try:
            # *** FIX: 使用 'valid_time' 坐标 ***
            era5_time = pd.to_datetime(self.ds.valid_time.sel(valid_time=time_point, method='nearest').values)
            print(f"🔍 处理时间点: {time_point} (使用ERA5时间: {era5_time})")
            
            # *** FIX: 使用 'valid_time' 坐标进行切片 ***
            ds_at_time = self.ds.sel(valid_time=time_point, method='nearest')

            systems['subtropical_high'] = self.extract_subtropical_high(ds_at_time, tc_lat, tc_lon)
            systems['ocean_heat'] = self.extract_ocean_heat(ds_at_time, tc_lat, tc_lon)
            systems['low_level_flow'] = self.extract_low_level_flow(ds_at_time, tc_lat, tc_lon)
            systems['atmospheric_stability'] = self.extract_atmospheric_stability(ds_at_time, tc_lat, tc_lon)
            systems['vertical_shear'] = self.extract_vertical_shear(ds_at_time, tc_lat, tc_lon)

        except Exception as e:
            print(f"⚠️ 提取环境系统失败: {e}")
            systems['error'] = str(e)
        return systems

    def extract_subtropical_high(self, ds_at_time, tc_lat, tc_lon):
        """提取副热带高压系统"""
        try:
            if 'z' in ds_at_time.data_vars and 'level' in ds_at_time.dims and 500 in ds_at_time.level.values:
                z500 = ds_at_time.sel(level=500)['z'] / 9.80665
                field = z500
                field_name = 'z500 (m)'
                high_by_height = True
            else:
                field = ds_at_time.msl / 100
                field_name = 'MSLP (hPa)'
                high_by_height = False

            # 在台风周围搜索高压系统
            # 搜索范围扩大到整个数据域，以确保捕捉到远距离的高压
            if high_by_height: # 位势高度找最大值
                max_val_idx = np.unravel_index(np.argmax(field.values), field.shape)
            else: # 海平面气压找最大值
                max_val_idx = np.unravel_index(np.argmax(field.values), field.shape)
            
            max_val = field.values[max_val_idx]
            high_lat = field.latitude.values[max_val_idx[0]]
            high_lon = field.longitude.values[max_val_idx[1]]

            bearing, distance = self._calculate_bearing_distance(tc_lat, tc_lon, high_lat, high_lon)

            if (not high_by_height) and max_val >= 1025: intensity_level = "强"
            elif (not high_by_height) and max_val >= 1020: intensity_level = "中等"
            else: intensity_level = "弱"

            return {
                "system_type": "SubtropicalHigh", "position": {"lat": float(high_lat), "lon": float(high_lon)},
                "intensity": {"value": float(max_val), "unit": "m" if high_by_height else "hPa", "level": intensity_level, "field": field_name},
                "relative_to_tc": {"bearing_deg": float(bearing), "distance_km": float(distance), "direction": self._bearing_to_direction(bearing)},
                "description": f"{intensity_level}副热带高压系统位于台风{self._bearing_to_direction(bearing)}方向{int(distance)}km处",
                "influence": self._assess_high_influence(distance, max_val if not high_by_height else 1020, bearing)
            }
        except Exception as e:
            print(f"⚠️ 提取副热带高压失败: {e}")
            return {"error": str(e)}

    def extract_ocean_heat(self, ds_at_time, tc_lat, tc_lon):
        """提取海洋热含量"""
        try:
            if 'sst' in ds_at_time.data_vars:
                temp_field = ds_at_time.sst
            elif 't2m' in ds_at_time.data_vars:
                temp_field = ds_at_time.t2m
            else:
                return {"error": "无温度数据"}
            
            if np.nanmean(temp_field.values) > 200:
                temp_field = temp_field - 273.15

            # 在台风位置插值获取温度
            point_temp = temp_field.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values
            
            if point_temp >= 29: heat_level, energy_support = "极高", "超级有利"
            elif point_temp >= 28: heat_level, energy_support = "高", "非常有利"
            elif point_temp >= 26.5: heat_level, energy_support = "中等", "基本有利"
            else: heat_level, energy_support = "低", "能量不足"

            return {
                "system_type": "OceanHeatContent", "position": {"lat": tc_lat, "lon": tc_lon},
                "intensity": {"value": float(point_temp), "unit": "°C", "level": heat_level},
                "properties": {"energy_support": energy_support, "suitable_for_development": bool(point_temp >= 26.5)},
                "description": f"海域温度{point_temp:.1f}°C，热含量等级{heat_level}，{energy_support}台风发展"
            }
        except Exception as e:
            print(f"⚠️ 提取海洋热含量失败: {e}")
            return {"error": str(e)}

    def extract_low_level_flow(self, ds_at_time, tc_lat, tc_lon):
        """提取低层风场"""
        try:
            u10 = ds_at_time.u10.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values
            v10 = ds_at_time.v10.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values

            mean_speed = np.sqrt(u10**2 + v10**2)
            mean_direction = (np.degrees(np.arctan2(u10, v10)) + 360) % 360
            
            if mean_speed >= 15: wind_level = "强"
            elif mean_speed >= 10: wind_level = "中等"
            else: wind_level = "弱"

            return {
                "system_type": "LowLevelFlow", "position": {"lat": tc_lat, "lon": tc_lon},
                "intensity": {"speed": float(mean_speed), "direction": float(mean_direction), "unit": "m/s", "level": wind_level, "vector": {"u": float(u10), "v": float(v10)}},
                "description": f"{wind_level}低层风场，风速{mean_speed:.1f}m/s，风向{mean_direction:.0f}°"
            }
        except Exception as e:
            print(f"⚠️ 提取低层风场失败: {e}")
            return {"error": str(e)}

    def extract_atmospheric_stability(self, ds_at_time, tc_lat, tc_lon):
        """提取大气稳定性"""
        try:
            if 't2m' in ds_at_time.data_vars:
                t2m = ds_at_time.t2m
                if np.nanmean(t2m.values) > 200:
                    t2m = t2m - 273.15
                
                point_t2m = t2m.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values
                stability = "中等" # 简化评估

                return {
                    "system_type": "AtmosphericStability", "position": {"lat": tc_lat, "lon": tc_lon},
                    "intensity": {"surface_temp": float(point_t2m), "unit": "°C"},
                    "properties": {"stability_level": stability},
                    "description": f"近地表温度{point_t2m:.1f}°C，大气稳定性{stability}"
                }
            else:
                return {"error": "无近地表温度数据"}
        except Exception as e:
            print(f"⚠️ 提取大气稳定性失败: {e}")
            return {"error": str(e)}

    def extract_vertical_shear(self, ds_at_time, tc_lat, tc_lon):
        """提取垂直风切变"""
        try:
            if 'u' in ds_at_time.data_vars and 'level' in ds_at_time.dims and 200 in ds_at_time.level.values and 850 in ds_at_time.level.values:
                u200 = ds_at_time.sel(level=200).u.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values
                v200 = ds_at_time.sel(level=200).v.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values
                u850 = ds_at_time.sel(level=850).u.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values
                v850 = ds_at_time.sel(level=850).v.interp(latitude=tc_lat, longitude=tc_lon, method="linear").values

                du, dv = u200 - u850, v200 - v850
                shear = float(np.sqrt(du**2 + dv**2))
                shear_dir = float((np.degrees(np.arctan2(du, dv)) + 360) % 360)

                if shear >= 20: level = "强"
                elif shear >= 12.5: level = "中等"
                else: level = "弱"

                return {
                    "system_type": "VerticalWindShear", "position": {"lat": tc_lat, "lon": tc_lon},
                    "intensity": {"magnitude": shear, "direction": shear_dir, "unit": "m/s", "level": level},
                    "description": f"垂直风切变 {shear:.1f} m/s ({level})，方向 {shear_dir:.0f}°"
                }
            else:
                return {"error": "未加载等压面风场，无法计算垂直风切变"}
        except Exception as e:
            print(f"⚠️ 提取垂直风切变失败: {e}")
            return {"error": str(e)}
            
    def _find_nearest_grid(self, lat, lon):
        lat_idx = np.abs(self.ds.latitude.values - lat).argmin()
        lon_idx = np.abs(self.ds.longitude.values - lon).argmin()
        return lat_idx, lon_idx

    def _calculate_bearing_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        bearing = np.degrees(np.arctan2(np.sin(dlon) * np.cos(lat2), np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)))
        return (bearing + 360) % 360, distance

    def _bearing_to_direction(self, bearing):
        directions = [ (22.5, "北"), (67.5, "东北"), (112.5, "东"), (157.5, "东南"), (202.5, "南"), (247.5, "西南"), (292.5, "西"), (337.5, "西北"), (360, "北")]
        for limit, direction in directions:
            if bearing < limit: return direction
        return "北"

    def _assess_high_influence(self, distance, pressure, bearing):
        score = 0
        if distance < 500: score += 3
        elif distance < 1000: score += 2
        elif distance < 1500: score += 1
        if pressure >= 1025: score += 2
        elif pressure >= 1020: score += 1
        if 225 <= bearing <= 315: score += 1
        if score >= 4: return "强影响"
        elif score >= 2: return "中等影响"
        else: return "弱影响"

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