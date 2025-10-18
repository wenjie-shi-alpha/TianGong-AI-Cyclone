"""Core logic for extracting TC environmental systems."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

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


class TCEnvironmentalSystemsExtractor:
    """
    热带气旋环境场影响系统提取器
    """

    def __init__(self, forecast_data_path, tc_tracks_path):
        # ... (初始化代码与上一版相同) ...
        self.ds = xr.open_dataset(forecast_data_path)
        # 保存原始NC文件名(含/不含扩展)供输出命名使用
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

        # 预计算 cos(lat) 及其安全版本（避免极区除零放大）；不改变数值策略，仅提前计算
        self._coslat = np.cos(np.deg2rad(self.lat))
        self._coslat_safe = np.where(np.abs(self._coslat) < 1e-6, np.nan, self._coslat)

        # 梯度缓存：存储 (id(array) -> (grad_y_raw, grad_x_raw))，保持与 np.gradient(axis=0/1) 完全一致
        self._grad_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def _raw_gradients(arr: np.ndarray):
            key = id(arr)
            if key in self._grad_cache:
                return self._grad_cache[key]
            gy = np.gradient(arr, axis=0)
            gx = np.gradient(arr, axis=1)
            self._grad_cache[key] = (gy, gx)
            return gy, gx
        self._raw_gradients = _raw_gradients  # 绑定实例

        # 经纬度索引辅助：功能等价于原多次 argmin 调用
        def _loc_idx(lat_val: float, lon_val: float):
            return (np.abs(self.lat - lat_val).argmin(), np.abs(self.lon - lon_val).argmin())
        self._loc_idx = _loc_idx

        # 初始化形状分析器
        self.shape_analyzer = WeatherSystemShapeAnalyzer(self.lat, self.lon)

        self.tc_tracks = pd.read_csv(tc_tracks_path)
        self.tc_tracks["time"] = pd.to_datetime(self.tc_tracks["time"])

        print(f"📊 加载{len(self.tc_tracks)}个热带气旋路径点")
        print(
            f"🌍 区域范围: {self.lat.min():.1f}°-{self.lat.max():.1f}°N, {self.lon.min():.1f}°-{self.lon.max():.1f}°E"
        )
        print(f"🔍 增强形状分析功能已启用")

    def close(self) -> None:
        """Release the underlying dataset handle so NC files can be deleted promptly."""

        dataset = getattr(self, "ds", None)
        if dataset is not None:
            try:
                dataset.close()
            except Exception:
                pass
            finally:
                self.ds = None

    def __enter__(self) -> "TCEnvironmentalSystemsExtractor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # --- 工具函数：距离计算和掩膜生成 ---

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        使用Haversine公式计算两点间的球面距离（单位：公里）
        """
        R = 6371.0  # 地球半径，公里
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
        """
        基于Haversine距离创建圆形掩膜，正确处理跨越日期变更线的情况
        
        Args:
            center_lat: 中心纬度
            center_lon: 中心经度
            radius_km: 半径（公里）
        
        Returns:
            掩膜数组（True表示在圆内）
        """
        # 创建经纬度网格
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        
        # 处理经度跨越日期变更线的情况
        # 将所有经度归一化到以center_lon为中心的[-180, 180]范围
        lon_normalized = lon_grid.copy()
        lon_diff = lon_grid - center_lon
        lon_normalized = np.where(lon_diff > 180, lon_grid - 360, lon_grid)
        lon_normalized = np.where(lon_diff < -180, lon_grid + 360, lon_normalized)
        
        # 计算距离
        distances = self._haversine_distance(lat_grid, lon_normalized, center_lat, center_lon)
        
        return distances <= radius_km

    def _normalize_longitude(self, lon_array, center_lon):
        """
        将经度数组归一化到以center_lon为中心的连续范围
        处理跨越0°/360°经线的情况
        
        Args:
            lon_array: 经度数组
            center_lon: 中心经度
        
        Returns:
            归一化后的经度数组
        """
        lon_normalized = lon_array.copy()
        lon_diff = lon_array - center_lon
        
        # 将超过180度的差值调整到[-180, 180]范围
        lon_normalized = np.where(lon_diff > 180, lon_array - 360, lon_array)
        lon_normalized = np.where(lon_diff < -180, lon_array + 360, lon_normalized)
        
        return lon_normalized

    # --- 核心系统提取函数 (深度重构) ---

    def extract_steering_system(self, time_idx, tc_lat, tc_lon):
        """
        [深度重构] 提取并解译引导气流和副热带高压系统。
        """
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            # 1. 识别副高系统
            subtropical_high_obj = self._identify_pressure_system(
                z500, tc_lat, tc_lon, "high", 5880
            )
            if not subtropical_high_obj:
                return None

            # 2. 增强形状分析
            enhanced_shape = self._get_enhanced_shape_info(z500, 5880, "high", tc_lat, tc_lon)

            # 3. 计算引导气流
            steering_speed, steering_direction, u_steering, v_steering = (
                self._calculate_steering_flow(z500, tc_lat, tc_lon)
            )

            # 4. 丰富化描述和属性
            # 4.1 强度定性分级
            intensity_val = subtropical_high_obj["intensity"]["value"]
            if intensity_val > 5900:
                level = "强"
            elif intensity_val > 5880:
                level = "中等"
            else:
                level = "弱"
            subtropical_high_obj["intensity"]["level"] = level

            # 4.2 更新形状信息
            if enhanced_shape:
                subtropical_high_obj["shape"].update(
                    {
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                        "area_km2": enhanced_shape["area_km2"],
                        "shape_type": enhanced_shape["shape_type"],
                        "orientation": enhanced_shape["orientation"],
                        "complexity": enhanced_shape["complexity"],
                    }
                )

                # 添加坐标信息
                if "coordinate_info" in enhanced_shape:
                    subtropical_high_obj["shape"]["coordinate_details"] = enhanced_shape[
                        "coordinate_info"
                    ]

            # 4.3 提取关键坐标点
            system_coords = self._get_system_coordinates(z500, 5880, "high", max_points=15)
            if system_coords:
                subtropical_high_obj["shape"]["coordinates"] = system_coords

            # 4.4 传统等值线坐标（保持兼容性）
            contour_coords = self._get_contour_coords(z500, 5880, tc_lon)
            if contour_coords:
                subtropical_high_obj["shape"]["contour_5880gpm"] = contour_coords
                if not enhanced_shape:
                    subtropical_high_obj["shape"]["description"] = "呈东西向伸展的脊线形态"

            # 4.4 相对位置和综合描述
            high_pos = subtropical_high_obj["position"]["center_of_mass"]
            bearing, rel_pos_desc = self._calculate_bearing(
                tc_lat, tc_lon, high_pos["lat"], high_pos["lon"]
            )
            subtropical_high_obj["position"]["relative_to_tc"] = rel_pos_desc

            desc = (
                f"一个强度为“{level}”的副热带高压系统位于台风的{rel_pos_desc}，"
                f"其主体形态稳定，为台风提供了稳定的{steering_direction:.0f}°方向、"
                f"速度为{steering_speed:.1f} m/s的引导气流。"
            )

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
                        },
                    },
                }
            )
            return subtropical_high_obj
        except Exception as e:
            # print(f"⚠️ 引导系统提取失败: {e}")
            return None

    def extract_vertical_wind_shear(self, time_idx, tc_lat, tc_lon, radius_km=500):
        """
        [深度重构] 提取并解译垂直风切变。
        使用台风中心500km圆域内的面积平均计算200-850hPa风矢量差。
        
        Parameters:
            time_idx: 时间索引
            tc_lat: 台风中心纬度
            tc_lon: 台风中心经度
            radius_km: 计算半径（公里），默认500km
        """
        try:
            u200, v200 = self._get_data_at_level("u", 200, time_idx), self._get_data_at_level(
                "v", 200, time_idx
            )
            u850, v850 = self._get_data_at_level("u", 850, time_idx), self._get_data_at_level(
                "v", 850, time_idx
            )
            if any(x is None for x in [u200, v200, u850, v850]):
                return None

            # 使用Haversine距离创建500km圆形掩膜
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            
            # 在圆域内计算各层风场的面积平均
            u200_mean = np.nanmean(u200[circular_mask])
            v200_mean = np.nanmean(v200[circular_mask])
            u850_mean = np.nanmean(u850[circular_mask])
            v850_mean = np.nanmean(v850[circular_mask])
            
            # 计算矢量差（先平均后相减）
            shear_u = u200_mean - u850_mean
            shear_v = v200_mean - v850_mean
            shear_mag = np.sqrt(shear_u**2 + shear_v**2)

            if shear_mag < 5:
                level, impact = "弱", "非常有利于发展"
            elif shear_mag < 10:
                level, impact = "中等", "基本有利发展"
            else:
                level, impact = "强", "显著抑制发展"

            # 方向定义为风从哪个方向来（修正公式：使用 atan2(-u, -v)）
            direction_from = np.degrees(np.arctan2(-shear_u, -shear_v)) % 360
            dir_desc, _ = self._bearing_to_desc(direction_from)

            desc = (
                f"台风中心{radius_km}公里范围内的垂直风切变来自{dir_desc}方向，"
                f"强度为\"{level}\"（{round(shear_mag, 1)} m/s），"
                f"当前风切变环境对台风的发展{impact}。"
            )

            return {
                "system_name": "VerticalWindShear",
                "description": desc,
                "position": {
                    "description": f"台风中心{radius_km}km圆域平均的200-850hPa风矢量差",
                    "lat": tc_lat,
                    "lon": tc_lon,
                    "radius_km": radius_km,
                },
                "intensity": {"value": round(shear_mag, 2), "unit": "m/s", "level": level},
                "shape": {
                    "description": f"一个从{dir_desc}指向的矢量",
                    "vector_coordinates": self._get_vector_coords(tc_lat, tc_lon, shear_u, shear_v),
                },
                "properties": {
                    "direction_from_deg": round(direction_from, 1),
                    "impact": impact,
                    "shear_vector_mps": {
                        "u": round(shear_u, 2),
                        "v": round(shear_v, 2),
                    },
                    "calculation_method": f"面积平均于{radius_km}km圆域",
                },
            }
        except Exception as e:
            # print(f"⚠️ 垂直风切变提取失败: {e}")
            return None


    def extract_ocean_heat_content(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        """
        [深度重构] 提取并解译海洋热含量（海表温度SST近似）。
        使用基于Haversine距离的圆形掩膜和局部子域进行等值线提取。
        """
        try:
            sst = self._get_sst_field(time_idx)
            if sst is None:
                return None

            # 使用Haversine距离创建圆形掩膜
            radius_km = radius_deg * 111  # 粗略转换：1度 ≈ 111公里
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            
            # 计算区域平均SST（使用圆形掩膜）
            sst_mean = np.nanmean(sst[circular_mask])

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

            # 提取局部SST数据用于等值线分析
            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            
            # 计算局部区域的索引范围（扩大到3-4倍半径以确保等值线完整）
            radius_points = int(radius_deg * 3 / self.lat_spacing)
            
            lat_start = max(0, lat_idx - radius_points)
            lat_end = min(len(self.lat), lat_idx + radius_points + 1)
            lon_start = max(0, lon_idx - radius_points)
            lon_end = min(len(self.lon), lon_idx + radius_points + 1)
            
            # 提取局部SST数据
            sst_local = sst[lat_start:lat_end, lon_start:lon_end]
            local_lat = self.lat[lat_start:lat_end]
            local_lon = self.lon[lon_start:lon_end]
            
            # 在局部区域提取26.5°C等值线
            contour_26_5 = self._get_contour_coords_local(
                sst_local, 26.5, local_lat, local_lon, tc_lon
            )

            # 增强形状分析：使用全局数据但限制在台风附近
            enhanced_shape = self._get_enhanced_shape_info(sst, 26.5, "high", tc_lat, tc_lon)

            shape_info = {
                "description": "26.5°C是台风发展的最低海温门槛，此线是生命线",
                "warm_water_boundary_26.5C": contour_26_5,
                "boundary_type": "local_region",  # 标注这是局部边界
                "extraction_radius_deg": radius_deg * 3,  # 记录提取范围
            }

            # 如果有增强形状分析，添加更多细节
            if enhanced_shape:
                shape_info.update(
                    {
                        "warm_water_area_km2": enhanced_shape["area_km2"],
                        "warm_region_shape": enhanced_shape["shape_type"],
                        "warm_region_orientation": enhanced_shape["orientation"],
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                    }
                )

                # 更新描述信息
                desc += f" 暖水区域面积约{enhanced_shape['area_km2']:.0f}km²，呈{enhanced_shape['shape_type']}，{enhanced_shape['orientation']}。"

            return {
                "system_name": "OceanHeatContent",
                "description": desc,
                "position": {
                    "description": f"台风中心周围{radius_deg}度半径内的海域",
                    "lat": tc_lat,
                    "lon": tc_lon,
                },
                "intensity": {"value": round(sst_mean.item(), 2), "unit": "°C", "level": level},
                "shape": shape_info,
                "properties": {"impact": impact},
            }
        except Exception as e:
            # print(f"⚠️ 海洋热含量提取失败: {e}")
            return None

    def extract_upper_level_divergence(self, time_idx, tc_lat, tc_lon):
        """
        提取并解译高空辐散系统（200hPa散度场）。
        高空辐散有利于低层辐合加强，促进台风发展。
        """
        try:
            u200 = self._get_data_at_level("u", 200, time_idx)
            v200 = self._get_data_at_level("v", 200, time_idx)
            if u200 is None or v200 is None:
                return None

            # 计算散度场 (加入极区防护和有限值过滤)
            with np.errstate(divide="ignore", invalid="ignore"):
                gy_u, gx_u = self._raw_gradients(u200)
                gy_v, gx_v = self._raw_gradients(v200)
                du_dx = gx_u / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
                dv_dy = gy_v / (self.lat_spacing * 111000)
                divergence = du_dx + dv_dy
            if not np.any(np.isfinite(divergence)):
                return None
            divergence[~np.isfinite(divergence)] = np.nan

            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            div_val_raw = divergence[lat_idx, lon_idx]
            if not np.isfinite(div_val_raw):
                # 使用周围 3x3 有限值平均替代
                r = 1
                sub = divergence[max(0, lat_idx-r):lat_idx+r+1, max(0, lon_idx-r):lon_idx+r+1]
                finite_sub = sub[np.isfinite(sub)]
                if finite_sub.size == 0:
                    return None
                div_val_raw = float(np.nanmean(finite_sub))
            # 合理范围裁剪 (典型散度量级 < 2e-4 s^-1)
            div_val_raw = float(np.clip(div_val_raw, -5e-4, 5e-4))
            div_value = div_val_raw * 1e5  # 转换为10^-5 s^-1单位

            if div_value > 5:
                level, impact = "强", "极其有利于台风发展和加强"
            elif div_value > 2:
                level, impact = "中等", "有利于台风维持和发展"
            elif div_value > -2:
                level, impact = "弱", "对台风发展影响较小"
            else:
                level, impact = "负值", "不利于台风发展"

            desc = (
                f"台风上方200hPa高度的散度值为{div_value:.1f}×10⁻⁵ s⁻¹，高空辐散强度为'{level}'，"
                f"{impact}。"
            )

            return {
                "system_name": "UpperLevelDivergence",
                "description": desc,
                "position": {"description": "台风中心上方200hPa高度", "lat": tc_lat, "lon": tc_lon},
                "intensity": {"value": round(div_value, 2), "unit": "×10⁻⁵ s⁻¹", "level": level},
                "shape": {"description": "高空辐散中心的空间分布"},
                "properties": {"impact": impact, "favorable_development": div_value > 0},
            }
        except Exception as e:
            return None

    def extract_intertropical_convergence_zone(self, time_idx, tc_lat, tc_lon):
        """
        提取并解译热带辐合带(ITCZ)。
        ITCZ是热带对流活动的主要区域，影响台风的生成和路径。
        """
        try:
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if u850 is None or v850 is None:
                return None

            # 计算850hPa涡度来识别ITCZ
            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)
            du_dy = gy_u / (self.lat_spacing * 111000)
            dv_dx = gx_v / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
            vorticity = dv_dx - du_dy

            # ITCZ通常位于5°N-15°N之间，寻找最大涡度带
            tropical_mask = (self.lat >= 0) & (self.lat <= 20)
            if not np.any(tropical_mask):
                return None

            tropical_vort = vorticity[tropical_mask, :]
            max_vort_lat_idx = np.unravel_index(np.nanargmax(tropical_vort), tropical_vort.shape)[0]
            itcz_lat = self.lat[tropical_mask][max_vort_lat_idx]

            distance_to_tc = abs(tc_lat - itcz_lat)
            if distance_to_tc < 5:
                influence = "直接影响台风发展"
            elif distance_to_tc < 10:
                influence = "对台风路径有显著影响"
            else:
                influence = "对台风影响较小"

            desc = f"热带辐合带当前位于约{itcz_lat:.1f}°N附近，与台风中心距离{distance_to_tc:.1f}度，{influence}。"

            return {
                "system_name": "InterTropicalConvergenceZone",
                "description": desc,
                "position": {
                    "description": f"热带辐合带位置",
                    "lat": round(itcz_lat, 1),
                    "lon": "跨经度带",
                },
                "intensity": {"description": "基于850hPa涡度确定的活跃程度"},
                "shape": {"description": "东西向延伸的辐合带"},
                "properties": {
                    "distance_to_tc_deg": round(distance_to_tc, 1),
                    "influence": influence,
                },
            }
        except Exception as e:
            return None

    def extract_westerly_trough(self, time_idx, tc_lat, tc_lon):
        """
        提取并解译西风槽系统。
        西风槽可以为台风提供额外的动力支持或影响其路径。
        """
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            # 寻找中纬度地区的槽线（位势高度相对低值区）
            mid_lat_mask = (self.lat >= 20) & (self.lat <= 60)
            if not np.any(mid_lat_mask):
                return None

            # 寻找500hPa高度场的波动
            z500_mid = z500[mid_lat_mask, :]
            trough_threshold = np.percentile(z500_mid, 25)  # 寻找低四分位数区域

            trough_systems = self._identify_pressure_system(
                z500, tc_lat, tc_lon, "low", trough_threshold
            )
            if not trough_systems:
                return None

            trough_lat = trough_systems["position"]["center_of_mass"]["lat"]
            trough_lon = trough_systems["position"]["center_of_mass"]["lon"]

            # 计算与台风的相对位置
            bearing, rel_pos_desc = self._calculate_bearing(tc_lat, tc_lon, trough_lat, trough_lon)
            distance = self._calculate_distance(tc_lat, tc_lon, trough_lat, trough_lon)

            if distance < 1000:
                influence = "直接影响台风路径和强度"
            elif distance < 2000:
                influence = "对台风有间接影响"
            else:
                influence = "影响较小"

            desc = f"在台风{rel_pos_desc}约{distance:.0f}公里处存在西风槽系统，{influence}。"

            # 添加详细的坐标信息
            trough_coords = self._get_system_coordinates(
                z500, trough_threshold, "low", max_points=12
            )
            shape_info = {"description": "南北向延伸的槽线系统"}

            if trough_coords:
                shape_info.update(
                    {
                        "coordinates": trough_coords,
                        "extent_desc": f"纬度跨度{trough_coords['span_deg'][1]:.1f}°，经度跨度{trough_coords['span_deg'][0]:.1f}°",
                    }
                )
                desc += f" 槽线主体跨越纬度{trough_coords['span_deg'][1]:.1f}°，经度{trough_coords['span_deg'][0]:.1f}°。"

            return {
                "system_name": "WesterlyTrough",
                "description": desc,
                "position": trough_systems["position"],
                "intensity": trough_systems["intensity"],
                "shape": shape_info,
                "properties": {
                    "distance_to_tc_km": round(distance, 0),
                    "bearing_from_tc": round(bearing, 1),
                    "influence": influence,
                },
            }
        except Exception as e:
            return None

    def extract_frontal_system(self, time_idx, tc_lat, tc_lon):
        """
        提取并解译锋面系统。
        锋面系统通过温度梯度和风切变影响台风的移动路径。
        """
        try:
            t850 = self._get_data_at_level("t", 850, time_idx)
            if t850 is None:
                return None

            # 转换温度单位
            if np.nanmean(t850) > 200:
                t850 = t850 - 273.15

            # 计算温度梯度来识别锋面 (防止极区 cos(latitude)=0 导致除零 -> inf)
            with np.errstate(divide="ignore", invalid="ignore"):
                gy_t, gx_t = self._raw_gradients(t850)
                dt_dy = gy_t / (self.lat_spacing * 111000)
                dt_dx = gx_t / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
                temp_gradient = np.sqrt(dt_dx**2 + dt_dy**2)

            # 清理异常值
            if not np.any(np.isfinite(temp_gradient)):
                return None
            temp_gradient[~np.isfinite(temp_gradient)] = np.nan

            # 寻找强温度梯度区域（锋面特征）
            front_threshold = np.percentile(temp_gradient, 90)  # 前10%的强梯度区域
            front_mask = temp_gradient > front_threshold

            if not np.any(front_mask):
                return None

            # 寻找离台风最近的锋面
            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            search_radius = 50  # 搜索半径格点数

            lat_start = max(0, lat_idx - search_radius)
            lat_end = min(len(self.lat), lat_idx + search_radius)
            lon_start = max(0, lon_idx - search_radius)
            lon_end = min(len(self.lon), lon_idx + search_radius)

            local_front = front_mask[lat_start:lat_end, lon_start:lon_end]
            if not np.any(local_front):
                return None

            # 使用有限值的最大值
            finite_vals = temp_gradient[front_mask][np.isfinite(temp_gradient[front_mask])]
            if finite_vals.size == 0:
                return None
            front_strength = np.max(finite_vals)

            # 数值合理性限制，极端情况裁剪，单位: °C/m
            if not np.isfinite(front_strength) or front_strength <= 0:
                return None
            # 典型锋面水平温度梯度 ~ 1e-5 到 数值模式中少见超过 1e-4
            front_strength = float(np.clip(front_strength, 0, 5e-4))

            if front_strength > 3e-5:
                level = "强"
            elif front_strength > 1e-5:
                level = "中等"
            else:
                level = "弱"

            strength_1e5 = front_strength * 1e5  # 转换为 ×10⁻⁵ °C/m 标度
            desc = (
                f"台风周围存在强度为'{level}'的锋面系统，温度梯度达到{strength_1e5:.1f}×10⁻⁵ °C/m，"
                f"可能影响台风的移动路径。"
            )

            # 提取锋面带的坐标信息
            frontal_coords = self._get_system_coordinates(
                temp_gradient, front_threshold, "high", max_points=15
            )
            shape_info = {"description": "线性的温度梯度带"}

            if frontal_coords:
                shape_info.update(
                    {
                        "coordinates": frontal_coords,
                        "extent_desc": f"锋面带跨越纬度{frontal_coords['span_deg'][1]:.1f}°，经度{frontal_coords['span_deg'][0]:.1f}°",
                        "orientation_note": "根据几何形状确定锋面走向",
                    }
                )
                desc += f" 锋面带主体跨越{frontal_coords['span_deg'][1]:.1f}°纬度和{frontal_coords['span_deg'][0]:.1f}°经度。"

            return {
                "system_name": "FrontalSystem",
                "description": desc,
                "position": {"description": "台风周围的锋面区域", "lat": tc_lat, "lon": tc_lon},
                "intensity": {
                    "value": round(strength_1e5, 2),
                    "unit": "×10⁻⁵ °C/m",
                    "level": level,
                },
                "shape": shape_info,
                "properties": {"impact": "影响台风路径和结构"},
            }
        except Exception as e:
            return None

    def extract_monsoon_trough(self, time_idx, tc_lat, tc_lon):
        """
        提取并解译季风槽系统。
        季风槽是热带气旋生成的重要环境，也影响现有台风的发展。
        """
        try:
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if u850 is None or v850 is None:
                return None

            # 计算850hPa相对涡度
            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)
            du_dy = gy_u / (self.lat_spacing * 111000)
            dv_dx = gx_v / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
            relative_vorticity = dv_dx - du_dy

            # 清理异常数值
            with np.errstate(invalid="ignore"):
                relative_vorticity[~np.isfinite(relative_vorticity)] = np.nan

            # 季风槽通常在热带地区，寻找正涡度带
            tropical_mask = (self.lat >= -30) & (self.lat <= 30)
            if not np.any(tropical_mask):
                return None

            tropical_vort = relative_vorticity[tropical_mask, :]
            monsoon_threshold = (
                np.percentile(tropical_vort[tropical_vort > 0], 75)
                if np.any(tropical_vort > 0)
                else 0
            )

            if monsoon_threshold <= 0:
                return None

            monsoon_mask = relative_vorticity > monsoon_threshold
            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)

            # 检查台风附近是否存在季风槽
            search_radius = 30
            lat_start = max(0, lat_idx - search_radius)
            lat_end = min(len(self.lat), lat_idx + search_radius)
            lon_start = max(0, lon_idx - search_radius)
            lon_end = min(len(self.lon), lon_idx + search_radius)

            local_monsoon = monsoon_mask[lat_start:lat_end, lon_start:lon_end]
            if not np.any(local_monsoon):
                return None

            finite_vort = relative_vorticity[monsoon_mask][
                np.isfinite(relative_vorticity[monsoon_mask])
            ]
            if finite_vort.size == 0:
                return None
            max_vorticity = float(np.max(finite_vort))
            # 裁剪到合理范围 (典型热带涡度 < 2e-3 s^-1)
            max_vorticity = float(np.clip(max_vorticity, 0, 2e-3)) * 1e5

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
        except Exception as e:
            return None

    # --- 主分析与导出函数 ---
    def analyze_and_export_as_json(self, output_dir="final_single_output"):
        """Public entry point that always releases file handles."""

        try:
            return self._analyze_and_export_as_json(output_dir)
        finally:
            self.close()

    def _analyze_and_export_as_json(self, output_dir="final_single_output"):
        # ... (此函数逻辑与上一版基本相同，无需修改) ...
        print("\n🔍 开始进行专家级环境场解译并构建JSON...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # === 新增: 如果输出已存在则跳过重算 ===
        # 判定标准: 对当前 NC 文件 (self.nc_stem) 所有粒子(若无粒子列则默认为 TC_01) 的
        # 目标文件 <ncstem>_TC_Analysis_<particle>.json 均已存在且非空, 则直接跳过
        existing_outputs = list(output_path.glob(f"{self.nc_stem}_TC_Analysis_*.json"))
        if existing_outputs:
            # 确定期望粒子集合
            if "particle" in self.tc_tracks.columns:
                expected_particles = sorted(set(str(p) for p in self.tc_tracks["particle"].unique()))
            else:
                expected_particles = ["TC_01"]
            # 已存在并且文件非空的粒子结果
            existing_particles = []
            for pfile in existing_outputs:
                # 文件名格式: <ncstem>_TC_Analysis_<pid>.json -> 提取 <pid>
                stem = pfile.stem
                if stem.startswith(f"{self.nc_stem}_TC_Analysis_"):
                    pid = stem.replace(f"{self.nc_stem}_TC_Analysis_", "")
                    try:
                        if pfile.stat().st_size > 10:  # 简单判定非空
                            existing_particles.append(pid)
                    except Exception:
                        pass
            if set(expected_particles).issubset(existing_particles):
                print(
                    f"⏩ 检测到当前NC对应的所有分析结果已存在于 '{output_path}' (共{len(existing_particles)}个)，跳过重算。"
                )
                return {pid: None for pid in expected_particles}  # 返回占位, 表示已跳过

        if "particle" not in self.tc_tracks.columns:
            print("警告: 路径文件 .csv 中未找到 'particle' 列，将所有路径点视为单个台风事件。")
            self.tc_tracks["particle"] = "TC_01"

        tc_groups = self.tc_tracks.groupby("particle")
        all_typhoon_events = {}

        for tc_id, track_df in tc_groups:
            print(f"\n🌀 正在处理台风事件: {tc_id}")
            event_data = {
                "tc_id": str(tc_id),
                "analysis_time": datetime.now().isoformat(),
                "time_series": [],
            }

            for _, track_point in track_df.sort_values(by="time").iterrows():
                time_idx, lat, lon = (
                    int(track_point.get("time_idx", 0)),
                    track_point["lat"],
                    track_point["lon"],
                )
                print(f"  -> 分析时间点: {track_point['time'].strftime('%Y-%m-%d %H:%M')}")

                environmental_systems = []
                systems_to_extract = [
                    self.extract_steering_system,
                    self.extract_vertical_wind_shear,
                    self.extract_ocean_heat_content,
                    self.extract_upper_level_divergence,
                    self.extract_intertropical_convergence_zone,
                    self.extract_westerly_trough,
                    self.extract_frontal_system,
                    self.extract_monsoon_trough,
                ]

                for func in systems_to_extract:
                    system_obj = func(time_idx, lat, lon)
                    if system_obj:
                        environmental_systems.append(system_obj)

                event_data["time_series"].append(
                    {
                        "time": track_point["time"].isoformat(),
                        "time_idx": time_idx,
                        "tc_position": {"lat": lat, "lon": lon},
                        "tc_intensity_hpa": track_point.get("intensity", None),
                        "environmental_systems": environmental_systems,
                    }
                )
            all_typhoon_events[str(tc_id)] = event_data

        for tc_id, data in all_typhoon_events.items():
            # 在输出文件名中加入原始NC文件名(去扩展)，格式: <ncstem>_TC_Analysis_<tc_id>.json
            json_filename = output_path / f"{self.nc_stem}_TC_Analysis_{tc_id}.json"
            print(f"💾 保存专家解译结果到: {json_filename}")

            # 递归转换numpy类型为Python原生类型
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    val = float(obj)
                    if not np.isfinite(val):
                        return None
                    return val
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj

            # 额外递归处理 Python float 中的 inf / nan
            def sanitize_inf_nan(o):
                if isinstance(o, dict):
                    return {k: sanitize_inf_nan(v) for k, v in o.items()}
                elif isinstance(o, list):
                    return [sanitize_inf_nan(v) for v in o]
                elif isinstance(o, float):
                    if math.isinf(o) or math.isnan(o):
                        return None
                    return o
                return o

            converted_data = convert_numpy_types(data)
            converted_data = sanitize_inf_nan(converted_data)

            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ 所有台风事件解译完成，结果保存在: {output_path}")
        return all_typhoon_events

    # --- 辅助与工具函数 ---
    def _get_sst_field(self, time_idx):
        # 优先查找SST数据，如果没有则使用2米温度作为近似
        for var_name in ["sst", "ts"]:
            if var_name in self.ds.data_vars:
                sst_data = self.ds[var_name].isel(time=time_idx).values
                return sst_data - 273.15 if np.nanmean(sst_data) > 200 else sst_data

        # 如果没有SST数据，使用2米温度作为近似（仅在海洋区域有效）
        for var_name in ["t2", "t2m"]:
            if var_name in self.ds.data_vars:
                t2_data = self.ds[var_name].isel(time=time_idx).values
                # 转换温度单位
                sst_approx = t2_data - 273.15 if np.nanmean(t2_data) > 200 else t2_data
                # 注意：这是一个近似，在陆地上会不准确
                print(f"⚠️  使用{var_name}作为海表温度近似")
                return sst_approx

        return None

    def _calculate_steering_flow(self, z500, tc_lat, tc_lon):
        gy, gx = self._raw_gradients(z500)
        dy = gy / (self.lat_spacing * 111000)
        dx = gx / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
        lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
        u_steering = -dx[lat_idx, lon_idx] / (9.8 * 1e-5)
        v_steering = dy[lat_idx, lon_idx] / (9.8 * 1e-5)
        speed = np.sqrt(u_steering**2 + v_steering**2)
        direction = (np.degrees(np.arctan2(u_steering, v_steering)) + 180) % 360
        return speed, direction, u_steering, v_steering

    def _get_contour_coords(self, data_field, level, center_lon, max_points=100):
        try:
            contours = find_contours(data_field, level)
            if not contours:
                return None
            # 寻找最长的等值线段，通常是主系统
            main_contour = sorted(contours, key=len, reverse=True)[0]

            # 对经度进行正确转换
            contour_lon = self.lon[main_contour[:, 1].astype(int)]
            contour_lat = self.lat[main_contour[:, 0].astype(int)]

            # 降采样以减少数据量
            step = max(1, len(main_contour) // max_points)
            return [
                [round(lon, 2), round(lat, 2)]
                for lon, lat in zip(contour_lon[::step], contour_lat[::step])
            ]
        except Exception:
            return None

    def _get_contour_coords_local(self, data_field, level, lat_array, lon_array, 
                                   center_lon, max_points=100):
        """
        在局部数据场上提取等值线坐标
        
        Args:
            data_field: 局部数据场（2D数组）
            level: 等值线阈值
            lat_array: 对应的纬度数组
            lon_array: 对应的经度数组
            center_lon: 中心经度（用于归一化）
            max_points: 最大返回点数
        
        Returns:
            等值线坐标列表 [[lon, lat], ...] 或 None
        """
        try:
            contours = find_contours(data_field, level)
            if not contours:
                return None
            
            # 寻找最长的等值线段
            main_contour = sorted(contours, key=len, reverse=True)[0]
            
            # 使用局部的经纬度数组进行索引映射
            contour_indices_lat = main_contour[:, 0].astype(int)
            contour_indices_lon = main_contour[:, 1].astype(int)
            
            # 确保索引在有效范围内
            contour_indices_lat = np.clip(contour_indices_lat, 0, len(lat_array) - 1)
            contour_indices_lon = np.clip(contour_indices_lon, 0, len(lon_array) - 1)
            
            contour_lon = lon_array[contour_indices_lon]
            contour_lat = lat_array[contour_indices_lat]
            
            # 对经度进行归一化处理，避免跨越日期变更线导致的跳变
            contour_lon_normalized = self._normalize_longitude(contour_lon, center_lon)
            
            # 降采样以减少数据量
            step = max(1, len(main_contour) // max_points)
            
            # 返回归一化后的坐标，但将超出[-180, 180]的经度转回[0, 360]范围
            coords = []
            for lon, lat in zip(contour_lon_normalized[::step], contour_lat[::step]):
                # 将归一化的经度转回标准[0, 360]范围（如果需要）
                if lon < 0:
                    lon = lon + 360
                coords.append([round(float(lon), 2), round(float(lat), 2)])
            
            return coords
        except Exception as e:
            # 调试时可以打印错误信息
            # print(f"局部等值线提取失败: {e}")
            return None

    def _get_enhanced_shape_info(self, data_field, threshold, system_type, center_lat, center_lon):
        """
        获取增强的形状信息，包含详细的坐标定位
        """
        try:
            shape_analysis = self.shape_analyzer.analyze_system_shape(
                data_field, threshold, system_type, center_lat, center_lon
            )
            if shape_analysis:
                # 基础信息
                basic_info = {
                    "area_km2": shape_analysis["basic_geometry"]["area_km2"],
                    "shape_type": shape_analysis["basic_geometry"]["description"],
                    "orientation": shape_analysis["orientation"]["direction_type"],
                    "complexity": shape_analysis["shape_complexity"]["description"],
                    "detailed_analysis": shape_analysis,
                }

                # 添加坐标信息
                if "contour_analysis" in shape_analysis and shape_analysis["contour_analysis"]:
                    contour_data = shape_analysis["contour_analysis"]
                    basic_info.update(
                        {
                            "coordinate_info": {
                                "main_contour_coords": contour_data.get(
                                    "simplified_coordinates", []
                                ),
                                "polygon_features": contour_data.get("polygon_features", {}),
                                "contour_length_km": contour_data.get("contour_length_km", 0),
                            }
                        }
                    )

                return basic_info
        except Exception as e:
            print(f"形状分析失败: {e}")
        return None

    def _get_system_coordinates(self, data_field, threshold, system_type, max_points=20):
        """
        专门提取气象系统的关键坐标点
        """
        try:
            # 创建系统掩膜
            if system_type == "high":
                mask = data_field >= threshold
            else:
                mask = data_field <= threshold

            if not np.any(mask):
                return None

            # 找到连通区域
            labeled_mask, num_features = label(mask)
            if num_features == 0:
                return None

            # 选择最大的连通区域
            flat_labels = labeled_mask.ravel()
            counts = np.bincount(flat_labels)[1: num_features + 1]
            if counts.size == 0:
                return None
            main_label = int(np.argmax(counts) + 1)
            main_region = labeled_mask == main_label

            # 提取边界坐标
            contours = find_contours(main_region.astype(float), 0.5)
            if not contours:
                return None

            main_contour = max(contours, key=len)

            # 简化多边形以获得关键点
            epsilon = len(main_contour) * 0.01  # 简化程度
            simplified = approximate_polygon(main_contour, tolerance=epsilon)

            # 限制点数
            if len(simplified) > max_points:
                step = len(simplified) // max_points
                simplified = simplified[::step]

            # 转换为地理坐标
            geo_coords = []
            for point in simplified:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                # 使用更紧凑的数组格式 [lon, lat]
                geo_coords.append([round(self.lon[lon_idx], 3), round(self.lat[lat_idx], 3)])

            # 计算系统范围
            if geo_coords:
                lons = [coord[0] for coord in geo_coords]
                lats = [coord[1] for coord in geo_coords]

                extent = {
                    "boundaries": [
                        round(min(lons), 3),
                        round(min(lats), 3),
                        round(max(lons), 3),
                        round(max(lats), 3),
                    ],  # [west, south, east, north]
                    "center": [round(np.mean(lons), 3), round(np.mean(lats), 3)],  # [lon, lat]
                    "span": [
                        round(max(lons) - min(lons), 3),
                        round(max(lats) - min(lats), 3),
                    ],  # [lon_span, lat_span]
                }

                return {
                    "vertices": geo_coords,  # 简化的数组格式
                    "vertex_count": len(geo_coords),
                    "extent": extent,
                    "span_deg": [extent["span"][0], extent["span"][1]],  # [lon_span, lat_span]
                }

            return None
        except Exception as e:
            print(f"坐标提取失败: {e}")
            return None

    def _generate_coordinate_description(self, coords_info, system_name="系统"):
        """
        生成可读的坐标描述文本
        """
        if not coords_info:
            return ""

        try:
            description_parts = []

            # 系统范围描述
            if "extent" in coords_info:
                extent = coords_info["extent"]
                boundaries = extent["boundaries"]  # [west, south, east, north]
                description_parts.append(
                    f"{system_name}主体位于{boundaries[0]:.1f}°E-{boundaries[2]:.1f}°E，"
                    f"{boundaries[1]:.1f}°N-{boundaries[3]:.1f}°N"
                )

            # 关键顶点描述
            if "vertices" in coords_info and coords_info["vertex_count"] > 0:
                vertex_count = coords_info["vertex_count"]
                description_parts.append(f"由{vertex_count}个关键顶点构成的多边形形状")

            # 尺度描述
            if "span_deg" in coords_info:
                lon_span, lat_span = coords_info["span_deg"]
                lat_km = lat_span * 111  # 纬度1度约111km
                center_lat = coords_info.get("extent", {}).get("center", [0, 30])[1]
                lon_km = lon_span * 111 * np.cos(np.radians(center_lat))
                description_parts.append(f"纬向跨度约{lat_km:.0f}km，经向跨度约{lon_km:.0f}km")

            return "，".join(description_parts) + "。" if description_parts else ""

        except Exception:
            return ""

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两点间的球面距离（单位：公里）"""
        R = 6371.0  # 地球半径，公里
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        dLon = math.radians(lon2 - lon1)
        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        y = math.sin(dLon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
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
        # 将 m/s 转换为经纬度偏移
        # 这是一个非常粗略的近似，仅用于可视化示意
        end_lat = lat + v * scale * 0.009  # 1 m/s ~ 0.009 deg lat
        end_lon = lon + u * scale * 0.009 / math.cos(math.radians(lat))
        return {
            "start": {"lat": round(lat, 2), "lon": round(lon, 2)},
            "end": {"lat": round(end_lat, 2), "lon": round(end_lon, 2)},
        }

    def _identify_pressure_system(self, *args, **kwargs):
        # ... (此函数与上一版相同) ...
        data_field, tc_lat, tc_lon, system_type, threshold = args
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
            center_y, center_x = (slc[0].start + slc[0].stop) / 2, (slc[1].start + slc[1].stop) / 2
            dist = np.sqrt((center_y - tc_lat_idx) ** 2 + (center_x - tc_lon_idx) ** 2)
            if dist < min_dist:
                min_dist, closest_feature_idx = dist, i
        if closest_feature_idx == -1:
            return None
        target_slc = objects_slices[closest_feature_idx]
        target_mask = labeled_array == (closest_feature_idx + 1)
        com_y, com_x = center_of_mass(target_mask)
        pos_lat, pos_lon = self.lat[int(com_y)], self.lon[int(com_x)]
        intensity_val = (
            np.max(data_field[target_mask])
            if system_type == "high"
            else np.min(data_field[target_mask])
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

    def _get_data_at_level(self, *args, **kwargs):
        # ... (此函数与上一版相同) ...
        var_name, level_hPa, time_idx = args
        if var_name not in self.ds.data_vars:
            return None
        var_data = self.ds[var_name]
        level_dim = next(
            (dim for dim in ["level", "isobaricInhPa", "pressure"] if dim in var_data.dims), None
        )
        if level_dim is None:
            return (
                var_data.isel(time=time_idx).values if "time" in var_data.dims else var_data.values
            )
        levels = self.ds[level_dim].values
        level_idx = np.abs(levels - level_hPa).argmin()
        return var_data.isel(time=time_idx, **{level_dim: level_idx}).values

    def _create_region_mask(self, *args, **kwargs):
        # ... (此函数与上一版相同) ...
        center_lat, center_lon, radius_deg = args
        lat_mask = (self.lat >= center_lat - radius_deg) & (self.lat <= center_lat + radius_deg)
        lon_mask = (self.lon >= center_lon - radius_deg) & (self.lon <= center_lon + radius_deg)
        return np.outer(lat_mask, lon_mask)


    # ================= 新增: 流式顺序处理函数 =================
