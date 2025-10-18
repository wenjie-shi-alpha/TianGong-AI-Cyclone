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

            # 1. 识别副高系统 (改进版 - 使用区域化处理)
            subtropical_high_obj = self._identify_subtropical_high_regional(
                z500, tc_lat, tc_lon, time_idx
            )
            if not subtropical_high_obj:
                # 如果区域化方法失败，回退到原方法
                subtropical_high_obj = self._identify_pressure_system(
                    z500, tc_lat, tc_lon, "high", 5880
                )
                if not subtropical_high_obj:
                    return None

            # 2. 增强形状分析
            enhanced_shape = self._get_enhanced_shape_info(z500, 5880, "high", tc_lat, tc_lon)

            # 3. 计算引导气流 (改进版 - 使用层平均风)
            steering_result = self._calculate_steering_flow_layered(time_idx, tc_lat, tc_lon)
            if not steering_result:
                # 如果层平均方法失败，回退到地转风方法
                steering_speed, steering_direction, u_steering, v_steering = (
                    self._calculate_steering_flow(z500, tc_lat, tc_lon)
                )
                steering_result = {
                    "speed": steering_speed,
                    "direction": steering_direction,
                    "u": u_steering,
                    "v": v_steering,
                    "method": "geostrophic_wind"
                }

            # 4. 提取脊线位置 (588线)
            ridge_info = self._extract_ridge_line(z500, tc_lat, tc_lon)

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

            # 4.2 更新形状信息（移除面积计算）
            if enhanced_shape:
                subtropical_high_obj["shape"].update(
                    {
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                        # 移除 area_km2 - 不需要计算面积
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

            # 4.3 提取闭合边界和特征点（科学安全的方法）
            # 获取动态阈值（如果有的话）
            if "extraction_info" in subtropical_high_obj and "dynamic_threshold" in subtropical_high_obj["extraction_info"]:
                dynamic_threshold = subtropical_high_obj["extraction_info"]["dynamic_threshold"]
            else:
                dynamic_threshold = 5880  # 默认值
            
            # 使用改进的闭合边界提取方法
            boundary_result = self._extract_closed_boundary_with_features(
                z500, tc_lat, tc_lon, 
                threshold=dynamic_threshold,
                lat_range=20.0,
                lon_range=40.0,
                target_points=50
            )
            
            if boundary_result:
                # 添加边界坐标（闭合）
                subtropical_high_obj["boundary_coordinates"] = boundary_result["boundary_coordinates"]
                
                # 添加关键特征点
                subtropical_high_obj["boundary_features"] = boundary_result["boundary_features"]
                
                # 添加边界度量信息
                subtropical_high_obj["boundary_metrics"] = boundary_result["boundary_metrics"]
                
                print(f"✅ 边界提取成功: {boundary_result['boundary_metrics']['total_points']}点, "
                      f"{'闭合' if boundary_result['boundary_metrics']['is_closed'] else '开放'}, "
                      f"方法: {boundary_result['boundary_metrics']['extraction_method']}")
            else:
                # 如果新方法失败，回退到旧方法
                print(f"⚠️ 新方法失败，使用旧方法提取边界")
                boundary_coords = self._extract_local_boundary_coords(
                    z500, tc_lat, tc_lon, threshold=dynamic_threshold, radius_deg=20
                )
                if boundary_coords:
                    subtropical_high_obj["boundary_coordinates"] = boundary_coords
                    subtropical_high_obj["boundary_note"] = "使用旧方法（新方法失败）"

            # 4.4 相对位置和综合描述
            high_pos = subtropical_high_obj["position"]["center_of_mass"]
            bearing, rel_pos_desc = self._calculate_bearing(
                tc_lat, tc_lon, high_pos["lat"], high_pos["lon"]
            )
            subtropical_high_obj["position"]["relative_to_tc"] = rel_pos_desc
            steering_speed = steering_result["speed"]
            steering_direction = steering_result["direction"]
            u_steering = steering_result["u"]
            v_steering = steering_result["v"]

            steering_speed = steering_result["speed"]
            steering_direction = steering_result["direction"]
            u_steering = steering_result["u"]
            v_steering = steering_result["v"]


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
                            "calculation_method": steering_result.get("method", "unknown")
                        },
                    },
                }
            )

            # 添加脊线信息
            if ridge_info:
                subtropical_high_obj["properties"]["ridge_line"] = ridge_info

            # 添加脊线信息
            if ridge_info:
                subtropical_high_obj["properties"]["ridge_line"] = ridge_info

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
            
            # 【改进】使用闭合边界提取方法（科学方法）
            boundary_result = self._extract_closed_ocean_boundary_with_features(
                sst, tc_lat, tc_lon, threshold=26.5,
                lat_range=radius_deg * 6,  # 使用6倍半径确保完整性
                lon_range=radius_deg * 12,
                target_points=50
            )

            shape_info = {
                "description": "26.5°C是台风发展的最低海温门槛，此线是生命线",
                "boundary_type": "closed_contour_with_features",  # 新方法标注
                "extraction_radius_deg": radius_deg * 3,  # 记录提取范围
            }

            # 如果成功提取闭合边界
            if boundary_result:
                shape_info["warm_water_boundary_26.5C"] = boundary_result["boundary_coordinates"]
                shape_info["boundary_features"] = boundary_result["boundary_features"]
                shape_info["boundary_metrics"] = boundary_result["boundary_metrics"]
                
                # 使用新方法计算的面积
                metrics = boundary_result["boundary_metrics"]
                if "warm_water_area_approx_km2" in metrics:
                    shape_info["warm_water_area_km2"] = metrics["warm_water_area_approx_km2"]
                    desc += f" 暖水区域面积约{metrics['warm_water_area_approx_km2']:.0f}km²"
                
                # 添加闭合性信息
                if metrics.get("is_closed"):
                    desc += f"，边界完整闭合（{metrics['total_points']}个采样点，周长{metrics['perimeter_km']:.0f}km）"
                
                # 添加关键特征信息
                features = boundary_result["boundary_features"]
                tc_rel = features.get("tc_relative_points", {})
                if "nearest_to_tc" in tc_rel:
                    nearest_dist = tc_rel["nearest_to_tc"]["distance_km"]
                    desc += f"，台风距暖水区边界最近{nearest_dist:.0f}km"
                
                # 暖涡信息
                warm_eddies = features.get("warm_eddy_centers", [])
                if warm_eddies:
                    desc += f"，检测到{len(warm_eddies)}个暖涡特征"
                    
            else:
                # 回退到旧方法
                print("⚠️ 闭合边界提取失败，回退到旧方法")
                contour_26_5 = self._get_contour_coords_local(
                    sst_local, 26.5, local_lat, local_lon, tc_lon
                )
                shape_info["warm_water_boundary_26.5C"] = contour_26_5
                shape_info["boundary_type"] = "fallback_local_region"
                
                # 旧的形状分析
                enhanced_shape = self._get_enhanced_shape_info(sst, 26.5, "high", tc_lat, tc_lon)
                if enhanced_shape:
                    shape_info.update({
                        "warm_water_area_km2": enhanced_shape["area_km2"],
                        "warm_region_shape": enhanced_shape["shape_type"],
                        "warm_region_orientation": enhanced_shape["orientation"],
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                    })
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
        
        改进点：
        1. 使用球面散度公式：div = (1/(a*cos(φ))) * ∂u/∂λ + (1/a) * ∂(v*cos(φ))/∂φ
        2. 在台风中心500km圆域内计算面积平均散度
        3. 统计最大辐散中心的位置和方位
        """
        try:
            u200 = self._get_data_at_level("u", 200, time_idx)
            v200 = self._get_data_at_level("v", 200, time_idx)
            if u200 is None or v200 is None:
                return None

            # 使用球面散度公式计算散度场
            # div = (1/(a*cos(φ))) * ∂u/∂λ + (1/a) * ∂(v*cos(φ))/∂φ
            with np.errstate(divide="ignore", invalid="ignore"):
                # 地球半径 (米)
                a = 6371000.0
                
                # 计算梯度
                gy_u, gx_u = self._raw_gradients(u200)
                
                # 计算 v*cos(φ)
                coslat = self._coslat_safe[:, np.newaxis]
                v_coslat = v200 * coslat
                gy_v_coslat, gx_v_coslat = self._raw_gradients(v_coslat)
                
                # 转换为弧度梯度
                dlambda = np.deg2rad(self.lon_spacing)  # 经度间隔（弧度）
                dphi = np.deg2rad(self.lat_spacing)     # 纬度间隔（弧度）
                
                # 球面散度：div = (1/(a*cos(φ))) * ∂u/∂λ + (1/a) * ∂(v*cos(φ))/∂φ
                du_dlambda = gx_u / dlambda
                dv_coslat_dphi = gy_v_coslat / dphi
                
                divergence = (du_dlambda / (a * coslat) + dv_coslat_dphi / a)
            
            if not np.any(np.isfinite(divergence)):
                return None
            divergence[~np.isfinite(divergence)] = np.nan

            # 创建500km圆形掩膜
            radius_km = 500
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            
            # 在掩膜区域内计算平均散度
            divergence_masked = np.where(circular_mask, divergence, np.nan)
            
            # 计算区域平均散度
            div_val_raw = float(np.nanmean(divergence_masked))
            if not np.isfinite(div_val_raw):
                return None
            
            # 找到掩膜区域内的最大辐散中心
            max_div_idx = np.nanargmax(divergence_masked)
            max_div_lat_idx, max_div_lon_idx = np.unravel_index(max_div_idx, divergence_masked.shape)
            max_div_lat = float(self.lat[max_div_lat_idx])
            max_div_lon = float(self.lon[max_div_lon_idx])
            max_div_value = float(divergence[max_div_lat_idx, max_div_lon_idx])
            
            # 计算最大辐散中心与台风中心的距离和方位
            distance_to_max = self._haversine_distance(tc_lat, tc_lon, max_div_lat, max_div_lon)
            
            # 计算方位角
            def calculate_bearing(lat1, lon1, lat2, lon2):
                """计算从点1到点2的方位角（度，正北为0°，顺时针）"""
                lat1_rad = np.deg2rad(lat1)
                lat2_rad = np.deg2rad(lat2)
                dlon_rad = np.deg2rad(lon2 - lon1)
                
                x = np.sin(dlon_rad) * np.cos(lat2_rad)
                y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
                bearing = np.rad2deg(np.arctan2(x, y))
                return (bearing + 360) % 360
            
            bearing = calculate_bearing(tc_lat, tc_lon, max_div_lat, max_div_lon)
            
            # 方位描述
            direction_names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
            direction_idx = int((bearing + 22.5) // 45) % 8
            direction = direction_names[direction_idx]
            
            # 合理范围裁剪 (典型散度量级 < 2e-4 s^-1)
            div_val_raw = float(np.clip(div_val_raw, -5e-4, 5e-4))
            max_div_value = float(np.clip(max_div_value, -5e-4, 5e-4))
            
            # 转换为10^-5 s^-1单位
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

            # 判断辐散中心是否偏移
            offset_note = ""
            if distance_to_max > 100:  # 如果最大辐散中心距离台风中心超过100km
                offset_note = f"最大辐散中心位于台风中心{direction}方向约{distance_to_max:.0f}公里处，强度为{max_div_value_scaled:.1f}×10⁻⁵ s⁻¹，"
                if distance_to_max > 200:
                    offset_note += "辐散中心明显偏移可能影响台风的对称结构。"
                else:
                    offset_note += "辐散中心略有偏移。"

            desc = (
                f"台风中心周围500公里范围内200hPa高度的平均散度值为{div_value:.1f}×10⁻⁵ s⁻¹，"
                f"高空辐散强度为'{level}'，{impact}。"
            )
            if offset_note:
                desc += offset_note

            return {
                "system_name": "UpperLevelDivergence",
                "description": desc,
                "position": {
                    "description": f"台风中心周围{radius_km}公里范围内200hPa高度",
                    "center_lat": tc_lat,
                    "center_lon": tc_lon,
                    "radius_km": radius_km
                },
                "intensity": {
                    "average_value": round(div_value, 2),
                    "max_value": round(max_div_value_scaled, 2),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": level
                },
                "divergence_center": {
                    "lat": round(max_div_lat, 2),
                    "lon": round(max_div_lon, 2),
                    "distance_to_tc_km": round(distance_to_max, 1),
                    "direction": direction,
                    "bearing_deg": round(bearing, 1)
                },
                "shape": {"description": "高空辐散中心的空间分布"},
                "properties": {
                    "impact": impact,
                    "favorable_development": div_value > 0,
                    "center_offset": distance_to_max > 100
                },
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

    # ================= 新增: 副热带高压和引导气流改进函数 =================
    
    def _identify_subtropical_high_regional(self, z500, tc_lat, tc_lon, time_idx):
        """
        使用区域化处理识别副热带高压
        
        改进:
        1. 在台风周围20°x40°区域内处理
        2. 计算高度异常场(相对于时间/纬向平均)
        3. 使用局部阈值而非全局固定5880gpm
        
        Returns:
            副高系统信息字典，或None
        """
        try:
            # 1. 定义局部区域 (台风周围 20°纬度 x 40°经度)
            lat_range = 20.0
            lon_range = 40.0
            
            lat_min = max(tc_lat - lat_range/2, self.lat.min())
            lat_max = min(tc_lat + lat_range/2, self.lat.max())
            lon_min = tc_lon - lon_range/2
            lon_max = tc_lon + lon_range/2
            
            # 创建区域掩膜
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            lon_mask_raw = (self.lon >= lon_min) & (self.lon <= lon_max)
            
            # 处理经度跨越0°/360°的情况
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = lon_mask_raw
            
            # 2. 提取局部区域数据
            region_z500 = z500[np.ix_(lat_mask, lon_mask)]
            
            # 3. 计算高度异常 (简化版:相对于区域平均)
            z500_mean = np.nanmean(region_z500)
            z500_anomaly = region_z500 - z500_mean
            
            # 4. 使用动态阈值 (75百分位或区域平均+标准差)
            threshold_percentile = np.nanpercentile(region_z500, 75)
            threshold_std = z500_mean + np.nanstd(region_z500)
            dynamic_threshold = min(threshold_percentile, threshold_std)
            
            # 确保阈值合理 (至少5860 gpm)
            dynamic_threshold = max(dynamic_threshold, 5860)
            
            # 5. 识别高压区域
            high_mask = region_z500 > dynamic_threshold
            
            if not np.any(high_mask):
                return None
            
            # 6. 标记连通区域
            labeled_array, num_features = label(high_mask)
            
            if num_features == 0:
                return None
            
            # 7. 选择最大/最强的连通区域
            max_area = 0
            best_feature_idx = -1
            
            for i in range(1, num_features + 1):
                feature_mask = (labeled_array == i)
                area = np.sum(feature_mask)
                
                if area > max_area:
                    max_area = area
                    best_feature_idx = i
            
            if best_feature_idx == -1:
                return None
            
            # 8. 计算副高属性
            target_mask = (labeled_array == best_feature_idx)
            com_y, com_x = center_of_mass(target_mask)
            
            # 转换回全局坐标
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            
            pos_lat = local_lat[int(com_y)]
            pos_lon = local_lon[int(com_x)]
            
            intensity_val = np.max(region_z500[target_mask])
            
            return {
                "position": {
                    "center_of_mass": {
                        "lat": round(float(pos_lat), 2),
                        "lon": round(float(pos_lon), 2)
                    }
                },
                "intensity": {
                    "value": round(float(intensity_val), 1),
                    "unit": "gpm"
                },
                "shape": {},
                "extraction_info": {
                    "method": "regional_processing",
                    "region_extent": {
                        "lat_range": [float(lat_min), float(lat_max)],
                        "lon_range": [float(lon_min), float(lon_max)]
                    },
                    "dynamic_threshold": round(float(dynamic_threshold), 1)
                }
            }
            
        except Exception as e:
            print(f"⚠️ 区域化副高识别失败: {e}")
            return None
    
    def _calculate_steering_flow_layered(self, time_idx, tc_lat, tc_lon, radius_deg=5.0):
        """
        使用850-300hPa层平均风计算引导气流
        
        改进:
        1. 计算多层风场的质量加权平均
        2. 在台风中心周围区域进行面积平均
        3. 考虑纬度相关的科里奥利参数
        
        Args:
            time_idx: 时间索引
            tc_lat: 台风中心纬度
            tc_lon: 台风中心经度
            radius_deg: 计算半径(度)，默认5度
        
        Returns:
            {"speed": ..., "direction": ..., "u": ..., "v": ..., "method": ...} 或 None
        """
        try:
            # 1. 定义层次 (850, 700, 500, 300 hPa)
            levels = [850, 700, 500, 300]
            weights = [0.3, 0.3, 0.2, 0.2]  # 低层权重更大
            
            u_weighted = 0
            v_weighted = 0
            total_weight = 0
            
            # 2. 对每一层计算面积平均风
            for level, weight in zip(levels, weights):
                u_level = self._get_data_at_level("u", level, time_idx)
                v_level = self._get_data_at_level("v", level, time_idx)
                
                if u_level is None or v_level is None:
                    continue
                
                # 创建区域掩膜
                region_mask = self._create_region_mask(tc_lat, tc_lon, radius_deg)
                
                # 面积平均
                u_mean = np.nanmean(u_level[region_mask])
                v_mean = np.nanmean(v_level[region_mask])
                
                u_weighted += weight * u_mean
                v_weighted += weight * v_mean
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            # 3. 归一化
            u_steering = u_weighted / total_weight
            v_steering = v_weighted / total_weight
            
            # 4. 计算速度和方向
            speed = np.sqrt(u_steering**2 + v_steering**2)
            
            # 风向: 风吹向的方向 (气象惯例)
            direction = (np.degrees(np.arctan2(u_steering, v_steering)) + 180) % 360
            
            return {
                "speed": float(speed),
                "direction": float(direction),
                "u": float(u_steering),
                "v": float(v_steering),
                "method": "layer_averaged_wind_850-300hPa"
            }
            
        except Exception as e:
            print(f"⚠️ 层平均引导气流计算失败: {e}")
            return None
    
    def _extract_ridge_line(self, z500, tc_lat, tc_lon, threshold=5880):
        """
        提取副高脊线位置(588线的东西端点)
        
        Args:
            z500: 500hPa位势高度场
            tc_lat: 台风中心纬度
            tc_lon: 台风中心经度
            threshold: 脊线阈值 (默认5880gpm)
        
        Returns:
            脊线信息字典，包含东西端点位置，或None
        """
        try:
            # 1. 提取等值线
            contours = find_contours(z500, threshold)
            
            if not contours or len(contours) == 0:
                return None
            
            # 2. 选择最长的等值线 (主脊线)
            main_contour = sorted(contours, key=len, reverse=True)[0]
            
            # 3. 转换为地理坐标
            contour_indices_lat = main_contour[:, 0].astype(int)
            contour_indices_lon = main_contour[:, 1].astype(int)
            
            # 确保索引有效
            contour_indices_lat = np.clip(contour_indices_lat, 0, len(self.lat) - 1)
            contour_indices_lon = np.clip(contour_indices_lon, 0, len(self.lon) - 1)
            
            contour_lons = self.lon[contour_indices_lon]
            contour_lats = self.lat[contour_indices_lat]
            
            # 4. 找到脊线的东西端点
            # 归一化经度到台风中心附近
            contour_lons_normalized = self._normalize_longitude(contour_lons, tc_lon)
            
            # 东端 (最大经度)
            east_idx = np.argmax(contour_lons_normalized)
            east_lon = float(contour_lons[east_idx])
            east_lat = float(contour_lats[east_idx])
            
            # 西端 (最小经度)
            west_idx = np.argmin(contour_lons_normalized)
            west_lon = float(contour_lons[west_idx])
            west_lat = float(contour_lats[west_idx])
            
            # 5. 计算脊线相对于台风的位置
            _, east_bearing = self._calculate_bearing(tc_lat, tc_lon, east_lat, east_lon)
            _, west_bearing = self._calculate_bearing(tc_lat, tc_lon, west_lat, west_lon)
            
            return {
                "east_end": {
                    "latitude": round(east_lat, 2),
                    "longitude": round(east_lon, 2),
                    "relative_position": east_bearing
                },
                "west_end": {
                    "latitude": round(west_lat, 2),
                    "longitude": round(west_lon, 2),
                    "relative_position": west_bearing
                },
                "threshold_gpm": threshold,
                "description": f"588线从{west_bearing}延伸至{east_bearing}"
            }
            
        except Exception as e:
            print(f"⚠️ 脊线提取失败: {e}")
            return None
    
    def _extract_local_boundary_coords(self, z500, tc_lat, tc_lon, threshold=5880, radius_deg=20, max_points=50):
        """
        在局部区域内提取副高边界坐标
        
        Args:
            z500: 500hPa位势高度场
            tc_lat: 台风中心纬度
            tc_lon: 台风中心经度
            threshold: 等值线阈值 (默认5880gpm)
            radius_deg: 局部区域半径（度），默认20度
            max_points: 最大返回点数
        
        Returns:
            边界坐标列表 [[lon, lat], ...] 或 None
        """
        try:
            # 1. 定义局部区域范围
            lat_min = max(tc_lat - radius_deg, self.lat.min())
            lat_max = min(tc_lat + radius_deg, self.lat.max())
            lon_min = tc_lon - radius_deg
            lon_max = tc_lon + radius_deg
            
            # 2. 创建区域掩膜
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            
            # 处理经度跨越0°/360°的情况
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
            
            # 3. 提取局部数据
            local_z500 = z500[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            
            # 4. 在局部数据上提取等值线
            boundary_coords = self._get_contour_coords_local(
                local_z500, threshold, local_lat, local_lon, tc_lon, max_points
            )
            
            return boundary_coords
            
        except Exception as e:
            print(f"⚠️ 局部边界提取失败: {e}")
            return None
    
    def _extract_closed_boundary_with_features(self, z500, tc_lat, tc_lon, threshold, 
                                               lat_range=20.0, lon_range=40.0, 
                                               target_points=50):
        """
        提取闭合边界并标注关键特征点（科学安全的方法）
        
        改进点:
        1. 使用连通区域标注确保边界闭合
        2. 自适应采样保留关键形态特征
        3. 自动识别并标注关键特征点
        4. 多重回退机制确保稳定性
        
        Args:
            z500: 500hPa位势高度场
            tc_lat: 台风中心纬度
            tc_lon: 台风中心经度
            threshold: 等值线阈值
            lat_range: 纬度范围（默认20度）
            lon_range: 经度范围（默认40度）
            target_points: 目标采样点数（默认50）
        
        Returns:
            dict: {
                "boundary_coordinates": [[lon, lat], ...],  # 闭合边界坐标
                "boundary_features": {
                    "extreme_points": {...},  # 极值点
                    "ridge_intersections": [...],  # 脊线交点
                    "curvature_extremes": [...],  # 曲率极值点
                    "tc_relative_points": {...}  # 相对台风的关键点
                },
                "boundary_metrics": {
                    "is_closed": bool,
                    "total_points": int,
                    "perimeter_km": float,
                    "angle_coverage_deg": float,
                    ...
                }
            }
        """
        try:
            from skimage.measure import label, find_contours
            from scipy.spatial.distance import cdist
            
            # 第1步: 定义局部区域并提取数据
            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2
            
            # 创建区域掩膜
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            
            # 处理经度跨越0°/360°
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
            
            # 提取局部数据
            local_z500 = z500[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            
            if local_z500.size == 0:
                print(f"⚠️ 局部区域无数据")
                return None
            
            # 第2步: 使用连通区域标注方法提取闭合边界（科学方法）
            boundary_coords = None
            method_used = None
            
            # 方法1: 连通区域标注（最优方法）
            try:
                # 创建二值掩膜
                mask = (local_z500 >= threshold).astype(int)
                
                # 标注连通区域
                labeled = label(mask, connectivity=2)
                
                if labeled.max() == 0:
                    raise ValueError("未找到连通区域")
                
                # 找到包含台风周围的连通区域（距台风中心最近的区域）
                tc_lat_idx = np.argmin(np.abs(local_lat - tc_lat))
                tc_lon_idx = np.argmin(np.abs(local_lon - tc_lon))
                
                # 获取台风附近的标签
                target_label = labeled[tc_lat_idx, tc_lon_idx]
                
                if target_label == 0:
                    # 如果台风位置不在高压区，选择最大连通区域
                    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
                    target_label = unique[np.argmax(counts)]
                
                # 提取该连通区域的外轮廓
                contours = find_contours((labeled == target_label).astype(float), 0.5)
                
                if contours and len(contours) > 0:
                    # 选择最长的轮廓（外边界）
                    main_contour = sorted(contours, key=len, reverse=True)[0]
                    boundary_coords = main_contour
                    method_used = "connected_component_labeling"
                    
            except Exception as e:
                print(f"⚠️ 连通区域方法失败: {e}，尝试方法2")
            
            # 方法2: 扩大区域重试（回退方法）
            if boundary_coords is None:
                try:
                    # 扩大到30°x60°
                    expanded_result = self._extract_closed_boundary_with_features(
                        z500, tc_lat, tc_lon, threshold,
                        lat_range=30.0, lon_range=60.0, target_points=target_points
                    )
                    if expanded_result:
                        expanded_result["boundary_metrics"]["method_note"] = "使用扩大区域(30x60)"
                        return expanded_result
                        
                except Exception as e:
                    print(f"⚠️ 扩大区域方法失败: {e}，尝试方法3")
            
            # 方法3: 原find_contours方法（最后兜底）
            if boundary_coords is None:
                try:
                    contours = find_contours(local_z500, threshold)
                    if contours and len(contours) > 0:
                        boundary_coords = sorted(contours, key=len, reverse=True)[0]
                        method_used = "direct_contour_extraction"
                except Exception as e:
                    print(f"⚠️ 所有方法均失败: {e}")
                    return None
            
            if boundary_coords is None or len(boundary_coords) == 0:
                return None
            
            # 第3步: 将像素坐标转换为地理坐标
            geo_coords = []
            for point in boundary_coords:
                lat_idx = int(np.clip(point[0], 0, len(local_lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(local_lon) - 1))
                
                lat_val = float(local_lat[lat_idx])
                lon_val = float(local_lon[lon_idx])
                
                # 归一化经度
                lon_normalized = self._normalize_longitude(np.array([lon_val]), tc_lon)[0]
                if lon_normalized < 0:
                    lon_normalized += 360
                    
                geo_coords.append([lon_normalized, lat_val])
            
            # 第4步: 智能采样（保留关键特征）
            sampled_coords = self._adaptive_boundary_sampling(
                geo_coords, target_points=target_points
            )
            
            # 第5步: 确保闭合（如果首尾距离>阈值，添加闭合点）
            if len(sampled_coords) > 2:
                first = sampled_coords[0]
                last = sampled_coords[-1]
                closure_dist = np.sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
                
                if closure_dist > 1.0:  # 如果首尾距离>1度，添加首点形成闭合
                    sampled_coords.append(first)
            
            # 第6步: 提取关键特征点
            features = self._extract_boundary_features(
                sampled_coords, tc_lat, tc_lon, threshold
            )
            
            # 第7步: 计算边界度量
            metrics = self._calculate_boundary_metrics(
                sampled_coords, tc_lat, tc_lon, method_used
            )
            
            # 返回完整结果
            return {
                "boundary_coordinates": sampled_coords,
                "boundary_features": features,
                "boundary_metrics": metrics
            }
            
        except Exception as e:
            print(f"⚠️ 闭合边界提取完全失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _adaptive_boundary_sampling(self, coords, target_points=50, method="auto"):
        """
        智能自适应边界采样
        
        支持三种采样策略:
        1. curvature: 基于曲率的自适应采样（高曲率区域密集采样）
        2. perimeter: 基于周长的比例采样（均匀分布）
        3. douglas_peucker: 道格拉斯-普克算法（保留关键点）
        
        Args:
            coords: 原始坐标列表 [[lon, lat], ...]
            target_points: 目标点数
            method: 采样方法 ("auto", "curvature", "perimeter", "douglas_peucker")
        
        Returns:
            采样后的坐标列表
        """
        if len(coords) <= target_points:
            return coords
        
        # 自动选择最佳方法
        if method == "auto":
            perimeter_deg = self._calculate_perimeter(coords)
            
            # 小系统（周长<50°）使用曲率方法
            if perimeter_deg < 50:
                method = "curvature"
            # 大系统使用道格拉斯-普克算法
            else:
                method = "douglas_peucker"
        
        # 方法1: 基于曲率的自适应采样
        if method == "curvature":
            return self._curvature_adaptive_sampling(coords, target_points)
        
        # 方法2: 基于周长的比例采样
        elif method == "perimeter":
            return self._perimeter_proportional_sampling(coords, target_points)
        
        # 方法3: 道格拉斯-普克算法
        elif method == "douglas_peucker":
            return self._douglas_peucker_sampling(coords, target_points)
        
        # 默认: 等间隔采样
        else:
            step = max(1, len(coords) // target_points)
            return coords[::step]
    
    def _curvature_adaptive_sampling(self, coords, target_points):
        """基于曲率的自适应采样（高曲率区域密集采样）"""
        if len(coords) < 3:
            return coords
        
        # 计算每个点的曲率
        curvatures = []
        for i in range(len(coords)):
            prev_idx = (i - 1) % len(coords)
            next_idx = (i + 1) % len(coords)
            
            p1 = np.array(coords[prev_idx])
            p2 = np.array(coords[i])
            p3 = np.array(coords[next_idx])
            
            # 使用Menger曲率公式
            v1 = p2 - p1
            v2 = p3 - p2
            
            cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p3 - p1)
            
            if denom > 1e-10:
                curvature = cross / denom
            else:
                curvature = 0.0
            
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        # 基于曲率分配采样权重
        # 归一化曲率到[0.5, 1.5]范围
        if curvatures.max() > 1e-10:
            weights = 0.5 + (curvatures / curvatures.max())
        else:
            weights = np.ones_like(curvatures)
        
        # 累积权重
        cum_weights = np.cumsum(weights)
        cum_weights = cum_weights / cum_weights[-1]  # 归一化到[0, 1]
        
        # 均匀采样累积权重空间
        target_weights = np.linspace(0, 1, target_points, endpoint=False)
        
        # 找到最接近的索引
        sampled_indices = []
        for tw in target_weights:
            idx = np.argmin(np.abs(cum_weights - tw))
            if idx not in sampled_indices:  # 避免重复
                sampled_indices.append(idx)
        
        sampled_indices = sorted(sampled_indices)
        return [coords[i] for i in sampled_indices]
    
    def _perimeter_proportional_sampling(self, coords, target_points):
        """基于周长的比例采样（沿周长均匀分布）"""
        if len(coords) < 2:
            return coords
        
        # 计算累积距离
        distances = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(distances[-1] + dist)
        
        total_dist = distances[-1]
        if total_dist < 1e-10:
            # 所有点重合，返回第一个点
            return [coords[0]]
        
        # 沿周长均匀采样
        target_distances = np.linspace(0, total_dist, target_points, endpoint=False)
        
        sampled_coords = []
        for td in target_distances:
            # 找到距离最接近的点
            idx = np.argmin(np.abs(np.array(distances) - td))
            sampled_coords.append(coords[idx])
        
        return sampled_coords
    
    def _douglas_peucker_sampling(self, coords, target_points):
        """道格拉斯-普克算法（保留关键特征点）"""
        if len(coords) <= target_points:
            return coords
        
        # 简化版道格拉斯-普克: 迭代移除最不重要的点
        current_coords = coords.copy()
        
        while len(current_coords) > target_points:
            min_importance = float('inf')
            min_idx = -1
            
            # 计算每个点的重要性（到前后点连线的距离）
            for i in range(1, len(current_coords) - 1):
                p1 = np.array(current_coords[i-1])
                p2 = np.array(current_coords[i])
                p3 = np.array(current_coords[i+1])
                
                # 点到线段的距离
                importance = self._point_to_line_distance(p2, p1, p3)
                
                if importance < min_importance:
                    min_importance = importance
                    min_idx = i
            
            # 移除最不重要的点
            if min_idx > 0:
                current_coords.pop(min_idx)
            else:
                break
        
        return current_coords
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的垂直距离"""
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        
        ab = b - a
        ap = p - a
        
        if np.linalg.norm(ab) < 1e-10:
            return np.linalg.norm(ap)
        
        # 投影比例
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        
        # 最近点
        closest = a + t * ab
        
        return np.linalg.norm(p - closest)
    
    def _calculate_perimeter(self, coords):
        """计算边界周长（度）"""
        if len(coords) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(coords)):
            next_idx = (i + 1) % len(coords)
            dx = coords[next_idx][0] - coords[i][0]
            dy = coords[next_idx][1] - coords[i][1]
            perimeter += np.sqrt(dx**2 + dy**2)
        
        return perimeter
    
    def _extract_boundary_features(self, coords, tc_lat, tc_lon, threshold):
        """提取边界关键特征点"""
        if not coords or len(coords) < 4:
            return {}
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # 1. 极值点
        north_idx = np.argmax(lats)
        south_idx = np.argmin(lats)
        east_idx = np.argmax(lons)
        west_idx = np.argmin(lons)
        
        extreme_points = {
            "north": {
                "lon": round(lons[north_idx], 2),
                "lat": round(lats[north_idx], 2),
                "index": north_idx
            },
            "south": {
                "lon": round(lons[south_idx], 2),
                "lat": round(lats[south_idx], 2),
                "index": south_idx
            },
            "east": {
                "lon": round(lons[east_idx], 2),
                "lat": round(lats[east_idx], 2),
                "index": east_idx
            },
            "west": {
                "lon": round(lons[west_idx], 2),
                "lat": round(lats[west_idx], 2),
                "index": west_idx
            }
        }
        
        # 2. 相对台风的关键点
        distances = []
        for lon, lat in coords:
            dist = self._haversine_distance(tc_lat, tc_lon, lat, lon)
            distances.append(dist)
        
        nearest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)
        
        tc_relative_points = {
            "nearest": {
                "lon": round(lons[nearest_idx], 2),
                "lat": round(lats[nearest_idx], 2),
                "index": nearest_idx,
                "distance_km": round(distances[nearest_idx], 1)
            },
            "farthest": {
                "lon": round(lons[farthest_idx], 2),
                "lat": round(lats[farthest_idx], 2),
                "index": farthest_idx,
                "distance_km": round(distances[farthest_idx], 1)
            }
        }
        
        # 3. 曲率极值点（找出最凸和最凹的点）
        curvature_extremes = []
        if len(coords) >= 5:
            curvatures = []
            for i in range(len(coords)):
                prev_idx = (i - 1) % len(coords)
                next_idx = (i + 1) % len(coords)
                
                p1 = np.array(coords[prev_idx])
                p2 = np.array(coords[i])
                p3 = np.array(coords[next_idx])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p3 - p1)
                
                if denom > 1e-10:
                    curvature = cross / denom
                else:
                    curvature = 0.0
                
                curvatures.append((i, curvature))
            
            # 找出曲率最大和最小的点（各取前2个）
            curvatures_sorted = sorted(curvatures, key=lambda x: abs(x[1]), reverse=True)
            
            for i, curv in curvatures_sorted[:4]:  # 取前4个高曲率点
                if abs(curv) > 0.01:  # 只记录显著的曲率点
                    curvature_extremes.append({
                        "lon": round(lons[i], 2),
                        "lat": round(lats[i], 2),
                        "index": i,
                        "curvature": round(curv, 4),
                        "type": "凸出" if curv > 0 else "凹陷"
                    })
        
        return {
            "extreme_points": extreme_points,
            "tc_relative_points": tc_relative_points,
            "curvature_extremes": curvature_extremes
        }
    
    def _calculate_boundary_metrics(self, coords, tc_lat, tc_lon, method_used):
        """计算边界度量指标"""
        if not coords or len(coords) < 2:
            return {}
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # 检查闭合性
        first = coords[0]
        last = coords[-1]
        closure_dist = np.sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
        is_closed = closure_dist < 1.0
        
        # 计算周长（km）
        perimeter_km = 0.0
        for i in range(len(coords)):
            next_idx = (i + 1) % len(coords) if is_closed else min(i + 1, len(coords) - 1)
            if next_idx != i:
                dist_km = self._haversine_distance(
                    lats[i], lons[i], lats[next_idx], lons[next_idx]
                )
                perimeter_km += dist_km
        
        # 计算方位角覆盖
        center_lon = np.mean(lons)
        center_lat = np.mean(lats)
        
        angles = []
        for lon, lat in coords:
            angle = np.arctan2(lat - center_lat, lon - center_lon) * 180 / np.pi
            angles.append(angle)
        
        angle_coverage = max(angles) - min(angles) if angles else 0
        if is_closed:
            angle_coverage = 360.0
        
        # 平均点间距
        avg_spacing_km = perimeter_km / len(coords) if len(coords) > 0 else 0
        
        # 长宽比
        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)
        aspect_ratio = lon_span / lat_span if lat_span > 0 else 0
        
        return {
            "is_closed": bool(is_closed),  # 转换为Python bool
            "total_points": int(len(coords)),  # 转换为Python int
            "perimeter_km": round(float(perimeter_km), 1),
            "avg_point_spacing_km": round(float(avg_spacing_km), 1),
            "angle_coverage_deg": round(float(angle_coverage), 1),
            "closure_distance_deg": round(float(closure_dist), 2),
            "aspect_ratio": round(float(aspect_ratio), 2),
            "lon_span_deg": round(float(lon_span), 2),
            "lat_span_deg": round(float(lat_span), 2),
            "extraction_method": method_used or "unknown"
        }

    def _extract_closed_ocean_boundary_with_features(self, sst, tc_lat, tc_lon, threshold=26.5, 
                                                      lat_range=20.0, lon_range=40.0, 
                                                      target_points=50):
        """
        提取海洋热含量闭合边界并标注关键特征点（专用于SST场）
        
        改进点:
        1. 使用连通区域标注确保26.5°C等温线边界闭合
        2. 曲率自适应采样保留暖涡/冷涡特征
        3. 自动识别并标注关键特征点（极值点、暖涡中心、相对台风位置）
        4. 多重回退机制确保稳定性
        
        技术特点:
        - 复用Steering系统的成功经验（90%代码复用）
        - 针对SST场特性优化（暖水区识别、暖涡提取）
        - 三重安全机制：连通标注 → 扩大区域 → 原始方法
        
        Args:
            sst: 海表温度场 (2D array)
            tc_lat: 台风中心纬度
            tc_lon: 台风中心经度
            threshold: 等温线阈值（默认26.5°C，台风发展最低海温门槛）
            lat_range: 纬度范围（默认20度）
            lon_range: 经度范围（默认40度）
            target_points: 目标采样点数（默认50）
        
        Returns:
            dict: {
                "boundary_coordinates": [[lon, lat], ...],  # 闭合边界坐标
                "boundary_features": {
                    "extreme_points": {...},  # 4个极值点（最北/南/东/西）
                    "warm_eddy_centers": [...],  # 暖涡中心（凸出部分）
                    "cold_intrusion_points": [...],  # 冷涡侵入点（凹陷部分）
                    "curvature_extremes": [...],  # 曲率极值点
                    "tc_relative_points": {...}  # 相对台风的关键点（最近/最远）
                },
                "boundary_metrics": {
                    "is_closed": bool,  # 边界是否闭合
                    "total_points": int,  # 总点数
                    "perimeter_km": float,  # 周长（公里）
                    "angle_coverage_deg": float,  # 方位角覆盖度（度）
                    "warm_water_area_approx_km2": float,  # 暖水区近似面积
                    ...
                }
            }
        """
        try:
            from skimage.measure import label, find_contours
            from scipy.spatial.distance import cdist
            
            # 第1步: 定义局部区域并提取数据
            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2
            
            # 创建区域掩膜
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            
            # 处理经度跨越0°/360°
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
            
            # 提取局部SST数据
            local_sst = sst[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            
            if local_sst.size == 0:
                print(f"⚠️ 局部区域无SST数据")
                return None
            
            # 第2步: 使用连通区域标注方法提取闭合边界（科学方法）
            boundary_coords = None
            method_used = None
            
            # 方法1: 连通区域标注（最优方法）
            try:
                # 创建二值掩膜（SST >= 26.5°C的暖水区）
                mask = (local_sst >= threshold).astype(int)
                
                # 标注连通区域
                labeled = label(mask, connectivity=2)
                
                if labeled.max() == 0:
                    raise ValueError("未找到暖水连通区域")
                
                # 找到包含台风的连通区域（距台风中心最近的暖水区）
                tc_lat_idx = np.argmin(np.abs(local_lat - tc_lat))
                tc_lon_idx = np.argmin(np.abs(local_lon - tc_lon))
                
                # 获取台风位置的标签
                target_label = labeled[tc_lat_idx, tc_lon_idx]
                
                if target_label == 0:
                    # 如果台风位置不在暖水区，选择最大连通区域
                    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
                    target_label = unique[np.argmax(counts)]
                
                # 提取该连通区域的外轮廓
                contours = find_contours((labeled == target_label).astype(float), 0.5)
                
                if contours and len(contours) > 0:
                    # 选择最长的轮廓（外边界）
                    main_contour = sorted(contours, key=len, reverse=True)[0]
                    boundary_coords = main_contour
                    method_used = "connected_component_labeling"
                    print(f"✅ 方法1成功: 连通区域标注提取到{len(main_contour)}个点")
                    
            except Exception as e:
                print(f"⚠️ 连通区域方法失败: {e}，尝试方法2")
            
            # 方法2: 扩大区域重试（回退方法）
            if boundary_coords is None:
                try:
                    print(f"🔄 方法2: 扩大区域到30°x60°")
                    # 扩大到30°x60°
                    expanded_result = self._extract_closed_ocean_boundary_with_features(
                        sst, tc_lat, tc_lon, threshold,
                        lat_range=30.0, lon_range=60.0, target_points=target_points
                    )
                    if expanded_result:
                        expanded_result["boundary_metrics"]["method_note"] = "使用扩大区域(30x60)"
                        return expanded_result
                        
                except Exception as e:
                    print(f"⚠️ 扩大区域方法失败: {e}，尝试方法3")
            
            # 方法3: 原find_contours方法（最后兜底）
            if boundary_coords is None:
                try:
                    print(f"🔄 方法3: 使用原始find_contours方法")
                    contours = find_contours(local_sst, threshold)
                    if contours and len(contours) > 0:
                        boundary_coords = sorted(contours, key=len, reverse=True)[0]
                        method_used = "direct_contour_extraction"
                        print(f"✅ 方法3成功: 提取到{len(boundary_coords)}个点")
                except Exception as e:
                    print(f"⚠️ 所有方法均失败: {e}")
                    return None
            
            if boundary_coords is None or len(boundary_coords) == 0:
                return None
            
            # 第3步: 将像素坐标转换为地理坐标
            geo_coords = []
            for point in boundary_coords:
                lat_idx = int(np.clip(point[0], 0, len(local_lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(local_lon) - 1))
                
                lat_val = float(local_lat[lat_idx])
                lon_val = float(local_lon[lon_idx])
                
                # 归一化经度
                lon_normalized = self._normalize_longitude(np.array([lon_val]), tc_lon)[0]
                if lon_normalized < 0:
                    lon_normalized += 360
                    
                geo_coords.append([lon_normalized, lat_val])
            
            # 第4步: 智能采样（保留暖涡/冷涡特征）
            sampled_coords = self._adaptive_boundary_sampling(
                geo_coords, target_points=target_points, method="curvature"
            )
            
            # 第5步: 确保闭合（如果首尾距离>阈值，添加闭合点）
            if len(sampled_coords) > 2:
                first = sampled_coords[0]
                last = sampled_coords[-1]
                closure_dist = np.sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
                
                if closure_dist > 1.0:  # 如果首尾距离>1度，添加首点形成闭合
                    sampled_coords.append(first)
                    print(f"🔒 边界闭合: 添加首点，闭合距离从{closure_dist:.2f}°降至0")
            
            # 第6步: 提取关键特征点（针对海洋热含量特性）
            features = self._extract_ocean_boundary_features(
                sampled_coords, tc_lat, tc_lon, threshold
            )
            
            # 第7步: 计算边界度量
            metrics = self._calculate_boundary_metrics(
                sampled_coords, tc_lat, tc_lon, method_used
            )
            
            # 额外计算暖水区近似面积（使用Green定理）
            metrics["warm_water_area_approx_km2"] = self._calculate_polygon_area_km2(sampled_coords)
            
            # 返回完整结果
            return {
                "boundary_coordinates": sampled_coords,
                "boundary_features": features,
                "boundary_metrics": metrics
            }
            
        except Exception as e:
            print(f"⚠️ OceanHeat闭合边界提取完全失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_ocean_boundary_features(self, coords, tc_lat, tc_lon, threshold):
        """
        提取海洋热含量边界的关键特征点
        
        针对SST边界的特殊处理:
        - 暖涡中心: 边界向外凸出的部分（高曲率凸点）
        - 冷涡侵入: 边界向内凹陷的部分（高曲率凹点）
        - 相对台风位置: 最近点（台风可能驶离）、最远点（暖水区延伸方向）
        
        Returns:
            dict: 包含各类特征点的字典
        """
        if not coords or len(coords) < 3:
            return {}
        
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
        
        # 1. 四个极值点（地理位置极值）
        north_idx = np.argmax(lats)
        south_idx = np.argmin(lats)
        east_idx = np.argmax(lons)
        west_idx = np.argmin(lons)
        
        extreme_points = {
            "northernmost": {"lon": float(lons[north_idx]), "lat": float(lats[north_idx])},
            "southernmost": {"lon": float(lons[south_idx]), "lat": float(lats[south_idx])},
            "easternmost": {"lon": float(lons[east_idx]), "lat": float(lats[east_idx])},
            "westernmost": {"lon": float(lons[west_idx]), "lat": float(lats[west_idx])},
        }
        
        # 2. 相对台风的关键点
        distances = [self._haversine_distance(tc_lat, tc_lon, lat, lon) 
                    for lon, lat in coords]
        
        nearest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)
        
        tc_relative_points = {
            "nearest_to_tc": {
                "lon": float(lons[nearest_idx]),
                "lat": float(lats[nearest_idx]),
                "distance_km": round(float(distances[nearest_idx]), 1),
                "description": "台风到暖水区边界的最短距离"
            },
            "farthest_from_tc": {
                "lon": float(lons[farthest_idx]),
                "lat": float(lats[farthest_idx]),
                "distance_km": round(float(distances[farthest_idx]), 1),
                "description": "暖水区延伸的最远点"
            }
        }
        
        # 3. 曲率极值点（暖涡和冷涡特征）
        curvature_extremes = []
        warm_eddy_centers = []
        cold_intrusion_points = []
        
        if len(coords) >= 5:
            # 计算每个点的Menger曲率
            curvatures = []
            for i in range(len(coords)):
                prev_idx = (i - 2) % len(coords)
                next_idx = (i + 2) % len(coords)
                
                p1 = np.array([lons[prev_idx], lats[prev_idx]])
                p2 = np.array([lons[i], lats[i]])
                p3 = np.array([lons[next_idx], lats[next_idx]])
                
                # Menger曲率公式
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                
                if a * b * c > 1e-10:
                    curvature = 4 * area / (a * b * c)
                else:
                    curvature = 0
                
                curvatures.append(curvature)
            
            curvatures = np.array(curvatures)
            
            # 找到局部极大值（高曲率点）
            high_curvature_threshold = np.percentile(curvatures, 90)
            high_curv_indices = np.where(curvatures > high_curvature_threshold)[0]
            
            for idx in high_curv_indices[:5]:  # 最多5个
                # 判断是凸出（暖涡）还是凹陷（冷涡）
                # 简化判断：相对台风中心的距离变化
                dist_to_tc = self._haversine_distance(tc_lat, tc_lon, lats[idx], lons[idx])
                avg_dist = np.mean(distances)
                
                point_info = {
                    "lon": float(lons[idx]),
                    "lat": float(lats[idx]),
                    "curvature": round(float(curvatures[idx]), 6)
                }
                
                if dist_to_tc > avg_dist * 1.1:
                    # 凸出部分 - 可能是暖涡中心
                    warm_eddy_centers.append({
                        **point_info,
                        "type": "warm_eddy",
                        "description": "暖水区向外延伸的暖涡"
                    })
                elif dist_to_tc < avg_dist * 0.9:
                    # 凹陷部分 - 可能是冷水侵入
                    cold_intrusion_points.append({
                        **point_info,
                        "type": "cold_intrusion",
                        "description": "冷水向暖水区侵入"
                    })
                
                curvature_extremes.append(point_info)
        
        return {
            "extreme_points": extreme_points,
            "warm_eddy_centers": warm_eddy_centers[:3],  # 最多3个暖涡
            "cold_intrusion_points": cold_intrusion_points[:3],  # 最多3个冷涡
            "curvature_extremes": curvature_extremes[:5],  # 最多5个高曲率点
            "tc_relative_points": tc_relative_points
        }
    
    def _calculate_polygon_area_km2(self, coords):
        """
        使用Green定理计算多边形面积（近似，适用于小区域）
        
        Args:
            coords: [[lon, lat], ...] 坐标列表
        
        Returns:
            float: 面积（平方公里）
        """
        if not coords or len(coords) < 3:
            return 0.0
        
        # 转换为米制坐标（近似）
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
        
        # 中心点
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # 转换为相对米坐标
        x_m = (lons - center_lon) * 111000 * np.cos(np.radians(center_lat))
        y_m = (lats - center_lat) * 111000
        
        # Shoelace公式
        area_m2 = 0.5 * abs(sum(x_m[i]*y_m[i+1] - x_m[i+1]*y_m[i] 
                                for i in range(len(x_m)-1)))
        
        # 转换为km²
        area_km2 = area_m2 / 1e6
        
        return round(float(area_km2), 1)

    # ================= 新增: 流式顺序处理函数 =================

