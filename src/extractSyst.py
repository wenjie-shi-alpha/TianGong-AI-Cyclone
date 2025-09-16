#!/usr/bin/env python3
"""
热带气旋环境场影响系统提取器（专家解译版）
基于已追踪的热带气旋路径，提取并详细解译影响其移动和强度的关键天气系统，
输出包含气象描述、定性分级和形状坐标的结构化JSON文件。
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import math
import warnings
import re

# 引入新的库用于图像处理和等值线提取
try:
    from scipy.ndimage import label, center_of_mass, find_objects, binary_erosion, binary_dilation
    from skimage.measure import find_contours, regionprops, approximate_polygon
    from skimage.morphology import convex_hull_image
    from scipy.spatial.distance import pdist
    from scipy.spatial import ConvexHull
except ImportError:
    print("错误：需要scipy和scikit-image库。请运行 'pip install scipy scikit-image' 进行安装。")
    exit()

warnings.filterwarnings("ignore")


class WeatherSystemShapeAnalyzer:
    """
    气象系统形状分析器
    专门用于分析气象系统的几何形状特征
    """

    def __init__(self, lat_grid, lon_grid):
        self.lat = lat_grid
        self.lon = lon_grid
        self.lat_spacing = np.abs(np.diff(lat_grid).mean())
        self.lon_spacing = np.abs(np.diff(lon_grid).mean())

    def analyze_system_shape(
        self, data_field, threshold, system_type="high", center_lat=None, center_lon=None
    ):
        """
        全面分析气象系统的形状特征

        Parameters:
        -----------
        data_field : numpy.ndarray
            二维气象数据场
        threshold : float
            用于定义系统边界的阈值
        system_type : str
            系统类型 ('high' 或 'low')
        center_lat, center_lon : float
            系统中心位置（用于计算相对特征）

        Returns:
        --------
        dict : 包含详细形状特征的字典
        """
        try:
            # 1. 创建二值掩膜
            if system_type == "high":
                mask = data_field >= threshold
            else:
                mask = data_field <= threshold

            if not np.any(mask):
                return None

            # 2. 连通区域分析
            labeled_mask, num_features = label(mask)
            if num_features == 0:
                return None

            # 3. 选择主要系统（最大或离中心最近的）
            main_region = self._select_main_system(
                labeled_mask, num_features, center_lat, center_lon
            )
            if main_region is None:
                return None

            # 4. 计算基础几何特征
            basic_features = self._calculate_basic_features(
                main_region, data_field, threshold, system_type
            )

            # 5. 计算形状复杂度特征
            complexity_features = self._calculate_complexity_features(main_region)

            # 6. 计算方向性特征
            orientation_features = self._calculate_orientation_features(main_region)

            # 7. 提取等值线特征
            contour_features = self._extract_contour_features(data_field, threshold, system_type)

            # 8. 计算多尺度特征
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

        except Exception as e:
            print(f"形状分析失败: {e}")
            return None

    def _select_main_system(self, labeled_mask, num_features, center_lat, center_lon):
        """选择主要的气象系统区域"""
        if center_lat is None or center_lon is None:
            # 选择最大的连通区域 (使用 bincount 等价加速)
            flat_labels = labeled_mask.ravel()
            counts = np.bincount(flat_labels)[1: num_features + 1]
            if counts.size == 0:
                return None
            main_label = int(np.argmax(counts) + 1)
        else:
            # 选择离指定中心最近的区域
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
        """计算基础几何特征"""
        # 使用regionprops进行高级形状分析
        props = regionprops(region_mask.astype(int), intensity_image=data_field)[0]

        # 计算实际的地理面积（km²）
        area_pixels = props.area
        area_km2 = (
            area_pixels
            * (self.lat_spacing * 111)
            * (self.lon_spacing * 111 * np.cos(np.deg2rad(np.mean(self.lat))))
        )

        # 计算周长（km）
        perimeter_pixels = props.perimeter
        perimeter_km = perimeter_pixels * np.sqrt(
            (self.lat_spacing * 111) ** 2 + (self.lon_spacing * 111) ** 2
        )

        # 计算紧凑度和形状指数
        compactness = 4 * np.pi * area_km2 / (perimeter_km**2) if perimeter_km > 0 else 0
        shape_index = perimeter_km / (2 * np.sqrt(np.pi * area_km2)) if area_km2 > 0 else 0

        # 计算长宽比和偏心率
        major_axis_length = props.major_axis_length * self.lat_spacing * 111  # km
        minor_axis_length = props.minor_axis_length * self.lat_spacing * 111  # km
        aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 1
        eccentricity = props.eccentricity

        # 计算强度统计
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
        """计算形状复杂度特征"""
        # 计算凸包
        convex_hull = convex_hull_image(region_mask)
        convex_area = np.sum(convex_hull)
        actual_area = np.sum(region_mask)

        # 凸性（solidity）
        solidity = actual_area / convex_area if convex_area > 0 else 0

        # 计算边界粗糙度
        contours = find_contours(region_mask.astype(float), 0.5)
        if contours:
            main_contour = max(contours, key=len)
            # 使用多边形近似来评估边界复杂度
            epsilon = 0.02 * len(main_contour)
            approx_contour = approximate_polygon(main_contour, tolerance=epsilon)
            boundary_complexity = (
                len(main_contour) / len(approx_contour) if len(approx_contour) > 0 else 1
            )
        else:
            boundary_complexity = 1

        # 分形维数近似
        fractal_dimension = self._estimate_fractal_dimension(region_mask)

        return {
            "solidity": round(solidity, 3),
            "boundary_complexity": round(boundary_complexity, 2),
            "fractal_dimension": round(fractal_dimension, 3),
            "description": self._describe_complexity(solidity, boundary_complexity),
        }

    def _calculate_orientation_features(self, region_mask):
        """计算方向性特征"""
        props = regionprops(region_mask.astype(int))[0]

        # 主轴方向角（弧度转度）
        orientation_rad = props.orientation
        orientation_deg = np.degrees(orientation_rad)

        # 标准化到0-180度
        if orientation_deg < 0:
            orientation_deg += 180

        # 确定主要延伸方向
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
        """提取等值线特征"""
        try:
            contours = find_contours(data_field, threshold)
            if not contours:
                return None

            # 选择最长的等值线
            main_contour = max(contours, key=len)

            # 转换为地理坐标
            contour_lats = self.lat[main_contour[:, 0].astype(int)]
            contour_lons = self.lon[main_contour[:, 1].astype(int)]

            # 计算等值线长度
            contour_length_km = 0
            for i in range(1, len(contour_lats)):
                dist = self._haversine_distance(
                    contour_lats[i - 1], contour_lons[i - 1], contour_lats[i], contour_lons[i]
                )
                contour_length_km += dist

            # 降采样等值线点以减少数据量
            step = max(1, len(main_contour) // 50)
            simplified_contour = [
                [round(lon, 2), round(lat, 2)]
                for lat, lon in zip(contour_lats[::step], contour_lons[::step])
            ]

            # 提取多边形坐标特征
            polygon_features = self._extract_polygon_coordinates(main_contour, data_field.shape)

            return {
                "contour_length_km": round(contour_length_km, 1),
                "contour_points": len(main_contour),
                "simplified_coordinates": simplified_contour,
                "polygon_features": polygon_features,
                "description": f"主等值线长度{contour_length_km:.0f}km，包含{len(main_contour)}个数据点",
            }
        except Exception:
            return None

    def _extract_polygon_coordinates(self, contour, shape):
        """提取多边形关键坐标点"""
        try:
            # 使用多边形近似来获取关键角点
            epsilon = 0.02 * len(contour)
            approx_polygon = approximate_polygon(contour, tolerance=epsilon)

            # 转换为地理坐标
            polygon_coords = []
            for point in approx_polygon:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                polygon_coords.append([round(self.lon[lon_idx], 2), round(self.lat[lat_idx], 2)])

            # 计算边界框
            if len(polygon_coords) > 0:
                lons = [coord[0] for coord in polygon_coords]
                lats = [coord[1] for coord in polygon_coords]
                bbox = [
                    round(min(lons), 2),
                    round(min(lats), 2),
                    round(max(lons), 2),
                    round(max(lats), 2),
                ]  # [west, south, east, north]

                # 计算中心点
                center = [round(np.mean(lons), 2), round(np.mean(lats), 2)]

                # 提取关键方向点
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
                    ],  # [lon_span, lat_span]
                }

            return None
        except Exception as e:
            return None

    def _calculate_multiscale_features(self, data_field, threshold, system_type):
        """计算多尺度特征"""
        features = {}

        # 定义多个阈值水平
        if system_type == "high":
            thresholds = [threshold, threshold + 20, threshold + 40]
            threshold_names = ["外边界", "中等强度", "强中心"]
        else:
            thresholds = [threshold, threshold - 20, threshold - 40]
            threshold_names = ["外边界", "中等强度", "强中心"]

        for i, (thresh, name) in enumerate(zip(thresholds, threshold_names)):
            if system_type == "high":
                mask = data_field >= thresh
            else:
                mask = data_field <= thresh

            if np.any(mask):
                area_pixels = np.sum(mask)
                area_km2 = (
                    area_pixels
                    * (self.lat_spacing * 111)
                    * (self.lon_spacing * 111 * np.cos(np.deg2rad(np.mean(self.lat))))
                )
                features[f"area_{name}_km2"] = round(area_km2, 1)
            else:
                features[f"area_{name}_km2"] = 0

        # 计算嵌套比例
        if features.get("area_外边界_km2", 0) > 0:
            features["core_ratio"] = round(
                features.get("area_强中心_km2", 0) / features["area_外边界_km2"], 3
            )
            features["middle_ratio"] = round(
                features.get("area_中等强度_km2", 0) / features["area_外边界_km2"], 3
            )

        return features

    def _describe_basic_shape(self, compactness, aspect_ratio, eccentricity):
        """描述基本形状特征"""
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
        """描述复杂度特征"""
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
        """估算分形维数（简化方法）"""
        try:
            # 使用盒计数法的简化版本
            sizes = [2, 4, 8, 16]
            counts = []

            for size in sizes:
                # 将图像分割成不同大小的盒子
                h, w = region_mask.shape
                count = 0
                for i in range(0, h, size):
                    for j in range(0, w, size):
                        box = region_mask[i : min(i + size, h), j : min(j + size, w)]
                        if np.any(box):
                            count += 1
                counts.append(count)

            # 计算分形维数
            if len(counts) > 1 and all(c > 0 for c in counts):
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                fractal_dim = -coeffs[0]
                return max(1.0, min(2.0, fractal_dim))  # 限制在合理范围内
            else:
                return 1.5  # 默认值
        except:
            return 1.5

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """计算两点间的球面距离（km）"""
        R = 6371.0  # 地球半径
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


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

    def extract_vertical_wind_shear(self, time_idx, tc_lat, tc_lon):
        """
        [深度重构] 提取并解译垂直风切变。
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

            lat_idx, lon_idx = (
                np.abs(self.lat - tc_lat).argmin(),
                np.abs(self.lon - tc_lon).argmin(),
            )
            shear_u = u200[lat_idx, lon_idx] - u850[lat_idx, lon_idx]
            shear_v = v200[lat_idx, lon_idx] - v850[lat_idx, lon_idx]
            shear_mag = np.sqrt(shear_u**2 + shear_v**2)

            if shear_mag < 5:
                level, impact = "弱", "非常有利于发展"
            elif shear_mag < 10:
                level, impact = "中等", "基本有利发展"
            else:
                level, impact = "强", "显著抑制发展"

            # 方向定义为风从哪个方向来
            direction_from = (np.degrees(np.arctan2(shear_u, shear_v)) + 180) % 360
            dir_desc, _ = self._bearing_to_desc(direction_from)

            desc = (
                f"台风核心区正受到来自{dir_desc}方向、强度为“{level}”的垂直风切变影响，"
                f"当前风切变环境对台风的发展{impact.split(' ')[-1]}。"
            )

            return {
                "system_name": "VerticalWindShear",
                "description": desc,
                "position": {
                    "description": "在台风中心点计算的200-850hPa风矢量差",
                    "lat": tc_lat,
                    "lon": tc_lon,
                },
                "intensity": {"value": round(shear_mag.item(), 2), "unit": "m/s", "level": level},
                "shape": {
                    "description": f"一个从{dir_desc}指向的矢量",
                    "vector_coordinates": self._get_vector_coords(tc_lat, tc_lon, shear_u, shear_v),
                },
                "properties": {
                    "direction_from_deg": round(direction_from.item(), 1),
                    "impact": impact,
                    "shear_vector_mps": {
                        "u": round(shear_u.item(), 2),
                        "v": round(shear_v.item(), 2),
                    },
                },
            }
        except Exception as e:
            # print(f"⚠️ 垂直风切变提取失败: {e}")
            return None

    def extract_ocean_heat_content(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        """
        [深度重构] 提取并解译海洋热含量（海表温度SST近似）。
        """
        try:
            sst = self._get_sst_field(time_idx)
            if sst is None:
                return None

            region_mask = self._create_region_mask(tc_lat, tc_lon, radius_deg)
            sst_mean = np.nanmean(sst[region_mask])

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

            contour_26_5 = self._get_contour_coords(sst, 26.5, tc_lon)

            # 增强形状分析：分析暖水区域形状
            enhanced_shape = self._get_enhanced_shape_info(sst, 26.5, "high", tc_lat, tc_lon)

            shape_info = {
                "description": "26.5°C是台风发展的最低海温门槛，此线是生命线",
                "warm_water_boundary_26.5C": contour_26_5,
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
    def analyze_and_export_as_json(self, output_dir="final_output"):
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
def streaming_from_csv(
    csv_path: Path,
    limit: int | None = None,
    search_range: float = 3.0,
    memory: int = 3,
    keep_nc: bool = False,
    initials_csv: Path | None = None,
):
    """逐行读取CSV, 每个NC文件执行: 下载 -> 追踪 -> 环境分析 -> (可选删除)

    与原批量模式最大区别: 不预先下载全部; 每个文件完成后即可释放磁盘。
    """
    if not csv_path.exists():
        print(f"❌ CSV不存在: {csv_path}")
        return
    import pandas as pd, re, traceback
    from trackTC import sanitize_filename, download_s3_public
    # initialTracker 提供基于初始点的追踪算法
    from initialTracker import track_file_with_initials as it_track_file_with_initials
    # 兼容: initialTracker 中提供的是 _load_all_points，这里用同名别名引用
    from initialTracker import _load_all_points as it_load_initial_points

    df = pd.read_csv(csv_path)
    required_cols = {"s3_url", "model_prefix", "init_time"}
    if not required_cols.issubset(df.columns):
        print(f"❌ CSV缺少必要列: {required_cols - set(df.columns)}")
        return
    if limit is not None:
        df = df.head(limit)
    print(f"📄 流式待处理数量: {len(df)} (limit={limit})")

    persist_dir = Path("data/nc_files")  # 仍放入该目录, 便于复用逻辑
    persist_dir.mkdir(parents=True, exist_ok=True)
    track_dir = Path("track_test"); track_dir.mkdir(exist_ok=True)
    final_dir = Path("final_output"); final_dir.mkdir(exist_ok=True)

    processed = 0
    skipped = 0
    for idx, row in df.iterrows():
        s3_url = row["s3_url"]
        model_prefix = row["model_prefix"]
        init_time = row["init_time"]
        fname = Path(s3_url).name
        m = re.search(r"(f\d{3}_f\d{3}_\d{2})", Path(fname).stem)
        forecast_tag = m.group(1) if m else "track"
        safe_prefix = sanitize_filename(model_prefix)
        safe_init = sanitize_filename(init_time.replace(":", "").replace("-", ""))
        track_csv = track_dir / f"tracks_{safe_prefix}_{safe_init}_{forecast_tag}.csv"
        nc_local = persist_dir / fname

        print(f"\n[{idx+1}/{len(df)}] ▶️ 处理: {fname}")

        # 如果 final 已存在则跳过整个流程
        existing_json = list(final_dir.glob(f"{Path(fname).stem}_TC_Analysis_*.json"))
        if existing_json:
            non_empty = [p for p in existing_json if p.stat().st_size > 10]
            if non_empty:
                print(f"⏭️  已存在最终JSON({len(non_empty)}) -> 跳过")
                skipped += 1
                continue

        # 下载 (若不存在)
        if not nc_local.exists():
            try:
                print(f"⬇️  下载NC: {s3_url}")
                download_s3_public(s3_url, nc_local)
            except Exception as e:
                print(f"❌ 下载失败, 跳过: {e}")
                skipped += 1
                continue
        else:
            print("📦 已存在NC文件, 复用")

        # 轨迹: 若不存在则计算 (使用 initialTracker)
        if not track_csv.exists():
            try:
                print("🧭 使用 initialTracker 执行追踪...")
                # 加载初始点
                initials_path = initials_csv or Path("input/western_pacific_typhoons_superfast.csv")
                initials_df = it_load_initial_points(initials_path)
                # 针对当前 NC 运行追踪, initialTracker 会为每个风暴输出一个 CSV
                per_storm_csvs = it_track_file_with_initials(Path(nc_local), initials_df, track_dir)
                if not per_storm_csvs:
                    print("⚠️ 无有效轨迹 -> 跳过环境分析")
                    if not keep_nc:
                        try:
                            nc_local.unlink(); print("🧹 已删除NC (无轨迹)")
                        except Exception: pass
                    skipped += 1
                    continue

                # 合并为单一轨迹文件, 增加 particle 与 time_idx 列，便于后续提取
                try:
                    import xarray as _xr
                    ds_times = []
                    with _xr.open_dataset(nc_local) as _ds:
                        ds_times = pd.to_datetime(_ds.time.values) if "time" in _ds.coords else []
                    def _nearest_time_idx(ts: pd.Timestamp) -> int:
                        if len(ds_times) == 0:
                            return 0
                        # 精确匹配优先
                        try:
                            return int(np.argmin(np.abs(ds_times - ts)))
                        except Exception:
                            return 0
                    parts = []
                    for p in per_storm_csvs:
                        df_i = pd.read_csv(p)
                        # 解析 storm_id 自文件名: track_<storm>_<ncstem>.csv
                        s = Path(p).stem
                        m_id = re.match(r"track_(.+?)_" + re.escape(Path(nc_local).stem) + r"$", s)
                        particle_id = m_id.group(1) if m_id else s.replace("track_", "")
                        df_i["particle"] = particle_id
                        # 统一时间并生成 time_idx
                        if "time" in df_i.columns:
                            df_i["time"] = pd.to_datetime(df_i["time"], errors="coerce")
                            df_i["time_idx"] = df_i["time"].apply(lambda t: _nearest_time_idx(t) if pd.notnull(t) else 0)
                        else:
                            # 若缺少时间, 用顺序索引代替
                            df_i["time_idx"] = np.arange(len(df_i))
                            # 合成时间列(可选)
                        parts.append(df_i)
                    tracks_df = pd.concat(parts, ignore_index=True)
                    tracks_df.to_csv(track_csv, index=False)
                    print(f"💾 合并保存轨迹: {track_csv.name} (含 {tracks_df['particle'].nunique()} 条路径)")
                except Exception as ce:
                    print(f"❌ 合并轨迹失败: {ce}")
                    raise
            except Exception as e:
                print(f"❌ 追踪失败: {e}")
                traceback.print_exc()
                if not keep_nc:
                    try:
                        nc_local.unlink(); print("🧹 已删除NC (追踪失败)")
                    except Exception: pass
                skipped += 1
                continue
        else:
            print("🗺️  已存在轨迹CSV, 直接环境分析")

        # 环境分析
        try:
            extractor = TCEnvironmentalSystemsExtractor(str(nc_local), str(track_csv))
            extractor.analyze_and_export_as_json("final_output")
            processed += 1
        except Exception as e:
            print(f"❌ 环境分析失败: {e}")
        finally:
            if not keep_nc:
                try:
                    nc_local.unlink(); print("🧹 已删除NC文件")
                except Exception as ee:
                    print(f"⚠️ 删除NC失败: {ee}")

    print("\n📊 流式处理结果:")
    print(f"  ✅ 完成: {processed}")
    print(f"  ⏭️ 跳过: {skipped}")
    print(f"  📁 输出目录: final_output")


def main():
    import argparse, sys, subprocess

    parser = argparse.ArgumentParser(description="一体化: 下载->追踪->环境分析")
    parser.add_argument("--csv", default="output/nc_file_urls.csv", help="含s3_url的列表CSV")
    parser.add_argument("--limit", type=int, default=1, help="限制处理前N个NC文件")
    parser.add_argument("--nc", default=None, help="直接指定单个NC文件 (跳过下载与追踪)")
    parser.add_argument("--tracks", default=None, help="直接指定轨迹CSV (跳过追踪)\n若与--nc同时给出则只做环境分析")
    parser.add_argument("--no-clean", action="store_true", help="分析后不删除NC")
    parser.add_argument("--keep-nc", action="store_true", help="同 --no-clean (兼容)")
    parser.add_argument("--auto", action="store_true", help="无轨迹则自动运行追踪")
    parser.add_argument("--search-range", type=float, default=3.0, help="追踪搜索范围")
    parser.add_argument("--memory", type=int, default=3, help="追踪记忆时间步")
    parser.add_argument("--initials", default=str(Path("input")/"western_pacific_typhoons_superfast.csv"), help="initialTracker 初始点CSV")
    parser.add_argument("--batch", action="store_true", help="使用旧的批量模式: 先全部下载+追踪, 再统一做环境分析")
    args = parser.parse_args()

    print("🌀 一体化热带气旋分析流程启动")
    print("=" * 60)

    nc_file: Path | None = None
    track_file: Path | None = None

    # 1. 单文件直通模式 (--nc) 或 CSV 多文件顺序流式模式 (默认) / 旧批量模式 (--batch)
    if args.nc:
        nc_file = Path(args.nc)
        if not nc_file.exists():
            print(f"❌ 指定NC不存在: {nc_file}")
            sys.exit(1)
        target_nc_files = [nc_file]
        print("📦 单文件分析模式")
    else:
        if args.batch:
            # 旧批量模式: 兼容原逻辑
            from trackTC import process_from_csv
            print("⬇️ [批量模式] 先统一下载/追踪后再做环境分析 (limit=", args.limit, ")")
            process_from_csv(Path(args.csv), limit=args.limit)
            cache_dir = Path("data/nc_files")
            if not cache_dir.exists():
                print("❌ 没有找到 data/nc_files 目录")
                sys.exit(1)
            cached = sorted(cache_dir.glob("*.nc"))
            if not cached:
                print("❌ 未发现任何NC文件")
                sys.exit(1)
            target_nc_files = cached[: args.limit] if args.limit is not None else cached
            print(f"📦 待环境分析NC数量: {len(target_nc_files)}")
        else:
            # 新的流式顺序处理: 逐条CSV -> 下载 -> 追踪 -> 环境分析 -> (可选清理)
            print("🚚 启用流式顺序处理: 每个NC独立完成(下载->追踪->环境分析->清理)")
            streaming_from_csv(
                csv_path=Path(args.csv),
                limit=args.limit,
                search_range=args.search_range,
                memory=args.memory,
                keep_nc=(args.no_clean or args.keep_nc),
                initials_csv=Path(args.initials) if args.initials else None,
            )
            print("🎯 流式处理完成 (无需进入批量后处理循环)")
            return

    final_output_dir = Path("final_output")
    final_output_dir.mkdir(exist_ok=True)

    processed = 0
    skipped = 0
    for idx, nc_file in enumerate(target_nc_files, start=1):
        nc_stem = nc_file.stem
        print(f"\n[{idx}/{len(target_nc_files)}] ▶️ 处理 NC: {nc_file.name}")
        # 检查是否已有输出
        existing = list(final_output_dir.glob(f"{nc_stem}_TC_Analysis_*.json"))
        non_empty = [p for p in existing if p.stat().st_size > 10]
        if non_empty:
            print(f"⏭️  已存在分析结果 ({len(non_empty)}) -> 跳过 {nc_stem}")
            skipped += 1
            continue

        # 寻找匹配的轨迹文件 (优先 forecast_tag 匹配)
        track_file = None
        if args.tracks:
            t = Path(args.tracks)
            if t.exists():
                track_file = t
        if track_file is None:
            tdir = Path("track_output")
            if tdir.exists():
                forecast_tag_match = re.search(r"(f\d{3}_f\d{3}_\d{2})", nc_stem)
                potential = []
                if forecast_tag_match:
                    tag = forecast_tag_match.group(1)
                    potential = list(tdir.glob(f"tracks_*_{tag}.csv"))
                tracks_all = sorted(tdir.glob("tracks_*.csv"))
                if potential:
                    track_file = potential[0]
                elif tracks_all:
                    # 退化选择: 选第一个 (提示不精确)
                    track_file = tracks_all[0]
                    print(f"⚠️ 未精确匹配 forecast_tag, 使用 {track_file.name}")
        if track_file is None:
            if args.auto:
                # 使用 initialTracker 自动生成轨迹 (基于初始点)
                from initialTracker import track_file_with_initials as it_track_file_with_initials
                # 兼容: initialTracker 中提供的是 _load_all_points，这里用同名别名引用
                from initialTracker import _load_all_points as it_load_initial_points
                print("🔄 使用 initialTracker 自动追踪当前NC以生成轨迹...")
                try:
                    initials_df = it_load_initial_points(Path(args.initials) if args.initials else Path("input/western_pacific_typhoons_superfast.csv"))
                    out_dir = Path("track_output"); out_dir.mkdir(exist_ok=True)
                    per_storm = it_track_file_with_initials(Path(nc_file), initials_df, out_dir)
                    if not per_storm:
                        print("⚠️ 无轨迹 -> 跳过该NC")
                        skipped += 1
                        continue
                    # 合并
                    import xarray as _xr, re as _re
                    ds_times = []
                    with _xr.open_dataset(nc_file) as _ds:
                        ds_times = pd.to_datetime(_ds.time.values) if "time" in _ds.coords else []
                    def _nearest_idx(ts: pd.Timestamp) -> int:
                        if len(ds_times) == 0:
                            return 0
                        return int(np.argmin(np.abs(ds_times - ts)))
                    parts = []
                    for p in per_storm:
                        dfi = pd.read_csv(p)
                        s = Path(p).stem
                        mid = _re.match(r"track_(.+?)_" + _re.escape(nc_stem) + r"$", s)
                        pid = mid.group(1) if mid else s.replace("track_", "")
                        dfi["particle"] = pid
                        if "time" in dfi.columns:
                            dfi["time"] = pd.to_datetime(dfi["time"], errors="coerce")
                            dfi["time_idx"] = dfi["time"].apply(lambda t: _nearest_idx(t) if pd.notnull(t) else 0)
                        else:
                            dfi["time_idx"] = np.arange(len(dfi))
                        parts.append(dfi)
                    tracks_df = pd.concat(parts, ignore_index=True)
                    ts0 = pd.to_datetime(tracks_df.iloc[0]["time"]).strftime("%Y%m%d%H") if "time" in tracks_df.columns and pd.notnull(tracks_df.iloc[0]["time"]) else "T000"
                    track_file = out_dir / f"tracks_auto_{nc_stem}_{ts0}.csv"
                    tracks_df.to_csv(track_file, index=False)
                    print(f"💾 自动轨迹文件: {track_file.name}")
                except Exception as e:
                    print(f"❌ 自动追踪失败: {e}")
                    skipped += 1
                    continue
            else:
                print("⚠️ 未找到对应轨迹且未启用 --auto, 跳过")
                skipped += 1
                continue

        print(f"✅ 使用轨迹文件: {track_file}")
        try:
            extractor = TCEnvironmentalSystemsExtractor(str(nc_file), str(track_file))
            extractor.analyze_and_export_as_json("final_output")
            processed += 1
        except Exception as e:
            print(f"❌ 分析失败 {nc_file.name}: {e}")
            continue

        if not (args.no_clean or args.keep_nc):
            try:
                nc_file.unlink()
                print(f"🧹 已删除 NC: {nc_file.name}")
            except Exception as e:
                print(f"⚠️ 删除NC失败: {e}")
        else:
            print("ℹ️ 按参数保留NC文件")

    print("\n🎉 多文件环境分析完成. 统计:")
    print(f"  ✅ 已分析: {processed}")
    print(f"  ⏭️ 跳过(已有结果/无轨迹): {skipped}")
    print(f"  📦 总计遍历: {len(target_nc_files)}")
    print("结果目录: final_output")


if __name__ == "__main__":
    main()
