"""Shape analytics for identifying and describing weather systems."""

from __future__ import annotations

import numpy as np

from .deps import (
    approximate_polygon,
    center_of_mass,
    find_contours,
    label,
)


class WeatherSystemShapeAnalyzer:
    """气象系统形状分析器 - 简化版，仅提取边界坐标."""

    def __init__(self, lat_grid, lon_grid, enable_detailed_analysis=True):
        """
        初始化形状分析器
        
        Args:
            lat_grid: 纬度网格
            lon_grid: 经度网格
            enable_detailed_analysis: 保留参数以兼容旧代码，但不再使用
        """
        self.lat = lat_grid
        self.lon = lon_grid
        self.enable_detailed_analysis = enable_detailed_analysis

    def analyze_system_shape(
        self, data_field, threshold, system_type="high", center_lat=None, center_lon=None
    ):
        """分析气象系统的形状，仅提取边界坐标信息.
        
        返回值包含：
        - boundary_coordinates: 简化后的边界坐标点列表
        - polygon_features: 多边形顶点、边界框、中心点等基本几何信息
        """
        try:
            if system_type == "high":
                mask = data_field >= threshold
            else:
                mask = data_field <= threshold

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

            # 提取边界坐标
            boundary_coords = self._extract_boundary_coordinates(data_field, threshold, system_type)
            if boundary_coords is None:
                return None

            return {
                "boundary_coordinates": boundary_coords["simplified_coordinates"],
                "polygon_features": boundary_coords["polygon_features"],
                "description": boundary_coords["description"],
            }

        except Exception as exc:
            print(f"形状分析失败: {exc}")
            return None

    def _select_main_system(self, labeled_mask, num_features, center_lat, center_lon):
        """选择主要的天气系统区域."""
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

    def _extract_boundary_coordinates(self, data_field, threshold, system_type):
        """提取系统的边界坐标."""
        try:
            contours = find_contours(data_field, threshold)
            if not contours:
                return None

            main_contour = max(contours, key=len)

            contour_lats = self.lat[main_contour[:, 0].astype(int)]
            contour_lons = self.lon[main_contour[:, 1].astype(int)]

            # 简化轮廓点，保留约50个代表性点
            step = max(1, len(main_contour) // 50)
            simplified_contour = [
                [round(lon, 2), round(lat, 2)]
                for lat, lon in zip(contour_lats[::step], contour_lons[::step])
            ]

            # 提取多边形特征
            polygon_features = self._extract_polygon_coordinates(main_contour, data_field.shape)

            return {
                "simplified_coordinates": simplified_contour,
                "polygon_features": polygon_features,
                "description": f"边界包含{len(main_contour)}个数据点，简化为{len(simplified_contour)}个代表性点",
            }
        except Exception:
            return None

    def _extract_polygon_coordinates(self, contour, shape):
        """提取多边形顶点和边界信息."""
        try:
            epsilon = 0.02 * len(contour)
            approx_polygon = approximate_polygon(contour, tolerance=epsilon)

            polygon_coords = []
            for point in approx_polygon:
                lat_idx = int(np.clip(point[0], 0, len(self.lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(self.lon) - 1))
                polygon_coords.append([round(self.lon[lon_idx], 2), round(self.lat[lat_idx], 2)])

            if len(polygon_coords) > 0:
                lons = [coord[0] for coord in polygon_coords]
                lats = [coord[1] for coord in polygon_coords]
                bbox = [
                    round(min(lons), 2),
                    round(min(lats), 2),
                    round(max(lons), 2),
                    round(max(lats), 2),
                ]

                center = [round(np.mean(lons), 2), round(np.mean(lats), 2)]

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
        except Exception:
            return None
