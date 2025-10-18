"""引导气流与副高相关的提取逻辑。"""

from __future__ import annotations

import numpy as np

from ..deps import center_of_mass, find_contours, label


class SteeringExtractionMixin:
    def extract_steering_system(self, time_idx, tc_lat, tc_lon):
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            subtropical_high_obj = self._identify_subtropical_high_regional(z500, tc_lat, tc_lon, time_idx)
            if not subtropical_high_obj:
                subtropical_high_obj = self._identify_pressure_system(z500, tc_lat, tc_lon, "high", 5880)
                if not subtropical_high_obj:
                    return None

            enhanced_shape = self._get_enhanced_shape_info(z500, 5880, "high", tc_lat, tc_lon)

            steering_result = self._calculate_steering_flow_layered(time_idx, tc_lat, tc_lon)
            if not steering_result:
                steering_speed, steering_direction, u_steering, v_steering = self._calculate_steering_flow(
                    z500, tc_lat, tc_lon
                )
                steering_result = {
                    "speed": steering_speed,
                    "direction": steering_direction,
                    "u": u_steering,
                    "v": v_steering,
                    "method": "geostrophic_wind",
                }

            ridge_info = self._extract_ridge_line(z500, tc_lat, tc_lon)

            intensity_val = subtropical_high_obj["intensity"]["value"]
            if intensity_val > 5900:
                level = "强"
            elif intensity_val > 5880:
                level = "中等"
            else:
                level = "弱"
            subtropical_high_obj["intensity"]["level"] = level

            if enhanced_shape:
                subtropical_high_obj["shape"].update(
                    {
                        "detailed_analysis": enhanced_shape["detailed_analysis"],
                        "shape_type": enhanced_shape["shape_type"],
                        "orientation": enhanced_shape["orientation"],
                        "complexity": enhanced_shape["complexity"],
                    }
                )
                if "coordinate_info" in enhanced_shape:
                    subtropical_high_obj["shape"]["coordinate_details"] = enhanced_shape["coordinate_info"]

            if "extraction_info" in subtropical_high_obj and "dynamic_threshold" in subtropical_high_obj["extraction_info"]:
                dynamic_threshold = subtropical_high_obj["extraction_info"]["dynamic_threshold"]
            else:
                dynamic_threshold = 5880

            boundary_result = self._extract_closed_boundary_with_features(
                z500,
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
                boundary_coords = self._extract_local_boundary_coords(
                    z500, tc_lat, tc_lon, threshold=dynamic_threshold, radius_deg=20
                )
                if boundary_coords:
                    subtropical_high_obj["boundary_coordinates"] = boundary_coords
                    subtropical_high_obj["boundary_note"] = "使用旧方法（新方法失败）"

            high_pos = subtropical_high_obj["position"]["center_of_mass"]
            bearing, rel_pos_desc = self._calculate_bearing(tc_lat, tc_lon, high_pos["lat"], high_pos["lon"])
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
                            "calculation_method": steering_result.get("method", "unknown"),
                        },
                    },
                }
            )

            if ridge_info:
                subtropical_high_obj["properties"]["ridge_line"] = ridge_info
                subtropical_high_obj["properties"]["ridge_line"] = ridge_info

            return subtropical_high_obj
        except Exception:
            return None

    def _identify_subtropical_high_regional(self, z500, tc_lat, tc_lon, time_idx):  # noqa: ARG002
        try:
            lat_range = 20.0
            lon_range = 40.0

            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2

            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            lon_mask_raw = (self.lon >= lon_min) & (self.lon <= lon_max)

            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = lon_mask_raw

            region_z500 = z500[np.ix_(lat_mask, lon_mask)]

            z500_mean = np.nanmean(region_z500)
            z500_anomaly = region_z500 - z500_mean

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
            best_feature_idx = -1
            for i in range(1, num_features + 1):
                feature_mask = labeled_array == i
                area = np.sum(feature_mask)
                if area > max_area:
                    max_area = area
                    best_feature_idx = i

            if best_feature_idx == -1:
                return None

            target_mask = labeled_array == best_feature_idx
            com_y, com_x = center_of_mass(target_mask)

            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            pos_lat = local_lat[int(com_y)]
            pos_lon = local_lon[int(com_x)]
            intensity_val = np.max(region_z500[target_mask])

            return {
                "position": {
                    "center_of_mass": {
                        "lat": round(float(pos_lat), 2),
                        "lon": round(float(pos_lon), 2),
                    }
                },
                "intensity": {"value": round(float(intensity_val), 1), "unit": "gpm"},
                "shape": {},
                "extraction_info": {
                    "method": "regional_processing",
                    "region_extent": {"lat_range": [float(lat_min), float(lat_max)], "lon_range": [float(lon_min), float(lon_max)]},
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

            u_weighted = 0
            v_weighted = 0
            total_weight = 0

            for level, weight in zip(levels, weights):
                u_level = self._get_data_at_level("u", level, time_idx)
                v_level = self._get_data_at_level("v", level, time_idx)
                if u_level is None or v_level is None:
                    continue
                region_mask = self._create_region_mask(tc_lat, tc_lon, radius_deg)
                u_mean = np.nanmean(u_level[region_mask])
                v_mean = np.nanmean(v_level[region_mask])
                u_weighted += weight * u_mean
                v_weighted += weight * v_mean
                total_weight += weight

            if total_weight == 0:
                return None

            u_steering = u_weighted / total_weight
            v_steering = v_weighted / total_weight
            speed = np.sqrt(u_steering**2 + v_steering**2)
            direction = (np.degrees(np.arctan2(u_steering, v_steering)) + 180) % 360

            return {
                "speed": float(speed),
                "direction": float(direction),
                "u": float(u_steering),
                "v": float(v_steering),
                "method": "layer_averaged_wind_850-300hPa",
            }

        except Exception as exc:
            print(f"⚠️ 层平均引导气流计算失败: {exc}")
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

    def _extract_ridge_line(self, z500, tc_lat, tc_lon, threshold=5880):
        try:
            contours = find_contours(z500, threshold)
            if not contours or len(contours) == 0:
                return None
            main_contour = sorted(contours, key=len, reverse=True)[0]
            contour_indices_lat = np.clip(main_contour[:, 0].astype(int), 0, len(self.lat) - 1)
            contour_indices_lon = np.clip(main_contour[:, 1].astype(int), 0, len(self.lon) - 1)
            contour_lons = self.lon[contour_indices_lon]
            contour_lats = self.lat[contour_indices_lat]
            contour_lons_normalized = self._normalize_longitude(contour_lons, tc_lon)
            east_idx = np.argmax(contour_lons_normalized)
            east_lon = float(contour_lons[east_idx])
            east_lat = float(contour_lats[east_idx])
            west_idx = np.argmin(contour_lons_normalized)
            west_lon = float(contour_lons[west_idx])
            west_lat = float(contour_lats[west_idx])
            _, east_bearing = self._calculate_bearing(tc_lat, tc_lon, east_lat, east_lon)
            _, west_bearing = self._calculate_bearing(tc_lat, tc_lon, west_lat, west_lon)
            return {
                "east_end": {
                    "latitude": round(east_lat, 2),
                    "longitude": round(east_lon, 2),
                    "relative_position": east_bearing,
                },
                "west_end": {
                    "latitude": round(west_lat, 2),
                    "longitude": round(west_lon, 2),
                    "relative_position": west_bearing,
                },
                "threshold_gpm": threshold,
                "description": f"588线从{west_bearing}延伸至{east_bearing}",
            }

        except Exception as exc:
            print(f"⚠️ 脊线提取失败: {exc}")
            return None

    def _extract_local_boundary_coords(self, z500, tc_lat, tc_lon, threshold=5880, radius_deg=20, max_points=50):
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
            local_z500 = z500[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]
            boundary_coords = self._get_contour_coords_local(
                local_z500, threshold, local_lat, local_lon, tc_lon, max_points
            )
            return boundary_coords

        except Exception as exc:
            print(f"⚠️ 局部边界提取失败: {exc}")
            return None


__all__ = ["SteeringExtractionMixin"]
