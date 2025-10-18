"""与风场相关的环境提取模块。"""

from __future__ import annotations

import numpy as np


class WindFieldExtractionMixin:
    def extract_vertical_wind_shear(self, time_idx, tc_lat, tc_lon, radius_km=500):
        try:
            u200, v200 = self._get_data_at_level("u", 200, time_idx), self._get_data_at_level("v", 200, time_idx)
            u850, v850 = self._get_data_at_level("u", 850, time_idx), self._get_data_at_level("v", 850, time_idx)
            if any(x is None for x in [u200, v200, u850, v850]):
                return None

            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            u200_mean = np.nanmean(u200[circular_mask])
            v200_mean = np.nanmean(v200[circular_mask])
            u850_mean = np.nanmean(u850[circular_mask])
            v850_mean = np.nanmean(v850[circular_mask])

            shear_u = u200_mean - u850_mean
            shear_v = v200_mean - v850_mean
            shear_mag = np.sqrt(shear_u**2 + shear_v**2)

            if shear_mag < 5:
                level, impact = "弱", "非常有利于发展"
            elif shear_mag < 10:
                level, impact = "中等", "基本有利发展"
            else:
                level, impact = "强", "显著抑制发展"

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
                    "shear_vector_mps": {"u": round(shear_u, 2), "v": round(shear_v, 2)},
                    "calculation_method": f"面积平均于{radius_km}km圆域",
                },
            }
        except Exception:
            return None

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
            divergence[~np.isfinite(divergence)] = np.nan

            radius_km = 500
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
            divergence_masked = np.where(circular_mask, divergence, np.nan)
            div_val_raw = float(np.nanmean(divergence_masked))
            if not np.isfinite(div_val_raw):
                return None

            max_div_idx = np.nanargmax(divergence_masked)
            max_div_lat_idx, max_div_lon_idx = np.unravel_index(max_div_idx, divergence_masked.shape)
            max_div_lat = float(self.lat[max_div_lat_idx])
            max_div_lon = float(self.lon[max_div_lon_idx])
            max_div_value = float(divergence[max_div_lat_idx, max_div_lon_idx])

            distance_to_max = self._haversine_distance(tc_lat, tc_lon, max_div_lat, max_div_lon)

            def calculate_bearing(lat1, lon1, lat2, lon2):
                lat1_rad = np.deg2rad(lat1)
                lat2_rad = np.deg2rad(lat2)
                dlon_rad = np.deg2rad(lon2 - lon1)
                x = np.sin(dlon_rad) * np.cos(lat2_rad)
                y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
                bearing = np.rad2deg(np.arctan2(x, y))
                return (bearing + 360) % 360

            bearing = calculate_bearing(tc_lat, tc_lon, max_div_lat, max_div_lon)
            direction_names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
            direction_idx = int((bearing + 22.5) // 45) % 8
            direction = direction_names[direction_idx]

            div_val_raw = float(np.clip(div_val_raw, -5e-4, 5e-4))
            max_div_value = float(np.clip(max_div_value, -5e-4, 5e-4))
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
        except Exception:
            return None


__all__ = ["WindFieldExtractionMixin"]
