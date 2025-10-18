"""锋面系统提取逻辑。"""

from __future__ import annotations

import numpy as np


class FrontalSystemMixin:
    def extract_frontal_system(self, time_idx, tc_lat, tc_lon):
        try:
            t850 = self._get_data_at_level("t", 850, time_idx)
            t500 = self._get_data_at_level("t", 500, time_idx)
            t1000 = self._get_data_at_level("t", 1000, time_idx)
            u925 = self._get_data_at_level("u", 925, time_idx)
            v925 = self._get_data_at_level("v", 925, time_idx)

            if t850 is None or t500 is None:
                return None

            if np.nanmean(t850) > 200:
                t850 = t850 - 273.15
            if np.nanmean(t500) > 200:
                t500 = t500 - 273.15
            if t1000 is not None and np.nanmean(t1000) > 200:
                t1000 = t1000 - 273.15

            if t1000 is not None:
                thickness = t1000 - t500
            else:
                thickness = t850 - t500

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

            for field in [thickness_gradient, temp_gradient, frontogenesis]:
                if field is not None:
                    field[~np.isfinite(field)] = np.nan
            if wind_convergence is not None:
                wind_convergence[~np.isfinite(wind_convergence)] = np.nan

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

            frontal_index = (
                0.5 * norm_thickness_grad + 0.25 * norm_temp_grad + 0.15 * norm_frontogenesis + 0.10 * norm_convergence
            )

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

            frontal_coords = None
            try:
                contour_threshold = np.percentile(valid_values, 90)
                local_front_for_contour = np.where(
                    front_mask & (frontal_index_local > contour_threshold),
                    frontal_index_local,
                    np.nan,
                )
                frontal_coords = self._get_system_coordinates_local(
                    local_front_for_contour,
                    contour_threshold,
                    "high",
                    tc_lat,
                    tc_lon,
                    max_points=20,
                )
            except Exception:
                pass

            shape_info = {"description": f"线性的{front_type}带，基于厚度场梯度识别", "type": front_type}

            if frontal_coords:
                shape_info.update(
                    {
                        "coordinates": frontal_coords,
                        "extent_desc": (
                            f"锋面带跨越纬度{frontal_coords['span_deg'][1]:.1f}°，"
                            f"经度{frontal_coords['span_deg'][0]:.1f}°"
                        ),
                    }
                )
                desc += (
                    f" 锋面带主体跨越{frontal_coords['span_deg'][1]:.1f}°纬度和"
                    f"{frontal_coords['span_deg'][0]:.1f}°经度。"
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
                "shape": shape_info,
                "properties": {
                    "impact": "影响台风路径和结构",
                    "distance_to_tc_km": round(float(distance_to_tc), 1),
                    "front_type": front_type,
                    "search_radius_km": search_radius_km,
                },
            }
        except Exception:
            return None


__all__ = ["FrontalSystemMixin"]
