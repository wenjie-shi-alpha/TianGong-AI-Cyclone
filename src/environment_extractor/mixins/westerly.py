"""西风槽提取逻辑。"""

from __future__ import annotations

import numpy as np


class WesterlyTroughMixin:
    def extract_westerly_trough(self, time_idx, tc_lat, tc_lon):
        try:
            z500 = self._get_data_at_level("z", 500, time_idx)
            if z500 is None:
                return None

            z500_zonal_mean = np.nanmean(z500, axis=1, keepdims=True)
            z500_anomaly = z500 - z500_zonal_mean

            mid_lat_mask = (self.lat >= 20) & (self.lat <= 60)
            if not np.any(mid_lat_mask):
                return None

            u500 = self._get_data_at_level("u", 500, time_idx)
            v500 = self._get_data_at_level("v", 500, time_idx)

            if u500 is None or v500 is None:
                pv_gradient = None
            else:
                gy_u, gx_u = self._raw_gradients(u500)
                gy_v, gx_v = self._raw_gradients(v500)
                du_dy = gy_u / (self.lat_spacing * 111000)
                dv_dx = gx_v / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
                vorticity = dv_dx - du_dy
                omega = 7.2921e-5
                f = 2 * omega * np.sin(np.deg2rad(self.lat))[:, np.newaxis]
                abs_vorticity = vorticity + f
                gy_pv, gx_pv = self._raw_gradients(abs_vorticity)
                pv_gradient = np.sqrt(gy_pv**2 + gx_pv**2)

            z500_anomaly_mid = z500_anomaly.copy()
            z500_anomaly_mid[~mid_lat_mask, :] = np.nan

            negative_anomaly = z500_anomaly_mid < 0
            if not np.any(negative_anomaly):
                return None

            neg_values = z500_anomaly_mid[negative_anomaly]
            if len(neg_values) == 0:
                return None

            trough_threshold_anomaly = np.percentile(neg_values, 25)

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
            distance = self._haversine_distance(tc_lat, tc_lon, trough_center_lat, trough_center_lon)

            distance_bottom = self._haversine_distance(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)
            bearing_bottom, _ = self._calculate_bearing(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)

            trough_intensity = abs(trough_bottom_anomaly)

            if trough_intensity > 150:
                strength = "强"
            elif trough_intensity > 80:
                strength = "中等"
            else:
                strength = "弱"

            if distance < 1000:
                if distance_bottom < 500:
                    influence = "槽前西南气流直接影响台风路径和强度，可能促进台风向东北方向移动"
                    interaction_potential = "高"
                else:
                    influence = "直接影响台风路径和强度"
                    interaction_potential = "高"
            elif distance < 2000:
                influence = "对台风有间接影响，可能通过引导气流影响台风移动"
                interaction_potential = "中"
            else:
                influence = "影响较小"
                interaction_potential = "低"

            u200 = self._get_data_at_level("u", 200, time_idx)
            jet_info = None
            if u200 is not None:
                jet_mask = u200 > 30
                if np.any(jet_mask & local_mask):
                    jet_info = "检测到200hPa急流，确认为动力活跃的西风槽"

            desc = (
                f"在台风{rel_pos_desc}约{distance:.0f}公里处存在{strength}西风槽系统，"
                f"槽底位于({trough_bottom_lat:.1f}°N, {trough_bottom_lon:.1f}°E)，"
                f"距台风中心{distance_bottom:.0f}公里。"
            )

            desc += f"槽轴呈南北向延伸，跨越{len(trough_axis)}个采样点。"

            if jet_info:
                desc += jet_info + "。"

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

            if pv_gradient is not None:
                pv_grad_at_bottom = float(pv_gradient[trough_bottom_lat_idx, trough_bottom_lon_idx])
                shape_info["pv_gradient_at_bottom"] = float(f"{pv_grad_at_bottom:.2e}")

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
                    "interaction_potential": interaction_potential,
                    "jet_detected": jet_info is not None,
                },
            }
        except Exception:
            return None


__all__ = ["WesterlyTroughMixin"]
