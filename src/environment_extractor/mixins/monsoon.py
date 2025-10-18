"""季风槽提取逻辑。"""

from __future__ import annotations

import numpy as np


class MonsoonTroughMixin:
    def extract_monsoon_trough(self, time_idx, tc_lat, tc_lon):
        try:
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if u850 is None or v850 is None:
                return None

            if tc_lat >= 0:
                lat_min, lat_max = 5, 25
                hemisphere = "北半球"
                expected_vort_sign = 1
            else:
                lat_min, lat_max = -25, -5
                hemisphere = "南半球"
                expected_vort_sign = -1

            search_radius_km = 1500
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)

            distance_mask = np.zeros((len(self.lat), len(self.lon)), dtype=bool)
            for i, lat in enumerate(self.lat):
                if not lat_mask[i]:
                    continue
                for j, lon in enumerate(self.lon):
                    dist = self._haversine_distance(tc_lat, tc_lon, lat, lon)
                    if dist <= search_radius_km:
                        distance_mask[i, j] = True

            if not np.any(distance_mask):
                return None

            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)
            du_dy = gy_u / (self.lat_spacing * 111000)
            dv_dx = gx_v / (self.lon_spacing * 111000 * self._coslat_safe[:, np.newaxis])
            relative_vorticity = dv_dx - du_dy
            with np.errstate(invalid="ignore"):
                relative_vorticity[~np.isfinite(relative_vorticity)] = np.nan

            if hemisphere == "南半球":
                relative_vorticity = -relative_vorticity

            masked_vort = np.where(distance_mask, relative_vorticity, np.nan)
            if not np.any(np.isfinite(masked_vort)):
                return None

            vort_threshold = np.nanpercentile(masked_vort[masked_vort > 0], 75) if np.any(masked_vort > 0) else 0
            if vort_threshold <= 0:
                return None

            trough_mask = masked_vort > vort_threshold
            if not np.any(trough_mask):
                return None

            max_vort_idx = np.unravel_index(np.nanargmax(masked_vort), masked_vort.shape)
            trough_bottom_lat = self.lat[max_vort_idx[0]]
            trough_bottom_lon = self.lon[max_vort_idx[1]]
            max_vorticity = masked_vort[max_vort_idx] * 1e5

            trough_lat_idx = max_vort_idx[0]
            vort_along_axis = masked_vort[trough_lat_idx, :]
            axis_threshold = vort_threshold * 0.7
            axis_mask = vort_along_axis > axis_threshold

            axis_lons = self.lon[axis_mask]
            if len(axis_lons) > 0:
                axis_lon_start = axis_lons[0]
                axis_lon_end = axis_lons[-1]
                axis_length_deg = axis_lon_end - axis_lon_start
                axis_length_km = axis_length_deg * 111 * np.cos(np.deg2rad(trough_bottom_lat))
            else:
                axis_lon_start = trough_bottom_lon
                axis_lon_end = trough_bottom_lon
                axis_length_km = 0

            u_at_trough = u850[trough_lat_idx, :]
            mean_u = np.nanmean(u_at_trough[axis_mask]) if np.any(axis_mask) else 0

            if mean_u > 2:
                wind_pattern = "西风为主"
                monsoon_confidence = "高"
            elif mean_u > 0:
                wind_pattern = "弱西风"
                monsoon_confidence = "中"
            else:
                wind_pattern = "东风分量"
                monsoon_confidence = "低"

            pressure_desc = ""
            try:
                mslp = self._get_data_at_level("msl", None, time_idx)
                if mslp is not None and not isinstance(mslp, tuple):
                    mslp_at_trough = mslp[trough_lat_idx, :]
                    if np.any(axis_mask):
                        mean_mslp = float(np.nanmean(mslp_at_trough[axis_mask]))
                        mean_mslp_hpa = mean_mslp / 100
                        pressure_desc = f"，气压约{mean_mslp_hpa:.0f} hPa"
            except Exception:
                pass

            distance_to_trough = self._haversine_distance(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)
            bearing, direction = self._calculate_bearing(tc_lat, tc_lon, trough_bottom_lat, trough_bottom_lon)

            if distance_to_trough < 500:
                influence = "台风位于季风槽内或紧邻，受水汽输送直接影响"
                impact_level = "强"
            elif distance_to_trough < 1000:
                influence = "台风受季风槽环流影响，水汽条件较好"
                impact_level = "中"
            else:
                influence = "季风槽对台风影响有限"
                impact_level = "弱"

            if max_vorticity > 10:
                vort_level = "强"
                vort_desc = "季风槽活跃，有利于台风发展"
            elif max_vorticity > 5:
                vort_level = "中等"
                vort_desc = "季风槽中等强度"
            else:
                vort_level = "弱"
                vort_desc = "季风槽较弱"

            desc = (
                f"在台风{direction}约{distance_to_trough:.0f}公里处检测到{hemisphere}季风槽，"
                f"槽底位于{trough_bottom_lat:.1f}°{'N' if trough_bottom_lat >= 0 else 'S'}, "
                f"{trough_bottom_lon:.1f}°E，"
                f"槽轴长度约{axis_length_km:.0f}公里，"
                f"最大涡度{max_vorticity:.1f}×10⁻⁵ s⁻¹（{vort_level}），"
                f"低层{wind_pattern}{pressure_desc}。{influence}。"
            )

            result = {
                "system_name": "MonsoonTrough",
                "description": desc,
                "position": {
                    "description": "季风槽槽底位置",
                    "lat": round(trough_bottom_lat, 2),
                    "lon": round(trough_bottom_lon, 2),
                },
                "intensity": {
                    "value": round(max_vorticity, 2),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": vort_level,
                    "description": vort_desc,
                },
                "shape": {
                    "description": f"东西向延伸的低压槽，长度约{axis_length_km:.0f}公里",
                    "type": "trough_axis",
                    "axis_length_km": round(axis_length_km, 1),
                },
                "properties": {
                    "distance_to_tc_km": round(distance_to_trough, 1),
                    "direction_from_tc": direction,
                    "bearing": round(bearing, 1),
                    "influence": influence,
                    "impact_level": impact_level,
                    "hemisphere": hemisphere,
                    "vorticity_level": vort_level,
                    "zonal_wind_pattern": wind_pattern,
                    "monsoon_confidence": monsoon_confidence,
                    "axis_lon_range": f"{axis_lon_start:.1f}°E - {axis_lon_end:.1f}°E",
                },
            }

            if axis_length_km > 0:
                boundary_coords = [
                    [axis_lon_start, trough_bottom_lat],
                    [trough_bottom_lon, trough_bottom_lat],
                    [axis_lon_end, trough_bottom_lat],
                ]
                result["boundary_coordinates"] = boundary_coords

            return result

        except Exception:
            return None


__all__ = ["MonsoonTroughMixin"]
