"""热带辐合带提取相关逻辑。"""

from __future__ import annotations

import numpy as np


class IntertropicalConvergenceZoneMixin:
    def extract_intertropical_convergence_zone(self, time_idx, tc_lat, tc_lon):
        try:
            u850 = self._get_data_at_level("u", 850, time_idx)
            v850 = self._get_data_at_level("v", 850, time_idx)
            if u850 is None or v850 is None:
                return None

            a = 6371000
            lat_rad = np.deg2rad(self.lat)
            lon_rad = np.deg2rad(self.lon)

            gy_u, gx_u = self._raw_gradients(u850)
            gy_v, gx_v = self._raw_gradients(v850)

            dlat = self.lat_spacing * np.pi / 180
            dlon = self.lon_spacing * np.pi / 180
            cos_lat = self._coslat_safe[:, np.newaxis]

            du_dlon = gx_u / dlon
            dv_dlat = gy_v / dlat

            divergence = (1 / (a * cos_lat)) * du_dlon + (1 / a) * (dv_dlat * cos_lat - v850 * np.sin(lat_rad)[:, np.newaxis])

            w700 = self._get_data_at_level("w", 700, time_idx)

            if tc_lat >= 0:
                lat_min, lat_max = 5, 20
                hemisphere = "北半球"
            else:
                lat_min, lat_max = -20, -5
                hemisphere = "南半球"

            tropical_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if not np.any(tropical_mask):
                return None

            convergence = -divergence
            tropical_conv = convergence[tropical_mask, :]
            conv_by_lat = np.nanmean(tropical_conv, axis=1)
            if not np.any(np.isfinite(conv_by_lat)):
                return None

            max_conv_idx = np.nanargmax(conv_by_lat)
            itcz_lat = self.lat[tropical_mask][max_conv_idx]
            max_convergence = conv_by_lat[max_conv_idx] * 1e5

            lat_idx = self._loc_idx(itcz_lat, tc_lon)[0]
            conv_at_lat = convergence[lat_idx, :]

            conv_threshold = (
                np.nanpercentile(conv_at_lat[conv_at_lat > 0], 50) if np.any(conv_at_lat > 0) else 0
            )
            strong_conv_mask = conv_at_lat > conv_threshold

            lon_ranges = []
            in_range = False
            start_lon = None
            for i, is_strong in enumerate(strong_conv_mask):
                if is_strong and not in_range:
                    start_lon = self.lon[i]
                    in_range = True
                elif not is_strong and in_range:
                    lon_ranges.append((start_lon, self.lon[i - 1]))
                    in_range = False
            if in_range:
                lon_ranges.append((start_lon, self.lon[-1]))

            best_range = None
            min_dist = float("inf")
            for lon_start, lon_end in lon_ranges:
                if lon_start <= tc_lon <= lon_end:
                    best_range = (lon_start, lon_end)
                    break
                dist = min(abs(tc_lon - lon_start), abs(tc_lon - lon_end))
                if dist < min_dist:
                    min_dist = dist
                    best_range = (lon_start, lon_end)

            distance_km = self._haversine_distance(tc_lat, tc_lon, itcz_lat, tc_lon)
            distance_deg = abs(tc_lat - itcz_lat)

            if distance_km < 500:
                influence = "直接影响台风发展"
                impact_level = "强"
            elif distance_km < 1000:
                influence = "对台风路径有显著影响"
                impact_level = "中"
            else:
                influence = "对台风影响较小"
                impact_level = "弱"

            if max_convergence > 5:
                conv_level = "强"
                conv_desc = "辐合活跃，有利于对流发展"
            elif max_convergence > 2:
                conv_level = "中等"
                conv_desc = "辐合中等，对对流有一定支持"
            else:
                conv_level = "弱"
                conv_desc = "辐合较弱"

            vertical_motion_desc = ""
            if w700 is not None:
                w_at_itcz = w700[lat_idx, :]
                mean_w = np.nanmean(w_at_itcz[strong_conv_mask]) if np.any(strong_conv_mask) else 0
                if mean_w < -0.05:
                    vertical_motion_desc = "，伴随强上升运动"
                elif mean_w < 0:
                    vertical_motion_desc = "，伴随上升运动"

            lon_range_str = f"{best_range[0]:.1f}°E-{best_range[1]:.1f}°E" if best_range else "跨经度带"

            desc = (
                f"{hemisphere}热带辐合带位于约{itcz_lat:.1f}°{'N' if itcz_lat >= 0 else 'S'}附近，"
                f"经度范围{lon_range_str}，"
                f"辐合强度{max_convergence:.2f}×10⁻⁵ s⁻¹（{conv_level}）{vertical_motion_desc}。"
                f"与台风中心距离{distance_km:.0f}公里（{distance_deg:.1f}度），{influence}。"
            )

            result = {
                "system_name": "InterTropicalConvergenceZone",
                "description": desc,
                "position": {
                    "description": "热带辐合带中心位置",
                    "lat": round(itcz_lat, 2),
                    "lon": tc_lon,
                    "lon_range": lon_range_str,
                },
                "intensity": {
                    "value": round(max_convergence, 2),
                    "unit": "×10⁻⁵ s⁻¹",
                    "level": conv_level,
                    "description": conv_desc,
                },
                "shape": {"description": "东西向延伸的辐合带", "type": "convergence_line"},
                "properties": {
                    "distance_to_tc_km": round(distance_km, 1),
                    "distance_to_tc_deg": round(distance_deg, 2),
                    "influence": influence,
                    "impact_level": impact_level,
                    "hemisphere": hemisphere,
                    "convergence_strength": conv_level,
                },
            }

            if best_range:
                sample_lons = [best_range[0], (best_range[0] + best_range[1]) / 2, best_range[1]]
                boundary_coords = [[lon, itcz_lat] for lon in sample_lons]
                result["boundary_coordinates"] = boundary_coords

            return result

        except Exception:
            return None


__all__ = ["IntertropicalConvergenceZoneMixin"]
