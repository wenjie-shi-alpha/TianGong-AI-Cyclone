"""海洋热含量相关的提取逻辑。"""

from __future__ import annotations

import numpy as np


class OceanHeatContentMixin:
    def extract_ocean_heat_content(self, time_idx, tc_lat, tc_lon, radius_deg=2.0):
        try:
            sst = self._get_sst_field(time_idx)
            if sst is None:
                return None

            radius_km = radius_deg * 111
            circular_mask = self._create_circular_mask_haversine(tc_lat, tc_lon, radius_km)
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

            lat_idx, lon_idx = self._loc_idx(tc_lat, tc_lon)
            radius_points = int(radius_deg * 3 / self.lat_spacing)
            lat_start = max(0, lat_idx - radius_points)
            lat_end = min(len(self.lat), lat_idx + radius_points + 1)
            lon_start = max(0, lon_idx - radius_points)
            lon_end = min(len(self.lon), lon_idx + radius_points + 1)

            sst_local = sst[lat_start:lat_end, lon_start:lon_end]
            local_lat = self.lat[lat_start:lat_end]
            local_lon = self.lon[lon_start:lon_end]

            boundary_result = self._extract_closed_ocean_boundary_with_features(
                sst,
                tc_lat,
                tc_lon,
                threshold=26.5,
                lat_range=radius_deg * 6,
                lon_range=radius_deg * 12,
                target_points=50,
            )

            shape_info = {
                "description": "26.5°C是台风发展的最低海温门槛，此线是生命线",
                "boundary_type": "closed_contour_with_features",
                "extraction_radius_deg": radius_deg * 3,
            }

            if boundary_result:
                shape_info["warm_water_boundary_26.5C"] = boundary_result["boundary_coordinates"]
                shape_info["boundary_features"] = boundary_result["boundary_features"]
                shape_info["boundary_metrics"] = boundary_result["boundary_metrics"]

                metrics = boundary_result["boundary_metrics"]
                if "warm_water_area_approx_km2" in metrics:
                    shape_info["warm_water_area_km2"] = metrics["warm_water_area_approx_km2"]
                    desc += f" 暖水区域面积约{metrics['warm_water_area_approx_km2']:.0f}km²"

                if metrics.get("is_closed"):
                    desc += (
                        f"，边界完整闭合（{metrics['total_points']}个采样点，"
                        f"周长{metrics['perimeter_km']:.0f}km）"
                    )

                features = boundary_result["boundary_features"]
                tc_rel = features.get("tc_relative_points", {})
                if "nearest_to_tc" in tc_rel:
                    nearest_dist = tc_rel["nearest_to_tc"]["distance_km"]
                    desc += f"，台风距暖水区边界最近{nearest_dist:.0f}km"

                warm_eddies = features.get("warm_eddy_centers", [])
                if warm_eddies:
                    desc += f"，检测到{len(warm_eddies)}个暖涡特征"

            else:
                print("⚠️ 闭合边界提取失败，回退到旧方法")
                contour_26_5 = self._get_contour_coords_local(
                    sst_local, 26.5, local_lat, local_lon, tc_lon
                )
                shape_info["warm_water_boundary_26.5C"] = contour_26_5
                shape_info["boundary_type"] = "fallback_local_region"

                enhanced_shape = self._get_enhanced_shape_info(sst, 26.5, "high", tc_lat, tc_lon)
                if enhanced_shape:
                    shape_info.update(
                        {
                            "warm_water_area_km2": enhanced_shape["area_km2"],
                            "warm_region_shape": enhanced_shape["shape_type"],
                            "warm_region_orientation": enhanced_shape["orientation"],
                            "detailed_analysis": enhanced_shape["detailed_analysis"],
                        }
                    )
                    desc += (
                        f" 暖水区域面积约{enhanced_shape['area_km2']:.0f}km²，呈{enhanced_shape['shape_type']}，"
                        f"{enhanced_shape['orientation']}。"
                    )

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
        except Exception:
            return None

    def _extract_closed_ocean_boundary_with_features(
        self,
        sst,
        tc_lat,
        tc_lon,
        threshold=26.5,
        lat_range=20.0,
        lon_range=40.0,
        target_points=50,
    ):
        try:
            from skimage.measure import label as sk_label, find_contours as sk_find_contours

            lat_min = max(tc_lat - lat_range / 2, self.lat.min())
            lat_max = min(tc_lat + lat_range / 2, self.lat.max())
            lon_min = tc_lon - lon_range / 2
            lon_max = tc_lon + lon_range / 2

            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            if lon_min < 0:
                lon_mask = (self.lon >= lon_min + 360) | (self.lon <= lon_max)
            elif lon_max > 360:
                lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max - 360)
            else:
                lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)

            local_sst = sst[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]

            if local_sst.size == 0:
                print("⚠️ 局部区域无SST数据")
                return None

            boundary_coords = None
            method_used = None

            try:
                mask = (local_sst >= threshold).astype(int)
                labeled = sk_label(mask, connectivity=2)
                if labeled.max() == 0:
                    raise ValueError("未找到暖水连通区域")

                tc_lat_idx = np.argmin(np.abs(local_lat - tc_lat))
                tc_lon_idx = np.argmin(np.abs(local_lon - tc_lon))
                target_label = labeled[tc_lat_idx, tc_lon_idx]

                if target_label == 0:
                    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
                    target_label = unique[np.argmax(counts)]

                contours = sk_find_contours((labeled == target_label).astype(float), 0.5)
                if contours and len(contours) > 0:
                    main_contour = sorted(contours, key=len, reverse=True)[0]
                    boundary_coords = main_contour
                    method_used = "connected_component_labeling"
                    print(f"✅ 方法1成功: 连通区域标注提取到{len(main_contour)}个点")

            except Exception as exc:
                print(f"⚠️ 连通区域方法失败: {exc}，尝试方法2")

            if boundary_coords is None:
                try:
                    print("🔄 方法2: 扩大区域到30°x60°")
                    expanded_result = self._extract_closed_ocean_boundary_with_features(
                        sst,
                        tc_lat,
                        tc_lon,
                        threshold,
                        lat_range=30.0,
                        lon_range=60.0,
                        target_points=target_points,
                    )
                    if expanded_result:
                        expanded_result["boundary_metrics"]["method_note"] = "使用扩大区域(30x60)"
                        return expanded_result
                except Exception as exc:
                    print(f"⚠️ 扩大区域方法失败: {exc}，尝试方法3")

            if boundary_coords is None:
                try:
                    print("🔄 方法3: 使用原始find_contours方法")
                    contours = sk_find_contours(local_sst, threshold)
                    if contours and len(contours) > 0:
                        boundary_coords = sorted(contours, key=len, reverse=True)[0]
                        method_used = "direct_contour_extraction"
                        print(f"✅ 方法3成功: 提取到{len(boundary_coords)}个点")
                except Exception as exc:
                    print(f"⚠️ 所有方法均失败: {exc}")
                    return None

            if boundary_coords is None or len(boundary_coords) == 0:
                return None

            geo_coords = []
            for point in boundary_coords:
                lat_idx = int(np.clip(point[0], 0, len(local_lat) - 1))
                lon_idx = int(np.clip(point[1], 0, len(local_lon) - 1))
                lat_val = float(local_lat[lat_idx])
                lon_val = float(local_lon[lon_idx])
                lon_normalized = self._normalize_longitude(np.array([lon_val]), tc_lon)[0]
                if lon_normalized < 0:
                    lon_normalized += 360
                geo_coords.append([lon_normalized, lat_val])

            sampled_coords = self._adaptive_boundary_sampling(geo_coords, target_points=target_points, method="curvature")

            if len(sampled_coords) > 2:
                first = sampled_coords[0]
                last = sampled_coords[-1]
                closure_dist = np.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)
                if closure_dist > 1.0:
                    sampled_coords.append(first)
                    print(f"🔒 边界闭合: 添加首点，闭合距离从{closure_dist:.2f}°降至0")

            features = self._extract_ocean_boundary_features(sampled_coords, tc_lat, tc_lon, threshold)
            metrics = self._calculate_boundary_metrics(sampled_coords, tc_lat, tc_lon, method_used)
            metrics["warm_water_area_approx_km2"] = self._calculate_polygon_area_km2(sampled_coords)

            return {
                "boundary_coordinates": sampled_coords,
                "boundary_features": features,
                "boundary_metrics": metrics,
            }

        except Exception as exc:
            print(f"⚠️ OceanHeat闭合边界提取完全失败: {exc}")
            import traceback as _traceback  # noqa: WPS433

            _traceback.print_exc()
            return None

    def _extract_ocean_boundary_features(self, coords, tc_lat, tc_lon, threshold):
        if not coords or len(coords) < 3:
            return {}

        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])

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

        distances = [self._haversine_distance(tc_lat, tc_lon, lat, lon) for lon, lat in coords]
        nearest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)

        tc_relative_points = {
            "nearest_to_tc": {
                "lon": float(lons[nearest_idx]),
                "lat": float(lats[nearest_idx]),
                "distance_km": round(float(distances[nearest_idx]), 1),
                "description": "台风到暖水区边界的最短距离",
            },
            "farthest_from_tc": {
                "lon": float(lons[farthest_idx]),
                "lat": float(lats[farthest_idx]),
                "distance_km": round(float(distances[farthest_idx]), 1),
                "description": "暖水区延伸的最远点",
            },
        }

        curvature_extremes = []
        warm_eddy_centers = []
        cold_intrusion_points = []

        if len(coords) >= 5:
            curvatures = []
            for i in range(len(coords)):
                prev_idx = (i - 2) % len(coords)
                next_idx = (i + 2) % len(coords)
                p1 = np.array([lons[prev_idx], lats[prev_idx]])
                p2 = np.array([lons[i], lats[i]])
                p3 = np.array([lons[next_idx], lats[next_idx]])
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                curvature = 0
                if a * b * c > 1e-10:
                    curvature = 4 * area / (a * b * c)
                curvatures.append(curvature)

            curvatures = np.array(curvatures)
            high_curvature_threshold = np.percentile(curvatures, 90)
            high_curv_indices = np.where(curvatures > high_curvature_threshold)[0]

            for idx in high_curv_indices[:5]:
                dist_to_tc = self._haversine_distance(tc_lat, tc_lon, lats[idx], lons[idx])
                avg_dist = np.mean(distances)
                point_info = {"lon": float(lons[idx]), "lat": float(lats[idx]), "curvature": round(float(curvatures[idx]), 6)}
                if dist_to_tc > avg_dist * 1.1:
                    warm_eddy_centers.append({**point_info, "type": "warm_eddy", "description": "暖水区向外延伸的暖涡"})
                elif dist_to_tc < avg_dist * 0.9:
                    cold_intrusion_points.append({**point_info, "type": "cold_intrusion", "description": "冷水向暖水区侵入"})
                curvature_extremes.append(point_info)

        return {
            "extreme_points": extreme_points,
            "warm_eddy_centers": warm_eddy_centers[:3],
            "cold_intrusion_points": cold_intrusion_points[:3],
            "curvature_extremes": curvature_extremes[:5],
            "tc_relative_points": tc_relative_points,
        }


__all__ = ["OceanHeatContentMixin"]
