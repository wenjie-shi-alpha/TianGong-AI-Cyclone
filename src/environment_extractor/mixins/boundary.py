"""Boundary extraction utilities shared across multiple systems."""

from __future__ import annotations

import numpy as np


class BoundaryExtractionMixin:
    def _extract_closed_boundary_with_features(
        self,
        z500,
        tc_lat,
        tc_lon,
        threshold,
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

            local_z500 = z500[np.ix_(lat_mask, lon_mask)]
            local_lat = self.lat[lat_mask]
            local_lon = self.lon[lon_mask]

            if local_z500.size == 0:
                print("⚠️ 局部区域无数据")
                return None

            boundary_coords = None
            method_used = None

            try:
                mask = (local_z500 >= threshold).astype(int)
                labeled = sk_label(mask, connectivity=2)
                if labeled.max() == 0:
                    raise ValueError("未找到连通区域")

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

            except Exception as exc:
                print(f"⚠️ 连通区域方法失败: {exc}，尝试方法2")

            if boundary_coords is None:
                try:
                    expanded_result = self._extract_closed_boundary_with_features(  # type: ignore[call-arg]
                        z500,
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
                    from skimage.measure import find_contours as sk_find_contours_direct

                    contours = sk_find_contours_direct(local_z500, threshold)
                    if contours and len(contours) > 0:
                        boundary_coords = sorted(contours, key=len, reverse=True)[0]
                        method_used = "direct_contour_extraction"
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

            sampled_coords = self._adaptive_boundary_sampling(geo_coords, target_points=target_points)

            if len(sampled_coords) > 2:
                first = sampled_coords[0]
                last = sampled_coords[-1]
                closure_dist = np.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)
                if closure_dist > 1.0:
                    sampled_coords.append(first)

            features = self._extract_boundary_features(sampled_coords, tc_lat, tc_lon, threshold)
            metrics = self._calculate_boundary_metrics(sampled_coords, tc_lat, tc_lon, method_used)

            return {
                "boundary_coordinates": sampled_coords,
                "boundary_features": features,
                "boundary_metrics": metrics,
            }

        except Exception as exc:
            print(f"⚠️ 闭合边界提取完全失败: {exc}")
            import traceback as _traceback  # noqa: WPS433

            _traceback.print_exc()
            return None

    def _adaptive_boundary_sampling(self, coords, target_points=50, method="auto"):
        if len(coords) <= target_points:
            return coords

        if method == "auto":
            perimeter_deg = self._calculate_perimeter(coords)
            if perimeter_deg < 50:
                method = "curvature"
            else:
                method = "douglas_peucker"

        if method == "curvature":
            return self._curvature_adaptive_sampling(coords, target_points)
        if method == "perimeter":
            return self._perimeter_proportional_sampling(coords, target_points)
        if method == "douglas_peucker":
            return self._douglas_peucker_sampling(coords, target_points)

        step = max(1, len(coords) // target_points)
        return coords[::step]

    def _curvature_adaptive_sampling(self, coords, target_points):
        if len(coords) < 3:
            return coords

        curvatures = []
        for i in range(len(coords)):
            prev_idx = (i - 1) % len(coords)
            next_idx = (i + 1) % len(coords)
            p1 = np.array(coords[prev_idx])
            p2 = np.array(coords[i])
            p3 = np.array(coords[next_idx])
            v1 = p2 - p1
            v2 = p3 - p2
            cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p3 - p1)
            curvature = cross / denom if denom > 1e-10 else 0.0
            curvatures.append(curvature)

        curvatures = np.array(curvatures)
        if curvatures.max() > 1e-10:
            weights = 0.5 + (curvatures / curvatures.max())
        else:
            weights = np.ones_like(curvatures)

        cum_weights = np.cumsum(weights)
        cum_weights = cum_weights / cum_weights[-1]
        target_weights = np.linspace(0, 1, target_points, endpoint=False)

        sampled_indices = []
        for tw in target_weights:
            idx = np.argmin(np.abs(cum_weights - tw))
            if idx not in sampled_indices:
                sampled_indices.append(idx)
        sampled_indices = sorted(sampled_indices)
        return [coords[i] for i in sampled_indices]

    def _perimeter_proportional_sampling(self, coords, target_points):
        if len(coords) < 2:
            return coords

        distances = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(distances[-1] + dist)

        total_dist = distances[-1]
        if total_dist < 1e-10:
            return [coords[0]]

        target_distances = np.linspace(0, total_dist, target_points, endpoint=False)
        sampled_coords = []
        for td in target_distances:
            idx = np.argmin(np.abs(np.array(distances) - td))
            sampled_coords.append(coords[idx])
        return sampled_coords

    def _douglas_peucker_sampling(self, coords, target_points):
        if len(coords) <= target_points:
            return coords

        current_coords = coords.copy()
        while len(current_coords) > target_points:
            min_importance = float("inf")
            min_idx = -1
            for i in range(1, len(current_coords) - 1):
                p1 = np.array(current_coords[i - 1])
                p2 = np.array(current_coords[i])
                p3 = np.array(current_coords[i + 1])
                importance = self._point_to_line_distance(p2, p1, p3)
                if importance < min_importance:
                    min_importance = importance
                    min_idx = i
            if min_idx > 0:
                current_coords.pop(min_idx)
            else:
                break
        return current_coords

    def _point_to_line_distance(self, point, line_start, line_end):
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        ab = b - a
        ap = p - a
        if np.linalg.norm(ab) < 1e-10:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def _calculate_perimeter(self, coords):
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
        if not coords or len(coords) < 4:
            return {}

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        north_idx = np.argmax(lats)
        south_idx = np.argmin(lats)
        east_idx = np.argmax(lons)
        west_idx = np.argmin(lons)

        extreme_points = {
            "north": {"lon": round(lons[north_idx], 2), "lat": round(lats[north_idx], 2), "index": north_idx},
            "south": {"lon": round(lons[south_idx], 2), "lat": round(lats[south_idx], 2), "index": south_idx},
            "east": {"lon": round(lons[east_idx], 2), "lat": round(lats[east_idx], 2), "index": east_idx},
            "west": {"lon": round(lons[west_idx], 2), "lat": round(lats[west_idx], 2), "index": west_idx},
        }

        distances = [self._haversine_distance(tc_lat, tc_lon, lat, lon) for lon, lat in coords]
        nearest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)

        tc_relative_points = {
            "nearest": {
                "lon": round(lons[nearest_idx], 2),
                "lat": round(lats[nearest_idx], 2),
                "index": nearest_idx,
                "distance_km": round(distances[nearest_idx], 1),
            },
            "farthest": {
                "lon": round(lons[farthest_idx], 2),
                "lat": round(lats[farthest_idx], 2),
                "index": farthest_idx,
                "distance_km": round(distances[farthest_idx], 1),
            },
        }

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
                curvature = cross / denom if denom > 1e-10 else 0.0
                curvatures.append((i, curvature))

            curvatures_sorted = sorted(curvatures, key=lambda x: abs(x[1]), reverse=True)
            for i, curv in curvatures_sorted[:4]:
                if abs(curv) > 0.01:
                    curvature_extremes.append(
                        {
                            "lon": round(lons[i], 2),
                            "lat": round(lats[i], 2),
                            "index": i,
                            "curvature": round(curv, 4),
                            "type": "凸出" if curv > 0 else "凹陷",
                        }
                    )

        return {
            "extreme_points": extreme_points,
            "tc_relative_points": tc_relative_points,
            "curvature_extremes": curvature_extremes,
        }

    def _calculate_boundary_metrics(self, coords, tc_lat, tc_lon, method_used):
        if not coords or len(coords) < 2:
            return {}

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        first = coords[0]
        last = coords[-1]
        closure_dist = np.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)
        is_closed = closure_dist < 1.0

        perimeter_km = 0.0
        for i in range(len(coords)):
            if is_closed:
                next_idx = (i + 1) % len(coords)
            else:
                next_idx = min(i + 1, len(coords) - 1)
            if next_idx != i:
                dist_km = self._haversine_distance(lats[i], lons[i], lats[next_idx], lons[next_idx])
                perimeter_km += dist_km

        center_lon = np.mean(lons)
        center_lat = np.mean(lats)
        angles = []
        for lon, lat in coords:
            angle = np.arctan2(lat - center_lat, lon - center_lon) * 180 / np.pi
            angles.append(angle)
        angle_coverage = max(angles) - min(angles) if angles else 0
        if is_closed:
            angle_coverage = 360.0

        avg_spacing_km = perimeter_km / len(coords) if len(coords) > 0 else 0
        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)
        aspect_ratio = lon_span / lat_span if lat_span > 0 else 0

        return {
            "is_closed": bool(is_closed),
            "total_points": int(len(coords)),
            "perimeter_km": round(float(perimeter_km), 1),
            "avg_point_spacing_km": round(float(avg_spacing_km), 1),
            "angle_coverage_deg": round(float(angle_coverage), 1),
            "closure_distance_deg": round(float(closure_dist), 2),
            "aspect_ratio": round(float(aspect_ratio), 2),
            "lon_span_deg": round(float(lon_span), 2),
            "lat_span_deg": round(float(lat_span), 2),
            "extraction_method": method_used or "unknown",
        }


__all__ = ["BoundaryExtractionMixin"]
