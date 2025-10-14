#!/usr/bin/env python3
"""Preprocess cyclone forecasts by aligning them with ground truth observations.

This script implements the data preparation workflow described in ``specdataset.md``.
For every deterministic forecast file found under ``data/track_single`` it:

* groups candidate trajectories by ``particle`` identifier (multiple tracks per run),
* selects, for each storm present in the file, the trajectory whose mean distance to
  the real path is minimal,
* aggregates the best track from **every available model** for the same storm and
  initial time,
* loads the corresponding environment analyses,
* converts verbose JSON payloads into concise text summaries, and
* writes JSONL samples that bundle history, ground truth, real environment analysis,
  and a list of model forecasts (each carrying its own track and environment summary).

The resulting dataset provides a comprehensive, multi-model view for each forecast
initialisation time that can be fed directly into prompt construction or downstream
generation pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ISO_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ENV_TIME_LIMIT_HOURS = 72
FORECAST_HORIZON_HOURS = 72
HISTORY_WINDOW_HOURS = 24
MIN_MATCH_POINTS = 3

SINGLE_RE = re.compile(
    r"^track_(?P<storm>[0-9A-Z]+)_(?P<model>[A-Za-z0-9]+)_(?P<version>v\d+)_"
    r"(?P<source>[A-Za-z0-9]+)_(?P<init>\d+)_f\d+_f\d+_(?P<cycle>\d+)\.csv$"
)
MULTI_RE = re.compile(
    r"^tracks_(?P<model>[A-Za-z0-9]+)_(?P<version>v\d+)_(?P<source>[A-Za-z0-9]+)_"
    r"(?P<init>[0-9T]+)_f\d+_f\d+_(?P<cycle>\d+)\.csv$"
)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_datetime_parse(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, ISO_TIME_FORMAT)
    except ValueError:
        logger.debug("无法解析时间 %s", value)
        return None


def _parse_init_time(raw: str) -> datetime:
    if "T" in raw:
        return datetime.strptime(raw, "%Y%m%dT%H%M%S")
    return datetime.strptime(raw, "%Y%m%d%H")


def _parse_environment_time(raw: object) -> Optional[datetime]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometres."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


@dataclass(frozen=True)
class TrackFileMeta:
    path: Path
    model: str
    version: str
    source: str
    init_time: datetime
    cycle_hours: int
    storm_id_hint: Optional[str]

    @property
    def init_tag(self) -> str:
        return self.init_time.strftime("%Y%m%d%H")


def parse_track_filename(path: Path) -> Optional[TrackFileMeta]:
    name = path.name
    single = SINGLE_RE.match(name)
    if single:
        meta = single.groupdict()
        init_time = _parse_init_time(meta["init"])
        return TrackFileMeta(
            path=path,
            model=meta["model"],
            version=meta["version"],
            source=meta["source"],
            init_time=init_time,
            cycle_hours=int(meta["cycle"]),
            storm_id_hint=meta["storm"],
        )
    multi = MULTI_RE.match(name)
    if multi:
        meta = multi.groupdict()
        init_time = _parse_init_time(meta["init"])
        return TrackFileMeta(
            path=path,
            model=meta["model"],
            version=meta["version"],
            source=meta["source"],
            init_time=init_time,
            cycle_hours=int(meta["cycle"]),
            storm_id_hint=None,
        )
    logger.warning("跳过无法解析的文件名: %s", name)
    return None


def _normalise_particle_id(particle: Optional[str], meta: TrackFileMeta) -> Optional[str]:
    particle = (particle or "").strip()
    if particle:
        return particle
    return meta.storm_id_hint


def _group_forecast_tracks(meta: TrackFileMeta) -> Dict[str, List[Dict[str, Optional[float]]]]:
    grouped: Dict[str, List[Dict[str, Optional[float]]]] = defaultdict(list)
    with meta.path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            particle = _normalise_particle_id(row.get("particle"), meta)
            if not particle:
                continue
            timestamp = _safe_datetime_parse(row["time"])
            if timestamp is None:
                continue
            grouped[particle].append(
                {
                    "time": timestamp,
                    "lat": _to_float(row.get("lat")),
                    "lon": _to_float(row.get("lon")),
                    "msl": _to_float(row.get("msl")),
                    "wind": _to_float(row.get("wind")),
                    "time_idx": int(row["time_idx"]) if row.get("time_idx") else None,
                }
            )
    for values in grouped.values():
        values.sort(key=lambda item: item["time"])  # type: ignore[arg-type]
    return grouped


class GroundTruthIndex:
    """Index real observations for quick lookup by storm and time."""

    def __init__(self, csv_path: Path):
        self._points: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        self._times: Dict[str, List[datetime]] = {}
        self._storm_names: Dict[str, str] = {}
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                storm_id = row["storm_id"].strip()
                stamp = datetime.fromisoformat(row["datetime"])
                point = {
                    "datetime": stamp,
                    "latitude": _to_float(row.get("latitude")),
                    "longitude": _to_float(row.get("longitude")),
                    "max_wind_wmo": _to_float(row.get("max_wind_wmo")),
                    "min_pressure_wmo": _to_float(row.get("min_pressure_wmo")),
                    "max_wind_usa": _to_float(row.get("max_wind_usa")),
                    "min_pressure_usa": _to_float(row.get("min_pressure_usa")),
                    "storm_speed": _to_float(row.get("storm_speed")),
                    "storm_direction": _to_float(row.get("storm_direction")),
                    "distance_to_land": _to_float(row.get("distance_to_land")),
                }
                self._points[storm_id].append(point)
                self._storm_names[storm_id] = row.get("storm_name", "").strip()
        for storm_id, items in self._points.items():
            items.sort(key=lambda item: item["datetime"])  # type: ignore[arg-type]
            self._times[storm_id] = [item["datetime"] for item in items]  # type: ignore[list-item]

    def has_storm(self, storm_id: str) -> bool:
        return storm_id in self._points

    def get_storm_name(self, storm_id: str) -> str:
        return self._storm_names.get(storm_id, "")

    def _nearest_index(self, storm_id: str, target: datetime) -> Optional[int]:
        times = self._times.get(storm_id)
        if not times:
            return None
        pos = bisect_left(times, target)
        candidates: List[Tuple[float, int]] = []
        if pos < len(times):
            candidates.append((abs((times[pos] - target).total_seconds()), pos))
        if pos > 0:
            idx = pos - 1
            candidates.append((abs((times[idx] - target).total_seconds()), idx))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        best_delta, best_idx = candidates[0]
        if best_delta > 3 * 3600:
            return None
        return best_idx

    def get_point(self, storm_id: str, target: datetime) -> Optional[Dict[str, object]]:
        idx = self._nearest_index(storm_id, target)
        if idx is None:
            return None
        return self._points[storm_id][idx]

    def get_points_between(
        self, storm_id: str, start: datetime, end: datetime
    ) -> List[Dict[str, object]]:
        if start > end:
            start, end = end, start
        points = self._points.get(storm_id, [])
        return [
            point.copy()
            for point in points
            if start <= point["datetime"] <= end  # type: ignore[operator]
        ]


class RealEnvironmentIndex:
    """Lazy loader for real (CDS) environment analyses."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._cache: Dict[str, Dict[datetime, List[Dict[str, object]]]] = {}

    def _load_month(self, month_key: str) -> Dict[datetime, List[Dict[str, object]]]:
        if month_key in self._cache:
            return self._cache[month_key]
        file_path = self._base_dir / f"cds_environment_analysis_{month_key}.json"
        if not file_path.exists():
            logger.debug("缺少真实环境场文件: %s", file_path)
            self._cache[month_key] = {}
            return self._cache[month_key]
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        by_time: Dict[datetime, List[Dict[str, object]]] = defaultdict(list)
        for entry in payload.get("environmental_analysis", []):
            stamp = datetime.fromisoformat(entry["time"]).replace(microsecond=0)
            by_time[stamp].append(entry)
        self._cache[month_key] = by_time
        return by_time

    def get_entry(
        self, when: datetime, lat: Optional[float], lon: Optional[float]
    ) -> Optional[Dict[str, object]]:
        month_key = when.strftime("%Y-%m")
        monthly = self._load_month(month_key)
        when_key = when.replace(microsecond=0)
        entries = monthly.get(when_key)
        if not entries:
            return None
        if lat is None or lon is None:
            return entries[0]
        best_entry = None
        best_distance = float("inf")
        for entry in entries:
            pos = entry.get("tc_position") or {}
            pos_lat = pos.get("lat")
            pos_lon = pos.get("lon")
            if isinstance(pos_lat, (int, float)) and isinstance(pos_lon, (int, float)):
                distance = haversine_distance_km(lat, lon, pos_lat, pos_lon)
            else:
                distance = float("inf")
            if distance < best_distance:
                best_distance = distance
                best_entry = entry
        return best_entry or entries[0]


class ForecastEnvironmentIndex:
    """Loader for deterministic forecast environment analyses."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._cache: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}

    def _candidate_paths(
        self, model: str, version: str, source: str, init_tag: str, storm_id: str
    ) -> List[Path]:
        suffix = f"{init_tag}_f000_f240_06_TC_Analysis_{storm_id}.json"
        return [
            self._base_dir / f"{model}_{version}_{source}_{suffix}",
        ]

    def load_environment(
        self, model: str, version: str, source: str, init_tag: str, storm_id: str
    ) -> Optional[Dict[str, object]]:
        key = (model, version, source, init_tag, storm_id)
        if key in self._cache:
            return self._cache[key]
        for candidate in self._candidate_paths(model, version, source, init_tag, storm_id):
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    self._cache[key] = json.load(handle)
                    self._cache[key]["__file_path"] = str(candidate)
                    return self._cache[key]
        logger.debug(
            "未找到预报环境场文件: %s_%s_%s_%s (%s)",
            model,
            version,
            source,
            init_tag,
            storm_id,
        )
        self._cache[key] = None  # type: ignore[assignment]
        return None


def summarise_environment_system(system: Dict[str, object]) -> str:
    name = str(system.get("system_name", "")).strip()
    label = SYSTEM_NAME_LABELS.get(name, name or "未知系统")
    parts: List[str] = []
    desc = system.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())
    intensity = system.get("intensity")
    if isinstance(intensity, dict):
        level = intensity.get("level")
        value = intensity.get("value")
        unit = intensity.get("unit")
        details: List[str] = []
        if level:
            details.append(str(level))
        if isinstance(value, (int, float)):
            if isinstance(unit, str) and unit:
                details.append(f"{value}{unit}")
            else:
                details.append(str(value))
        elif isinstance(value, str):
            details.append(value)
        if details:
            joined = " / ".join(details)
            if not any(joined in segment for segment in parts):
                parts.append(f"强度: {joined}")
    properties = system.get("properties")
    if isinstance(properties, dict):
        for key in ("impact", "influence", "trend", "direction_text"):
            text = properties.get(key)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        steering = properties.get("steering_flow")
        if isinstance(steering, dict):
            speed = steering.get("speed_mps")
            direction = steering.get("direction_deg")
            if isinstance(speed, (int, float)) and isinstance(direction, (int, float)):
                parts.append(f"引导气流约{speed:.1f}m/s，方向{direction:.0f}°")
    return f"{label}: {'；'.join(parts) if parts else '暂无摘要'}"


SYSTEM_NAME_LABELS: Dict[str, str] = {
    "SubtropicalHigh": "副热带高压",
    "VerticalWindShear": "垂直风切变",
    "OceanHeatContent": "海洋热含量",
    "UpperLevelDivergence": "高空辐散",
    "InterTropicalConvergenceZone": "热带辐合带",
    "WesterlyTrough": "西风槽",
    "FrontalSystem": "锋面系统",
    "MonsoonTrough": "季风槽",
    "LowLevelFlow": "低层风场",
    "AtmosphericStability": "大气稳定度",
    "BlockingHigh": "阻塞高压",
    "MaddenJulianOscillation": "MJO",
}


def summarise_environment_entry(entry: Dict[str, object]) -> str:
    systems = entry.get("environmental_systems")
    if not isinstance(systems, list):
        return "环境信息缺失"
    summaries = [summarise_environment_system(system) for system in systems if isinstance(system, dict)]
    return " ".join(summaries) if summaries else "环境系统数量为0"


def _combine_environment_time_series(
    time_series: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    combined: "OrderedDict[datetime, Dict[str, object]]" = OrderedDict()
    system_maps: Dict[datetime, "OrderedDict[str, Dict[str, object]]"] = {}
    for raw_entry in time_series:
        if not isinstance(raw_entry, dict):
            continue
        stamp = _parse_environment_time(raw_entry.get("time"))
        if stamp is None:
            continue
        bucket = combined.get(stamp)
        if bucket is None:
            bucket = {"time": stamp, "environmental_systems": []}
            combined[stamp] = bucket
        systems = raw_entry.get("environmental_systems")
        if not isinstance(systems, list):
            continue
        system_map = system_maps.setdefault(stamp, OrderedDict())
        for system in systems:
            if not isinstance(system, dict):
                continue
            name = system.get("system_name")
            key = str(name) if name else f"system_{len(system_map)}"
            if key in system_map:
                continue
            system_map[key] = system
    for stamp, bucket in combined.items():
        systems = system_maps.get(stamp)
        if systems:
            bucket["environmental_systems"] = list(systems.values())
        else:
            bucket["environmental_systems"] = []
    return list(combined.values())


def _json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _format_system_details(system: Dict[str, object], indent: str = "  ") -> List[str]:
    lines: List[str] = []
    system_name = system.get("system_name")
    label = SYSTEM_NAME_LABELS.get(str(system_name), str(system_name) or "未知系统")
    lines.append(f"{indent}- 系统: {label} ({system_name})")
    description = system.get("description")
    if isinstance(description, str) and description.strip():
        lines.append(f"{indent}  描述: {description.strip()}")
    intensity = system.get("intensity")
    if intensity:
        lines.append(f"{indent}  强度: {_json_dumps(intensity)}")
    position = system.get("position")
    if position:
        lines.append(f"{indent}  位置: {_json_dumps(position)}")
    properties = system.get("properties")
    if properties:
        lines.append(f"{indent}  属性: {_json_dumps(properties)}")
    shape = system.get("shape")
    if shape:
        lines.append(f"{indent}  形态: {_json_dumps(shape)}")
    extras = {
        key: value
        for key, value in system.items()
        if key
        not in {
            "system_name",
            "description",
            "intensity",
            "position",
            "properties",
            "shape",
        }
    }
    if extras:
        lines.append(f"{indent}  其他: {_json_dumps(extras)}")
    return lines


def _format_environment_timeline_text(timeline: Sequence[Dict[str, object]]) -> str:
    paragraphs: List[str] = []
    for entry in timeline:
        time_text = entry.get("time", "未知时间")
        summary = entry.get("summary", "")
        header = f"{time_text}: {summary}" if summary else f"{time_text}:"
        lines: List[str] = [header]
        systems = entry.get("systems")
        if isinstance(systems, list) and systems:
            for system in systems:
                if isinstance(system, dict):
                    lines.extend(_format_system_details(system, indent="  "))
        else:
            lines.append("  - 无环境系统信息")
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def _align_track_and_environment(
    points: Sequence[Dict[str, object]],
    timeline: Sequence[Dict[str, object]],
    tolerance_hours: int = 3,
) -> List[Dict[str, object]]:
    if not points:
        return []
    env_map: Dict[datetime, str] = {}
    env_times: List[datetime] = []
    env_detail_map: Dict[datetime, Dict[str, object]] = {}
    for entry in timeline:
        stamp = _parse_environment_time(entry.get("time"))
        if stamp is None:
            continue
        summary = entry.get("summary", "")
        env_map[stamp] = summary
        env_detail_map[stamp] = entry
        env_times.append(stamp)
    env_times.sort()

    alignments: List[Dict[str, object]] = []
    for point in points:
        stamp = _parse_environment_time(point.get("datetime"))
        if stamp is None:
            continue
        summary = env_map.get(stamp)
        detail_entry = env_detail_map.get(stamp)
        if summary is None and env_times:
            nearest = min(
                env_times,
                key=lambda t: abs((t - stamp).total_seconds()),
            )
            delta_hours = abs((nearest - stamp).total_seconds()) / 3600
            if delta_hours <= tolerance_hours:
                summary = env_map.get(nearest)
                detail_entry = env_detail_map.get(nearest)
        alignment = {
            "time": stamp.strftime("%Y-%m-%d %H:%M"),
            "track_point": {
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "msl": point.get("msl"),
                "wind": point.get("wind"),
            },
            "environment_summary": summary,
        }
        if isinstance(detail_entry, dict):
            systems = detail_entry.get("systems")
            if isinstance(systems, list):
                alignment["environment_systems"] = systems
        alignments.append(alignment)
    return alignments


def summarise_environment_series(
    time_series: Sequence[Dict[str, object]], limit_hours: int = ENV_TIME_LIMIT_HOURS
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    if not time_series:
        return summaries
    combined = _combine_environment_time_series(time_series)
    if not combined:
        return summaries
    first_time = combined[0]["time"]
    if not isinstance(first_time, datetime):
        return summaries
    limit = first_time + timedelta(hours=limit_hours)
    for entry in combined:
        stamp = entry.get("time")
        if not isinstance(stamp, datetime):
            continue
        if stamp > limit:
            break
        summary = summarise_environment_entry(entry)
        systems = entry.get("environmental_systems") or []
        try:
            systems_serialised = json.loads(json.dumps(systems, ensure_ascii=False))
        except TypeError:
            systems_serialised = systems
        summaries.append(
            {
                "time": stamp.strftime("%Y-%m-%d %H:%M"),
                "summary": summary,
                "systems": systems_serialised,
                "raw_time_iso": stamp.replace(microsecond=0).isoformat(),
            }
        )
    return summaries


def evaluate_forecast_track(
    storm_id: str,
    init_time: datetime,
    points: Sequence[Dict[str, object]],
    truth_index: GroundTruthIndex,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> Optional[Dict[str, object]]:
    if not truth_index.has_storm(storm_id):
        return None
    end_time = init_time + timedelta(hours=horizon_hours)
    filtered: List[Dict[str, object]] = [
        point for point in points if init_time <= point["time"] <= end_time  # type: ignore[operator]
    ]
    if not filtered:
        return None
    matched: List[Dict[str, object]] = []
    distances: List[float] = []
    for point in filtered:
        truth = truth_index.get_point(storm_id, point["time"])  # type: ignore[arg-type]
        if truth is None:
            continue
        lat_f = point.get("lat")
        lon_f = point.get("lon")
        lat_t = truth.get("latitude")
        lon_t = truth.get("longitude")
        if not all(isinstance(val, (int, float)) for val in (lat_f, lon_f, lat_t, lon_t)):
            continue
        distance = haversine_distance_km(float(lat_f), float(lon_f), float(lat_t), float(lon_t))
        distances.append(distance)
        matched.append(
            {
                "time": point["time"],
                "forecast": point,
                "truth": truth,
                "distance_km": distance,
            }
        )
    if len(matched) < MIN_MATCH_POINTS:
        return None
    return {
        "filtered_points": filtered,
        "matches": matched,
        "mean_distance": float(sum(distances) / len(distances)),
        "max_distance": float(max(distances)),
    }


def _serialise_points(points: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    serialised: List[Dict[str, object]] = []
    for point in points:
        item = point.copy()
        stamp = item.get("datetime")
        if isinstance(stamp, datetime):
            item["datetime"] = stamp.strftime(ISO_TIME_FORMAT)
        serialised.append(item)
    return serialised


def _initialise_sample_entry(
    storm_id: str,
    init_time: datetime,
    truth_index: GroundTruthIndex,
    real_env_index: RealEnvironmentIndex,
) -> Dict[str, object]:
    storm_name = truth_index.get_storm_name(storm_id)
    history = truth_index.get_points_between(
        storm_id, init_time - timedelta(hours=HISTORY_WINDOW_HOURS), init_time
    )
    ground_truth = truth_index.get_points_between(
        storm_id, init_time, init_time + timedelta(hours=FORECAST_HORIZON_HOURS)
    )
    history_serialised = _serialise_points(history)
    ground_truth_serialised = _serialise_points(ground_truth)

    first_truth_point = ground_truth[0] if ground_truth else None
    target_lat = first_truth_point.get("latitude") if first_truth_point else None  # type: ignore[union-attr]
    target_lon = first_truth_point.get("longitude") if first_truth_point else None  # type: ignore[union-attr]
    truth_env = real_env_index.get_entry(init_time, target_lat, target_lon)
    truth_env_summary = summarise_environment_entry(truth_env) if truth_env else "环境信息缺失"
    analysis_time_str = None
    analysis_text = truth_env_summary
    if truth_env:
        raw_time = truth_env.get("time")
        analysis_time = _parse_environment_time(raw_time)
        if analysis_time:
            analysis_time_str = analysis_time.strftime("%Y-%m-%d %H:%M")
            if truth_env_summary:
                analysis_text = f"{analysis_time_str}: {truth_env_summary}"

    return {
        "storm_id": storm_id,
        "storm_name": storm_name,
        "init_time": init_time.strftime(ISO_TIME_FORMAT),
        "history": history_serialised,
        "ground_truth": ground_truth_serialised,
        "environment": {
            "analysis_summary": truth_env_summary,
            "analysis_time": analysis_time_str,
            "analysis_text": analysis_text,
        },
        "model_forecasts": [],
        "_model_index": {},
    }


def _forecast_track_from_evaluation(
    evaluation: Dict[str, object],
    track_file: Path,
) -> Dict[str, object]:
    forecast_points: List[Dict[str, object]] = []
    for point in evaluation["filtered_points"]:  # type: ignore[assignment]
        stamp = point["time"]
        timestamp = stamp.strftime(ISO_TIME_FORMAT) if isinstance(stamp, datetime) else str(stamp)
        forecast_points.append(
            {
                "datetime": timestamp,
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "msl": point.get("msl"),
                "wind": point.get("wind"),
                "time_idx": point.get("time_idx"),
            }
        )
    return {
        "points": forecast_points,
        "track_file": str(track_file),
    }


def _build_model_forecast_entry(
    storm_id: str,
    meta: TrackFileMeta,
    evaluation: Dict[str, object],
    forecast_env_index: ForecastEnvironmentIndex,
) -> Dict[str, object]:
    track = _forecast_track_from_evaluation(evaluation, meta.path)

    forecast_env = forecast_env_index.load_environment(
        meta.model, meta.version, meta.source, meta.init_tag, storm_id
    )
    forecast_env_timeline: List[Dict[str, object]] = []
    forecast_env_path = None
    if forecast_env:
        time_series = forecast_env.get("time_series")
        if isinstance(time_series, list):
            forecast_env_timeline = summarise_environment_series(time_series)
        forecast_env_path = forecast_env.get("__file_path")

    particle_id = evaluation.get("particle_id") or storm_id
    initial_env_summary = (
        forecast_env_timeline[0]["summary"] if forecast_env_timeline else None
    )

    aligned_timeline = _align_track_and_environment(
        track["points"],
        forecast_env_timeline,
    )

    selection_score = float(evaluation.get("mean_distance", float("inf")))
    max_distance = evaluation.get("max_distance")

    return {
        "model": {
            "name": meta.model,
            "version": meta.version,
            "source": meta.source,
            "cycle_hours": meta.cycle_hours,
        },
        "particle_id": particle_id,
        "track": track,
        "environment": {
            "initial_summary": initial_env_summary,
            "timeline": forecast_env_timeline,
            "timeline_text": _format_environment_timeline_text(forecast_env_timeline)
            if forecast_env_timeline
            else None,
            "forecast_summaries": forecast_env_timeline,
            "forecast_environment_file": forecast_env_path,
        },
        "track_environment_alignment": aligned_timeline,
        "_selection_score": selection_score,
        "_selection_max_distance": float(max_distance) if isinstance(max_distance, (int, float)) else None,
    }


def _add_model_forecast(
    aggregate: Dict[str, object],
    meta: TrackFileMeta,
    storm_id: str,
    evaluation: Dict[str, object],
    forecast_env_index: ForecastEnvironmentIndex,
) -> None:
    model_entry = _build_model_forecast_entry(storm_id, meta, evaluation, forecast_env_index)
    model_key = (
        meta.model,
        meta.version,
        meta.source,
        meta.cycle_hours,
    )
    index_map: Dict[Tuple[str, str, str, int], int] = aggregate.setdefault("_model_index", {})  # type: ignore[assignment]
    forecasts: List[Dict[str, object]] = aggregate.setdefault("model_forecasts", [])  # type: ignore[assignment]

    new_score = model_entry.get("_selection_score")
    if not isinstance(new_score, (int, float)):
        new_score = float("inf")
    existing_idx = index_map.get(model_key)
    if existing_idx is not None:
        existing_entry = forecasts[existing_idx]
        existing_score = existing_entry.get("_selection_score")
        if not isinstance(existing_score, (int, float)) or new_score < existing_score:
            forecasts[existing_idx] = model_entry
    else:
        index_map[model_key] = len(forecasts)
        forecasts.append(model_entry)


def _mean_distance_sort_key(entry: Dict[str, object]) -> float:
    value = entry.get("_selection_score")
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def iter_forecast_samples(
    meta: TrackFileMeta,
    grouped_tracks: Dict[str, List[Dict[str, object]]],
    truth_index: GroundTruthIndex,
) -> Iterator[Tuple[str, Dict[str, object]]]:
    best_per_storm: Dict[str, Dict[str, object]] = {}
    for particle_id, points in grouped_tracks.items():
        storm_id = particle_id
        if not truth_index.has_storm(storm_id):
            continue
        evaluation = evaluate_forecast_track(storm_id, meta.init_time, points, truth_index)
        if not evaluation:
            continue
        evaluation["particle_id"] = particle_id
        existing = best_per_storm.get(storm_id)
        if not existing or evaluation["mean_distance"] < existing["mean_distance"]:  # type: ignore[index]
            best_per_storm[storm_id] = evaluation
    for storm_id, evaluation in best_per_storm.items():
        yield storm_id, evaluation


def prepare_samples(
    truth_csv: Path,
    track_dir: Path,
    real_env_dir: Path,
    forecast_env_dir: Path,
    output_path: Path,
    limit: Optional[int] = None,
) -> int:
    truth_index = GroundTruthIndex(truth_csv)
    real_env_index = RealEnvironmentIndex(real_env_dir)
    forecast_env_index = ForecastEnvironmentIndex(forecast_env_dir)
    aggregated: Dict[Tuple[str, datetime], Dict[str, object]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_files = sorted(track_dir.glob("*.csv"))
    for idx, track_path in enumerate(track_files):
        if limit is not None and idx >= limit:
            break
        meta = parse_track_filename(track_path)
        if not meta:
            continue
        grouped = _group_forecast_tracks(meta)
        if not grouped:
            continue
        for storm_id, evaluation in iter_forecast_samples(meta, grouped, truth_index):
            key = (storm_id, meta.init_time)
            aggregate = aggregated.get(key)
            if aggregate is None:
                aggregate = _initialise_sample_entry(
                    storm_id,
                    meta.init_time,
                    truth_index,
                    real_env_index,
                )
                aggregated[key] = aggregate
            _add_model_forecast(aggregate, meta, storm_id, evaluation, forecast_env_index)

    records: List[Dict[str, object]] = []
    for aggregate in aggregated.values():
        forecasts = aggregate.get("model_forecasts") or []
        if forecasts:
            forecasts.sort(key=_mean_distance_sort_key)
            for forecast in forecasts:
                if isinstance(forecast, dict):
                    forecast.pop("_selection_score", None)
                    forecast.pop("_selection_max_distance", None)
        cleaned = {k: v for k, v in aggregate.items() if k != "_model_index"}
        records.append(cleaned)

    records.sort(key=lambda item: (item["init_time"], item["storm_id"]))  # type: ignore[index]

    with output_path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对齐真实观测与模式预报，构建预报样本的基础数据。"
    )
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path("input") / "western_pacific_typhoons_superfast.csv",
        help="真实气旋路径CSV路径",
    )
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=Path("data") / "track_single",
        help="模型预报路径目录",
    )
    parser.add_argument(
        "--real-env-dir",
        type=Path,
        default=Path("data") / "cds_output_trusted",
        help="真实环境场目录",
    )
    parser.add_argument(
        "--forecast-env-dir",
        type=Path,
        default=Path("data") / "final_single_output_trusted",
        help="模式环境场目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preprocessed_data") / "matched_samples.jsonl",
        help="输出的JSONL文件路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="调试用，只处理前N个预报文件",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="增加日志输出（-v 或 -vv）",
    )
    return parser.parse_args(argv)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    produced = prepare_samples(
        truth_csv=args.truth_csv,
        track_dir=args.track_dir,
        real_env_dir=args.real_env_dir,
        forecast_env_dir=args.forecast_env_dir,
        output_path=args.output,
        limit=args.limit,
    )
    logger.info("共写出 %s 条预报样本至 %s", produced, args.output)


if __name__ == "__main__":
    main()
