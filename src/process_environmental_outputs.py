"""Utility for extracting distance-independent fields from TC environment JSON outputs.

The processed payload keeps descriptive information (location cues, textual shape
descriptions, intensity) while dropping area/perimeter style metrics that depend on
distance calculations. Results are written to the processed output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_INPUT_DIR = Path("data/final_single_output")
DEFAULT_OUTPUT_DIR = Path("data/final_single_output_processed")

# Keys whose names imply reliance on distance/area style metrics. These will be pruned
# from nested dictionaries to avoid surfacing unreliable magnitudes.
_DISTANCE_KEY_MARKERS: tuple[str, ...] = (
    "distance",
    "area",
    "perimeter",
    "radius",
    "length",
    "axis",
    "span",
    "km",
    "core_ratio",
    "middle_ratio",
    "approx",
)


def _should_drop_key(key: str) -> bool:
    key_lower = key.lower()
    return any(marker in key_lower for marker in _DISTANCE_KEY_MARKERS)


def _compress_coordinates(coords: list[Any], max_pairs: Optional[int] = None) -> Optional[str]:
    """Convert coordinate arrays to a compact string representation."""

    if not isinstance(coords, list) or not coords:
        return None

    pairs: list[str] = []
    iterable = coords[: max_pairs] if isinstance(max_pairs, int) and max_pairs > 0 else coords
    for item in iterable:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lat, lon = item[0], item[1]
            try:
                pairs.append(f"({float(lat):.2f},{float(lon):.2f})")
            except (TypeError, ValueError):
                continue

    if not pairs:
        return None

    serialized = ";".join(pairs)
    if len(iterable) < len(coords):
        serialized += ";..."
    return serialized


def _filter_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove keys whose names indicate distance-based quantities."""

    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if _should_drop_key(key):
            continue

        if isinstance(value, dict):
            nested = _filter_mapping(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            # Preserve lists that contain non-numeric descriptive content.
            filtered_list = []
            for item in value:
                if isinstance(item, dict):
                    nested = _filter_mapping(item)
                    if nested:
                        filtered_list.append(nested)
                else:
                    filtered_list.append(item)
            if filtered_list:
                cleaned[key] = filtered_list
        else:
            cleaned[key] = value

    return cleaned


def _build_shape_summary(shape: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(shape, dict):
        return None

    summary: Dict[str, Any] = {}

    for key in ("description", "shape_type", "orientation", "complexity"):
        value = shape.get(key)
        if isinstance(value, str) and value:
            summary[key] = value

    detailed = shape.get("detailed_analysis")
    if isinstance(detailed, dict):
        basic = detailed.get("basic_geometry")
        if isinstance(basic, dict):
            desc = basic.get("description")
            if isinstance(desc, str) and desc:
                summary.setdefault("shape_type", desc)

        orientation = detailed.get("orientation")
        if isinstance(orientation, dict):
            orient_desc = orientation.get("description")
            if isinstance(orient_desc, str) and orient_desc:
                summary.setdefault("orientation_details", orient_desc)

        complexity = detailed.get("shape_complexity")
        if isinstance(complexity, dict):
            complexity_desc = complexity.get("description")
            if isinstance(complexity_desc, str) and complexity_desc:
                summary.setdefault("complexity_details", complexity_desc)

        contour = detailed.get("contour_analysis")
        if isinstance(contour, dict):
            coords = contour.get("simplified_coordinates")
            coords_str = _compress_coordinates(coords)
            if coords_str:
                summary.setdefault("boundary_coords", coords_str)

    return summary or None


def _process_system(system: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        key: system[key]
        for key in ("system_name", "description")
        if key in system
    }

    if "intensity" in system:
        result["intensity"] = _filter_mapping(system["intensity"])

    if "position" in system:
        result["position"] = _filter_mapping(system["position"])

    if "properties" in system:
        filtered_props = _filter_mapping(system["properties"])
        if filtered_props:
            result["properties"] = filtered_props

    shape_summary = _build_shape_summary(system.get("shape"))
    if shape_summary:
        result["shape_summary"] = shape_summary

    return result


def _process_time_series_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    processed = {
        key: entry[key]
        for key in ("time", "time_idx", "tc_position")
        if key in entry
    }

    systems = entry.get("environmental_systems") or []
    processed_systems = [_process_system(system) for system in systems]
    if processed_systems:
        processed["environmental_systems"] = processed_systems

    return processed


def process_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    processed: Dict[str, Any] = {
        key: payload[key]
        for key in ("tc_id", "analysis_time")
        if key in payload
    }

    time_series = payload.get("time_series") or []
    processed["time_series"] = [_process_time_series_entry(entry) for entry in time_series]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(processed, fp, ensure_ascii=False, indent=2)


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract distance-independent fields from TC environment JSON outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing original JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write processed JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = list(iter_input_files(input_dir))
    if not files:
        print(f"⚠️ No JSON files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        target = output_dir / path.name
        process_file(path, target)
        print(f"✅ Processed {path.name} -> {target.relative_to(output_dir.parent)}")


if __name__ == "__main__":
    main()
