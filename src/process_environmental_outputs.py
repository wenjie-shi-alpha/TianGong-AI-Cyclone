"""Filter TC environment outputs to keep only the trusted, distance-independent data.

The specification in ``spec.md`` identifies which parts of the JSON payloads are
reliable (system identity, position, intensity, steering flow, qualitative shape
descriptions, and raw coordinates) and which parts must be discarded (any distance-
based magnitudes such as area or perimeter). This module prunes the untrusted fields
and writes curated JSON files for both the deterministic ``final_single_output`` data
and the monthly ``cds_output`` data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_FINAL_INPUT_DIR = Path("data/final_single_output")
DEFAULT_FINAL_OUTPUT_DIR = Path("data/final_single_output_trusted")
DEFAULT_CDS_INPUT_DIR = Path("data/cds_output")
DEFAULT_CDS_OUTPUT_DIR = Path("data/cds_output_trusted")
DEFAULT_SINGLE_OUTPUT_SUFFIX = "_trusted"

_UNTRUSTED_KEY_MARKERS: Tuple[str, ...] = (
    "distance",
    "area",
    "perimeter",
    "radius",
    "length",
    "axis",
    "span",
    "km",
    "ratio",
    "core_ratio",
    "middle_ratio",
    "approx",
)

_UNTRUSTED_STRING_MARKERS_LOWER = ("km",)
_UNTRUSTED_STRING_MARKERS_LITERAL = ("公里", "千米")

_POSITION_DROP_KEYS = {"bearing_deg"}

_SHAPE_TEXT_KEYS = (
    "description",
    "shape_type",
    "orientation",
    "complexity",
    "warm_region_shape",
    "warm_region_orientation",
)

_COORDINATE_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("coordinates", "vertices"),
    ("coordinates",),
    ("coordinate_details", "main_contour_coords"),
    ("coordinate_details", "polygon"),
    ("coordinate_details", "polygon_features", "polygon"),
    ("detailed_analysis", "contour_analysis", "simplified_coordinates"),
    ("detailed_analysis", "contour_analysis", "polygon_features", "polygon"),
    ("warm_water_boundary_26.5C",),
    ("warm_water_boundary",),
)

# Pattern to match area descriptions like "暖水区域面积约3080km²" or similar
# This matches various forms of area descriptions with km² or similar units
_AREA_DESCRIPTION_PATTERN = re.compile(
    r'[，。\s]*[暖冷热]?水?[区域]*面积约?\s*\d+[\d,\.]*\s*(?:km²|平方公里|公里²|千米²)[，。\s]*',
    re.UNICODE
)

def _should_drop_key(key: str) -> bool:
    key_lower = key.lower()
    return any(marker in key_lower for marker in _UNTRUSTED_KEY_MARKERS)


def _string_contains_untrusted_units(value: str) -> bool:
    value_lower = value.lower()
    if any(marker in value_lower for marker in _UNTRUSTED_STRING_MARKERS_LOWER):
        return True
    return any(marker in value for marker in _UNTRUSTED_STRING_MARKERS_LITERAL)


def _clean_area_descriptions(text: str) -> str:
    """Remove area-related descriptions from text.
    
    Args:
        text: Original text that may contain area descriptions
        
    Returns:
        Cleaned text with area descriptions removed
    """
    if not isinstance(text, str):
        return text
    
    original = text
    
    # Remove area-related phrases
    cleaned = _AREA_DESCRIPTION_PATTERN.sub('', text)
    
    # Only apply cleanup if something was changed
    if cleaned != original:
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Fix spacing after period: ensure there's no space before next character starts
        cleaned = re.sub(r'([。！？])\s+', r'\1', cleaned)
        # Clean up spaces before Chinese punctuation
        cleaned = re.sub(r'\s+([，。！？])', r'\1', cleaned)
        # Clean up multiple punctuation marks
        cleaned = re.sub(r'([，。])+', r'\1', cleaned)
        cleaned = cleaned.strip()
        # Remove leading punctuation
        cleaned = re.sub(r'^[，。\s]+', '', cleaned)
        # Remove trailing punctuation except period/exclamation/question
        cleaned = re.sub(r'[，]+$', '', cleaned)
    
    return cleaned


def sanitize_value(value: Any, drop_keys: Optional[Iterable[str]] = None) -> Any:
    drop_key_set = set(drop_keys or ())

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for key, raw in obj.items():
                if key in drop_key_set or _should_drop_key(key):
                    continue
                nested = _sanitize(raw)
                if nested is not None:
                    cleaned[key] = nested
            return cleaned or None

        if isinstance(obj, list):
            cleaned_list: List[Any] = []
            for item in obj:
                nested = _sanitize(item)
                if nested is not None:
                    cleaned_list.append(nested)
            return cleaned_list or None

        if isinstance(obj, str):
            # First clean area descriptions, then check for untrusted units
            cleaned_str = _clean_area_descriptions(obj)
            return None if _string_contains_untrusted_units(cleaned_str) else cleaned_str

        if isinstance(obj, (int, float, bool)):
            return obj

        return None

    return _sanitize(value)


def _dig(data: Any, path: Sequence[str]) -> Any:
    current = data
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _normalize_coordinate_payload(value: Any) -> Any:
    sanitized = sanitize_value(value)
    if sanitized is None:
        return None

    if isinstance(sanitized, dict):
        if "vertices" in sanitized:
            return _normalize_coordinate_payload(sanitized["vertices"])
        if "polygon" in sanitized:
            return _normalize_coordinate_payload(sanitized["polygon"])
        if "main_contour_coords" in sanitized:
            return _normalize_coordinate_payload(sanitized["main_contour_coords"])
        if "simplified_coordinates" in sanitized:
            return _normalize_coordinate_payload(sanitized["simplified_coordinates"])
        if {"lat", "lon"}.issubset(sanitized.keys()):
            lat = sanitized.get("lat")
            lon = sanitized.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                return {"lat": lat, "lon": lon}
        return sanitized or None

    if isinstance(sanitized, list):
        normalized: List[Any] = []
        for item in sanitized:
            norm = _normalize_coordinate_payload(item)
            if norm is not None:
                normalized.append(norm)
        return normalized or None

    if isinstance(sanitized, (int, float)):
        return sanitized

    return None


def _find_boundary_coordinates(shape: Dict[str, Any]) -> Any:
    for path in _COORDINATE_PATHS:
        raw = _dig(shape, path)
        coords = _normalize_coordinate_payload(raw)
        if coords:
            return coords
    return None


def clean_shape(shape: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(shape, dict):
        return None

    cleaned: Dict[str, Any] = {}

    for key in _SHAPE_TEXT_KEYS:
        value = shape.get(key)
        if isinstance(value, str) and value and not _string_contains_untrusted_units(value):
            cleaned[key] = value

    vector_coords = sanitize_value(shape.get("vector_coordinates"))
    if vector_coords:
        cleaned["vector_coordinates"] = vector_coords

    boundary = _find_boundary_coordinates(shape)
    if boundary:
        cleaned["boundary_coordinates"] = boundary

    return cleaned or None


def process_system(system: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(system, dict):
        return None

    cleaned: Dict[str, Any] = {}
    for field in ("system_name", "system_type", "description", "category"):
        value = system.get(field)
        if isinstance(value, str) and value:
            cleaned[field] = value

    position = sanitize_value(system.get("position"), drop_keys=_POSITION_DROP_KEYS)
    if position:
        cleaned["position"] = position

    intensity = sanitize_value(system.get("intensity"))
    if intensity:
        cleaned["intensity"] = intensity

    properties = sanitize_value(system.get("properties"))
    if properties:
        cleaned["properties"] = properties

    metrics = sanitize_value(system.get("metrics"))
    if metrics:
        cleaned["metrics"] = metrics

    shape = clean_shape(system.get("shape"))
    if shape:
        cleaned["shape"] = shape

    if not cleaned:
        return None
    return cleaned


def process_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None

    cleaned: Dict[str, Any] = {}
    base_fields = ("time", "time_idx", "forecast_hour", "analysis_time", "lead_time_hours")
    for field in base_fields:
        if field in entry:
            sanitized = sanitize_value(entry[field])
            if sanitized is not None:
                cleaned[field] = sanitized

    tc_fields = ("tc_position", "tc_intensity", "tc_intensity_hpa", "tc_intensity_kts")
    for field in tc_fields:
        if field in entry:
            sanitized = sanitize_value(entry[field])
            if sanitized:
                cleaned[field] = sanitized

    for key, value in entry.items():
        if key in cleaned or key in ("environmental_systems",):
            continue
        if key in base_fields or key in tc_fields:
            continue
        sanitized = sanitize_value(value)
        if sanitized is not None:
            cleaned[key] = sanitized

    systems: List[Dict[str, Any]] = []
    for system in entry.get("environmental_systems", []):
        processed = process_system(system)
        if processed:
            systems.append(processed)
    if systems:
        cleaned["environmental_systems"] = systems

    return cleaned or None


def build_processed_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    processed: Dict[str, Any] = {}

    preserved_top_level = ("tc_id", "analysis_time", "analysis_window", "dataset", "source")
    for key in preserved_top_level:
        if key in payload:
            processed[key] = payload[key]

    metadata = sanitize_value(payload.get("metadata"))
    if metadata:
        processed["metadata"] = metadata

    time_series: List[Dict[str, Any]] = []
    for entry in payload.get("time_series", []):
        processed_entry = process_entry(entry)
        if processed_entry:
            time_series.append(processed_entry)
    if "time_series" in payload:
        processed["time_series"] = time_series

    analysis_series: List[Dict[str, Any]] = []
    for entry in payload.get("environmental_analysis", []):
        processed_entry = process_entry(entry)
        if processed_entry:
            analysis_series.append(processed_entry)
    if "environmental_analysis" in payload:
        processed["environmental_analysis"] = analysis_series

    for key, value in payload.items():
        if key in processed or key in {"time_series", "environmental_analysis", "metadata"}:
            continue
        sanitized = sanitize_value(value)
        if sanitized is not None:
            processed[key] = sanitized

    return processed


def process_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    processed_payload = build_processed_payload(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(processed_payload, fp, ensure_ascii=False, indent=2)


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def run_for_directory(label: str, input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        print(f"⚠️ Skipping {label}: input directory does not exist ({input_dir})")
        return

    files = list(iter_input_files(input_dir))
    if not files:
        print(f"⚠️ No JSON files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        target = output_dir / path.name
        process_file(path, target)
        try:
            rel_target = target.relative_to(output_dir.parent)
        except ValueError:
            rel_target = target
        print(f"[{label}] ✅ Processed {path.name} -> {rel_target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter TC environment JSON outputs to keep only trusted fields."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Process a single input directory instead of the default directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for single-directory processing (defaults to '<input>_trusted').",
    )
    parser.add_argument(
        "--final-input-dir",
        type=Path,
        default=DEFAULT_FINAL_INPUT_DIR,
        help="Directory containing final_single_output JSON files.",
    )
    parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_DIR,
        help="Directory to write processed final_single_output JSON files.",
    )
    parser.add_argument(
        "--cds-input-dir",
        type=Path,
        default=DEFAULT_CDS_INPUT_DIR,
        help="Directory containing cds_output JSON files.",
    )
    parser.add_argument(
        "--cds-output-dir",
        type=Path,
        default=DEFAULT_CDS_OUTPUT_DIR,
        help="Directory to write processed cds_output JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_dir:
        input_dir: Path = args.input_dir
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = input_dir.parent / f"{input_dir.name}{DEFAULT_SINGLE_OUTPUT_SUFFIX}"
        run_for_directory("custom", input_dir, output_dir)
        return

    run_for_directory("final", args.final_input_dir, args.final_output_dir)
    run_for_directory("cds", args.cds_input_dir, args.cds_output_dir)


if __name__ == "__main__":
    main()

