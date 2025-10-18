from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest

from environment_extractor.extractor import TCEnvironmentalSystemsExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPECTED_DIR = DATA_DIR / "testExtract"
TRACK_DIR = DATA_DIR / "test" / "tracks"


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy types into Python builtins."""
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def assert_deep_almost_equal(actual: Any, expected: Any, path: str = "root") -> None:
    """Assert two nested structures are equal within a numeric tolerance."""
    actual = to_builtin(actual)
    expected = to_builtin(expected)

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict got {type(actual)!r}"
        assert actual.keys() == expected.keys(), f"{path}: keys differ {actual.keys()} vs {expected.keys()}"
        for key in expected:
            assert_deep_almost_equal(actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list got {type(actual)!r}"
        assert len(actual) == len(expected), f"{path}: list lengths differ {len(actual)} vs {len(expected)}"
        for idx, (a_item, e_item) in enumerate(zip(actual, expected)):
            assert_deep_almost_equal(a_item, e_item, f"{path}[{idx}]")
        return

    if isinstance(expected, bool):
        assert actual is expected, f"{path}: expected {expected!r} got {actual!r}"
        return

    if isinstance(expected, numbers.Real):
        assert isinstance(actual, numbers.Real), f"{path}: expected numeric got {type(actual)!r}"
        if math.isnan(expected):
            assert math.isnan(actual), f"{path}: expected NaN got {actual!r}"
        else:
            assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6), (
                f"{path}: values differ (actual={actual!r}, expected={expected!r})"
            )
        return

    assert actual == expected, f"{path}: expected {expected!r} got {actual!r}"


def resolve_path(path_str: str) -> Path:
    """Resolve a path stored in JSON (relative to project root)."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_track_for_dataset(dataset_name: str) -> Path:
    matches = [p for p in TRACK_DIR.glob("*.csv") if dataset_name in p.name]
    if len(matches) != 1:
        raise ValueError(f"Unable to uniquely match track file for dataset '{dataset_name}'")
    return matches[0]


def dataset_nc_path(dataset_name: str) -> Path:
    return DATA_DIR / f"{dataset_name}.nc"


def load_track_position(track_path: Path, time_idx: int) -> tuple[float, float]:
    track_df = pd.read_csv(track_path)
    row = track_df.iloc[time_idx]
    return float(row["lat"]), float(row["lon"])


def run_extractor_case(
    method_name: str,
    nc_path: Path,
    track_path: Path,
    time_idx: int,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    method_kwargs: dict[str, Any] | None = None,
) -> Any:
    if latitude is None or longitude is None:
        latitude, longitude = load_track_position(track_path, time_idx)

    kwargs = method_kwargs or {}
    with TCEnvironmentalSystemsExtractor(nc_path, track_path) as extractor:
        method = getattr(extractor, method_name)
        result = method(time_idx, latitude, longitude, **kwargs)
    return to_builtin(result)


# --- Steering system regression ------------------------------------------------

STEERING_EXPECTED_FILES: Iterable[Path] = [
    EXPECTED_DIR / "steering/test_results_auro_v5_improved.json",
    EXPECTED_DIR / "steering/test_results_four_v5_improved.json",
    EXPECTED_DIR / "steering/test_results_pang_v5_improved.json",
]


@pytest.mark.parametrize("expected_file", STEERING_EXPECTED_FILES, ids=lambda p: p.stem)
def test_steering_system_regression(expected_file: Path) -> None:
    expected_raw = json.loads(expected_file.read_text(encoding="utf-8"))
    info = expected_raw["test_info"]
    nc_path = resolve_path(info["nc_file"])
    track_path = resolve_path(info["track_file"])
    time_idx = info["time_idx"]
    typhoon_position = expected_raw["typhoon_position"]
    actual = run_extractor_case(
        "extract_steering_system",
        nc_path,
        track_path,
        time_idx,
        latitude=typhoon_position["latitude"],
        longitude=typhoon_position["longitude"],
    )
    assert_deep_almost_equal(actual, expected_raw["steering_system"])


# --- Vertical wind shear regression -------------------------------------------

WIND_SHEAR_EXPECTED_FILES: Iterable[Path] = [
    EXPECTED_DIR / "windShear/test_results_auro.json",
    EXPECTED_DIR / "windShear/test_results_four.json",
    EXPECTED_DIR / "windShear/test_results_pang.json",
]


@pytest.mark.parametrize("expected_file", WIND_SHEAR_EXPECTED_FILES, ids=lambda p: p.stem)
def test_vertical_wind_shear_regression(expected_file: Path) -> None:
    expected_raw = json.loads(expected_file.read_text(encoding="utf-8"))
    info = expected_raw["test_info"]
    nc_path = resolve_path(info["nc_file"])
    track_path = resolve_path(info["track_file"])
    time_idx = info["time_idx"]
    lat = info["tc_position"]["lat"]
    lon = info["tc_position"]["lon"]
    actual = run_extractor_case(
        "extract_vertical_wind_shear",
        nc_path,
        track_path,
        time_idx,
        latitude=lat,
        longitude=lon,
    )
    assert_deep_almost_equal(actual, expected_raw["wind_shear"])


# --- Upper-level divergence regression ----------------------------------------

DIVERGENCE_EXPECTED_FILES: Iterable[Path] = [
    EXPECTED_DIR / "divergence/AURO_divergence.json",
    EXPECTED_DIR / "divergence/FOUR_divergence.json",
    EXPECTED_DIR / "divergence/PANG_divergence.json",
]


@pytest.mark.parametrize("expected_file", DIVERGENCE_EXPECTED_FILES, ids=lambda p: p.stem)
def test_upper_level_divergence_regression(expected_file: Path) -> None:
    expected_raw = json.loads(expected_file.read_text(encoding="utf-8"))
    info = expected_raw["test_info"]
    nc_path = resolve_path(info["nc_file"])
    track_path = resolve_path(info["track_file"])
    time_idx = info["time_idx"]
    lat = info["tc_position"]["lat"]
    lon = info["tc_position"]["lon"]
    actual = run_extractor_case(
        "extract_upper_level_divergence",
        nc_path,
        track_path,
        time_idx,
        latitude=lat,
        longitude=lon,
    )
    assert_deep_almost_equal(actual, expected_raw["divergence"])


# --- Frontal system regression -------------------------------------------------

FRONTAL_EXPECTED_FILES: Iterable[Path] = [
    EXPECTED_DIR / "frontal/AURO_v100_IFS_2025061000_f000_f240_06_time0_frontal.json",
    EXPECTED_DIR / "frontal/PANG_v100_IFS_2022032900_f000_f240_06_time0_frontal.json",
    EXPECTED_DIR / "frontal/FOUR_v200_GFS_2020093012_f000_f240_06_time0_frontal.json",
    EXPECTED_DIR / "frontal/FOUR_v200_GFS_2020093012_f000_f240_06_time5_frontal.json",
]


@pytest.mark.parametrize("expected_file", FRONTAL_EXPECTED_FILES, ids=lambda p: p.stem)
def test_frontal_system_regression(expected_file: Path) -> None:
    expected_raw = json.loads(expected_file.read_text(encoding="utf-8"))
    nc_path = resolve_path(expected_raw["file"])
    track_path = resolve_path(expected_raw["track_file"])
    time_idx = expected_raw["time_idx"]
    tc_position = expected_raw["tc_position"]
    actual = run_extractor_case(
        "extract_frontal_system",
        nc_path,
        track_path,
        time_idx,
        latitude=tc_position["lat"],
        longitude=tc_position["lon"],
    )
    assert_deep_almost_equal(actual, expected_raw["frontal_system"])


# --- Westerly trough regression -----------------------------------------------

WESTERLY_EXPECTED_FILES: Iterable[Path] = [
    EXPECTED_DIR / "westerlyTrough/AURO_v100_IFS_2025061000_f000_f240_06_westerly_trough.json",
    EXPECTED_DIR / "westerlyTrough/PANG_v100_IFS_2022032900_f000_f240_06_westerly_trough.json",
    EXPECTED_DIR / "westerlyTrough/FOUR_v200_GFS_2020093012_f000_f240_06_westerly_trough.json",
]


@pytest.mark.parametrize("expected_file", WESTERLY_EXPECTED_FILES, ids=lambda p: p.stem)
def test_westerly_trough_regression(expected_file: Path) -> None:
    expected_entries = json.loads(expected_file.read_text(encoding="utf-8"))
    dataset_name = expected_file.stem.replace("_westerly_trough", "")
    nc_path = dataset_nc_path(dataset_name)
    track_path = find_track_for_dataset(dataset_name)

    for idx, entry in enumerate(expected_entries):
        time_idx = entry["time_idx"]
        tc_position = entry["tc_position"]
        actual = run_extractor_case(
            "extract_westerly_trough",
            nc_path,
            track_path,
            time_idx,
            latitude=tc_position["lat"],
            longitude=tc_position["lon"],
        )
        assert_deep_almost_equal(actual, entry["westerly_trough"], path=f"{expected_file.stem}[{idx}]")


# --- Ocean heat content regression --------------------------------------------

OCEAN_HEAT_CASES = [
    ("test_ocean_heat_AURO.json", "AURO_v100_IFS_2025061000_f000_f240_06"),
    ("test_ocean_heat_PANG.json", "PANG_v100_IFS_2022032900_f000_f240_06"),
    ("test_ocean_heat_FOUR.json", "FOUR_v200_GFS_2020093012_f000_f240_06"),
]


@pytest.mark.parametrize("filename,dataset", OCEAN_HEAT_CASES, ids=lambda x: x[1])
def test_ocean_heat_content_regression(filename: str, dataset: str) -> None:
    expected_path = EXPECTED_DIR / "oceanHeat" / filename
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    nc_path = dataset_nc_path(dataset)
    track_path = find_track_for_dataset(dataset)
    actual = run_extractor_case(
        "extract_ocean_heat_content",
        nc_path,
        track_path,
        time_idx=0,
        method_kwargs={"radius_deg": 2.0},
    )
    assert_deep_almost_equal(actual, expected)


# --- ITCZ regression ----------------------------------------------------------

ITCZ_CASES = [
    ("AURO_v100_IFS_2025061000_f000_f240_06_itcz.json", "AURO_v100_IFS_2025061000_f000_f240_06"),
    ("PANG_v100_IFS_2022032900_f000_f240_06_itcz.json", "PANG_v100_IFS_2022032900_f000_f240_06"),
    ("FOUR_v200_GFS_2020093012_f000_f240_06_itcz.json", "FOUR_v200_GFS_2020093012_f000_f240_06"),
]


@pytest.mark.parametrize("filename,dataset", ITCZ_CASES, ids=lambda x: x[1])
def test_itcz_regression(filename: str, dataset: str) -> None:
    expected_path = EXPECTED_DIR / "itcz" / filename
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    nc_path = dataset_nc_path(dataset)
    track_path = find_track_for_dataset(dataset)
    actual = run_extractor_case(
        "extract_intertropical_convergence_zone",
        nc_path,
        track_path,
        time_idx=0,
    )
    assert_deep_almost_equal(actual, expected)


# --- Monsoon trough regression -------------------------------------------------

MONSOON_CASES = [
    ("AURO_v100_IFS_2025061000_f000_f240_06_monsoon.json", "AURO_v100_IFS_2025061000_f000_f240_06"),
    ("PANG_v100_IFS_2022032900_f000_f240_06_monsoon.json", "PANG_v100_IFS_2022032900_f000_f240_06"),
]


@pytest.mark.parametrize("filename,dataset", MONSOON_CASES, ids=lambda x: x[1])
def test_monsoon_trough_regression(filename: str, dataset: str) -> None:
    expected_path = EXPECTED_DIR / "monsoonTrough" / filename
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    nc_path = dataset_nc_path(dataset)
    track_path = find_track_for_dataset(dataset)
    actual = run_extractor_case(
        "extract_monsoon_trough",
        nc_path,
        track_path,
        time_idx=0,
    )
    assert_deep_almost_equal(actual, expected)


# --- Subtropical high regression (direct snapshot) ----------------------------

@pytest.mark.parametrize(
    "expected_file",
    [EXPECTED_DIR / "subtropicalHigh/AURO_v100_IFS_2025061000_f000_f240_06_subtropical_high_t0.json"],
    ids=lambda p: p.stem,
)
def test_subtropical_high_snapshot(expected_file: Path) -> None:
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    dataset = expected_file.stem.replace("_subtropical_high_t0", "")
    nc_path = dataset_nc_path(dataset)
    track_path = find_track_for_dataset(dataset)
    actual = run_extractor_case(
        "extract_steering_system",
        nc_path,
        track_path,
        time_idx=0,
    )
    assert_deep_almost_equal(actual, expected)
