"""Environment extractor package exposing the refactored public API."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from .cli import build_parser, main
from .extractor import TCEnvironmentalSystemsExtractor
from .pipeline import process_nc_files, streaming_from_csv
from .shape_analysis import WeatherSystemShapeAnalyzer

__all__ = [
    "TCEnvironmentalSystemsExtractor",
    "WeatherSystemShapeAnalyzer",
    "build_parser",
    "main",
    "process_nc_files",
    "streaming_from_csv",
]
