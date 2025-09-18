"""Lightweight batch containers that mimic the original Aurora Batch interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class _Metadata:
    lat: np.ndarray
    lon: np.ndarray
    time: List[pd.Timestamp]
    atmos_levels: List[int] | np.ndarray


@dataclass
class _SimpleBatch:
    """A minimal batch object compatible with the legacy tracking algorithm."""

    atmos_vars: Dict[str, np.ndarray]
    surf_vars: Dict[str, np.ndarray]
    static_vars: Dict[str, np.ndarray]
    metadata: _Metadata

    def to(self, _: str) -> "_SimpleBatch":
        """Interface compatibility stub; no device movement is required."""
        return self


__all__ = ["_Metadata", "_SimpleBatch"]
