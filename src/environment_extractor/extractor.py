"""Core interface for tropical cyclone environmental system extraction."""

from __future__ import annotations

from .base import BaseExtractor
from .mixins import (
    AnalysisMixin,
    BoundaryExtractionMixin,
    FrontalSystemMixin,
    IntertropicalConvergenceZoneMixin,
    MonsoonTroughMixin,
    OceanHeatContentMixin,
    SteeringExtractionMixin,
    WesterlyTroughMixin,
    WindFieldExtractionMixin,
)


class TCEnvironmentalSystemsExtractor(
    BoundaryExtractionMixin,
    SteeringExtractionMixin,
    WindFieldExtractionMixin,
    OceanHeatContentMixin,
    IntertropicalConvergenceZoneMixin,
    WesterlyTroughMixin,
    FrontalSystemMixin,
    MonsoonTroughMixin,
    AnalysisMixin,
    BaseExtractor,
):
    """
    热带气旋环境场影响系统提取器
    """


__all__ = ["TCEnvironmentalSystemsExtractor"]
