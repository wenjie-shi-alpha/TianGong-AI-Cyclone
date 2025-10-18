"""Collection of mixins composing the TC environmental systems extractor."""

from .analysis import AnalysisMixin
from .boundary import BoundaryExtractionMixin
from .frontal import FrontalSystemMixin
from .itcz import IntertropicalConvergenceZoneMixin
from .monsoon import MonsoonTroughMixin
from .ocean import OceanHeatContentMixin
from .steering import SteeringExtractionMixin
from .westerly import WesterlyTroughMixin
from .wind import WindFieldExtractionMixin

__all__ = [
    "AnalysisMixin",
    "BoundaryExtractionMixin",
    "FrontalSystemMixin",
    "IntertropicalConvergenceZoneMixin",
    "MonsoonTroughMixin",
    "OceanHeatContentMixin",
    "SteeringExtractionMixin",
    "WesterlyTroughMixin",
    "WindFieldExtractionMixin",
]
