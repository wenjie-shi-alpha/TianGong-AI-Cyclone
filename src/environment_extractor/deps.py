"""Optional third-party dependency imports for environment extractor modules."""

from __future__ import annotations

from typing import Callable


class MissingDependencyError(ImportError):
    """Raised when the required optional scientific python stack is missing."""


ERROR_MESSAGE = "错误：需要scipy和scikit-image库。请运行 'pip install scipy scikit-image' 进行安装。"


def _missing_dependency_stub(*_args, **_kwargs):  # pragma: no cover - runtime guard
    raise MissingDependencyError(ERROR_MESSAGE)


try:  # pragma: no cover - delegated to runtime execution
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        center_of_mass,
        find_objects,
        label,
    )
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    from skimage.measure import approximate_polygon, find_contours, regionprops
    from skimage.morphology import convex_hull_image

    OPTIONAL_DEPS_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - import side-effect guard
    OPTIONAL_DEPS_AVAILABLE = False
    _IMPORT_ERROR = exc

    binary_dilation = _missing_dependency_stub
    binary_erosion = _missing_dependency_stub
    center_of_mass = _missing_dependency_stub
    find_objects = _missing_dependency_stub
    label = _missing_dependency_stub
    ConvexHull = _missing_dependency_stub  # type: ignore[assignment]
    pdist = _missing_dependency_stub
    approximate_polygon = _missing_dependency_stub
    find_contours = _missing_dependency_stub
    regionprops = _missing_dependency_stub
    convex_hull_image = _missing_dependency_stub


def ensure_available() -> None:
    """Raise a descriptive error if optional dependencies are missing."""

    if not OPTIONAL_DEPS_AVAILABLE:
        raise MissingDependencyError(ERROR_MESSAGE) from _IMPORT_ERROR


__all__ = [
    "approximate_polygon",
    "binary_dilation",
    "binary_erosion",
    "center_of_mass",
    "convex_hull_image",
    "ensure_available",
    "find_contours",
    "find_objects",
    "label",
    "pdist",
    "regionprops",
    "ConvexHull",
    "MissingDependencyError",
    "ERROR_MESSAGE",
    "OPTIONAL_DEPS_AVAILABLE",
]
