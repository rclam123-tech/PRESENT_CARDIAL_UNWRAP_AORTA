"""Display-only aortic calcium unwrap.

A read-only, downstream visualization layer that PROJECTS an already-scored
``calcium_deposits`` mask onto a flattened 2D map of the aorta. It never
recomputes, resamples, or interpolates the Agatston score (Hard invariant #1).

See ``README.md`` for the phased build / gate structure.
"""

from .geometry import (
    affine_to_ras,
    make_affine,
    physical_to_voxel,
    to_ras,
    voxel_to_physical,
)
from .centerline import (
    AnalyticCenterline,
    Centerline,
    PolylineFileCenterline,
    SegmentationCenterline,
)
from .frame import accumulated_twist, rotation_minimizing_frame
from .wall import estimate_wall_radius
from .mask_io import CalciumHandoff, calcium_points, load_calcium_from_files
from .unwrap_a import (
    CenterlineProjectionUnwrap,
    CurvatureCorrectedUnwrap,
    UnwrapResult,
)
from .raster import UnwrapRaster, back_project, rasterize

__all__ = [
    "make_affine", "voxel_to_physical", "physical_to_voxel", "to_ras",
    "affine_to_ras", "Centerline", "AnalyticCenterline", "PolylineFileCenterline",
    "SegmentationCenterline",
    "rotation_minimizing_frame", "accumulated_twist", "estimate_wall_radius",
    "calcium_points",
    "CalciumHandoff", "load_calcium_from_files", "CenterlineProjectionUnwrap",
    "CurvatureCorrectedUnwrap", "UnwrapResult", "rasterize", "back_project",
    "UnwrapRaster",
]
