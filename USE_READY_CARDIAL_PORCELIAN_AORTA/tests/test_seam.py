"""Circumferential seam / theta-wraparound handling (hardening pass item 3).

A lesion straddling the theta = +/-pi branch cut must reconstruct on the 2D map
as a SINGLE connected footprint with its full area, not two split pieces.

Connectivity is evaluated at a voxel-scale pixel (0.7 mm); at the finer 0.35 mm
deliverable pixel the footprint is sampled sparsely (gaps between voxels), which
is a separate rasterization-density property unrelated to the seam.
"""

import numpy as np

from aortic_unwrap import phantoms
from aortic_unwrap.geometry import voxel_to_physical
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap
from aortic_unwrap.raster import rasterize, connected_components

PIXEL = 0.7  # voxel-scale: footprint is connected, so component counts are meaningful


def _raster(ph):
    pts = voxel_to_physical(ph.calcium_idx, ph.affine)
    u = CurvatureCorrectedUnwrap(ph.centerline)
    return rasterize(u(pts), pixel=PIXEL)


def test_seam_lesion_splits_without_periodic_handling():
    rast = _raster(phantoms.bent_tube_seam())  # theta0 = pi
    n, _ = connected_components(rast, periodic=False)
    assert n == 2, f"expected the seam to split the lesion, got {n} pieces"


def test_seam_lesion_is_single_footprint_with_periodic_handling():
    rast = _raster(phantoms.bent_tube_seam())
    n, merged = connected_components(rast, periodic=True)
    assert n == 1, f"periodic handling should merge into 1 footprint, got {n}"
    # The single footprint must contain the FULL area (no orphaned pixels).
    occupied = int(rast.image.sum())
    assert int((merged == 1).sum()) == occupied


def test_area_is_seam_placement_invariant():
    """Occupied-pixel area must not change with the periodic flag (no loss/dup)."""
    rast = _raster(phantoms.bent_tube_seam())
    occupied = int(rast.image.sum())
    for periodic in (False, True):
        _, merged = connected_components(rast, periodic=periodic)
        assert int((merged > 0).sum()) == occupied


def test_control_lesion_not_split():
    """A lesion away from the seam is one footprint either way."""
    rast = _raster(phantoms.bent_tube_seam(theta0=0.0))
    n_np, _ = connected_components(rast, periodic=False)
    n_p, _ = connected_components(rast, periodic=True)
    assert n_np == 1 and n_p == 1
