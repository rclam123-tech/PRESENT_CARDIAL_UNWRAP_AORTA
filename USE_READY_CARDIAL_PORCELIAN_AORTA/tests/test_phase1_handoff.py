"""Phase 1 -- mask handoff contract (coordinate correctness)."""

import numpy as np

from aortic_unwrap.geometry import physical_to_voxel, to_ras, voxel_to_physical
from aortic_unwrap.mask_io import calcium_points, roundtrip_error


def test_roundtrip_subvoxel(built_phantoms):
    for ph in built_phantoms.values():
        h = calcium_points(ph.calcium_deposits, ph.affine, system="RAS",
                           ct_shape=ph.ct.shape)
        rt = roundtrip_error(h.points_ras, h.affine_ras)
        assert rt < 0.5 * float(np.min(h.spacing)), ph.name


def test_no_mirror_or_flip(built_phantoms):
    """Recovered voxels must land back inside the scored mask (no flip)."""
    for ph in built_phantoms.values():
        h = calcium_points(ph.calcium_deposits, ph.affine, system="RAS")
        rec = np.round(physical_to_voxel(h.points_ras, h.affine_ras)).astype(int)
        in_mask = ph.calcium_deposits[rec[:, 0], rec[:, 1], rec[:, 2]]
        assert in_mask.mean() > 0.999, ph.name


def test_lps_to_ras_conversion(built_phantoms):
    """If the same grid is declared LPS, points must equal RAS with x,y flipped."""
    ph = built_phantoms["straight_cylinder"]
    ras = calcium_points(ph.calcium_deposits, ph.affine, system="RAS").points_ras
    lps = calcium_points(ph.calcium_deposits, ph.affine, system="LPS").points_ras
    # LPS interpretation flips x and y relative to RAS.
    assert np.allclose(lps, to_ras(ras, "LPS"))
    assert np.allclose(lps[:, 0], -ras[:, 0])
    assert np.allclose(lps[:, 1], -ras[:, 1])
    assert np.allclose(lps[:, 2], ras[:, 2])


def test_shape_mismatch_rejected(built_phantoms):
    ph = built_phantoms["straight_cylinder"]
    import pytest
    with pytest.raises(ValueError):
        calcium_points(ph.calcium_deposits, ph.affine, ct_shape=(1, 2, 3))
