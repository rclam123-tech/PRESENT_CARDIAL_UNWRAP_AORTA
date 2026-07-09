"""Phase 6a -- phantom-validate the SegmentationCenterline extractor.

The extractor is validated against the analytic centerlines (the only oracle).
Position error is analytic -> extracted (ground-truth coverage); it is NOT
penalized by the phantom tubes being uncapped (they extend to the grid boundary,
so the extraction legitimately runs longer than the analytic span).
"""

import numpy as np
from scipy.spatial import cKDTree

from aortic_unwrap.centerline import SegmentationCenterline

ANALYTIC_KAPPA = {
    "straight_cylinder": 0.0,
    "bent_tube": 1.0 / 60.0,
    "tapered_tube": 0.0,
    "bent_aneurysm": 1.0 / 55.0,
}


def test_position_error_subvoxel(built_phantoms):
    for name, ph in built_phantoms.items():
        vox = float(np.min(ph.meta["spacing"]))
        cl = SegmentationCenterline.from_mask(ph.aorta_mask, ph.affine, system="RAS")
        d, _ = cKDTree(cl.points).query(ph.centerline.points)  # analytic -> extracted
        assert np.median(d) < vox, f"{name}: median pos err {np.median(d):.3f}"


def test_curvature_plausible(built_phantoms):
    for name, ph in built_phantoms.items():
        cl = SegmentationCenterline.from_mask(ph.aorta_mask, ph.affine, system="RAS")
        d_ext, _ = cKDTree(ph.centerline.points).query(cl.points)
        k = float(np.median(cl.kappa[d_ext < 2.0]))
        an = ANALYTIC_KAPPA[name]
        if an == 0.0:
            assert k < 0.01, f"{name}: kappa {k:.5f} should be ~0"
        else:
            assert abs(k - an) / an < 0.30, f"{name}: kappa {k:.5f} vs {an:.5f}"


def test_single_clean_path_and_endpoints(built_phantoms):
    ph = built_phantoms["bent_tube"]
    cl = SegmentationCenterline.from_mask(ph.aorta_mask, ph.affine, system="RAS")
    # single simple polyline -> 2 endpoints, 0 branch points, no NaNs
    assert cl.endpoints_ras.shape == (2, 3)
    assert np.all(np.isfinite(cl.points))
    assert cl.length > 0
