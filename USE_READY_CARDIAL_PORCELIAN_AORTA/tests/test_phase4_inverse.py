"""Phase 4 -- raster + stored inverse map + round trip."""

import numpy as np

from aortic_unwrap.geometry import physical_to_voxel, voxel_to_physical
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap
from aortic_unwrap.raster import rasterize, back_project, UnwrapRaster
from aortic_unwrap import metrics

PIXEL = 0.35


def _voxelize(points, affine, shape):
    idx = np.round(physical_to_voxel(points, affine)).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[ok]
    m = np.zeros(shape, bool)
    m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return m


def test_dice_and_roundtrip(built_phantoms):
    for name, ph in built_phantoms.items():
        pts = voxel_to_physical(ph.calcium_idx, ph.affine)
        u = CurvatureCorrectedUnwrap(ph.centerline)
        rast = rasterize(u(pts), pixel=PIXEL)
        recovered = _voxelize(back_project(rast, u), ph.affine, ph.shape)
        assert metrics.dice(ph.calcium_deposits, recovered) >= 0.90, name
        rt = metrics.roundtrip_2d3d(pts, u, rast)
        assert rt["max_mm"] <= 1.0, name  # ~one-voxel radial-shell floor


def test_npz_save_load_roundtrip(tmp_path, built_phantoms):
    ph = built_phantoms["bent_tube"]
    pts = voxel_to_physical(ph.calcium_idx, ph.affine)
    u = CurvatureCorrectedUnwrap(ph.centerline)
    rast = rasterize(u(pts), pixel=PIXEL)
    p = tmp_path / "u.npz"
    rast.save(p)
    reloaded = UnwrapRaster.load(p)
    assert np.array_equal(rast.image, reloaded.image)
    assert np.allclose(rast.s_map, reloaded.s_map)
    # The reloaded inverse map alone is enough to back-project.
    back = back_project(reloaded, u)
    assert back.shape[0] == int(reloaded.image.sum())
