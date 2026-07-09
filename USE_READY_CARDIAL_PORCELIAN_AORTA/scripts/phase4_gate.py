"""Phase 4 GATE -- 2D raster of the scored mask + stored inverse map.

Finalizes the deliverable: rasterize the curvature-corrected unwrap of the
scored calcium into a 2D image, persist the per-pixel inverse map as a compact
.npz, and verify the round trip both ways:

  * lesion Dice/IoU between the original 3D scored mask and the mask recovered by
    back-projecting the 2D raster, and
  * point -> 2D -> 3D round-trip error (bounded by ~one pixel).

No interactive click-back UI is built (deferred feature).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402
from aortic_unwrap.geometry import physical_to_voxel, voxel_to_physical  # noqa: E402
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap  # noqa: E402
from aortic_unwrap.raster import rasterize, back_project, UnwrapRaster  # noqa: E402
from aortic_unwrap import metrics  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

# Pixel = half the in-plane voxel spacing (0.7 mm) so the 2D raster does not
# under-sample the obliquely-projected vessel wall.
PIXEL_MM = 0.35
# A 2D SURFACE map collapses radially-stacked voxels (a ~2-voxel-thick calcium
# shell) onto one pixel and stores one representative radius. So Dice ceilings
# just below 1.0 and the round-trip error floors at ~one voxel (the discarded
# radial position), independent of pixel size. Thresholds reflect that limit.
DICE_MIN = 0.90
RT_MAX_MM = 1.0  # ~ one voxel (0.7 mm) + pixel


def voxelize(points, affine, shape):
    idx = np.round(physical_to_voxel(points, affine)).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[ok]
    m = np.zeros(shape, bool)
    m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return m


def save_png(ph, raster):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    u, v = raster.pixel_centers()
    ax.imshow(raster.image, origin="lower", cmap="hot",
              extent=[u[0], u[-1], v[0], v[-1]], aspect="auto")
    ax.set_xlabel("u = arclength s (mm)")
    ax.set_ylabel("v = area-corrected circumf. (mm)")
    ax.set_title(f"{ph.name}: unwrapped scored calcium")
    p = OUT / f"phase4_unwrap_{ph.name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def main():
    ok = True
    print(f"{'phantom':<18} {'#px':>5} {'Dice':>7} {'IoU':>7} "
          f"{'rt median':>10} {'rt max':>8}  npz")
    print("-" * 78)
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        pts = voxel_to_physical(ph.calcium_idx, ph.affine)
        unwrap = CurvatureCorrectedUnwrap(ph.centerline)
        res = unwrap(pts)
        raster = rasterize(res, pixel=PIXEL_MM)

        npz = OUT / f"unwrap_{name}.npz"
        raster.save(npz)
        # Prove persistence + that the stored inverse map is self-sufficient.
        reloaded = UnwrapRaster.load(npz)
        back = back_project(reloaded, unwrap)
        recovered = voxelize(back, ph.affine, ph.shape)

        d = metrics.dice(ph.calcium_deposits, recovered)
        j = metrics.iou(ph.calcium_deposits, recovered)
        rt = metrics.roundtrip_2d3d(pts, unwrap, reloaded)
        png = save_png(ph, raster)

        good = d >= DICE_MIN and rt["max_mm"] <= RT_MAX_MM
        ok &= good
        print(f"{name:<18} {int(reloaded.image.sum()):>5d} {d:>7.3f} {j:>7.3f} "
              f"{rt['median_mm']:>10.3f} {rt['max_mm']:>8.3f}  {npz.name}"
              + ("" if good else "  <-- FAIL"))
    print("-" * 78)
    print("PHASE 4 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
