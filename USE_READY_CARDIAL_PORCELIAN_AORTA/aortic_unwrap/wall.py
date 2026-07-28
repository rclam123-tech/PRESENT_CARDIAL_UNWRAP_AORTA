"""Per-vertex wall radius from the aorta segmentation, by ray-casting.

Shared, geometry-only helper (no CT, no scorer). Originally lived in repo A's
``unwrap_ct.py``; moved here so BOTH the grayscale CT panorama (repo A) and the
binary calcium projection (repo B) estimate the wall radius identically, and so
the wrap-aware vertex assignment (`Centerline.assign`) has a single definition of
"how wide is the wall here". Behaviour is unchanged from the original: 24 rays,
r_max 40 mm, r_step 0.5 mm, nearest-neighbour sampling of the segmentation, 90th
percentile over rays, 4 mm floor.
"""

from __future__ import annotations

import numpy as np

from .geometry import physical_to_voxel


def _ray_directions(cl, theta):
    """Unit ray directions (n_vertex, n_theta, 3) in each vertex's RMF plane."""
    cos = np.cos(theta)[None, :, None]
    sin = np.sin(theta)[None, :, None]
    return cos * cl.N1[:, None, :] + sin * cl.N2[:, None, :]


def estimate_wall_radius(seg, seg_affine_ras, cl, n_probe: int = 24,
                         r_max: float = 40.0, r_step: float = 0.5,
                         r_min: float = 4.0, pct: float = 90.0) -> np.ndarray:
    """Per-vertex wall radius (mm) from the segmentation, by ray-casting.

    At each centerline vertex, cast ``n_probe`` rays outward in the RMF plane
    and sample the segmentation nearest-neighbour. Each ray contributes its
    OUTERMOST in-segment radius, which steps over the small interior holes a
    real segmentation has. The vertex radius is the ``pct``-th percentile over
    rays -- robust to a few rays escaping down a branch -- clipped below at
    ``r_min``.

    Parameters
    ----------
    seg : 3D bool array
        Binary aorta segmentation.
    seg_affine_ras : 4x4
        Index -> RAS affine of ``seg``.
    cl : Centerline
        Supplies ``.points`` (RAS) and the RMF normals ``.N1`` / ``.N2``.
    """
    seg = np.asarray(seg, bool)
    theta = np.linspace(0.0, 2.0 * np.pi, n_probe, endpoint=False)
    radii = np.arange(r_step, r_max + r_step, r_step)

    dirs = _ray_directions(cl, theta)                       # (n, n_probe, 3)
    pts = (cl.points[:, None, None, :]
           + radii[None, None, :, None] * dirs[:, :, None, :])
    n, n_theta, n_r = pts.shape[:3]

    idx = physical_to_voxel(pts.reshape(-1, 3), seg_affine_ras)
    idx = np.rint(idx).astype(int)                          # nearest neighbour
    inside_grid = np.all((idx >= 0) & (idx < np.array(seg.shape)), axis=1)
    hit = np.zeros(len(idx), bool)
    ok = idx[inside_grid]
    hit[inside_grid] = seg[ok[:, 0], ok[:, 1], ok[:, 2]]
    hit = hit.reshape(n, n_theta, n_r)

    # Outermost in-segment radius per ray: the largest r whose sample is inside.
    # np.where on a reversed axis would also work; argmax on the reversed hit
    # mask finds the first hit from the outside in.
    rev = hit[:, :, ::-1]
    any_hit = rev.any(axis=2)
    first_from_outside = rev.argmax(axis=2)
    r_out = np.where(any_hit, radii[n_r - 1 - first_from_outside], np.nan)

    with np.errstate(invalid="ignore"):
        R = np.nanpercentile(r_out, pct, axis=1)
    # A vertex whose every ray missed (should not happen for a centerline that
    # lies inside the mask) falls back to the floor.
    R = np.where(np.isfinite(R), R, r_min)
    return np.clip(R, r_min, None)
