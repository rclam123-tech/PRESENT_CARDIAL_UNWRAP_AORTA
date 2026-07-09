"""Validation metrics shared by the Phase 3/4/5 gates.

Area distortion is measured *exactly* (not by counting rasterized pixels, which
carries a half-voxel rim bias). For each scored voxel we push a tiny analytic
surface quad around its known (s, theta) through the actual unwrap code and
compare the quad's area in (u, v) to its true surface area sqrt(g)*ds*dtheta.

This makes the degenerate case unambiguous: on a straight cylinder the unwrap is
an isometry, so the local Jacobian is 1 to machine precision -> 0% distortion.
Any meaningful distortion there is a bug, not geometry (the Phase 3 fork).
"""

from __future__ import annotations

import numpy as np

from .geometry import voxel_to_physical


def dice(a, b) -> float:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = np.count_nonzero(a & b)
    s = np.count_nonzero(a) + np.count_nonzero(b)
    return 1.0 if s == 0 else 2.0 * inter / s


def iou(a, b) -> float:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else inter / union


def _shoelace(quad):
    """Signed area of quads given as (..., 4, 2); returns (...,) absolute area."""
    x = quad[..., 0]
    y = quad[..., 1]
    xn = np.roll(x, -1, axis=-1)
    yn = np.roll(y, -1, axis=-1)
    return 0.5 * np.abs(np.sum(x * yn - xn * y, axis=-1))


def area_distortion(phantom, unwrap, ds=0.5, dtheta=0.02):
    """Per-voxel area-distortion factor (projected element / true element).

    Returns dict with the array and summary stats. A value of 1.0 means the
    unwrap preserves area locally; >1 inflates, <1 shrinks.
    """
    s0 = phantom.s_gt
    th0 = phantom.theta_gt
    M = len(s0)

    # 4 corners of a (ds, dtheta) cell around each voxel, in CCW order.
    offs = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    s_corners = (s0[:, None] + offs[None, :, 0] * ds).ravel()
    t_corners = (th0[:, None] + offs[None, :, 1] * dtheta).ravel()

    surf = phantom.surface(s_corners, t_corners)        # (M*4, 3) physical
    res = unwrap(surf)
    uv = np.stack([res.u, res.v], axis=1).reshape(M, 4, 2)
    proj_area = _shoelace(uv)                            # (M,) area in (u,v)

    # True surface area of each cell via sqrt(g) at the center.
    sqrtg = np.array([phantom.metric(s0[i], th0[i]) for i in range(M)])
    true_area = sqrtg * ds * dtheta

    factor = proj_area / true_area
    return {
        "factor": factor,
        "median": float(np.median(factor)),
        "mean": float(np.mean(factor)),
        "p95": float(np.percentile(factor, 95)),
        "median_abs_pct": float(np.median(np.abs(factor - 1.0)) * 100),
        "lesion_true_area": float(phantom.lesion_true_area),
        "lesion_projected_area": float(phantom.lesion_true_area * np.mean(factor)),
    }


def localization_error(phantom, unwrap):
    """Implemented (s, theta) vs analytic ground truth, in mm."""
    pts = voxel_to_physical(phantom.calcium_idx, phantom.affine)
    res = unwrap(pts)
    ds = np.abs(res.s - phantom.s_gt)
    dth = np.arctan2(np.sin(res.theta - phantom.theta_gt),
                     np.cos(res.theta - phantom.theta_gt))
    dcirc = np.abs(dth) * phantom.r_gt
    return {
        "median_ds_mm": float(np.median(ds)),
        "max_ds_mm": float(np.max(ds)),
        "median_dcirc_mm": float(np.median(dcirc)),
        "max_dcirc_mm": float(np.max(dcirc)),
    }


def roundtrip_2d3d(points, unwrap, raster):
    """point -> 2D pixel -> back to 3D, max error (mm). Bounded by ~pixel size."""
    from .raster import back_project

    res = unwrap(points)
    col = np.clip(((res.u - raster.u_min) / raster.pixel).astype(int), 0, raster.image.shape[1] - 1)
    row = np.clip(((res.v - raster.v_min) / raster.pixel).astype(int), 0, raster.image.shape[0] - 1)
    s = raster.s_map[row, col]
    theta = raster.theta_map[row, col]
    r = raster.r_map[row, col]
    back = unwrap.inverse(s, theta, r)
    err = np.linalg.norm(back - points, axis=1)
    return {"median_mm": float(np.median(err)), "max_mm": float(np.max(err))}
