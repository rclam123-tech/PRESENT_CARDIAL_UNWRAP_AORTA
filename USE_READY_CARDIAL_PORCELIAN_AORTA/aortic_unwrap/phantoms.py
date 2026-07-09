"""Phase 0 -- analytic phantoms with known geometry and known calcium.

Hard invariant #4 (Master Prompt): phantoms with known geometry are the ONLY
source of ground-truth area. Each phantom is rasterized onto a realistic,
anisotropic, sign-flipped (LPS-style) physical CT grid so that the coordinate
plumbing is genuinely exercised, and each carries one synthetic calcium lesion
at a KNOWN (s, theta) on the wall with a KNOWN surface area.

Four phantoms, escalating difficulty:
  1. straight cylinder        -- degenerate, zero area distortion expected
  2. constant-curvature bend  -- tests curvature
  3. tapered tube             -- tests radius variation
  4. bent tube with aneurysm  -- tests curvature + eccentric radius (the arch)

Each phantom emits, in the SAME shapes a real scorer/segmentation would:
  * ``ct``               float32 volume (HU-like)
  * ``aorta_mask``       bool volume
  * ``calcium_deposits`` bool volume  (a label volume on the CT grid)
  * ``centerline``       AnalyticCenterline (physical RAS)
  * ground-truth table   per-calcium-voxel true (s, theta, r) + lesion area
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .centerline import AnalyticCenterline
from .geometry import make_affine, voxel_to_physical

# Realistic-ish CT HU values for a non-contrast study.
HU_AIR = -1000.0
HU_SOFT = 40.0  # vessel wall / blood pool
HU_CALCIUM = 600.0


@dataclass
class Phantom:
    name: str
    affine: np.ndarray
    shape: tuple
    ct: np.ndarray
    aorta_mask: np.ndarray
    calcium_deposits: np.ndarray
    centerline: AnalyticCenterline
    radius_fn: object  # callable s -> radius
    calcium_idx: np.ndarray  # (M, 3) voxel indices of scored voxels
    s_gt: np.ndarray         # (M,) true arclength per scored voxel
    theta_gt: np.ndarray     # (M,) true circumferential angle per scored voxel
    r_gt: np.ndarray         # (M,) true wall radius per scored voxel
    lesion_true_area: float  # mm^2, analytic surface area of the lesion patch
    lesion_s0: float
    lesion_theta0: float
    meta: dict = field(default_factory=dict)

    # -- analytic surface of the tube wall -----------------------------------
    def surface(self, s, theta) -> np.ndarray:
        """Physical points on the tube wall at arclength ``s``, angle ``theta``."""
        s = np.atleast_1d(np.asarray(s, float))
        theta = np.atleast_1d(np.asarray(theta, float))
        C = self.centerline.point_at(s)
        T, N1, N2 = self.centerline.frame_at(s)
        r = np.asarray(self.radius_fn(s), float).reshape(-1, 1)
        ct = np.cos(theta).reshape(-1, 1)
        st = np.sin(theta).reshape(-1, 1)
        return C + r * (ct * N1 + st * N2)

    def metric(self, s, theta, ds=0.25, dtheta=0.01) -> float:
        """Surface area element sqrt(g) at (s, theta), by central differences."""
        s = float(s)
        theta = float(theta)
        dSds = (self.surface(s + ds, theta)[0] - self.surface(s - ds, theta)[0]) / (2 * ds)
        dSdt = (self.surface(s, theta + dtheta)[0] - self.surface(s, theta - dtheta)[0]) / (2 * dtheta)
        return float(np.linalg.norm(np.cross(dSds, dSdt)))


def _grid_for_bbox(bbox_min, bbox_max, spacing, direction):
    """Axis-aligned (diagonal-direction) grid + affine covering a world bbox."""
    spacing = np.asarray(spacing, float)
    direction = np.asarray(direction, float)
    step = np.diag(direction) * spacing  # signed world step per +1 voxel index
    origin = np.empty(3)
    shape = np.empty(3, int)
    for a in range(3):
        span = bbox_max[a] - bbox_min[a]
        n = int(np.ceil(span / abs(step[a]))) + 1
        shape[a] = n
        origin[a] = bbox_max[a] if step[a] < 0 else bbox_min[a]
    affine = make_affine(spacing, direction, origin)
    return affine, tuple(int(x) for x in shape)


def _curvature_outer_angle(centerline, s0):
    """Angle (in the RMF) pointing along the OUTER wall of the bend at s0.

    Placing the lesion here maximizes the curvature term (1 - kappa r cos phi)
    so the bent/aneurysm phantoms genuinely stress the area gate.
    """
    s = np.array([s0 - 1.0, s0, s0 + 1.0])
    P = centerline.point_at(s)
    accel = P[0] - 2 * P[1] + P[2]  # ~ -d2C/ds2 direction (toward outer wall)
    _, N1, N2 = centerline.frame_at([s0])
    a = accel @ N1[0]
    b = accel @ N2[0]
    if a == 0 and b == 0:
        return 0.0
    return float(np.arctan2(b, a))


def _build(name, center_fn, t_max, radius_fn, spacing, direction,
           lesion_frac=0.5, lesion_ds=10.0, lesion_dtheta=np.deg2rad(40.0),
           n_centerline=400, place_outer=True, lesion_theta0=None, rng_seed=0):
    centerline = AnalyticCenterline.from_function(center_fn, t_max, n=n_centerline)
    L = centerline.length

    # ---- physical grid that comfortably contains the tube ------------------
    s_samples = np.linspace(0, L, 200)
    rmax = float(np.max(radius_fn(s_samples)))
    margin = rmax + 5.0
    pts = centerline.points
    bbox_min = pts.min(axis=0) - margin
    bbox_max = pts.max(axis=0) + margin
    affine, shape = _grid_for_bbox(bbox_min, bbox_max, spacing, direction)

    # ---- rasterize the solid tube into the grid ----------------------------
    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                             np.arange(shape[2]), indexing="ij")
    idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    phys = voxel_to_physical(idx, affine)
    knn = centerline.nearest(phys)
    cl_pts = centerline.points[knn]
    _, N1, N2 = centerline.frame_at(centerline.s[knn])
    d = phys - cl_pts
    a = np.sum(d * N1, axis=1)
    b = np.sum(d * N2, axis=1)
    radial = np.hypot(a, b)
    r_at = np.asarray(radius_fn(centerline.s[knn]), float)
    inside = radial <= r_at

    aorta_mask = inside.reshape(shape)
    ct = np.full(shape, HU_AIR, np.float32)
    ct[aorta_mask] = HU_SOFT

    # ---- place a calcium lesion at a known (s, theta) on the wall ----------
    s0 = lesion_frac * L
    if lesion_theta0 is not None:
        theta0 = float(lesion_theta0)
    elif place_outer:
        theta0 = _curvature_outer_angle(centerline, s0)
    else:
        theta0 = 0.0

    # Dense (s, theta) sampling of the lesion patch -> voxels.
    ns, nth = 80, 80
    sg = np.linspace(s0 - lesion_ds / 2, s0 + lesion_ds / 2, ns)
    tg = np.linspace(theta0 - lesion_dtheta / 2, theta0 + lesion_dtheta / 2, nth)
    SS, TT = np.meshgrid(sg, tg, indexing="ij")
    surf = _surface_vec(centerline, radius_fn, SS.ravel(), TT.ravel())
    from .geometry import physical_to_voxel
    vidx = np.round(physical_to_voxel(surf, affine)).astype(int)

    # Aggregate true (s, theta, r) per unique voxel (mean of contributing samples).
    calcium = np.zeros(shape, bool)
    acc = {}
    s_flat = SS.ravel(); t_flat = TT.ravel()
    r_flat = np.asarray(radius_fn(s_flat), float)
    for n in range(len(vidx)):
        i, j, k = vidx[n]
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            continue
        key = (i, j, k)
        if key not in acc:
            acc[key] = [0, 0.0, 0.0, 0.0, 0.0]  # count, s, sin, cos, r
        acc[key][0] += 1
        acc[key][1] += s_flat[n]
        acc[key][2] += np.sin(t_flat[n])
        acc[key][3] += np.cos(t_flat[n])
        acc[key][4] += r_flat[n]

    cidx, s_gt, theta_gt, r_gt = [], [], [], []
    for (i, j, k), (c, ssum, sin_s, cos_s, rsum) in acc.items():
        calcium[i, j, k] = True
        cidx.append((i, j, k))
        s_gt.append(ssum / c)
        theta_gt.append(np.arctan2(sin_s / c, cos_s / c))
        r_gt.append(rsum / c)
    cidx = np.array(cidx, int)
    s_gt = np.array(s_gt); theta_gt = np.array(theta_gt); r_gt = np.array(r_gt)

    aorta_mask[cidx[:, 0], cidx[:, 1], cidx[:, 2]] = True
    ct[cidx[:, 0], cidx[:, 1], cidx[:, 2]] = HU_CALCIUM

    # ---- analytic lesion surface area (the ONLY area ground truth) ----------
    lesion_true_area = _patch_area(centerline, radius_fn,
                                   s0 - lesion_ds / 2, s0 + lesion_ds / 2,
                                   theta0 - lesion_dtheta / 2, theta0 + lesion_dtheta / 2)

    return Phantom(
        name=name, affine=affine, shape=shape, ct=ct, aorta_mask=aorta_mask,
        calcium_deposits=calcium, centerline=centerline, radius_fn=radius_fn,
        calcium_idx=cidx, s_gt=s_gt, theta_gt=theta_gt, r_gt=r_gt,
        lesion_true_area=lesion_true_area, lesion_s0=s0, lesion_theta0=theta0,
        meta={"length": L, "rmax": rmax, "spacing": tuple(spacing)},
    )


def _surface_vec(centerline, radius_fn, s, theta):
    C = centerline.point_at(s)
    _, N1, N2 = centerline.frame_at(s)
    r = np.asarray(radius_fn(s), float).reshape(-1, 1)
    ct = np.cos(theta).reshape(-1, 1)
    st = np.sin(theta).reshape(-1, 1)
    return C + r * (ct * N1 + st * N2)


def _patch_area(centerline, radius_fn, s_lo, s_hi, th_lo, th_hi, n=60):
    """Analytic surface area of an (s, theta) patch = integral of sqrt(g)."""
    sg = np.linspace(s_lo, s_hi, n)
    tg = np.linspace(th_lo, th_hi, n)
    ds = (s_hi - s_lo) / (n - 1)
    dth = (th_hi - th_lo) / (n - 1)
    SS, TT = np.meshgrid(sg, tg, indexing="ij")
    # sqrt(g) via central differences of the analytic surface.
    h_s, h_t = ds, dth
    s_p = _surface_vec(centerline, radius_fn, (SS + h_s).ravel(), TT.ravel())
    s_m = _surface_vec(centerline, radius_fn, (SS - h_s).ravel(), TT.ravel())
    t_p = _surface_vec(centerline, radius_fn, SS.ravel(), (TT + h_t).ravel())
    t_m = _surface_vec(centerline, radius_fn, SS.ravel(), (TT - h_t).ravel())
    dSds = (s_p - s_m) / (2 * h_s)
    dSdt = (t_p - t_m) / (2 * h_t)
    sqrtg = np.linalg.norm(np.cross(dSds, dSdt), axis=1)
    # Trapezoid weights on a regular grid ~ midpoint sum.
    return float(np.mean(sqrtg) * (s_hi - s_lo) * (th_hi - th_lo))


# --------------------------------------------------------------------------- #
# Concrete phantoms
# --------------------------------------------------------------------------- #
SPACING = (0.7, 0.7, 0.8)
DIRECTION = np.diag([-1.0, -1.0, 1.0])  # LPS-style sign flip on x, y


def straight_cylinder():
    R = 12.0
    return _build(
        "straight_cylinder",
        center_fn=lambda t: np.array([0.0, 0.0, t]),
        t_max=120.0,
        radius_fn=lambda s: np.full_like(np.atleast_1d(s), R, float) if np.ndim(s) else R,
        spacing=SPACING, direction=DIRECTION, place_outer=False,
    )


def bent_tube():
    R = 11.0
    rad = 60.0  # radius of curvature of the centerline
    return _build(
        "bent_tube",
        center_fn=lambda t: np.array([rad * np.cos(t), 0.0, rad * np.sin(t)]),
        t_max=np.deg2rad(150.0),
        radius_fn=lambda s: np.full_like(np.atleast_1d(s), R, float) if np.ndim(s) else R,
        spacing=SPACING, direction=DIRECTION,
    )


def tapered_tube():
    r0, r1 = 14.0, 7.0
    L = 120.0

    def radius_fn(s):
        s = np.asarray(s, float)
        return r0 + (r1 - r0) * np.clip(s / L, 0, 1)

    return _build(
        "tapered_tube",
        center_fn=lambda t: np.array([0.0, 0.0, t]),
        t_max=L, radius_fn=radius_fn,
        spacing=SPACING, direction=DIRECTION, place_outer=False,
    )


def bent_aneurysm():
    R0 = 11.0
    rad = 55.0
    t_max = np.deg2rad(160.0)
    L_approx = rad * t_max

    def radius_fn(s):
        s = np.asarray(s, float)
        # Gaussian bulge centered mid-arch.
        bulge = 9.0 * np.exp(-0.5 * ((s - 0.5 * L_approx) / (0.10 * L_approx)) ** 2)
        return R0 + bulge

    return _build(
        "bent_aneurysm",
        center_fn=lambda t: np.array([rad * np.cos(t), 0.0, rad * np.sin(t)]),
        t_max=t_max, radius_fn=radius_fn,
        spacing=SPACING, direction=DIRECTION,
    )


def bent_tube_seam(theta0=np.pi):
    """Bent tube with the lesion centered on the theta = +/-pi branch cut.

    Used to test circumferential-seam handling: the lesion straddles the angle
    wraparound, so a naive raster splits it into two pieces at opposite ends of
    the v axis. Deliberately NOT in ALL_PHANTOMS so the standard gates are
    unaffected.
    """
    R = 11.0
    rad = 60.0
    return _build(
        "bent_tube_seam",
        center_fn=lambda t: np.array([rad * np.cos(t), 0.0, rad * np.sin(t)]),
        t_max=np.deg2rad(150.0),
        radius_fn=lambda s: np.full_like(np.atleast_1d(s), R, float) if np.ndim(s) else R,
        spacing=SPACING, direction=DIRECTION, lesion_theta0=theta0,
    )


ALL_PHANTOMS = {
    "straight_cylinder": straight_cylinder,
    "bent_tube": bent_tube,
    "tapered_tube": tapered_tube,
    "bent_aneurysm": bent_aneurysm,
}


def build_all():
    return {name: fn() for name, fn in ALL_PHANTOMS.items()}
