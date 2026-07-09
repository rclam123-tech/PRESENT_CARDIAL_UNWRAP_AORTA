"""Phase 3 -- Architecture A: centerline-projection unwrap.

The cheapest unwrap that might pass (build philosophy rule #2). For each calcium
physical point:

  1. assign the nearest centerline vertex (KD-tree),
  2. refine arclength s by projecting onto the local tangent (sub-vertex),
  3. measure the radial offset in the rotation-minimizing (N1, N2) plane,
  4. emit (u, v) = (s, theta * r).

No surface mesh, no parameterization, no branch clipping. Most of an aorta is
near-cylindrical and this handles it exactly; the open question the Phase 3 gate
answers with data is whether it survives the arch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnwrapResult:
    u: np.ndarray          # longitudinal coordinate = arclength s (mm)
    v: np.ndarray          # circumferential coordinate = theta * r (mm)
    s: np.ndarray          # arclength (mm)
    theta: np.ndarray      # circumferential angle (rad), in the RMF
    r: np.ndarray          # radial distance from centerline (mm)
    cl_index: np.ndarray   # nearest centerline vertex index


class CenterlineProjectionUnwrap:
    def __init__(self, centerline):
        self.cl = centerline

    def __call__(self, points) -> UnwrapResult:
        cl = self.cl
        pts = np.atleast_2d(np.asarray(points, float))
        k = cl.nearest(pts)
        base = cl.points[k]
        T = cl.T[k]
        N1 = cl.N1[k]
        N2 = cl.N2[k]
        d = pts - base
        along = np.sum(d * T, axis=1)          # sub-vertex arclength refinement
        a = np.sum(d * N1, axis=1)
        b = np.sum(d * N2, axis=1)
        r = np.hypot(a, b)
        theta = np.arctan2(b, a)
        s = cl.s[k] + along
        return UnwrapResult(u=s, v=theta * r, s=s, theta=theta, r=r, cl_index=k)

    def inverse(self, s, theta, r) -> np.ndarray:
        """Map unwrap coordinates back to physical 3D points.

        Uses the interpolated frame at arclength ``s`` so the inverse is
        continuous between centerline vertices. The inverse consumes the stored
        (s, theta, r) directly, so it is identical for the raw and the
        curvature-corrected unwrap (only the 2D layout coordinate v differs).
        """
        s = np.atleast_1d(np.asarray(s, float))
        theta = np.atleast_1d(np.asarray(theta, float))
        r = np.atleast_1d(np.asarray(r, float))
        C = self.cl.point_at(s)
        _, N1, N2 = self.cl.frame_at(s)
        ct = np.cos(theta).reshape(-1, 1)
        st = np.sin(theta).reshape(-1, 1)
        return C + r.reshape(-1, 1) * (ct * N1 + st * N2)


class CurvatureCorrectedUnwrap(CenterlineProjectionUnwrap):
    """Architecture A with an area-preserving circumferential coordinate.

    Raw A sets v = theta * r, whose area element r*ds*dtheta ignores the bend:
    the true tube surface element is r*(1 - kappa*r*cos(theta - theta_inner)),
    larger on the outer wall and smaller on the inner wall. That mismatch is the
    17%/41% distortion the Phase 3 gate measured on the bent/arch phantoms.

    The fix keeps the longitudinal coordinate u = s and the angular position
    theta EXACTLY as raw A computes them (so localization is unchanged), and only
    rescales the circumferential coordinate so its Jacobian equals the true area
    element::

        v(theta) = integral_0^theta r*(1 - kappa*r*cos(t - theta_inner)) dt
                 = r*theta - kappa*r^2*(sin(theta - theta_inner) + sin(theta_inner))

    With this v, d(u,v)/d(s,theta) has determinant r*(1 - kappa*r*cos(...)),
    i.e. it matches sqrt(g) to first order in kappa*r -> ~0% area distortion.
    The (sin(theta_inner)) term is a per-section constant that just anchors
    v(theta=0)=0 (no longitudinal shear); it does not affect area or position.
    """

    def __call__(self, points) -> UnwrapResult:
        res = super().__call__(points)
        k = res.cl_index
        kappa = self.cl.kappa[k]
        theta_inner = self.cl.theta_inner[k]
        v = (res.r * res.theta
             - kappa * res.r ** 2
             * (np.sin(res.theta - theta_inner) + np.sin(theta_inner)))
        return UnwrapResult(u=res.s, v=v, s=res.s, theta=res.theta, r=res.r,
                            cl_index=k)
