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

Optional aorta boundary gate (Phase 7)
--------------------------------------
When an ``aorta_mask`` (+ its own RAS affine) is supplied, vertex assignment uses
the wrap-aware ``Centerline.assign`` instead of plain ``nearest`` (so the arch's
antiparallel-limb ambiguity is resolved by the radial offset), and each point is
tagged ``in_aorta``. A point is in-scope iff it is BOTH inside the segmentation
(gate a) AND within ``boundary_factor * wall_radius`` of a centerline vertex
(gate b). With no mask the behaviour is byte-identical to before (``nearest``,
``in_aorta=None``) -- so the Phase 3 gate/tests are unchanged.
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
    cl_index: np.ndarray   # chosen centerline vertex index
    # Boundary gate (None when no aorta_mask was supplied -> ungated):
    in_aorta: np.ndarray | None = None   # gate(a) AND gate(b): in-scope point
    in_seg: np.ndarray | None = None     # gate(a) alone: inside the segmentation


class CenterlineProjectionUnwrap:
    def __init__(self, centerline, aorta_mask=None, aorta_affine_ras=None,
                 boundary_factor: float = 1.15):
        """``boundary_factor`` is the per-point inclusion tolerance for gate (b):
        a point is kept iff its radial offset ``r <= boundary_factor *
        wall_radius[vertex]``. This is NOT repo A's ``radial_extent`` (a MIP
        sampling SPAN); it is an inclusion tolerance on an individual point. The
        default 1.15 admits ~1-2 voxels of segmentation-boundary uncertainty
        (~0.8-1.6 mm at 0.793 mm in-plane spacing) beyond the ray-cast wall
        radius, so a plaque bulging at the adventitial edge is not clipped by a
        tightly-drawn mask, while calcium a full radius outside the wall
        (coronary / mitral-annular / vertebral) is still rejected.
        """
        self.cl = centerline
        self.aorta_mask = (None if aorta_mask is None
                           else np.asarray(aorta_mask).astype(bool))
        self.aorta_affine_ras = (None if aorta_affine_ras is None
                                 else np.asarray(aorta_affine_ras, float))
        self.boundary_factor = float(boundary_factor)

    def _inside_seg(self, pts):
        """Gate (a): points inside the aorta segmentation, seg-affine NN."""
        from .geometry import physical_to_voxel
        idx = np.rint(physical_to_voxel(pts, self.aorta_affine_ras)).astype(int)
        shape = np.array(self.aorta_mask.shape)
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        inside = np.zeros(len(pts), bool)
        g = idx[ok]
        inside[ok] = self.aorta_mask[g[:, 0], g[:, 1], g[:, 2]]
        return inside

    def __call__(self, points) -> UnwrapResult:
        cl = self.cl
        pts = np.atleast_2d(np.asarray(points, float))

        if self.aorta_mask is not None:
            wall_radius = cl.wall_radius(self.aorta_mask, self.aorta_affine_ras)
            k, _, _ = cl.assign(pts, wall_radius, factor=self.boundary_factor)
            in_wall = k >= 0                     # gate (b): assign found a vertex
            # Rejected points keep their nearest vertex only for coordinate
            # bookkeeping; they are flagged out-of-aorta via in_aorta.
            k = np.where(in_wall, k, cl.nearest(pts))
            in_seg = self._inside_seg(pts)       # gate (a)
            in_aorta = in_seg & in_wall
        else:
            k = cl.nearest(pts)
            in_seg = in_aorta = None

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
        return UnwrapResult(u=s, v=theta * r, s=s, theta=theta, r=r, cl_index=k,
                            in_aorta=in_aorta, in_seg=in_seg)

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
                            cl_index=k, in_aorta=res.in_aorta, in_seg=res.in_seg)
