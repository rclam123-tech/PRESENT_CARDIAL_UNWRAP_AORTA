"""Centerline interface and its two implementations.

Environment rule (Master Prompt): centerline extraction is a *pluggable*
interface so the phantom-validated pipeline never depends on an interactive 3D
Slicer session.

* ``AnalyticCenterline`` -- built from a parametric curve, no external deps. Used
  by the phantoms, where we know the exact geometry.
* ``PolylineFileCenterline`` -- loads a polyline exported once from Slicer/VMTK
  as CSV or legacy VTK PolyData.
* ``SegmentationCenterline`` -- extracts a centerline directly from a binary
  aorta segmentation via a physical-space distance transform + min-cost path
  (no Slicer). Used for real patient data (Phase 6).

Both produce the same object: ordered physical (RAS) vertices, cumulative
arclength ``s``, a cached rotation-minimizing frame, and a KD-tree for
nearest-vertex queries. Everything downstream consumes only this interface.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import frame as _frame


class Centerline:
    """Ordered physical-space polyline with arclength + RMF.

    Parameters
    ----------
    points : (N, 3) array
        Vertices in physical RAS coordinates, ordered proximal -> distal.
    """

    def __init__(self, points):
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("centerline points must be (N, 3)")
        # Drop consecutive duplicates which would break tangents/arclength.
        keep = np.ones(len(pts), bool)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
        self.points = pts[keep]
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.T, self.N1, self.N2 = _frame.rotation_minimizing_frame(self.points)
        self._tree = cKDTree(self.points)
        self._compute_curvature()

    def _compute_curvature(self):
        """Centerline curvature magnitude kappa(s) and the angle (in the RMF) of
        the curvature vector -- i.e. the direction of the INNER wall of the bend.

        Computed purely from the polyline (works on real data, not just
        phantoms). Used by the area-preserving (curvature-corrected) unwrap: the
        tube surface area element is r*(1 - kappa*r*cos(theta - theta_inner)).
        """
        n = len(self.points)
        if n < 3:
            self.kappa = np.zeros(n)
            self.theta_inner = np.zeros(n)
            return
        d1 = np.gradient(self.points, self.s, axis=0)
        d2 = np.gradient(d1, self.s, axis=0)
        # Keep only the component of d2 orthogonal to the tangent (the curvature
        # vector); the tangential part is parametrization noise.
        tdot = np.sum(d2 * self.T, axis=1, keepdims=True)
        kvec = d2 - tdot * self.T
        self.kappa = np.linalg.norm(kvec, axis=1)
        a = np.sum(kvec * self.N1, axis=1)
        b = np.sum(kvec * self.N2, axis=1)
        self.theta_inner = np.arctan2(b, a)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def nearest(self, pts):
        """Nearest vertex index for each physical point (..., 3)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        _, idx = self._tree.query(pts)
        return idx

    def wall_radius(self, seg, seg_affine_ras):
        """Per-vertex wall radius (mm), ray-cast from the segmentation once.

        Cached on the instance: the wrap-aware ``assign`` needs a per-vertex
        "how wide is the wall here", and that should not depend on the caller
        re-deriving it. Assumes one segmentation per centerline (the aorta seg
        the centerline was built from); pass the same ``seg``/affine each call.
        """
        cached = getattr(self, "_wall_radius", None)
        if cached is None:
            from .wall import estimate_wall_radius
            cached = estimate_wall_radius(seg, seg_affine_ras, self)
            self._wall_radius = cached
        return cached

    def assign(self, pts, wall_radius, factor=1.15, k=5):
        """Wrap-aware nearest-vertex assignment, bounded by the wall radius.

        Plain nearest-vertex (``nearest``) is ambiguous where the vessel doubles
        back on itself (the arch): for a point on the descending limb a vertex on
        the *ascending* limb can be almost as close, and because the two limbs run
        roughly ANTIPARALLEL there, the offset to that wrong vertex is almost
        entirely RADIAL -- the along-tangent distance ``|(p-C).T|`` is small for
        the wrong vertex too, so it cannot separate the limbs. The radial offset
        can: the correct limb is the one whose wall actually reaches the point.

        For each point, among its ``k`` nearest vertices, keep only candidates
        whose radial offset ``r = hypot((p-C).N1, (p-C).N2)`` is within
        ``factor * wall_radius[vertex]``, then pick the survivor minimizing
        ``r / wall_radius[vertex]`` (tie-broken on ``|along|``). If no candidate's
        wall reaches the point it is out-of-aorta -> vertex ``-1``.

        Returns ``(vertex_index, r, along)`` per point so callers need not
        recompute the projection. Where nothing survives, ``vertex_index`` is
        ``-1`` and ``r``/``along`` are those of the nearest candidate (diagnostic
        only -- the point is excluded downstream). Does NOT modify ``nearest``.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        wall_radius = np.asarray(wall_radius, float)
        n_pts = len(pts)
        kq = int(min(k, len(self.points)))
        _, cand = self._tree.query(pts, k=kq)
        cand = cand.reshape(n_pts, kq)                       # (n_pts, kq) vertex ids

        # Project each point onto each candidate vertex's frame.
        C = self.points[cand]                                # (n_pts, kq, 3)
        d = pts[:, None, :] - C
        along = np.sum(d * self.T[cand], axis=2)             # (n_pts, kq)
        a = np.sum(d * self.N1[cand], axis=2)
        b = np.sum(d * self.N2[cand], axis=2)
        r = np.hypot(a, b)                                   # (n_pts, kq)

        wr = wall_radius[cand]                               # (n_pts, kq)
        within = r <= factor * wr                            # survivors
        # Minimize r / wall_radius; tie-break toward smaller |along|. Rejected
        # candidates get +inf so they are never selected.
        ratio = np.where(within, r / wr, np.inf)
        key = ratio + 1e-9 * np.abs(along)                   # |along| tiebreak
        best = np.argmin(key, axis=1)                        # (n_pts,)
        rows = np.arange(n_pts)

        vtx = cand[rows, best]
        r_out = r[rows, best]
        along_out = along[rows, best]
        vtx = np.where(within[rows, best], vtx, -1)

        # Warn (never raise) if sub-vertex refinement is extrapolating: the
        # along-tangent shift should stay within the local vertex spacing.
        n_v = len(self.points)
        spacing = np.full(n_v, np.inf)
        seg_len = np.diff(self.s)
        if len(seg_len):
            spacing[:-1] = seg_len
            spacing[-1] = seg_len[-1]
            if n_v > 2:
                spacing[1:-1] = np.maximum(seg_len[:-1], seg_len[1:])
        valid = vtx >= 0
        extrap = valid & (np.abs(along_out) > spacing[np.where(valid, vtx, 0)])
        if np.any(extrap):
            warnings.warn(
                f"assign: |along| exceeds local vertex spacing for "
                f"{int(extrap.sum())} point(s); sub-vertex refinement is "
                f"extrapolating (centerline may be too coarse here).",
                stacklevel=2,
            )
        return vtx, r_out, along_out

    def point_at(self, s) -> np.ndarray:
        """Linearly interpolate a physical point at arclength ``s``."""
        s = np.atleast_1d(np.asarray(s, float))
        out = np.empty((len(s), 3))
        for d in range(3):
            out[:, d] = np.interp(s, self.s, self.points[:, d])
        return out

    def frame_at(self, s):
        """Interpolate (T, N1, N2) at arclength ``s`` and re-orthonormalize."""
        s = np.atleast_1d(np.asarray(s, float))

        def _interp(arr):
            out = np.empty((len(s), 3))
            for d in range(3):
                out[:, d] = np.interp(s, self.s, arr[:, d])
            return out

        T = _interp(self.T)
        N1 = _interp(self.N1)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N1 = N1 - np.sum(N1 * T, axis=1, keepdims=True) * T
        N1 /= np.linalg.norm(N1, axis=1, keepdims=True)
        N2 = np.cross(T, N1)
        return T, N1, N2


class AnalyticCenterline(Centerline):
    """Centerline sampled from a parametric function ``f(t) -> (3,)``."""

    @classmethod
    def from_function(cls, f, t_max: float, n: int = 400, t_min: float = 0.0):
        ts = np.linspace(t_min, t_max, n)
        pts = np.array([np.asarray(f(t), float) for t in ts])
        return cls(pts)


class PolylineFileCenterline(Centerline):
    """Centerline loaded from a file exported by Slicer / VMTK.

    Supported formats:

    * ``.csv`` -- columns x,y,z (header optional), one vertex per row.
    * ``.vtk`` -- legacy ASCII PolyData with POINTS (+ optional LINES ordering).

    Points are assumed to already be in the canonical world frame. If the export
    is LPS (the Slicer/ITK default), pass ``system='LPS'`` to convert to RAS.
    """

    @classmethod
    def from_file(cls, path, system: str = "RAS"):
        from .geometry import to_ras

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            pts = cls._read_csv(path)
        elif suffix == ".vtk":
            pts = cls._read_vtk(path)
        else:
            raise ValueError(f"unsupported centerline format: {suffix}")
        return cls(to_ras(pts, system))

    @staticmethod
    def _read_csv(path: Path) -> np.ndarray:
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            try:
                xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            except (ValueError, IndexError):
                continue  # header or non-numeric row
            rows.append(xyz)
        if not rows:
            raise ValueError(f"no numeric x,y,z rows found in {path}")
        return np.array(rows, float)

    @staticmethod
    def _read_vtk(path: Path) -> np.ndarray:
        tokens = path.read_text().split()
        try:
            i = tokens.index("POINTS")
        except ValueError as exc:
            raise ValueError(f"no POINTS section in {path}") from exc
        npts = int(tokens[i + 1])
        flat = np.array(tokens[i + 3 : i + 3 + 3 * npts], float)
        pts = flat.reshape(npts, 3)
        # Honor an explicit LINES connectivity order if present.
        if "LINES" in tokens:
            j = tokens.index("LINES")
            count = int(tokens[j + 2])  # total ints in the connectivity array
            conn = np.array(tokens[j + 3 : j + 3 + count], int)
            order, k = [], 0
            while k < len(conn):
                m = conn[k]
                order.extend(conn[k + 1 : k + 1 + m].tolist())
                k += 1 + m
            if order:
                pts = pts[np.array(order, int)]
        return pts


class SegmentationCenterline(Centerline):
    """Centerline extracted from a binary aorta segmentation (no Slicer).

    Pipeline (Phase 6a):

    1. Crop to the segmentation bounding box.
    2. Physical-space Euclidean distance transform *inside* the mask,
       ``ndimage.distance_transform_edt(mask, sampling=voxel_spacing)`` so the
       0.793/0.793/3.0 mm anisotropy is honored -- EDT is distance-to-wall in mm.
    3. Cost field ``1/(EDT+eps)`` (cheap along the centered ridge, expensive near
       the wall), impassable outside the mask.
    4. Auto-detect the two tube ends as the geodesically-farthest points via a
       double sweep of ``MCP_Geometric`` (sampling-aware) on that cost field.
       A long main tube accumulates more centered cost than any short branch, so
       the two ends land on the main lumen, not on supra-aortic spurs.
    5. Min-cost path between them with ``skimage.graph.route_through_array``.
    6. Spline-smooth in physical space and resample uniformly.

    Because it returns a single simple path, the result is a clean
    2-endpoint / 0-branch polyline by construction (unlike skeletonization).

    Naive skeletonization was rejected: on a test patient it produced 36 endpoints /
    60 branch points and kappa*r up to 6.
    """

    @classmethod
    def from_mask(cls, mask, affine, system: str = "RAS", n_resample: int = 300,
                  smooth=None, eps: float = 1e-3, pad: int = 2,
                  return_debug: bool = False):
        from scipy import ndimage
        from scipy.interpolate import splev, splprep
        from skimage.graph import MCP_Geometric, route_through_array

        from .geometry import to_ras, voxel_to_physical, voxel_spacing

        mask = np.asarray(mask).astype(bool)
        if not mask.any():
            raise ValueError("segmentation mask is empty")
        spacing = voxel_spacing(affine)

        # -- 1. crop to bounding box (keeps EDT/MCP cheap on big volumes) -------
        occ = np.argwhere(mask)
        lo = np.maximum(occ.min(0) - pad, 0)
        hi = np.minimum(occ.max(0) + pad + 1, mask.shape)
        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        # -- 2. physical-space EDT (anisotropy honored via sampling) -----------
        # Pad with a background border so a mask touching the crop/image boundary
        # (e.g. the aorta cut at the scan FOV) still gets a correct
        # distance-to-cut-face, rather than a spuriously large EDT at the edge.
        sub_p = np.pad(sub, 1, constant_values=False)
        edt = ndimage.distance_transform_edt(sub_p, sampling=spacing)[1:-1, 1:-1, 1:-1]

        # -- 3. cost field: cheap on the centered ridge, barrier outside -------
        cost = np.where(sub, 1.0 / (edt + eps), np.inf)

        # EDT ridge = local maxima of the distance map (the medial axis). Tube
        # endpoints are restricted to the ridge so a double sweep lands on the
        # centered ends, not on a low-EDT rim/corner of a flat cut face.
        ridge = (edt >= ndimage.maximum_filter(edt, size=3) - 1e-9) & sub

        # -- 4. two geodesically-farthest EDT-ridge ends (double sweep) --------
        def farthest(src):
            mcp = MCP_Geometric(cost, sampling=tuple(spacing))
            cum, _ = mcp.find_costs([tuple(int(x) for x in src)])
            reachable = np.isfinite(cum) & ridge
            cum2 = np.where(reachable, cum, -np.inf)
            return np.unravel_index(np.argmax(cum2), cum2.shape)

        seed = np.unravel_index(np.argmax(edt), edt.shape)  # deepest -> on ridge
        end_a = farthest(seed)
        end_b = farthest(end_a)

        # -- 5. min-cost path between the ends ----------------------------------
        indices, _ = route_through_array(cost, end_a, end_b,
                                         fully_connected=True, geometric=True)
        path_sub = np.array(indices, float)             # (M,3) cropped index
        path_full = path_sub + lo                        # full-grid index
        pts = to_ras(voxel_to_physical(path_full, affine), system)

        # -- 6. de-staircase, spline-smooth, resample uniformly ----------------
        keep = np.ones(len(pts), bool)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
        pts = pts[keep]

        # Anisotropy-aware pre-smooth: low-pass the raw path at the COARSEST
        # voxel scale. The min-cost path steps by whole voxels, so on 3 mm slices
        # it staircases with period ~ the slice thickness; that staircase, not
        # anatomy, is what spikes curvature (kappa*r up to ~4.5) through the arch.
        # A Gaussian of sigma ~ one coarse voxel removes it while preserving the
        # low-frequency arch. For near-isotropic phantoms sigma ~ 1 point, i.e.
        # a no-op (their position error is unchanged).
        step = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        sigma = float(np.max(spacing)) / step if step > 0 else 0.0
        if sigma > 0.3 and len(pts) > 6:
            pts_s = np.stack(
                [ndimage.gaussian_filter1d(pts[:, d], sigma, mode="nearest")
                 for d in range(3)], axis=1)
        else:
            pts_s = pts

        if smooth is None:
            # Light spline (the Gaussian did the de-staircasing): allow ~half a
            # fine voxel of deviation, just to interpolate/resample smoothly.
            smooth = len(pts_s) * (0.5 * float(np.min(spacing))) ** 2
        k = 3 if len(pts_s) > 3 else max(1, len(pts_s) - 1)
        tck, _ = splprep([pts_s[:, 0], pts_s[:, 1], pts_s[:, 2]], s=smooth, k=k)
        uu = np.linspace(0.0, 1.0, n_resample)
        smoothed = np.array(splev(uu, tck)).T

        obj = cls(smoothed)
        ends_full = np.array([end_a, end_b], float) + lo
        obj.endpoints_ras = to_ras(voxel_to_physical(ends_full, affine), system)
        obj.raw_path_ras = pts
        obj.edt_max = float(edt.max())
        if return_debug:
            obj.debug = {"edt": edt, "bbox_lo": lo, "bbox_hi": hi,
                         "end_a": end_a, "end_b": end_b, "sub": sub}
        return obj
