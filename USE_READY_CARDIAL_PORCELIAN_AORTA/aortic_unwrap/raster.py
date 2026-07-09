"""Phase 4 -- 2D raster of the scored mask + stored inverse map.

The deliverable is a flattened 2D image of the already-scored calcium, plus an
inverse map so any map pixel can be traced back to a 3D location. The inverse
map is nearly free: we keep the (cl_index, s, theta, r) the unwrap already
computed for the representative point in each pixel.

NO interactive click-back UI is built here (that is a deferred feature). We only
finalize the raster and persist the inverse map as a compact ``.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnwrapRaster:
    image: np.ndarray        # (H, W) bool: scored-calcium occupancy
    u_min: float
    v_min: float
    pixel: float             # pixel edge length in mm (square)
    # Per-pixel inverse map (only meaningful where image is True):
    cl_index: np.ndarray     # (H, W) int, -1 where empty
    s_map: np.ndarray        # (H, W) float
    theta_map: np.ndarray    # (H, W) float
    r_map: np.ndarray        # (H, W) float

    def pixel_centers(self):
        H, W = self.image.shape
        u = self.u_min + (np.arange(W) + 0.5) * self.pixel
        v = self.v_min + (np.arange(H) + 0.5) * self.pixel
        return u, v

    def save(self, path):
        np.savez_compressed(
            path,
            image=self.image, u_min=self.u_min, v_min=self.v_min,
            pixel=self.pixel, cl_index=self.cl_index, s_map=self.s_map,
            theta_map=self.theta_map, r_map=self.r_map,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(
            image=z["image"], u_min=float(z["u_min"]), v_min=float(z["v_min"]),
            pixel=float(z["pixel"]), cl_index=z["cl_index"], s_map=z["s_map"],
            theta_map=z["theta_map"], r_map=z["r_map"],
        )


def rasterize(unwrap_result, pixel: float = 0.7, pad: float = 2.0) -> UnwrapRaster:
    """Rasterize an UnwrapResult into a 2D occupancy image + inverse map."""
    res = unwrap_result
    u, v = res.u, res.v
    u_min = float(u.min()) - pad
    v_min = float(v.min()) - pad
    W = int(np.ceil((u.max() + pad - u_min) / pixel)) + 1
    H = int(np.ceil((v.max() + pad - v_min) / pixel)) + 1

    image = np.zeros((H, W), bool)
    cl_index = np.full((H, W), -1, int)
    s_map = np.zeros((H, W))
    theta_map = np.zeros((H, W))
    r_map = np.zeros((H, W))

    col = np.clip(((u - u_min) / pixel).astype(int), 0, W - 1)
    row = np.clip(((v - v_min) / pixel).astype(int), 0, H - 1)
    for n in range(len(u)):
        rr, cc = row[n], col[n]
        image[rr, cc] = True
        # Last writer wins is fine; representative point per pixel.
        cl_index[rr, cc] = res.cl_index[n]
        s_map[rr, cc] = res.s[n]
        theta_map[rr, cc] = res.theta[n]
        r_map[rr, cc] = res.r[n]

    return UnwrapRaster(image=image, u_min=u_min, v_min=v_min, pixel=pixel,
                        cl_index=cl_index, s_map=s_map, theta_map=theta_map,
                        r_map=r_map)


def back_project(raster: UnwrapRaster, unwrap) -> np.ndarray:
    """3D physical points for every occupied pixel, via the stored inverse map."""
    rows, cols = np.nonzero(raster.image)
    s = raster.s_map[rows, cols]
    theta = raster.theta_map[rows, cols]
    r = raster.r_map[rows, cols]
    return unwrap.inverse(s, theta, r)


def connected_components(raster, periodic=True):
    """Label connected calcium footprints on the 2D map.

    The circumferential coordinate theta is periodic: theta = +pi and theta = -pi
    are the SAME meridian, but rasterizing v = theta*r puts them at opposite ends
    of the v axis, so a lesion straddling the seam splits into two pieces. With
    ``periodic=True`` those across-seam pieces are merged (using the stored
    per-pixel theta + r), so a seam-straddling lesion is one footprint with its
    full area, not two.

    Returns ``(n_components, labels)`` where ``labels`` is the raster-shaped
    integer label image (0 = background).
    """
    from scipy.ndimage import label

    lbl, n = label(raster.image, structure=np.ones((3, 3), int))
    if not periodic or n <= 1:
        return n, lbl

    rows, cols = np.nonzero(raster.image)
    th = raster.theta_map[rows, cols]
    rr = raster.r_map[rows, cols]
    # Angular size of one pixel ~ pixel / r; pixels within ~3 of +/-pi are seam.
    dth_pix = raster.pixel / np.maximum(rr, 1e-6)
    near_pos = th > (np.pi - 3 * dth_pix)
    near_neg = th < (-np.pi + 3 * dth_pix)

    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pos_i = np.where(near_pos)[0]
    neg_i = np.where(near_neg)[0]
    for pi in pos_i:
        for ni in neg_i:
            # Same arclength column (within one pixel) -> same meridian seam.
            if abs(int(cols[pi]) - int(cols[ni])) <= 1:
                union(int(lbl[rows[pi], cols[pi]]), int(lbl[rows[ni], cols[ni]]))

    roots = {find(int(lbl[r_, c_])) for r_, c_ in zip(rows, cols)}
    # Relabel to a contiguous, root-merged label image.
    remap = {root: i + 1 for i, root in enumerate(sorted(roots))}
    merged = np.zeros_like(lbl)
    for r_, c_ in zip(rows, cols):
        merged[r_, c_] = remap[find(int(lbl[r_, c_]))]
    return len(roots), merged
