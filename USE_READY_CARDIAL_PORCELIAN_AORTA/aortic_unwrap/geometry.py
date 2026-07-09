"""Coordinate / affine geometry.

Hard invariant #2 (Master Prompt): all geometry is done in physical/world
coordinates. The image affine (direction cosines * spacing, plus origin) is
carried through every voxel <-> point conversion. No bare index arithmetic and
no silent axis reordering happens anywhere else in the package -- it all funnels
through here.

Affine convention
-----------------
A 4x4 matrix ``A`` maps a *voxel index* (i, j, k) to a *physical world point*
(x, y, z)::

    [x y z 1]^T = A @ [i j k 1]^T

with ``A[:3, :3] = direction @ diag(spacing)`` and ``A[:3, 3] = origin``.
This is the standard NIfTI sform / ITK convention.

Coordinate systems
------------------
NIfTI stores world points in RAS+; NRRD / DICOM / ITK store them in LPS+. The
two differ by flipping the sign of the x and y axes. Mixing a mask saved in one
convention with a CT saved in the other is the classic silent corruptor this
package guards against, so every world point is tagged with its system and
converted to a single canonical frame (RAS) before anything downstream uses it.
"""

from __future__ import annotations

import numpy as np

# Diagonal sign-flip that converts a point/affine between LPS and RAS. It is its
# own inverse, so the same matrix maps both ways.
_LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def make_affine(spacing, direction, origin) -> np.ndarray:
    """Assemble a 4x4 index->world affine from spacing, direction, origin."""
    spacing = np.asarray(spacing, float)
    direction = np.asarray(direction, float).reshape(3, 3)
    origin = np.asarray(origin, float)
    A = np.eye(4)
    A[:3, :3] = direction @ np.diag(spacing)
    A[:3, 3] = origin
    return A


def voxel_to_physical(idx, affine) -> np.ndarray:
    """Voxel index (..., 3) -> physical world point (..., 3)."""
    idx = np.atleast_2d(np.asarray(idx, float))
    h = np.concatenate([idx, np.ones((len(idx), 1))], axis=1)
    return (affine @ h.T).T[:, :3]


def physical_to_voxel(pts, affine) -> np.ndarray:
    """Physical world point (..., 3) -> (continuous) voxel index (..., 3)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    inv = np.linalg.inv(affine)
    return (inv @ h.T).T[:, :3]


def to_ras(pts, system: str) -> np.ndarray:
    """Convert world points from ``system`` ('RAS' or 'LPS') to canonical RAS."""
    system = system.upper()
    pts = np.atleast_2d(np.asarray(pts, float))
    if system == "RAS":
        return pts
    if system == "LPS":
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return (_LPS_RAS @ h.T).T[:, :3]
    raise ValueError(f"unknown coordinate system {system!r} (expected RAS or LPS)")


def affine_to_ras(affine, system: str) -> np.ndarray:
    """Re-express an index->world affine so that its output is RAS."""
    system = system.upper()
    if system == "RAS":
        return np.asarray(affine, float)
    if system == "LPS":
        return _LPS_RAS @ np.asarray(affine, float)
    raise ValueError(f"unknown coordinate system {system!r} (expected RAS or LPS)")


def voxel_spacing(affine) -> np.ndarray:
    """Physical edge length of one voxel along each index axis."""
    return np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)
