"""Rotation-minimizing frame (RMF) along a centerline.

Hard invariant #3 (Master Prompt): the circumferential coordinate uses a
rotation-minimizing / parallel-transport (Bishop) frame, NOT Frenet-Serret.
Frenet's normal is defined from the curvature vector, so it spins by the torsion
integral and -- worse -- becomes undefined and flips sign wherever curvature
passes through zero (straight segments, inflection points). An aorta is mostly
near-straight with one high-torsion arch, i.e. exactly the regime where Frenet
misbehaves. The RMF carries an initial normal along the curve with the least
possible rotation about the tangent, so it stays continuous everywhere.

Implementation: the *double reflection* method of Wang, Juttler, Zheng & Liu,
"Computation of Rotation Minimizing Frames", ACM TOG 2008. It is exact to second
order, needs no curvature/torsion, and never divides by curvature, so it is
stable through straight runs and the arch alike.
"""

from __future__ import annotations

import numpy as np


def tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents at each centerline vertex via central differences."""
    points = np.asarray(points, float)
    t = np.empty_like(points)
    t[1:-1] = points[2:] - points[:-2]
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return t / n


def _initial_normal(t0: np.ndarray) -> np.ndarray:
    """Any unit vector orthogonal to the first tangent (stable choice)."""
    # Cross the tangent with whichever world axis it is least aligned to, so the
    # cross product is well conditioned regardless of tangent direction.
    axis = np.eye(3)[np.argmin(np.abs(t0))]
    r = np.cross(t0, axis)
    return r / np.linalg.norm(r)


def rotation_minimizing_frame(points: np.ndarray, r0: np.ndarray | None = None):
    """Return (T, N1, N2): unit tangent and two RMF normals at each vertex.

    ``r0`` optionally seeds the first normal (must be ~orthogonal to the first
    tangent); otherwise a stable default is chosen.
    """
    points = np.asarray(points, float)
    n = len(points)
    T = tangents(points)

    N1 = np.empty_like(points)
    if r0 is None:
        N1[0] = _initial_normal(T[0])
    else:
        r0 = np.asarray(r0, float)
        # Re-orthogonalize against the tangent so a sloppy seed is still valid.
        r0 = r0 - (r0 @ T[0]) * T[0]
        N1[0] = r0 / np.linalg.norm(r0)

    # Double-reflection transport of N1 from vertex i to i+1.
    for i in range(n - 1):
        v1 = points[i + 1] - points[i]
        c1 = v1 @ v1
        if c1 == 0.0:  # duplicate vertex -> carry frame unchanged
            N1[i + 1] = N1[i]
            continue
        r_l = N1[i] - (2.0 / c1) * (v1 @ N1[i]) * v1
        t_l = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - t_l
        c2 = v2 @ v2
        if c2 == 0.0:  # tangent unchanged -> reflected normal already correct
            r_next = r_l
        else:
            r_next = r_l - (2.0 / c2) * (v2 @ r_l) * v2
        # Project back onto the plane orthogonal to the true tangent and renorm,
        # killing accumulated round-off.
        r_next = r_next - (r_next @ T[i + 1]) * T[i + 1]
        N1[i + 1] = r_next / np.linalg.norm(r_next)

    N2 = np.cross(T, N1)
    N2 /= np.linalg.norm(N2, axis=1, keepdims=True)
    return T, N1, N2


def accumulated_twist(T: np.ndarray, N1: np.ndarray) -> float:
    """Total rotation (radians) of N1 about the tangent across the curve.

    For a true RMF this is ~0 by construction; on a straight curve the frame is
    constant so this is 0 to machine precision. Used by the Phase 2 gate.
    """
    total = 0.0
    for i in range(len(T) - 1):
        # Component of N1 rotation that is *about* the shared tangent direction.
        a = N1[i]
        b = N1[i + 1]
        # Signed angle of b relative to a, measured in the plane orthogonal to T.
        t = T[i + 1]
        a_perp = a - (a @ t) * t
        b_perp = b - (b @ t) * t
        na, nb = np.linalg.norm(a_perp), np.linalg.norm(b_perp)
        if na == 0 or nb == 0:
            continue
        a_perp /= na
        b_perp /= nb
        cross = np.cross(a_perp, b_perp)
        sin = cross @ t
        cos = np.clip(a_perp @ b_perp, -1.0, 1.0)
        total += np.arctan2(sin, cos)
    return float(total)
