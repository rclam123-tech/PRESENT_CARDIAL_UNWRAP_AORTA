"""Phase 2 -- arclength + rotation-minimizing frame."""

import numpy as np

from aortic_unwrap.frame import accumulated_twist, rotation_minimizing_frame


def test_straight_has_no_twist(built_phantoms):
    cl = built_phantoms["straight_cylinder"].centerline
    assert abs(accumulated_twist(cl.T, cl.N1)) < 1e-6


def test_frame_continuous_no_flips_no_nans(built_phantoms):
    for name, ph in built_phantoms.items():
        cl = ph.centerline
        stacked = np.c_[cl.T, cl.N1, cl.N2]
        assert np.all(np.isfinite(stacked)), name
        dots = np.sum(cl.N1[:-1] * cl.N1[1:], axis=1)
        assert dots.min() > 0.5, name  # no sudden sign flip between vertices


def test_arclength_matches_analytic(built_phantoms):
    analytic = {
        "straight_cylinder": 120.0,
        "bent_tube": 60.0 * np.deg2rad(150.0),
        "tapered_tube": 120.0,
        "bent_aneurysm": 55.0 * np.deg2rad(160.0),
    }
    for name, ph in built_phantoms.items():
        assert abs(ph.centerline.length - analytic[name]) / analytic[name] < 0.005


def test_frame_orthonormal():
    # A helix exercises a frame that genuinely rotates in 3D.
    t = np.linspace(0, 4 * np.pi, 300)
    pts = np.c_[np.cos(t), np.sin(t), 0.3 * t]
    T, N1, N2 = rotation_minimizing_frame(pts)
    assert np.allclose(np.sum(T * N1, axis=1), 0, atol=1e-9)
    assert np.allclose(np.sum(T * N2, axis=1), 0, atol=1e-9)
    assert np.allclose(np.sum(N1 * N2, axis=1), 0, atol=1e-9)
    assert np.allclose(np.linalg.norm(N1, axis=1), 1, atol=1e-9)
