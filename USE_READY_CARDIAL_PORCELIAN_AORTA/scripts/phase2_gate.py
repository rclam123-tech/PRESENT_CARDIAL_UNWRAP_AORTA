"""Phase 2 GATE -- centerline arclength + rotation-minimizing frame.

Checks:
  * straight phantom: accumulated frame twist ~ 0 (frame does not spin),
  * bent phantom: frame continuous, no flips, no NaNs at low curvature,
  * computed arclength matches the analytic phantom length to tolerance.

Also reports how a Frenet normal would fare (undefined where curvature -> 0),
to justify invariant #3's rotation-minimizing requirement.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402
from aortic_unwrap.frame import accumulated_twist  # noqa: E402


def frenet_undefined_fraction(points):
    """Fraction of vertices where the Frenet normal is ill-defined (kappa~0)."""
    d1 = np.gradient(points, axis=0)
    d2 = np.gradient(d1, axis=0)
    # curvature vector magnitude ~ |d2 - (d2.t)t|
    t = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    proj = np.sum(d2 * t, axis=1, keepdims=True) * t
    kappa = np.linalg.norm(d2 - proj, axis=1)
    return float(np.mean(kappa < 1e-6 * np.max(kappa + 1e-12)))


def analytic_length(name, ph):
    return {
        "straight_cylinder": 120.0,
        "bent_tube": 60.0 * np.deg2rad(150.0),
        "tapered_tube": 120.0,
        "bent_aneurysm": 55.0 * np.deg2rad(160.0),
    }[name]


def main():
    ok = True
    print(f"{'phantom':<18} {'twist(rad)':>11} {'min N1.N1':>10} {'NaNs':>5} "
          f"{'len':>8} {'analytic':>9} {'err%':>7}")
    print("-" * 80)
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        cl = ph.centerline
        twist = accumulated_twist(cl.T, cl.N1)
        dots = np.sum(cl.N1[:-1] * cl.N1[1:], axis=1)
        min_dot = float(np.min(dots))
        nans = int(np.count_nonzero(~np.isfinite(np.c_[cl.T, cl.N1, cl.N2])))
        L = cl.length
        La = analytic_length(name, ph)
        err = abs(L - La) / La * 100

        checks = [min_dot > 0.5, nans == 0, err < 0.5]
        if name == "straight_cylinder":
            checks.append(abs(twist) < 1e-6)
        ok &= all(checks)
        print(f"{name:<18} {twist:>11.2e} {min_dot:>10.4f} {nans:>5d} "
              f"{L:>8.2f} {La:>9.2f} {err:>7.3f}"
              + ("" if all(checks) else "  <-- FAIL"))
    print("-" * 80)
    # Illustrate why not Frenet.
    straight = phantoms.straight_cylinder()
    frac = frenet_undefined_fraction(straight.centerline.points)
    print(f"(Frenet normal undefined on {frac*100:.0f}% of straight-cylinder "
          f"vertices -> would flip; RMF stays defined everywhere.)")
    print("PHASE 2 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
