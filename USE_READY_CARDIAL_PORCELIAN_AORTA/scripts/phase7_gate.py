"""Phase 7 GATE -- the aorta boundary gate (in_aorta) on phantoms.

Validates the wrap-aware ``Centerline.assign`` + ``in_aorta`` gate layered on top
of Architecture A. Real anatomy has no (s, theta) oracle (invariant #4), so this
is phantom-only. Three checks, each able to fail on a plausible bug:

  1. DISTRACTOR: a calcium blob OUTSIDE the aorta segmentation (coronary /
     vertebral style) is excluded (``in_aorta`` False) while genuine wall calcium
     is kept -- boundary precision AND recall both 1.0. An ungated nearest-vertex
     projection would paint the distractor onto the aortic map. This is the
     load-bearing check: on real ``seg & CT>=130`` masks every calcium voxel is
     inside the segmentation by construction, so gate (a) alone is tautological;
     the weight is on excluding genuinely out-of-aorta calcium (here, gate a AND
     gate b agree it is out).

  2. ARCH wrong-limb: on an asymmetric hairpin, a point inside the THICK limb but
     nudged toward the THIN limb is Euclidean-nearest to a thin-limb vertex, so
     ``nearest`` mis-maps it to the thin limb's arclength. ``assign`` refuses it
     (vertex -1) instead of painting it onto the wrong limb, while a genuine
     thick-wall point still resolves to the thick limb.

     Honest scope: with k=5 the safe realized behaviour for a genuine
     perpendicular trap is EXCLUSION, not "resolve to the correct limb". When a
     whole stretch of the wrong limb is perpendicular-closer, the correct limb is
     beyond the k nearest, so no survivor remains and assign returns -1 (the
     algorithm's own out-of-aorta rule). Exclusion is the correct outcome: it
     never fabricates a footprint on the wrong limb.

  3. LOCALIZATION UNCHANGED: with the gate on, in-aorta phantom calcium localizes
     to within a hair of the pre-gate pinned baseline, and no genuine calcium is
     dropped (recall 1.0).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import metrics, phantoms  # noqa: E402
from aortic_unwrap.geometry import voxel_to_physical  # noqa: E402
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap  # noqa: E402

BASELINE = (Path(__file__).resolve().parents[1] / "tests"
            / "baseline_pre_phase7.json")

# Localization must not move more than a sub-vertex hair from the baseline: the
# gate reassigns some points to an adjacent vertex, absorbed almost entirely by
# the along-tangent refinement. These tolerances pass that micro-shift but catch
# any real mis-assignment (which moves localization by a vertex spacing or more).
MED_TOL_MM = 0.05
MAX_TOL_MM = 0.5


def check_distractor(boundary_factor=1.15):
    ph = phantoms.bent_tube()
    real = voxel_to_physical(ph.calcium_idx, ph.affine)
    dist, _ = phantoms.out_of_aorta_blob(ph)
    pts = np.vstack([real, dist])
    is_real = np.concatenate([np.ones(len(real), bool), np.zeros(len(dist), bool)])
    uw = CurvatureCorrectedUnwrap(ph.centerline, aorta_mask=ph.aorta_mask,
                                  aorta_affine_ras=ph.affine,
                                  boundary_factor=boundary_factor)
    pred = uw(pts).in_aorta
    tp = int((pred & is_real).sum())
    fp = int((pred & ~is_real).sum())
    fn = int((~pred & is_real).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"n_real": int(len(real)), "n_distractor": int(len(dist)),
            "precision": precision, "recall": recall}


def check_arch(boundary_factor=1.15):
    cl, wall, meta = phantoms.arch_hairpin()
    z0 = 0.5 * meta["Lz"]
    trap = np.array([[-1.5, 0.0, z0]])                 # inside thick, toward thin
    ctrl = np.array([[meta["sep"] + 10.0, 0.0, z0]])   # thick outer wall
    kn = int(cl.nearest(trap)[0])
    ka = int(cl.assign(trap, wall, factor=boundary_factor, k=5)[0][0])
    kc = int(cl.assign(ctrl, wall, factor=boundary_factor, k=5)[0][0])
    return {
        "nearest_trap_s": float(cl.s[kn]),
        "thin_s_lo": float(meta["thin_s_lo"]),
        "thick_s_hi": float(meta["thick_s_hi"]),
        "nearest_maps_to_wrong_limb": bool(cl.s[kn] > meta["thin_s_lo"]),
        "assign_excludes_trap": bool(ka < 0),
        "assign_ctrl_s": float(cl.s[kc]) if kc >= 0 else -1.0,
        "ctrl_resolves_correct": bool(kc >= 0 and cl.s[kc] < meta["thick_s_hi"]
                                      and abs(cl.s[kc] - z0) < 3.0),
    }


def check_localization():
    base = json.loads(BASELINE.read_text())["phantoms"]
    rows = {}
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        uw = CurvatureCorrectedUnwrap(ph.centerline, aorta_mask=ph.aorta_mask,
                                      aorta_affine_ras=ph.affine)
        pts = voxel_to_physical(ph.calcium_idx, ph.affine)
        in_aorta_all = bool(uw(pts).in_aorta.all())
        loc = metrics.localization_error(ph, uw)
        b = base[name]
        rows[name] = {
            "d_med_ds": abs(loc["median_ds_mm"] - b["loc_median_ds_mm"]),
            "d_med_dcirc": abs(loc["median_dcirc_mm"] - b["loc_median_dcirc_mm"]),
            "d_max_ds": abs(loc["max_ds_mm"] - b["loc_max_ds_mm"]),
            "recall_all": in_aorta_all,
        }
    return rows


def main():
    ok = True
    print("=" * 70)
    d = check_distractor()
    d_ok = d["precision"] == 1.0 and d["recall"] == 1.0
    ok &= d_ok
    print(f"[1] distractor  precision={d['precision']:.3f} recall={d['recall']:.3f} "
          f"(real={d['n_real']}, distractor={d['n_distractor']})  "
          f"{'PASS' if d_ok else 'FAIL'}")

    a = check_arch()
    a_ok = (a["nearest_maps_to_wrong_limb"] and a["assign_excludes_trap"]
            and a["ctrl_resolves_correct"])
    ok &= a_ok
    print(f"[2] arch        nearest(trap) s={a['nearest_trap_s']:.1f} -> wrong-limb="
          f"{a['nearest_maps_to_wrong_limb']}; assign excludes trap="
          f"{a['assign_excludes_trap']}; control resolves="
          f"{a['ctrl_resolves_correct']}  {'PASS' if a_ok else 'FAIL'}")

    rows = check_localization()
    l_ok = True
    for name, r in rows.items():
        good = (r["d_med_ds"] <= MED_TOL_MM and r["d_med_dcirc"] <= MED_TOL_MM
                and r["d_max_ds"] <= MAX_TOL_MM and r["recall_all"])
        l_ok &= good
        print(f"    {name:16s} d_med_ds={r['d_med_ds']:.2e} "
              f"d_med_dcirc={r['d_med_dcirc']:.2e} d_max_ds={r['d_max_ds']:.3f} "
              f"recall_all={r['recall_all']}  {'ok' if good else 'FAIL'}")
    ok &= l_ok
    print(f"[3] localization unchanged vs baseline  {'PASS' if l_ok else 'FAIL'}")
    print("=" * 70)
    print("PHASE 7 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
