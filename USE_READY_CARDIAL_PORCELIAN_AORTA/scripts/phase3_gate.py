"""Phase 3 GATE -- Architecture A unwrap + the project's decision fork.

Per the resolved fork (libigl unavailable; tool is display/localization), we use
the *curvature-corrected* Architecture A: it applies the tube area Jacobian to
the circumferential coordinate while leaving (s, theta) -- hence localization --
exactly as raw A computes them.

For each phantom we report, side by side:
  * raw A area distortion %      (the failing baseline)
  * corrected A area distortion %
  * localization error (ds, dcirc) for the corrected unwrap

Decision rule (build plan + reviewed decision):
  * straight cylinder MUST stay near-zero distortion (else it's a bug),
  * localization MUST be preserved vs raw A,
  * curvature-fixable phantoms (bend, taper) MUST be within ~5% area distortion,
  * the aneurysm-on-arch residual is an ACCEPTED, documented intrinsic
    limitation (Gaussian-curvature distortion of a centerline-frame unwrap,
    removable only by Architecture B) -- it must show a large improvement vs
    raw A but is not treated as a hard failure. See outputs/error_budget.md.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402
from aortic_unwrap.unwrap_a import (  # noqa: E402
    CenterlineProjectionUnwrap,
    CurvatureCorrectedUnwrap,
)
from aortic_unwrap import metrics  # noqa: E402

AREA_THRESHOLD_PCT = 5.0


def main():
    rows = []
    print(f"{'phantom':<18} {'raw A %':>9} {'corrected %':>12} {'factor':>8} "
          f"{'loc ds(mm)':>11} {'loc dcirc(mm)':>14}")
    print("-" * 76)
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        raw = CenterlineProjectionUnwrap(ph.centerline)
        cor = CurvatureCorrectedUnwrap(ph.centerline)
        ad_raw = metrics.area_distortion(ph, raw)
        ad_cor = metrics.area_distortion(ph, cor)
        loc = metrics.localization_error(ph, cor)
        rows.append((name, ad_raw, ad_cor, loc))
        print(f"{name:<18} {ad_raw['median_abs_pct']:>9.2f} "
              f"{ad_cor['median_abs_pct']:>12.2f} {ad_cor['median']:>8.4f} "
              f"{loc['median_ds_mm']:>11.3f} {loc['median_dcirc_mm']:>14.3f}")
    print("-" * 76)

    straight = next(r for r in rows if r[0] == "straight_cylinder")
    straight_ok = straight[2]["median_abs_pct"] < 0.5
    # Bend + taper are curvature-fixable and must clear the threshold. The
    # aneurysm is an accepted, documented intrinsic residual (handled below).
    fixable = ["bent_tube", "tapered_tube"]
    fixable_pct = {r[0]: r[2]["median_abs_pct"] for r in rows if r[0] in fixable}
    fixable_ok = all(p <= AREA_THRESHOLD_PCT for p in fixable_pct.values())
    aneur = next(r for r in rows if r[0] == "bent_aneurysm")
    aneur_improved = aneur[2]["median_abs_pct"] < aneur[1]["median_abs_pct"] * 0.5
    others_pct = {r[0]: r[2]["median_abs_pct"] for r in rows if r[0] != "straight_cylinder"}

    # Localization must be preserved vs raw A: re-check it didn't move.
    loc_preserved = True
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        l_raw = metrics.localization_error(ph, CenterlineProjectionUnwrap(ph.centerline))
        l_cor = metrics.localization_error(ph, CurvatureCorrectedUnwrap(ph.centerline))
        if abs(l_raw["median_ds_mm"] - l_cor["median_ds_mm"]) > 1e-9 or \
           abs(l_raw["median_dcirc_mm"] - l_cor["median_dcirc_mm"]) > 1e-9:
            loc_preserved = False

    print(f"straight cylinder near-zero distortion (<0.5%): "
          f"{straight[2]['median_abs_pct']:.3f}%  -> {'OK' if straight_ok else 'BUG'}")
    for nm in fixable:
        p = fixable_pct[nm]
        flag = "within" if p <= AREA_THRESHOLD_PCT else "EXCEEDS"
        print(f"  {nm:<16} {p:6.2f}%  {flag} {AREA_THRESHOLD_PCT:.0f}% threshold")
    print(f"  {'bent_aneurysm':<16} {aneur[2]['median_abs_pct']:6.2f}%  "
          f"ACCEPTED intrinsic residual (raw {aneur[1]['median_abs_pct']:.1f}% -> "
          f"corrected {aneur[2]['median_abs_pct']:.1f}%; remedy = Architecture B)")
    print(f"localization preserved vs raw A (s, theta unchanged): "
          f"{'YES' if loc_preserved else 'NO'}")

    if straight_ok and fixable_ok and loc_preserved and aneur_improved:
        print("\nDECISION: curvature-corrected Architecture A accepted -> "
              "proceed to Phase 4. Straight=exact, bend/taper within threshold, "
              "localization preserved; aneurysm residual documented as an "
              "intrinsic limitation (see error_budget.md).")
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print("PHASE 3 GATE:", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
