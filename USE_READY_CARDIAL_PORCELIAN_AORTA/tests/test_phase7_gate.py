"""Phase 7 -- aorta boundary gate, mirrored from scripts/phase7_gate.py.

Each assertion is able to fail on a plausible bug (no tautological shape checks):
removing the gate breaks the distractor + arch checks; swapping assign() back to
nearest() breaks the arch check; mis-assigning in-aorta calcium breaks the
localization check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase7_gate  # noqa: E402


def test_distractor_excluded_precision_recall_one():
    d = phase7_gate.check_distractor()
    assert d["n_distractor"] > 0            # fixture actually placed a distractor
    # Every out-of-aorta distractor is rejected AND every wall calcium is kept.
    assert d["precision"] == 1.0
    assert d["recall"] == 1.0


def test_arch_nearest_wrong_assign_refuses():
    a = phase7_gate.check_arch()
    # The trap is real: plain nearest() maps it onto the WRONG (thin) limb.
    assert a["nearest_maps_to_wrong_limb"]
    # assign() refuses to paint it onto the wrong limb (out-of-aorta, vertex -1).
    assert a["assign_excludes_trap"]
    # ...while genuine thick-wall calcium still resolves to the correct limb.
    assert a["ctrl_resolves_correct"]


def test_in_aorta_localization_unchanged_vs_baseline():
    rows = phase7_gate.check_localization()
    for name, r in rows.items():
        assert r["recall_all"], f"{name}: gate dropped genuine aortic calcium"
        assert r["d_med_ds"] <= phase7_gate.MED_TOL_MM, name
        assert r["d_med_dcirc"] <= phase7_gate.MED_TOL_MM, name
        assert r["d_max_ds"] <= phase7_gate.MAX_TOL_MM, name
