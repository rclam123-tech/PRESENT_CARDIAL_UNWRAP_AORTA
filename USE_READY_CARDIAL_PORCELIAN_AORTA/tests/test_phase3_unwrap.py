"""Phase 3 -- curvature-corrected Architecture A: area + localization."""

import numpy as np

from aortic_unwrap.unwrap_a import (
    CenterlineProjectionUnwrap,
    CurvatureCorrectedUnwrap,
)
from aortic_unwrap import metrics


def test_straight_zero_distortion(built_phantoms):
    ph = built_phantoms["straight_cylinder"]
    ad = metrics.area_distortion(ph, CurvatureCorrectedUnwrap(ph.centerline))
    assert ad["median_abs_pct"] < 0.5  # degenerate case must be exact


def test_correction_within_threshold_except_aneurysm(built_phantoms):
    for name in ("bent_tube", "tapered_tube"):
        ph = built_phantoms[name]
        ad = metrics.area_distortion(ph, CurvatureCorrectedUnwrap(ph.centerline))
        assert ad["median_abs_pct"] <= 5.0, name


def test_correction_improves_arch(built_phantoms):
    ph = built_phantoms["bent_aneurysm"]
    raw = metrics.area_distortion(ph, CenterlineProjectionUnwrap(ph.centerline))
    cor = metrics.area_distortion(ph, CurvatureCorrectedUnwrap(ph.centerline))
    # Big improvement, but a documented intrinsic residual remains.
    assert cor["median_abs_pct"] < raw["median_abs_pct"] * 0.5
    assert cor["median_abs_pct"] < 15.0


def test_localization_preserved_by_correction(built_phantoms):
    for name, ph in built_phantoms.items():
        raw = metrics.localization_error(ph, CenterlineProjectionUnwrap(ph.centerline))
        cor = metrics.localization_error(ph, CurvatureCorrectedUnwrap(ph.centerline))
        assert abs(raw["median_ds_mm"] - cor["median_ds_mm"]) < 1e-9, name
        assert abs(raw["median_dcirc_mm"] - cor["median_dcirc_mm"]) < 1e-9, name
        assert cor["median_ds_mm"] < 0.2 and cor["median_dcirc_mm"] < 0.2, name
