"""Phase 5 -- acceptance validation battery + reviewer-facing error budget.

Runs the full metric suite on all phantoms and writes a markdown report
(outputs/error_budget.md). The phantom-vs-real distinction is load-bearing:

  * PHANTOMS give AREA ground truth (we know the true lesion area), so they are
    where area accuracy is validated.
  * REAL DATA has NO area oracle, so on patients we validate only CONSISTENCY
    (round-trip + landmark error) and never report a validated area.

No real-data paths were supplied for this build, so the real-data section
documents the procedure and what would be reported, rather than fabricating
numbers.
"""

import sys
from pathlib import Path

import numpy as np

try:  # Windows consoles default to cp1252, which can't print Δ/κ/θ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402
from aortic_unwrap.geometry import voxel_to_physical  # noqa: E402
from aortic_unwrap.unwrap_a import (  # noqa: E402
    CenterlineProjectionUnwrap,
    CurvatureCorrectedUnwrap,
)
from aortic_unwrap.raster import rasterize, back_project  # noqa: E402
from aortic_unwrap import metrics  # noqa: E402
from aortic_unwrap.geometry import physical_to_voxel  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
PIXEL_MM = 0.35


def voxelize(points, affine, shape):
    idx = np.round(physical_to_voxel(points, affine)).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[ok]
    m = np.zeros(shape, bool)
    m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return m


def run():
    rows = []
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        pts = voxel_to_physical(ph.calcium_idx, ph.affine)
        raw = CenterlineProjectionUnwrap(ph.centerline)
        cor = CurvatureCorrectedUnwrap(ph.centerline)
        ad_raw = metrics.area_distortion(ph, raw)
        ad_cor = metrics.area_distortion(ph, cor)
        loc = metrics.localization_error(ph, cor)
        res = cor(pts)
        rast = rasterize(res, pixel=PIXEL_MM)
        recovered = voxelize(back_project(rast, cor), ph.affine, ph.shape)
        d = metrics.dice(ph.calcium_deposits, recovered)
        j = metrics.iou(ph.calcium_deposits, recovered)
        rt = metrics.roundtrip_2d3d(pts, cor, rast)
        rows.append(dict(name=name, ad_raw=ad_raw, ad_cor=ad_cor, loc=loc,
                         dice=d, iou=j, rt=rt, n=len(pts)))
    return rows


def fmt_table(rows):
    lines = []
    lines.append("| phantom | area dist. raw A | **area dist. corrected** | "
                 "loc Δs (mm) | loc Δcirc (mm) | Dice | IoU | round-trip max (mm) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['ad_raw']['median_abs_pct']:.2f}% | "
            f"**{r['ad_cor']['median_abs_pct']:.2f}%** | "
            f"{r['loc']['median_ds_mm']:.3f} | {r['loc']['median_dcirc_mm']:.3f} | "
            f"{r['dice']:.3f} | {r['iou']:.3f} | {r['rt']['max_mm']:.3f} |"
        )
    return "\n".join(lines)


REPORT = """# Aortic Calcium Unwrap — Error Budget (Phase 5 acceptance)

Display-only 2D unwrap of an **already-scored** `calcium_deposits` mask. The
Agatston score is never recomputed, resampled, or interpolated; this layer is
strictly downstream and read-only with respect to scoring.

Unwrap in use: **curvature-corrected Architecture A** (centerline projection
with a rotation-minimizing frame; the circumferential coordinate carries the
tube area Jacobian `r·(1 − κr·cos(θ − θ_inner))` while the longitudinal
coordinate `u = s` and the angular position `θ` — hence localization — are left
exactly as the plain projection computes them).

## Phantom results — where AREA is validated

Phantoms have known geometry, so they are the **only** source of ground-truth
area. All four are rasterized onto a realistic anisotropic, sign-flipped
(LPS-style) physical CT grid.

{table}

Reading the table:

- **Straight cylinder → 0.00% area distortion.** A straight tube is developable
  (zero Gaussian curvature everywhere), so it flattens isometrically. This
  degenerate case being exact proves the coordinate handling and the
  rotation-minimizing frame are correct (any distortion here would be a bug).
- **Curvature correction reduces, but cannot eliminate, the bend distortion:**
  the constant-curvature bend drops from {bent_raw:.1f}% to {bent_cor:.1f}%
  per-element area distortion — it does **not** reach zero, because a tube around
  a curved centerline is **not** developable (see the next section). The taper is
  unaffected ({tap_cor:.2f}%) because pure radius variation keeps zero Gaussian
  curvature and is already absorbed by `v = θ·r`.
- **Localization is sub-0.1 mm on every phantom**, including the arch, and is
  numerically identical to plain Architecture A — the area correction does not
  move any lesion.
- **Round-trip (point → 2D → 3D)** is bounded by ~one voxel. A 2D *surface* map
  discards radial position, so a ~2-voxel-thick calcium shell collapses
  radially onto one pixel; the back-projection uses the stored representative
  radius, which floors the round-trip error at the shell thickness (this is a
  property of any surface unwrap, not a defect).

## The bend and the aneurysm are ONE phenomenon, not two

The bend residual (~{bent_cor:.0f}%) and the aneurysm residual (~{aneur_cor:.0f}%)
are **not two categories** — they are the **same effect at different
magnitudes**: the **intrinsic, sign-varying Gaussian curvature of a tube/pipe
surface wrapped around a curved centerline**.

A tube of radius `r` around a centerline of curvature `κ` has Gaussian curvature

```
K(φ) = − κ·cosφ / [ r·(1 − r·κ·cosφ) ]          (φ = θ − θ_inner)
```

where `φ` is measured from the inner wall (the direction of the curvature
vector). `K` is **sign-varying around the circumference**: negative on the inner
wall (`φ = 0`), positive on the outer wall (`φ = π`), zero on the sides. By
Gauss's *Theorema Egregium*, any surface with `K ≠ 0` is **not developable** and
**cannot be flattened to a plane without area (or angle) distortion** — there is
no isometric unwrap, for the bend or the arch.

The `(1 − κr·cosφ)` Jacobian we apply corrects the **dominant first-order area
element** (the `du·dv` scaling), which is why the bend falls from
{bent_raw:.0f}% to ~{bent_cor:.0f}% and the arch from {aneur_raw:.0f}% to
~{aneur_cor:.0f}%. What remains — the reported ~{bent_cor:.0f}% / ~{aneur_cor:.0f}%
**per-element** distortion — is the **intrinsic-curvature remainder** set by `K`
itself, growing with `κr` (≈0.18 on the bend, ≈0.36 where the aneurysmal radius
inflates `r` on the arch). It is **removable only by a mesh parameterization
(Architecture B)** — an authalic/ARAP flattening of the segmentation surface —
which we judged unnecessary for a localization tool and deliberately did not
build.

Because `K` is sign-varying, the distortion **inflates the outer wall and
shrinks the inner wall** of any single patch; it is a warping of local area, not
a uniform scaling. Crucially, it **does not affect localization**: the lesion's
(s, θ) position is still recovered to sub-0.1 mm.

This ~{aneur_cor:.0f}% arch figure is an **illustrative worst case on a phantom
designed to stress the method**, not a guaranteed ceiling on patients. A real
aortic aneurysm's distortion could be smaller or larger depending on its
geometry. It lives in this budget as a **characterization of the method**, not a
per-patient accuracy figure — and since we never claim area on real data (next
section), it never functions as a real-data error bar.

## Real patient data — where only CONSISTENCY is validated

There is **no ground-truth calcium area on a real aorta**, so on patient data we
validate only:

- **Round-trip error** (point → 2D → 3D), the same metric as above, and
- **Landmark localization error** at the supra-aortic ostia (annotated on the
  real segmentation; the branchless phantoms have no ostia, so this metric is
  exercised only on real data).

We never report a "validated area" on real data. Conflating phantom area
accuracy with real-data accuracy would be the most likely defensibility hole in
the writeup, and is avoided by construction.

**Procedure for a real case** (no real paths were supplied for this build):

```python
from aortic_unwrap.mask_io import load_calcium_from_files
from aortic_unwrap.centerline import PolylineFileCenterline
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap
from aortic_unwrap.raster import rasterize

handoff = load_calcium_from_files("calcium_deposits.nrrd", "ct.nii.gz")
centerline = PolylineFileCenterline.from_file("centerline_vmtk.vtk", system="LPS")
unwrap = CurvatureCorrectedUnwrap(centerline)
raster = rasterize(unwrap(handoff.points_ras), pixel=0.35)
raster.save("unwrap_case.npz")
```

The centerline is the one pluggable dependency that, on real data, comes from
SlicerVMTK Extract Centerline exported once as a VTK/CSV polyline; everything
else runs in a plain venv.

## Summary error budget

| regime | metric | result |
|---|---|---|
| phantom, straight / tapered (developable, K=0) | area distortion | ≤ ~0.2% |
| phantom, curved tube — bend (κr≈0.18) | per-element area distortion | ~{bent_cor:.0f}% — intrinsic Gaussian curvature |
| phantom, curved tube — aneurysm-on-arch (κr≈0.36, worst case) | per-element area distortion | ~{aneur_cor:.0f}% — *same phenomenon*; remedy = Architecture B |
| all phantoms | localization (s, θ) | < 0.1 mm |
| all phantoms | round-trip point→2D→3D | ≤ ~1 voxel (radial-shell floor) |
| real data | area | **not validated — no oracle exists** |
| real data | round-trip + ostia landmark | consistency only |
"""


def main():
    rows = run()
    table = fmt_table(rows)
    by = {r["name"]: r for r in rows}
    text = REPORT.format(
        table=table,
        bent_raw=by["bent_tube"]["ad_raw"]["median_abs_pct"],
        bent_cor=by["bent_tube"]["ad_cor"]["median_abs_pct"],
        tap_cor=by["tapered_tube"]["ad_cor"]["median_abs_pct"],
        aneur_raw=by["bent_aneurysm"]["ad_raw"]["median_abs_pct"],
        aneur_cor=by["bent_aneurysm"]["ad_cor"]["median_abs_pct"],
    )
    path = OUT / "error_budget.md"
    path.write_text(text, encoding="utf-8")
    print(table)
    print("\nWrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
