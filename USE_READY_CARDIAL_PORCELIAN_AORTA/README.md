# Aortic Calcium Unwrap (display-only)

A standalone Python package that projects an **already-scored** `calcium_deposits`
mask onto a flattened ("unwrapped") 2D map of the aorta, for visualization and
lesion localization.

> The native-axial Agatston score is **never** recomputed, resampled, or
> interpolated. This layer is strictly downstream of, and read-only with respect
> to, scoring.

It runs end-to-end in a plain `venv` with **no 3D Slicer dependency**: the whole
pipeline is validated on analytic phantoms, and the one real-data dependency
(the centerline) is a pluggable interface that loads a polyline exported once
from Slicer/VMTK.

## The four hard invariants

1. **Score is sacred.** The unwrap is read-only w.r.t. the calcium score.
2. **Physical coordinates everywhere.** Every voxel↔point conversion carries the
   image affine (direction cosines · spacing + origin); LPS/RAS handled
   explicitly. See `geometry.py`.
3. **Rotation-minimizing frame**, never Frenet (which flips at low curvature and
   twists through the arch). Double-reflection Bishop frame in `frame.py`.
4. **Phantoms are the only area oracle.** Real data has no ground-truth calcium
   area — on patients we report only consistency (round-trip + landmark), never
   "validated area".

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or: pip install -e .[test]
```

## Run the gates

Each phase ends at an executable gate that prints PASS/FAIL. Run them in order,
or all at once:

```bash
python scripts/run_all_gates.py        # everything, with a summary
python scripts/phase0_gate.py          # ...or one phase at a time
pytest -q                              # the same checks as assertions
```

Artifacts (sanity PNGs, the unwrap `.npz` files, and `error_budget.md`) are
written to `outputs/`.

## Architecture decision (the project's fork)

The build plan forks on **data, not opinion**:

- **Architecture A — centerline projection.** Per calcium point: nearest
  centerline vertex → `(s, θ·r)`. No mesh. Cheapest; handles near-cylindrical
  vessels exactly.
- **Architecture B — mesh parameterization** (authalic/ARAP). Principled area
  control, but needs a clean manifold mesh and branch handling.

The Phase 3 gate measured Architecture A on all four phantoms:

| phantom | raw A area dist. | **corrected A** | localization |
|---|---|---|---|
| straight cylinder | 0.00% | **0.00%** | <0.1 mm |
| tapered tube | 0.17% | **0.17%** | <0.1 mm |
| constant-curvature bend | 17.5% | **3.9%** | <0.1 mm |
| aneurysm-on-arch | 41.1% | **10.3%** | <0.1 mm |

Raw A failed the arch area gate (41%). Rather than escalate to the full mesh
pipeline (Architecture B; `libigl` could not be built in this environment, and
the tool is display/localization-only), we use **curvature-corrected
Architecture A**: it applies the tube area Jacobian
`r·(1 − κr·cos(θ − θ_inner))` to the *circumferential* coordinate while leaving
`u = s` and the angular position `θ` — and therefore localization — untouched.
This restores area to ≤4% on the bend and improves the arch 4×.

The aneurysm's remaining ~10% is **intrinsic Gaussian-curvature distortion of a
centerline-frame unwrap** (a large aneurysmal radius on a high-curvature arch),
removable only by Architecture B. It is a documented worst-case on a phantom
*designed* to stress the method, does not affect localization, and is never used
as a real-data error bar. Full detail in `outputs/error_budget.md`.

## Using it on real data

```python
from aortic_unwrap.mask_io import load_calcium_from_files
from aortic_unwrap.centerline import PolylineFileCenterline
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap
from aortic_unwrap.raster import rasterize

# 1. Consume the scored mask + CT (resolves LPS/RAS from the file headers).
handoff = load_calcium_from_files("calcium_deposits.nrrd", "ct.nii.gz")

# 2. Centerline exported once from SlicerVMTK Extract Centerline (VTK or CSV).
centerline = PolylineFileCenterline.from_file("centerline.vtk", system="LPS")

# 3. Unwrap + rasterize + persist the inverse map.
unwrap = CurvatureCorrectedUnwrap(centerline)
raster = rasterize(unwrap(handoff.points_ras), pixel=0.35)
raster.save("unwrap_case.npz")     # 2D image + per-pixel (s, θ, r) inverse map
```

`calcium_deposits` is consumed as a **binary label volume on the CT grid**
(index/grid based); the loader asserts mask/CT co-registration and converts
scored voxels to physical RAS points through the affine.

## Layout

```
aortic_unwrap/
  geometry.py     affine / voxel<->physical / LPS<->RAS         (invariant #2)
  frame.py        double-reflection rotation-minimizing frame   (invariant #3)
  centerline.py   Centerline interface + analytic + file-loading impls
  phantoms.py     four analytic phantoms + ground-truth area     (Phase 0)
  mask_io.py      scored-mask handoff contract + real-data loaders (Phase 1)
  unwrap_a.py     centerline projection + curvature-corrected unwrap (Phase 3)
  raster.py       2D raster + stored inverse map (.npz)           (Phase 4)
  metrics.py      area distortion, localization, Dice/IoU, round-trip
scripts/          one executable gate per phase + run_all_gates.py
tests/            the gates as pytest assertions
outputs/          generated PNGs, .npz, error_budget.md (gitignored)
```

## What is intentionally NOT built

- **Architecture B** (mesh parameterization / branch clipping) — deferred; only
  needed if accurate *area* depiction at aneurysmal arches becomes a requirement.
- **Interactive click-back-to-3D UI** — the inverse map is stored (Phase 4), but
  the interactive viewer is a deferred feature.
