# Aortic Calcium Unwrap

**A display-only pipeline that flattens an already-scored aortic calcium mask onto a 2D map of the vessel wall, for lesion localization — without ever touching the calcium score.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License](https://img.shields.io/badge/license-see%20below-lightgrey)

> The package lives in [`CARDIAL_PORCELIAN_AORTA/`](CARDIAL_PORCELIAN_AORTA/) (my file; do not click or you will get error). This page is the project overview; that directory has the full technical README.

---

## What this is

Given a CT and a segmentation in which aortic wall calcification has **already been scored** (Agatston), this tool projects the scored voxels onto a flattened ("unwrapped") 2D map of the aorta so a reader can see *where* on the vessel wall the calcium sits. The motivating case is the **porcelain aorta** — circumferential wall calcification — where a flat map makes the distribution legible at a glance.

It is strictly a **visualization / localization** layer. It is downstream of scoring and read-only with respect to it.

## What this is *not*

- **Not a scoring tool.** The Agatston score is never recomputed, resampled, or interpolated. If you want a number, this is the wrong repo.
- **Not validated for clinical use.** Accuracy is established only on analytic phantoms (see below). On real anatomy the tool reports *self-consistency*, never validated accuracy.
- **Not dependent on 3D Slicer at runtime.** The whole pipeline runs in a plain `venv`. Slicer/VMTK is used once, offline, to export a centerline — and even that is a pluggable interface with a no-Slicer fallback (`SegmentationCenterline`).

## Design guarantees (the four invariants)

1. **The score is sacred.** The unwrap is read-only with respect to the calcium score.
2. **Physical coordinates everywhere.** Every voxel↔point conversion carries the image affine; LPS/RAS is resolved explicitly from file headers, never assumed.
3. **Rotation-minimizing frame, never Frenet.** A double-reflection Bishop frame is stable through straight runs and the arch alike, where Frenet's normal flips and twists.
4. **Phantoms are the only accuracy oracle.** Real data has no ground-truth calcium area, so on real cases the tool reports only consistency (round-trip + landmark), never a "validated area."

## Results (reproducible)

Every number below is emitted by the phase gates and is reproduced by `pytest`. Area distortion is measured analytically per voxel (not by counting rasterized pixels), so a perfect unwrap reads exactly 0%.

| phantom | raw area dist. | **corrected** | localization Δ | Dice |
|---|---|---|---|---|
| straight cylinder | 0.00% | **0.00%** | < 0.1 mm | 0.93 |
| tapered tube | 0.17% | **0.17%** | < 0.1 mm | 0.92 |
| constant-curvature bend | 17.48% | **3.86%** | < 0.1 mm | 0.96 |
| aneurysm-on-arch | 41.09% | **10.34%** | < 0.1 mm | 0.99 |

The curvature-corrected unwrap rescales only the *circumferential* coordinate by the true tube-surface area Jacobian, leaving arclength and angular position — and therefore localization — untouched. The aneurysm-on-arch residual (~10%) is intrinsic Gaussian-curvature distortion of any centerline-frame unwrap; it is a documented worst case on a phantom built to stress the method, does not affect localization, and is never used as a real-data error bar. Full derivation in `CARDIAL_PORCELIAN_AORTA/outputs/error_budget.md`.

## Quickstart

```bash
cd CARDIAL_PORCELIAN_AORTA
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install -e .[test]
```

Run the validation gates (each prints PASS/FAIL):

```bash
python scripts/run_all_gates.py    # all phases, with a summary
pytest -q                          # the same checks as assertions (24 tests)
```

Artifacts (sanity PNGs, `.npz` unwraps, `error_budget.md`) are written to `outputs/`.

## Using it on real data

```python
from aortic_unwrap.mask_io import load_calcium_from_files
from aortic_unwrap.centerline import PolylineFileCenterline
from aortic_unwrap.unwrap_a import CurvatureCorrectedUnwrap
from aortic_unwrap.raster import rasterize

handoff   = load_calcium_from_files("calcium_deposits.nrrd", "ct.nii.gz")
centerline = PolylineFileCenterline.from_file("centerline.vtk", system="LPS")
unwrap     = CurvatureCorrectedUnwrap(centerline)
raster     = rasterize(unwrap(handoff.points_ras), pixel=0.35)
raster.save("unwrap_case.npz")     # 2D image + per-pixel (s, θ, r) inverse map
```

On real cases the pipeline reports **consistency only** — round-trip error, frame twist, fold cells, and `(s, θ)` occupancy — because real anatomy has no ground-truth surface area or canonical `(s, θ)` coordinate map. These are plausibility checks, not accuracy claims.

## Repository layout

```
CARDIAL_PORCELIAN_AORTA/
  aortic_unwrap/      core package
    geometry.py       affine / voxel↔physical / LPS↔RAS      (invariant #2)
    frame.py          double-reflection rotation-minimizing frame (invariant #3)
    centerline.py     Centerline interface: analytic, file, segmentation
    phantoms.py       four analytic phantoms + ground-truth area
    mask_io.py        scored-mask handoff contract + real-data loaders
    unwrap_a.py       centerline projection + curvature-corrected unwrap
    raster.py         2D raster + stored inverse map (.npz)
    metrics.py        area distortion, localization, Dice/IoU, round-trip
  scripts/            one executable gate per phase + run_all_gates.py
  tests/              the gates as pytest assertions
  outputs/            generated PNGs, .npz, error_budget.md
```


## Data provenance

No protected health information or identifiable patient imaging data should be committed to this repository.

The included outputs are intended to be generated from synthetic phantoms or de-identified research data only. Before publishing any derived figures from real datasets, confirm that the dataset license allows redistribution and cite the source appropriately.

If third-party research data is used, add:

- **Dataset name:**
- **Source / citation:**
- **DOI or URL:**
- **License / terms of use:**

> These are third-party research data. Confirm the dataset's license permits redistribution of derived images before publishing this repo, and cite the source as required.

## License

<!-- TODO: resolve ownership before choosing a license. -->
_No license is set yet._ Until a `LICENSE` file exists, all rights are reserved and the code cannot be reused by others.

## Status & scope

This is a research prototype. Intentionally **not** built:

- **Mesh-parameterization unwrap** (authalic/ARAP) — deferred; only needed if accurate *area* depiction at aneurysmal arches becomes a requirement.
- **Interactive click-back-to-3D viewer** — the inverse map is stored, but the UI is deferred.
