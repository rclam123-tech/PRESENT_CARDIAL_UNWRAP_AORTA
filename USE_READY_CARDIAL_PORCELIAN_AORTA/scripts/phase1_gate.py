"""Phase 1 GATE -- mask handoff contract.

Resolves the coordinate question (calcium_deposits is an INDEX/grid-based label
volume on the CT grid), then verifies:
  * round-trip physical -> index -> physical is sub-voxel, and
  * recovered voxels land back inside the calcium mask (no mirror/flip), shown
    in an overlay PNG.

Runs on every phantom. (A real case would slot in via load_calcium_from_files;
no real-data paths were supplied, so the phantom battery stands in.)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402
from aortic_unwrap.geometry import physical_to_voxel  # noqa: E402
from aortic_unwrap.mask_io import calcium_points, roundtrip_error  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def overlay_png(ph, handoff):
    cz = int(np.round(ph.calcium_idx.mean(axis=0)[2]))
    rec = np.round(physical_to_voxel(handoff.points_ras, handoff.affine_ras)).astype(int)
    sel = rec[rec[:, 2] == cz]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(ph.ct[:, :, cz].T, cmap="gray", origin="lower")
    ax.scatter(sel[:, 0], sel[:, 1], s=8, edgecolor="cyan", facecolor="none",
               label="recovered calcium pts")
    ax.set_title(f"{ph.name}: calcium points on CT (axial k={cz})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    p = OUT / f"phase1_overlay_{ph.name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def main():
    voxel_diag = None
    ok = True
    print(f"{'phantom':<18} {'roundtrip(mm)':>14} {'subvoxel?':>10} "
          f"{'in-mask frac':>13}  png")
    print("-" * 74)
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        # Resolve handoff: phantom world is RAS; mask is a grid label volume.
        handoff = calcium_points(ph.calcium_deposits, ph.affine, system="RAS",
                                 ct_shape=ph.ct.shape)
        rt = roundtrip_error(handoff.points_ras, handoff.affine_ras)
        vox = float(np.min(handoff.spacing))
        subvoxel = rt < 0.5 * vox

        rec = np.round(physical_to_voxel(handoff.points_ras, handoff.affine_ras)).astype(int)
        in_mask = ph.calcium_deposits[rec[:, 0], rec[:, 1], rec[:, 2]]
        frac = float(np.mean(in_mask))

        png = overlay_png(ph, handoff)
        good = subvoxel and frac > 0.999
        ok &= good
        print(f"{name:<18} {rt:>14.2e} {str(subvoxel):>10} {frac:>13.4f}  {png.name}"
              + ("" if good else "  <-- FAIL"))
    print("-" * 74)
    print("PHASE 1 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
