"""Phase 0 GATE -- generate all four phantoms, print stats, save sanity PNGs.

PASS when every phantom produces a volume + aorta mask + calcium mask + a
ground-truth table, and the saved PNG shows the calcium where it was placed.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def save_png(ph):
    """Sagittal + axial slices through the calcium centroid."""
    cz = int(np.round(ph.calcium_idx.mean(axis=0)[2]))
    cx = int(np.round(ph.calcium_idx.mean(axis=0)[0]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

    # Axial slice at the calcium k-centroid.
    ax[0].imshow(ph.ct[:, :, cz].T, cmap="gray", origin="lower")
    cal = ph.calcium_deposits[:, :, cz]
    ys, xs = np.nonzero(cal.T)
    ax[0].scatter(xs, ys, s=4, c="red")
    ax[0].set_title(f"{ph.name}\naxial k={cz}")

    # Sagittal-ish slice at the calcium i-centroid.
    ax[1].imshow(ph.ct[cx, :, :].T, cmap="gray", origin="lower")
    cal = ph.calcium_deposits[cx, :, :]
    ys, xs = np.nonzero(cal.T)
    ax[1].scatter(xs, ys, s=4, c="red")
    ax[1].set_title(f"slice i={cx}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    p = OUT / f"phase0_{ph.name}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main():
    ok = True
    print(f"{'phantom':<18} {'shape':<16} {'len(mm)':>8} {'#cal vox':>9} "
          f"{'area(mm^2)':>11}  png")
    print("-" * 78)
    for name, fn in phantoms.ALL_PHANTOMS.items():
        ph = fn()
        n = len(ph.calcium_idx)
        checks = [
            ph.ct.shape == ph.shape,
            ph.aorta_mask.shape == ph.shape,
            ph.calcium_deposits.shape == ph.shape,
            n > 0,
            np.count_nonzero(ph.calcium_deposits) == n,
            ph.lesion_true_area > 0,
            len(ph.s_gt) == n and len(ph.theta_gt) == n and len(ph.r_gt) == n,
        ]
        png = save_png(ph)
        ok &= all(checks)
        print(f"{name:<18} {str(ph.shape):<16} {ph.meta['length']:>8.1f} "
              f"{n:>9d} {ph.lesion_true_area:>11.1f}  {png.name}"
              + ("" if all(checks) else "  <-- CHECK FAILED"))
    print("-" * 78)
    print("PHASE 0 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
