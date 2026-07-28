"""Phase 1 -- the mask handoff contract.

This is the seam between the existing (FINAL, read-only) Agatston scorer and the
new unwrap layer. The plan calls it the riskiest seam in the project: a silent
LPS/RAS flip or off-by-one here corrupts everything downstream while still
looking plausible.

Resolved coordinate question
----------------------------
``calcium_deposits`` is a **binary label volume sampled on the CT voxel grid** --
i.e. it is INDEX/grid based, not a list of physical points. The scored voxels are
``np.argwhere(mask)``. We therefore convert to physical world points by pushing
each voxel index through the image affine (direction cosines * spacing + origin),
never by bare index arithmetic.

The mask and the CT are assumed co-registered (identical grid + affine), which is
what an in-pipeline scorer that labels CT voxels produces. The loader asserts
this. World points are returned in canonical RAS regardless of the file's native
convention (NIfTI=RAS, NRRD/ITK=LPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import affine_to_ras, voxel_to_physical, voxel_spacing


@dataclass
class CalciumHandoff:
    """Result of consuming the scored mask + CT."""

    points_ras: np.ndarray   # (M, 3) physical RAS points of scored voxels
    voxel_idx: np.ndarray    # (M, 3) the originating voxel indices
    affine_ras: np.ndarray   # 4x4 index->RAS affine
    shape: tuple
    spacing: np.ndarray
    n_scored: int
    # Optional aorta segmentation for the boundary gate, kept on its OWN grid
    # (NOT forced onto the CT grid) so it can be looked up in physical RAS,
    # nearest-neighbour. None -> the downstream unwrap falls back to a
    # radial-only gate and says so out loud.
    aorta_mask: np.ndarray | None = None
    aorta_affine_ras: np.ndarray | None = None


def calcium_points(calcium_mask, affine, system: str = "RAS",
                   ct_shape=None, aorta_mask=None,
                   aorta_affine_ras=None) -> CalciumHandoff:
    """Convert a scored binary mask to physical RAS points.

    Parameters
    ----------
    calcium_mask : 3D bool/int array
        The scored ``calcium_deposits`` label volume on the CT grid.
    affine : 4x4
        Index -> world affine of the CT/mask grid (native convention).
    system : 'RAS' | 'LPS'
        World convention of ``affine`` (NIfTI -> RAS, NRRD/ITK -> LPS).
    ct_shape : optional tuple
        If given, asserts the mask grid matches the CT grid (co-registration).
    aorta_mask, aorta_affine_ras : optional
        Aorta segmentation for the boundary gate, on its OWN grid with an
        already-RAS affine. Carried through unchanged; NOT forced onto the CT
        grid and NOT co-registration-checked (it is looked up in physical RAS).
    """
    mask = np.asarray(calcium_mask)
    if mask.ndim != 3:
        raise ValueError("calcium_deposits must be a 3D label volume")
    if ct_shape is not None and tuple(mask.shape) != tuple(ct_shape):
        raise ValueError(
            f"calcium mask shape {mask.shape} != CT shape {ct_shape}; "
            "mask and CT must be co-registered on the same grid"
        )
    affine_ras = affine_to_ras(affine, system)
    idx = np.argwhere(mask.astype(bool))
    pts = voxel_to_physical(idx, affine_ras)
    return CalciumHandoff(
        points_ras=pts,
        voxel_idx=idx,
        affine_ras=affine_ras,
        shape=tuple(mask.shape),
        spacing=voxel_spacing(affine_ras),
        n_scored=len(idx),
        aorta_mask=(None if aorta_mask is None
                    else np.asarray(aorta_mask).astype(bool)),
        aorta_affine_ras=(None if aorta_affine_ras is None
                          else np.asarray(aorta_affine_ras, float)),
    )


def roundtrip_error(points_ras, affine_ras) -> float:
    """Max physical->index->physical recovery error (mm). Should be sub-voxel."""
    from .geometry import physical_to_voxel
    idx = physical_to_voxel(points_ras, affine_ras)
    back = voxel_to_physical(idx, affine_ras)
    return float(np.max(np.linalg.norm(back - points_ras, axis=1)))


# --------------------------------------------------------------------------- #
# Real-data loaders (NIfTI CT, NRRD segmentation/mask). Optional deps.
# --------------------------------------------------------------------------- #
def load_nifti(path):
    """Load a NIfTI volume. Returns (array, affine, 'RAS')."""
    import nibabel as nib

    img = nib.load(str(path))
    # nibabel's affine is always index->RAS by construction.
    return np.asanyarray(img.dataobj), np.asarray(img.affine, float), "RAS"


def load_nrrd(path):
    """Load an NRRD volume. Returns (array, affine, 'LPS').

    NRRD/ITK store space directions + origin in LPS by default. We assemble the
    index->world affine from ``space directions`` and ``space origin`` and report
    the convention so the caller can canonicalize to RAS.
    """
    import nrrd

    data, header = nrrd.read(str(path))
    directions = np.array(header["space directions"], float)  # rows are axes
    origin = np.array(header["space origin"], float)
    affine = np.eye(4)
    affine[:3, :3] = directions.T  # columns must be the per-index-axis world step
    affine[:3, 3] = origin
    space = header.get("space", "left-posterior-superior").lower()
    system = "LPS" if space.startswith("left") else "RAS"
    return data, affine, system


def load_calcium_from_files(calcium_path, ct_path,
                            aorta_seg=None) -> CalciumHandoff:
    """Convenience real-data path: read CT + calcium mask, return the handoff.

    Picks the loader by extension (.nii/.nii.gz -> NIfTI, .nrrd -> NRRD).

    ``aorta_seg`` (optional) is read the same way and carried on the handoff for
    the boundary gate. It is kept on its OWN affine (canonicalized to RAS) and is
    NOT forced onto the CT grid -- the gate looks it up in physical RAS,
    nearest-neighbour (order=0, never linear on a label). If omitted, downstream
    falls back to the radial-only gate.
    """
    def _load(p):
        p = Path(p)
        name = p.name.lower()
        if name.endswith((".nii", ".nii.gz")):
            return load_nifti(p)
        if name.endswith(".nrrd"):
            return load_nrrd(p)
        raise ValueError(f"unrecognized volume extension: {p.name}")

    ct, ct_affine, _ = _load(ct_path)
    mask, mask_affine, mask_system = _load(calcium_path)

    aorta_mask = aorta_affine_ras = None
    if aorta_seg is not None:
        aseg, aseg_affine, aseg_system = _load(aorta_seg)
        aorta_mask = np.asarray(aseg) > 0
        aorta_affine_ras = affine_to_ras(aseg_affine, aseg_system)

    return calcium_points(mask, mask_affine, system=mask_system,
                          ct_shape=ct.shape, aorta_mask=aorta_mask,
                          aorta_affine_ras=aorta_affine_ras)
