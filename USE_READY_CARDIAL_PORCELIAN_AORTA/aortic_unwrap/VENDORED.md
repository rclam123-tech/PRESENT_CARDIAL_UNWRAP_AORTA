# Vendored shared package (STOPGAP — not a solved problem)

The modules listed below are the **shared core** of the aortic-unwrap work. They
are duplicated, byte-for-byte, into two separate public repositories:

- `rclam123-tech/aortic_ct_unwrap_public`  (grayscale CT panorama)
- `rclam123-tech/PRESENT_CARDIAL_UNWRAP_AORTA`  (binary calcium projection)

**Single source of truth:** the copy under
`PRESENT_CARDIAL_UNWRAP_AORTA/USE_READY_CARDIAL_PORCELIAN_AORTA/aortic_unwrap/`.
Every change lands there first and is then copied verbatim into
`aortic_ct_unwrap_public/aortic-calcium-unwrap-public/aortic_unwrap/`.

The two copies are kept in sync **by content hash, not by commit identity** — the
manifest below is the contract, and `scripts/check_vendored.py` (mirrored as
`tests/test_vendored.py`) recomputes these hashes and fails on any drift. There is
deliberately **no upstream commit SHA recorded here**: it would be circular (the
commit does not exist until after this file is written) and would always be one
commit stale.

**This is a stopgap, not the real fix.** Copying source across two public repos is
fragile: a fix applied to one copy silently rots the other until the hash check
catches it. The real fix is to extract this core into **one installable package**
(`pip install aortic-unwrap`) that both repos depend on. Until that exists, treat
the hash check as a tripwire, not a solution.

Only these five modules are shared and hashed. Repo-B-only modules
(`unwrap_a.py`, `raster.py`, `phantoms.py`, `metrics.py`) are **not** vendored
into repo A and are **not** in this manifest, so ordinary repo-B-only edits do
not trip repo A's check. Repo A's `aortic_unwrap/__init__.py` is intentionally
empty (it imports submodules directly).

## Manifest (sha256)

```
6a6938e580e1cf5b9466c06d6e189f27da1a8f23674814521f7abea33ee87e6c  geometry.py
cededcb7eba9247a3d69c0595d5ce4b24bc4a718dda894c487d4991f3d8067e4  frame.py
f4a071037b29f3716170a50af3a3ff065682dff90efc8e3f81f434ebd0579a66  centerline.py
eef4411788bfd1a861f43afccdff33bace4c5df7c121fe2548214af0f79e500a  mask_io.py
73ed7edbbac6c2f8f6741d7a6b01bbbbe24a59b78e3ce3a34ddf8cd2ebc161e5  wall.py
```
