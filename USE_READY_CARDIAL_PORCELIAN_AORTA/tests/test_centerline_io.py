"""Real-data centerline hook: load polyline from CSV / VTK."""

import numpy as np

from aortic_unwrap.centerline import PolylineFileCenterline


def _sample_points():
    t = np.linspace(0, 2, 40)
    return np.c_[t, np.sin(t), 0.5 * t]


def test_csv_roundtrip(tmp_path):
    pts = _sample_points()
    p = tmp_path / "cl.csv"
    p.write_text("x,y,z\n" + "\n".join(f"{x},{y},{z}" for x, y, z in pts))
    cl = PolylineFileCenterline.from_file(p, system="RAS")
    assert np.allclose(cl.points, pts)
    assert cl.length > 0


def test_vtk_polydata(tmp_path):
    pts = _sample_points()
    n = len(pts)
    lines = ["# vtk DataFile Version 3.0", "centerline", "ASCII",
             "DATASET POLYDATA", f"POINTS {n} float"]
    lines += [f"{x} {y} {z}" for x, y, z in pts]
    lines += [f"LINES 1 {n + 1}", str(n) + " " + " ".join(str(i) for i in range(n))]
    p = tmp_path / "cl.vtk"
    p.write_text("\n".join(lines))
    cl = PolylineFileCenterline.from_file(p, system="RAS")
    assert np.allclose(cl.points, pts)


def test_lps_csv_flips_to_ras(tmp_path):
    pts = _sample_points()
    p = tmp_path / "cl.csv"
    p.write_text("\n".join(f"{x},{y},{z}" for x, y, z in pts))
    cl = PolylineFileCenterline.from_file(p, system="LPS")
    assert np.allclose(cl.points[:, 0], -pts[:, 0])
    assert np.allclose(cl.points[:, 1], -pts[:, 1])
    assert np.allclose(cl.points[:, 2], pts[:, 2])
