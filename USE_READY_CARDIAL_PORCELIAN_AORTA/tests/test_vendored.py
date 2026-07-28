"""Vendored-drift tripwire, mirrored from scripts/check_vendored.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_vendored  # noqa: E402


def test_vendored_shared_modules_match_manifest():
    ok, problems = check_vendored.check()
    assert ok, "vendored shared modules drifted from the manifest:\n" + "\n".join(problems)
