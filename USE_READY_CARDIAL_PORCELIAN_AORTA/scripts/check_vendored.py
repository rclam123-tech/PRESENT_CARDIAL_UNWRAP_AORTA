"""Verify the vendored shared modules against the sha256 manifest in
``aortic_unwrap/VENDORED.md``.

This is a drift tripwire for the stopgap that duplicates the shared core across
two public repos (see VENDORED.md). It recomputes the sha256 of each file listed
in the manifest and fails if any differs -- and NOTHING ELSE (it does not diff
against the other repo, which is not present in a single checkout).
"""

import hashlib
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "aortic_unwrap"
MANIFEST = PKG / "VENDORED.md"
_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)\s*$")


def parse_manifest(text):
    """{filename: sha256} from the fenced manifest block in VENDORED.md."""
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m:
            out[m.group(2)] = m.group(1)
    if not out:
        raise ValueError(f"no sha256 manifest lines found in {MANIFEST}")
    return out


def check():
    """Return (ok, list_of_problem_strings)."""
    manifest = parse_manifest(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for fname, want in manifest.items():
        f = PKG / fname
        if not f.exists():
            problems.append(f"{fname}: MISSING")
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            problems.append(f"{fname}: drift\n    manifest {want}\n    actual   {got}")
    return (not problems), problems


def main():
    ok, problems = check()
    if ok:
        print(f"VENDORED: OK ({len(parse_manifest(MANIFEST.read_text()))} shared "
              f"files match the manifest)")
        return 0
    print("VENDORED: DRIFT DETECTED")
    for p in problems:
        print("  " + p)
    print("\nThe shared core diverged from the manifest. Re-sync from the source "
          "of truth (PRESENT_CARDIAL_UNWRAP_AORTA) and update VENDORED.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
