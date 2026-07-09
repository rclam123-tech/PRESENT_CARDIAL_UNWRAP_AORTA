"""Run every phase gate in order and summarize PASS/FAIL

Each gate is an independent, executable check (the staged build's contract).
This driver just runs them sequentially for a one-shot status.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATES = [
    ("Phase 0  phantom harness", "phase0_gate.py"),
    ("Phase 1  mask handoff", "phase1_gate.py"),
    ("Phase 2  centerline + RMF", "phase2_gate.py"),
    ("Phase 3  unwrap decision", "phase3_gate.py"),
    ("Phase 4  raster + inverse", "phase4_gate.py"),
    ("Phase 5  acceptance report", "phase5_report.py"),
]


def main():
    results = []
    for label, script in GATES:
        print(f"\n{'=' * 78}\n### {label}  ({script})\n{'=' * 78}")
        proc = subprocess.run([sys.executable, str(HERE / script)])
        results.append((label, proc.returncode == 0))
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
