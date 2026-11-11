"""check_determinism.py -- the plan's Step 12E, as a script rather than a promise.

Copies every result file, re-runs the WHOLE pipeline (both the sequential and the pooled
path, since they must agree), and diffs the results byte for byte. Any difference is a bug,
not a finding. Exits non-zero on any difference.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
RESULTS = ROOT / "results"
SNAP = ROOT / "logs" / "determinism_snapshot"

FILES = ["protocol_checks.json", "operator_checks.json", "baseline_table.json",
         "screen_C.json", "headtohead_C.json", "calibration_C.json", "method_out.json"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    missing = [f for f in FILES if not (RESULTS / f).is_file()]
    if missing:
        print(f"ERROR: run code/run_screen.py first; missing {missing}")
        return 2
    SNAP.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(RESULTS / f, SNAP / f)
    before = {f: sha256(RESULTS / f) for f in FILES}
    print("snapshot taken:", {k: v[:12] for k, v in before.items()})

    print("re-running the whole pipeline ...")
    r = subprocess.run([sys.executable, str(CODE / "run_screen.py")],
                       cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        print("ERROR: the second pipeline run failed")
        return 1

    after = {f: sha256(RESULTS / f) for f in FILES}
    bad = {f: (before[f], after[f]) for f in FILES if before[f] != after[f]}
    if bad:
        for f, (a, b) in bad.items():
            print(f"NONDETERMINISM in {f}: {a[:16]} != {b[:16]}")
        return 1
    print("DETERMINISTIC: all 7 result files byte-identical across two full pipeline runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
