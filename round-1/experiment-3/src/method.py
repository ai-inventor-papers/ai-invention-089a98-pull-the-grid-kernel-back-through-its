#!/usr/bin/env python3
"""method.py -- entry point for the screen-C experiment.

"Is the Gaussian process doing the work?" A mechanism ablation of the phase-aware
power-flow surrogate kernel on the IEEE European LV feeder.

This script runs the whole study end to end:

  1. verify the archived three-phase AC ground truth and rebuild the analytic
     sensitivity operator S = d|V_far|/dx (zero AC solves);
  2. run every arm and every baseline on the same 108 folds;
  3. write the result tables, the figures and the audit;
  4. write results/method_out.json and this workspace's method_out.json.

The heavy lifting lives in code/; this file is the reproducible driver with the
one-command interface, so a reader does not have to know the module layout.

    uv run method.py                # everything (about 4 minutes on 2 vCPUs)
    uv run method.py --quick        # smoke test only, no sweep
    uv run method.py --no-figures   # skip figure rendering

ZERO LLM calls. No OpenRouter client is imported anywhere in this artifact.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE))

STEPS = [
    ("build + verify the shared state, run the full sweep, write result tables",
     "run_screen.py"),
    ("independently re-derive every headline number through a different code path",
     "audit_headline.py"),
    ("render the figures in the house style", "make_figures.py"),
]


def run(script: str) -> int:
    print(f"\n{'='*72}\n[{script}] {dict(STEPS)[script] if False else ''}\n{'='*72}",
          flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(CODE / script)],
                       cwd=str(CODE.parent))
    print(f"[{script}] exit {r.returncode} in {time.time()-t0:.1f}s", flush=True)
    return r.returncode


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    no_figs = "--no-figures" in argv
    print("openrouter_calls = 0 | openrouter_usd = 0.0 | zero LLM calls", flush=True)

    if quick:
        return run("smoke.py")

    rc = run("run_screen.py")
    if rc != 0:
        print("ERROR: the sweep failed; see logs/run_screen.log")
        return rc
    rc = run("audit_headline.py")
    if rc != 0:
        print("ERROR: the independent audit disagreed with the pipeline. Do not trust the "
              "numbers until this is resolved; see results/audit_headline.json")
        return rc
    if not no_figs:
        rc = run("make_figures.py")
        if rc != 0:
            print("WARNING: figure rendering failed; the result tables are still valid")
    print("\nDONE. Main outputs:")
    for p in ("results/method_out.json", "method_out.json", "results/screen_C.json",
              "results/baseline_table.json", "results/headtohead_C.json",
              "results/calibration_C.json", "results/audit_headline.json"):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
