"""T4: ONE-FOLD END-TO-END (n = 80, scheme 0, seed 0) through every arm and baseline.

Confirmation signals, in order of importance:
  1. every arm's ratio to the mean predictor is strictly below 1.0 (guard g1);
  2. the five baselines are all below 1.0 and b5 is the strongest, at roughly 0.10
     normalised at n = 80 on the 2.0x regime;
  3. SP-R1a-GP at k = 11 lands in the vicinity of 0.10;
  4. the selected lengthscales are interior to the grid, not at 0.05 or 8.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "code")
from engine import ARM_SPECS, BASELINE_NAMES, run_fold  # noqa: E402
from folds import fold_indices  # noqa: E402
from method_core import prepare  # noqa: E402

if __name__ == "__main__":
    prep = prepare()
    print("\n=== T4 one fold: n=80, scheme 0, seed 0 ===")
    tr, te = fold_indices(80, 0, 0)
    rec = run_fold(prep["ops"], prep["ph"], prep["X"], prep["Y"], tr, te,
                   prep["Vp1"], prep["Vp2"], prep["M110"], prep["M55"], prep["Mhop"],
                   collect_calibrations=True)
    print("mean predictor  %.4f" % rec["mean_predictor_nrmse"])
    print("{:<22} {:<10} {:>8} {:>8} {:>8}".format("arm", "kind", "nrmse", "/mean", "margin%"))
    for b in BASELINE_NAMES:
        d = rec["baselines"][b]
        print("{:<22} {:<10} {:>8.4f} {:>8.3f} {:>8s}".format(
            b, "baseline", d["nrmse"], d["nrmse"] / rec["mean_predictor_nrmse"], "-"))
    print("-" * 60)
    for name, *_ in ARM_SPECS:
        c = rec["cells"][name]
        if "nrmse" not in c:
            print("{:<22} {:<10} {:>8}".format(name, "SPECTRAL", c.get("status")))
            continue
        print("{:<22} {:<10} {:>8.4f} {:>8.3f} {:>+8.1f}".format(
            name, "SPECTRAL", c["nrmse"], c["ratio_to_mean"],
            100 * c["margin_vs_best_baseline"]))
    print("\nbest baseline:", rec["best_baseline"], "%.4f" % rec["best_baseline_nrmse"])
    print("best independent:", rec["best_independent_baseline"],
          "%.4f" % rec["best_independent_nrmse"])
    print("\nLinDistFlow calibrations (nrmse):")
    for k, v in rec["lindist_calibrations"].items():
        print("  {:<26} {:.4f}".format(k, v))
    print("\nselected hyperparameters (headline = train-only CV criterion):")
    for b in BASELINE_NAMES:
        h = rec["baselines"][b]["hyper"]
        print("  {:<22}".format(b), h if not isinstance(h, dict)
              else f"CV l={h['hyper'][0]} s2={h['hyper'][1]} n2={h['hyper'][2]} | "
                   f"MLL would pick l={h['alt_hyper'][0]} (cv_rmse {h['cv_rmse']:.4f} "
                   f"vs {h['alt_cv_rmse']:.4f})")
    for name, *_ in ARM_SPECS:
        c = rec["cells"][name]
        if "lengthscale" in c:
            print("  {:<22} CV l={} s2={} n2={} | MLL l={}".format(
                name, c["hyper"]["hyper"][0], c["hyper"]["hyper"][1],
                c["hyper"]["hyper"][2], c["alt_lengthscale"]))
    g1 = [n for n, *_ in ARM_SPECS
          if rec["cells"].get(n, {}).get("ratio_to_mean", 0) >= 1.0]
    b1bad = [b for b in BASELINE_NAMES
             if rec["baselines"][b]["nrmse"] / rec["mean_predictor_nrmse"] >= 1.0]
    print("\nGUARD g1 (spectral ratio >= 1.0):", g1 if g1 else "clear")
    print("GUARD g1 (baseline ratio >= 1.0):", b1bad if b1bad else "clear")
    ends = [n for n in rec["cells"] if rec["cells"][n].get("grid_endpoint_hit")]
    print("GUARD g3 (lengthscale at grid endpoint):", ends if ends else "clear")
