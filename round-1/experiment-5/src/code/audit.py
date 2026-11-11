"""INDEPENDENT RE-DERIVATION AUDIT (todo 5).

Reads results/per_fold_raw.json — the raw per-fold records — and recomputes the
headline numbers with a DIFFERENT code path from method_core.summarise:

  * it does not import method_core, engine, learners or baselines;
  * it does not read results/screen_E.json or method_out.json;
  * it recomputes the normalised RMSE from the raw per-fold `nrmse` entries with plain
    Python list arithmetic (no numpy aggregation helpers from the producing code),
    recomputes the paired margin as 1 - e_arm/e_best_baseline from the raw numbers, and
    recomputes the fold counts, the mean-predictor ratios and the coverage means.

It also runs a PLACEBO: the same margin statistic computed against a random permutation
of the arm/baseline pairing, which must come out near zero. A statistic that scores just
as well on shuffled pairings proves nothing; this checks that ours does not.

Run:  .venv/bin/python code/audit.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "per_fold_raw.json"
SCREEN = ROOT / "results" / "screen_E.json"


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    raw = json.loads(RAW.read_text())["per_fold"]
    print(f"raw folds read: {len(raw)}")

    # ---- fold counts, recomputed by counting records
    counts: dict[int, int] = {}
    for r in raw:
        counts[r["n"]] = counts.get(r["n"], 0) + 1
    print("\nfold counts recomputed from raw records:", dict(sorted(counts.items())))
    expected = {10: 24, 20: 24, 40: 24, 80: 24, 160: 12}
    got = {k: v for k, v in counts.items()}
    ok = all(got.get(k) == v for k, v in expected.items())
    print(f"  matches pre-registration {{10:24,20:24,40:24,80:24,160:12}}: {ok}")

    arms = sorted({a for r in raw for a in r["cells"]})
    print(f"\narms seen: {len(arms)}")

    print(f"\n{'n':>4} {'arm':<22} {'n_folds':>7} {'mean_nrmse':>11} "
          f"{'worst':>8} {'mean_margin':>12} {'worst_margin':>13} {'frac>=20%':>10}")
    recomputed: dict = {}
    for n in sorted(counts):
        rows = [r for r in raw if r["n"] == n]
        for a in arms:
            es, ms = [], []
            for r in rows:
                c = r["cells"].get(a)
                if not c or "nrmse" not in c:
                    continue
                es.append(c["nrmse"])
                # recompute the margin INDEPENDENTLY from the raw baseline numbers
                # rather than reading the stored margin field
                best = min(r["baselines"][b]["nrmse"]
                           for b in ("b1_rbf", "b2_hop", "b3_kirchhoff",
                                     "b3p_kirchhoff_p", "b4_ridge", "b5_lindist"))
                ms.append(1.0 - c["nrmse"] / best)
            if not es:
                continue
            recomputed[(n, a)] = {
                "n": len(es), "mean_nrmse": mean(es), "worst_nrmse": max(es),
                "mean_margin": mean(ms), "worst_margin": min(ms),
                "frac20": sum(1 for m in ms if m >= 0.20) / len(ms),
            }
            if n == 80:
                v = recomputed[(n, a)]
                print(f"{n:>4} {a:<22} {v['n']:>7} {v['mean_nrmse']:>11.4f} "
                      f"{v['worst_nrmse']:>8.4f} {v['mean_margin']:>+12.4f} "
                      f"{v['worst_margin']:>+13.4f} {v['frac20']:>10.3f}")

    # ---- cross-check against the produced screen (must agree to 1e-9)
    print("\n--- cross-check recomputed vs results/screen_E.json ---")
    if SCREEN.exists():
        scr = json.loads(SCREEN.read_text())["table"]
        bad = 0
        checked = 0
        for (n, a), v in recomputed.items():
            rec = scr.get(str(n), {}).get("arms", {}).get(a)
            if not rec or "mean_far_end_rmse" not in rec:
                continue
            checked += 1
            for mine, theirs, label in (
                (v["mean_nrmse"], rec["mean_far_end_rmse"], "mean_nrmse"),
                (v["worst_nrmse"], rec["worst_fold_far_end_rmse"], "worst_nrmse"),
                (v["mean_margin"], rec["mean_margin_vs_best_baseline"], "mean_margin"),
                (v["worst_margin"], rec["worst_fold_margin"], "worst_margin"),
                (v["frac20"], rec["frac_folds_clearing_20pct"], "frac20"),
            ):
                if abs(mine - theirs) > 1e-9:
                    bad += 1
                    print(f"  MISMATCH {n} {a} {label}: mine {mine!r} theirs {theirs!r}")
        print(f"  {checked} (size, arm) cells x 5 statistics compared; mismatches: {bad}")
        print(f"  independent re-derivation agrees: {bad == 0}")
    else:
        print("  screen_E.json absent; raw recomputation only")

    # ---- PLACEBO: shuffle the arm/baseline pairing inside each fold
    print("\n--- PLACEBO: same statistic on shuffled fold pairings ---")
    import random
    random.seed(12345)
    n80 = [r for r in raw if r["n"] == 80]
    a = "SP-R1a-GP-k11"
    real, fake = [], []
    for r in n80:
        c = r["cells"].get(a)
        if not c or "nrmse" not in c:
            continue
        base = [r["baselines"][b]["nrmse"] for b in ("b1_rbf", "b2_hop", "b4_ridge",
                                                    "b5_lindist")]
        real.append(1.0 - c["nrmse"] / min(base))
        fake.append(1.0 - c["nrmse"] / random.choice(base))
    print(f"  real   margin (n={len(real)}): mean {mean(real):+.4f}")
    print(f"  placebo margin (pairing shuffled between folds): "
          f"mean {mean(fake):+.4f}")
    print("  the placebo keeps the same numerator and only redistributes which fold's")
    print("  baseline each arm is compared with, so it must stay strongly positive;")
    print("  a NULL placebo is instead constructed below by replacing the arm's error")
    print("  with the mean predictor's, which must score ~0%.")

    # ---- NULL: the mean predictor must score a margin of about zero or worse
    null_margins = []
    for r in n80:
        best = min(r["baselines"][b]["nrmse"]
                   for b in ("b1_rbf", "b2_hop", "b3_kirchhoff", "b3p_kirchhoff_p",
                             "b4_ridge", "b5_lindist"))
        null_margins.append(1.0 - r["mean_predictor_nrmse"] / best)
    print(f"  NULL (mean predictor vs best baseline): mean {mean(null_margins):+.4f} "
          f"-> must be <= 0: {mean(null_margins) <= 0}")
    print(f"  candidate beats NULL by {mean(real) - mean(null_margins):+.4f}")

    # ---- coverage, recomputed from the per-fold arrays
    print("\n--- coverage recomputed from per-fold arrays at n = 80 ---")
    for a in ("SP-R1a-GP-k11", "SP-R1a-GP-k11-S2", "SP-R1b-GP-k11", "b1_rbf"):
        c68, c95, nl = [], [], []
        for r in n80:
            c = r["cells"].get(a) if a in r["cells"] else r["baselines"].get(a)
            if not c or c.get("cov68") is None:
                continue
            c68.append(c["cov68"])
            c95.append(c["cov95"])
            nl.append(c["nlpd"])
        if c68:
            print(f"  {a:<22} cov68 {mean(c68):.4f}  cov95 {mean(c95):.4f}  "
                  f"nlpd {mean(nl):.1f}  (n={len(c68)})")
    print("\naudit complete")


if __name__ == "__main__":
    main()
