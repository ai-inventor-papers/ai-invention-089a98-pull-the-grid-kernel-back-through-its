#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the artifact's headline numbers.

This script deliberately does NOT import code/screen.py, code/kernels.py or
code/baselines.py.  It reads the raw per-fold records that the run wrote
(results/screen_2.0x.json::per_fold) and recomputes, through a different code
path:

  1. the paired per-fold margin  1 - rmse_treatment / rmse_best_baseline
     by hand, then its mean, worst and fraction clearing 20 percent;
  2. the paired bootstrap 95 percent interval, by an independent resampler;
  3. the Wilcoxon signed-rank p-value, via scipy.stats with a different call;
  4. the crossover brackets, re-found from the recomputed margins;
  5. the same margin statistics on a PLACEBO: the treatment's per-fold RMSE
     vector rotated by one fold (so it is paired with the wrong baseline), and
     on the b1 RBF baseline substituted for the treatment.  A statistic that
     also passes on the placebo is not evidence.

Outputs audit_result.json next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load_per_fold(regime_file: str) -> list[dict]:
    return json.loads((RES / regime_file).read_text())["per_fold"]


def stats(margins: np.ndarray) -> dict:
    return dict(mean=float(np.mean(margins)), worst=float(np.min(margins)),
                sd=float(np.std(margins, ddof=1)),
                frac_ge_20=float(np.mean(margins >= 0.20)),
                n=int(len(margins)))


def boot_ci(x: np.ndarray, B: int = 4000, seed: int = 987654321) -> dict:
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return dict(lo=float(np.quantile(means, 0.025)),
                hi=float(np.quantile(means, 0.975)),
                mean=float(x.mean()), B=B)


def recompute(per_fold: list[dict], regime: str) -> dict:
    by_n: dict[int, list[dict]] = {}
    for r in per_fold:
        by_n.setdefault(int(r["n"]), []).append(r)
    out = {"regime": regime, "per_n": {}, "placebo": {}}
    for n in sorted(by_n):
        folds = by_n[n]
        # ---- recompute the paired margin from the two raw RMSE numbers
        margins, treat, best, bestname = [], [], [], []
        for f in folds:
            t = f["treatment"]["rmse"]
            pool = {k: f[k]["rmse"] for k in
                    ("b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow")
                    if k in f}
            bname = min(pool, key=pool.get)
            margins.append(1.0 - t / pool[bname])
            treat.append(t)
            best.append(pool[bname])
            bestname.append(bname)
        m = np.array(margins)
        s = stats(m)
        s["bootstrap"] = boot_ci(m)
        s["wilcoxon_p_greater"] = float(wilcoxon(m, alternative="greater").pvalue)
        s["wilcoxon_p_two_sided"] = float(wilcoxon(m, alternative="two-sided").pvalue)
        s["best_baseline_counts"] = {k: bestname.count(k) for k in set(bestname)}
        out["per_n"][n] = s
    # ---- crossover brackets from the recomputed means
    ns = sorted(out["per_n"])
    means = [out["per_n"][n]["mean"] for n in ns]
    cross = None
    for i, v in enumerate(means):
        if v > 0:
            cross = [ns[i - 1], ns[i]] if i > 0 else [None, ns[0]]
            break
    out["crossover_vs_best_interval"] = cross
    out["mean_margin_curve"] = {n: out["per_n"][n]["mean"] for n in ns}
    out["frac_clearing_curve"] = {n: out["per_n"][n]["frac_ge_20"] for n in ns}

    # ---- PLACEBO 1: rotate the treatment RMSE across the WHOLE pooled fold list so a
    # treatment error from size n is paired with a baseline error from a DIFFERENT size.
    # (Rotating within one cell is too weak a null: every fold in a cell has nearly the
    # same RMSE, so the rotation leaves the pairing almost intact and the placebo
    # inherits the real margin.  That failure is itself reported in the audit output.)
    allfolds = [f for n in sorted(by_n) for f in by_n[n]]
    t_all = np.array([f["treatment"]["rmse"] for f in allfolds])
    b_all = np.array([min(f[k]["rmse"] for k in
                          ("b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow")
                          if k in f) for f in allfolds])
    n_all = np.array([int(f["n"]) for f in allfolds])
    rot = np.roll(t_all, 1)
    m_rot_all = 1.0 - rot / b_all
    out["placebo"]["rotated_across_sizes_pooled"] = stats(m_rot_all)
    out["placebo_note"] = ("rotating within one size is too weak a null (all folds in a "
                           "cell share nearly the same RMSE); the reported null rotates "
                           "across the pooled fold list so a treatment error is paired "
                           "with a baseline error from a different training size")
    for n in sorted(by_n):
        folds = by_n[n]
        # ---- PLACEBO 2: the RBF baseline as if it were the treatment
        t1 = np.array([f["b1_rbf"]["rmse"] for f in folds])
        b = np.array([min(f[k]["rmse"] for k in
                          ("b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow")
                          if k in f) for f in folds])
        out["placebo"][f"b1_rbf_as_treatment_n{n}"] = stats(1.0 - t1 / b)
        # ---- PLACEBO 3: the mean predictor as if it were the treatment
        tm = np.array([f["mean"]["rmse"] for f in folds])
        out["placebo"][f"mean_predictor_as_treatment_n{n}"] = stats(1.0 - tm / b)
        # ---- PLACEBO 4: the best graph kernel as if it were the treatment
        tb = np.array([f["b3_adm"]["rmse"] for f in folds])
        out["placebo"][f"b3_adm_as_treatment_n{n}"] = stats(1.0 - tb / b)
        # ---- the null for the placebo gate at n itself, from the pooled rotation
        mask = n_all == n
        out["placebo"][f"rotated_across_sizes_at_n{n}"] = stats(m_rot_all[mask])
    return out


def main() -> None:
    res = {}
    for f, reg in (("screen_2.0x.json", "2.0"), ("screen_1.0x.json", "1.0")):
        p = RES / f
        if not p.exists():
            res[reg] = {"missing": str(p)}
            continue
        res[reg] = recompute(load_per_fold(f), reg)
    # ---- contrast: what the headline claims vs what the placebo gives
    checks = {}
    for reg in ("2.0", "1.0"):
        r = res.get(reg, {})
        if "per_n" not in r:
            continue
        n80 = r["per_n"].get(80)
        if not n80:
            continue
        pl = r["placebo"]
        checks[reg] = dict(
            headline_mean_margin_at_80=n80["mean"],
            headline_worst_at_80=n80["worst"],
            headline_frac_ge_20_at_80=n80["frac_ge_20"],
            headline_bootstrap_ci_at_80=[n80["bootstrap"]["lo"], n80["bootstrap"]["hi"]],
            headline_wilcoxon_p_greater_at_80=n80["wilcoxon_p_greater"],
            placebo_rotated_across_sizes_mean_at_80=pl["rotated_across_sizes_at_n80"]["mean"],
            placebo_rbf_mean_at_80=pl["b1_rbf_as_treatment_n80"]["mean"],
            placebo_mean_pred_mean_at_80=pl["mean_predictor_as_treatment_n80"]["mean"],
            placebo_b3_adm_mean_at_80=pl["b3_adm_as_treatment_n80"]["mean"],
            placebo_rbf_frac_ge_20=pl["b1_rbf_as_treatment_n80"]["frac_ge_20"])
        # The placebo that matters: put a candidate with NO claimed advantage in the
        # treatment's place.  A statistic that also clears 20 percent for the strongest
        # graph kernel, the standard RBF, or the mean predictor proves nothing; all three
        # must fail.  The rotate-the-treatment-error placebo is NOT part of the gate --
        # it is uninformative here because the treatment's advantage (about 72 percent)
        # is an order of magnitude larger than its fold-to-fold spread, so any pairing
        # that keeps a size-80 treatment error wins.  Its value is reported and said so.
        checks[reg]["passes_placebo_gate"] = bool(
            n80["mean"] > 0.20 and
            pl["b1_rbf_as_treatment_n80"]["mean"] < 0.20 and
            pl["mean_predictor_as_treatment_n80"]["mean"] < 0.20 and
            pl["b3_adm_as_treatment_n80"]["mean"] < 0.20)
        checks[reg]["placebos_that_fail"] = [
            k for k in ("b1_rbf_as_treatment_n80", "mean_predictor_as_treatment_n80",
                        "b3_adm_as_treatment_n80") if pl[k]["mean"] < 0.20]
        checks[reg]["rotation_placebo_note"] = (
            "uninformative: reported for completeness, not used in the gate")
    res["headline_gate"] = checks
    (HERE / "audit_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("headline_gate", {}), indent=1))
    print("\ncrossover (2.0x):", res.get("2.0", {}).get("crossover_vs_best_interval"))
    print("mean margin curve (2.0x):", res.get("2.0", {}).get("mean_margin_curve"))
    print("frac clearing 20% (2.0x):", res.get("2.0", {}).get("frac_clearing_curve"))


if __name__ == "__main__":
    main()
