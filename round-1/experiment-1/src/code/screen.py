"""The screen: every candidate and every canonical baseline, on identical folds.

One fold = one (n, seed, scheme) triple.  For each fold the runner computes, from the
same train rows and scored on the same test block:

    treatment   GP with the pullback metric S_far (45 x 110)
    b1          RBF GP on per-column-standardised injections
    b2          hop-count graph-Laplacian kernel (best of two forms)
    b3          admittance-weighted graph-Laplacian kernel
    b4          ridge on standardised injections, scale-free penalty grid
    b5          LinDistFlow order 2 / Z-bus correction, 16-cell train-only selection
    b5_legacy   the legacy cell, reported beside b5
    mean        the mean predictor (the guard-g1 reference)

Everything that can be tuned is tuned on train rows only.  The per-fold margin is a
PAIRED quantity: 1 - rmse_treatment / rmse_best_baseline_of_that_same_fold.
"""
from __future__ import annotations

import gc
import numpy as np

from kernels import gp_fit_predict, nlpd_coverage, sqdist
from baselines import (b1_rbf_injections, b2_hop_count, b3_admittance_laplacian,
                       b4_ridge, b5_lindistflow_grid, b5_legacy_cell)
from operators import build_operator

CANON = ("b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow")


def metric_features(S: np.ndarray, Xtr: np.ndarray, Xte: np.ndarray):
    """F(x) = S (x - mx) / sc with mx and sc from TRAIN ROWS ONLY."""
    mx = Xtr.mean(0)
    Ftr = (Xtr - mx) @ S.T
    Fte = (Xte - mx) @ S.T
    sc = np.sqrt(np.einsum("ij,ij->i", Ftr, Ftr).mean()) + 1e-12
    return Ftr / sc, Fte / sc, mx, sc


def norm_rmse(pred: np.ndarray, Y: np.ndarray, vsd: float) -> float:
    return float(np.sqrt(np.mean((pred - Y) ** 2)) / vsd)


def run_fold_metric(S: np.ndarray, X, Y, tr, te, vsd, want_var: bool = True,
                    select: str = "mll") -> dict:
    """The primary treatment only, with an arbitrary (3m x 110) operator S.

    Used by the mechanism factorial and the controls, so every cell goes through
    exactly the same feature construction and the same GP.
    """
    Ftr, Fte, mx, sc = metric_features(S, X[tr], X[te])
    r = gp_fit_predict(Ftr, Fte, Y[tr], want_var=want_var, select=select)
    rec = dict(rmse=norm_rmse(r["mu"], Y[te], vsd), ell=r["ell"], s2=r["s2"],
               n2=r["n2"], mll=r["mll"], feat_scale=sc)
    if want_var:
        rec.update(nlpd_coverage(Y[te], r["mu"], r["var"], Y[tr].std(0)))
    mu_mean = np.tile(Y[tr].mean(0), (len(te), 1))
    rec["mean_rmse"] = norm_rmse(mu_mean, Y[te], vsd)
    rec["rmse_over_mean"] = float(rec["rmse"] / rec["mean_rmse"])
    return rec


def run_fold(X, Y, tr, te, physics, targets, phase_labels, vsd,
             want_var: bool = True, do_b5: bool = True,
             do_graph: bool = True, select: str = "mll") -> dict:
    Xtr, Xte = X[tr], X[te]
    Ytr, Yte = Y[tr], Y[te]
    out = {"n_train": int(len(tr)), "n_test": int(len(te))}

    # ---- mean predictor (guard g1 reference)
    mu_mean = np.tile(Ytr.mean(0), (len(te), 1))
    out["mean"] = dict(rmse=norm_rmse(mu_mean, Yte, vsd))

    # ---- primary treatment
    S = build_operator(physics, targets, phase_labels, "pu_correct", "noload")
    Ftr, Fte, mx, sc = metric_features(S, Xtr, Xte)
    r = gp_fit_predict(Ftr, Fte, Ytr, want_var=want_var, select=select)
    rec = dict(rmse=norm_rmse(r["mu"], Yte, vsd), ell=r["ell"], s2=r["s2"],
               n2=r["n2"], mll=r["mll"], feat_scale=sc)
    if want_var:
        rec.update(nlpd_coverage(Yte, r["mu"], r["var"], Ytr.std(0)))
    out["treatment"] = rec

    # ---- b1
    r = b1_rbf_injections(Xtr, Xte, Ytr, want_var=want_var, select=select)
    rec = dict(rmse=norm_rmse(r["mu"], Yte, vsd), ell=r["ell"], s2=r["s2"], n2=r["n2"],
               mll=r["mll"])
    if want_var:
        rec.update(nlpd_coverage(Yte, r["mu"], r["var"], Ytr.std(0)))
    out["b1_rbf"] = rec

    # ---- b2 / b3 graph kernels
    if do_graph:
        r = b2_hop_count(physics, phase_labels, Xtr, Xte, Ytr, want_var=want_var,
                         select=select)
        rec = dict(rmse=norm_rmse(r["mu"], Yte, vsd), ell=r["ell"], s2=r["s2"],
                   n2=r["n2"], mll=r["mll"], form=r["form"],
                   forms_mll=r["forms_mll"])
        if want_var:
            rec.update(nlpd_coverage(Yte, r["mu"], r["var"], Ytr.std(0)))
        out["b2_hop"] = rec

        r = b3_admittance_laplacian(physics, phase_labels, Xtr, Xte, Ytr,
                                   want_var=want_var, select=select)
        rec = dict(rmse=norm_rmse(r["mu"], Yte, vsd), ell=r["ell"], s2=r["s2"],
                   n2=r["n2"], mll=r["mll"])
        if want_var:
            rec.update(nlpd_coverage(Yte, r["mu"], r["var"], Ytr.std(0)))
        out["b3_adm"] = rec

    # ---- b4
    r = b4_ridge(Xtr, Xte, Ytr, want_var=want_var)
    rec = dict(rmse=norm_rmse(r["mu"], Yte, vsd), lam=r["lam"], lam_max=r["lam_max"])
    if want_var:
        rec.update(nlpd_coverage(Yte, r["mu"], r["var"], Ytr.std(0)))
    out["b4_ridge"] = rec

    # ---- b5, the reconciliation
    if do_b5:
        r = b5_lindistflow_grid(physics, Xtr, Xte, Ytr, targets, phase_labels, {})
        out["b5_lindistflow"] = dict(
            rmse=norm_rmse(r["pred"], Yte, vsd), cell=r["cell"],
            cv_rmse=r["cv_rmse"],
            all_cells_rmse={k: norm_rmse(v, Yte, vsd)
                            for k, v in r["all_cells_pred"].items()},
            all_cells_cv=r["all_cells_cv"])
        rl = b5_legacy_cell(physics, Xtr, Xte, Ytr, targets, phase_labels)
        out["b5_legacy"] = dict(rmse=norm_rmse(rl["pred"], Yte, vsd), cell=rl["cell"])

    # ---- paired comparison against the canonical pool
    pool = {k: out[k]["rmse"] for k in CANON if k in out}
    best_name = min(pool, key=pool.get)
    out["best_baseline_name"] = best_name
    out["best_baseline_rmse"] = float(pool[best_name])
    out["margin_vs_best"] = float(1.0 - out["treatment"]["rmse"] / pool[best_name])
    if "b5_lindistflow" in out:
        out["margin_vs_b5"] = float(1.0 - out["treatment"]["rmse"] /
                                    out["b5_lindistflow"]["rmse"])
    out["guard_g1_treatment"] = float(out["treatment"]["rmse"] / out["mean"]["rmse"])
    out["guard_g1_b1"] = float(out["b1_rbf"]["rmse"] / out["mean"]["rmse"])
    return out


def summarise_folds(folds: list[dict]) -> dict:
    """Aggregate a list of per-fold records for one (regime, n)."""
    tr = np.array([f["treatment"]["rmse"] for f in folds])
    marg = np.array([f["margin_vs_best"] for f in folds])
    mg5 = np.array([f["margin_vs_b5"] for f in folds if "margin_vs_b5" in f])
    out = dict(
        n_folds=len(folds),
        mean_far_end_rmse=float(tr.mean()),
        worst_fold_far_end_rmse=float(tr.max()),
        std_far_end_rmse=float(tr.std()),
        mean_rmse_over_mean_predictor=float(np.mean([f["guard_g1_treatment"] for f in folds])),
        mean_margin_vs_best_baseline=float(marg.mean()),
        worst_fold_margin=float(marg.min()),
        std_fold_margin=float(marg.std(ddof=1)) if len(marg) > 1 else 0.0,
        frac_folds_clearing_20pct=float(np.mean(marg >= 0.20)),
        mean_nlpd=float(np.mean([f["treatment"]["nlpd"] for f in folds
                                 if "nlpd" in f["treatment"]])),
        mean_coverage68=float(np.mean([f["treatment"]["coverage68"] for f in folds
                                       if "coverage68" in f["treatment"]])),
        mean_coverage95=float(np.mean([f["treatment"]["coverage95"] for f in folds
                                       if "coverage95" in f["treatment"]])),
        best_baseline_counts={},
    )
    for f in folds:
        b = f["best_baseline_name"]
        out["best_baseline_counts"][b] = out["best_baseline_counts"].get(b, 0) + 1
    if len(mg5):
        out["mean_margin_vs_lindistflow"] = float(mg5.mean())
        out["worst_fold_margin_vs_lindistflow"] = float(mg5.min())
    for k in CANON:
        vals = [f[k]["rmse"] for f in folds if k in f]
        if vals:
            out[f"mean_rmse_{k}"] = float(np.mean(vals))
            out[f"guard_g1_{k}"] = float(np.mean(vals) /
                                         np.mean([f["mean"]["rmse"] for f in folds]))
    out["per_fold_rmse_treatment"] = [float(v) for v in tr]
    out["per_fold_margin"] = [float(v) for v in marg]
    out["per_fold_margin_vs_b5"] = [float(v) for v in mg5]
    out["per_fold_rmse_baselines"] = {
        k: [float(f[k]["rmse"]) for f in folds if k in f] for k in
        ("b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow", "b5_legacy", "mean")
    }
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 424242) -> dict:
    """Paired bootstrap over folds (resampling folds, NOT data: all folds share draws)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return dict(mean=float(x.mean()),
                lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)),
                n_boot=int(n_boot),
                resampling_unit="folds")


def wilcoxon_p(x: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on the paired per-fold differences."""
    from scipy.stats import wilcoxon
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0):
        return dict(statistic=0.0, p_value=1.0, n=int(len(x)), note="all differences zero")
    try:
        s = wilcoxon(x, zero_method="wilcox", alternative="greater")
        return dict(statistic=float(s.statistic), p_value=float(s.pvalue), n=int(len(x)))
    except ValueError as e:
        return dict(statistic=float("nan"), p_value=float("nan"), n=int(len(x)),
                    note=str(e))


def crossover_curve(summary_by_n: dict) -> dict:
    """Two crossover numbers, each reported as the interval it lies in."""
    ns = sorted(summary_by_n)
    out = {}
    for tag, key in (("vs_best", "mean_margin_vs_best_baseline"),
                     ("vs_lindistflow", "mean_margin_vs_lindistflow")):
        vals = [summary_by_n[n].get(key) for n in ns]
        found = None
        for i, v in enumerate(vals):
            if v is not None and v > 0:
                found = i
                break
        if found is None:
            out[f"crossover_{tag}"] = dict(interval=None, note="never positive on this grid")
        elif found == 0:
            out[f"crossover_{tag}"] = dict(interval=[None, ns[0]],
                                           note=f"already positive at n={ns[0]}")
        else:
            out[f"crossover_{tag}"] = dict(interval=[ns[found - 1], ns[found]])
        out[f"mean_margin_curve_{tag}"] = {int(n): (None if v is None else float(v))
                                           for n, v in zip(ns, vals)}
    every = None
    for n in ns:
        if summary_by_n[n]["frac_folds_clearing_20pct"] >= 1.0:
            every = n
            break
    out["crossover_everyfold20"] = dict(value=int(every) if every is not None else None,
                                        grid=ns)
    out["frac_clearing_20pct_curve"] = {int(n): float(summary_by_n[n]["frac_folds_clearing_20pct"])
                                        for n in ns}
    return out
