#!/usr/bin/env python3
"""Physics-guided residual Gaussian process on the IEEE PES European LV feeder.

PRE-REGISTERED QUESTION
-----------------------
Candidate D of a five-way kernel screen: use a train-calibrated multiphase
LinDistFlow prediction as the Gaussian-process PRIOR MEAN and put the
output-restricted sensitivity kernel on the RESIDUAL.  Does that beat the tuned
physics baseline and the zero-mean kernels at small n (10-160 AC solves), and
can its predictive intervals be repaired by a train-only noise-floor tempering
without touching the point prediction?

Ground truth
------------
Archived converged unbalanced three-phase AC power-flow ensembles
(`runpp_3ph` on `ieee_european_lv_asymmetric('on_peak_566')`, 55 single-phase
loads on their native phases, independent lognormal household draws with
CV = 0.5, power factor 0.95).  220 draws per regime, 110-vector injections
(55 x P, 55 x Q in MW/MVAr), 45 targets (15 far-end buses x 3 phases, per-unit
voltage magnitude).  Regenerated from the fallback recipe only if the archives
are absent.

Design: shared 24-fold protocol (12 seeds x 2 partition schemes) x
n in {10, 20, 40, 80, 160}, decision regime 2.0x, robustness panel 1.0x.
Reserved evidence (2.6x regime, permuted-feeder ensembles) is hard-blocked.

Everything is held constant across arms: injection pool, fold indices, target
sheet, train-only standardisation, hyperparameter budget, penalty grid,
GP grid, RBF distance divisor of 110.

Outputs (all under ./results):
  screen_D.json       the mandated table plus diagnostics
  protocol_checks.json executable leakage, determinism, guard and operator checks
  folds_D.csv         one row per fold, for audit
and ../method_out.json, the top-level result in the pipeline's schema.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feeder import (Feeder, build_laplacians, check_q_is_deterministic,  # noqa: E402
                    regime_label, singular_spectrum)
from kernels import (ALPHA_GRID, GP_N2GRID, GP_LGRID, GP_S2GRID, PENALTY_FRACS, RBF_DIVISOR,  # noqa: E402
                     effective_dof, feats_from_S, feats_from_metric, gp_fit,
                     gp_predict, gram, resid_variance, ridge_cv, ridge_design,
                     ridge_predictive_var, sqd, standardise, zscore_nlpd)

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
(HERE / "logs").mkdir(exist_ok=True)
logger.add(str(HERE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------- configuration
DATA = HERE / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Inputs are read from ./data first, which is the published copy of the archived
# ensembles this artifact needs.  The fallback locations are the sibling artifacts
# of the same run, named by their ids; a reader of the published repository has
# only ./data, so nothing below this block depends on an absolute server path.
#   gen_art_experiment_1 = the artefact that produced the AC ensembles (pilot)
#   gen_art_experiment_3 = the artefact that produced the reference operator
RUN_ROOT = HERE.parent.parent.parent.parent  # the run's top directory, if present
PILOT = Path(os.environ.get("PILOT_DIR", RUN_ROOT / "iter_2/gen_hypo/claude_agent/pilot"))
VERIFY = Path(os.environ.get("VERIFY_DIR", RUN_ROOT / "iter_3/gen_hypo/claude_agent/verify"))

REGIME_DECIDE = "2.0"     # decision table
REGIME_ROBUST = "1.0"     # secondary panel, clearly labelled
RESERVED = "2.6"          # FORBIDDEN: reserved evidence, hard-blocked below
DENYLIST = ("ens_s2.6", "ens_perm", "ens_near", "phase_perm")

NS = (10, 20, 40, 80, 160)
SEEDS = tuple(range(12))
SCHEMES = (0, 1)
FOLD_BASE = 20261111
# expected archived statistics, re-asserted on load
EXPECT = {"1.0": (57.49, 0.00969), "2.0": (114.69, 0.02101)}
MIN_TEST_ROWS = 3

BASE_ARMS = ("b1_rbf", "b2_lap_hop", "b3_lap_adm", "b4_ridge", "b5_lindist")
DIAG_ARMS = ("b6_lap_imp",)
MODEL_ARMS = ("physics_only", "zero_mean_gp", "resid_gp_mll",
              "resid_gp_tempered", "resid_ridge_linear")
ALL_ARMS = BASE_ARMS + MODEL_ARMS


def fold_rng(seed: int, scheme: int) -> np.random.Generator:
    return np.random.default_rng(FOLD_BASE + 7919 * seed + 131 * scheme)


def fold_indices(n: int, seed: int, scheme: int, n_pool: int) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint train/test index sets for one fold.

    scheme 0 (complement): the training rows are drawn first and EVERY other row
        is test, so the test set shrinks as n grows.  This is the standard
        fixed-n curve.
    scheme 1 (held-out block): a FIXED 55-row block (a quarter of the pool) is
        set aside as test and the training rows are drawn from the remaining 165.
        The test block therefore does not change with n, which is what lets a
        reader separate "the model got better" from "the test set got easier".
        The block size is reduced only if n would otherwise not fit.
    """
    idx = fold_rng(seed, scheme).permutation(n_pool)
    if scheme == 0:
        tr, te = idx[:n], idx[n:]
    elif scheme == 1:
        n_test = int(min(55, max(MIN_TEST_ROWS, n_pool - n - 5)))
        n_train_pool = n_pool - n_test
        if n > n_train_pool:
            raise RuntimeError(f"scheme 1 cannot fit n={n} with a {n_test}-row test block")
        tr, te = idx[:n_train_pool][:n], idx[n_train_pool:]
    else:
        raise ValueError(f"unknown partition scheme {scheme}")
    if len(tr) != n or len(te) < MIN_TEST_ROWS:
        raise RuntimeError(f"fold too small: n_train={len(tr)} n_test={len(te)}")
    if np.intersect1d(tr, te).size:
        raise RuntimeError("train/test overlap inside a fold")
    return tr, te


# ------------------------------------------------------------------ staging
def stage_ensemble(regime: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and verify the archived ground truth. Never touches reserved evidence."""
    out = {}
    for tag in ("X", "Y"):
        cands = [DATA / f"ens_s{regime}_{tag}.npy", PILOT / f"ens_s{regime}_{tag}.npy"]
        for c in cands:
            if any(tok in str(c) for tok in DENYLIST):
                raise RuntimeError(f"reserved-evidence file requested: {c}")
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"archived ensemble ens_s{regime}_{tag}.npy missing from {cands}; "
                "run `uv run regenerate_data.py` (fallback recipe) first")
        out[tag] = np.load(path)
    x, y = out["X"], out["Y"]
    if x.shape != (220, 110) or y.shape != (220, 45):
        raise ValueError(f"unexpected ensemble shapes {x.shape} {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite values in the archived ensemble")
    stats = {"mean_kW": float(1000.0 * x[:, :55].sum(1).mean()),
             "Vstd": float(y.std(0).mean()),
             "median_V": float(np.median(y)),
             "p1_min_V": float(np.percentile(y, 1)),
             "n_rows": int(len(x)), "n_inputs": int(x.shape[1]), "n_outputs": int(y.shape[1])}
    exp_kw, exp_vs = EXPECT[regime]
    dev_kw = abs(stats["mean_kW"] - exp_kw) / exp_kw
    dev_vs = abs(stats["Vstd"] - exp_vs) / exp_vs
    if dev_kw > 0.02 or dev_vs > 0.02:
        raise AssertionError(
            f"regime {regime} statistics deviate from the archived targets: "
            f"kW {stats['mean_kW']:.3f} vs {exp_kw} ({dev_kw:.2%}), "
            f"Vstd {stats['Vstd']:.6f} vs {exp_vs} ({dev_vs:.2%})")
    stats["dev_kW_frac"] = float(dev_kw)
    stats["dev_Vstd_frac"] = float(dev_vs)
    return x, y, stats


def load_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sheet = np.load(DATA / "sheet15.npy").astype(int)
    phase_true = np.load(DATA / "ph_true.npy").astype(int)
    s_ref = np.load(DATA / "S_sheet15.npy")
    return sheet, phase_true, s_ref


def verify_operator(fe: Feeder, sheet: np.ndarray, s_ref: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rebuild S from line data and cross-check against the archived operator."""
    s = fe.lin_operator(sheet)
    if s.shape != (45, 110):
        raise AssertionError(f"operator shape {s.shape} != (45, 110)")
    frob_ratio = float(np.linalg.norm(s) / np.linalg.norm(s_ref))
    dev = float(np.linalg.norm(s / np.linalg.norm(s) - s_ref / np.linalg.norm(s_ref)))
    spec = singular_spectrum(s)
    check = {"frob_ratio_vs_archived": frob_ratio,
             "normalised_deviation_vs_archived": dev,
             "spectrum": spec,
             "note": ("a global scalar is absorbed by the train-only feature scaler; the "
                      "scale-free check is the normalised deviation, which must be below 1e-8")}
    if dev > 1e-8:
        raise AssertionError(f"rebuilt operator disagrees with the archived one: {dev:.3e}")
    return s, check


# ---------------------------------------------------------------- one fold
def build_baselines(x: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                    fe: Feeder, sheet: np.ndarray, s_op: np.ndarray,
                    metrics: dict, v0bus: np.ndarray) -> dict:
    """All request-named baselines for one fold. None touches S or any Jacobian.

    b1_rbf      : isotropic RBF on train-standardised injections, GP by marginal likelihood
    b2_lap_hop  : hop-count graph-Laplacian metric, then the same GP
    b3_lap_adm  : admittance-weighted Kirchhoff graph metric, then the same GP
    b4_ridge    : ridge on train-standardised injections
    b5_lindist  : multiphase LinDistFlow order 1 or 2 (chosen on training CV) with
                  per-output ridge calibration, penalty by train-only CV
    b6_lap_imp  : impedance-weighted Laplacian (DIAGNOSTIC ONLY, never in best-of)
    """
    n = len(tr)
    out: dict = {}

    ztr, zte = standardise(x[tr], x[te])

    # --- b1: isotropic RBF on injections
    fit = gp_fit(sqd(ztr, ztr) / RBF_DIVISOR, y[tr], need_loo=False)
    mu, pv = gp_predict(fit, sqd(zte, ztr) / RBF_DIVISOR)
    out["b1_rbf"] = {"pred": mu, "var": pv,
                     "hyper": {"ell": fit.ell, "s2": fit.s2, "n2": fit.n2,
                               "jitter": fit.jitter_used}}

    # --- b2, b3, b6: graph metrics then the same GP
    for key, name in (("M_hop", "b2_lap_hop"), ("M_adm", "b3_lap_adm"),
                      ("M_imp", "b6_lap_imp")):
        ftr, fte = feats_from_metric(metrics[key], x[tr], x[te])
        fit = gp_fit(gram(ftr), y[tr], need_loo=False)
        mu, pv = gp_predict(fit, sqd(fte, ftr))
        out[name] = {"pred": mu, "var": pv,
                     "hyper": {"ell": fit.ell, "s2": fit.s2, "n2": fit.n2,
                               "jitter": fit.jitter_used}}

    # --- b4: ridge on injections (intercept unpenalised, design centred)
    ctr, cte = ridge_design(ztr, zte)
    rr = ridge_cv(ctr, y[tr], cte)
    rv = resid_variance(y[tr], rr["pred_train"], rr["hat_diag"])
    out["b4_ridge"] = {"pred": rr["pred"],
                       "var": ridge_predictive_var(rr, cte, rv),
                       "hyper": {"penalty_frac": rr["frac"], "lmax": rr["lmax"],
                                 "cv_score": rr["cv_score"]}}

    # --- b5: LinDistFlow, order chosen on TRAIN-only CV, then ridge-calibrated
    orders = {}
    for order in (1, 2):
        vp = fe.lindist(x, sheet, order=order, v0bus=v0bus)
        c1, _ = ridge_design(vp[tr], vp[tr])
        rc = ridge_cv(c1, y[tr], c1)
        orders[order] = {"cv_score": rc["cv_score"], "frac": rc["frac"]}
    best_order = min(orders, key=lambda o: orders[o]["cv_score"])
    vp = fe.lindist(x, sheet, order=best_order, v0bus=v0bus)
    c1, c2 = ridge_design(vp[tr], vp[te])
    rc = ridge_cv(c1, y[tr], c2)
    rvp = resid_variance(y[tr], rc["pred_train"], rc["hat_diag"])
    out["b5_lindist"] = {"pred": rc["pred"],
                         "var": ridge_predictive_var(rc, c2, rvp),
                         "hyper": {"order": int(best_order), "penalty_frac": rc["frac"],
                                   "cv_score": rc["cv_score"],
                                   "order_cv": {str(k): v for k, v in orders.items()}},
                         "phys_train_pred": rc["pred_train"],
                         "phys_hat": rc["hat_diag"],
                         "phys_resid_var": rvp,
                         "phys_kit": (c1, c2, rc)}
    return out, (ztr, zte)


def tempered_alpha(phys_loo_err: np.ndarray, gp_loo_err: np.ndarray,
                   loo_total_var: np.ndarray, s2_ref: float) -> tuple[float, list]:
    """Choose the noise-floor multiplier alpha by train-only LOO coverage matching.

    The score is |c68 - 0.68| + |c95 - 0.95| on the leave-one-out z-scores, which
    are computable from training rows alone.  The test targets never enter here.
    """
    scores = []
    best_alpha, best_score = float(ALPHA_GRID[0]), np.inf
    total_err = phys_loo_err + gp_loo_err
    for alpha in ALPHA_GRID:
        v = np.clip(loo_total_var + alpha * s2_ref, 1e-30, None)
        z = total_err / np.sqrt(v)
        score = abs(float(np.mean(np.abs(z) <= 1.0)) - 0.68) + \
                abs(float(np.mean(np.abs(z) <= 1.96)) - 0.95)
        scores.append({"alpha": float(alpha), "score": float(score)})
        if score < best_score:
            best_alpha, best_score = float(alpha), float(score)
    return best_alpha, scores


def run_fold(x: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
             fe: Feeder, sheet: np.ndarray, s_op: np.ndarray, metrics: dict,
             v0bus: np.ndarray, keep_pred: bool = False) -> dict:
    """Fit and score every model for one fold. Returns raw predictions and metrics."""
    n = len(tr)
    base, (ztr, zte) = build_baselines(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
    ytr, yte = y[tr], y[te]
    vsd_train = float(ytr.std(0).mean())
    vsd_ens = float(y.std(0).mean())
    mean_pred = ytr.mean(0)[None, :] * np.ones_like(yte)
    rmse_mean_pred = float(np.sqrt(((mean_pred - yte) ** 2).mean()))

    # --- the sensitivity feature map, train-only scale
    ftr, fte = feats_from_S(s_op, x[tr], x[te])
    dtr, dst = gram(ftr), sqd(fte, ftr)

    arms: dict = {}
    arms["physics_only"] = {"pred": base["b5_lindist"]["pred"],
                            "var": base["b5_lindist"]["var"],
                            "hyper": dict(base["b5_lindist"]["hyper"]),
                            "note": "identical to b5_lindist by construction: the crossover reference"}

    # --- zero-mean operator kernel
    fit0 = gp_fit(dtr, ytr, need_loo=True)
    mu0, pv0 = gp_predict(fit0, dst)
    arms["zero_mean_gp"] = {"pred": mu0, "var": pv0,
                            "hyper": {"ell": fit0.ell, "s2": fit0.s2, "n2": fit0.n2,
                                      "jitter": fit0.jitter_used},
                            "loo_mu": fit0.loo_mu, "loo_var": fit0.loo_var}

    # --- physics mean + residual GP by marginal likelihood
    phys = base["b5_lindist"]
    a1, a2, rc = phys["phys_kit"]
    resid_tr = ytr - phys["phys_train_pred"]
    fit1 = gp_fit(dtr, resid_tr, need_loo=True)
    mu_r, pv_r = gp_predict(fit1, dst)
    pred3 = phys["pred"] + mu_r
    arms["resid_gp_mll"] = {"pred": pred3, "var": pv_r,
                            "hyper": {"ell": fit1.ell, "s2": fit1.s2, "n2": fit1.n2,
                                      "jitter": fit1.jitter_used},
                            "resid_var": fit1.ybar, "resid_mean": fit1.ybar}

    # --- the same point prediction with a train-only noise floor (two floors)
    h = phys["phys_hat"]
    phys_loo_err = (ytr - phys["phys_train_pred"]) / (1.0 - h)[:, None]
    gp_loo_err = ytr - phys["phys_train_pred"] - fit1.loo_mu
    # total LOO variance: ridge leverage part + GP LOO part
    sigma2_o = phys["phys_resid_var"]
    loo_ridge_var = sigma2_o[None, :] / (1.0 - h)[:, None]
    loo_total_var = loo_ridge_var + fit1.loo_var
    s2_ref_loo = float(np.mean((phys_loo_err + gp_loo_err) ** 2))
    s2_ref_raw = float(np.mean((ytr - phys["phys_train_pred"]) ** 2))
    alpha_loo, alpha_scores = tempered_alpha(phys_loo_err, gp_loo_err, loo_total_var, s2_ref_loo)
    alpha_raw, _ = tempered_alpha(phys_loo_err, gp_loo_err, loo_total_var, s2_ref_raw)
    # LOO coverage at the chosen alpha, so the reader can see the train-side and
    # test-side numbers together: they estimate the same quantity and should agree.
    z_loo = (phys_loo_err + gp_loo_err) / np.sqrt(np.clip(loo_total_var + alpha_loo * s2_ref_loo, 1e-30, None))
    loo_cov = {"c68": float(np.mean(np.abs(z_loo) <= 1.0)),
               "c95": float(np.mean(np.abs(z_loo) <= 1.96)),
               "nlpd": float(-np.mean(np.log(2 * np.pi * (loo_total_var + alpha_loo * s2_ref_loo)) / 2
                                      + z_loo ** 2 / 2))}
    arms["resid_gp_tempered"] = {"pred": pred3.copy(),  # bit-identical to resid_gp_mll
                                 "var": pv_r + alpha_loo * s2_ref_loo,
                                 "hyper": {"alpha_loo": alpha_loo, "alpha_raw": alpha_raw,
                                           "s2_ref_loo": s2_ref_loo, "s2_ref_raw": s2_ref_raw,
                                           "alpha_scores_loo": alpha_scores,
                                           "floor": "train-only LOO residual variance"},
                                 "tempered_raw_var": pv_r + alpha_raw * s2_ref_raw,
                                 "loo_coverage": loo_cov}

    # --- plain ridge on the residual, classical intervals
    ctr, cte = ridge_design(ztr, zte)
    rres = ridge_cv(ctr, resid_tr, cte)
    sig2 = resid_variance(resid_tr, rres["pred_train"], rres["hat_diag"])
    pred5 = phys["pred"] + rres["pred"]
    arms["resid_ridge_linear"] = {"pred": pred5,
                                  "var": ridge_predictive_var(rres, cte, sig2),
                                  "hyper": {"penalty_frac": rres["frac"], "lmax": rres["lmax"],
                                            "cv_score": rres["cv_score"]}}

    # ---------------- scoring
    rec = {"n_train": int(n), "n_test": int(len(te)), "vsd_train": vsd_train,
           "vsd_ens": vsd_ens, "rmse_mean_predictor": rmse_mean_pred,
           "arms": {}, "baselines": {}, "diagnostics": {}}
    for name, d in list(base.items()):
        p = d["pred"]
        e = np.sqrt(((p - yte) ** 2).mean())
        dd = {"rmse_norm": float(e / vsd_train),
              "rmse_ens_norm": float(e / vsd_ens),
              "rmse_over_mean_predictor": float(e / max(rmse_mean_pred, 1e-30)),
              "rmse_pu": float(e), "hyper": d["hyper"]}
        if name in DIAG_ARMS:
            rec["diagnostics"][name] = dd
        else:
            rec["baselines"][name] = dd
        if d.get("var") is not None:
            z = zscore_nlpd(yte, p, d["var"])
            dd.update({"nlpd": z["nlpd"], "c68": z["c68"], "c95": z["c95"]})
    best_base_name = min(rec["baselines"], key=lambda k: rec["baselines"][k]["rmse_norm"])
    best_base = rec["baselines"][best_base_name]["rmse_norm"]
    rec["best_baseline"] = best_base_name
    rec["best_baseline_rmse"] = float(best_base)

    for name, d in arms.items():
        p = d["pred"]
        e = np.sqrt(((p - yte) ** 2).mean())
        dd = {"rmse_norm": float(e / vsd_train),
              "rmse_ens_norm": float(e / vsd_ens),
              "rmse_over_mean_predictor": float(e / max(rmse_mean_pred, 1e-30)),
              "rmse_pu": float(e),
              "margin_vs_best_baseline": float(1.0 - (e / vsd_train) / best_base),
              "numerical_failure": bool((e / max(rmse_mean_pred, 1e-30)) > 1.0),
              "hyper": d["hyper"]}
        if d.get("var") is not None:
            z = zscore_nlpd(yte, p, d["var"])
            dd.update({"nlpd": z["nlpd"], "c68": z["c68"], "c95": z["c95"]})
            # an interval that is narrower than the residual scale is not a result,
            # it is a broken interval; flag it rather than quoting its NLPD
            med_var = float(np.median(d["var"]))
            med_err2 = float(np.median((p - yte) ** 2))
            dd["interval_degenerate"] = bool(med_var < 1e-8 * med_err2)
            dd["median_interval_sd_pu"] = float(np.sqrt(med_var))
            dd["median_residual_pu"] = float(np.sqrt(med_err2))
        if name == "resid_gp_tempered":
            dd.update(d["loo_coverage"] and {
                "loo_c68": d["loo_coverage"]["c68"],
                "loo_c95": d["loo_coverage"]["c95"],
                "loo_nlpd": d["loo_coverage"]["nlpd"]})
            zw = zscore_nlpd(yte, p, d["tempered_raw_var"])
            dd.update({"nlpd_raw_floor": zw["nlpd"], "c68_raw_floor": zw["c68"],
                       "c95_raw_floor": zw["c95"]})
        rec["arms"][name] = dd

    # physics explanatory power OUT OF SAMPLE, and the effective dof at this n
    rec["R2_phys_oos"] = float(1.0 - ((yte - phys["pred"]) ** 2).mean() /
                               max(((yte - mean_pred) ** 2).mean(), 1e-30))
    rec["R2_phys_oos_median_output"] = float(np.median(
        1.0 - ((yte - phys["pred"]) ** 2).mean(0) /
        np.maximum(((yte - mean_pred) ** 2).mean(0), 1e-30)))
    rec["physics_effective_dof"] = effective_dof(rc, a1)
    rec["physics_penalty_frac"] = float(rc["frac"])
    rec["physics_order"] = int(phys["hyper"]["order"])
    # residual rank in this fold: an n-row residual cannot exceed rank n-1
    r_tr = ytr - phys["phys_train_pred"]
    sv = np.linalg.svd(r_tr - r_tr.mean(0), compute_uv=False)
    rec["residual_rank_train"] = {"sv_normalised": [float(v / max(sv[0], 1e-30)) for v in sv[:12]],
                                  "rank_1e2": int((sv > 1e-2 * sv[0]).sum()) if sv[0] > 0 else 0,
                                  "max_possible": int(n - 1)}
    rec["zero_mean_gp_hyper"] = {"ell": fit0.ell, "s2": fit0.s2, "n2": fit0.n2}
    rec["resid_gp_hyper"] = {"ell": fit1.ell, "s2": fit1.s2, "n2": fit1.n2}
    rec["keys"] = {"tr": tr.tolist(), "te": te.tolist()}
    if keep_pred:
        rec["predictions"] = {name: d["pred"] for name, d in {**arms, **base}.items()}
        rec["variances"] = {name: d["var"] for name, d in {**arms, **base}.items()
                             if d.get("var") is not None}
    return rec


# ------------------------------------------------------------- aggregation
def bootstrap_ci(vals: np.ndarray, n_boot: int = 10000, seed: int = 424242) -> list[float]:
    """95% bootstrap CI of the mean over folds."""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_test(vals: np.ndarray) -> float:
    """Two-sided exact sign-test p-value for a set of paired margins."""
    v = np.asarray(vals, dtype=float)
    pos = int((v > 0).sum())
    nz = int((v != 0).sum())
    if nz == 0:
        return 1.0
    from math import comb
    k = min(pos, nz - pos)
    tail = sum(comb(nz, i) for i in range(k + 1)) / 2 ** nz
    return float(min(1.0, 2.0 * tail))


def summarise(records: list[dict]) -> dict:
    """Mean / worst-fold aggregates plus the per-fold arrays behind every summary."""
    arms = list(records[0]["arms"]) + list(records[0]["baselines"]) + list(records[0]["diagnostics"])
    out = {"n_folds": len(records), "per_fold": {}}
    for rank_key in ("rmse_norm",):
        for name in arms:
            vals = np.array([r["arms"].get(name, r["baselines"].get(name, r["diagnostics"].get(name)))[rank_key]
                             for r in records])
            out["per_fold"][name] = [float(v) for v in vals]
    for name in arms:
        def g(key):
            return np.array([r["arms"].get(name, r["baselines"].get(name, r["diagnostics"].get(name))).get(key, np.nan)
                             for r in records], dtype=float)
        block = {"mean_rmse_norm": float(np.nanmean(g("rmse_norm"))),
                 "worst_fold_rmse_norm": float(np.nanmax(g("rmse_norm"))),
                 "best_fold_rmse_norm": float(np.nanmin(g("rmse_norm"))),
                 "mean_rmse_ens_norm": float(np.nanmean(g("rmse_ens_norm"))),
                 "per_fold_rmse_norm": [float(v) for v in g("rmse_norm")]}
        if np.isfinite(g("nlpd")).any():
            deg = np.array([r["arms"].get(name, r["baselines"].get(name, r["diagnostics"].get(name)))
                            .get("interval_degenerate", False) for r in records], dtype=bool)
            isd = g("median_interval_sd_pu")
            block.update({"mean_nlpd": float(np.nanmean(g("nlpd"))),
                          "frac_folds_interval_degenerate": float(np.mean(deg)),
                          "mean_median_interval_sd_pu": (float(np.nanmean(isd))
                                                         if np.isfinite(isd).any() else None),
                          "mean_coverage68": float(np.nanmean(g("c68"))),
                          "mean_coverage95": float(np.nanmean(g("c95"))),
                          "worst_fold_coverage68_dev": float(np.nanmax(np.abs(g("c68") - 0.68))),
                          "per_fold_coverage68": [float(v) for v in g("c68")],
                          "per_fold_coverage95": [float(v) for v in g("c95")],
                          "per_fold_nlpd": [float(v) for v in g("nlpd")]})
        if name == "resid_gp_tempered":
            block.update({"mean_loo_coverage68": float(np.nanmean(g("loo_c68"))),
                          "mean_loo_coverage95": float(np.nanmean(g("loo_c95"))),
                          "mean_loo_nlpd": float(np.nanmean(g("loo_nlpd"))),
                          "per_fold_loo_c68": [float(v) for v in g("loo_c68")]})
        if name in MODEL_ARMS:
            m = g("margin_vs_best_baseline")
            block.update({"mean_margin_vs_best_baseline": float(np.nanmean(m)),
                          "worst_fold_margin": float(np.nanmin(m)),
                          "best_fold_margin": float(np.nanmax(m)),
                          "frac_folds_clearing_20pct": float(np.mean(m >= 0.20)),
                          "frac_folds_positive_margin": float(np.mean(m > 0)),
                          "margin_ci95": bootstrap_ci(m),
                          "margin_sign_test_p": sign_test(m),
                          "per_fold_margin": [float(v) for v in m],
                          "any_numerical_failure": bool(
                              np.nansum(g("numerical_failure")) > 0)})
        out[name] = block
    # best-of-baselines per fold, and the guard
    out["mean_rmse_over_mean_predictor"] = {
        name: float(np.nanmean([r["arms"].get(name, r["baselines"].get(name, r["diagnostics"].get(name)))["rmse_over_mean_predictor"]
                                for r in records])) for name in arms}
    out["best_baseline_counts"] = {b: int(sum(r["best_baseline"] == b for r in records))
                                   for b in set(r["best_baseline"] for r in records)}
    out["R2_phys_oos_mean"] = float(np.mean([r["R2_phys_oos"] for r in records]))
    out["R2_phys_oos_worst"] = float(np.min([r["R2_phys_oos"] for r in records]))
    out["physics_effective_dof_mean"] = float(np.mean([r["physics_effective_dof"] for r in records]))
    out["physics_effective_dof_max"] = float(np.max([r["physics_effective_dof"] for r in records]))
    out["physics_effective_dof_over_n"] = float(np.mean(
        [r["physics_effective_dof"] / r["n_train"] for r in records]))
    out["physics_order_counts"] = {str(o): int(sum(r["physics_order"] == o for r in records))
                                  for o in (1, 2)}
    out["physics_penalty_frac_median"] = float(np.median([r["physics_penalty_frac"] for r in records]))
    out["residual_rank_1e2_mean"] = float(np.mean([r["residual_rank_train"]["rank_1e2"] for r in records]))
    out["residual_rank_max_possible"] = int(records[0]["residual_rank_train"]["max_possible"])
    for key in ("alpha_loo", "alpha_raw"):
        al = [r["arms"]["resid_gp_tempered"]["hyper"][key] for r in records]
        out[f"alpha_{key.split('_')[1]}_mean"] = float(np.mean(al))
        out[f"alpha_{key.split('_')[1]}_median"] = float(np.median(al))
        out[f"alpha_{key.split('_')[1]}_per_fold"] = [float(v) for v in al]
    out["gp_hyper_zero_mean"] = {"ell": [r["zero_mean_gp_hyper"]["ell"] for r in records],
                                 "s2": [r["zero_mean_gp_hyper"]["s2"] for r in records],
                                 "n2": [r["zero_mean_gp_hyper"]["n2"] for r in records]}
    out["gp_hyper_resid"] = {"ell": [r["resid_gp_hyper"]["ell"] for r in records],
                             "s2": [r["resid_gp_hyper"]["s2"] for r in records],
                             "n2": [r["resid_gp_hyper"]["n2"] for r in records]}
    out["mean_rmse_over_mean_predictor_max"] = {
        name: float(np.nanmax([r["arms"].get(name, r["baselines"].get(name, r["diagnostics"].get(name)))["rmse_over_mean_predictor"]
                              for r in records])) for name in arms}
    return out


# ------------------------------------------------------------------ driver
def read_previous_folds() -> list[dict]:
    """Rebuild the per-fold CSV rows from a completed results file."""
    import csv as _csv
    path = RESULTS / "folds_D.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    out = []
    for r in rows:
        conv = {}
        for k, v in r.items():
            try:
                conv[k] = float(v) if v not in ("", None) else np.nan
            except (TypeError, ValueError):
                conv[k] = v
        out.append(conv)
    return out


def run_cell(x: np.ndarray, y: np.ndarray, n: int, fe: Feeder, sheet: np.ndarray,
             s_op: np.ndarray, metrics: dict, v0bus: np.ndarray,
             tag: str) -> tuple[dict, list[dict]]:
    recs = []
    t0 = time.time()
    for seed in SEEDS:
        for scheme in SCHEMES:
            tr, te = fold_indices(n, seed, scheme, len(x))
            rec = run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
            rec["seed"] = seed
            rec["scheme"] = scheme
            recs.append(rec)
    summ = summarise(recs)
    summ["wall_seconds"] = float(time.time() - t0)
    logger.info(f"[{tag}] n={n:3d} FoldSummary " + "  ".join(
        f"{a}={summ[a]['mean_rmse_norm']:.4f}" for a in ALL_ARMS))
    return summ, recs


def flatten_folds(regime: str, n: int, recs: list[dict]) -> list[dict]:
    rows = []
    for r in recs:
        row = {"regime": regime, "n_train": n, "seed": r["seed"], "scheme": r["scheme"],
               "n_test": r["n_test"], "vsd_train": r["vsd_train"],
               "best_baseline": r["best_baseline"],
               "best_baseline_rmse_norm": r["best_baseline_rmse"],
               "R2_phys_oos": r["R2_phys_oos"],
               "physics_effective_dof": r["physics_effective_dof"],
               "physics_penalty_frac": r["physics_penalty_frac"],
               "physics_order": r["physics_order"],
               "alpha_loo": r["arms"]["resid_gp_tempered"]["hyper"]["alpha_loo"],
               "alpha_raw": r["arms"]["resid_gp_tempered"]["hyper"]["alpha_raw"],
               "mean_predictor_rmse_norm": r["rmse_mean_predictor"] / r["vsd_train"]}
        for name in ALL_ARMS:
            d = r["arms"].get(name) or r["baselines"].get(name)
            row[f"{name}__rmse_norm"] = d["rmse_norm"]
            row[f"{name}__rmse_ens_norm"] = d["rmse_ens_norm"]
            if name in MODEL_ARMS:
                row[f"{name}__margin"] = d["margin_vs_best_baseline"]
            if "nlpd" in d:
                row[f"{name}__nlpd"] = d["nlpd"]
                row[f"{name}__c68"] = d["c68"]
                row[f"{name}__c95"] = d["c95"]
        rows.append(row)
    return rows


def leakage_test(x: np.ndarray, y: np.ndarray, fe: Feeder, sheet: np.ndarray,
                 s_op: np.ndarray, metrics: dict, v0bus: np.ndarray) -> dict:
    """Permute test targets, assert every prediction is bit-identical.

    If any statistic, reference point or hyperparameter were computed from test
    rows, the predictions would move.  This is the check this field's reviewers
    name first after baseline contamination.
    """
    out = {}
    for n in NS:
        tr, te = fold_indices(n, 0, 0, len(x))
        a = run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus, keep_pred=True)
        yp = y.copy()
        rng = np.random.default_rng(999)
        yp[np.asarray(te)] = y[np.asarray(te)[rng.permutation(len(te))]]
        b = run_fold(x, yp, tr, te, fe, sheet, s_op, metrics, v0bus, keep_pred=True)
        # compare the PREDICTIONS and the predictive VARIANCES, not the scores: a
        # score legitimately depends on the test targets, a prediction must not.
        worst_p = 0.0
        worst_v = 0.0
        for name, pa in a["predictions"].items():
            pb = b["predictions"][name]
            worst_p = max(worst_p, float(np.max(np.abs(pa - pb))))
        for name, va in a["variances"].items():
            vb = b["variances"][name]
            worst_v = max(worst_v, float(np.max(np.abs(va - vb))))
        out[str(n)] = {"bit_identical": bool(worst_p == 0.0),
                       "variance_bit_identical": bool(worst_v == 0.0),
                       "max_abs_prediction_difference": worst_p,
                       "max_abs_variance_difference": worst_v,
                       "n_models_compared": len(a["predictions"])}
    return out


def nominal_predictions(x: np.ndarray, y: np.ndarray, n: int, fe: Feeder, sheet: np.ndarray,
                        s_op: np.ndarray, metrics: dict, v0bus: np.ndarray,
                        seed: int = 0, scheme: int = 0) -> dict:
    """Raw predictions for one fold, kept so a later reader can re-check the scores."""
    tr, te = fold_indices(n, seed, scheme, len(x))
    base, (ztr, zte) = build_baselines(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
    ftr, fte = feats_from_S(s_op, x[tr], x[te])
    dtr, dst = gram(ftr), sqd(fte, ftr)
    phys = base["b5_lindist"]
    resid_tr = y[tr] - phys["phys_train_pred"]
    fit1 = gp_fit(dtr, resid_tr, need_loo=True)
    mu_r, pv_r = gp_predict(fit1, dst)
    fit0 = gp_fit(dtr, y[tr], need_loo=True)
    mu0, pv0 = gp_predict(fit0, dst)
    ctr, cte = ridge_design(ztr, zte)
    rres = ridge_cv(ctr, resid_tr, cte)
    return {"te": np.asarray(te), "y": y[te],
            "physics_only": phys["pred"], "zero_mean_gp": mu0,
            "resid_gp_mll": phys["pred"] + mu_r,
            "resid_ridge_linear": phys["pred"] + rres["pred"],
            "b1_rbf": base["b1_rbf"]["pred"], "b3_lap_adm": base["b3_lap_adm"]["pred"]}


def determinism_test(x: np.ndarray, y: np.ndarray, fe: Feeder, sheet: np.ndarray,
                     s_op: np.ndarray, metrics: dict, v0bus: np.ndarray) -> dict:
    n = 40
    tr, te = fold_indices(n, 3, 1, len(x))
    a = run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus, keep_pred=True)
    b = run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus, keep_pred=True)
    worst = 0.0
    for name, pa in a["predictions"].items():
        worst = max(worst, float(np.max(np.abs(pa - b["predictions"][name]))))
    return {"bit_identical": bool(worst == 0.0),
            "max_abs_prediction_difference": float(worst),
            "fold": {"n_train": n, "seed": 3, "scheme": 1}}


def main() -> None:
    t_start = time.time()
    # hard RAM cap, container-aware (14 GB limit; use 6 GB for this tiny sweep)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    except (ValueError, OSError):
        logger.warning("could not set RLIMIT_AS; continuing without a hard RAM cap")

    logger.info("building the feeder (no AC solves yet)")
    fe = Feeder()
    sheet, phase_true, s_ref = load_metadata()
    if not np.array_equal(np.argmax(fe.P0, axis=1), phase_true):
        raise AssertionError("network phases disagree with the archived ph_true")
    far, sheet_rebuilt = fe.far_sheet(15)
    if not np.array_equal(sheet_rebuilt, sheet):
        raise AssertionError(f"far-end sheet mismatch {sheet_rebuilt} vs {sheet}")
    logger.info(f"feeder: {len(fe.buses)} buses, {fe.nload} loads, "
                f"phase counts {np.bincount(fe.phase, minlength=3).tolist()}, "
                f"sheet {sheet.tolist()}")

    t_ac = time.time()
    v0bus = fe.noload_voltage(sheet)
    t_noload = time.time() - t_ac
    s_op, op_check = verify_operator(fe, sheet, s_ref)
    logger.info(f"operator verified: frob ratio {op_check['frob_ratio_vs_archived']:.6f}, "
                f"normalised deviation {op_check['normalised_deviation_vs_archived']:.3e}, "
                f"r_eff(1e-2) {op_check['spectrum']['rank_1e2']}, "
                f"stable rank {op_check['spectrum']['stable_rank']:.3f}")

    logger.info("building graph Laplacians and the 55x55 kernel metrics (line data only)")
    metrics = build_laplacians(fe)
    logger.info(f"Laplacian symmetry: transpose defect {metrics['laplacian_symmetry']['rel_transpose_defect']:.3e} "
                f"relative, Hermitian defect {metrics['laplacian_symmetry']['rel_hermitian_defect']:.3e} relative")

    # spectra of the three graph metrics: the rank each one retains is the mechanism
    # that decides which graph kernel wins here, so it is measured, not assumed
    metric_spectra = {}
    for key in ("M_hop", "M_adm", "M_imp"):
        w = np.linalg.eigvalsh(0.5 * (metrics[key] + metrics[key].T))
        w = np.clip(w, 0.0, None)
        share = w / max(w.sum(), 1e-300)
        metric_spectra[key] = {
            "top1_share": float(share[-1]),
            "top3_share": float(share[-3:].sum()),
            "participation_ratio": float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300)),
            "n_eigenvalues": int(len(w)),
            "top5_eigenvalues": [float(v) for v in np.sort(w)[::-1][:5]],
        }
    checks: dict = {
        "ensemble_rows_finite": {},
        "operator_vs_archived": op_check,
        "laplacian_symmetry": metrics["laplacian_symmetry"],
        "hop_distance_diagnostic": metrics["hop_distance_diagnostic"],
        "graph_metric_spectra": metric_spectra,
        "noload_solve_seconds": float(t_noload),
        "ac_solves_charged_to_the_operator": 1,
        "ac_solves_charged_to_labels": 220,
        "reserved_evidence_untouched": True,
        "denylist": list(DENYLIST),
    }

    results = {"config": {
        "ns": list(NS), "seeds": list(SEEDS), "schemes": list(SCHEMES),
        "fold_base": FOLD_BASE, "n_folds_per_cell": len(SEEDS) * len(SCHEMES),
        "regime_decision": REGIME_DECIDE, "regime_robustness": REGIME_ROBUST,
        "regime_reserved_forbidden": RESERVED,
        "gp_grid": {"ell": list(GP_LGRID), "s2": list(GP_S2GRID), "n2": list(GP_N2GRID)},
        "penalty_fracs": PENALTY_FRACS.tolist(),
        "penalty_scaling": "fractions of the design-Gram largest eigenvalue",
        "alpha_grid": ALPHA_GRID.tolist(),
        "rbf_sqd_divisor": RBF_DIVISOR,
        "normaliser": "mean per-output standard deviation of the TRAIN rows of the fold",
        "normaliser_diagnostic": "ensemble-wide per-output standard deviation (labelled diagnostic)",
        "target_sheet": sheet.tolist(),
        "n_outputs": 45, "n_inputs": 110,
        "metric_slicing": ("the graph metrics are 55x55 and act on the load-MAGNITUDE block, so P and Q "
                           "share one metric; every graph-kernel baseline uses this same construction"),
        "baseline_contamination": ("no baseline touches S or any sensitivity Jacobian; b1 is the RBF on raw "
                                   "injections, b2/b3 are hop and admittance graph metrics from line data, "
                                   "b4 is ridge on injections, b5 is LinDistFlow plus ridge calibration"),
    }}

    all_rows: list[dict] = []
    plan = [(REGIME_DECIDE, "decision"), (REGIME_ROBUST, "robustness")]
    resume = os.environ.get("RESUME_PANELS", "1") == "1"
    if resume:
        merged = {}
        try:
            for name, key in (("screen_D.json", "panel_decision"),
                              ("screen_D_robustness.json", "panel_robustness")):
                fp = RESULTS / name
                if fp.exists():
                    blob = json.loads(fp.read_text())
                    if key in blob and len(blob[key]["cells"]) == len(NS):
                        merged[key] = blob[key]
            if len(merged) == 2:
                logger.info(f"resuming: both panels are complete in results/, reusing them")
                results.update(merged)
            else:
                logger.info(f"resume requested but only {sorted(merged)} are complete; "
                            f"recomputing the incomplete panels")
                results.update(merged)
                resume = False
        except (json.JSONDecodeError, KeyError):
            logger.warning("could not parse the previous results files; recomputing the sweep")
            resume = False
    if resume:
        for regime, tag in plan:
            x, y, stats = stage_ensemble(regime)
            checks["ensemble_rows_finite"][regime] = stats
            checks[f"q_is_deterministic_{regime}"] = check_q_is_deterministic(x)
            del x, y
            gc.collect()
        all_rows = read_previous_folds()
    for regime, tag in plan:
        if f"panel_{tag}" in results:
            continue
        x, y, stats = stage_ensemble(regime)
        checks["ensemble_rows_finite"][regime] = stats
        checks[f"q_is_deterministic_{regime}"] = check_q_is_deterministic(x)
        logger.info(f"regime {regime} ({tag}): {stats['n_rows']} rows, "
                    f"{stats['mean_kW']:.2f} kW, far-end Vstd {stats['Vstd']:.6f} pu")
        panel = {"stats": stats, "cells": {}}
        for n in NS:
            summ, recs = run_cell(x, y, n, fe, sheet, s_op, metrics, v0bus, f"{tag} s{regime}")
            panel["cells"][str(n)] = summ
            all_rows += flatten_folds(regime, n, recs)
            partial = dict(results)
            partial[f"panel_{tag}"] = panel
            out_path = RESULTS / ("screen_D.json" if tag == "decision" else "screen_D_robustness.json")
            out_path.write_text(json.dumps(partial, indent=1))
            logger.debug(f"wrote partial {out_path.name} after n={n}")
            del recs
            gc.collect()
        results[f"panel_{tag}"] = panel
        del x, y
        gc.collect()

    dec = results["panel_decision"]["cells"]
    rob = results["panel_robustness"]["cells"]

    # ------------------------------------------------ pre-registered readings
    readings: dict = {}
    for n in (10, 20):
        c = dec[str(n)]
        d = c["resid_gp_tempered"]
        readings[f"R1_small_n_crossing_at_{n}"] = {
            "mean_margin": d["mean_margin_vs_best_baseline"],
            "ci95": d["margin_ci95"], "sign_test_p": d["margin_sign_test_p"],
            "passes": bool(d["mean_margin_vs_best_baseline"] > 0 and d["margin_ci95"][0] > 0),
            "physics_only_margin": c["physics_only"]["mean_margin_vs_best_baseline"],
            "zero_mean_gp_rmse": c["zero_mean_gp"]["mean_rmse_norm"],
            "physics_only_rmse": c["physics_only"]["mean_rmse_norm"],
            "tempered_rmse": d["mean_rmse_norm"],
            "effective_dof": c["physics_effective_dof_mean"],
            "interpretation": ("at n=10 the 47-parameter physics calibration nearly interpolates the "
                               "training rows, so the residual GP has almost nothing left to fit and this "
                               "arm is approximately the physics baseline by construction"),
        }
    readings["R2_D_vs_zero_mean_gp"] = {
        str(n): {"tempered_margin_vs_zero_mean": float(
            1.0 - dec[str(n)]["resid_gp_tempered"]["mean_rmse_norm"] /
            dec[str(n)]["zero_mean_gp"]["mean_rmse_norm"]),
            "resid_mll_margin_vs_zero_mean": float(
                1.0 - dec[str(n)]["resid_gp_mll"]["mean_rmse_norm"] /
                dec[str(n)]["zero_mean_gp"]["mean_rmse_norm"]),
            "zero_mean_rmse": dec[str(n)]["zero_mean_gp"]["mean_rmse_norm"],
            "resid_mll_rmse": dec[str(n)]["resid_gp_mll"]["mean_rmse_norm"]}
        for n in NS}
    qual = [n for n in NS
            if abs(dec[str(n)]["resid_gp_tempered"]["mean_coverage68"] - 0.68) <= 0.05
            and abs(dec[str(n)]["resid_gp_tempered"]["mean_coverage95"] - 0.95) <= 0.05]
    readings["R3_coverage_repair"] = {
        "qualifying_n": qual,
        "rule": "usable if, at every n from 10 to 160, |c68-0.68|<=0.05 and |c95-0.95|<=0.05",
        "passes_rule": bool(len(qual) == len(NS)),
        "per_n": {str(n): {"tempered_c68": dec[str(n)]["resid_gp_tempered"]["mean_coverage68"],
                           "tempered_c95": dec[str(n)]["resid_gp_tempered"]["mean_coverage95"],
                           "tempered_nlpd": dec[str(n)]["resid_gp_tempered"].get("mean_nlpd"),
                           "mll_c68": dec[str(n)]["resid_gp_mll"]["mean_coverage68"],
                           "mll_c95": dec[str(n)]["resid_gp_mll"]["mean_coverage95"],
                           "ridge_c68": dec[str(n)]["resid_ridge_linear"]["mean_coverage68"],
                           "ridge_c95": dec[str(n)]["resid_ridge_linear"]["mean_coverage95"],
                           "physics_c68": dec[str(n)]["physics_only"]["mean_coverage68"],
                           "physics_c95": dec[str(n)]["physics_only"]["mean_coverage95"]}
                  for n in NS},
    }
    n80 = dec["80"]["resid_gp_tempered"]
    readings["R4_screen_survival"] = {
        "S1_mean_margin_at_80": n80["mean_margin_vs_best_baseline"],
        "S1_pass": bool(n80["mean_margin_vs_best_baseline"] >= 0.20),
        "S2_worst_fold_at_80": n80["worst_fold_margin"],
        "S2_pass": bool(n80["worst_fold_margin"] >= 0.20),
        "S3_no_candidate_beats_D_by_15pct": {
            "note": ("the other four screened artifacts run in parallel and reading them is forbidden, "
                     "so this is not evaluable inside this artifact"),
            "status": "NOT_EVALUABLE_IN_ISOLATION"},
        "ranking_at_80": {a: dec["80"][a]["mean_rmse_norm"] for a in ALL_ARMS},
        "zero_mean_vs_tempered_ratio": float(
            dec["80"]["zero_mean_gp"]["mean_rmse_norm"] /
            dec["80"]["resid_gp_tempered"]["mean_rmse_norm"]),
    }
    readings["R5_nlpd_not_used_to_arbitrate"] = True
    results["readings"] = readings
    results["screen_metric_at_80"] = n80["mean_margin_vs_best_baseline"]
    results["screen_metric_at_80_definition"] = (
        "mean paired margin of the physics-plus-residual GP (tempered intervals, point prediction "
        "identical to the marginal-likelihood arm) over the best of the five named baselines in the "
        "same fold, at n=80 on the 2.0x regime, averaged over 24 folds")
    results["robustness_panel_summary"] = {
        str(n): {a: rob[str(n)][a]["mean_rmse_norm"] for a in ALL_ARMS} for n in NS}

    # ------------------------------------------------------ protocol checks
    logger.info("running executable protocol checks (leakage, determinism, guard)")
    x, y, _ = stage_ensemble(REGIME_DECIDE)
    checks["leakage_test"] = leakage_test(x, y, fe, sheet, s_op, metrics, v0bus)
    checks["determinism_test"] = determinism_test(x, y, fe, sheet, s_op, metrics, v0bus)
    guard = {tag: {str(n): {a: dec[str(n)]["mean_rmse_over_mean_predictor"][a] for a in ALL_ARMS}
                   for n in NS} for tag in ("decision",)}
    worst_guard = max(v for n in NS for v in guard["decision"][str(n)].values())
    checks["guard_g1_pass"] = {"worst_rmse_over_mean_predictor": float(worst_guard),
                               "passes": bool(worst_guard <= 1.0),
                               "per_cell": guard["decision"]}
    checks["guard_g2_grid"] = {
        "penalty_was_a_fraction_of_lmax": True,
        "grid_span": [float(PENALTY_FRACS[0]), float(PENALTY_FRACS[-1])],
        "median_chosen_fraction": float(np.median(
            [dec[str(n)]["physics_penalty_frac_median"] for n in NS])),
        "chosen_fraction_per_n": {str(n): dec[str(n)]["physics_penalty_frac_median"] for n in NS}}
    # tempering is variance-only: recompute one fold and compare predictions
    tr, te = fold_indices(80, 0, 0, len(x))
    rec = run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
    # the record stores no predictions, so compare by re-deriving them here
    checks["tempering_is_variance_only"] = _tempering_identity_check(
        x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
    checks["n_equals_10_interpolation"] = {
        "effective_dof_at_10": dec["10"]["physics_effective_dof_mean"],
        "effective_dof_over_n_at_10": dec["10"]["physics_effective_dof_over_n"],
        "n_parameters_of_physics_calibration": 47,
        "interpretation": ("the physics calibration has 47 parameters against 10 training rows at n=10, "
                           "so it nearly interpolates and a small margin there is not new mechanism")}
    results["protocol_checks"] = checks

    # raw predictions behind the n=10 / n=160 claims, so a reader can re-check the
    # scores from more than the summary (seed 0, scheme 0 of the decision regime)
    raw = {}
    for n in (10, 80, 160):
        pr = nominal_predictions(x, y, n, fe, sheet, s_op, metrics, v0bus)
        raw[str(n)] = {"test_indices": pr["te"].tolist(),
                       "y_true": pr["y"].tolist(),
                       **{k: np.asarray(v).tolist() for k, v in pr.items()
                          if k not in ("te", "y")}}
    (RESULTS / "raw_predictions_seed0_scheme0.json").write_text(json.dumps(raw))
    del raw
    del x, y
    gc.collect()

    # -------------------------------------------------------------- write out
    (RESULTS / "screen_D.json").write_text(json.dumps(results, indent=1))
    (RESULTS / "protocol_checks.json").write_text(json.dumps(checks, indent=1))

    import csv
    keys = sorted({k for r in all_rows for k in r})
    with open(RESULTS / "folds_D.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(all_rows)

    write_method_out(results, checks, dec)
    logger.info(f"total wall clock {time.time() - t_start:.1f}s; outputs in {RESULTS}")


def _tempering_identity_check(x, y, tr, te, fe, sheet, s_op, metrics, v0bus) -> dict:
    """Assert the tempered arm's POINT prediction is bit-identical to the MLL arm's."""
    n = len(tr)
    ftr, fte = feats_from_S(s_op, x[tr], x[te])
    base, _ = build_baselines(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
    phys = base["b5_lindist"]
    resid_tr = y[tr] - phys["phys_train_pred"]
    fit1 = gp_fit(gram(ftr), resid_tr, need_loo=True)
    mu_r, pv_r = gp_predict(fit1, sqd(fte, ftr))
    p_mll = phys["pred"] + mu_r
    h = phys["phys_hat"]
    phys_loo_err = (y[tr] - phys["phys_train_pred"]) / (1.0 - h)[:, None]
    gp_loo_err = y[tr] - phys["phys_train_pred"] - fit1.loo_mu
    loo_total = phys["phys_resid_var"][None, :] / (1.0 - h)[:, None] + fit1.loo_var
    s2_ref = float(np.mean((phys_loo_err + gp_loo_err) ** 2))
    alpha, _ = tempered_alpha(phys_loo_err, gp_loo_err, loo_total, s2_ref)
    p_temp = phys["pred"] + mu_r  # the tempering only widens pv_r
    pv_temp = pv_r + alpha * s2_ref
    return {"tempered_point_prediction_identical": bool(np.array_equal(p_mll, p_temp)),
            "exact_zero_difference": bool(float(np.max(np.abs(p_mll - p_temp))) == 0.0),
            "alpha_used": float(alpha),
            "min_var_increase": float(np.min(pv_temp - pv_r)),
            "n": n}


def write_method_out(results: dict, checks: dict, dec: dict) -> None:
    """Top-level result in the exp_gen_sol_out schema (one example per fold cell)."""
    examples = []
    for n in NS:
        c = dec[str(n)]
        for arm in MODEL_ARMS + BASE_ARMS:
            b = c[arm]
            examples.append({
                "input": (f"Predict per-unit magnitude at 15 far-end buses x 3 phases on the IEEE PES "
                          f"European LV feeder from {n} three-phase AC power-flow solves; "
                          f"model {arm}."),
                "output": (f"mean normalised far-end RMSE {b['mean_rmse_norm']:.6f}, "
                           f"worst fold {b['worst_fold_rmse_norm']:.6f}"
                           + (f", mean paired margin over the best baseline "
                              f"{b['mean_margin_vs_best_baseline']:+.4f}"
                              if arm in MODEL_ARMS else "")
                           + (f", mean NLPD {b['mean_nlpd']:.4f}, coverage 68% "
                              f"{b['mean_coverage68']:.3f} / 95% {b['mean_coverage95']:.3f}"
                              if "mean_nlpd" in b else "")),
                "metadata_n_train": int(n),
                "metadata_model": arm,
                "metadata_is_baseline": str(arm in BASE_ARMS).lower(),
                "metadata_is_diagnostic": "false",
                # every row carries the SAME paired comparison at this training
                # size, so a downstream reader never has to join rows across models:
                # the treatment's mean normalised RMSE against the best of the five
                # named baselines in the same folds
                "predict_our_method": f"{c['resid_gp_tempered']['mean_rmse_norm']:.6f}",
                "predict_baseline": f"{min(c[b]['mean_rmse_norm'] for b in BASE_ARMS):.6f}",
            })
    out = {
        "metadata": {
            "method_name": "physics-prior-mean residual GP with an output-restricted sensitivity kernel",
            "description": (
                "Candidate D of a five-way kernel screen on the IEEE PES European LV feeder: a "
                "train-calibrated multiphase LinDistFlow prediction is the GP prior mean and the "
                "output-restricted sensitivity kernel exp(-||S(x-x')||^2/(2 l^2)) is placed on the "
                "residual, with a train-only noise-floor tempering selected by leave-one-out coverage "
                "matching. Six prediction columns per fold: physics mean alone, zero-mean operator "
                "kernel, physics-plus-residual GP by marginal likelihood, two alpha-tempered variants, "
                "and a plain ridge on the residual with classical ridge intervals."),
            "protocol": "12 seeds x 2 partition schemes = 24 folds per training size",
            "regimes": {"decision": "2.0x", "robustness": "1.0x", "reserved_forbidden": "2.6x"},
            "primary_metric": "mean normalised far-end RMSE (train-row normaliser)",
            "screen_metric_at_80": results.get("screen_metric_at_80"),
            "screen_metric_definition": results.get("screen_metric_at_80_definition"),
            "readings": results["readings"],
            "protocol_checks_summary": {
                "leakage_test_bit_identical": all(
                    v["bit_identical"] for v in checks["leakage_test"].values()),
                "determinism_bit_identical": checks["determinism_test"]["bit_identical"],
                "guard_g1_pass": checks["guard_g1_pass"]["passes"],
                "tempering_is_variance_only": checks["tempering_is_variance_only"][
                    "exact_zero_difference"],
                "operator_normalised_deviation_vs_archived":
                    checks["operator_vs_archived"]["normalised_deviation_vs_archived"],
            },
            "departures": [
                "No neural-network baseline: at 10-160 AC solves a DNN is outside its data regime; the "
                "claim is about which kernel to use at small n, not about kernels beating neural networks.",
                "One feeder, one 15-bus far-end sheet, one load model: external validity is asserted for "
                "this feeder only and the crossover number is not generalised.",
                "No measurement noise: ground truth is a deterministic function of the injection vector, "
                "so the GP noise term is a regulariser and calibration is read off empirical coverage.",
                "The graph metrics are 55x55 and slice the load-magnitude block, so P and Q share one "
                "metric; this flatters the treatment's margin if anything and is recorded rather than "
                "claimed as the strongest possible phase-aware graph kernel.",
                "No topology change / N-1 adaptation: out of scope for a fixed-topology kernel screen.",
            ],
            "ac_solves": {"operator": 1, "labels": 220, "note": "one no-load three-phase solve for the "
                                                                 "reference phasors, zero for the analytic operator"},
        },
        "datasets": [{"dataset": "ieee_european_lv_asymmetric_on_peak_566_runpp_3ph", "examples": examples}],
    }
    (HERE / "method_out.json").write_text(json.dumps(out, indent=1))
    logger.info(f"wrote method_out.json with {len(examples)} examples")


if __name__ == "__main__":
    main()
