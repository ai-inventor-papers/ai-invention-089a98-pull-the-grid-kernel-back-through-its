"""Fold engine: one pass over the pre-registered fold schedule computing the five
shared baselines ONCE per fold and then every spectral cell against those baselines.

Everything here is train-only except the final `predict` on held-out rows.
"""
from __future__ import annotations

import numpy as np

import learners
from baselines import (b1_rbf, b2_hop, b3_kirchhoff, b3p_kirchhoff_p, b4_ridge,
                       b5_calibrations, b5_lindist, b6_pca_gp, radius_normalise,
                       standardise)
from learners import gp_fit_predict, normalise_distance, ridge_fit_predict, sqd

ARM_SPECS = (
    # (arm name, route, k, variant, learner, scaling)
    ("SP-R1a-GP-k3", "R1", 3, "real", "gp", "S1"),
    ("SP-R1a-GP-k5", "R1", 5, "real", "gp", "S1"),
    ("SP-R1a-GP-k11", "R1", 11, "real", "gp", "S1"),
    ("SP-R1a-GP-k22", "R1", 22, "real", "gp", "S1"),
    ("SP-R1a-GP-k45", "R1", 45, "real", "gp", "S1"),
    ("SP-R1a-LIN-k11", "R1", 11, "real", "ridge", "S1"),
    ("SP-R1b-GP-k11", "R1", 11, "reim", "gp", "S1"),
    ("SP-R2-GP-k11", "R2", 11, "real", "gp", "S1"),
    ("SP-R3-GP-k11", "R3", 11, "real", "gp", "S1"),
    ("SP-R4-GP-k11", "R4", 11, "real", "gp", "S1"),
    ("SP-R4-GP-k45", "R4", 45, "real", "gp", "S1"),
    ("SP-R1a-GP-k11-S2", "R1", 11, "real", "gp", "S2"),
)
HEADLINE_ARM = "SP-R1a-GP-k11"
BASELINE_NAMES = ("b1_rbf", "b2_hop", "b3_kirchhoff", "b3p_kirchhoff_p", "b4_ridge",
                  "b5_lindist", "b6_pca_gp_k11")
# b3 is excluded from the independent set: it is derived from the same block Laplacian
# the candidate's spectrum comes from, so it is a related construction, not an opponent.
# b6 is the dose-matched generic-compression control, not a "best baseline" competitor,
# so it is scored separately in the mechanism comparison.
INDEPENDENT_BASELINES = ("b1_rbf", "b2_hop", "b4_ridge", "b5_lindist")
BEST_BASELINE_POOL = ("b1_rbf", "b2_hop", "b3_kirchhoff", "b3p_kirchhoff_p", "b4_ridge",
                      "b5_lindist")


def project_injections(Xtr, Xte, Fk):
    """A = [P @ Fk, Q @ Fk]; the P and Q blocks share one loading-independent block."""
    Atr = np.hstack([Xtr[:, :55] @ Fk, Xtr[:, 55:] @ Fk])
    Ate = np.hstack([Xte[:, :55] @ Fk, Xte[:, 55:] @ Fk])
    return Atr, Ate


def _ser_hyper(hp):
    """JSON-safe rendering of a learner's selected hyperparameters."""
    if isinstance(hp, dict):
        return {"hyper": [float(x) for x in hp["hyper"]],
                "alt_hyper": [float(x) for x in hp["alt_hyper"]],
                "criterion": hp["criterion"], "jittered": bool(hp["jittered"]),
                "cv_rmse": None if hp["cv_rmse"] is None else float(hp["cv_rmse"]),
                "alt_cv_rmse": None if hp["alt_cv_rmse"] is None else float(hp["alt_cv_rmse"])}
    if isinstance(hp, tuple):
        return [_jsonable(x) for x in hp]
    return hp


def _jsonable(x):
    if isinstance(x, (str, bool)) or x is None:
        return x
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def _metrics(mu, pv, Yte, ysd):
    e = np.sqrt(((mu - Yte) ** 2).mean()) / ysd
    out = {"nrmse": float(e), "nlpd": None, "cov68": None, "cov95": None}
    if pv is not None:
        pv = np.maximum(pv, 1e-300)
        z = (Yte - mu) / np.sqrt(pv)
        out["nlpd"] = float(-np.mean(np.log(2 * np.pi * pv) / 2 + z ** 2 / 2))
        out["cov68"] = float(np.mean(np.abs(z) <= 1.0))
        out["cov95"] = float(np.mean(np.abs(z) <= 1.96))
    return out


def run_fold(ops, ph, X, Y, tr, te, Vp1, Vp2, M110, M55, Mhop,
             arm_specs=ARM_SPECS, lgrid=None, collect_calibrations=False):
    """Compute the five shared baselines and every spectral cell on one fold."""
    ysd = Y[tr].std(0).mean()
    Ytr, Yte = Y[tr], Y[te]
    rec: dict = {"n_train": int(len(tr)), "n_test": int(len(te)), "ysd_train": float(ysd)}

    # ---- mean predictor (guard g1's denominator)
    mean_pred = np.tile(Ytr.mean(0), (len(te), 1))
    rec["mean_predictor_nrmse"] = float(np.sqrt(((mean_pred - Yte) ** 2).mean()) / ysd)

    # ---- the five shared baselines, ONCE per fold
    bl: dict = {}
    mu, pv, hp, clip = b1_rbf(X[tr], Ytr, X[te], lgrid=lgrid)
    bl["b1_rbf"] = (_metrics(mu, pv, Yte, ysd), hp, clip)
    mu, pv, hp, clip = b2_hop(X[tr], Ytr, X[te], Mhop)
    bl["b2_hop"] = (_metrics(mu, pv, Yte, ysd), hp, clip)
    mu, pv, hp, clip = b3_kirchhoff(X[tr], Ytr, X[te], M110)
    bl["b3_kirchhoff"] = (_metrics(mu, pv, Yte, ysd), hp, clip)
    mu, pv, hp, clip = b3p_kirchhoff_p(X[tr], Ytr, X[te], M55)
    bl["b3p_kirchhoff_p"] = (_metrics(mu, pv, Yte, ysd), hp, clip)
    mu, _, hp, clip = b4_ridge(X[tr], Ytr, X[te])
    bl["b4_ridge"] = (_metrics(mu, None, Yte, ysd), hp, clip)
    mu, _, hp, order = b5_lindist(Vp1[tr], Vp2[tr], Ytr, Vp1[te], Vp2[te])
    bl["b5_lindist"] = (_metrics(mu, None, Yte, ysd), hp, 0)
    # dose-matched generic-compression control: 2k = 22 coordinates, k = 11
    mu, pv, hp, clip = b6_pca_gp(X[tr], Ytr, X[te], 11)
    bl["b6_pca_gp_k11"] = (_metrics(mu, pv, Yte, ysd), hp, clip)

    rec["baselines"] = {k: dict(v[0], hyper=_ser_hyper(v[1]), clipped=v[2])
                        for k, v in bl.items()}
    best_name = min(BEST_BASELINE_POOL, key=lambda k: bl[k][0]["nrmse"])
    best_e = bl[best_name][0]["nrmse"]
    best_indep_name = min(INDEPENDENT_BASELINES, key=lambda k: bl[k][0]["nrmse"])
    best_indep_e = bl[best_indep_name][0]["nrmse"]
    rec["best_baseline"] = best_name
    rec["best_baseline_nrmse"] = float(best_e)
    rec["best_independent_baseline"] = best_indep_name
    rec["best_independent_nrmse"] = float(best_indep_e)
    rec["pca_control_nrmse"] = float(bl["b6_pca_gp_k11"][0]["nrmse"])

    if collect_calibrations:
        cal = b5_calibrations(Vp1[tr], Vp2[tr], Ytr, Vp1[te], Vp2[te], order)
        rec["lindist_calibrations"] = {
            k: float(np.sqrt(((v - Yte) ** 2).mean()) / ysd) for k, v in cal.items()}

    # ---- the spectral cells
    cells: dict = {}
    for name, route, k, variant, learner_kind, scaling in arm_specs:
        if route == "R5" and ops.routes["R5"].F is None:
            cells[name] = {"status": "unavailable", "reason": ops.routes["R5"].note}
            continue
        Fk = ops.features(route, k, variant)
        if not np.isfinite(Fk).all():
            cells[name] = {"status": "nan_features"}
            continue
        Atr, Ate = project_injections(X[tr], X[te], Fk)
        if scaling == "S1":
            Atr, Ate, clip = standardise(Atr, Ate)
        elif scaling == "S2":
            Atr, Ate = radius_normalise(Atr, Ate)
            clip = 0
        else:
            raise ValueError(scaling)
        if not (np.isfinite(Atr).all() and np.isfinite(Ate).all()):
            cells[name] = {"status": "nan_after_scaling"}
            continue
        if learner_kind == "gp":
            Dtr, Dst = normalise_distance(sqd(Atr, Atr), sqd(Ate, Atr), Atr.shape[1])
            mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst, lgrid=lgrid or learners.L_GRID)
            m = _metrics(mu, pv, Yte, ysd)
        elif learner_kind == "ridge":
            mu, lam = ridge_fit_predict(Atr, Ytr, Ate)
            hp = ("ridge", float(lam))
            m = _metrics(mu, None, Yte, ysd)
        else:
            raise ValueError(learner_kind)
        m["ratio_to_mean"] = float(m["nrmse"] / rec["mean_predictor_nrmse"])
        m["margin_vs_best_baseline"] = float(1.0 - m["nrmse"] / best_e)
        m["margin_vs_best_independent"] = float(1.0 - m["nrmse"] / best_indep_e)
        m["margin_vs_each"] = {b: float(1.0 - m["nrmse"] / bl[b][0]["nrmse"])
                               for b in BASELINE_NAMES}
        m["hyper"] = list(hp) if isinstance(hp, tuple) else hp
        if isinstance(hp, dict):
            m["lengthscale"] = hp["hyper"][0]
            m["grid_endpoint_hit"] = bool(
                abs(hp["hyper"][0] - hp["grid_lo"]) < 1e-12
                or abs(hp["hyper"][0] - hp["grid_hi"]) < 1e-12)
            m["alt_lengthscale"] = hp["alt_hyper"][0]
            m["jittered"] = hp["jittered"]
        m["clipped_columns"] = int(clip)
        m["k"] = int(k)
        m["route"] = route
        m["variant"] = variant
        m["scaling"] = scaling
        cells[name] = m
    rec["cells"] = cells
    return rec
