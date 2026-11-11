"""arms.py -- one fold, every arm and every baseline, on one shared partition.

The point of this module is that the treatment (a GP on the analytic operator's
coordinates) and every competitor see exactly the same train rows, the same test
rows, the same target sheet, the same train-only standardisation and the same
error denominator.  Everything that differs between two rows of the output table
is the model, never the data or the selection protocol.

`run_fold` returns plain Python floats only, so the caller can hash the whole
table and diff two full runs byte for byte.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import operators as opmod
from protocol import (EPS, fold_indices, gp_batch, gram, mean_predictor_rmse, normalisers,
                      predictive_scores, rmse_raw, sqd)
from ridge_blr import fit_affine, fit_blr, fit_diag_ridge, fit_ridge

G: dict[str, Any] = {}


def init_worker(state: dict[str, Any]) -> None:
    G.clear()
    G.update(state)


CALIB_ROUTES = ("cv-eigen", "loo", "blr", "affine", "diag-loo")


def _calib_member(Vtr: np.ndarray, Vte: np.ndarray, Ytr: np.ndarray, route: str) -> dict:
    if route == "cv-eigen":
        r = fit_ridge(Vtr, Ytr, Vte, mode="cv")
        return dict(pred=r["pred"], frac=r["frac"], df=r["df"])
    if route == "loo":
        r = fit_ridge(Vtr, Ytr, Vte, mode="gcv")
        return dict(pred=r["pred"], frac=r["frac"], df=r["df"])
    if route == "blr":
        b = fit_blr(Vtr, Ytr, Vte)
        return dict(pred=b["pred"], frac=b["frac"], df=b["df"])
    if route == "affine":
        a = fit_affine(Vtr, Ytr, Vte)
        return dict(pred=a["pred"], frac=0.0, df=a["df"])
    if route == "diag-loo":
        d = fit_diag_ridge(Vtr, Ytr, Vte)
        return dict(pred=d["pred"], frac=d["frac"], df=d["df"])
    raise ValueError(f"unknown calibration route {route!r}")       # pragma: no cover


def lindist_family(Vp1: np.ndarray, Vp2: np.ndarray, tr: np.ndarray, te: np.ndarray,
                   Ytr: np.ndarray, Yte: np.ndarray, vsd_train: float) -> dict[str, Any]:
    """The LinDistFlow family: 2 orders x 5 calibration routes = 10 members.

    Each member is scored on ONE shared inner K-fold partition of the TRAINING rows
    only, so the choice among members cannot see a test row.  The winner is refit on all
    training rows and its test RMSE is what the protocol scores against.  Every member's
    test RMSE is also recorded: that is the reconciliation table the plan requires, since
    the run's own two documents report this same baseline 1.6x apart.
    """
    orders = {1: (Vp1[tr], Vp1[te]), 2: (Vp2[tr], Vp2[te])}
    n = int(len(tr))
    k = max(2, min(5, n // 4))
    parts = np.array_split(np.arange(n), k)
    members: dict[str, dict[str, Any]] = {}
    for order, (Vr, Ve) in orders.items():
        for route in CALIB_ROUTES:
            tot, used = 0.0, 0
            ok = True
            for part in parts:
                inner_tr = np.setdiff1d(np.arange(n), part)
                if len(inner_tr) < 3:
                    ok = False
                    break
                try:
                    m = _calib_member(Vr[inner_tr], Vr[part], Ytr[inner_tr], route)
                except (np.linalg.LinAlgError, RuntimeError, ValueError):
                    ok = False
                    break
                tot += float(np.mean((m["pred"] - Ytr[part]) ** 2))
                used += 1
            if not ok or used == 0:
                continue
            key = f"order{order}/{route}"
            final = _calib_member(Vr, Ve, Ytr, route)
            e = rmse_raw(final["pred"], Yte)
            members[key] = dict(inner_cv_mse=float(tot / used), order=int(order),
                                route=route, rmse=float(e),
                                rmse_norm=float(e / (vsd_train + EPS)),
                                frac=float(final["frac"]), df=float(final["df"]))
    if not members:                                                # pragma: no cover
        raise RuntimeError("lindist_family: no admissible member")
    win = min(members, key=lambda kk: members[kk]["inner_cv_mse"])
    Vr, Ve = orders[members[win]["order"]]
    return dict(winner=win, order=int(members[win]["order"]), route=members[win]["route"],
                rmse=float(members[win]["rmse"]),
                rmse_norm=float(members[win]["rmse_norm"]),
                frac=float(members[win]["frac"]), df=float(members[win]["df"]),
                inner_cv_mse=float(members[win]["inner_cv_mse"]),
                members=members, pred=_calib_member(Vr, Ve, Ytr, members[win]["route"])["pred"])


# ---------------------------------------------------------------------------
# one fold
# ---------------------------------------------------------------------------
def run_fold(job: tuple[int, int, int]) -> dict[str, Any] | None:
    n, seed, scheme = job
    got = fold_indices(n, seed, scheme)
    if got is None:
        return None
    tr, te = got
    X, Y = G["X"], G["Y"]
    ops: opmod.OperatorBundle = G["ops"]
    Vp1_all, Vp2_all = G["Vp1"], G["Vp2"]
    graph_bases = G["graph_bases"]
    S, S11, V11, Sr, R11 = ops.S, ops.S11, ops.V11, ops.Sr, ops.R11

    Ytr, Yte = Y[tr], Y[te]
    Xtr, Xte = X[tr], X[te]
    vsd_train, vsd_all = normalisers(Ytr, Y)
    mp = mean_predictor_rmse(Ytr, Yte, vsd_train)
    res: dict[str, Any] = dict(n=int(n), seed=int(seed), scheme=int(scheme),
                              fold_id=f"n{n}_s{seed}_c{scheme}",
                              n_train=int(len(tr)), n_test=int(len(te)),
                              vsd_train=vsd_train, vsd_all=vsd_all,
                              mean_pred_rmse=mp,
                              numerical_failure=[], arms={}, baselines={})

    def score(name: str, pred: np.ndarray, extra: dict[str, Any]) -> None:
        e = rmse_raw(pred, Yte)
        nr = e / (vsd_train + EPS)
        rec: dict[str, Any] = dict(rmse=float(e), rmse_norm=float(nr),
                                   rmse_all=float(e / (vsd_all + EPS)),
                                   ratio_g1=float(nr / (mp + EPS)))
        if rec["ratio_g1"] > 1.0:
            rec["numerical_failure"] = True
            res["numerical_failure"].append(name)
        rec.update(extra)
        res["arms"][name] = rec

    def score_base(name: str, pred: np.ndarray, extra: dict[str, Any]) -> None:
        e = rmse_raw(pred, Yte)
        nr = e / (vsd_train + EPS)
        res["baselines"][name] = dict(rmse=float(e), rmse_norm=float(nr),
                                      rmse_all=float(e / (vsd_all + EPS)),
                                      ratio_g1=float(nr / (mp + EPS)), **extra)

    # ---------------- baselines -------------------------------------------
    Ztr, Zte = opmod.standardised_injections(Xtr, Xte)

    g1 = gp_batch(gram(Ztr), Ytr, sqd(Zte, Ztr))
    sc = predictive_scores(g1["mu"], g1["pv"], Yte)
    score_base("b1_rbf_gp", g1["mu"], dict(hyper=g1["hyper"], df=g1["df"], **sc))

    for bname, spec in graph_bases.items():
        Phi, P = spec["Phi"], spec["P"]
        Ftr_g, Fte_g = opmod.graph_features(Phi, Xtr, Xte, P.shape[1])
        rg = gp_batch(gram(Ftr_g), Ytr, sqd(Fte_g, Ftr_g))
        scg = predictive_scores(rg["mu"], rg["pv"], Yte)
        score_base(bname, rg["mu"], dict(hyper=rg["hyper"], df=rg["df"],
                                         n_features=int(Phi.shape[1]), **scg))

    r_raw = fit_ridge(Ztr, Ytr, Zte, mode="gcv")
    score_base("b4_ridge_raw110", r_raw["pred"], dict(frac=r_raw["frac"], df=r_raw["df"],
                                                     mode=r_raw["mode"]))

    fam = lindist_family(Vp1_all, Vp2_all, tr, te, Ytr, Yte, vsd_train)
    score_base("b5_lindist", fam["pred"], dict(winner=fam["winner"], order=fam["order"],
                                              route=fam["route"], frac=fam["frac"],
                                              df=fam["df"],
                                              inner_cv_mse=fam["inner_cv_mse"],
                                              n_members=len(fam["members"])))
    res["b5_members"] = fam["members"]
    res["b5_winner"] = fam["winner"]

    # ---------------- treatment: GP on the operator coordinates -------------
    Ftr, Fte, info = opmod.coord_map(S, Xtr, Xte, "iso")
    gS = gp_batch(gram(Ftr), Ytr, sqd(Fte, Ftr))
    scS = predictive_scores(gS["mu"], gS["pv"], Yte)
    score("gp_S45", gS["mu"], dict(df=gS["df"], hyper=gS["hyper"], scale=info["scale"],
                                  n_features=45, **scS))

    Ftr11, Fte11, info11 = opmod.coord_map(S11, Xtr, Xte, "iso")
    gS11 = gp_batch(gram(Ftr11), Ytr, sqd(Fte11, Ftr11))
    sc11 = predictive_scores(gS11["mu"], gS11["pv"], Yte)
    score("gp_S11_aux", gS11["mu"], dict(df=gS11["df"], hyper=gS11["hyper"],
                                        scale=info11["scale"], n_features=11, **sc11))

    # ---------------- linear arms -----------------------------------------
    r = fit_ridge(Ftr, Ytr, Fte, mode="gcv")
    score("lin_S45_iso", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=45,
                                        mode="gcv",
                                        at_grid_endpoint=r["at_grid_endpoint"],
                                        cov_unreliable=r["cov_unreliable"]))
    r_cv = fit_ridge(Ftr, Ytr, Fte, mode="cv")
    score("lin_S45_iso_cv", r_cv["pred"], dict(frac=r_cv["frac"], df=r_cv["df"],
                                              n_features=45, mode="cv",
                                              at_grid_endpoint=r_cv["at_grid_endpoint"],
                                              cov_unreliable=r_cv["cov_unreliable"]))

    Ftr_s, Fte_s, _ = opmod.coord_map(S, Xtr, Xte, "std")
    r = fit_ridge(Ftr_s, Ytr, Fte_s, mode="gcv")
    score("lin_S45_std", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=45,
                                        mode="gcv",
                                        at_grid_endpoint=r["at_grid_endpoint"],
                                        cov_unreliable=r["cov_unreliable"]))

    r = fit_ridge(Ftr11, Ytr, Fte11, mode="gcv")
    score("lin_S11", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=11, mode="gcv",
                                    scale=info11["scale"],
                                    at_grid_endpoint=r["at_grid_endpoint"],
                                    cov_unreliable=r["cov_unreliable"]))

    Ftr_v, Fte_v, _ = opmod.coord_map(V11, Xtr, Xte, "std")
    r = fit_ridge(Ftr_v, Ytr, Fte_v, mode="gcv")
    score("lin_V11", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=11, mode="gcv",
                                    at_grid_endpoint=r["at_grid_endpoint"],
                                    cov_unreliable=r["cov_unreliable"]))

    # ---------------- degree-2 arms ---------------------------------------
    qtr, qte = opmod.quad_features(Ftr, Fte)
    n_feat_q = int(qtr.shape[1])
    if n >= 40:
        r = fit_ridge(qtr, Ytr, qte, mode="gcv")
        score("quad_S45", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=n_feat_q,
                                         mode="gcv",
                                         df_equals_feature_count=bool(
                                             abs(r["df"] - n_feat_q) < 1e-8),
                                         at_grid_endpoint=r["at_grid_endpoint"]))
    else:
        res["arms"]["quad_S45"] = dict(status="under_determined", p=n_feat_q, n=int(n),
                                      df=None, rmse=None, rmse_norm=None,
                                      ratio_to_mean_predictor=None,
                                      note="1080 features against <40 rows is not a "
                                           "reportable RMSE; the well-posed nonlinearity "
                                           "test at this size is quad_S11")

    qtr11, qte11 = opmod.quad_features(Ftr_v, Fte_v)
    r = fit_ridge(qtr11, Ytr, qte11, mode="gcv")
    score("quad_S11", r["pred"], dict(frac=r["frac"], df=r["df"],
                                     n_features=int(qtr11.shape[1]), mode="gcv",
                                     df_equals_feature_count=bool(
                                         abs(r["df"] - qtr11.shape[1]) < 1e-8),
                                     at_grid_endpoint=r["at_grid_endpoint"]))

    # ---------------- Bayesian linear regression ---------------------------
    b = fit_blr(Ftr, Ytr, Fte)
    scb = predictive_scores(b["pred"], b["pv"], Yte)
    score("blr_S45", b["pred"], dict(frac=b["frac"], df=b["df"], n_features=45,
                                    sigma_mult=b["sigma_mult"], sigma2=b["sigma2"],
                                    log_evidence=b["log_evidence"], **scb))

    # ---------------- honesty controls ------------------------------------
    score("lin_raw110", r_raw["pred"], dict(frac=r_raw["frac"], df=r_raw["df"],
                                           n_features=110, mode="gcv"))

    Ptr, Pte, pinfo = opmod.pca_features(Xtr, Xte, k=45)
    r = fit_ridge(Ptr, Ytr, Pte, mode="gcv")
    score("lin_pca45", r["pred"], dict(frac=r["frac"], df=r["df"],
                                      n_features=int(Ptr.shape[1]), mode="gcv",
                                      pca_k=pinfo["k_actual"],
                                      pca_explained=pinfo["explained_variance"]))

    Ftr_r, Fte_r, info_r = opmod.coord_map(Sr, Xtr, Xte, "iso")
    r = fit_ridge(Ftr_r, Ytr, Fte_r, mode="gcv")
    score("lin_rand45", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=45,
                                       mode="gcv", scale=info_r["scale"]))

    Ftr_11r, Fte_11r, info_11r = opmod.coord_map(R11, Xtr, Xte, "iso")
    r = fit_ridge(Ftr_11r, Ytr, Fte_11r, mode="gcv")
    score("lin_rand11", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=11,
                                       mode="gcv", scale=info_11r["scale"]))

    # phase-label shuffle: the same line data, the same linear arm, a permuted
    # load->phase assignment. If this matches lin_S45_iso, the phase awareness is inert.
    Ftr_sh, Fte_sh, info_sh = opmod.coord_map(ops.S_shuf, Xtr, Xte, "iso")
    r = fit_ridge(Ftr_sh, Ytr, Fte_sh, mode="gcv")
    score("lin_S45_phaseshuf", r["pred"], dict(frac=r["frac"], df=r["df"], n_features=45,
                                              mode="gcv", scale=info_sh["scale"]))

    # ---------------- decisive bookkeeping --------------------------------
    base_names = ["b1_rbf_gp", "b2_hop_laplacian_gp", "b3_admittance_laplacian_gp",
                  "b4_ridge_raw110", "b5_lindist"]
    best_name = min(base_names, key=lambda kk: res["baselines"][kk]["rmse_norm"])
    best_base = float(res["baselines"][best_name]["rmse_norm"])
    res["best_baseline_name"] = best_name
    res["best_baseline_rmse_norm"] = best_base
    for name, rec in res["arms"].items():
        if rec.get("rmse_norm") is None:
            continue
        rec["margin"] = float(1.0 - rec["rmse_norm"] / (best_base + EPS))
    return res
