"""Process-pool worker: one task per fold, with the feeder object built once per worker.

The task payload is a small dict; every worker rebuilds the Physics object from the
feeder table (about 0.7 s) and caches the impedance-pair array per transformer mode.
BLAS threads are pinned to 1 per worker so the pool does not oversubscribe the box.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                  # noqa: E402

_PH = None
_DATA = {}


def _physics():
    global _PH
    if _PH is None:
        from feeder import Physics
        _PH = Physics()
    return _PH


def _ensemble(scale):
    if scale not in _DATA:
        from data import load_ensemble
        _DATA[scale] = load_ensemble(scale)
    return _DATA[scale]


def init_worker():
    _physics()


def screen_task(task: dict) -> dict:
    """Run one (regime, n, seed, scheme) fold through the full candidate set."""
    from screen import run_fold
    ph = _physics()
    ens = _ensemble(task["regime"])
    X, Y = ens["X"], ens["Y"]
    tr = np.asarray(task["train"], dtype=int)
    te = np.asarray(task["test"], dtype=int)
    far, s15 = ph.sheet(15)
    vsd = float(Y[tr].std(0).mean())        # TRAIN-ROW normaliser only
    rec = run_fold(X, Y, tr, te, ph, s15, ph.phase, vsd,
                   want_var=True, do_b5=task.get("do_b5", True),
                   do_graph=task.get("do_graph", True),
                   select=task.get("select", "mll"))
    rec.update(regime=task["regime"], n=int(task["n"]), seed=int(task["seed"]),
               scheme=int(task["scheme"]), vsd_train=vsd,
               vsd_ensemble=float(Y.std(0).mean()),
               fingerprint=task["fingerprint"])
    return rec


def select_task(task: dict) -> dict:
    """Same fold, hyperparameters by train-only CV instead of marginal likelihood."""
    from screen import run_fold
    ph = _physics()
    ens = _ensemble(task["regime"])
    X, Y = ens["X"], ens["Y"]
    tr = np.asarray(task["train"], dtype=int)
    te = np.asarray(task["test"], dtype=int)
    far, s15 = ph.sheet(15)
    vsd = float(Y[tr].std(0).mean())
    rec = run_fold(X, Y, tr, te, ph, s15, ph.phase, vsd, want_var=False,
                   do_b5=False, do_graph=task.get("do_graph", False),
                   select="cv")
    rec.update(regime=task["regime"], n=int(task["n"]), seed=int(task["seed"]),
               scheme=int(task["scheme"]), vsd_train=vsd, fingerprint=task["fingerprint"])
    return rec


def mechanism_task(task: dict) -> dict:
    """One metric x target-sheet x fold cell of the mechanism factorial.

    Also used for the width-matched reduction controls (phase aggregates, PCA-11,
    random projection to 11), which are feature maps of the same GP rather than
    operators: the `control` key selects the feature construction.
    """
    from screen import run_fold_metric, norm_rmse, metric_features
    from operators import (truncate_rank, partial_whiten, random_operator_like,
                           shuffled_phase_operator, rotate_injections, build_operator)
    from kernels import gp_fit_predict
    ph = _physics()
    ens = _ensemble(task["regime"])
    X, Y = ens["X"], ens["Y"]
    tr = np.asarray(task["train"], dtype=int)
    te = np.asarray(task["test"], dtype=int)
    out = {"regime": task["regime"], "n": int(task["n"]), "seed": int(task["seed"]),
           "scheme": int(task["scheme"]), "metric": task["metric"],
           "targets": task["target"], "kind": task.get("kind", "metric")}

    if task.get("control"):
        c = task["control"]
        Ftr, Fte = _reduction_features(c, X, tr, te)
        r = gp_fit_predict(Ftr, Fte, Y[tr], want_var=True)
        vsd = float(Y[tr].std(0).mean())
        mu_mean = np.tile(Y[tr].mean(0), (len(te), 1))
        from kernels import nlpd_coverage
        out.update(dict(control=c, rmse=norm_rmse(r["mu"], Y[te], vsd),
                        ell=r["ell"], s2=r["s2"], n2=r["n2"],
                        mean_rmse=norm_rmse(mu_mean, Y[te], vsd),
                        rmse_over_mean=float(norm_rmse(r["mu"], Y[te], vsd) /
                                             norm_rmse(mu_mean, Y[te], vsd)),
                        n_features=int(Ftr.shape[1])))
        out.update(nlpd_coverage(Y[te], r["mu"], r["var"], Y[tr].std(0)))
        return out

    far, s15 = ph.sheet(15)
    if task["kind"] == "near_targets":
        from data import load_near_sheet
        near = load_near_sheet()
        if not near["available"]:
            out["error"] = "near sheet not archived"
            return out
        Yt = near["Y"]
        ok = np.isfinite(Yt).all(1)
        Yt = Yt[ok]
        n_rows = len(Yt)
        if task.get("n_draws") == n_rows:
            # the folds were drawn from the near sheet's OWN pool (120 rows), so they
            # index it directly; filtering the 220-row folds instead would leave too few
            # test rows at n = 10 and would change the scheme-1 block size
            trm, tem = tr, te
            Xt = X[:n_rows]
        else:
            trm = tr[tr < n_rows]
            tem = te[te < n_rows]
            Xt = X[:n_rows]
        if len(trm) != len(tr) or len(tem) < 5:
            out["error"] = (f"near sheet has fewer rows than this fold needs "
                            f"({len(trm)} train / {len(tem)} test of {n_rows} rows)")
            return out
        tg = list(near["buses"])
    else:
        Yt, tg, Xt, trm, tem = Y, list(s15), X, tr, te
    vsd = float(Yt[trm].std(0).mean())
    S = task.get("operator")
    if S is None:
        S = build_operator(ph, tg, ph.phase, "pu_correct", "noload")
    S = np.asarray(S)
    rec = run_fold_metric(S, Xt, Yt, trm, tem, vsd)
    out.update(rec)
    return out


def control_task(task: dict) -> dict:
    """One control cell on one fold (same folds, matched n, fixed loads and targets)."""
    from screen import run_fold_metric
    from operators import (random_operator_like, shuffled_phase_operator,
                           rotate_injections, build_operator)
    from baselines import standardise_train, b4_ridge
    from kernels import gp_fit_predict
    ph = _physics()
    ens = _ensemble(task["regime"])
    X, Y = ens["X"], ens["Y"]
    tr = np.asarray(task["train"], dtype=int)
    te = np.asarray(task["test"], dtype=int)
    far, s15 = ph.sheet(15)
    vsd = float(Y[tr].std(0).mean())
    out = {"regime": task["regime"], "n": int(task["n"]), "seed": int(task["seed"]),
           "scheme": int(task["scheme"]), "control": task["control"]}
    S_true = build_operator(ph, s15, ph.phase, "pu_correct", "noload")
    c = task["control"]
    if c == "shuffled_phase":
        S, _ = shuffled_phase_operator(ph, s15, ph.phase, "pu_correct", "noload", seed=0)
        out["operator_note"] = "operator rebuilt from a permuted load->phase assignment"
    elif c == "random_operator":
        S = random_operator_like(S_true, seed=777)
    elif c == "rotated_injections":
        Xr = rotate_injections(X, seed=31415)
        rec = run_fold_metric(S_true, Xr, Y, tr, te, vsd)
        out.update(rec)
        out["operator_note"] = "injections rotated by a fixed random orthogonal Q"
        return out
    elif c == "physically_moved_phase":
        raise RuntimeError("reserved confirmation evidence -- must not be run here")
    else:
        raise ValueError(c)
    rec = run_fold_metric(S, X, Y, tr, te, vsd)
    out.update(rec)
    return out


def _reduction_features(kind: str, X, tr, te):
    """Width-matched reductions and per-phase aggregates, all fitted on TRAIN rows only.

    Every one of these is a plausible way to get a low-dimensional map from the
    injections.  If any of them matches the operator, the treatment's advantage is
    aggregation or generic dimension reduction, not output-restricted orientation.
    """
    from baselines import standardise_train
    Xtr, Xte = X[tr], X[te]
    if kind in ("phase_aggregate_6", "phase_aggregate_9"):
        def agg(A):
            P = A[:, :55]
            Q = A[:, 55:]
            # per-phase totals on the REAL phases
            ph = _physics().phase
            cols = [P[:, ph == j].sum(1) for j in range(3)] + \
                   [Q[:, ph == j].sum(1) for j in range(3)]
            F = np.stack(cols, axis=1)
            if kind == "phase_aggregate_9":
                imb = np.stack([P[:, ph == (j + 1) % 3].sum(1) -
                                P[:, ph == j].sum(1) for j in range(3)], axis=1)
                imbq = np.stack([Q[:, ph == (j + 1) % 3].sum(1) -
                                 Q[:, ph == j].sum(1) for j in range(3)], axis=1)
                F = np.hstack([F, imb, imbq])
            return F
        Ftr, Fte = agg(Xtr), agg(Xte)
        mx = Ftr.mean(0)
        sx = Ftr.std(0)
        sx = np.where(sx < 1e-12, 1.0, sx)
        return (Ftr - mx) / sx, (Fte - mx) / sx
    if kind == "pca_11":
        Ftrs, Ftes, _, _ = standardise_train(Xtr, Xte)
        mu = Ftrs.mean(0)
        C = np.cov(Ftrs - mu, rowvar=False)
        w, V = np.linalg.eigh(C)
        Pk = V[:, np.argsort(w)[::-1][:11]]
        A = (Ftrs - mu) @ Pk
        B = (Ftes - mu) @ Pk
        sc = np.sqrt(np.einsum("ij,ij->i", A, A).mean()) + 1e-12
        return A / sc, B / sc
    if kind == "randproj_11":
        Ftrs, Ftes, _, _ = standardise_train(Xtr, Xte)
        rng = np.random.default_rng(999)
        R = rng.normal(size=(Ftrs.shape[1], 11)) / np.sqrt(Ftrs.shape[1])
        A = Ftrs @ R
        B = Ftes @ R
        sc = np.sqrt(np.einsum("ij,ij->i", A, A).mean()) + 1e-12
        return A / sc, B / sc
    raise ValueError(kind)


def extra_baseline_task(task: dict) -> dict:
    """Extended-pool baselines: MLP (n=80) and ARD-RBF (n in 20/40/80), 3 far outputs."""
    from baselines import b6_mlp, standardise_train
    from kernels import ard_rbf, nlpd_coverage
    from screen import norm_rmse
    ph = _physics()
    ens = _ensemble(task["regime"])
    X, Y = ens["X"], ens["Y"]
    tr = np.asarray(task["train"], dtype=int)
    te = np.asarray(task["test"], dtype=int)
    far, s15 = ph.sheet(15)
    vsd = float(Y[tr].std(0).mean())
    out = {"regime": task["regime"], "n": int(task["n"]), "seed": int(task["seed"]),
           "scheme": int(task["scheme"]), "baseline": task["baseline"]}
    if task["baseline"] == "b6_mlp":
        r = b6_mlp(X[tr], X[te], Y[tr])
        out["rmse"] = norm_rmse(r["mu"], Y[te], vsd)
        out["seed_spread"] = r["seed_spread"]
    elif task["baseline"] == "b7_ard":
        # 3 outputs: the three phases of one far bus (the middle one of the sheet)
        b = 7
        cols = [3 * b + k for k in range(3)]
        Ftr, Fte, _, _ = standardise_train(X[tr], X[te])
        r = ard_rbf(Ftr, Fte, Y[tr][:, cols])
        out["rmse"] = norm_rmse(r["mu"], Y[te][:, cols], vsd)
        out["ls_summary"] = r["ls_summary"]
        out["outputs"] = [int(s15[b]), "a/b/c"]
    return out
