"""Baseline pool.  Every baseline is fitted on the same fold's train rows and
scored on the same test block, with all tuning done on train rows only.

Canonical pool (decides the pre-registered verdicts)
  b1  isotropic RBF Gaussian process on per-column-standardised injections
  b2  hop-count graph-Laplacian kernel, two forms, best kept
  b3  admittance-weighted graph-Laplacian kernel (built from line sequence
      impedances only -- never from any sensitivity Jacobian)
  b4  ridge on standardised injections, scale-free penalty grid, train-only CV
  b5  multiphase LinDistFlow order 2 with the exact second-order Z-bus current
      correction, per-output ridge calibration, penalty and cell by train-only CV
      (16-cell factorial: order x V0 convention x target standardisation x grid)

Extended pool (reported as a conservative headline, does not decide verdicts)
  b6  small MLP on standardised injections (n = 80 only, 3 initialisations)
  b7  ARD-RBF with genuine per-dimension lengthscales (3 outputs, n in 20/40/80)
  plus the impedance-weighted Laplacian and the unrestricted 55-d Z-bus metric

Deliberate exclusion from the canonical pool, stated: the unrestricted 55x55
admittance Z-bus metric.  It is the full-state metric's load-restricted cousin,
and putting it in the pool would let a same-object comparison suppress the
treatment's margin.  It appears only in the mechanism section, where it is the point.
"""
from __future__ import annotations

import itertools
import numpy as np

from kernels import gp_fit_predict, sqdist, gram_dist, ELL_GRID

N_LOAD = 55
B5_ORDERS = (1, 2)
B5_V0MODES = ("flat", "noload")
B5_STANDARDISE = (False, True)
B5_GRIDS = ("absolute", "scale_free")
B5_ABSOLUTE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
B5_SCALEFREE_FRACTIONS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


# ---------------------------------------------------------------- b1 / features
def standardise_train(Xtr: np.ndarray, Xte: np.ndarray):
    mx = Xtr.mean(0)
    sx = Xtr.std(0)
    sx = np.where(sx < 1e-12, 1.0, sx)
    return (Xtr - mx) / sx, (Xte - mx) / sx, mx, sx


def b1_rbf_injections(Xtr, Xte, Ytr, want_var=False, select='mll') -> dict:
    Ftr, Fte, mx, sx = standardise_train(Xtr, Xte)
    out = gp_fit_predict(Ftr, Fte, Ytr, want_var=want_var, select=select)
    out["features"] = "per-column-standardised injections (110-d)"
    return out


# ------------------------------------------------- graph kernels b2 / b3
def _laplacian_from_weights(W: np.ndarray) -> np.ndarray:
    W = 0.5 * (W + W.T)
    return np.diag(W.sum(1)) - W


def hop_laplacian(physics, phase_labels) -> np.ndarray:
    """Unweighted load-graph Laplacian over shared-path adjacency (55 x 55)."""
    import networkx as nx
    n = physics.nload
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lca = nx.lowest_common_ancestor(physics.tree, int(physics.lbus[i]),
                                            int(physics.lbus[j]))
            # shared path length in hops from the source to the LCA
            W[i, j] = float(physics.hop[int(lca)])
    return _laplacian_from_weights(W)


def admittance_laplacian(physics, phase_labels) -> np.ndarray:
    """Load-graph Laplacian with edge weight 1/|Z_shared| (55 x 55).

    Z_shared(l,l') is the sequence-impedance block accumulated from the source to
    the pair's lowest common ancestor, restricted to the two loads' phases.  It is
    built from the line table only -- no sensitivity Jacobian, no phase expansion.
    """
    import networkx as nx
    n = physics.nload
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lca = nx.lowest_common_ancestor(physics.tree, int(physics.lbus[i]),
                                            int(physics.lbus[j]))
            Z = physics.Zto[int(lca)]
            z = Z[int(phase_labels[i]), int(phase_labels[j])]
            mag = abs(z)
            if mag < 1e-14:
                continue
            W[i, j] = W[j, i] = 1.0 / mag
    return _laplacian_from_weights(W)


def _metric_of_laplacian(L: np.ndarray) -> np.ndarray:
    """PSD pseudo-inverse of a (possibly singular) graph Laplacian."""
    Ls = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(Ls)
    inv = np.where(w > max(1e-10 * w.max(), 1e-14), 1.0 / np.maximum(w, 1e-300), 0.0)
    M = (V * inv) @ V.T
    return 0.5 * (M + M.T)


def hop_metric_psd(physics, phase_labels) -> np.ndarray:
    """Metric form of b2: PSD projection of the hop-distance matrix."""
    import networkx as nx
    n = physics.nload
    D = np.zeros((n, n))
    hp = np.array([physics.hop[int(b)] for b in physics.lbus], dtype=float)
    for i in range(n):
        for j in range(n):
            lca = nx.lowest_common_ancestor(physics.tree, int(physics.lbus[i]),
                                            int(physics.lbus[j]))
            D[i, j] = hp[i] + hp[j] - 2.0 * physics.hop[int(lca)]
    # classical MDS -> PSD Gram of squared distances
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(0.5 * (B + B.T))
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T


def _metric_features(M: np.ndarray, Xtr, Xte):
    """Apply a 55x55 magnitude metric to the P and Q blocks separately."""
    Ptr, Pte = Xtr[:, :N_LOAD], Xte[:, :N_LOAD]
    Qtr, Qte = Xtr[:, N_LOAD:], Xte[:, N_LOAD:]
    Ftr = np.hstack([Ptr @ M, Qtr @ M])
    Fte = np.hstack([Pte @ M, Qte @ M])
    mx = Ftr.mean(0)
    Ftr = Ftr - mx
    Fte = Fte - mx
    sc = np.sqrt(np.einsum("ij,ij->i", Ftr, Ftr).mean()) + 1e-12
    return Ftr / sc, Fte / sc


def graph_kernel_candidate(M: np.ndarray, Xtr, Xte, Ytr, want_var=False,
                           select='mll') -> dict:
    Ftr, Fte = _metric_features(M, Xtr, Xte)
    return gp_fit_predict(Ftr, Fte, Ytr, want_var=want_var, select=select)


def b2_hop_count(physics, phase_labels, Xtr, Xte, Ytr, want_var=False,
                 select='mll') -> dict:
    """Two forms, each given its best train-only score; the better one is kept."""
    out = {}
    M1 = hop_metric_psd(physics, phase_labels)
    out["metric_form"] = graph_kernel_candidate(M1, Xtr, Xte, Ytr, want_var, select)
    Lh = hop_laplacian(physics, phase_labels)
    M2 = _metric_of_laplacian(Lh)
    out["pinv_form"] = graph_kernel_candidate(M2, Xtr, Xte, Ytr, want_var, select)
    score_key = "mll" if select == "mll" else "cv_score"
    best = ("metric_form" if out["metric_form"].get(score_key, -np.inf) >=
            out["pinv_form"].get(score_key, -np.inf) else "pinv_form")
    res = dict(out[best])
    res["form"] = best
    res["forms_mll"] = {k: float(v["mll"]) for k, v in out.items()}
    res["features"] = f"hop-count graph-Laplacian metric ({best})"
    return res


def b3_admittance_laplacian(physics, phase_labels, Xtr, Xte, Ytr, want_var=False,
                            select='mll') -> dict:
    Lw = admittance_laplacian(physics, phase_labels)
    M = _metric_of_laplacian(Lw)
    res = graph_kernel_candidate(M, Xtr, Xte, Ytr, want_var, select)
    res["features"] = "admittance-weighted graph-Laplacian metric (pinv form)"
    return res


def b3_impedance_weighted(physics, phase_labels, Xtr, Xte, Ytr, want_var=False) -> dict:
    """The impedance-weighted variant kept as an extended-pool demonstration."""
    Lw = admittance_laplacian(physics, phase_labels)
    M = hop_metric_psd(physics, phase_labels) @ np.eye(physics.nload)  # placeholder-free
    M = _metric_of_laplacian(Lw)
    res = graph_kernel_candidate(M, Xtr, Xte, Ytr, want_var)
    res["features"] = "impedance-weighted Laplacian (extended pool demonstration)"
    return res


def full_state_z_metric(physics, phase_labels) -> np.ndarray:
    """Unrestricted 55x55 Z-bus magnitude metric: |Z_bus(load_i, load_j)| at the
    loads' own phases, symmetrised, PSD-projected.  Built from line data only."""
    n = physics.nload
    Z = np.zeros((n, n))
    import networkx as nx
    for i in range(n):
        for j in range(n):
            lca = nx.lowest_common_ancestor(physics.tree, int(physics.lbus[i]),
                                            int(physics.lbus[j]))
            Z[i, j] = abs(physics.Zto[int(lca)][int(phase_labels[i]), int(phase_labels[j])])
    Z = 0.5 * (Z + Z.T)
    w, V = np.linalg.eigh(Z)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T


# ------------------------------------------------------------------ b4 ridge
def _cv_folds(n: int, k: int) -> list[np.ndarray]:
    k = max(2, min(k, n // 2))
    return [f for f in np.array_split(np.arange(n), k) if len(f) > 0]


def b4_ridge(Xtr, Xte, Ytr, want_var=False) -> dict:
    """Ridge on standardised injections with a SCALE-FREE penalty grid.

    The grid is a set of fractions of the largest eigenvalue of the centred Gram,
    spanning 1e-8 to 10 in decade steps, chosen by train-only k-fold CV with
    k = max(2, n//4).  An absolute grid was recorded in this run tree as producing
    normalised RMSE of order 50 (protocol check, guard g1).
    """
    Ftr, Fte, mx, sx = standardise_train(Xtr, Xte)
    n, p = Ftr.shape
    # an INTERCEPT is mandatory: the standardised features are zero-mean per column
    # while the targets sit near 1.0 pu, so a no-intercept ridge returns a zero-mean
    # prediction and a normalised RMSE of order 50.  That failure mode was recorded in
    # this run tree and is reproduced here deliberately so the fix is explicit.
    ybar = Ytr.mean(0)
    Yc = Ytr - ybar
    G = Ftr.T @ Ftr
    lam_max = float(np.linalg.eigvalsh(G)[-1])
    grid = [f * lam_max for f in B5_SCALEFREE_FRACTIONS]
    k = max(2, n // 4)
    folds = _cv_folds(n, k)
    scores = []
    for lam in grid:
        s = 0.0
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 3:
                continue
            G1 = Ftr[tri].T @ Ftr[tri]
            c = np.linalg.solve(G1 + lam * np.eye(p), Ftr[tri].T @ Yc[tri])
            s += float(np.mean((Ftr[f] @ c - Yc[f]) ** 2)) / len(folds)
        scores.append(s)
    lam = grid[int(np.argmin(scores))]
    c = np.linalg.solve(G + lam * np.eye(p), Ftr.T @ Yc)
    mu = Fte @ c + ybar
    out = dict(mu=mu, lam=float(lam), lam_max=float(lam_max), n_cv_folds=int(k),
               features="ridge on standardised injections, scale-free grid")
    if want_var:
        # a Gaussian K=1 predictive variance in LOG-voltage space, since the levels of
        # this target are far from zero and a raw-unit Gaussian would be meaningless
        mb, sb = Ytr.mean(0), Ytr.std(0) + 1e-12
        mu_tr = Ftr @ c
        v = np.mean(((Ytr - mu_tr) / sb) ** 2, axis=0) + 1e-6
        out["var"] = np.tile(v[None, :] * sb[None, :] ** 2, (mu.shape[0], 1)) + 1e-18
    return out


# -------------------------------------------------- b5 LinDistFlow (reconciliation)
def _ridge_calibrate(Vtr: np.ndarray, Vte: np.ndarray, Ytr: np.ndarray,
                     grid, standardise: bool, n_cv: int) -> tuple[np.ndarray, float]:
    """Per-output ridge calibration of a physics prediction, penalty by train-only CV."""
    if standardise:
        mb, sb = Ytr.mean(0), Ytr.std(0) + 1e-12
        ytr = (Ytr - mb) / sb
    else:
        mb, sb = np.zeros(Ytr.shape[1]), np.ones(Ytr.shape[1])
        ytr = Ytr
    Atr = np.hstack([Vtr, np.ones((len(Vtr), 1))])
    Ate = np.hstack([Vte, np.ones((len(Vte), 1))])
    n, p = Atr.shape
    k = max(2, min(n_cv, n // 2))
    folds = _cv_folds(n, k)
    scores = []
    for lam in grid:
        s = 0.0
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 3:
                continue
            G = Atr[tri].T @ Atr[tri]
            c = np.linalg.solve(G + lam * np.eye(p), Atr[tri].T @ ytr[tri])
            s += float(np.mean((Atr[f] @ c - ytr[f]) ** 2)) / len(folds)
        scores.append(s)
    lam = grid[int(np.argmin(scores))]
    G = Atr.T @ Atr
    c = np.linalg.solve(G + lam * np.eye(p), Atr.T @ ytr)
    pred = (Ate @ c) * sb + mb
    # train-only CV score in the ORIGINAL units, so cells are comparable
    if standardise:
        cv_rmse = float(np.sqrt(min(scores)))
    else:
        cv_rmse = float(np.sqrt(min(scores)))
    return pred, cv_rmse


def b5_lindistflow_grid(physics, Xtr, Xte, Ytr, targets, phase_labels,
                        v0_caches: dict) -> dict:
    """The 16-cell factorial, selected by TRAIN-ONLY cross-validated error.

    Cells: (order in {1,2}) x (V0 in {flat, noload}) x (target standardisation
    inside the calibration on/off) x (penalty grid absolute vs scale-free).
    Every cell's TEST error is recorded so the disagreement between the two prior
    artifacts' readings of this baseline is visible rather than smoothed.
    """
    cells = {}
    n = len(Xtr)
    n_cv = max(2, n // 4)
    for order, v0m, std, gridkind in itertools.product(
            B5_ORDERS, B5_V0MODES, B5_STANDARDISE, B5_GRIDS):
        key = f"order{order}_v0-{v0m}_std{int(std)}_{gridkind}"
        Vfull = physics.lindist(np.vstack([Xtr, Xte]), targets, phase_labels,
                                order=order, v0mode=v0m, tr_mode="pu_correct")
        Vtr, Vte = Vfull[:n], Vfull[n:]
        if gridkind == "absolute":
            grid = list(B5_ABSOLUTE_GRID)
        else:
            A = np.hstack([Vtr, np.ones((n, 1))])
            lam_max = float(np.linalg.eigvalsh(A.T @ A)[-1])
            grid = [f * lam_max for f in B5_SCALEFREE_FRACTIONS]
        pred, cv = _ridge_calibrate(Vtr, Vte, Ytr, grid, std, n_cv)
        cells[key] = dict(pred=pred, cv_rmse=cv,
                          config=dict(order=order, v0=v0m, standardise=bool(std),
                                      grid=gridkind))
    best_key = min(cells, key=lambda k: cells[k]["cv_rmse"])
    res = dict(cells[best_key])
    res["cell"] = best_key
    res["all_cells_cv"] = {k: float(v["cv_rmse"]) for k, v in cells.items()}
    res["all_cells_pred"] = {k: v["pred"] for k, v in cells.items()}
    res["features"] = f"LinDistFlow order-2 / Z-bus correction ({best_key})"
    return res


def b5_legacy_cell(physics, Xtr, Xte, Ytr, targets, phase_labels) -> dict:
    """The 'legacy' reading: order 2, no-load V0, no target standardisation,
    absolute penalty grid.  Reported beside the selected cell so the 1.7x
    cross-artifact disagreement stays visible."""
    n = len(Xtr)
    Vfull = physics.lindist(np.vstack([Xtr, Xte]), targets, phase_labels,
                            order=2, v0mode="noload", tr_mode="pu_correct")
    Vtr, Vte = Vfull[:n], Vfull[n:]
    pred, cv = _ridge_calibrate(Vtr, Vte, Ytr, list(B5_ABSOLUTE_GRID), False,
                                max(2, n // 4))
    return dict(pred=pred, cv_rmse=cv,
                cell="order2_v0-noload_std0_absolute",
                features="LinDistFlow order-2 legacy cell")


# ----------------------------------------------------------------------- b6 MLP
def b6_mlp(Xtr, Xte, Ytr, seeds=(0, 1, 2), hidden=(32, 32), steps=2000,
           lr=1e-2) -> dict:
    """Small MLP on standardised injections, tanh, Adam, mean of 3 initialisations.

    Implemented in numpy so the whole pipeline has one dependency set.  Only used
    at n = 80 in the 2.0x regime (reduced scope, see the artifact plan).
    """
    Ftr, Fte, mx, sx = standardise_train(Xtr, Xte)
    Ybar = Ytr.mean(0)
    Ysd = Ytr.std(0) + 1e-12
    Yc = (Ytr - Ybar) / Ysd
    preds = []
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        dims = [Ftr.shape[1], *hidden, Yc.shape[1]]
        Ws, bs = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            Ws.append(rng.normal(0.0, np.sqrt(2.0 / a), (a, b)))
            bs.append(np.zeros(b))
        params = Ws + bs
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = len(Ftr)
        L = len(Ws)
        for step in range(1, steps + 1):
            # forward
            acts = [Ftr]
            h = Ftr
            for i in range(L - 1):
                h = np.tanh(h @ Ws[i] + bs[i])
                acts.append(h)
            out = h @ Ws[-1] + bs[-1]
            g = 2.0 * (out - Yc) / (n * Yc.shape[1])
            # backward
            gW = [None] * L
            gb = [None] * L
            gW[L - 1] = acts[L - 1].T @ g
            gb[L - 1] = g.sum(0)
            gh = g @ Ws[L - 1].T
            for i in range(L - 2, -1, -1):
                gz = gh * (1.0 - acts[i + 1] ** 2)
                gW[i] = acts[i].T @ gz
                gb[i] = gz.sum(0)
                if i > 0:
                    gh = gz @ Ws[i].T
            grads = gW + gb
            # Adam; parameters are reassigned into their own list (never `-=` on an
            # aliased element, which silently broadcast a (p,) step onto a (d,p) block
            # and is the exact failure this replaces)
            for j in range(len(params)):
                gr = grads[j]
                m[j] = b1 * m[j] + (1 - b1) * gr
                v[j] = b2 * v[j] + (1 - b2) * gr ** 2
                mh = m[j] / (1 - b1 ** step)
                vh = v[j] / (1 - b2 ** step)
                params[j] = params[j] - lr * mh / (np.sqrt(vh) + eps)
            Ws = params[:L]
            bs = params[L:]
        h = Fte
        for i in range(len(Ws) - 1):
            h = np.tanh(h @ Ws[i] + bs[i])
        preds.append((h @ Ws[-1] + bs[-1]) * Ysd + Ybar)
    mu = np.mean(preds, axis=0)
    spread = float(np.mean(np.std(np.stack(preds), axis=0)))
    return dict(mu=mu, seed_spread=spread, n_seeds=len(seeds),
                features="MLP 110-32-32-45, tanh, Adam, mean of 3 initialisations")
