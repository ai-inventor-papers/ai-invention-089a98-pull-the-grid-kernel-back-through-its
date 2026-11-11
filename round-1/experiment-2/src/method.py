#!/usr/bin/env python3
"""Screen candidate B - whiten versus truncate the pullback sensitivity kernel.

Question
--------
Does partially whitening the 45x110 pullback sensitivity operator S before the
RBF (the "grid kernel" / Kirchhoff-sensitivity kernel) repair the raw kernel's
small-n failure, and does rank truncation do anything at all?

Design (pre-registered in 3_invention_loop/iter_1/gen_art/gen_art_experiment_2)
-----------------------------------------------------------------------------
Ground truth : archived 220-draw runpp_3ph ensemble of the IEEE PES European LV
               test feeder (pandapower `ieee_european_lv_asymmetric`), 2.0x
               stressed regime, 55 single-phase loads on their real phases,
               45 far-end outputs = 15 far-end buses x a,b,c.  REUSED, not
               re-solved on the measurement path.
Operator     : rebuilt analytically from line data (ZERO AC solves).
Treatment    : 2-D grid, truncation rank k x partial-whitening exponent gamma,
               plus a data-independent singular-value-floor family.
Baselines    : b1a median-normalised RBF on injections, b1b raw-scale RBF,
               b2 hop-count graph kernel, b3 admittance-weighted multiphase
               (110-dim) Kirchhoff kernel, b4 scale-free ridge,
               b5p honest per-output ridge on multiphase LinDistFlow order 2
               (the physics / crossover reference), b5s the archived p17
               joint-map calibration kept as a labelled secondary row only.
Folds        : 12 seeds x 2 partition schemes, n in {10,20,40,80,160}.
               24 folds at n<=80, 12 at n=160 -> 108 fold-cells.
Metric       : far-end test RMSE in per-unit voltage, normalised by the mean
               per-output standard deviation of the TRAINING rows of each fold.

Stages
------
  checks     - input assertions, operator rebuild, structural facts
  baselines  - per-fold baseline table (shared by every candidate arm)
  sweep      - main k x gamma and floor grid, checkpointed per (n, fold)
  select     - train-only selection of (k, gamma) + shuffled-phase control
  stats      - aggregation, bootstrap/Wilcoxon, figures, JSON products
  all        - run every stage in order
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"
for _d in (DATA, RESULTS, RAW, FIGS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS / "run.log"), rotation="30 MB", level="DEBUG", enqueue=False)

# ---------------------------------------------------------------------------
# pre-registered protocol constants
# ---------------------------------------------------------------------------
SEED_BASE = 20261111
SEEDS = tuple(range(12))
SCHEMES = (0, 1)
N_GRID = (10, 20, 40, 80, 160)
K_GRID = (3, 5, 8, 10, 11, 15, 20, 45)
G_GRID = (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00)
F_GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
K_GRID_COARSE = (3, 8, 11, 20, 45)
G_GRID_COARSE = (0.0, 0.30, 0.50, 0.70)
LGRID = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
S2G = (0.3, 1.0, 3.0)
N2G = (1e-8, 1e-5, 1e-3, 1e-1)
SEL_REPS = 5
JITTERS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
N_PROC = 2

CANDIDATE_ID = "B_whitening_rank"
MECHANISM = (
    "partial whitening of the target-restricted sensitivity operator re-weights its "
    "near-null directions, while rank truncation discards them"
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def jdump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True))


def jload(path: Path) -> Any:
    return json.loads(path.read_text())


def sqd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, guarded against negative round-off."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0.0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def folds_for(n: int) -> list[dict[str, Any]]:
    """The shared screen fold recipe.

    scheme 0 = complement split (train = first n of a full permutation, test = rest)
    scheme 1 = fixed-half split (train = first n of the first 110 of the permutation,
               test = the other 110).  At n=160 scheme 1 cannot draw n from 110 rows
               and the cell is skipped, so n=160 has 12 folds not 24.
    """
    out = []
    for seed in SEEDS:
        for scheme in SCHEMES:
            rng = np.random.default_rng(SEED_BASE + 7919 * seed + 131 * scheme)
            idx = rng.permutation(220)
            if scheme == 0:
                tr, te = idx[:n], idx[n:]
            else:
                h = 110
                tr, te = idx[:h][:n], idx[h:]
            if len(tr) < n or len(te) < 3:
                continue
            out.append(
                dict(
                    n=int(n),
                    seed=int(seed),
                    scheme=int(scheme),
                    fold_id=int(n * 100 + seed * 2 + scheme),
                    tr=tr.astype(int),
                    te=te.astype(int),
                )
            )
    return out


# ---------------------------------------------------------------------------
# operator construction
# ---------------------------------------------------------------------------
def _import_physics():
    sys.path.insert(0, str(DATA))
    from physics import Physics  # type: ignore

    return Physics()


def build_S(ph, targets: Sequence[int], phase: np.ndarray, v0: float = 1.05) -> np.ndarray:
    """d|V| / d(P,Q) in pu per MW / MVAr, shape (3*T, 2*L).

    Reproduces Physics.lin_operator but with an explicit phase-label argument so the
    shuffled-phase control can rebuild the same operator on permuted labels.
    """
    angs = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])
    V0v = v0 * np.exp(1j * angs)
    n = len(phase)
    Zt = np.zeros((3 * len(targets), n), dtype=complex)
    for i, tb in enumerate(targets):
        for l in range(n):
            Zt[3 * i:3 * i + 3, l] = ph.shared(ph.lbus[l], tb)[:, phase[l]]
    Zt = Zt / ph.Zb
    V0t = np.tile(V0v, len(targets))
    V0l = V0v[phase]
    Wt = np.conj(V0t) / np.abs(V0t)
    dPd = -Zt / (ph.Sb * np.conj(V0l))[None, :]
    S = np.zeros((Zt.shape[0], 2 * n))
    S[:, :n] = np.real(Wt[:, None] * dPd)
    S[:, n:] = np.real(Wt[:, None] * (1j * dPd))
    return S


def op_whiten(U, sv, Vt, k: int, gamma: float) -> np.ndarray:
    """The (k, 110) feature map  diag(w_1..w_k) @ Vt[:k, :]  with
    w_i = (sv_i / sv_0) ** (1 - gamma).

    gamma = 0 -> w_i = sv_i/sv_0   (raw scaled operator truncated at k)
    gamma = 1 -> w_i = 1           (full whitening, all k directions at equal weight)

    NOTE this is the ratio form; the direction text states the exponent the other way
    round (it calls gamma=0 the raw operator while writing the fully whitened map).
    The ratio form is what the working pilot code uses and what STEP 3 asserts.
    """
    w = (sv[:k] / sv[0]) ** (1.0 - gamma)
    return w[:, None] * Vt[:k, :]


def op_floor(U, sv, Vt, f: float) -> np.ndarray:
    """gamma = 0 with every singular value below f * sv_max zeroed.

    Equivalent to truncation at k(f) but reported as a data-independent family.
    """
    ratio = sv / sv[0]
    mask = ratio >= f
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        idx = np.array([0])
    return ratio[idx][:, None] * Vt[idx, :]


def featurise(M: np.ndarray, Xtr: np.ndarray, Xte: np.ndarray):
    """Features renormalised by the TRAIN RMS feature norm (leak-free)."""
    Ftr = Xtr @ M.T
    Fte = Xte @ M.T
    sc = math.sqrt(float((Ftr ** 2).sum(1).mean())) + 1e-12
    return Ftr / sc, Fte / sc


# ---------------------------------------------------------------------------
# Gaussian process (multi-output, one shared hyperparameter triple)
# ---------------------------------------------------------------------------
def gp_corr(corr: list[tuple[Any, np.ndarray, np.ndarray]], Ytr: np.ndarray,
            s2g: Sequence[float] = S2G, n2g: Sequence[float] = N2G,
            mll_mode: str = "archived"):
    """Shared-hyperparameter multi-output GP given candidate correlation matrices.

    corr: list of (label, Ktr, Kte) with Ktr (n,n) and Kte (m,n) correlations in [~0,1].
    Standardises Y per output on TRAIN rows, picks (label, s2, n2) by one shared
    criterion over all 45 outputs, and returns (mu, pv, chosen, Ktr, Kte).

    Hyperparameter selection criterion.  The p17 / p8_full / pilot convention -
    reproduced by the plan pseudocode - scores
        -0.5 * ||K^-1 Yc||^2_F  - 0.5 * m * log|K|,
    i.e. the squared norm of the solved coefficients.  The marginal likelihood is
        -0.5 * tr(Yc^T K^-1 Yc)  - 0.5 * m * log|K|,
    which replaces ||K^-1 Yc||^2 by the quadratic form sum(VtY^2 / d).  These are
    different statistics whenever K is not a multiple of the identity, and the first
    is not a likelihood.  mll_mode="archived" keeps the archived convention so this
    screen's arms are comparable with the earlier tables; mll_mode="quadratic" is the
    correct marginal likelihood and is reported as a sensitivity row.  Both are
    computed from the same eigendecomposition, so the comparison is free.

    The eigendecomposition of Ktr (PSD up to round-off; negative eigenvalues clipped
    at 0) makes every (s2, n2) point O(n^2) and is verified against a dense Cholesky
    reference to 2e-15 in test_gp.py.
    """
    n, m = Ytr.shape
    yb = Ytr.mean(0)
    ys = Ytr.std(0) + 1e-12
    Yc = (Ytr - yb) / ys
    best = None
    best_q = None
    for lab, Ktr, Kte in corr:
        ev, V = np.linalg.eigh(0.5 * (Ktr + Ktr.T))
        ev = np.clip(ev, 0.0, None)
        VtY = V.T @ Yc                                  # (n, m)
        for s2 in s2g:
            for n2 in n2g:
                d = s2 * ev + n2
                logdet = float(np.log(d).sum())
                alpha = VtY / d[:, None]                # = V^T K^-1 Yc
                a2 = float((alpha ** 2).sum())          # ||K^-1 Yc||^2_F
                quad = float((VtY ** 2 / d[:, None]).sum())   # tr(Yc^T K^-1 Yc)
                sc_arch = -0.5 * a2 - 0.5 * m * logdet
                sc_quad = -0.5 * quad - 0.5 * m * logdet
                if not np.isfinite(sc_arch) or not np.isfinite(sc_quad):
                    continue
                if best is None or sc_arch > best[0]:
                    best = (sc_arch, lab, float(s2), float(n2), ev, V, VtY, Kte, Ktr)
                if best_q is None or sc_quad > best_q[0]:
                    best_q = (sc_quad, lab, float(s2), float(n2), ev, V, VtY, Kte, Ktr)
    chosen = best if mll_mode == "archived" else best_q
    if chosen is None:
        lab, Ktr, Kte = corr[0]
        ev, V = np.linalg.eigh(0.5 * (Ktr + Ktr.T))
        ev = np.clip(ev, 0.0, None)
        chosen = (float("-inf"), lab, 1.0, 1e-1, ev, V, V.T @ Yc, Kte, Ktr)
    _, lab, s2, n2, ev, V, VtY, Kte, Ktr = chosen
    d = s2 * ev + n2
    alpha = VtY / d[:, None]
    a = V @ alpha
    Ks = s2 * Kte
    mu = (Ks @ a) * ys + yb
    # diag(Ks K^-1 Ks^T), K^-1 = V diag(1/d) V^T
    VtKs = Kte @ V                                      # (n_te, n)
    quad = (s2 ** 2) * ((VtKs ** 2) / d[None, :]).sum(1)
    pv = np.maximum(s2 - quad, 1e-12)[:, None] * (ys ** 2)[None, :]
    return mu, pv, (lab, float(s2), float(n2)), Ktr, Kte


def gp_train_fit(Dtr: np.ndarray, Ytr: np.ndarray, hp: tuple, jitter: float = 0.0):
    """In-sample TRAIN predictions at a FIXED hyperparameter triple hp = (l, s2, n2).

    Used only for the tempered-noise diagnostic, which needs a train-only residual
    variance. The fit is K^-1 applied to the train block alone; the test block is never
    touched, so this cannot leak. Returns predictions in the original Y units.
    """
    lab, s2, n2 = hp
    n, m = Ytr.shape
    yb = Ytr.mean(0)
    ys = Ytr.std(0) + 1e-12
    Yc = (Ytr - yb) / ys
    K = s2 * Dtr + (n2 + jitter) * np.eye(n)
    ev, V = np.linalg.eigh(0.5 * (K + K.T))
    ev = np.clip(ev, 0.0, None)
    alpha = (V.T @ Yc) / ev[:, None]
    a = V @ alpha
    mu = (K @ a) * ys + yb
    return mu


def _heldout_resid_var(Ktr: np.ndarray, Ytr: np.ndarray, hp: tuple, seed: int) -> float:
    """Mean squared held-out residual over the TRAIN rows, at fixed hyperparameters.

    Two-way half split of the train block, both directions averaged. Uses only train
    rows, so it is a legal noise calibration for the tempered-noise diagnostic row.
    """
    lab, s2, n2 = hp
    n, m = Ytr.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    h = n // 2
    if h < 2 or n - h < 2:
        return float(np.var(Ytr, axis=0).mean())
    tot, cnt = 0.0, 0
    for fit, sco in ((perm[:h], perm[h:]), (perm[h:], perm[:h])):
        Kff = s2 * Ktr[np.ix_(fit, fit)] + n2 * np.eye(len(fit))
        Ksf = s2 * Ktr[np.ix_(sco, fit)]
        yb = Ytr[fit].mean(0)
        ys = Ytr[fit].std(0) + 1e-12
        Yc = (Ytr[fit] - yb) / ys
        ev, V = np.linalg.eigh(0.5 * (Kff + Kff.T))
        ev = np.clip(ev, 0.0, None)
        a = V @ ((V.T @ Yc) / ev[:, None])
        pred = (Ksf @ a) * ys + yb
        tot += float(((pred - Ytr[sco]) ** 2).sum())
        cnt += Ytr[sco].size
    return tot / max(cnt, 1)


def rbf_corr(Dtr: np.ndarray, Dte: np.ndarray, lgrid: Sequence[float] = LGRID):
    """Correlation candidates exp(-D / (2 l^2)) for a squared-distance matrix."""
    return [(float(l), np.exp(-Dtr / (2.0 * l * l)), np.exp(-Dte / (2.0 * l * l)))
            for l in lgrid]


def score_gp(mu, pv, Yte, vsd: float) -> dict[str, float]:
    e = rmse(mu, Yte) / vsd
    z = (Yte - mu) / np.sqrt(pv)
    nlpd = float(-np.mean(0.5 * np.log(2 * np.pi * pv) + 0.5 * z ** 2))
    return dict(rmse=float(e), nlpd=nlpd,
                c68=float(np.mean(np.abs(z) <= 1.0)),
                c95=float(np.mean(np.abs(z) <= 1.96)))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def ridge_scale_free(Ztr: np.ndarray, Zte: np.ndarray, Ytr: np.ndarray) -> np.ndarray:
    """b4 - ridge on standardised injections, scale-free penalty, unpenalised intercept."""
    n = len(Ztr)
    Atr = np.hstack([Ztr, np.ones((n, 1))])
    Ate = np.hstack([Zte, np.ones((len(Zte), 1))])
    k = max(2, n // 4)
    folds = np.array_split(np.arange(n), k)
    D = np.ones(Atr.shape[1])
    D[-1] = 0.0
    # precompute every fold's Gram and cross-product ONCE (25 lambdas reuse them)
    pre = []
    for f in folds:
        tri = np.setdiff1d(np.arange(n), f)
        if len(tri) < 3:
            continue
        A1 = Atr[tri]
        G1 = A1.T @ A1
        pre.append((f, Atr[f], G1, A1.T @ Ytr[tri],
                    float(np.linalg.eigvalsh(G1).max()) / A1.shape[1]))
    best = None
    for lam in np.logspace(-6, 6, 25):
        sse = 0.0
        for f, Af, G1, rhs, sc1 in pre:
            try:
                c = np.linalg.solve(G1 + (lam * sc1) * np.diag(D), rhs)
            except np.linalg.LinAlgError:
                sse += 1e18
                continue
            sse += float(((Af @ c - Ytr[f]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lam)
    lam = best[1]
    Gbase = Atr.T @ Atr
    scale = float(np.linalg.eigvalsh(Gbase).max()) / Atr.shape[1]
    c = np.linalg.solve(Gbase + (lam * scale) * np.diag(D), Atr.T @ Ytr)
    return Ate @ c


def lindist_ridge_primary(Vtr: np.ndarray, Vte: np.ndarray, Ytr: np.ndarray) -> np.ndarray:
    """b5-primary - per-output ridge on the physics output column + free intercept.

    Penalty grid logspace(-8, 3, 12) relative to the design trace, intercept
    UNPENALISED, lambda chosen per output by train-only K-fold CV (K = max(2, n//4)).
    This is the honest calibration; probe8.log measured 0.1665 normalised RMSE at n=10
    against 0.3805 for the archived joint-map calibration.
    """
    n = len(Vtr)
    Atr = np.hstack([Vtr, np.ones((n, 1))])
    Ate = np.hstack([Vte, np.ones((len(Vte), 1))])
    k = max(2, n // 4)
    folds = np.array_split(np.arange(n), k)
    D = np.ones(Atr.shape[1])
    D[-1] = 0.0
    grid = np.logspace(-8, 3, 12)
    sse = np.zeros((len(grid), Ytr.shape[1]))
    for gi, lam in enumerate(grid):
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 3:
                continue
            A1 = Atr[tri]
            G1 = A1.T @ A1
            tr1 = float(np.trace(G1)) / A1.shape[1]
            try:
                c = np.linalg.solve(G1 + lam * tr1 * np.diag(D), A1.T @ Ytr[tri])
            except np.linalg.LinAlgError:
                sse[gi] += 1e18
                continue
            sse[gi] += ((Atr[f] @ c - Ytr[f]) ** 2).sum(0)
    lam_idx = np.argmin(sse, axis=0)                    # per output
    out = np.zeros((len(Vte), Ytr.shape[1]))
    Gbase = Atr.T @ Atr
    trb = float(np.trace(Gbase)) / Atr.shape[1]
    shared = Atr.T @ Ytr
    for j in range(Ytr.shape[1]):
        lam = grid[lam_idx[j]]
        try:
            c = np.linalg.solve(Gbase + lam * trb * np.diag(D), shared[:, j])
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(Atr, Ytr[:, j], rcond=None)[0]
        out[:, j] = Ate @ c
    return out


def lindist_ridge_p17(Vtr: np.ndarray, Vte: np.ndarray, Ytr: np.ndarray) -> np.ndarray:
    """b5-secondary - the archived p17 joint-map calibration, kept for continuity only.

    One 46-column joint map, a single trace-scaled ridge applied to ALL coefficients
    INCLUDING the intercept column, CV grid (1e-4 .. 10) on the raw 45-output block.
    Reproduces p17_table.ridge_calib so a reader can map this table onto the earlier one.
    THIS IS NOT THE MARGIN REFERENCE.
    """
    n = len(Vtr)
    Atr = np.hstack([Vtr, np.ones((n, 1))])
    Ate = np.hstack([Vte, np.ones((len(Vte), 1))])
    k = max(2, n // 4)
    folds = np.array_split(np.arange(n), k)
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        sc = 0.0
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 3:
                continue
            A1 = Atr[tri]
            G = A1.T @ A1 + lam * np.trace(A1.T @ A1) / A1.shape[1] * np.eye(A1.shape[1])
            try:
                c = np.linalg.solve(G, A1.T @ Ytr[tri])
            except np.linalg.LinAlgError:
                sc += 1e18
                continue
            sc += float(((Atr[f] @ c - Ytr[f]) ** 2).mean()) / len(folds)
        if best is None or sc < best[0]:
            best = (sc, lam)
    lam = best[1]
    G = Atr.T @ Atr + lam * np.trace(Atr.T @ Atr) / Atr.shape[1] * np.eye(Atr.shape[1])
    c = np.linalg.solve(G, Atr.T @ Ytr)
    return Ate @ c


# ---------------------------------------------------------------------------
# admittance-weighted multiphase Kirchhoff operator (b3), 110-dimensional
# ---------------------------------------------------------------------------
def admittance_operator(ph) -> dict[str, Any]:
    """Grounded multiphase block Laplacian -> Z_bus -> 110-dim real feature map.

    L is complex-SYMMETRIC but not Hermitian, so scipy.linalg.eigvals (general) is
    used, and the spectrum of the Hermitian part is reported beside it.  We never
    call eigh on L.
    """
    import networkx as nx
    from scipy.linalg import eigvals

    from physics import zmat  # type: ignore

    buses = ph.buses
    idx = {int(b): i for i, b in enumerate(buses)}
    M = len(buses)
    Lb = np.zeros((3 * M, 3 * M), dtype=complex)
    for _, r in ph.net.line.iterrows():
        u, v = int(r.from_bus), int(r.to_bus)
        Mz = zmat(r.r_ohm_per_km * r.length_km, r.x_ohm_per_km * r.length_km,
                  r.r0_ohm_per_km * r.length_km, r.x0_ohm_per_km * r.length_km)
        Y = np.linalg.inv(Mz)
        iu = slice(3 * idx[u], 3 * idx[u] + 3)
        iv = slice(3 * idx[v], 3 * idx[v] + 3)
        Lb[iu, iu] += Y
        Lb[iv, iv] += Y
        Lb[iu, iv] -= Y
        Lb[iv, iu] -= Y
    src = idx[1]
    keep = np.array([i for i in range(3 * M) if i not in (3 * src, 3 * src + 1, 3 * src + 2)])
    Lg = Lb[np.ix_(keep, keep)]
    Zbus = np.linalg.inv(Lg)

    dev = float(np.abs(Lg @ Zbus - np.eye(Lg.shape[0])).max() / np.abs(np.eye(Lg.shape[0])).max())
    asymT = float(np.abs(Lb - Lb.T).max())
    asymH = float(np.abs(Lb - Lb.conj().T).max())
    scaleL = float(np.abs(Lb).max())

    # spectra: general (correct for a complex-symmetric matrix) and Hermitian part
    ev_general = np.sort(np.abs(eigvals(Lb)))[::-1]
    Lh = 0.5 * (Lb + Lb.conj().T)
    ev_herm = np.sort(np.abs(np.linalg.eigvalsh(Lh)))[::-1]

    def blk(b: int) -> int:
        return 3 * idx[b] - 3 if idx[b] > src else 3 * idx[b]

    lidx = [blk(int(b)) for b in ph.lbus]
    nload = ph.nload
    Zl = np.zeros((nload, nload), dtype=complex)
    for i in range(nload):
        for j in range(nload):
            Zl[i, j] = Zbus[lidx[i] + int(ph.phase[i]), lidx[j] + int(ph.phase[j])]
    Zl = Zl / ph.Zb
    return dict(
        Zl=Zl,
        identity_max_rel_dev=dev,
        asym_T=asymT,
        asym_H=asymH,
        L_scale=scaleL,
        spectrum_method="scipy.linalg.eigvals on the full grounded block Laplacian (general, non-Hermitian)",
        top10_general=[float(x) for x in ev_general[:10]],
        top10_hermitian_part=[float(x) for x in ev_herm[:10]],
        n_buses=int(M),
    )


def adm_feature_map(Zl: np.ndarray, X: np.ndarray) -> np.ndarray:
    """110-dim real feature: Re and Im of the own-phase voltage drop Z_l @ s.

    Fixes the 55-versus-110 mismatch of probe12: the compact treatment is
    110-dimensional, so the admittance competitor is built 110-dimensional.
    """
    nl = Zl.shape[0]
    P = X[:, :nl] / 0.8
    Q = X[:, nl:] / 0.8
    s = P + 1j * Q
    dV = s @ Zl.T
    return np.hstack([dV.real, dV.imag])


def hop_distance_matrix(ph) -> np.ndarray:
    """Hop distance between two LOAD buses, tree metric on the BFS tree from bus 1."""
    import networkx as nx

    n = ph.nload
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = (ph.hop[ph.lbus[i]] + ph.hop[ph.lbus[j]]
                       - 2 * ph.hop[nx.lowest_common_ancestor(ph.tree, ph.lbus[i], ph.lbus[j])])
    return H


def hop_feature_map(H: np.ndarray, lam: float, X: np.ndarray) -> np.ndarray:
    """Kernel-induced 110-dim feature map for the hop-count graph kernel.

    The hop kernel K_h[l, l'] = exp(-lam * hop(l, l')) is defined on the 55 LOADS, but
    the GP needs a metric between DRAWS.  The standard induced construction takes the
    factor Phi with K_h = Phi Phi^T (eigen-decomposition, negative eigenvalues clipped)
    and embeds a draw by u(x) = sum_l x_l Phi[l, :]; the draw-space kernel is then
    exp(-||u_i - u_j||^2 / 2).  Applied blockwise to (P, Q) so the competitor is
    110-dimensional like the treatment.

    This deliberately does NOT reuse the archived `F = P @ M` pattern of p8_full lapZ
    and lapY, which silently squares the metric.
    """
    Kh = np.exp(-lam * H)
    ev, V = np.linalg.eigh(0.5 * (Kh + Kh.T))
    ev = np.clip(ev, 0.0, None)
    Lh = V * np.sqrt(ev)[None, :]
    nl = H.shape[0]
    return np.hstack([X[:, :nl] @ Lh, X[:, nl:] @ Lh])


# ---------------------------------------------------------------------------
# STAGE: checks
# ---------------------------------------------------------------------------
def regenerate_ensemble(n_draws: int = 220, seed: int = 11,
                        scale: float = 2.0, cv: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: rebuild the 220-draw runpp_3ph ensemble exactly as ens.py does.

    Used ONLY when the archived .npy inputs are absent (a fresh clone of the published
    repository will not carry them). The recipe is the one recorded in the run tree:
    pandapower `ieee_european_lv_asymmetric('on_peak_566')`, per-load lognormal
    multiplier with mean 1 and coefficient of variation cv, active power
    P0.sum(axis=1) * scale * m clipped to [0.02e-3, 60e-3] MW, reactive power equal to
    active times tan(arccos 0.95), assigned to each load's OWN phase, solved with
    runpp_3ph(calculate_voltage_angles=True, max_iter=50, numba=False).
    """
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.ieee_european_lv_asymmetric("on_peak_566")
    al = net.asymmetric_load
    cols = ["p_a_mw", "p_b_mw", "p_c_mw"]
    P0 = al[cols].values.copy()
    nl = len(al)
    phase = np.argmax(P0, axis=1)
    sheet = np.load(DATA / "sheet15.npy") if (DATA / "sheet15.npy").exists() else None
    if sheet is None:
        raise FileNotFoundError("sheet15.npy is required to define the target sheet")

    rng = np.random.default_rng(seed)
    X = np.zeros((n_draws, 2 * nl))
    Y = np.zeros((n_draws, len(sheet) * 3))
    mu_l = -0.5 * np.log(1 + cv ** 2)
    sd_l = np.sqrt(np.log(1 + cv ** 2))
    for i in range(n_draws):
        m = rng.lognormal(mu_l, sd_l, nl)
        p = np.clip(P0.sum(1) * scale * m, 0.02e-3, 60e-3)
        q = p * np.tan(np.arccos(0.95))
        for j, (c1, c2) in enumerate([("p_a_mw", "q_a_mvar"), ("p_b_mw", "q_b_mvar"),
                                      ("p_c_mw", "q_c_mvar")]):
            al[c1] = np.where(phase == j, p, 0.0)
            al[c2] = np.where(phase == j, q, 0.0)
        pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
        v = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].values
        Y[i] = v[sheet].ravel() if net.converged else np.nan
        X[i] = np.concatenate([p, q])
    return X, Y


def ensure_inputs() -> dict[str, Any]:
    """Load the archived inputs; regenerate (and say so) if they are missing."""
    need = ["ens_s2.0_X.npy", "ens_s2.0_Y.npy", "sheet15.npy", "ph_true.npy"]
    missing = [f for f in need if not (DATA / f).exists()]
    info: dict[str, Any] = dict(regenerated=False, missing_at_start=missing)
    if missing:
        logger.warning("inputs missing {}; regenerating the ensemble", missing)
        ph = _import_physics()
        sheet = ph.sheet(15)[1]
        np.save(DATA / "sheet15.npy", sheet)
        np.save(DATA / "ph_true.npy", ph.phase)
        X, Y = regenerate_ensemble()
        np.save(DATA / "ens_s2.0_X.npy", X)
        np.save(DATA / "ens_s2.0_Y.npy", Y)
        info["regenerated"] = True
        info["regeneration_recipe"] = (
            "pandapower ieee_european_lv_asymmetric('on_peak_566'), lognormal(cv=0.5) "
            "per-load multipliers, 2.0x scale, P clipped to [0.02e-3, 60e-3] MW, "
            "Q = P tan(arccos 0.95), own-phase assignment, runpp_3ph, seed 11")
    return info


def stage_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    regen = ensure_inputs()
    X = np.load(DATA / "ens_s2.0_X.npy")
    Y = np.load(DATA / "ens_s2.0_Y.npy")
    sheet = np.load(DATA / "sheet15.npy")
    phl = np.load(DATA / "ph_true.npy")
    checks["input_regeneration"] = regen

    checks["X_shape"] = list(X.shape)
    checks["Y_shape"] = list(Y.shape)
    checks["finite_rows"] = int(np.isfinite(Y).all(1).sum())
    checks["mean_total_load_kW"] = float(1000.0 * X[:, :55].sum(1).mean())
    checks["far_end_sd_mean"] = float(Y.std(0).mean())
    checks["sheet15"] = [int(v) for v in sheet]
    checks["phase_counts"] = [int(v) for v in np.bincount(phl, minlength=3)]
    checks["n_loads"] = int(phl.size)
    checks["y_min_row_p1"] = float(np.percentile(Y.min(1), 1))

    assert list(X.shape) == [220, 110], "X shape"
    assert list(Y.shape) == [220, 45], "Y shape"
    assert checks["sheet15"] == [681, 697, 713, 729, 745, 761, 777, 793, 809, 825,
                                 841, 857, 873, 889, 906], "sheet15 mismatch"
    assert checks["phase_counts"] == [21, 19, 15], "phase label counts mismatch"
    # Level assertions hold for the ARCHIVED ensemble. A regenerated ensemble is a fresh
    # 220-draw sample from the same law, so its sample mean and sd differ by sampling
    # noise; per the plan we report the deviation and proceed rather than hard-fail.
    dev_kW = abs(checks["mean_total_load_kW"] - 114.6928)
    dev_sd = abs(checks["far_end_sd_mean"] - 0.021013)
    checks["deviation_from_archive"] = dict(mean_total_load_kW=float(dev_kW),
                                            far_end_sd=float(dev_sd))
    if not regen["regenerated"]:
        assert checks["finite_rows"] == 220, "not every archived row converged"
        assert dev_kW < 0.01, "mean total load mismatch"
        assert dev_sd < 1e-5, "far-end sd mismatch"
        checks["ensemble_source"] = "archived (iter_2 pilot ens_s2.0_X.npy / ens_s2.0_Y.npy)"
    else:
        assert checks["finite_rows"] == 220, "regenerated ensemble has non-converged rows"
        checks["ensemble_source"] = "REGENERATED in this run (archived inputs absent)"
        if dev_kW > 0.01 * 114.6928 or dev_sd > 0.01 * 0.021013:
            logger.warning("regenerated ensemble deviates from the archive: load {:.3f} kW "
                           "sd {:.6f}", dev_kW, dev_sd)
            checks["regeneration_deviation_gt_1pct"] = True

    ph = _import_physics()
    S = build_S(ph, sheet, phl)
    San = ph.lin_operator(sheet)
    checks["S_vs_Physics_lin_operator_max_rel"] = float(np.abs(S - San).max() / np.abs(S).max())

    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    svn = sv / sv[0]
    facts = dict(
        shape=list(S.shape),
        singular_values_normalised_first20=[float(v) for v in svn[:20]],
        stable_rank=float((sv ** 2).sum() / sv[0] ** 2),
        rank_1e2=int((svn > 1e-2).sum()),
        rank_1e3=int((svn > 1e-3).sum()),
        rank_1e4=int((svn > 1e-4).sum()),
        rank_at_double_precision_floor=int((sv > np.finfo(float).eps * sv[0] * S.shape[0]).sum()),
        condition_number=float(sv[0] / sv[-1]),
        condition_number_note=(
            "cond(S) is NOT a reproducible number for this operator. S is rank deficient "
            "(29 of 45 singular values sit above the double-precision floor, so the last "
            "sixteen are pure round-off), therefore cond(S) is the reciprocal of a "
            "round-off residue. The same matrix gives 4.99e+32 through "
            "numpy.linalg.svd(S, compute_uv=False) and 1.13e+16 through "
            "numpy.linalg.svd(S, full_matrices=False) - different LAPACK paths, "
            "elementwise identical input. The archived record quotes 1.28873e+32. The "
            "reproducible statements are the effective ranks at 1e-2/1e-3/1e-4, the stable "
            "rank, and condition_number_at_rank23 which is the condition number of the "
            "numerically nonsingular core."),
        condition_number_at_rank23=float(sv[0] / sv[22]),
    )
    jdump(facts, RESULTS / "structural_facts.json")

    assert facts["rank_1e2"] == 11 and facts["rank_1e3"] == 19 and facts["rank_1e4"] == 23
    assert abs(facts["stable_rank"] - 2.1768) < 0.01
    # the exact digits of cond(S) are a round-off residue; assert only that S is
    # numerically singular far beyond double precision and that the nonsingular core
    # is well conditioned. Anything from 1e12 to 1e33 is consistent with rank deficiency.
    assert 1e12 < facts["condition_number"] < 1e35
    assert facts["condition_number_at_rank23"] < 1e4
    archived = [1.0, 0.9434, 0.5037, 0.1157, 0.1024, 0.0593, 0.0515, 0.0425, 0.0242,
                0.0193, 0.0104, 0.0066, 0.0065, 0.0053, 0.0051, 0.0030, 0.0025,
                0.0024, 0.0020, 0.0009]
    got = [round(float(v), 4) for v in svn[:20]]
    assert got == archived, f"singular spectrum mismatch: {got}"

    # cross-check against the archived operator (row space + elementwise)
    cross: dict[str, Any] = {}
    arch_path = DATA / "S_sheet15.npy"
    if arch_path.exists():
        Ar = np.load(arch_path)
        cross["elementwise_max_abs_over_max_abs"] = float(np.abs(S - Ar).max() / np.abs(Ar).max())
        U2 = np.linalg.svd(Ar, full_matrices=False)[0]
        for k in (11, 45):
            pa = np.linalg.svd(U[:, :k].T @ U2[:, :k], compute_uv=False)
            cross[f"top{k}_principal_angles_rad"] = [float(v) for v in
                                                     np.arccos(np.clip(pa, -1, 1))]
        worst = max(cross.get("top11_principal_angles_rad", [0.0]))
        cross["agree"] = bool(worst < 1e-3 or cross["elementwise_max_abs_over_max_abs"] < 1e-9)
    facts["operator_crosscheck_vs_archive"] = cross

    # --- exponent-orientation assertions (the first thing to check) ------------
    x = np.asarray(X)
    M0 = op_whiten(U, sv, Vt, 45, 0.0)
    phi0 = x @ M0.T                          # (220, 45): diag(w) Vt x with w = sv/sv0
    rot = (phi0 @ U.T).T                     # U @ phi0^T, back in the 45-dim output frame
    ref = (x @ S.T).T / sv[0]                # S x / sv0
    exp_rot_dev = float(np.abs(rot - ref).max() / np.abs(ref).max())
    # squared distances must agree because U is orthogonal
    exp_dist_dev = float(np.abs(sqd(phi0, phi0) - sqd(ref.T, ref.T)).max()
                         / max(float(sqd(ref.T, ref.T).max()), 1e-30))
    M1 = op_whiten(U, sv, Vt, 45, 1.0)
    M05 = op_whiten(U, sv, Vt, 45, 0.5)
    cond1 = float(np.linalg.cond(M1))
    cond05 = float(np.linalg.cond(M05))
    assert exp_rot_dev < 1e-10, f"gamma-0 ratio form wrong: {exp_rot_dev}"
    assert cond1 < 1.001, f"gamma=1 must be semi-orthogonal: cond={cond1}"
    target05 = float((sv[0] / sv[-1]) ** 0.5)
    assert abs(cond05 - target05) / target05 < 1e-6, f"gamma=0.5 cond {cond05} vs {target05}"
    exponent_checks = dict(
        gamma0_rotation_identity_rel_dev=exp_rot_dev,
        cond_gamma0=float(np.linalg.cond(M0)),
        cond_gamma05=cond05,
        cond_gamma05_expected=target05,
        cond_gamma10=cond1,
        note=("STEP 3 asserts Phi_{45,0}(x) == S x / sv0 elementwise. That is impossible: "
              "the ratio form drops the left singular vector U, an orthogonal rotation "
              "that leaves every RBF kernel bit-identical. The assertion is therefore "
              "implemented as (a) U @ Phi_{45,0}(x) == S x / sv0 to 1e-10 and (b) the "
              "squared-distance matrices agreeing. Recorded as a correction."),
    )
    checks["exponent_orientation"] = exponent_checks

    # --- admittance operator (b3) ---------------------------------------------
    adm = admittance_operator(ph)
    facts["admittance_laplacian_identity_max_rel_dev"] = adm["identity_max_rel_dev"]
    facts["admittance_laplacian_asymmetry"] = dict(
        max_abs_L_minus_LT=adm["asym_T"],
        max_abs_L_minus_LH=adm["asym_H"],
        laplacian_scale=adm["L_scale"],
    )
    facts["admittance_spectrum_method"] = adm["spectrum_method"]
    facts["admittance_spectrum_top10_general"] = adm["top10_general"]
    facts["admittance_spectrum_top10_hermitian_part"] = adm["top10_hermitian_part"]
    assert adm["identity_max_rel_dev"] < 1e-10, "Z_bus != inv(grounded Y)"

    # --- fold counts ----------------------------------------------------------
    counts = {str(n): len(folds_for(n)) for n in N_GRID}
    checks["folds_per_n"] = counts
    assert counts["10"] == 24 and counts["20"] == 24 and counts["40"] == 24
    assert counts["80"] == 24 and counts["160"] == 12
    checks["total_fold_cells"] = int(sum(counts.values()))
    assert checks["total_fold_cells"] == 108

    # --- cache the operators ---------------------------------------------------
    # Shuffled-phase operator: permute the 55 load->phase ASSIGNMENTS among the loads.
    # `rng.permutation(55)` applied to the label ARRAY would produce integers 0..54,
    # which are not phase indices; the control the archived p8_full `shufLab` arm built
    # is phase[perm] - a permutation of which load carries which phase, with the data
    # left on the real phases. Recorded as a correction to the plan text.
    perm = np.random.default_rng(31).permutation(55)
    phl_perm = phl[perm]
    S_shuf = build_S(ph, sheet, phl_perm)
    Ush, svsh, Vtsh = np.linalg.svd(S_shuf, full_matrices=False)
    checks["shuffled_phase_operator"] = dict(
        seed=31,
        construction=("phase_shuffled = phase_true[perm] with perm = default_rng(31)"
                      ".permutation(55); the operator's load-to-phase assignment is "
                      "permuted while the data stay on the real phases"),
        permutation=[int(v) for v in perm],
        n_phase_labels_changed=int((phl_perm != phl).sum()),
        shuffled_phase_counts=[int(v) for v in np.bincount(phl_perm, minlength=3)],
        singular_values_normalised_first20=[float(v) for v in (svsh / svsh[0])[:20]],
    )
    # null control: a GLOBAL cyclic relabel of all three phases is a symmetry of the
    # feeder, so this operator must be essentially identical to the real one.
    S_cyc = build_S(ph, sheet, (phl + 1) % 3)
    sv_cyc = np.linalg.svd(S_cyc, compute_uv=False)
    checks["shuffled_phase_operator"]["cyclic_relabel_null_control"] = dict(
        construction="phase_true + 1 mod 3 (global cyclic relabel, a symmetry)",
        max_abs_rel_dev_from_real=float(np.abs(S_cyc - S).max() / np.abs(S).max()),
        singular_values_normalised_first20=[float(v) for v in (sv_cyc / sv_cyc[0])[:20]],
    )
    np.savez(
        DATA / "operators.npz",
        S=S, U=U, sv=sv, Vt=Vt,
        S_shuf=S_shuf, U_shuf=Ush, sv_shuf=svsh, Vt_shuf=Vtsh,
        Zl=adm["Zl"], hop=hop_distance_matrix(ph),
        sheet=sheet, phase=phl,
    )
    checks["inputs_regenerated"] = False
    checks["reused_ensembles"] = (
        "archived 220-draw runpp_3ph pools from iter_2/gen_hypo/claude_agent/pilot "
        "(ens_s2.0_X.npy, ens_s2.0_Y.npy); zero AC solves on the measurement path"
    )
    facts["singular_values_normalised_first45"] = [float(v) for v in svn]
    jdump(facts, RESULTS / "structural_facts.json")
    jdump(checks, RESULTS / "protocol_checks.json")
    logger.info("structural facts: ranks {}/{}/{} stable_rank {:.4f} cond {:.3e}",
                facts["rank_1e2"], facts["rank_1e3"], facts["rank_1e4"],
                facts["stable_rank"], facts["condition_number"])
    logger.info("exponent orientation: cond(gamma=0)={:.3e} cond(0.5)={:.4f} cond(1)={:.6f}",
                exponent_checks["cond_gamma0"], cond05, cond1)
    return checks


# ---------------------------------------------------------------------------
# shared worker state
# ---------------------------------------------------------------------------
_G: dict[str, Any] = {}


def _init_worker() -> None:
    _G["X"] = np.load(DATA / "ens_s2.0_X.npy")
    _G["Y"] = np.load(DATA / "ens_s2.0_Y.npy")
    z = np.load(DATA / "operators.npz")
    _G["z"] = {k: z[k] for k in z.files}


def _lindist_all() -> tuple[np.ndarray, np.ndarray]:
    """Multiphase LinDistFlow order 1 and 2 for all 220 rows (one no-load AC solve)."""
    cache = DATA / "lindist_1_2.npy"
    if cache.exists():
        arr = np.load(cache)
        return arr[0], arr[1]
    ph = _import_physics()
    sheet = np.load(DATA / "sheet15.npy")
    X = np.load(DATA / "ens_s2.0_X.npy")
    V0b = ph.noload_voltage(sheet)
    V1 = ph.lindist(X, sheet, order=1, v0bus=V0b)
    V2 = ph.lindist(X, sheet, order=2, v0bus=V0b)
    np.save(cache, np.stack([V1, V2]))
    return V1, V2


# ---------------------------------------------------------------------------
# STAGE: baselines
# ---------------------------------------------------------------------------
def _baselines_for_fold(fold: dict[str, Any]) -> dict[str, Any]:
    X, Y = _G["X"], _G["Y"]
    z = _G["z"]
    tr, te = fold["tr"], fold["te"]
    Ytr, Yte = Y[tr], Y[te]
    vsd = float(Ytr.std(0).mean())
    vsd_all = float(Y.std(0).mean())
    Mp = Ytr.mean(0)
    meanpred = rmse(np.tile(Mp, (len(te), 1)), Yte) / vsd

    out: dict[str, Any] = dict(
        n=fold["n"], fold_id=fold["fold_id"], seed=fold["seed"], scheme=fold["scheme"],
        n_train=int(len(tr)), n_test=int(len(te)), vsd_train=vsd, vsd_all=vsd_all,
        mean_predictor_rmse=meanpred,
    )

    # ---- b1a / b1b : RBF on injections ---------------------------------------
    mu_x = X[tr].mean(0)
    sd_x = X[tr].std(0)
    sd_x[sd_x < 1e-12] = 1.0
    Z = (X - mu_x) / sd_x
    Dtr_raw = sqd(Z[tr], Z[tr])
    Dte_raw = sqd(Z[te], Z[tr])
    iu = np.triu_indices(len(tr), k=1)
    med = float(np.median(Dtr_raw[iu])) if len(tr) > 1 else 1.0
    med = max(med, 1e-12)
    mu1, pv1, hp1, _, _ = gp_corr(rbf_corr(Dtr_raw / med, Dte_raw / med), Ytr)
    out["b1a_rbf_median"] = score_gp(mu1, pv1, Yte, vsd)
    out["b1a_rbf_median"]["hp"] = hp1
    mu2, pv2, hp2, _, _ = gp_corr(rbf_corr(Dtr_raw, Dte_raw), Ytr)
    out["b1b_rbf_raw"] = score_gp(mu2, pv2, Yte, vsd)
    out["b1b_rbf_raw"]["hp"] = hp2

    # ---- b2 : hop-count graph kernel -----------------------------------------
    # The draw-space metric is induced from the 55-load hop kernel by a kernel factor;
    # lambda is chosen by marginal likelihood over the induced feature maps.
    H = z["hop"]
    hop_corr = []
    for lam in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        Fh = hop_feature_map(H, lam, X)
        Ftr, Fte = featurise(np.eye(Fh.shape[1]), Fh[tr], Fh[te])
        # the induced map fixes the draw-space lengthscale at 1; only s2 and noise search
        hop_corr.append((float(lam), np.exp(-sqd(Ftr, Ftr) / 2.0),
                         np.exp(-sqd(Fte, Ftr) / 2.0)))
    mu3, pv3, hp3, _, _ = gp_corr(hop_corr, Ytr)
    out["b2_hop_laplacian"] = score_gp(mu3, pv3, Yte, vsd)
    out["b2_hop_laplacian"]["hp"] = hp3

    # ---- b3 : admittance-weighted multiphase Kirchhoff kernel, 110-dim -------
    # Z_bus is data-INDEPENDENT (line data only), never a fitted sensitivity Jacobian.
    # The noise power is not a tuning knob here: the feature scale is set on train rows
    # and the lengthscale is searched by marginal likelihood with a grid that spans the
    # train median squared distance, so the grid cannot be truncated relative to the
    # distance scale the way the archived RBF grid was.
    Fadm = adm_feature_map(z["Zl"], X)
    Ftr, Fte = featurise(np.eye(Fadm.shape[1]), Fadm[tr], Fadm[te])
    Dadm_tr, Dadm_te = sqd(Ftr, Ftr), sqd(Fte, Ftr)
    med_adm = max(float(np.median(Dadm_tr[np.triu_indices(len(tr), k=1)])), 1e-12)
    mu4, pv4, hp4, _, _ = gp_corr(
        rbf_corr(Dadm_tr / med_adm, Dadm_te / med_adm), Ytr)
    out["b3_admittance"] = score_gp(mu4, pv4, Yte, vsd)
    out["b3_admittance"]["hp"] = hp4
    out["b3_admittance"]["median_sqd_train"] = med_adm

    # ---- b4 : ridge on standardised injections -------------------------------
    out["b4_ridge"] = dict(rmse=rmse(ridge_scale_free(Z[tr], Z[te], Ytr), Yte) / vsd)

    # ---- b5-primary : honest per-output ridge on LinDistFlow order 2 ---------
    V1, V2 = _G["V1"], _G["V2"]
    cands = {}
    for order, Vp in (("order1", V1), ("order2", V2)):
        pred = lindist_ridge_primary(Vp[tr], Vp[te], Ytr)
        cands[order] = dict(pred=pred, rmse=rmse(pred, Yte) / vsd)
    best_order = min(cands, key=lambda o: cands[o]["rmse"])
    out["b5p_lindist_honest"] = dict(rmse=cands[best_order]["rmse"], order=best_order,
                                     rmse_order1=cands["order1"]["rmse"],
                                     rmse_order2=cands["order2"]["rmse"])
    # both denominators, so the effect of the train-only choice is quantified
    out["b5p_lindist_honest"]["rmse_vsd_all"] = float(
        rmse(cands[best_order]["pred"], Yte) / vsd_all)

    # ---- b5-secondary : archived p17 joint-map calibration -------------------
    pred17 = lindist_ridge_p17(V2[tr], V2[te], Ytr)
    out["b5s_p17_jointmap"] = dict(rmse=rmse(pred17, Yte) / vsd,
                                   rmse_vsd_all=float(rmse(pred17, Yte) / vsd_all))

    prim = [out["b1a_rbf_median"]["rmse"], out["b2_hop_laplacian"]["rmse"],
            out["b3_admittance"]["rmse"], out["b4_ridge"]["rmse"],
            out["b5p_lindist_honest"]["rmse"]]
    names = ["b1a_rbf_median", "b2_hop_laplacian", "b3_admittance", "b4_ridge",
             "b5p_lindist_honest"]
    j = int(np.argmin(prim))
    out["best_baseline"] = dict(name=names[j], rmse=float(prim[j]))
    return out


def stage_baselines() -> None:
    t0 = time.time()
    all_folds: list[dict[str, Any]] = []
    for n in N_GRID:
        all_folds.extend(folds_for(n))
    logger.info("baselines for {} fold-cells", len(all_folds))
    _init_worker()
    _G["V1"], _G["V2"] = _lindist_all()
    rows = []
    for i, f in enumerate(all_folds):
        rows.append(_baselines_for_fold(f))
        if (i + 1) % 12 == 0:
            logger.info("  {}/{} fold-cells done ({:.1f}s)", i + 1, len(all_folds),
                        time.time() - t0)
    jdump(rows, RAW / "baselines.json")

    import pandas as pd

    flat = []
    for r in rows:
        flat.append(dict(
            n=r["n"], fold_id=r["fold_id"], seed=r["seed"], scheme=r["scheme"],
            n_train=r["n_train"], n_test=r["n_test"], vsd_train=r["vsd_train"],
            mean_predictor=r["mean_predictor_rmse"],
            b1a_rbf_median=r["b1a_rbf_median"]["rmse"],
            b1b_rbf_raw=r["b1b_rbf_raw"]["rmse"],
            b2_hop_laplacian=r["b2_hop_laplacian"]["rmse"],
            b3_admittance=r["b3_admittance"]["rmse"],
            b4_ridge=r["b4_ridge"]["rmse"],
            b5p_lindist_honest=r["b5p_lindist_honest"]["rmse"],
            b5p_lindist_honest_vsd_all=r["b5p_lindist_honest"]["rmse_vsd_all"],
            b5s_p17_jointmap=r["b5s_p17_jointmap"]["rmse"],
            best_baseline_name=r["best_baseline"]["name"],
            best_baseline_rmse=r["best_baseline"]["rmse"],
        ))
    df = pd.DataFrame(flat)
    df.to_csv(RESULTS / "baselines_per_fold.csv", index=False)
    logger.info("baselines written ({:.1f}s); T4 check n=10:", time.time() - t0)
    d10 = df[df.n == 10]
    logger.info("  b5p honest {:.4f} | b5s p17 {:.4f} | b1a {:.4f} | meanpred {:.3f}",
                d10.b5p_lindist_honest.mean(), d10.b5s_p17_jointmap.mean(),
                d10.b1a_rbf_median.mean(), d10.mean_predictor.mean())
    # T4 - baseline order check, recorded so it is machine-verifiable rather than a log
    # line someone has to read. The archived p17 calibration is EXPECTED to be worse (it
    # penalises the intercept and its CV grid floor is reached); if the two are the other
    # way round, the probe3_ldf calibration bug has been re-introduced.
    t4 = dict(
        n=10,
        b5p_honest_per_output_ridge_mean=float(d10.b5p_lindist_honest.mean()),
        b5s_p17_jointmap_mean=float(d10.b5s_p17_jointmap.mean()),
        honest_beats_p17=bool(d10.b5p_lindist_honest.mean()
                              < d10.b5s_p17_jointmap.mean()),
        honest_in_expected_range=bool(0.10 < d10.b5p_lindist_honest.mean() < 0.30),
        p17_in_expected_range=bool(0.30 < d10.b5s_p17_jointmap.mean() < 0.55),
        note=("probe8.log recorded 0.1665 honest against 0.3805 for the archived joint-map "
              "calibration at n=10 on the p17 fold family. This artifact uses the shared "
              "screen fold family (seed base 20261111, not p17's 31337), so the levels are "
              "close but not identical. The ordering is the invariant that matters."),
    )
    t4["passes"] = bool(t4["honest_beats_p17"] and t4["honest_in_expected_range"])
    pc_path = RESULTS / "protocol_checks.json"
    if pc_path.exists():
        pc = jload(pc_path)
    else:
        pc = {}
    pc["T4_baseline_order"] = t4
    t7 = {}
    for nn in (10, 160):
        sub = df[df.n == nn]
        t7[str(nn)] = dict(
            b1a_median_normalised=float(sub.b1a_rbf_median.mean()),
            b1b_raw_scale=float(sub.b1b_rbf_raw.mean()),
            mean_predictor=float(sub.mean_predictor.mean()),
            median_normalisation_gain=float((sub.b1b_rbf_raw.mean()
                                             - sub.b1a_rbf_median.mean())
                                            / max(sub.b1b_rbf_raw.mean(), 1e-12)))
    pc["T7_rbf_defect_check"] = t7
    pc["baselines_per_fold_rows"] = int(len(df))
    pc["baselines_per_fold_nan_cells"] = int(df.isna().sum().sum())
    assert int(df.isna().sum().sum()) == 0, "baselines CSV contains NaN"
    jdump(pc, pc_path)
    logger.info("T4 recorded: honest {:.4f} vs p17 {:.4f}, passes={}",
                t4["b5p_honest_per_output_ridge_mean"], t4["b5s_p17_jointmap_mean"],
                t4["passes"])
    if not t4["passes"]:
        logger.warning("T4 baseline-order check FAILED - margins are not usable")


# ---------------------------------------------------------------------------
# STAGE: sweep
# ---------------------------------------------------------------------------
def _variant_names() -> list[tuple[str, str, Any]]:
    v = []
    for k in K_GRID:
        for g in G_GRID:
            v.append((f"k{k}_g{g}", "whiten", (k, g)))
    for f in F_GRID:
        v.append((f"floor_{f:g}", "floor", (f,)))
    return v


def _jitter_sweep(Ktr: np.ndarray, Kte: np.ndarray, Ytr: np.ndarray, Yte: np.ndarray,
                  hp: tuple, vsd: float, mean_pred: float) -> dict[str, Any]:
    """Refit at a fixed (l, s2) with a hard jitter added to the diagonal.

    Purpose: separate a NUMERICAL stabilisation artefact from a STATISTICAL gain. If a
    small hard jitter rescues the arm, the arm was numerically fragile, not better. Per
    guard g1 an entry whose normalised RMSE exceeds the mean predictor is recorded as a
    numerical failure (`numerical_failure: true`), never quoted as a finding - the
    archived precedent is an unscaled ridge penalty grid error of 50.0 at n=10.

    The jitter is ADDED on top of the selected noise level n2 (total diagonal
    s2*Ktr + (n2 + j) I), because REPLACING the noise with j would test a different
    model, not the numerical stability of the selected one. j=0 is the selected fit and
    is included as the reference row. A first implementation replaced n2 with j and, on
    the n=40 fold-4000 cell, reported normalised RMSE 4.6e6 at j=1e-8 purely because the
    selected noise was 0.1 and the replacement made the kernel singular - exactly the
    guard-g1 failure mode, recorded as a correction.
    """
    lab, s2, n2 = hp
    n, m = Ytr.shape
    yb = Ytr.mean(0)
    ys = Ytr.std(0) + 1e-12
    Yc = (Ytr - yb) / ys
    out: dict[str, Any] = {}
    for j in (0.0, *JITTERS):
        K = s2 * Ktr + (n2 + j) * np.eye(n)
        try:
            Lc = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            out[f"{j:g}"] = dict(rmse=None, numerical_failure=True,
                                 reason="Cholesky failed")
            continue
        al = np.linalg.solve(Lc, Yc)
        a = np.linalg.solve(Lc.T, al)
        mu = ((s2 * Kte) @ a) * ys + yb
        e = float(rmse(mu, Yte) / vsd)
        out[f"{j:g}"] = dict(rmse=e, numerical_failure=bool(e > mean_pred),
                             reason="normalised RMSE exceeds the mean predictor"
                             if e > mean_pred else None)
    return out


def _sweep_cell(args: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    n, fold, base = args
    X, Y = _G["X"], _G["Y"]
    z = _G["z"]
    tr, te = fold["tr"], fold["te"]
    Ytr, Yte = Y[tr], Y[te]
    vsd = base["vsd_train"]
    Xtr, Xte = X[tr], X[te]
    U, sv, Vt = z["U"], z["sv"], z["Vt"]

    variants: dict[str, Any] = {}
    for name, kind, param in _variant_names():
        M = op_whiten(U, sv, Vt, param[0], param[1]) if kind == "whiten" \
            else op_floor(U, sv, Vt, param[0])
        Ftr, Fte = featurise(M, Xtr, Xte)
        Dtr = sqd(Ftr, Ftr)
        Dte = sqd(Fte, Ftr)
        mu, pv, hp, Ktr, Kte = gp_corr(rbf_corr(Dtr, Dte), Ytr)
        s = score_gp(mu, pv, Yte, vsd)
        s["hp"] = [hp[0], hp[1], hp[2]]  # l, s2, n2 (label == l for RBF arms)
        s["dim"] = int(M.shape[0])
        variants[name] = s

    # ---- diagnostics for the raw (k=45, gamma=0) and the gamma=0.5 arms -------
    diags: dict[str, Any] = {}
    for tag, (kk, gg) in (("raw", (45, 0.0)), ("whitened_g0p5", (45, 0.5))):
        M = op_whiten(U, sv, Vt, kk, gg)
        Ftr, Fte = featurise(M, Xtr, Xte)
        Dtr, Dte = sqd(Ftr, Ftr), sqd(Fte, Ftr)
        mu, pv, hp, Ktr, Kte = gp_corr(rbf_corr(Dtr, Dte), Ytr)
        Kfull = hp[1] * Ktr + hp[2] * np.eye(len(tr))  # hp = (l, s2, n2)
        _ = hp[0]
        sval = np.linalg.svd(Kfull, compute_uv=False)
        try:
            piv = float(np.min(np.diag(np.linalg.cholesky(Kfull))))
        except np.linalg.LinAlgError:
            piv = float("nan")
        diags[tag] = dict(
            cond_K=float(sval[0] / max(sval[-1], 1e-300)),
            min_cholesky_pivot=piv,
            hyperparams=dict(l=hp[0], s2=hp[1], n2=hp[2]),
            jitter_sweep=_jitter_sweep(Ktr, Kte, Ytr, Yte, hp, vsd,
                                       base["mean_predictor_rmse"]),
        )

    cell = dict(n=int(n), fold_id=int(fold["fold_id"]), seed=int(fold["seed"]),
                scheme=int(fold["scheme"]), n_train=int(len(tr)), n_test=int(len(te)),
                vsd_train=float(vsd), variants=variants, diagnostics=diags,
                best_baseline=base["best_baseline"],
                mean_predictor_rmse=float(base["mean_predictor_rmse"]))
    jdump(cell, RAW / f"cell_n{n}_f{fold['fold_id']}.json")
    return cell


def stage_sweep(only_n: Iterable[int] | None = None) -> None:
    _init_worker()
    bases = {(r["n"], r["fold_id"]): r for r in jload(RAW / "baselines.json")}
    order = list(only_n) if only_n else [80, 40, 20, 160, 10]
    for n in order:
        folds = folds_for(n)
        todo = [f for f in folds if not (RAW / f"cell_n{n}_f{f['fold_id']}.json").exists()]
        logger.info("n={}: {} folds, {} to compute", n, len(folds), len(todo))
        t0 = time.time()
        if todo:
            payload = [(n, f, bases[(n, f["fold_id"])]) for f in todo]
            with ProcessPoolExecutor(max_workers=N_PROC, initializer=_init_worker) as ex:
                for i, _ in enumerate(ex.map(_sweep_cell, payload)):
                    if (i + 1) % 6 == 0:
                        logger.info("  n={}  {}/{} cells ({:.1f}s)", n, i + 1, len(todo),
                                    time.time() - t0)
        logger.info("n={} done in {:.1f}s", n, time.time() - t0)


# ---------------------------------------------------------------------------
# STAGE: train-only selection of (k, gamma)
# ---------------------------------------------------------------------------
def _select_one(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    fold, base = args
    X, Y = _G["X"], _G["Y"]
    z = _G["z"]
    tr, te = fold["tr"], fold["te"]
    Ytr = Y[tr]
    vsd = base["vsd_train"]
    U, sv, Vt = z["U"], z["sv"], z["Vt"]
    Xtr = X[tr]
    n = len(tr)

    configs = [(k, g) for k in K_GRID_COARSE for g in G_GRID_COARSE]
    scores = np.zeros(len(configs))
    counts = np.zeros(len(configs))
    # cache features per config once (independent of the inner split)
    feats = []
    for (k, g) in configs:
        M = op_whiten(U, sv, Vt, k, g)
        Ftr, _ = featurise(M, Xtr, Xtr)
        feats.append(sqd(Ftr, Ftr))

    for rep in range(SEL_REPS):
        rng = np.random.default_rng(fold["fold_id"] * 100 + rep)
        perm = rng.permutation(n)
        h = n // 2
        if h < 2 or n - h < 2:
            continue
        A, B = perm[:h], perm[h:]
        for ci, D in enumerate(feats):
            for (fit, sco) in ((A, B), (B, A)):
                Dab, Dbb = D[np.ix_(fit, fit)], D[np.ix_(sco, fit)]
                mu, _, _, _, _ = gp_corr(rbf_corr(Dab, Dbb), Ytr[fit])
                e = rmse(mu, Ytr[sco]) / vsd
                scores[ci] += e
                counts[ci] += 1
    ok = counts > 0
    if not ok.any():
        pick = (45, 0.0)
        tie = "no inner fold possible; defaulted to (45, 0.0)"
    else:
        sc = np.where(ok, scores / np.maximum(counts, 1), np.inf)
        best = float(sc[ok].min())
        cand = [i for i in range(len(configs)) if ok[i] and sc[i] <= best + 1e-12]
        # tie-break: LARGER gamma, then SMALLER k
        cand.sort(key=lambda i: (-configs[i][1], configs[i][0]))
        pick = configs[cand[0]]
        tie = "ties broken toward larger gamma then smaller k"
    sel_k, sel_g = pick

    # shuffled-phase control at the selected (k, gamma): operator on permuted labels,
    # data on real phases.  An INCONSISTENT operator by construction - it is a
    # phase-blindness check for the whitening gain, NOT a test of phase identity.
    Msh = op_whiten(z["U_shuf"], z["sv_shuf"], z["Vt_shuf"], sel_k, sel_g)
    Ftr, Fte = featurise(Msh, Xtr, X[te])
    mu, pv, hp, _, _ = gp_corr(rbf_corr(sqd(Ftr, Ftr), sqd(Fte, Ftr)), Ytr)
    shuf = score_gp(mu, pv, Y[te], vsd)
    shuf["hp"] = [hp[0], hp[1], hp[2]]

    # ---- tempered-noise diagnostic at the same (k, gamma) --------------------
    # Marginal-likelihood noise replaced by the TRAIN empirical residual variance of the
    # marginal-likelihood fit. This separates overconfidence that is a noise-inference
    # artefact from one that is a property of the metric: NLPD and coverage are then
    # computed against a residual variance measured on train rows only, so nothing about
    # the test block can enter the calibration.
    Msel = op_whiten(U, sv, Vt, sel_k, sel_g)
    Ftr_s, Fte_s = featurise(Msel, Xtr, X[te])
    D_s_tr, D_s_te = sqd(Ftr_s, Ftr_s), sqd(Fte_s, Ftr_s)
    mu_s, pv_s, hp_s, _, _ = gp_corr(rbf_corr(D_s_tr, D_s_te), Ytr)
    # Train-ONLY HELD-OUT residual variance: fit on half the TRAIN rows, score on the
    # other half, both directions, at the same selected hyperparameters. In-sample
    # residuals are useless here - the marginal-likelihood noise is ~1e-8, so in-sample
    # residuals are ~0 and the tempered NLPD explodes to -1e14 (a guard-g1 numerical
    # failure, which a first implementation of this row produced). Held-out TRAIN
    # residuals are bounded and cannot see the test block.
    K_s_tr = np.exp(-D_s_tr / (2 * hp_s[0] ** 2))
    resid_train = _heldout_resid_var(K_s_tr, Ytr, hp_s,
                                     seed=int(fold["fold_id"] * 100 + 97))
    pv_temp = np.full_like(mu_s, max(resid_train, 1e-18))
    tempered = score_gp(mu_s, pv_temp, Y[te], vsd)
    tempered["train_heldout_residual_variance"] = resid_train
    tempered["note"] = ("predictive variance fixed at the train-only HELD-OUT residual "
                        "variance of the marginal-likelihood fit; a diagnostic row, not "
                        "a candidate arm")

    return dict(n=int(fold["n"]), fold_id=int(fold["fold_id"]), seed=int(fold["seed"]),
                scheme=int(fold["scheme"]), selected=[int(sel_k), float(sel_g)],
                tie_break=tie, inner_score_table=dict(
                    configs=[f"k{k}_g{g}" for k, g in configs],
                    mean_score=[float(s / c) if c else None
                                for s, c in zip(scores, counts)]),
                shuffled_phase=shuf, tempered_noise=tempered)


def stage_select() -> None:
    _init_worker()
    bases = {(r["n"], r["fold_id"]): r for r in jload(RAW / "baselines.json")}
    out = []
    for n in N_GRID:
        folds = folds_for(n)
        payload = [(f, bases[(n, f["fold_id"])]) for f in folds]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_PROC, initializer=_init_worker) as ex:
            for r in ex.map(_select_one, payload):
                out.append(r)
        logger.info("selection n={} done ({:.1f}s)", n, time.time() - t0)
    jdump(out, RAW / "selection.json")


# ---------------------------------------------------------------------------
# STAGE: statistics, arms, figures, outputs
# ---------------------------------------------------------------------------
ARM_SPECS = [
    ("A0_raw", "gamma 0, k 45 - the archived treatment"),
    ("A1_sel", "train-selected (k, gamma) - the candidate deliverable"),
    ("A2_fixed_0p5", "gamma 0.5, k 45 - the pilot chosen exponent, no selection"),
    ("A3_fixed_0p5_k11", "gamma 0.5, k 11 - the pilot chosen pair"),
    ("A4_gamma1", "gamma 1.0, k 45 - the catastrophic arm, mechanism test"),
    ("A5_trunc_inert", "gamma 0, k in {8, 11, 20} - the inert family, falsification test"),
    ("A6_floor", "gamma 0 with the singular-value-floor family - data-independent"),
    ("A7_oracle", "per-fold TEST-argmin variant - selection optimism gap"),
    ("A8_shuffled", "shuffled-phase operator at the selected (k, gamma) - consistency control"),
    ("A10_tempered",
     "tempered-noise diagnostic at the selected (k, gamma): marginal-likelihood mean, "
     "predictive variance fixed at the TRAIN empirical residual variance"),
    ("A9_meanpred", "the do-nothing reference - guard g1"),
]


def _bootstrap_ci(d: np.ndarray, n_boot: int = 2000, seed: int = 20260101):
    if d.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _wilcoxon(d: np.ndarray) -> float | None:
    from scipy.stats import wilcoxon

    if d.size < 6 or np.allclose(d, 0.0):
        return None
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


def stage_stats() -> None:
    from scipy.stats import wilcoxon  # noqa: F401  (import check)

    bases = {(r["n"], r["fold_id"]): r for r in jload(RAW / "baselines.json")}
    sel = {(r["n"], r["fold_id"]): r for r in jload(RAW / "selection.json")}
    per_n: dict[str, Any] = {}
    structures = jload(RESULTS / "structural_facts.json")
    checks = jload(RESULTS / "protocol_checks.json")

    for n in N_GRID:
        folds = folds_for(n)
        cells = []
        for f in folds:
            p = RAW / f"cell_n{n}_f{f['fold_id']}.json"
            if p.exists():
                cells.append(jload(p))
        if not cells:
            logger.warning("no cells for n={}", n)
            continue
        fids = [c["fold_id"] for c in cells]
        vnames = [name for name, _, _ in _variant_names()]

        # ---- assemble arm-level per-fold series -----------------------------
        def series_from_variant(vname: str) -> dict[str, list[float]]:
            r, m, l, c68, c95, hp = [], [], [], [], [], []
            for c in cells:
                v = c["variants"][vname]
                bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
                r.append(v["rmse"])
                m.append(1.0 - v["rmse"] / bb)
                l.append(v["nlpd"])
                c68.append(v["c68"])
                c95.append(v["c95"])
                hp.append(v["hp"])
            return dict(rmse=r, margin=m, nlpd=l, c68=c68, c95=c95, hp=hp)

        arms: dict[str, Any] = {}
        # A0, A2, A3, A4
        arms["A0_raw"] = series_from_variant("k45_g0.0")
        arms["A2_fixed_0p5"] = series_from_variant("k45_g0.5")
        arms["A3_fixed_0p5_k11"] = series_from_variant("k11_g0.5")
        arms["A4_gamma1"] = series_from_variant("k45_g1.0")

        # A1 selected
        s1 = dict(rmse=[], margin=[], nlpd=[], c68=[], c95=[], hp=[])
        picked = []
        for c in cells:
            k, g = sel[(n, c["fold_id"])]["selected"]
            vname = f"k{int(k)}_g{float(g)}"
            v = c["variants"][vname]
            bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
            s1["rmse"].append(v["rmse"])
            s1["margin"].append(1.0 - v["rmse"] / bb)
            s1["nlpd"].append(v["nlpd"])
            s1["c68"].append(v["c68"])
            s1["c95"].append(v["c95"])
            s1["hp"].append(v["hp"])
            picked.append([int(k), float(g)])
        arms["A1_sel"] = s1

        # A5 inert family = mean over k in {8,11,20} at gamma 0
        ks = ("k8_g0.0", "k11_g0.0", "k20_g0.0")
        a5 = dict(rmse=[], margin=[], nlpd=[], c68=[], c95=[], hp=[])
        for i, c in enumerate(cells):
            bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
            rs = [c["variants"][kk]["rmse"] for kk in ks]
            a5["rmse"].append(float(np.mean(rs)))
            a5["margin"].append(1.0 - float(np.mean(rs)) / bb)
            a5["nlpd"].append(float(np.mean([c["variants"][kk]["nlpd"] for kk in ks])))
            a5["c68"].append(float(np.mean([c["variants"][kk]["c68"] for kk in ks])))
            a5["c95"].append(float(np.mean([c["variants"][kk]["c95"] for kk in ks])))
            a5["hp"].append(c["variants"]["k11_g0.0"]["hp"])
        arms["A5_trunc_inert"] = a5

        # A6 floor family = best-of-family is not allowed; report the family mean and
        # the single best pre-registered floor (1e-3, the rank-19 point).
        a6 = series_from_variant("floor_0.001")
        a6_family_mean = []
        for c in cells:
            a6_family_mean.append(float(np.mean([c["variants"][f"floor_{f:g}"]["rmse"]
                                                 for f in F_GRID])))
        a6["family_mean_rmse"] = a6_family_mean
        arms["A6_floor"] = a6

        # A7 oracle: per-fold test-argmin over all 64 variants
        a7 = dict(rmse=[], margin=[], nlpd=[], c68=[], c95=[], hp=[], oracle_variant=[])
        for c in cells:
            bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
            j = int(np.argmin([c["variants"][v]["rmse"] for v in vnames]))
            vn = vnames[j]
            v = c["variants"][vn]
            a7["rmse"].append(v["rmse"])
            a7["margin"].append(1.0 - v["rmse"] / bb)
            a7["nlpd"].append(v["nlpd"])
            a7["c68"].append(v["c68"])
            a7["c95"].append(v["c95"])
            a7["hp"].append(v["hp"])
            a7["oracle_variant"].append(vn)
        arms["A7_oracle"] = a7

        # A8 shuffled-phase control
        a8 = dict(rmse=[], margin=[], nlpd=[], c68=[], c95=[], hp=[])
        for c in cells:
            v = sel[(n, c["fold_id"])]["shuffled_phase"]
            bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
            a8["rmse"].append(v["rmse"])
            a8["margin"].append(1.0 - v["rmse"] / bb)
            a8["nlpd"].append(v["nlpd"])
            a8["c68"].append(v["c68"])
            a8["c95"].append(v["c95"])
            a8["hp"].append(v["hp"])
        arms["A8_shuffled"] = a8

        # A10 tempered-noise diagnostic at the selected (k, gamma). Same mean prediction
        # as A1; only the predictive variance is replaced by the train residual variance.
        # This is what tells a reader whether the coverage deficit is a noise-inference
        # artefact rather than a property of the metric.
        a10 = dict(rmse=[], margin=[], nlpd=[], c68=[], c95=[], hp=[])
        for c in cells:
            v = sel[(n, c["fold_id"])]["tempered_noise"]
            bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
            a10["rmse"].append(v["rmse"])
            a10["margin"].append(1.0 - v["rmse"] / bb)
            a10["nlpd"].append(v["nlpd"])
            a10["c68"].append(v["c68"])
            a10["c95"].append(v["c95"])
            a10["hp"].append([float("nan")] * 3)
        arms["A10_tempered"] = a10

        # A9 mean predictor
        a9 = dict(rmse=[], margin=[], nlpd=[], c68=[], c95=[], hp=[])
        for c in cells:
            bb = bases[(n, c["fold_id"])]["best_baseline"]["rmse"]
            mp = c["mean_predictor_rmse"]
            a9["rmse"].append(mp)
            a9["margin"].append(1.0 - mp / bb)
            for key in ("nlpd", "c68", "c95"):
                a9[key].append(float("nan"))
            a9["hp"].append([float("nan")] * 3)
        arms["A9_meanpred"] = a9

        # ---- summary per arm -------------------------------------------------
        summary: dict[str, Any] = {}
        for aname, adesc in ARM_SPECS:
            a = arms[aname]
            r = np.asarray(a["rmse"], dtype=float)
            m = np.asarray(a["margin"], dtype=float)
            mp = np.asarray(a9["rmse"], dtype=float)
            summary[aname] = dict(
                description=adesc,
                folds=len(r),
                mean_far_end_rmse=float(r.mean()),
                worst_fold_far_end_rmse=float(r.max()),
                mean_rmse_over_mean_predictor=float((r / mp).mean()),
                mean_margin_vs_best_baseline=float(m.mean()),
                worst_fold_margin=float(m.min()),
                frac_folds_clearing_20pct=float(np.mean(m >= 0.20)),
                mean_nlpd=float(np.nanmean(a["nlpd"])) if np.isfinite(a["nlpd"]).any() else None,
                mean_coverage68=float(np.nanmean(a["c68"])) if np.isfinite(a["c68"]).any() else None,
                mean_coverage95=float(np.nanmean(a["c95"])) if np.isfinite(a["c95"]).any() else None,
                per_fold=dict(rmse=[float(x) for x in r],
                              margin=[float(x) for x in m],
                              nlpd=[None if not np.isfinite(x) else float(x) for x in a["nlpd"]],
                              c68=[None if not np.isfinite(x) else float(x) for x in a["c68"]],
                              c95=[None if not np.isfinite(x) else float(x) for x in a["c95"]],
                              hyperparams=[list(map(float, h)) for h in a["hp"]],
                              fold_id=[int(x) for x in fids],
                              seed=[int(c["seed"]) for c in cells],
                              scheme=[int(c["scheme"]) for c in cells]),
            )

        # ---- selected distribution ------------------------------------------
        kc: dict[str, int] = {}
        gc_: dict[str, int] = {}
        joint: dict[str, int] = {}
        for (k, g) in picked:
            kc[str(int(k))] = kc.get(str(int(k)), 0) + 1
            gc_[f"{float(g):g}"] = gc_.get(f"{float(g):g}", 0) + 1
            key = f"k{int(k)}_g{float(g):g}"
            joint[key] = joint.get(key, 0) + 1

        # ---- curves ----------------------------------------------------------
        def curve(filt) -> dict[str, float]:
            out = {}
            for key in filt:
                vals = []
                for c in cells:
                    if key in c["variants"]:
                        vals.append(c["variants"][key]["rmse"])
                if vals:
                    out[key] = float(np.mean(vals))
            return out

        gamma_at_k45 = {f"{g:g}": float(np.mean([c["variants"][f"k45_g{g}"]["rmse"]
                                                 for c in cells])) for g in G_GRID}
        k_at_g0 = {str(k): float(np.mean([c["variants"][f"k{k}_g0.0"]["rmse"]
                                          for c in cells])) for k in K_GRID}
        k_at_g05 = {str(k): float(np.mean([c["variants"][f"k{k}_g0.5"]["rmse"]
                                           for c in cells])) for k in K_GRID}
        k_at_g1 = {str(k): float(np.mean([c["variants"][f"k{k}_g1.0"]["rmse"]
                                          for c in cells])) for k in K_GRID}
        floor_f = {f"{f:g}": float(np.mean([c["variants"][f"floor_{f:g}"]["rmse"]
                                            for c in cells])) for f in F_GRID}
        floor_k = {f"{f:g}": int(np.mean([c["variants"][f"floor_{f:g}"]["dim"]
                                          for c in cells])) for f in F_GRID}

        # ---- diagnostics -----------------------------------------------------
        diag_out: dict[str, Any] = {}
        for tag in ("raw", "whitened_g0p5"):
            conds = [c["diagnostics"][tag]["cond_K"] for c in cells]
            pivs = [c["diagnostics"][tag]["min_cholesky_pivot"] for c in cells]
            js: dict[str, Any] = {}
            for j in (0.0, *JITTERS):
                ent = [c["diagnostics"][tag]["jitter_sweep"][f"{j:g}"] for c in cells]
                vals = [e["rmse"] for e in ent if e.get("rmse") is not None]
                nfail = sum(1 for e in ent if e.get("numerical_failure"))
                js[f"{j:g}"] = dict(
                    mean_rmse=float(np.mean(vals)) if vals else None,
                    n_folds_numerical_failure=int(nfail),
                    n_folds=int(len(ent)))
            diag_out[tag] = dict(cond_K=float(np.mean(conds)),
                                 cond_K_max=float(np.max(conds)),
                                 min_cholesky_pivot=float(np.nanmean(pivs)),
                                 min_cholesky_pivot_min=float(np.nanmin(pivs)),
                                 jitter_sweep=js,
                                 jitter_note=(
                                     "each entry carries the fold count and how many of "
                                     "them exceeded the mean predictor (guard g1: such an "
                                     "entry is a numerical failure, not a finding)"))

        # ---- baselines block -------------------------------------------------
        bn = [bases[(n, f["fold_id"])] for f in folds if (n, f["fold_id"]) in bases]
        winner: dict[str, int] = {}
        for b in bn:
            nm = b["best_baseline"]["name"]
            winner[nm] = winner.get(nm, 0) + 1
        baselines_block = dict(
            folds=len(bn),
            b1a=float(np.mean([b["b1a_rbf_median"]["rmse"] for b in bn])),
            b1b=float(np.mean([b["b1b_rbf_raw"]["rmse"] for b in bn])),
            b2=float(np.mean([b["b2_hop_laplacian"]["rmse"] for b in bn])),
            b3=float(np.mean([b["b3_admittance"]["rmse"] for b in bn])),
            b4=float(np.mean([b["b4_ridge"]["rmse"] for b in bn])),
            b5_primary=float(np.mean([b["b5p_lindist_honest"]["rmse"] for b in bn])),
            b5_primary_vsd_all=float(np.mean([b["b5p_lindist_honest"]["rmse_vsd_all"]
                                              for b in bn])),
            b5_p17secondary=float(np.mean([b["b5s_p17_jointmap"]["rmse"] for b in bn])),
            b5_p17secondary_vsd_all=float(np.mean([b["b5s_p17_jointmap"]["rmse_vsd_all"]
                                                   for b in bn])),
            mean_predictor=float(np.mean([b["mean_predictor_rmse"] for b in bn])),
            winner_counts=winner,
            lindist_order_chosen=dict(
                order1=sum(1 for b in bn if b["b5p_lindist_honest"]["order"] == "order1"),
                order2=sum(1 for b in bn if b["b5p_lindist_honest"]["order"] == "order2")),
        )

        per_n[str(n)] = dict(
            folds=len(cells),
            arms=summary,
            selected_distribution=dict(k_counts=kc, gamma_counts=gc_, joint=joint,
                                       tie_break="larger gamma then smaller k"),
            curves=dict(gamma_at_k45=gamma_at_k45, k_at_gamma0=k_at_g0,
                        k_at_gamma0p5=k_at_g05, k_at_gamma1=k_at_g1,
                        floor_f=floor_f, floor_f_effective_rank=floor_k),
            diagnostics=diag_out,
            baselines=baselines_block,
            crossover={},
        )

    # ---- crossovers ----------------------------------------------------------
    ns_all = [str(n) for n in N_GRID]
    ns_sorted = sorted(per_n, key=int)
    for label, arm in (("raw", "A0_raw"), ("selected", "A1_sel")):
        mm = [per_n[str(n)]["arms"][arm]["mean_margin_vs_best_baseline"] for n in ns_sorted]
        wf = [per_n[str(n)]["arms"][arm]["worst_fold_margin"] for n in ns_sorted]
        first_mean = next((int(n) for n, v in zip(ns_sorted, mm) if v > 0), None)
        first_all = next((int(n) for n, v in zip(ns_sorted, wf) if v > 0), None)
        for n in ns_sorted:
            per_n[str(n)]["crossover"][f"{label}_mean_margin"] = first_mean
            per_n[str(n)]["crossover"][f"{label}_every_fold"] = first_all
    # also baselines crossover: the physics baseline only (no GP) is always available
    for n in ns_sorted:
        per_n[str(n)]["crossover"]["note"] = (
            "mean_margin_crossover = first n at which the mean paired margin over the "
            "best per-fold baseline is positive; every_fold_crossover = first n at which "
            "the worst fold margin is positive")

    # ---- paired statistics ---------------------------------------------------
    n80 = per_n.get("80")
    paired: dict[str, Any] = {}
    if n80 is not None:
        def arr(arm, key):
            return np.asarray(n80["arms"][arm]["per_fold"][key], dtype=float)

        comparisons = {
            "A1_vs_best_baseline": arr("A1_sel", "margin"),
            "A1_vs_A0_raw": arr("A0_raw", "rmse") - arr("A1_sel", "rmse"),
            "A2_vs_A0_raw": arr("A0_raw", "rmse") - arr("A2_fixed_0p5", "rmse"),
            "A1_vs_A7_oracle": arr("A1_sel", "rmse") - arr("A7_oracle", "rmse"),
            "A4_vs_A0_raw": arr("A0_raw", "rmse") - arr("A4_gamma1", "rmse"),
        }
        for key, d in comparisons.items():
            lo, hi = _bootstrap_ci(d)
            paired[key] = dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()),
                               n=int(d.size),
                               boot95_ci=[lo, hi], wilcoxon_p=_wilcoxon(d))
    # A1-vs-baseline is a margin, so report a separate paired RMSE comparison too
    for nn in ns_sorted:
        pass

    sel_opt = None
    if n80 is not None:
        sel_opt = float(n80["arms"]["A1_sel"]["mean_far_end_rmse"]
                        - n80["arms"]["A7_oracle"]["mean_far_end_rmse"])

    # ---- survival ------------------------------------------------------------
    s1 = bool(n80["arms"]["A1_sel"]["mean_margin_vs_best_baseline"] >= 0.20) if n80 else False
    s2 = bool(n80["arms"]["A1_sel"]["worst_fold_margin"] >= 0.20) if n80 else False
    survival = dict(S1_mean_ge_20pct=s1, S2_worst_fold_ge_20pct=s2,
                    S3_no_candidate_beats_by_15pct="evaluated downstream",
                    survives=bool(s1 and s2),
                    note=("S1 and S2 are decided here at n=80 on the 2.0x regime; S3 needs "
                          "the other four screened candidates and is decided downstream. "
                          "A screen where nothing clears +20% is a real result about the "
                          "request and is reported as such, never re-thresholded."))

    ranking = []
    if n80 is not None:
        cand = [(a, n80["arms"][a]) for a in n80["arms"]
                if a not in ("A9_meanpred",)]
        cand.sort(key=lambda t: (-t[1]["mean_margin_vs_best_baseline"],
                                 -t[1]["worst_fold_margin"],
                                 abs((t[1]["mean_coverage68"] or 0.68) - 0.68)))
        for rank, (a, s) in enumerate(cand, 1):
            ranking.append(dict(rank=rank, arm=a, description=s["description"],
                                mean_margin=s["mean_margin_vs_best_baseline"],
                                worst_fold_margin=s["worst_fold_margin"],
                                mean_rmse=s["mean_far_end_rmse"],
                                frac_folds_clearing_20pct=s["frac_folds_clearing_20pct"],
                                clears_20pct=bool(s["mean_margin_vs_best_baseline"] >= 0.20)))
    ranking_report = dict(
        measure=("mean paired margin in far-end normalised test RMSE over the best "
                 "per-fold baseline, n=80, 2.0x regime"),
        ordering=ranking,
        nothing_clears_20pct=bool(not any(r["clears_20pct"] for r in ranking)),
        statement=("No arm clears the pre-registered +20% mean-margin bar at n=80; this is "
                   "reported as the result of the screen, not re-thresholded."
                   if not any(r["clears_20pct"] for r in ranking)
                   else "At least one arm clears the +20% mean-margin bar at n=80."),
    )

    corrections = [
        ("whitening exponent", (
            "The direction text writes S_gamma = U diag((sigma/sigma_1)^gamma) Vt and calls "
            "gamma=0 the raw operator. That is inverted: at gamma=0 it yields U Vt, the fully "
            "whitened semi-orthogonal map, and at gamma=1 the raw operator. This artifact "
            "implements the ratio form w_i = (sigma_i/sigma_0)^(1-gamma) used by the working "
            "pilot code, and asserts the orientation before any modelling.")),
        ("gamma-0 assertion is impossible as written", (
            "STEP 3 asserts Phi_{45,0}(x) == S x / sigma_0 elementwise. The ratio form drops "
            "the left singular vector U, an orthogonal rotation that leaves every RBF kernel "
            "bit-identical, so the assertion is implemented as U @ Phi_{45,0}(x) == S x/sigma_0 "
            "to 1e-10 plus equality of the squared-distance matrices. Measured deviation is "
            "recorded in results/protocol_checks.json.")),
        ("LinDistFlow baseline recalibrated", (
            "The archived p17 joint-map calibration (one 46-column trace-scaled ridge "
            "penalising the intercept, CV grid floor 1e-4 reached in 20/20 seeds) is replaced "
            "as the margin reference by per-output ridge with an unpenalised intercept on a "
            "logspace(-8,3,12) trace-relative grid. The p17 calibration is retained as a "
            "labelled secondary row so this table maps onto the earlier one.")),
        ("normalisation denominator", (
            "The archived p17 and final_table tables divided by the ALL-ROW per-output sd, "
            "leaking test-set spread into the metric. This artifact divides by the TRAINING "
            "rows of each fold, so it will not reproduce p17 to the digit even where the model "
            "is identical; both denominators are reported for the LinDistFlow row at every n.")),
        ("RBF lengthscale grid", (
            "The archived RBF arms searched lengthscales up to 8 while squared distances in 110 "
            "standardised dimensions are near 220, making exp(-220/128)=0.18 an almost constant "
            "kernel. Both a median-normalised (primary) and the raw-scale (secondary) form are "
            "reported so the archived 'RBF sits at the mean predictor' claim can be re-read.")),
        ("admittance competitor dimensionality", (
            "probe12 measured the admittance-Laplacian competitor at 55 dimensions against a "
            "110-dimensional treatment. Here it is built at 110 dimensions: a real feature map "
            "taking the real and imaginary parts of the own-phase voltage drop Z_l @ s.")),
        ("fold count", (
            "12 seeds x 2 schemes gives 24 folds at n<=80 and 12 at n=160, i.e. 108 fold-cells, "
            "not the 20 seeds x 3 schemes (60 folds) of the earlier direction and not the 30 "
            "folds per cell claimed in earlier text. Scheme 1 cannot draw 160 rows from a "
            "110-row half, so that cell is skipped by construction. Every results cell carries "
            "its own fold count.")),
        ("seeds reserve splits, not data", (
            "Seeds 12 to 29 are reserved and were not evaluated. Reserving seeds reserves "
            "SPLITS, not data: the same 220 draws underlie every fold. The genuinely untouched "
            "evidence is the 2.6x regime and the moved-phase feeder.")),
        ("selection-optimism row added", (
            "A per-fold oracle (best of 64 variants chosen on TEST error) is reported beside the "
            "train-selected arm, so a train-CV-selected grid cannot be confused with "
            "grid-shopping. Reported as selection_optimism_at_80.")),
        ("shuffled-phase control scope", (
            "This artifact carries only the cheap consistency form of the shuffled-phase "
            "control - the selected whitening re-run with the operator's load-to-phase "
            "assignment permuted while the data stay on real phases. It is not the "
            "physically-moved-phase control, which the direction reserves for the survivor, so "
            "this artifact can say nothing about phase identity.")),
        ("supervision beyond the spec", (
            "A6_floor reports the pre-registered floor 1e-3 (the rank-19 point) as its single "
            "row, with the family mean carried alongside; selecting the best floor on test "
            "error would be exactly the grid-shopping the oracle row exists to bound.")),
    ]

    screen = dict(
        candidate=CANDIDATE_ID,
        mechanism_one_line=MECHANISM,
        protocol=dict(
            n_grid=list(N_GRID), k_grid=list(K_GRID), gamma_grid=list(G_GRID),
            f_grid=list(F_GRID), seed_base=SEED_BASE, seeds_used="0-11",
            schemes=[0, 1],
            folds_per_n={str(n): len(folds_for(n)) for n in N_GRID},
            total_fold_cells=int(sum(len(folds_for(n)) for n in N_GRID)),
            reserved=["seeds 12-29", "n equal 60", "n equal 120", "scheme 2",
                      "regimes 1.0x and 2.6x", "moved-phase feeder", "near-end targets"],
            reserved_note=("Reserving seeds reserves SPLITS, not data: every fold draws from "
                           "the same 220-row pool, so the genuinely untouched evidence is the "
                           "2.6x regime and the moved-phase feeder."),
            denominator=("mean per-output standard deviation of the TRAINING rows of each fold "
                         "(leak-free); the all-row denominator is reported beside it for the "
                         "LinDistFlow row so the change from the archived tables is quantified"),
            baseline_primary=("per-output ridge with an unpenalised intercept on a "
                              "logspace(-8,3,12) trace-relative grid chosen by train-only CV "
                              "on multiphase LinDistFlow"),
            margin_reference=("minimum over b1a, b2, b3, b4 and b5-primary, per fold"),
            reused_ensembles=("archived 220-draw runpp_3ph pools at the 2.0x regime, "
                              "never re-solved on the measurement path"),
            gp_hyperparameters=dict(lgrid=list(LGRID), s2=list(S2G), noise=list(N2G),
                                    convention="one shared triple per arm and fold, chosen by "
                                               "marginal likelihood over all 45 outputs"),
            selection=dict(k_coarse=list(K_GRID_COARSE), gamma_coarse=list(G_GRID_COARSE),
                           reps=SEL_REPS, tie_break="larger gamma then smaller k"),
            nproc=N_PROC,
            likwid=None,
        ),
        structural_facts={k: v for k, v in structures.items()},
        per_n=per_n,
        screen_metric_at_80=(n80["arms"]["A1_sel"]["mean_margin_vs_best_baseline"]
                             if n80 else None),
        selection_optimism_at_80=sel_opt,
        survival=survival,
        ranking_report=ranking_report,
        corrections_to_prior_text=corrections,
    )
    screen["protocol"].pop("likwid", None)
    # the paired statistics live inside the n=80 cell, so the top level holds exactly the
    # ten contract keys and no more
    if n80 is not None:
        screen["per_n"]["80"]["paired_statistics"] = paired
        screen["per_n"]["80"]["paired_statistics_note"] = (
            "paired per-fold differences at n=80 (24 folds). The folds share one 220-row "
            "draw pool, so these bootstrap intervals describe split-to-split variability, "
            "NOT variability across new injection draws. A1_vs_best_baseline is the margin "
            "itself (positive = candidate better than that fold's best baseline); "
            "A1_vs_A0_raw, A2_vs_A0_raw and A4_vs_A0_raw are RMSE differences signed so "
            "positive = the first-named arm is better; A1_vs_A7_oracle is signed so positive "
            "means the train-selected rule is WORSE than the per-fold oracle.")
    # RBF claim re-tested against BOTH the alternative scaling and the mean predictor,
    # not assumed. Two separate questions: (i) is the median-normalised form materially
    # better than the raw-scale form the archived arms used (a grid artefact)? (ii) does
    # either form actually sit AT the mean predictor at every n<=160 as claimed?
    for nk in ns_all:
        if nk not in per_n:
            continue
        bb = per_n[nk]["baselines"]
        gap = (bb["b1b"] - bb["b1a"]) / max(bb["b1b"], 1e-12)
        ratio_a = bb["b1a"] / max(bb["mean_predictor"], 1e-12)
        ratio_b = bb["b1b"] / max(bb["mean_predictor"], 1e-12)
        per_n[nk]["baselines"]["rbf_median_normalisation_gain"] = float(gap)
        per_n[nk]["baselines"]["b1a_over_mean_predictor"] = float(ratio_a)
        per_n[nk]["baselines"]["b1b_over_mean_predictor"] = float(ratio_b)
        at_mean = (ratio_a > 0.97) and (ratio_b > 0.97)
        per_n[nk]["baselines"]["rbf_restatement"] = (
            "At n={} the median-normalised RBF (b1a) scores {:.4f} and the raw-scale form "
            "(b1b, what the archived arms used) {:.4f}, against a mean predictor of "
            "{:.4f}. b1a is {:.1%} {} than b1b, so the median heuristic {} the archived "
            "result. Measured as a ratio to the mean predictor the two forms sit at "
            "{:.2f}x and {:.2f}x, so the archived claim that an injection RBF sits at the "
            "mean predictor {} at this n.".format(
                nk, bb["b1a"], bb["b1b"], bb["mean_predictor"], abs(gap),
                "better" if gap > 0 else "worse",
                "DOES explain" if gap > 0.02 else "does NOT explain",
                ratio_a, ratio_b,
                "DOES reproduce" if at_mean else "does NOT reproduce"))
    screen["corrections_to_prior_text"].append(
        ("RBF mean-predictor claim re-tested, not restated",
         "The archived claim that an isotropic RBF on injections sits at the mean predictor "
         "is re-tested by comparing the median-normalised form (b1a, primary) against the "
         "raw 110-dim scale (b1b, what the archived arms used) at every n. The measured "
         "gap is recorded in per_n[n].baselines.rbf_median_normalisation_gain."))
    screen["corrections_to_prior_text"].append(
        ("GP hyperparameter criterion",
         "The archived p17 / p8_full / pilot GP scored -0.5||K^-1 Yc||_F^2 - 0.5 m log|K|, "
         "the squared norm of the solved coefficients, which is not the marginal likelihood "
         "(-0.5 tr(Yc^T K^-1 Yc) - 0.5 m log|K|). This artifact keeps the archived criterion "
         "so its arms are comparable with the earlier tables, and verifies it against a "
         "dense reference in test_gp.py; both criteria are implemented in gp_corr and the "
         "correct one is available as mll_mode='quadratic'."))
    screen["corrections_to_prior_text"].append(
        ("shuffled-phase operator construction",
         "STEP 3 of the plan writes default_rng(31).permutation(55) 'applied to the 55 load "
         "phases', which if applied to the label array yields integers 0..54 that are not "
         "phase indices and raises IndexError inside build_S. The executable control - the "
         "one the archived p8_full shufLab arm built - is phase_shuffled = phase_true[perm], "
         "a permutation of which load carries which phase. Implemented that way and recorded "
         "in protocol_checks.json."))
    screen["corrections_to_prior_text"].append(
        ("jitter sweep interpretation",
         "The jitter sweep ADDS j to the selected noise level rather than replacing it, and "
         "includes j=0 as the selected-fit reference row. Replacing the noise with j tests a "
         "different model, not the numerical stability of the selected one; a first "
         "implementation did that and produced guard-g1 failures (normalised RMSE 1.7e2 "
         "against a mean predictor of 1.2 on n=40 fold 4000) that were numerical artefacts."))
    screen["corrections_to_prior_text"].append(
        ("cond(S) is a round-off residue, not a reproducible number",
         "S is rank deficient (29 of 45 singular values above the double-precision floor), so "
         "cond(S) is the reciprocal of a round-off residue: 4.99e+32 through one LAPACK path "
         "and 1.13e+16 through another, on elementwise identical input, against the archived "
         "1.28873e+32. The reproducible statements are the effective ranks at 1e-2/1e-3/1e-4, "
         "the stable rank 2.1768, and cond over the rank-23 core (2230). Reported in "
         "structural_facts.json with this note attached."))
    screen["corrections_to_prior_text"].append(
        ("tempered-noise diagnostic added",
         "The direction rests confidence on coverage, so a tempered-noise row (A10) is "
         "reported beside every Gaussian-process arm: the marginal-likelihood mean with the "
         "predictive variance fixed at the TRAIN empirical residual variance. If A10's "
         "coverage is near nominal while A1's is not, the coverage deficit is a "
         "noise-inference artefact rather than a property of the metric."))

    jdump(screen, RESULTS / "screen_B.json")

    # ---- method_out.json (exp_gen_sol_out contract) --------------------------
    examples = []
    for n in N_GRID:
        for f in folds_for(n):
            p = RAW / f"cell_n{n}_f{f['fold_id']}.json"
            if not p.exists():
                continue
            c = jload(p)
            b = bases[(n, f["fold_id"])]
            kk, gg = sel[(n, f["fold_id"])]["selected"]
            v = c["variants"][f"k{int(kk)}_g{float(gg)}"]
            examples.append(dict(
                input=(f"fold n={n} seed={f['seed']} scheme={f['scheme']}: predict the 45 "
                       f"far-end phase voltages with the whitened pullback kernel at the "
                       f"train-selected (k={int(kk)}, gamma={float(gg)}), 2.0x regime"),
                output=(f"far-end normalised test RMSE {v['rmse']:.6f}; best per-fold "
                        f"baseline {b['best_baseline']['name']} at "
                        f"{b['best_baseline']['rmse']:.6f}; paired margin "
                        f"{1 - v['rmse'] / b['best_baseline']['rmse']:+.4f}; "
                        f"NLPD {v['nlpd']:.4f}; coverage68 {v['c68']:.4f}; "
                        f"coverage95 {v['c95']:.4f}"),
                metadata_n=int(n), metadata_fold_id=int(f["fold_id"]),
                metadata_seed=int(f["seed"]), metadata_scheme=int(f["scheme"]),
                metadata_selected_k=int(kk), metadata_selected_gamma=float(gg),
                metadata_vsd_train=float(b["vsd_train"]),
                # the schema requires every predict_* field to be a STRING
                predict_whitened_kernel=f'{v["rmse"]:.6f}',
                predict_best_baseline=f'{b["best_baseline"]["rmse"]:.6f}',
                predict_mean_predictor=f'{b["mean_predictor_rmse"]:.6f}',
            ))
    method_out = dict(
        metadata=dict(
            method_name=CANDIDATE_ID,
            description=("Screen candidate B: whiten versus truncate the target-restricted "
                         "pullback sensitivity operator of a three-phase unbalanced LV feeder."),
            screen_B=screen,
            protocol_checks=checks,
        ),
        datasets=[dict(dataset="ieee_pes_european_lv_asymmetric_on_peak_566_2.0x",
                       examples=examples)],
    )
    jdump(method_out, RESULTS / "method_out.json")
    jdump(screen["protocol"], RESULTS / "protocol_record.json")
    logger.info("screen_B.json, method_out.json written ({} examples)", len(examples))
    return screen


# ---------------------------------------------------------------------------
# STAGE: figures
# ---------------------------------------------------------------------------
def stage_figures() -> None:
    """Render figures through the aii-data-fig-gen generator (the run's house style).

    Every figure is drawn from the numbers already in results/screen_B.json, so the
    picture cannot disagree with the table. The spec JSONs themselves are written to
    results/figure_specs.json, which is the machine-readable form of every plotted
    series, and the raw per-fold arrays stay in screen_B.json under per_n[*].arms[*]
    .per_fold, so any claim in a figure can be re-checked from the underlying folds and
    not only from the plotted mean.
    """
    import subprocess

    # the figure generator is resolved from an env var so the repository carries no
    # absolute server path; set AII_FIG_GEN to the chart_gen.py of the aii-data-fig-gen
    # skill. When it is absent the figures are skipped and every number still lands in
    # results/ and figures/*_spec.json.
    skill = Path(os.environ.get(
        "AII_FIG_GEN",
        str(ROOT.parents[0] / "skills" / "aii-data-fig-gen" / "scripts" / "chart_gen.py")))
    if not skill.exists():
        skill = Path("/ai-inventor/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py")
    if not skill.exists():
        logger.warning("chart_gen.py not found; skipping figures")
        return
    screen = jload(RESULTS / "screen_B.json")
    ns = [str(n) for n in N_GRID]
    nnum = [int(n) for n in N_GRID]
    specs: dict[str, dict[str, Any]] = {}

    # --- gamma curve at fixed rank -------------------------------------------------
    gvals = [float(g) for g in G_GRID]
    specs["gamma_curve"] = dict(
        type="line",
        title="Partial-whitening exponent at fixed truncation rank k=45",
        xlabel="partial-whitening exponent gamma",
        ylabel="mean far-end normalised RMSE",
        series=[dict(label=f"n={n}", x=gvals,
                     values=[screen["per_n"][n]["curves"]["gamma_at_k45"][f"{g:g}"]
                             for g in gvals])
                for n in ns])

    # --- truncation-rank curve at gamma 0 and gamma 1 ------------------------------
    kvals = [int(k) for k in K_GRID]
    ser = [dict(label=f"n={n}, gamma=0", x=kvals,
                values=[screen["per_n"][n]["curves"]["k_at_gamma0"][str(k)] for k in kvals])
           for n in ns]
    ser += [dict(label=f"n={n}, gamma=1", x=kvals,
                 values=[screen["per_n"][n]["curves"]["k_at_gamma1"][str(k)] for k in kvals])
            for n in ns]
    specs["k_curve"] = dict(
        type="line",
        title="Truncation rank at gamma=0 (raw) and gamma=1 (full whitening)",
        xlabel="truncation rank k", ylabel="mean far-end normalised RMSE", series=ser)

    # --- jitter sweep --------------------------------------------------------------
    jit = [0.0, *JITTERS]
    jlab = [f"{j:g}" for j in jit]
    ser = []
    for n in ns:
        for tag, lab in (("raw", "raw"), ("whitened_g0p5", "gamma=0.5")):
            js = screen["per_n"][n]["diagnostics"][tag]["jitter_sweep"]
            # x is the log10 of (j + 1e-9) so j=0 plots at the far left on a real axis;
            # the jitter levels themselves are recorded in results/figure_specs.json
            ser.append(dict(label=f"n={n} {lab}",
                            x=[float(np.log10(j + 1e-9)) for j in jit],
                            values=[js[j]["mean_rmse"] for j in jlab]))
    specs["jitter_sweep"] = dict(
        type="line",
        title="Jitter sweep: ADDED diagonal jitter over the selected noise level",
        xlabel="log10 of the added diagonal jitter j (leftmost point is j = 0)",
        ylabel="mean far-end normalised RMSE", series=ser)

    # --- RMSE versus n for every arm plus the physics baseline ---------------------
    arm_order = ["A0_raw", "A1_sel", "A2_fixed_0p5", "A3_fixed_0p5_k11", "A4_gamma1",
                 "A5_trunc_inert", "A6_floor", "A7_oracle", "A8_shuffled"]
    ser = [dict(label=a, x=nnum,
                values=[screen["per_n"][n]["arms"][a]["mean_far_end_rmse"] for n in ns])
           for a in arm_order]
    ser.append(dict(label="best per-fold baseline",
                    x=nnum,
                    values=[screen["per_n"][n]["baselines"]["b5_primary"] for n in ns]))
    ser.append(dict(label="mean predictor",
                    x=nnum,
                    values=[screen["per_n"][n]["arms"]["A9_meanpred"]["mean_far_end_rmse"]
                            for n in ns]))
    specs["rmse_vs_n"] = dict(
        type="line",
        title="Far-end normalised RMSE versus number of AC runs (2.0x regime)",
        xlabel="n (AC runs in the training set)",
        ylabel="mean far-end normalised RMSE", series=ser)

    # --- selection heatmap over the pooled 108 fold-cells --------------------------
    gk = [f"{g:g}" for g in G_GRID]
    kk = [str(k) for k in K_GRID]
    mat = []
    for kv in kk:
        row = []
        for gv in gk:
            key = f"k{kv}_g{gv}"
            row.append(sum(screen["per_n"][n]["selected_distribution"]["joint"].get(key, 0)
                           for n in ns))
        mat.append(row)
    specs["selected_hist"] = dict(
        type="heatmap",
        title="Where train-only selection lands, pooled over the 108 fold-cells",
        xlabel="selected gamma", ylabel="selected truncation rank k",
        cbar_label="fold-cells selecting this pair",
        row_labels=kk, col_labels=gk, matrix=mat)

    # --- paired margin over the best baseline, with its spread --------------------
    ser = []
    for a in ("A1_sel", "A0_raw"):
        ser.append(dict(label=a, x=nnum,
                        values=[100.0 * screen["per_n"][n]["arms"][a]
                                ["mean_margin_vs_best_baseline"] for n in ns]))
    ser.append(dict(label="A1_sel worst fold", x=nnum,
                    values=[100.0 * screen["per_n"][n]["arms"]["A1_sel"]["worst_fold_margin"]
                            for n in ns]))
    specs["margin_vs_n"] = dict(
        type="line",
        title="Paired margin over the best per-fold baseline (percent)",
        xlabel="n (AC runs in the training set)",
        ylabel="margin in normalised RMSE (%)", series=ser)

    rendered = []
    for name, spec in specs.items():
        p = FIGS / f"{name}_spec.json"
        p.write_text(json.dumps(spec, indent=2, default=str))
        out = FIGS / name
        try:
            r = subprocess.run([sys.executable, str(skill), "--spec", str(p),
                                "--out", str(out)], capture_output=True, text=True,
                               timeout=300)
            if r.returncode != 0:
                logger.warning("figure {} failed: {}", name, (r.stderr or "")[-1200:])
            else:
                logger.info("figure {} rendered", name)
                rendered.append(name)
        except subprocess.TimeoutExpired:
            logger.warning("figure {} timed out", name)

    jdump(specs, RESULTS / "figure_specs.json")
    jdump(dict(rendered=rendered, specs=list(specs)), RESULTS / "figure_index.json")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["checks", "baselines", "sweep", "select", "stats",
                             "figures", "all"])
    ap.add_argument("--n", type=int, nargs="*", default=None,
                    help="restrict the sweep to these n values")
    args = ap.parse_args()

    ram_cap = 12 * 1024 ** 3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (ram_cap, ram_cap))
    except (ValueError, OSError):
        pass

    t0 = time.time()
    if args.stage in ("checks", "all"):
        stage_checks()
    if args.stage in ("baselines", "all"):
        stage_baselines()
        gc.collect()
    if args.stage in ("sweep", "all"):
        stage_sweep(args.n)
        gc.collect()
    if args.stage in ("select", "all"):
        stage_select()
        gc.collect()
    if args.stage in ("stats", "all"):
        stage_stats()
        gc.collect()
    if args.stage in ("figures", "all"):
        stage_figures()
    logger.info("stage {} complete in {:.1f}s", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
