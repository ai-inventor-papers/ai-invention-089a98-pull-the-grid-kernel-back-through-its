"""One kernel implementation, shared verbatim by every model in the screen.

Everything here is fitted on TRAINING rows only.  No function in this module
ever receives the test targets; the predictions it returns are the only thing
that leaves it.  That is what makes the leakage test in method.py meaningful.

Contents
--------
sqd, gram          : squared Euclidean distances without an (n,m,d) intermediate
feats_from_S       : the output-restricted sensitivity feature map (train-only scale)
feats_from_metric  : 55x55 graph-metric feature map (train-only scale)
gp_fit / gp_predict: GP with an isotropic kernel, hyperparameters by marginal
                     likelihood on the training block, LOO predictive moments
                     for the calibration arms
ridge_cv           : per-output ridge with the penalty chosen by train-only CV
                     over a scale-free grid (fractions of the design-Gram lmax)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import solve_triangular

# --- pinned constants, identical to the other artifacts of this screen --------
GP_LGRID: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GP_S2GRID: Tuple[float, ...] = (0.3, 1.0, 3.0)
GP_N2GRID: Tuple[float, ...] = (1e-8, 1e-5, 1e-3, 1e-1)
PENALTY_FRACS: np.ndarray = np.logspace(-6, 6, 25)
JITTER: Tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
RBF_DIVISOR = 110.0  # squared distance divided by the input dimension
ALPHA_GRID: np.ndarray = np.logspace(np.log10(0.05), np.log10(20.0), 13)


def sqd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of a and rows of b."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def gram(f: np.ndarray) -> np.ndarray:
    g = f @ f.T
    d = np.diag(g)
    return np.maximum(d[:, None] + d[None, :] - 2.0 * g, 0.0)


# ---------------------------------------------------------------------------
# feature maps
# ---------------------------------------------------------------------------
def feats_from_S(s: np.ndarray, xtr: np.ndarray, xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sx features with the scale taken from TRAIN rows only."""
    sc = np.sqrt(((xtr @ s.T) ** 2).sum(1).mean()) + 1e-12
    return (xtr @ s.T) / sc, (xte @ s.T) / sc


def feats_from_metric(m: np.ndarray, xtr: np.ndarray, xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Graph-metric features through the symmetric square root: F = P M^{1/2}.

    This is the standard PSD metric feature map, because it makes the squared
    distance between two rows exactly the Mahalanobis quantity
    ||f_i - f_j||^2 = (p_i - p_j)^T M (p_i - p_j).
    The scale is taken from TRAIN rows only, as ||p M^{1/2}||_RMS.

    CORRECTION RECORDED: the earlier construction of this run used F = P M with a
    scale of sqrt(mean p^T M p), which is dimensionally inconsistent (the scale is
    the norm of P M^{1/2} while the feature is P M) and made the unit-weight and
    impedance-weight graph baselines degenerate to the constant mean predictor.
    The Mahalanobis form above is the correct construction and is what every
    graph-kernel baseline here uses; the change is reported in the results file.

    The 55x55 metric acts on the load-MAGNITUDE block, so P and Q share one
    metric (the construction this run has used throughout; recorded in results).
    """
    m = np.asarray(m, dtype=float)
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    m_half = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T
    ptr, pte = xtr[:, :55], xte[:, :55]
    ftr, fte = ptr @ m_half, pte @ m_half
    sc = float(np.sqrt(max((ftr ** 2).sum(1).mean(), 1e-30)))
    return ftr / sc, fte / sc


def standardise(xtr: np.ndarray, xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mx, sx = xtr.mean(0), xtr.std(0)
    sx = np.where(sx < 1e-12, 1.0, sx)
    return (xtr - mx) / sx, (xte - mx) / sx


# ---------------------------------------------------------------------------
# GP
# ---------------------------------------------------------------------------
@dataclass
class GPFit:
    ell: float
    s2: float
    n2: float
    L: np.ndarray          # lower Cholesky factor of K = s2 exp(-D/(2 l^2)) + n2 I
    alpha: np.ndarray
    ybar: np.ndarray
    ysd: np.ndarray
    kinv_diag: np.ndarray
    loo_mu: np.ndarray
    loo_var: np.ndarray
    mll: float
    jitter_used: float
    info: dict = field(default_factory=dict)


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L L^T x = b, with L lower triangular.

    Deliberately uses numpy's Cholesky plus an explicit triangular solve.  The
    scipy build in this image (1.16.3) returns the UPPER factor from
    `cho_factor(a, lower=True)` while `cho_solve` assumes the flag it was given,
    which silently produced wrong predictors on ill-conditioned kernels: a ~1000x
    error in the log marginal likelihood at ell=1, n2=1e-3, and a hyperparameter
    choice that cost 2.5x in test RMSE.  Caught by the T3/T4 numeric expectations
    in test_stages.py, not by chance.  `assert_cholesky_convention` below re-checks
    the convention at run time.
    """
    z = solve_triangular(L, b, lower=True, check_finite=False)
    return solve_triangular(L.T, z, lower=False, check_finite=False)


def assert_cholesky_convention() -> dict:
    """Verify the triangular-solve convention against a direct dense solve."""
    rng = np.random.default_rng(12345)
    a = rng.standard_normal((40, 7))
    k = a @ a.T + 0.05 * np.eye(40)
    b = rng.standard_normal((40, 3))
    L = np.linalg.cholesky(k)
    err = float(np.max(np.abs(_chol_solve(L, b) - np.linalg.solve(k, b))))
    err_upper = float(np.max(np.abs(L - np.triu(L).T)))
    return {"chol_solve_vs_direct_max_abs_error": err,
            "factor_is_lower_triangular": bool(err_upper == 0.0),
            "passes": bool(err < 1e-10)}


def gp_fit(dtr: np.ndarray, ytr: np.ndarray, need_loo: bool = True) -> GPFit:
    """Shared-hyperparameter GP by marginal likelihood; LOO moments from K^-1 diagonal.

    Hyperparameters (lengthscale, signal variance, noise variance) are shared by
    all outputs, as in every GP-for-power-flow reference used by this screen.
    """
    n, m = ytr.shape
    ybar = ytr.mean(0)
    ysd = ytr.std(0) + 1e-12
    yc = (ytr - ybar) / ysd
    eye = np.eye(n)
    best = None
    for ell in GP_LGRID:
        dl = np.exp(-dtr / (2.0 * ell * ell))
        for s2 in GP_S2GRID:
            for n2 in GP_N2GRID:
                base = s2 * dl + n2 * eye
                L = None
                jit_used = np.nan
                for jit in JITTER:
                    try:
                        L = np.linalg.cholesky(base + jit * eye)
                        jit_used = jit
                        break
                    except np.linalg.LinAlgError:
                        continue
                if L is None:
                    continue
                # the quadratic form of the log marginal likelihood is
                # y^T K^-1 y = ||L^-1 y||^2, i.e. the FORWARD triangular solve only.
                # Using the full solve K^-1 y here instead is a real bug: it squares
                # the wrong vector and drives the hyperparameter search to an
                # over-smooth corner (this artifact had it, and it cost 2.5x RMSE
                # at n=160 before the T3/T4 numeric expectations caught it).
                fwd = solve_triangular(L, yc, lower=True, check_finite=False)
                alpha = _chol_solve(L, yc)
                mll = -0.5 * float((fwd ** 2).sum()) - m * float(np.log(np.diag(L)).sum())
                if best is None or mll > best["mll"]:
                    best = {"mll": mll, "ell": ell, "s2": s2, "n2": n2, "L": L,
                            "alpha": alpha, "jit": jit_used}
    if best is None:
        raise RuntimeError("GP grid exhausted without a single successful Cholesky factorisation")
    L = best["L"]
    if need_loo:
        kinv = _chol_solve(L, eye)
        kinv_diag = np.clip(np.diag(kinv), 1e-12, None)
        # K^-1 y for the LOO identities (Rasmussen & Williams eq. 5.12)
        kinv_y = _chol_solve(L, yc)
        loo_mu = ybar[None, :] + ysd[None, :] * (yc - kinv_y / kinv_diag[:, None])
        loo_var = (ysd ** 2)[None, :] * np.clip(1.0 / kinv_diag[:, None] - best["n2"], 1e-12, None)
    else:
        kinv_diag = np.full(n, np.nan)
        loo_mu = np.full_like(ytr, np.nan)
        loo_var = np.full_like(ytr, np.nan)
    return GPFit(ell=best["ell"], s2=best["s2"], n2=best["n2"], L=L,
                 alpha=best["alpha"], ybar=ybar, ysd=ysd, kinv_diag=kinv_diag,
                 loo_mu=loo_mu, loo_var=loo_var, mll=best["mll"], jitter_used=best["jit"])


def gp_predict(fit: GPFit, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance, shape (n_test, n_out). No explicit inverse."""
    ks = fit.s2 * np.exp(-dst / (2.0 * fit.ell * fit.ell))
    mu = (ks @ fit.alpha) * fit.ysd[None, :] + fit.ybar[None, :]
    sol = _chol_solve(fit.L, ks.T)
    var = fit.s2 - (ks.T * sol).sum(0)
    var = np.clip(var, 1e-12, None)[:, None]
    return mu, var * (fit.ysd ** 2)[None, :] + 1e-18


# ---------------------------------------------------------------------------
# ridge, with the penalty by train-only cross-validation
# ---------------------------------------------------------------------------
def cv_folds(n: int) -> Sequence[np.ndarray]:
    """max(2, n//4) contiguous folds as in the shared protocol."""
    return np.array_split(np.arange(n), max(2, n // 4))


def _ridge_center(xtr: np.ndarray, xte: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Split a design matrix into a penalised block plus an UNPENALISED intercept.

    The intercept of a calibration must not be shrunk: penalising it makes the
    heavily-regularised model predict zero instead of the training mean, which
    turns any physics baseline into a straw man at small n.  The standard
    equivalent is to centre the design and the targets, fit the penalty on the
    centred block, and recover the intercept as ybar - xbar @ coef.
    """
    mx = xtr.mean(0)
    mxe = xte.mean(0)
    return xtr - mx, xte - mxe, mx, mxe


def _ridge_path(w: np.ndarray, v: np.ndarray, b: np.ndarray, wt: np.ndarray,
                penal_fracs: np.ndarray, lmax: float) -> np.ndarray:
    """Coefficients of the CENTRED ridge for every penalty, one eigendecomposition."""
    proj = v.T @ b
    out = np.empty((len(penal_fracs), b.shape[0], b.shape[1]))
    for pi, frac in enumerate(penal_fracs):
        out[pi] = v @ (proj / (w + frac * lmax)[:, None])
    _ = wt
    return out


def ridge_cv(xc_tr: np.ndarray, ytr: np.ndarray, xc_te: np.ndarray,
             penal_fracs: np.ndarray = PENALTY_FRACS) -> dict:
    """Per-output ridge on a centred design; penalty = fraction of the Gram lmax.

    The intercept is NOT penalised (see `_ridge_center`).  A scale-free penalty
    grid is used because these design matrices have condition numbers above 1e6:
    an absolute grid would make the model either an interpolator or a constant.

    Exact and fast: one eigendecomposition per cross-validation fold yields the
    whole 25-point coefficient path.
    """
    n = len(xc_tr)
    ybar = ytr.mean(0)
    yc = ytr - ybar
    g = xc_tr.T @ xc_tr
    g = 0.5 * (g + g.T)
    w, v = np.linalg.eigh(g)
    w = np.clip(w, 0.0, None)
    lmax = float(w[-1]) if w[-1] > 0 else 1.0
    folds = cv_folds(n)
    scores = np.zeros(len(penal_fracs))
    for f in folds:
        tri = np.setdiff1d(np.arange(n), f)
        if len(tri) < 3:
            continue
        a1, y1 = xc_tr[tri], ytr[tri]
        yb1 = y1.mean(0)
        g1 = a1.T @ a1
        g1 = 0.5 * (g1 + g1.T)
        w1, v1 = np.linalg.eigh(g1)
        w1 = np.clip(w1, 0.0, None)
        lm1 = float(w1[-1]) if w1[-1] > 0 else 1.0
        path = _ridge_path(w1, v1, a1.T @ (y1 - yb1), None, penal_fracs, lm1)
        for pi in range(len(penal_fracs)):
            pred = xc_tr[f] @ path[pi] + yb1[None, :]
            scores[pi] += float(((pred - ytr[f]) ** 2).mean()) / len(folds)
    best = int(np.argmin(scores))
    frac = float(penal_fracs[best])
    coef = _ridge_path(w, v, xc_tr.T @ yc, None, np.array([frac]), lmax)[0]
    # hat-matrix diagonal at the chosen penalty, from the same eigendecomposition
    aw = (xc_tr @ v) / np.sqrt(w + frac * lmax)[None, :]
    hat = np.clip((aw ** 2).sum(1), 0.0, 1.0 - 1e-9)
    return {"pred": xc_te @ coef + ybar[None, :], "coef": coef, "frac": frac,
            "lmax": lmax, "pred_train": xc_tr @ coef + ybar[None, :],
            "hat_diag": hat, "cv_scores": scores.tolist(),
            "cv_score": float(scores[best]), "eig": (w, v), "ybar": ybar}


def ridge_predictive_var(ridge: dict, xc_te: np.ndarray, resid_var: np.ndarray) -> np.ndarray:
    """Classical ridge intervals: s2_o (1 + a^T (A^T A + lam lmax I)^-1 a).

    resid_var is the per-output residual variance with dof = trace(H), computed
    from TRAINING rows only.
    """
    w, v = ridge["eig"]
    frac, lmax = ridge["frac"], ridge["lmax"]
    ate_v = xc_te @ v
    lev = (ate_v ** 2 / (w + frac * lmax)[None, :]).sum(1)
    return np.asarray(resid_var)[None, :] * (1.0 + lev)[:, None]


def ridge_design(xtr: np.ndarray, xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centred (penalised-block) design matrices for train and test rows."""
    mx = xtr.mean(0)
    return xtr - mx, xte - mx


def resid_variance(y: np.ndarray, pred: np.ndarray, hat_diag: np.ndarray) -> np.ndarray:
    """Per-output residual variance with dof = max(trace(H), 1) (training rows only)."""
    rss = ((y - pred) ** 2).sum(0)
    dof = max(float(hat_diag.sum()), 1.0)
    return np.clip(rss / dof, 1e-24, None)


def effective_dof(ridge: dict, atr: np.ndarray) -> float:
    """trace(H) at the chosen penalty."""
    return float(ridge["hat_diag"].sum())


def zscore_nlpd(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> dict:
    """NLPD and empirical 68%/95% coverage of Gaussian predictive intervals."""
    var = np.clip(var, 1e-300, None)
    z = (y - mu) / np.sqrt(var)
    return {
        "nlpd": float(-np.mean(np.log(2.0 * np.pi * var) / 2.0 + z ** 2 / 2.0)),
        "c68": float(np.mean(np.abs(z) <= 1.0)),
        "c95": float(np.mean(np.abs(z) <= 1.96)),
        "z": z,
    }
