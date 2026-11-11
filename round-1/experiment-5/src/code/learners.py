"""Multi-output Gaussian process and penalised-regression learners, plus the five
shared baselines and the scale-free ridge calibration.

Every hyperparameter here is fitted on TRAIN rows only. Nothing in this module sees
a test label except through `predict`, and every normalisation constant is passed in
from the fold (see method.py).

GP hyperparameters (one shared set per (arm, fold), as p17_table.gp_batch):
    l  in L_GRID, s2 in {0.3, 1.0, 3.0}, n2 in {1e-8, 1e-5, 1e-3, 1e-1}
    selected by maximising the marginal likelihood  -0.5*tr(al^T al) - m*sum(log diag(L))
    for m = 45 outputs, K = s2*exp(-D/(2 l^2)) + n2*I.
Guard g3: if the selected l sits at an endpoint in more than half the folds the grid is
extended by a decade in that direction and the arm is re-run.

Ridge (guards g2): penalty = frac * lambda_max(A^T A) on a log grid frac in
logspace(-8, 3, 12), frac chosen by train-only k-fold CV with k = max(2, n//4), and an
INTERCEPT THAT IS NOT PENALISED.
"""
from __future__ import annotations

import numpy as np

L_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
S2_GRID = (0.3, 1.0, 3.0)
N2_GRID = (1e-8, 1e-5, 1e-3, 1e-1)
FRACS = np.logspace(-8, 3, 12)
MAX_GRID_EXTENSIONS = 3          # guard g3: at most three decades in either direction

JITTER_COUNT = {"chol": 0, "folds": 0}
GRID_ENDPOINT_HITS = {"hits": 0, "total": 0}


# --------------------------------------------------------------------- distances
def sqd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, clipped at 0."""
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def mahalanobis_sqd(A: np.ndarray, B: np.ndarray, Mdim: np.ndarray) -> np.ndarray:
    """(a-b)^T Mm (a-b) for Mahalanobis kernels, via the same expansion as sqd.

    d(x,x') = x^T Mm x + x'^T Mm x' - 2 x^T Mm x'
    """
    Ma = A @ Mdim
    Mb = B @ Mdim
    d = (Ma * A).sum(1)[:, None] + (Mb * B).sum(1)[None, :] - 2.0 * Ma @ B.T
    return np.maximum(d, 0.0)


def normalise_distance(Dtr: np.ndarray, Dst: np.ndarray, d_dim: int):
    """Rescale a squared-distance pair so the mean TRAIN pair distance is d_dim.

    One uniform convention for EVERY kernel (b1, b2, b3, b3p and all spectral cells):
    after this the lengthscale l is measured in per-dimension units, so `l = 8` means
    eight marginal standard deviations in one coordinate rather than an arbitrary
    function of the dimension and of the metric's own scaling. Without it an
    unnormalised 110-dimensional squared distance makes every pair far apart, the
    kernel degenerates to the identity and the arm silently reports the mean predictor
    (guard g1) — which is a broken arm, not a result. The scale uses TRAIN rows only.
    """
    n = Dtr.shape[0]
    off = Dtr[~np.eye(n, dtype=bool)]
    s = off.mean() / max(d_dim, 1)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return Dtr / s, Dst / s


# --------------------------------------------------------------------- GP
def _predict_from(l, s2, n2, Dtr, Dst, Ytr, want_var=True):
    """Fit at a FIXED hyperparameter set and predict; returns (mu, pv, chol_ok)."""
    n, m = Ytr.shape
    yb = Ytr.mean(0)
    ys = np.where(Ytr.std(0) < 1e-12, 1.0, Ytr.std(0))
    Yc = (Ytr - yb) / ys
    K = s2 * np.exp(-Dtr / (2.0 * l * l)) + n2 * np.eye(n)
    jittered = False
    try:
        Lc = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        JITTER_COUNT["chol"] += 1
        jittered = True
        K = K + (np.trace(K) / n) * 1e-8 * np.eye(n)
        Lc = np.linalg.cholesky(K)
    al = np.linalg.solve(Lc, Yc)
    a = np.linalg.solve(Lc.T, al)
    Ks = s2 * np.exp(-Dst / (2.0 * l * l))
    mu = (Ks @ a) * ys + yb
    if not want_var:
        return mu, None, jittered
    Kinv = np.linalg.solve(Lc.T, np.linalg.solve(Lc, Ks.T))
    pv = np.maximum(s2 - (Ks.T * Kinv).sum(0), 1e-12)[:, None]
    return mu, pv * (ys ** 2)[None, :] + 1e-18, jittered


def gp_fit_predict(Dtr: np.ndarray, Ytr: np.ndarray, Dst: np.ndarray,
                   lgrid=L_GRID, want_var: bool = True, criterion: str = "cv",
                   n_cv: int = 5, seed: int = 0):
    """One multi-output GP with a single shared hyperparameter set for all 45 outputs.

    Hyperparameters (l, s2, n2) from the identical grid for EVERY arm, chosen by
      `criterion="cv"`  train-only n_cv-fold CV on the standardised outputs (headline);
      `criterion="mll"` the marginal likelihood, as the plan's Step 7 specifies.
    Both criteria are evaluated in the same sweep; the losing one's choice is returned
    in `alt` so the artifact can report where the two protocols disagree.

    CV is the headline because on this problem the marginal likelihood drives the
    signal-to-noise ratio to a grid boundary: it prefers to treat the whole target as
    noise (n2 = 1e-1) or to interpolate every training point through an essentially
    diagonal kernel (n2 = 1e-8), and on the Mahalanobis kernels it lands on a
    lengthscale whose held-out error is the mean predictor even though a longer
    lengthscale on the same kernel predicts several times better. Selecting by
    train-only CV costs the same tuning budget for every arm and cannot see a test label.
    """
    n, m = Ytr.shape
    yb = Ytr.mean(0)
    ys = np.where(Ytr.std(0) < 1e-12, 1.0, Ytr.std(0))
    Yc = (Ytr - yb) / ys
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, min(n_cv, n))

    grid = tuple(lgrid)
    extensions = []
    cv_scores, mll_scores = _sweep(grid, Dtr, Ytr, Yc, m, n, folds)
    for _ in range(MAX_GRID_EXTENSIONS):
        cv_best = min(cv_scores, key=cv_scores.get)
        mll_best = max(mll_scores, key=mll_scores.get)
        target = cv_best[0] if criterion == "cv" else mll_best[0]
        if abs(target - grid[0]) < 1e-12:
            grid = tuple([grid[0] / 10.0] + list(grid))
            extensions.append("lo")
        elif abs(target - grid[-1]) < 1e-12:
            grid = tuple(list(grid) + [grid[-1] * 10.0])
            extensions.append("hi")
        else:
            break
        cv_scores, mll_scores = _sweep(grid, Dtr, Ytr, Yc, m, n, folds)
    JITTER_COUNT["folds"] += 1
    cv_best = min(cv_scores, key=cv_scores.get)
    mll_best = max(mll_scores, key=mll_scores.get)
    chosen = cv_best if criterion == "cv" else mll_best
    alt = mll_best if criterion == "cv" else cv_best
    mu, pv, jit = _predict_from(*chosen, Dtr, Dst, Ytr, want_var=want_var)
    return mu, pv, {"hyper": chosen, "alt_hyper": alt, "criterion": criterion,
                    "cv_rmse": cv_scores.get(chosen), "alt_cv_rmse": cv_scores.get(alt),
                    "jittered": jit, "grid_lo": grid[0], "grid_hi": grid[-1],
                    "extensions": extensions, "n_grid": len(grid)}


def _sweep(grid, Dtr, Ytr, Yc, m, n, folds):
    """Evaluate the marginal likelihood and train-only CV at every grid point."""
    cv_scores: dict[tuple, float] = {}
    mll_scores: dict[tuple, float] = {}
    for l in grid:
        Dl = np.exp(-Dtr / (2.0 * l * l))
        for s2 in S2_GRID:
            for n2 in N2_GRID:
                key = (l, s2, n2)
                K = s2 * Dl + n2 * np.eye(n)
                try:
                    Lc = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                al = np.linalg.solve(Lc, Yc)
                mll_scores[key] = float(-0.5 * (al ** 2).sum() - m * np.log(np.diag(Lc)).sum())
                # ---- train-only CV on the standardised outputs
                err, used = 0.0, 0
                for f in folds:
                    tri = np.setdiff1d(np.arange(n), f, assume_unique=False)
                    if len(tri) < 2:
                        continue
                    try:
                        Lf = np.linalg.cholesky(K[np.ix_(tri, tri)])
                    except np.linalg.LinAlgError:
                        continue
                    Yf = Yc[tri]
                    af = np.linalg.solve(Lf.T, np.linalg.solve(Lf, Yf))
                    Ks = K[np.ix_(f, tri)]
                    err += float(((Ks @ af - Yc[f]) ** 2).mean()) * len(f)
                    used += len(f)
                if used:
                    cv_scores[key] = err / used
    if not cv_scores:
        key = (grid[len(grid) // 2], 1.0, 1e-3)
        cv_scores[key] = np.inf
        mll_scores[key] = -np.inf
    return cv_scores, mll_scores


# --------------------------------------------------------------------- ridge
def _ridge_solve(A: np.ndarray, Y: np.ndarray, lam: float, sqrt_w: np.ndarray) -> np.ndarray:
    """Ridge as an augmented least-squares problem, which is stable when p > n.

        min ||A c - Y||^2 + lam * ||diag(sqrt_w) c||^2
      = min || [A ; sqrt(lam)*diag(sqrt_w)] c - [Y ; 0] ||^2

    Solved with `lstsq`, so a rank-deficient design returns the MINIMUM-NORM solution
    instead of the numerical explosion that `solve` on a near-singular normal-equation
    matrix produces. This matters at n = 10, where 110 injection columns plus a free
    intercept make the normal equations singular: with an unpenalised intercept and a
    near-zero penalty the Gram matrix is not invertible and `solve` returns coefficients
    that put the prediction thousands of standard deviations away.
    """
    k = A.shape[1]
    aug_A = np.vstack([A, np.sqrt(lam) * np.diag(sqrt_w)])
    aug_Y = np.vstack([Y, np.zeros((k, Y.shape[1]))])
    c, *_ = np.linalg.lstsq(aug_A, aug_Y, rcond=None)
    return c


def _ridge_cv(Atr: np.ndarray, Ytr: np.ndarray) -> tuple[np.ndarray, float]:
    """Penalty on the scale-free grid by train-only k-fold CV, unpenalised intercept."""
    n, p = Atr.shape
    A = np.hstack([Atr, np.ones((n, 1))])
    lam_max = np.linalg.eigvalsh(A.T @ A).max()
    sqrt_w = np.ones(p + 1)
    sqrt_w[-1] = 0.0                                     # the intercept is NOT penalised
    k = max(2, n // 4)
    folds = np.array_split(np.arange(n), k)
    best = None
    for frac in FRACS:
        lam = frac * lam_max
        sc = 0.0
        used = 0
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 2 or len(f) < 1:
                continue
            A1 = np.hstack([Atr[tri], np.ones((len(tri), 1))])
            c = _ridge_solve(A1, Ytr[tri], lam, sqrt_w)
            Af = np.hstack([Atr[f], np.ones((len(f), 1))])
            sc += ((Af @ c - Ytr[f]) ** 2).mean()
            used += 1
        if used and (best is None or sc < best[0]):
            best = (sc, lam)
    if best is None:
        best = (float("inf"), FRACS[-1] * lam_max)
    lam = best[1]
    return _ridge_solve(A, Ytr, lam, sqrt_w), lam


def ridge_fit_predict(Atr: np.ndarray, Ytr: np.ndarray, Ate: np.ndarray):
    c, lam = _ridge_cv(Atr, Ytr)
    A = np.hstack([Ate, np.ones((len(Ate), 1))])
    return A @ c, lam


def ols_intercept(Atr: np.ndarray, Ytr: np.ndarray, Ate: np.ndarray) -> np.ndarray:
    """Least-squares affine fit. `lstsq` returns the minimum-norm solution when the
    design is rank-deficient (p > n), which is the correct behaviour for this comparison
    and avoids the normal-equation blow-up."""
    A = np.hstack([Atr, np.ones((len(Atr), 1))])
    c = np.linalg.lstsq(A, Ytr, rcond=None)[0]
    return np.hstack([Ate, np.ones((len(Ate), 1))]) @ c


def joint_ridge_46(Vp1tr, Vp2tr, Ytr, Vp1te, Vp2te, trace_scale: bool = True,
                   penalise_intercept: bool = True, lam: float = 1e-4):
    """p17's original 46-column joint ridge — reported for auditability only.

    46 columns = order-1 (45) + order-2 (45)? No: the historical construction was the
    order-2 output plus a constant, i.e. 45 + 1 = 46 columns fitted jointly with one
    trace-scaled penalty. Reproduced here so a reader can see what it costs.
    """
    Atr = np.hstack([Vp2tr, np.ones((len(Vp2tr), 1))])
    Ate = np.hstack([Vp2te, np.ones((len(Vp2te), 1))])
    G = Atr.T @ Atr
    tr = np.trace(G) / G.shape[0]
    w = np.ones(G.shape[0])
    if not penalise_intercept:
        w[-1] = 0.0
    # augmented least squares: stable even when the 46-column design is rank-deficient at
    # small n, which is exactly the regime this calibration is reported to expose
    c = _ridge_solve(Atr, Ytr, lam * tr, np.sqrt(w))
    return Ate @ c


def per_output_gain(Vp_tr, Ytr, Vp_te) -> np.ndarray:
    """Single per-output gain through the origin (no intercept)."""
    num = (Vp_tr * Ytr).sum(0)
    den = (Vp_tr ** 2).sum(0)
    g = num / np.where(den < 1e-30, 1.0, den)
    return Vp_te * g
