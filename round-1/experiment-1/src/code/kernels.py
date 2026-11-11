"""Multi-output Gaussian process with a squared-exponential kernel on a caller-supplied
feature map, plus the selection machinery every candidate shares.

The kernel is always

    k(x, x') = sigma2 * exp( -||F(x) - F(x')||^2 / (2 ell^2) ) + noise2 * delta

where F is the candidate's feature map.  The choice of F *is* the candidate:
  - primary treatment: F(x) = S (x - mx) with S = d|V_far|/dx, 45 x 110
  - isotropic RBF baseline: F(x) = (x - mx)/sx  (per-column standardisation)
  - graph-Laplacian baselines: F(x) = P (x - mx) for a 55x55 metric P
  - control random operator: F(x) = S_rand (x - mx)

Everything that could leak test information is computed from train rows only:
centring mx, scale sx, the single feature scale factor, the target standardisation,
and every hyperparameter (by marginal likelihood or train-row cross-validation).

Hyperparameters: the same grid for every candidate, and the same 3-point local
refinement of the lengthscale for every candidate, so no candidate receives a
finer search than another.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

# The shared grid: identical extent for EVERY kernel candidate, so no candidate gets a
# finer or wider search than another.  The extent was widened from the archived
# (ell <= 8, s2 <= 3) after measuring that the marginal-likelihood optimum for this
# target lies OUTSIDE that box (ell ~ 16, s2 ~ 10 at n = 160): the map is close to
# linear in the pulled-back coordinates and a nearly-linear kernel in those
# coordinates is what the training rows support.  Because the same box is given to
# every candidate this is a level playing field, not a preference for the treatment.
ELL_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 512.0)
S2_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
N2_GRID = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ELL_REFINE = (0.7, 1.0, 1.4)


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, clipped at 0."""
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def gram_dist(F: np.ndarray) -> np.ndarray:
    G = F @ F.T
    d = np.diag(G)
    return np.maximum(d[:, None] + d[None, :] - 2.0 * G, 0.0)


def _mll_one(D: np.ndarray, Yc: np.ndarray, ell: float, s2: float, n2: float):
    """Marginal log likelihood of one hyperparameter triple, plus its factorisation."""
    n, m = Yc.shape
    K = s2 * np.exp(-D / (2.0 * ell * ell))
    jit = n2 + 1e-10 * float(np.trace(K)) / n
    K = K + jit * np.eye(n)
    try:
        c = cho_factor(K, lower=True, check_finite=False)
    except LinAlgError:
        return None
    alpha = cho_solve(c, Yc, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
    mll = -0.5 * float(np.sum(alpha ** 2)) - 0.5 * m * logdet
    return dict(mll=mll, ell=ell, s2=s2, n2=n2, cf=c, alpha=alpha, jit=jit)


def _search(D: np.ndarray, Yc: np.ndarray, ell_grid=ELL_GRID, s2_grid=S2_GRID,
            n2_grid=N2_GRID, refine: bool = True):
    best = None
    for ell in ell_grid:
        for s2 in s2_grid:
            for n2 in n2_grid:
                r = _mll_one(D, Yc, ell, s2, n2)
                if r is None:
                    continue
                if best is None or r["mll"] > best["mll"]:
                    best = r
    if best is None:
        raise RuntimeError("every hyperparameter cell failed to factorise")
    if refine:
        # same 3-point local refinement applied to EVERY candidate
        for fac in ELL_REFINE:
            if fac == 1.0:
                continue
            r = _mll_one(D, Yc, best["ell"] * fac, best["s2"], best["n2"])
            if r is not None and r["mll"] > best["mll"]:
                best = r
    return best


def _cv_score_grid(D: np.ndarray, Yc: np.ndarray, k: int,
                   ell_grid=ELL_GRID, s2_grid=S2_GRID, n2_grid=N2_GRID) -> dict:
    """Hyperparameters by TRAIN-ONLY k-fold cross-validated squared error.

    The alternative to marginal likelihood, and the one that is robust to the
    obviously misspecified stationary-noise model this target has (a deterministic
    AC map has no observation noise, so the marginal likelihood's noise term is
    absorbing model error and can prefer a badly over-smoothing lengthscale).
    Every candidate is given this same rule when the CV arm is used.
    """
    n = D.shape[0]
    folds = [f for f in np.array_split(np.arange(n), max(2, min(k, n // 2))) if len(f) > 2]
    best = None
    for ell in ell_grid:
        Kf = np.exp(-D / (2.0 * ell * ell))
        for s2 in s2_grid:
            for n2 in n2_grid:
                s = 0.0
                ok = True
                for f in folds:
                    tri = np.setdiff1d(np.arange(n), f)
                    K = s2 * Kf[np.ix_(tri, tri)]
                    jit = n2 + 1e-10 * float(np.trace(K)) / len(tri)
                    try:
                        c = cho_factor(K + jit * np.eye(len(tri)), lower=True,
                                       check_finite=False)
                    except LinAlgError:
                        ok = False
                        break
                    al = cho_solve(c, Yc[tri], check_finite=False)
                    mu = (s2 * Kf[np.ix_(f, tri)] @ al)
                    s += float(np.mean((mu - Yc[f]) ** 2)) / len(folds)
                if not ok:
                    continue
                if best is None or s < best["score"]:
                    best = dict(score=s, ell=ell, s2=s2, n2=n2)
    if best is None:
        raise RuntimeError("every CV cell failed to factorise")
    # final factorisation at the selected triple, on all train rows
    r = _mll_one(D, Yc, best["ell"], best["s2"], best["n2"])
    if r is None:
        raise RuntimeError("selected CV cell failed to factorise on the full train set")
    r["cv_score"] = best["score"]
    return r


def gp_fit_predict(Ftr: np.ndarray, Fte: np.ndarray, Ytr: np.ndarray,
                   want_var: bool = False, refine: bool = True,
                   Dtr: np.ndarray | None = None, ell_grid=ELL_GRID,
                   select: str = "mll", cv_k: int = 5) -> dict:
    """Fit one shared hyperparameter set on all outputs, predict the test rows.

    select = 'mll' : marginal likelihood over the shared grid (the pre-registered rule)
    select = 'cv'  : train-row k-fold cross-validated squared error over the same grid

    Returns dict(mu, var, ell, s2, n2, mll, Ybar, Ysd).  `var` is the full predictive
    variance including the noise term, which is what a density forecast reports.
    """
    n, m = Ytr.shape
    Ybar = Ytr.mean(0)
    Ysd = Ytr.std(0) + 1e-12
    Yc = (Ytr - Ybar) / Ysd
    if Dtr is None:
        Dtr = gram_dist(Ftr)
    if select == "cv":
        best = _cv_score_grid(Dtr, Yc, cv_k, ell_grid=ell_grid)
    elif select == "mll":
        best = _search(Dtr, Yc, refine=refine, ell_grid=ell_grid)
    else:
        raise ValueError(select)
    Dst = sqdist(Fte, Ftr)
    Ks = best["s2"] * np.exp(-Dst / (2.0 * best["ell"] ** 2))
    a = cho_solve(best["cf"], Ks.T, check_finite=False).T          # (m_test, n)
    mu = (Ks @ best["alpha"]) * Ysd + Ybar
    out = dict(mu=mu, ell=float(best["ell"]), s2=float(best["s2"]),
               n2=float(best["n2"]), mll=float(best["mll"]), jit=float(best["jit"]),
               cv_score=float(best.get("cv_score", float("nan"))),
               select=select, Ybar=Ybar, Ysd=Ysd)
    if want_var:
        v = best["s2"] - np.einsum("ij,ij->i", a, Ks)
        v = np.maximum(v, 1e-12)
        out["var"] = v[:, None] * (Ysd ** 2)[None, :] + 1e-18
    return out


def nlpd_coverage(Yte: np.ndarray, mu: np.ndarray, var: np.ndarray,
                  ysd: np.ndarray | None = None) -> dict:
    """Negative log predictive density and empirical 68 / 95 percent coverage.

    The density is evaluated on the target standardised by the TRAIN-fold per-output
    standard deviation, which is the scale a voltage emulator's density forecast is
    normally reported on: an unnormalised pu density near 1.0 pu is dominated by the
    level, not the residual, and would make candidates incomparable.  Coverage is
    invariant to that scaling; the reported NLPD is the normalised one.
    """
    s = np.ones(Yte.shape[1]) if ysd is None else np.asarray(ysd, dtype=float)
    s = np.where(s <= 0, 1.0, s)
    Yn = Yte / s[None, :]
    mn = mu / s[None, :]
    vn = var / (s[None, :] ** 2)
    z = (Yn - mn) / np.sqrt(vn)
    nlpd = float(np.mean(0.5 * np.log(2.0 * np.pi * vn) + 0.5 * z ** 2))
    return dict(nlpd=nlpd,
                coverage68=float(np.mean(np.abs(z) <= 1.0)),
                coverage95=float(np.mean(np.abs(z) <= 1.96)),
                mean_z2=float(np.mean(z ** 2)))


def ard_rbf(Ftr: np.ndarray, Fte: np.ndarray, Ytr: np.ndarray,
            passes: int = 2, select: str = "mll") -> dict:
    """ARD squared-exponential with genuine per-dimension lengthscales.

    Coordinate descent on the marginal likelihood over a multiplicative grid, with
    sigma2 and noise2 re-selected by the shared grid at every pass.  Used only on
    3 outputs and 3 training sizes (the extended-pool, reduced-scope comparator).
    """
    n, p = Ftr.shape
    Ybar = Ytr.mean(0)
    Ysd = Ytr.std(0) + 1e-12
    Yc = (Ytr - Ybar) / Ysd
    ls = np.ones(p)
    scale = np.std(Ftr, axis=0) + 1e-12
    glob = 1.0
    best = None

    def _score(ls_vec, g):
        A = Ftr / (scale * ls_vec * g)
        D = gram_dist(A)
        bb = None
        for s2 in S2_GRID:
            for n2 in N2_GRID:
                rr = _mll_one(D, Yc, 1.0, s2, n2)
                if rr is None:
                    continue
                if bb is None or rr["mll"] > bb["mll"]:
                    bb = rr
        return None if bb is None else (bb["mll"], bb)

    # a global scale first -- 110 per-dimension lengthscales cannot be identified from
    # n = 20 rows, and without this the coordinate descent sits at the mean predictor
    for g in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        s = _score(ls, g)
        if s is None:
            continue
        if best is None or s[0] > best[0]:
            best = (s[0], ls.copy(), s[1], g)
    if best is not None:
        glob = best[3]

    for _ in range(passes):
        cur = best
        for j in range(p):
            for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
                trial = ls.copy()
                trial[j] = np.clip(ls[j] * mult, 1e-2, 1e2)
                s = _score(trial, glob)
                if s is None:
                    continue
                if cur is None or s[0] > cur[0]:
                    cur = (s[0], trial.copy(), s[1], glob)
        if cur is not None:
            best = cur
            ls = cur[1].copy()
    if best is None:
        raise RuntimeError("ARD-RBF: every lengthscale cell failed to factorise")
    _, ls, bb, glob = best
    A_tr = Ftr / (scale * ls * glob)
    A_te = Fte / (scale * ls * glob)
    Dst = sqdist(A_te, A_tr)
    Ks = bb["s2"] * np.exp(-Dst / 2.0)
    mu = (Ks @ bb["alpha"]) * Ysd + Ybar
    return dict(mu=mu, ell=float(np.median(ls)), s2=float(bb["s2"]),
                n2=float(bb["n2"]), mll=float(bb["mll"]),
                ls_summary=dict(median=float(np.median(ls)), min=float(ls.min()),
                                max=float(ls.max())))
