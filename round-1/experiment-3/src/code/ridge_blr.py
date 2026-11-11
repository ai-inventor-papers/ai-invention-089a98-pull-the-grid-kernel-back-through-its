"""ridge_blr.py -- the single ridge / Bayesian-linear code path.

One implementation is used by every linear arm, every honesty control and every
LinDistFlow calibration route, so no comparison can be confounded by two different
linear solvers.

Conventions (all train-only by construction):
  * features are centred on TRAIN means (intercept exact, df counted on the centred design);
  * targets are standardised PER OUTPUT on TRAIN mean/sd before the shared penalty is
    selected -- so a shared penalty is not silently a different regularisation strength
    for each output -- and predictions are mapped back;
  * every penalty is a FRACTION of the largest eigenvalue of the centred design Gram
    (guard g2): never an absolute number, so the grid cannot blow up;
  * 'loo' is the exact leave-one-out criterion of a linear smoother, which is what makes
    n = 10 folds stable (the pilot's absolute-penalty K-fold route returned RMSE ~50).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

EPS = 1e-12
# The mandated scale-free grid: 25 penalties spanning [1e-6, 1e6] x lmax(Gram).
PENALTY_FRACS: tuple[float, ...] = tuple(float(x) for x in np.logspace(-6, 6, 25))
NOISE_MULTS: tuple[float, ...] = tuple(float(x) for x in np.logspace(-4, 2, 13))
GRID_MIN_FRAC, GRID_MAX_FRAC = 1e-6, 1e6
# Guard g2's actual invariant: the grid must be FRACTIONS of lmax, and must span at
# least the mandated range. A wider grid (used as a sensitivity check on the degree-2
# arm, whose spectrum is more collinear) is allowed; an absolute-magnitude grid is not.
GRID_MIN_ALLOWED, GRID_MAX_ALLOWED = 1e-16, 1e12
# A penalty whose maximum leverage exceeds this leaves a row that the fit interpolates:
# its leave-one-out residual is 0/0, so the criterion cannot rank it. Excluded, not used.
LEVERAGE_CAP = 0.95


def _check_grid(grid: Sequence[float]) -> None:
    """Guard g2: penalties are fractions of lmax spanning at least the mandated range."""
    g = np.asarray(grid, dtype=float)
    assert g.ndim == 1 and g.size >= 2, "guard g2: penalty grid must be a 1-D sequence"
    assert np.all(np.diff(g) > 0), "guard g2: penalty grid must be strictly increasing"
    assert g[0] > 0, "guard g2: penalties must be positive fractions, never zero"
    assert g[0] <= GRID_MIN_FRAC and g[-1] >= GRID_MAX_FRAC, (
        f"guard g2: grid must span at least [{GRID_MIN_FRAC:g}, {GRID_MAX_FRAC:g}] x lmax; "
        f"got [{g[0]:g}, {g[-1]:g}]")
    assert g[0] >= GRID_MIN_ALLOWED and g[-1] <= GRID_MAX_ALLOWED, (
        "guard g2: grid bounds are outside the admissible fraction window")


def _thin_svd(Ac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    if s.size == 0:                                            # pragma: no cover
        return U, s, Vt
    keep = s > s[0] * 1e-13
    return U[:, keep], s[keep], Vt[keep]


def _prep(Ftr: np.ndarray, Ytr: np.ndarray) -> dict[str, Any]:
    muF = Ftr.mean(0)
    Ac = Ftr - muF
    yb = Ytr.mean(0)
    ys = Ytr.std(0) + EPS
    Yc = (Ytr - yb) / ys
    U, s, Vt = _thin_svd(Ac)
    g = s ** 2
    return dict(muF=muF, Ac=Ac, yb=yb, ys=ys, Yc=Yc, U=U, s=s, Vt=Vt, g=g,
                lmax=float(g[0]), r=int(s.size), n=int(Ac.shape[0]), p=int(Ftr.shape[1]))


def fit_ridge(Ftr: np.ndarray, Ytr: np.ndarray, Fte: np.ndarray,
              mode: str = "gcv", inner_folds: int | None = None,
              grid: Sequence[float] = PENALTY_FRACS) -> dict[str, Any]:
    """Ridge with a scale-free penalty grid; penalty chosen train-only.

    mode 'gcv' : generalised cross-validation (Golub-Heath-Wahba), the standard criterion
                 for a linear smoother:  GCV(lam) = RSS / (1 - df/n)^2.
    mode 'loo' : exact leave-one-out for a linear smoother.  Identical in the well-posed
                 regime, but its denominator (1 - h_ii) -> 0 when the effective rank
                 approaches n, at which point it can prefer an interpolating fit on an
                 artificial zero residual. GCV degrades gracefully where LOO does not.
    mode 'cv'  : K-fold cross-validation over the TRAIN rows (K = max(2, n // 4)).
    """
    _check_grid(grid)

    d = _prep(Ftr, Ytr)
    Ac, Yc, U, s, Vt, g = d["Ac"], d["Yc"], d["U"], d["s"], d["Vt"], d["g"]
    lmax, n, p, r = d["lmax"], d["n"], d["p"], d["r"]
    W = U.T @ Yc                                               # (r, m)

    inner = None
    if mode == "cv":
        k = inner_folds if inner_folds is not None else max(2, n // 4)
        inner = []
        for part in np.array_split(np.arange(n), k):
            tri = np.setdiff1d(np.arange(n), part)
            if len(tri) < 2:
                continue
            U1, s1, Vt1 = _thin_svd(Ac[tri])
            inner.append((part, U1, s1, Vt1, s1 ** 2, U1.T @ Yc[tri], float((s1 ** 2)[0])))
        if not inner:                                          # pragma: no cover
            raise RuntimeError("fit_ridge: no usable inner fold")

    crit = np.empty(len(grid))
    dfs = np.empty(len(grid))
    inadmissible = np.zeros(len(grid), dtype=bool)
    for i, frac in enumerate(grid):
        lam = frac * lmax
        sh = g / (g + lam)
        df = float(sh.sum())
        dfs[i] = df
        if mode == "gcv":
            # GCV: RSS / (1 - df/n)^2.  The denominator is the same shrinkage that makes
            # an interpolating fit look good to LOO, applied to df rather than to each
            # leverage; it grows without bound as df -> n, so an interpolating penalty is
            # penalised instead of rewarded.
            fit = U @ (sh[:, None] * W)
            rss = float(np.mean((Yc - fit) ** 2))
            crit[i] = rss / max((1.0 - df / n) ** 2, 1e-12)
        elif mode == "loo":
            fit = U @ (sh[:, None] * W)
            h = (U ** 2) @ sh
            # LEVERAGE CAP.  A linear smoother with h_ii -> 1 has no defined
            # leave-one-out residual: its LOO error is 0/0 and it can never be
            # validated train-only.  This is exactly the p >= n interpolation trap that
            # produced the pilot's RMSE ~ 50 at small n.  Such a penalty is marked
            # inadmissible and excluded from selection rather than being allowed to win
            # on an artificial zero residual.
            if float(h.max()) > LEVERAGE_CAP:
                inadmissible[i] = True
                crit[i] = np.nan
                continue
            resid = (Yc - fit) / (1.0 - h)[:, None]
            crit[i] = float(np.mean(resid ** 2))
        elif mode == "cv":
            tot, used = 0.0, 0
            for part, U1, s1, Vt1, g1, W1, lm1 in inner:
                coef = Vt1.T @ ((s1 / (g1 + frac * lm1))[:, None] * W1)
                tot += float(np.mean((Yc[part] - Ac[part] @ coef) ** 2))
                used += 1
            crit[i] = tot / max(used, 1)
        else:                                                  # pragma: no cover
            raise ValueError(f"unknown ridge mode {mode!r}")

    admissible = ~np.isnan(crit)
    if not admissible.any():                                   # pragma: no cover
        # every penalty leaves a high-leverage row: fall back to the strongest penalty
        admissible = np.ones(len(grid), dtype=bool)
    masked = np.where(admissible, crit, np.inf)
    j = int(np.argmin(masked))
    frac = float(grid[j])
    lam = frac * lmax
    coef = Vt.T @ ((s / (g + lam))[:, None] * W)
    pred = ((Fte - d["muF"]) @ coef) * d["ys"][None, :] + d["yb"][None, :]
    endpoint = (j == 0) or (j == len(grid) - 1)
    return dict(pred=pred, frac=frac, df=float(dfs[j]), n_features=p, n=n,
                crit=float(crit[j]), mode=mode,
                n_inadmissible=int(inadmissible.sum()),
                cov_unreliable=bool(p > n / 2 and endpoint),
                at_grid_endpoint=bool(endpoint))


def fit_blr(Ftr: np.ndarray, Ytr: np.ndarray, Fte: np.ndarray,
            grid: Sequence[float] = PENALTY_FRACS,
            noise_mults: Sequence[float] = NOISE_MULTS,
            validate_dense: bool = False) -> dict[str, Any]:
    """Bayesian linear regression with evidence-maximised prior precision and noise variance.

    w ~ N(0, alpha^-1 I), eps ~ N(0, sigma2 I), alpha = frac * lmax.  Evidence is summed
    over the (standardised) outputs and maximised over the (frac, sigma2) product grid.
    The predictive variance it returns is what puts NLPD and 68/95 coverage on the same
    footing as the GP's -- without it the calibration comparison would be a point
    predictor against a probabilistic one.
    """
    _check_grid(grid)

    d = _prep(Ftr, Ytr)
    Ac, Yc, U, s, Vt, g = d["Ac"], d["Yc"], d["U"], d["s"], d["Vt"], d["g"]
    lmax, n, r = d["lmax"], d["n"], d["r"]
    W = U.T @ Yc
    yperp2 = float(np.sum((Yc - U @ W) ** 2))
    var_y = float(np.var(Yc) + EPS)

    m_out = int(Yc.shape[1])
    best = None
    for frac in grid:
        alpha = frac * lmax
        for smul in noise_mults:
            sigma2 = smul * var_y
            beta = 1.0 / sigma2
            bg = beta * g
            den = alpha + bg
            # Marginal log evidence, summed over the standardised outputs.
            # Per output:  log p(y) = -0.5 [ n log 2pi + log|Sigma| + y' Sigma^-1 y ]
            #   log|Sigma| = sum_i log(alpha + beta g_i) - r log(alpha) - n log(beta)   [r=rank]
            #   y'Sigma^-1 y = beta*||y_perp||^2 + beta*alpha*sum_i W_i^2/(alpha+beta g_i)
            # The log|Sigma| and n log 2pi blocks are per output (factor m_out); the
            # quadratic block is already summed over outputs through W and y_perp.
            logdet_term = float(np.log(den).sum()) - r * np.log(alpha) - n * np.log(beta)
            quad_total = (beta * alpha * float((W ** 2 / den[:, None]).sum())
                          + beta * yperp2)
            logev = -0.5 * (m_out * (n * np.log(2.0 * np.pi) + logdet_term) + quad_total)
            if best is None or logev > best[0]:
                best = (logev, float(frac), float(alpha), float(sigma2), float(smul))
    _, frac, alpha, sigma2, smul = best

    coef = Vt.T @ ((s / (g + alpha))[:, None] * W)
    Ate = Fte - d["muF"]
    pred = (Ate @ coef) * d["ys"][None, :] + d["yb"][None, :]
    quad = (((Ate @ Vt.T) ** 2) / (g[None, :] + alpha)).sum(1)
    pv_std = sigma2 * (1.0 + quad)                             # (m_test,)
    pv = pv_std[:, None] * (d["ys"] ** 2)[None, :]             # back to physical units
    df = float(np.sum(g / (g + alpha)))

    out: dict[str, Any] = dict(pred=pred, pv=np.maximum(pv, 1e-300), frac=frac, alpha=alpha,
                               sigma2=sigma2, sigma_mult=smul, df=df, n_features=d["p"],
                               n=n, log_evidence=float(best[0]))
    if validate_dense:
        # Independent n x n form on the TRAIN rows:  Sigma_y = sigma2 I + alpha^-1 Ac Ac^T
        cov = sigma2 * np.eye(n) + (1.0 / alpha) * (Ac @ Ac.T)
        _, logdet = np.linalg.slogdet(cov)
        sol = np.linalg.solve(cov, Yc)
        logev_dense = -0.5 * (Yc.shape[1] * (n * np.log(2 * np.pi) + logdet)
                              + float(np.sum(Yc * sol)))
        out["log_evidence_dense"] = float(logev_dense)
        out["eigen_vs_dense_abs_diff"] = float(abs(logev_dense - out["log_evidence"]))
        out["eigen_vs_dense_rel_diff"] = float(
            abs(logev_dense - out["log_evidence"]) / max(abs(logev_dense), 1e-30))
    return out


def fit_affine(Ftr: np.ndarray, Ytr: np.ndarray, Fte: np.ndarray) -> dict[str, Any]:
    """Unpenalised affine map (least squares). Weakest calibrator; the family's floor."""
    muF, yb = Ftr.mean(0), Ytr.mean(0)
    ys = Ytr.std(0) + EPS
    Ac = Ftr - muF
    Yc = (Ytr - yb) / ys
    coef = np.linalg.lstsq(Ac, Yc, rcond=None)[0]
    pred = ((Fte - muF) @ coef) * ys[None, :] + yb[None, :]
    rank = int(np.linalg.matrix_rank(Ac))
    return dict(pred=pred, frac=0.0, df=float(rank),
                crit=float(np.mean((Yc - Ac @ coef) ** 2)))


def fit_diag_ridge(Ftr: np.ndarray, Ytr: np.ndarray, Fte: np.ndarray,
                   grid: Sequence[float] = PENALTY_FRACS) -> dict[str, Any]:
    """Per-output scalar affine recalibration (diagonal map), penalty by exact LOO.

    This is the classic per-output linear recalibration of a physics surrogate and is
    included so the physics baseline is not handicapped by being forced to share a
    coefficient across 45 outputs.
    """
    nm, m = Ftr.shape
    pred = np.empty((Fte.shape[0], m))
    fracs = np.empty(m)
    dfs = np.empty(m)
    crit = np.empty(m)
    for j in range(m):
        x = Ftr[:, j]
        mx = x.mean()
        xc = x - mx
        yb = Ytr[:, j].mean()
        yc = (Ytr[:, j] - yb) / (Ytr[:, j].std() + EPS)
        U, s, Vt = _thin_svd(xc[:, None])
        g = s ** 2
        if g.size == 0:                                        # pragma: no cover
            pred[:, j] = Ytr[:, j].mean()
            fracs[j], dfs[j], crit[j] = np.nan, 0.0, np.nan
            continue
        lmax = float(g[0])
        w = (U.T @ yc).reshape(-1)[0]
        best = None
        for frac in grid:
            lam = frac * lmax
            sh = g / (g + lam)
            fit = U @ (sh * w)
            h = (U ** 2) @ sh
            if float(h.max()) > LEVERAGE_CAP:
                continue
            c = float(np.mean(((yc - fit) / (1.0 - h)) ** 2))
            if best is None or c < best[0]:
                best = (c, frac, lam, sh, w)
        if best is None:                                       # pragma: no cover
            best = (np.nan, float(grid[-1]), float(grid[-1]) * lmax,
                    g / (g + grid[-1] * lmax), w)
        c, frac, lam, sh, w = best
        coef = Vt.T @ (sh * w)
        pred[:, j] = ((Fte[:, j] - mx) @ coef) * (Ytr[:, j].std() + EPS) + yb
        fracs[j] = frac
        dfs[j] = float(sh.sum())
        crit[j] = c
    return dict(pred=pred, frac=float(np.median(fracs)), df=float(dfs.sum()),
                crit=float(np.mean(crit)))
