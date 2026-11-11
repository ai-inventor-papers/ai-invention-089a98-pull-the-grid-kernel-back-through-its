"""The five shared baselines, computed once per fold and shared by every candidate.

b1 isotropic RBF GP on per-column-standardised injections, squared distance divided
   by d = 110 so the lengthscale is O(1) in per-dimension units (guard g1).
b2 hop-count graph-Laplacian kernel GP: M_ij = hop(bi) + hop(bj) - 2 hop(lca(bi,bj))
   over the 55 load buses, Mahalanobis-RBF on the 55 load magnitudes.
b3 admittance-weighted multiphase Kirchhoff metric at the FULL 110 dimensions,
   built from LINE DATA ONLY (each load's own-phase column of the load-by-load block
   Zbus by the shared-path rule, never from any Jacobian); also b3p, the 55-dim P-only
   version, because a prior hypothesis compared a 55-dim competitor to a 110-dim
   treatment.
b4 ridge on standardised injections, unpenalised intercept, scale-free penalty grid.
b5 multiphase LinDistFlow with the exact second-order Z-bus current correction,
   calibrated per output by train-only ridge CV with a free unpenalised intercept and
   the expansion order (1 or 2) chosen by inner train-only CV. Four further
   calibrations of the same LinDistFlow outputs are reported for auditability.

NON-CIRCULARITY. b1, b2, b4 and b5 are constructible without the candidate's Laplacian
or the treatment operator. b3 IS derived from the same Laplacian the candidate's
spectrum comes from and is therefore not an independent baseline; it is included
because it is the request-named phase-aware graph kernel and because the metric form
versus the spectral form is this artifact's substantive question.
"""
from __future__ import annotations

import numpy as np

from learners import (_ridge_solve, gp_fit_predict, mahalanobis_sqd, normalise_distance,
                      ols_intercept, per_output_gain, joint_ridge_46, ridge_fit_predict, sqd)

ANGS = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])


# --------------------------------------------------------------------- metric
def admittance_metric(Zll_own: np.ndarray, phase: np.ndarray, Sb: float) -> np.ndarray:
    """b3's 110-dim Mahalanobis metric from the 55x55 own-phase block Zbus.

    Kc = -Zcol / (Sb * conj(V0l)) is the first-order current-to-voltage map restricted
    to each load's own phase; K110 = [Kc, 1j*Kc] stacks the P and Q actions.
    M110 = Re(K110^H K110), symmetrised and clipped to PSD.
    """
    V0l = 1.05 * np.exp(1j * ANGS)[phase]
    Kc = -Zll_own / (Sb * np.conj(V0l))[None, :]
    K110 = np.hstack([Kc, 1j * Kc])
    M = (K110.conj().T @ K110).real
    M = 0.5 * (M + M.T)
    ev, U = np.linalg.eigh(M)
    ev = np.clip(ev, 0.0, None)
    return (U * ev) @ U.T


def hop_metric(Dl_loads: np.ndarray, hop: dict, lbus: np.ndarray) -> np.ndarray:
    """b2: hop-count graph Laplacian over the 55 load buses (tree metric)."""
    return Dl_loads.astype(float)


# --------------------------------------------------------------------- standardisation
def standardise(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    clip = int((sd < 1e-12).sum())
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, clip


def radius_normalise(Atr: np.ndarray, Ate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt((Atr ** 2).sum(1).mean())
    r = r if r > 1e-30 else 1.0
    return Atr / r, Ate / r


# --------------------------------------------------------------------- b1
def b1_rbf(Xtr, Ytr, Xte, lgrid=None):
    Ztr, Zte, clip = standardise(Xtr, Xte)
    kwargs = {} if lgrid is None else {"lgrid": lgrid}
    Dtr, Dst = normalise_distance(sqd(Ztr, Ztr), sqd(Zte, Ztr), Xtr.shape[1])
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst, **kwargs)
    return mu, pv, hp, clip


# --------------------------------------------------------------------- b2
def b2_hop(Xtr, Ytr, Xte, Mhop):
    Ptr, Pte = Xtr[:, :55], Xte[:, :55]
    Ztr, Zte, clip = standardise(Ptr, Pte)
    Dtr, Dst = normalise_distance(mahalanobis_sqd(Ztr, Ztr, Mhop),
                                  mahalanobis_sqd(Zte, Ztr, Mhop), 55)
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst)
    return mu, pv, hp, clip


# --------------------------------------------------------------------- b3
def b3_kirchhoff(Xtr, Ytr, Xte, M110):
    Ztr, Zte, clip = standardise(Xtr, Xte)
    Dtr, Dst = normalise_distance(mahalanobis_sqd(Ztr, Ztr, M110),
                                  mahalanobis_sqd(Zte, Ztr, M110), 110)
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst)
    return mu, pv, hp, clip


def b3p_kirchhoff_p(Xtr, Ytr, Xte, M55):
    Ptr, Pte = Xtr[:, :55], Xte[:, :55]
    Ztr, Zte, clip = standardise(Ptr, Pte)
    Dtr, Dst = normalise_distance(mahalanobis_sqd(Ztr, Ztr, M55),
                                  mahalanobis_sqd(Zte, Ztr, M55), 55)
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst)
    return mu, pv, hp, clip


# --------------------------------------------------------------------- b6 (control)
def b6_pca_gp(Xtr, Ytr, Xte, k: int):
    """THE DOSE-MATCHED CONTROL: GP on the top-k principal components of the
    standardised injections.

    It receives exactly the same reduction in dimension (2k coordinates, the same count
    the spectral cells get from k eigenpairs) and the same learner, but the directions
    are chosen from the injection covariance with no network information at all. If the
    spectral cell's margin survives this, the margin comes from WHICH directions the
    feeder's Laplacian selects; if it does not, the margin was generic compression and
    must be reported as such. This is the first control a reviewer would name.
    """
    Ztr, Zte, clip = standardise(Xtr, Xte)
    U, s, Vt = np.linalg.svd(Ztr, full_matrices=False)
    B = Vt[:k].T
    Dtr, Dst = normalise_distance(sqd(Ztr @ B, Ztr @ B), sqd(Zte @ B, Ztr @ B), 2 * k)
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst)
    return mu, pv, hp, clip


# --------------------------------------------------------------------- b4
def b4_ridge(Xtr, Ytr, Xte, ncomp: int = 0):
    Ztr, Zte, clip = standardise(Xtr, Xte)
    if ncomp:                                     # truncated-SVD components, if requested
        U, s, Vt = np.linalg.svd(Ztr, full_matrices=False)
        B = Vt[:ncomp].T
        mu, lam = ridge_fit_predict(Ztr @ B, Ytr, Zte @ B)
    else:
        mu, lam = ridge_fit_predict(Ztr, Ytr, Zte)
    return mu, None, ("ridge", lam), clip


# --------------------------------------------------------------------- b5
def b5_lindist(Vp1tr, Vp2tr, Ytr, Vp1te, Vp2te):
    """The tuned physics baseline: order chosen by inner train-only CV, per-output ridge."""
    n = len(Ytr)
    k = max(2, n // 4)
    folds = np.array_split(np.arange(n), k)
    scores = {}
    for order, (A, B) in {1: (Vp1tr, Vp1te), 2: (Vp2tr, Vp2te)}.items():
        sc, used = 0.0, 0
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 3:
                continue
            pr, _ = ridge_fit_predict(A[tri], Ytr[tri], A[f])
            sc += ((pr - Ytr[f]) ** 2).mean()
            used += 1
        scores[order] = sc / max(used, 1)
    order = min(scores, key=scores.get)
    A, B = (Vp1tr, Vp1te) if order == 1 else (Vp2tr, Vp2te)
    mu, lam = ridge_fit_predict(A, Ytr, B)
    return mu, None, ("lindist", order, float(lam), scores), order


def b5_calibrations(Vp1tr, Vp2tr, Ytr, Vp1te, Vp2te, order: int) -> dict[str, np.ndarray]:
    """The five LinDistFlow calibrations, reported per fold for auditability.

    'p17_joint46_trace'  : p17's 46-column joint trace-scaled ridge (penalises intercept)
    'joint46_ols'        : plain OLS on the same 46 columns
    'joint46_ridge_free' : 46-column ridge with the intercept unpenalised
    'per_output_ols'     : per-output affine OLS
    'per_output_gain'    : single per-output gain through the origin
    'per_output_ridge_cv': the one used as b5 (per-output ridge, train-only CV)
    """
    A, B = (Vp1tr, Vp1te) if order == 1 else (Vp2tr, Vp2te)
    out = {}
    out["p17_joint46_trace"] = joint_ridge_46(Vp1tr, Vp2tr, Ytr, Vp1te, Vp2te,
                                              trace_scale=True, penalise_intercept=True)
    out["joint46_ols"] = joint_ridge_46(Vp1tr, Vp2tr, Ytr, Vp1te, Vp2te,
                                        trace_scale=False, penalise_intercept=False,
                                        lam=0.0)
    # 46-column ridge with a free intercept: solve the KKT system directly
    Acol = np.hstack([B, np.ones((len(B), 1))])
    At = np.hstack([A, np.ones((len(A), 1))])
    G = At.T @ At
    ann = np.ones(G.shape[0])
    ann[-1] = 0.0
    lam = 1e-4 * np.trace(G) / G.shape[0]
    # augmented least squares via _ridge_solve: the 46-column design is rank-deficient at
    # n = 10, where a normal-equation solve would return a numerically meaningless fit
    out["joint46_ridge_free"] = Acol @ _ridge_solve(At, Ytr, lam, np.sqrt(ann))
    out["per_output_ols"] = ols_intercept(A, Ytr, B)
    out["per_output_gain"] = per_output_gain(A, Ytr, B)
    mu, _ = ridge_fit_predict(A, Ytr, B)
    out["per_output_ridge_cv"] = mu
    return out
