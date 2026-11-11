"""protocol.py -- the frozen fold recipe, guards, metrics, GP machinery, paired inference.

Everything here is deterministic numpy/scipy linear algebra. No LLM client is imported
anywhere in this artifact (openrouter_calls = 0).

Guards enforced here:
  * g1  any fold-level RMSE worse than that fold's mean-predictor RMSE is flagged
        numerical_failure and never quoted as a result;
  * g2  every ridge/BLR penalty is a FRACTION of the largest eigenvalue of the centred
        design Gram (see ridge_blr.py), never an absolute number;
  * every normaliser/statistic comes from TRAIN rows only (see arms.py: only X[tr],
        Y[tr] are ever passed into a fitting function).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
from scipy import stats

EPS = 1e-12

# ---------------------------------------------------------------------------
# the frozen fold recipe (contract-exact; identical in all five screen artifacts)
# ---------------------------------------------------------------------------
FOLD_SIZES: tuple[int, ...] = (10, 20, 40, 80, 160)      # 60 and 120 are RESERVED
FOLD_SEEDS: tuple[int, ...] = tuple(range(12))            # 12..29 are RESERVED
FOLD_SCHEMES: tuple[int, ...] = (0, 1)
FOLD_BASE = 20261111
FOLD_STEP_SEED = 7919
FOLD_STEP_SCHEME = 131
N_ENSEMBLE = 220
SCHEME1_TRAIN_BLOCK = 110


def fold_indices(n: int, seed: int, scheme: int) -> tuple[np.ndarray, np.ndarray] | None:
    """One partition. Returns (train_idx, test_idx), or None when the cell is skipped.

    scheme 0: train = permutation[:n],        test = permutation[n:]
    scheme 1: train = permutation[:110][:n],  test = permutation[110:]
              -> at n = 160 only 12 folds exist, because the train block is 110 rows.
    A cell that cannot be formed is SKIPPED, never substituted.
    """
    rng = np.random.default_rng(FOLD_BASE + FOLD_STEP_SEED * seed + FOLD_STEP_SCHEME * scheme)
    idx = rng.permutation(N_ENSEMBLE)
    if scheme == 0:
        tr, te = idx[:n], idx[n:]
    elif scheme == 1:
        tr, te = idx[:SCHEME1_TRAIN_BLOCK][:n], idx[SCHEME1_TRAIN_BLOCK:]
    else:                                                  # pragma: no cover
        raise ValueError(f"unknown scheme {scheme}")
    if len(tr) < n or len(te) < 3:
        return None
    return np.sort(tr), np.sort(te)


def all_jobs() -> list[tuple[int, int, int]]:
    return [(n, s, c) for n in FOLD_SIZES for s in FOLD_SEEDS for c in FOLD_SCHEMES]


def realised_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for n in FOLD_SIZES:
        out[str(n)] = sum(1 for s in FOLD_SEEDS for c in FOLD_SCHEMES
                          if fold_indices(n, s, c) is not None)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rmse_raw(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def normalisers(Ytr: np.ndarray, Yall: np.ndarray) -> tuple[float, float]:
    """Primary denominator: mean per-output sd on the TRAIN rows (leakage-free).
    Secondary denominator: the same statistic over the whole ensemble (comparability with
    the run's earlier p17 table, which used it). Both recorded per fold.
    """
    return float(Ytr.std(0).mean()), float(Yall.std(0).mean())


def mean_predictor_rmse(Ytr: np.ndarray, Yte: np.ndarray, vsd_train: float) -> float:
    pred = np.repeat(Ytr.mean(0)[None, :], Yte.shape[0], axis=0)
    return rmse_raw(pred, Yte) / (vsd_train + EPS)


def predictive_scores(mu: np.ndarray, pv: np.ndarray, Yte: np.ndarray) -> dict[str, float]:
    pv = np.maximum(pv, 1e-300)
    z = (Yte - mu) / np.sqrt(pv)
    return dict(nlpd=float(np.mean(0.5 * np.log(2 * np.pi * pv) + 0.5 * z ** 2)),
                coverage68=float(np.mean(np.abs(z) <= 1.0)),
                coverage95=float(np.mean(np.abs(z) <= 1.96)))


# ---------------------------------------------------------------------------
# Gaussian-process machinery (the run's p17 harness, unchanged)
# ---------------------------------------------------------------------------
GP_LGRID: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GP_S2GRID: tuple[float, ...] = (0.3, 1.0, 3.0)
GP_N2GRID: tuple[float, ...] = (1e-8, 1e-5, 1e-3, 1e-1)


def sqd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def gram(F: np.ndarray) -> np.ndarray:
    return sqd(F, F)


def gp_batch(Dtr: np.ndarray, Ytr: np.ndarray, Dst: np.ndarray,
             lgrid: Sequence[float] = GP_LGRID) -> dict[str, Any]:
    """Multi-output GP: ONE hyperparameter set shared by all 45 outputs, chosen by
    marginal likelihood on the TRAIN block, per-output train standardisation.

    Kernel: k = s2 * exp(-||f_i - f_j||^2 / (2 l^2)) + n2 * delta.
    df (effective degrees of freedom) = trace(K (K + n2 I)^{-1}) on the TRAIN Gram.
    """
    n, m = Ytr.shape
    yb = Ytr.mean(0)
    ys = Ytr.std(0) + EPS
    Yc = (Ytr - yb) / ys
    best = None
    for ell in lgrid:
        Dl = np.exp(-Dtr / (2.0 * ell * ell))
        for s2 in GP_S2GRID:
            for n2 in GP_N2GRID:
                K = s2 * Dl + n2 * np.eye(n)
                try:
                    Lc = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                al = np.linalg.solve(Lc, Yc)
                mll = -0.5 * float((al ** 2).sum()) - m * float(np.log(np.diag(Lc)).sum())
                if best is None or mll > best[0]:
                    best = (mll, ell, s2, n2, Lc, al, K)
    if best is None:                                        # pragma: no cover
        raise RuntimeError("gp_batch: no admissible hyperparameter set")
    _, ell, s2, n2, Lc, al, K = best
    a = np.linalg.solve(Lc.T, al)
    Ks = s2 * np.exp(-Dst / (2.0 * ell * ell))
    mu = (Ks @ a) * ys + yb
    Kinv_qt = np.linalg.solve(Lc.T, np.linalg.solve(Lc, Ks.T))
    pv = np.maximum(s2 - (Ks.T * Kinv_qt).sum(0), 1e-12)[:, None] * (ys ** 2)[None, :] + 1e-18

    lam = np.maximum(np.linalg.eigvalsh(0.5 * (K + K.T)), 0.0)
    df_gp = float(np.sum(lam / (lam + n2)))

    return dict(mu=mu, pv=pv, lengthscale=float(ell), signal_var=float(s2),
                noise_var=float(n2), mll=float(best[0]), df=df_gp,
                hyper=f"l={ell:g},s2={s2:g},n2={n2:g}")


# ---------------------------------------------------------------------------
# paired inference (the decisive quantity)
# ---------------------------------------------------------------------------
def paired_inference(d_abs: np.ndarray, d_rel: np.ndarray, n_boot: int = 10000,
                     seed: int = 7) -> dict[str, Any]:
    """Fold-level paired comparison of a linear arm against the GP arm.

    d_abs = rmse_linear - rmse_gp (negative => the linear arm is better)
    d_rel = rmse_linear / rmse_gp - 1
    """
    d_abs = np.asarray(d_abs, dtype=float)
    d_rel = np.asarray(d_rel, dtype=float)
    n = int(d_abs.size)

    # EXACT TIES ARE DROPPED from the sign test, which is the standard convention for a
    # paired sign test (and what `wilcoxon(zero_method="wilcox")` does). Counting a tie as
    # "not a linear win" would let a test with many ties look decisively significant while
    # carrying almost no information: with all 24 folds tied, binomtest(0, 24) returns
    # p = 2^-24, which would report a perfect null as a decisive result. Ties are reported
    # separately and excluded from the denominator.
    nonzero = d_abs != 0
    n_eff = int(nonzero.sum())
    n_lin = int(np.sum(d_abs < 0))
    n_gp = int(np.sum(d_abs > 0))

    out: dict[str, Any] = dict(
        n_folds=n, n_effective=int(n_eff),
        n_linear_better=n_lin, n_ties=int(n - n_eff), n_gp_better=n_gp,
        mean_d_abs=float(np.mean(d_abs)) if n else None,
        sd_d_abs=float(np.std(d_abs, ddof=1)) if n > 1 else 0.0,
        mean_d_rel=float(np.mean(d_rel)) if n else None,
        sd_d_rel=float(np.std(d_rel, ddof=1)) if n > 1 else 0.0,
        min_attainable_p=float(2.0 ** (-n_eff)) if n_eff else None,
        sign_test_note="two-sided exact sign test over the NON-TIED folds",
    )
    out["sign_test_p"] = (float(stats.binomtest(n_lin, n_eff, 0.5).pvalue)
                          if n_eff > 0 else None)
    if n_eff > 1:
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(d_abs, zero_method="wilcox").pvalue)
        except ValueError:                                  # pragma: no cover
            out["wilcoxon_p"] = None
    if n > 1:
        rng = np.random.default_rng(seed)
        b_rel = d_rel[rng.integers(0, n, size=(n_boot, n))].mean(1)
        b_abs = d_abs[rng.integers(0, n, size=(n_boot, n))].mean(1)
        out["boot_ci_rel"] = [float(np.percentile(b_rel, 2.5)), float(np.percentile(b_rel, 97.5))]
        out["boot_ci_abs"] = [float(np.percentile(b_abs, 2.5)), float(np.percentile(b_abs, 97.5))]
    return out


# ---------------------------------------------------------------------------
# canonical hashing so two runs (or five artifacts) can be shown to agree
# ---------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# Canonical float precision for the shared-baseline hash.
#
# Two runs of identical code on identical data can differ in the last bit of a double when
# a reduction order changes (a BLAS thread count, a different CPU). That is not a finding,
# but it WOULD change a raw-f64 hash and make the artifact's side of the cross-artifact
# agreement proof fail for no reason. Rounding to 12 significant digits -- far beyond what
# any baseline claim depends on, far below a ULP -- makes the hash stable under that noise
# while still catching any real disagreement. The rounding is recorded in the emitted table
# so a reader can see exactly what was hash-committed.
HASH_SIG_DIGITS = 12


def _round_floats(o: Any, sig: int = HASH_SIG_DIGITS) -> Any:
    if isinstance(o, bool) or isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        if v == 0.0 or not np.isfinite(v):
            return v
        return float(f"{v:.{sig}g}")
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, dict):
        return {k: _round_floats(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round_floats(v, sig) for v in o]
    if isinstance(o, np.ndarray):
        return [_round_floats(v, sig) for v in o.tolist()]
    return o


def canonical_hash(obj: Any, sig: int = HASH_SIG_DIGITS) -> str:
    """Order-independent, float-precision-stable SHA-256 of a JSON-serialisable object."""
    blob = json.dumps(_round_floats(obj, sig), sort_keys=True, separators=(",", ":"),
                      default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
