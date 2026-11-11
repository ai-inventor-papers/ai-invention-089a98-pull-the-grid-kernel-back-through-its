"""Pick ONE uniform distance convention that leaves every kernel non-degenerate.

Why a convention is needed at all: a squared distance in 55-110 dimensions concentrates
around its mean, so an unnormalised kernel matrix is nearly constant (all-ones) or
nearly diagonal depending on the metric's own units, and the GP silently reports the
mean predictor. The plan prescribes exactly this diagnostic (guard g1 / fallback F5).

Conventions compared, all computed on TRAIN rows only:
  mean     D / mean(off-diagonal)          -> mean distance 1
  median   D / median(off-diagonal)        -> median distance 1
  std      D / std(off-diagonal)           -> unit distance spread
  perdim   D / (mean(off-diagonal)/d)      -> lengthscale in per-dimension units
For each, the GP is swept over a decade-wide grid and we record the best achievable
normalised RMSE. The convention that gives every arm a non-degenerate, interior
lengthscale is the one adopted.
"""
import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import mahalanobis_sqd, standardise  # noqa: E402
from engine import project_injections  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import gp_fit_predict, sqd  # noqa: E402
from method_core import prepare  # noqa: E402

WIDE = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def conv(D, kind, d, n_tr):
    """Scale a squared-distance matrix using TRAIN-block statistics only."""
    off = D[:n_tr, :n_tr][~np.eye(n_tr, dtype=bool)]
    if kind == "mean":
        s = off.mean()
    elif kind == "median":
        s = np.median(off)
    elif kind == "std":
        s = off.std()
    elif kind == "perdim":
        s = off.mean() / max(d, 1)
    else:
        raise ValueError(kind)
    return D / max(s, 1e-30)


if __name__ == "__main__":
    prep = prepare()
    X, Y = prep["X"], prep["Y"]
    tr, te = fold_indices(80, 0, 0)
    Ytr, Yte = Y[tr], Y[te]
    ysd = Ytr.std(0).mean()
    mean_e = np.sqrt(((np.tile(Ytr.mean(0), (len(te), 1)) - Yte) ** 2).mean()) / ysd

    Ztr, Zte, _ = standardise(X[tr], X[te])
    Ptr, Pte, _ = standardise(X[tr, :55], X[te, :55])
    sets = {
        "b1_rbf": (sqd(Ztr, Ztr), sqd(Zte, Ztr), 110),
        "b2_hop": (mahalanobis_sqd(Ptr, Ptr, prep["Mhop"]),
                   mahalanobis_sqd(Pte, Ptr, prep["Mhop"]), 55),
        "b3_M110": (mahalanobis_sqd(Ztr, Ztr, prep["M110"]),
                    mahalanobis_sqd(Zte, Ztr, prep["M110"]), 110),
        "b3p_M55": (mahalanobis_sqd(Ptr, Ptr, prep["M55"]),
                    mahalanobis_sqd(Pte, Ptr, prep["M55"]), 55),
    }
    for k in (3, 5, 11, 22, 45):
        Fk = prep["ops"].features("R1", k, "real")
        Atr, Ate = project_injections(X[tr], X[te], Fk)
        Atr, Ate, _ = standardise(Atr, Ate)
        sets[f"SP-R1a-k{k}"] = (sqd(Atr, Atr), sqd(Ate, Atr), Atr.shape[1])

    for kind in ("mean", "median", "std", "perdim"):
        print(f"\n=== convention: {kind} ===")
        for tag, (Dtr, Dst, d) in sets.items():
            n_tr = Dtr.shape[0]
            A, B = conv(Dtr, kind, d, n_tr), conv(Dst, kind, d, n_tr)
            mu, pv, hp = gp_fit_predict(A, Ytr, B, lgrid=WIDE)
            e = np.sqrt(((mu - Yte) ** 2).mean()) / ysd
            off = A[~np.eye(n_tr, dtype=bool)]
            flag = "  <-- DEGENERATE" if e / mean_e > 0.98 else ""
            print(f"  {tag:<12} scale mean {off.mean():7.3f} sd {off.std():7.3f} "
                  f"l={hp[0]:<6} nrmse {e:.4f} ratio {e/mean_e:.3f}{flag}")
