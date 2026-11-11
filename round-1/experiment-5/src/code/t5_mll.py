"""Diagnose why the Mahalanobis kernels (b2, b3) select a near-identity lengthscale.

Prints, for one fold, the marginal likelihood AND the held-out normalised RMSE at every
lengthscale in a decade-wide grid, plus the distance distribution. If the MLL surface
itself peaks at a short lengthscale while a longer one predicts far better, the arm is
degenerate by its own objective and the honest fix is a kernel-family statement, not a
silent re-tune. If the MLL peaks where RMSE is bad for a different reason (numerics),
that must be found here before any number is reported.
"""
import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import mahalanobis_sqd, standardise  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import sqd  # noqa: E402
from method_core import prepare  # noqa: E402

WIDE = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)


def surface(Dtr, Dst, Ytr, Yte, ysd, tag):
    n, m = Ytr.shape
    yb, ys = Ytr.mean(0), Ytr.std(0)
    ys = np.where(ys < 1e-12, 1.0, ys)
    Yc = (Ytr - yb) / ys
    print(f"\n--- {tag} ---")
    off = Dtr[~np.eye(n, dtype=bool)]
    print("  train D: min %.3e  5%% %.3f  median %.3f  mean %.3f  95%% %.3f  max %.3e"
          % (off.min(), np.percentile(off, 5), np.median(off), off.mean(),
             np.percentile(off, 95), off.max()))
    print("  %8s %12s %12s %10s" % ("l", "mll", "nrmse", "s2/n2"))
    for l in WIDE:
        best = None
        for s2 in (0.3, 1.0, 3.0):
            for n2 in (1e-8, 1e-5, 1e-3, 1e-1):
                K = s2 * np.exp(-Dtr / (2 * l * l)) + n2 * np.eye(n)
                try:
                    Lc = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                al = np.linalg.solve(Lc, Yc)
                mll = -0.5 * (al ** 2).sum() - m * np.log(np.diag(Lc)).sum()
                if best is None or mll > best[0]:
                    best = (mll, s2, n2, Lc, al)
        if best is None:
            print("  %8.2f  all Cholesky failed" % l)
            continue
        mll, s2, n2, Lc, al = best
        a = np.linalg.solve(Lc.T, al)
        Ks = s2 * np.exp(-Dst / (2 * l * l))
        mu = (Ks @ a) * ys + yb
        e = np.sqrt(((mu - Yte) ** 2).mean()) / ysd
        print("  %8.2f %12.2f %12.4f %10.2e" % (l, mll, e, s2 / n2))


if __name__ == "__main__":
    prep = prepare()
    X, Y = prep["X"], prep["Y"]
    tr, te = fold_indices(80, 0, 0)
    Ytr, Yte = Y[tr], Y[te]
    ysd = Ytr.std(0).mean()
    Ztr, Zte, _ = standardise(X[tr], X[te])
    Ptr, Pte, _ = standardise(X[tr, :55], X[te, :55])
    surf = {
        "b1_rbf": (sqd(Ztr, Ztr), sqd(Zte, Ztr)),
        "b3_M110": (mahalanobis_sqd(Ztr, Ztr, prep["M110"]),
                    mahalanobis_sqd(Zte, Ztr, prep["M110"])),
        "b2_hop": (mahalanobis_sqd(Ptr, Ptr, prep["Mhop"]),
                   mahalanobis_sqd(Pte, Ptr, prep["Mhop"])),
    }
    for tag, (Dtr, Dst) in surf.items():
        n = Dtr.shape[0]
        s = Dtr[~np.eye(n, dtype=bool)].mean()
        surface(Dtr / s, Dst / s, Ytr, Yte, ysd, tag)
    print("\nb3 M110 spectral summary:")
    ev = np.linalg.eigvalsh(prep["M110"])
    print("  eig min %.3e max %.3e cond %.3e  frac<1e-8*max %.3f"
          % (ev.min(), ev.max(), ev.max() / max(ev.min(), 1e-300),
             float(np.mean(ev < 1e-8 * ev.max()))))
