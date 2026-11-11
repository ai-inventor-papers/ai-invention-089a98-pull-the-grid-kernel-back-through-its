"""Is the spectral GP at l=80 just linear regression, and is my ridge baseline weakened?

At the selected lengthscale (l = 80 in normalised distance units, where the mean squared
distance is 22) the RBF kernel is exp(-D/12800) ~ 1 - D/12800: a linear kernel plus a
constant. So the spectral GP should reduce to Bayesian linear regression on the same 22
coordinates, and SP-R1a-LIN-k11 (ridge on those coordinates) should match it. It does not
- the GP is ~4x better - so either (a) the GP is doing something beyond linear, or
(b) my ridge CV penalty is over-regularising, i.e. the baseline is weakened.

The plan's central warning is (b): a weakened baseline manufactures the candidate's
margin. This script measures OLS, ridge at several penalties, and the GP's own optimum on
IDENTICAL coordinates, and reports which explanation holds.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import standardise  # noqa: E402
from engine import project_injections  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import (FRACS, gp_fit_predict, normalise_distance, ols_intercept, sqd)  # noqa: E402
from method_core import prepare  # noqa: E402


def nrmse(mu, Yte, ysd):
    return float(np.sqrt(((mu - Yte) ** 2).mean()) / ysd)


if __name__ == "__main__":
    prep = prepare()
    ops, X, Y = prep["ops"], prep["X"], prep["Y"]
    Fk = ops.features("R1", 11, "real")
    rows = []
    for sch in (0, 1):
        for s in range(12):
            f = fold_indices(80, s, sch)
            if f is None:
                continue
            tr, te = f
            ysd = Y[tr].std(0).mean()
            Atr, Ate = project_injections(X[tr], X[te], Fk)
            Atr, Ate, _ = standardise(Atr, Ate)
            r = {"ols": nrmse(ols_intercept(Atr, Y[tr], Ate), Y[te], ysd)}
            lam_max = np.linalg.eigvalsh(
                np.hstack([Atr, np.ones((len(Atr), 1))]).T @
                np.hstack([Atr, np.ones((len(Atr), 1))])).max()
            A1 = np.hstack([Atr, np.ones((len(Atr), 1))])
            B1 = np.hstack([Ate, np.ones((len(Ate), 1))])
            for frac in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
                ann = np.ones(A1.shape[1]); ann[-1] = 0.0
                c = np.linalg.solve(A1.T @ A1 + frac * lam_max * np.diag(ann), A1.T @ Y[tr])
                r[f"ridge{frac:g}"] = nrmse(B1 @ c, Y[te], ysd)
            Dtr, Dst = normalise_distance(sqd(Atr, Atr), sqd(Ate, Atr), Atr.shape[1])
            mu, pv, hp = gp_fit_predict(Dtr, Y[tr], Dst)
            r["gp"] = nrmse(mu, Y[te], ysd)
            r["gp_l"] = hp["hyper"][0]
            r["gp_n2"] = hp["hyper"][2]
            rows.append(r)

    print(f"\n=== n = 80, {len(rows)} folds, same 2k = 22 spectral coordinates ===")
    keys = [k for k in rows[0] if k not in ("gp_l", "gp_n2")]
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {k:<16} mean {v.mean():.4f}  worst {v.max():.4f}  best {v.min():.4f}")
    print(f"  gp selected l   mean {np.mean([r['gp_l'] for r in rows]):.0f}   "
          f"n2 counts {np.unique([r['gp_n2'] for r in rows], return_counts=True)}")
    gp = np.array([r["gp"] for r in rows])
    ols = np.array([r["ols"] for r in rows])
    print(f"\n  GP vs OLS mean ratio {gp.mean()/ols.mean():.3f}")
    print("  -> if that ratio is ~1 the GP at l=80 IS linear regression, and any ridge arm")
    print("     materially worse than OLS is a weakened baseline that must be fixed.")
