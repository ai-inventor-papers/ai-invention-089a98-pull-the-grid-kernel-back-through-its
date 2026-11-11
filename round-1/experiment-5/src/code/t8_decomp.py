"""DECOMPOSITION: is the 82% margin from the SPECTRUM or from wide-RBF interpolation?

The candidate's GP selects l ~ 80 in normalised units, where every kernel entry is
~0.99-1.0. A Gaussian RBF at such a width is a high-order polynomial feature expansion;
with n2 = 1e-8 it interpolates. That machinery is available to ANY coordinate set. So the
honest question is not "does the spectral GP beat ridge on the same coordinates" (it does,
4x - measured in t7_lincheck) but "does it beat a GP with the SAME machinery on a
coordinate set that carries no feeder information".

This is the 2x2 the artifact must report:
                       linear (OLS)      nonlinear (wide-RBF GP)
  spectral coords      A                 B
  PCA coords           C                 D
  random coords        E                 F

  Spectrum's contribution at fixed learner = A vs C, and B vs D.
  Learner's contribution at fixed coords    = B/A, D/C.
If B ~= D, the spectrum contributes nothing at the headline operating point and the
margin is generic interpolation.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import standardise  # noqa: E402
from engine import project_injections  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import gp_fit_predict, normalise_distance, ols_intercept, sqd  # noqa: E402
from method_core import prepare  # noqa: E402

K = 11


def pair(Atr, Ate, Ytr, Yte):
    """OLS and the fixed GP protocol on identical coordinates."""
    ysd = Ytr.std(0).mean()
    ols = float(np.sqrt(((ols_intercept(Atr, Ytr, Ate) - Yte) ** 2).mean()) / ysd)
    Dtr, Dst = normalise_distance(sqd(Atr, Atr), sqd(Ate, Atr), Atr.shape[1])
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst)
    gp = float(np.sqrt(((mu - Yte) ** 2).mean()) / ysd)
    z = (Yte - mu) / np.sqrt(np.maximum(pv, 1e-300))
    return ols, gp, hp, float(np.mean(np.abs(z) <= 1.0))


if __name__ == "__main__":
    prep = prepare()
    ops, X, Y = prep["ops"], prep["X"], prep["Y"]
    F_spec = ops.features("R1", K, "real")
    rng = np.random.default_rng(11)
    F_rand = rng.normal(size=(55, K))

    acc: dict = {k: {"ols": [], "gp": [], "cov68": [], "l": []}
                 for k in ("spectral", "pca", "random")}
    for sch in (0, 1):
        for s in range(12):
            f = fold_indices(80, s, sch)
            if f is None:
                continue
            tr, te = f
            Ztr, Zte, _ = standardise(X[tr], X[te])
            _, _, Vt = np.linalg.svd(Ztr, full_matrices=False)
            sets = {
                "spectral": project_injections(X[tr], X[te], F_spec),
                "pca": (Ztr @ Vt[:2 * K].T, Zte @ Vt[:2 * K].T),
                "random": project_injections(X[tr], X[te], F_rand),
            }
            for tag, (Atr, Ate) in sets.items():
                Atr, Ate, _ = standardise(Atr, Ate)
                o, g, hp, c68 = pair(Atr, Ate, Y[tr], Y[te])
                acc[tag]["ols"].append(o)
                acc[tag]["gp"].append(g)
                acc[tag]["cov68"].append(c68)
                acc[tag]["l"].append(hp["hyper"][0])

    print(f"\n=== n = 80, {len(acc['spectral']['gp'])} folds, 2k = {2*K} coordinates ===")
    print(f"{'coords':<12} {'OLS':>8} {'wide-RBF GP':>13} {'GP/OLS':>8} "
          f"{'cov68':>7} {'mean l':>8}")
    for tag, d in acc.items():
        o, g = np.mean(d["ols"]), np.mean(d["gp"])
        print(f"{tag:<12} {o:8.4f} {g:13.4f} {g/o:8.3f} "
              f"{np.mean(d['cov68']):7.3f} {np.mean(d['l']):8.0f}")
    sp, pc = np.mean(acc["spectral"]["gp"]), np.mean(acc["pca"]["gp"])
    print(f"\n  learner contribution at spectral coords: "
          f"{np.mean(acc['spectral']['ols'])/sp:.2f}x")
    print(f"  learner contribution at PCA coords     : "
          f"{np.mean(acc['pca']['ols'])/pc:.2f}x")
    print(f"  SPECTRUM contribution at the GP learner: {pc/sp:.2f}x better than PCA")
    print(f"  SPECTRUM contribution at the OLS learner: "
          f"{np.mean(acc['pca']['ols'])/np.mean(acc['spectral']['ols']):.2f}x better than PCA")
    json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
               for k, v in acc.items()}, open("results/decomposition_n80.json", "w"), indent=1)
    print("\nwrote results/decomposition_n80.json")
