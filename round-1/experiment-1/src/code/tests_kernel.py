"""T4 -- GP correctness on a synthetic problem with a known answer.

Checks (a) interpolation at the training points with the smallest noise level,
(b) reversion to the training mean and sigma2 away from the data,
(c) marginal-likelihood recovery of the generating lengthscale within a factor 2.
A mis-transposed prediction still returns plausible numbers, so this is the test
that catches it.
"""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernels import gp_fit_predict, sqdist, gram_dist, _search   # noqa: E402


def run() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    # (a) interpolation
    x = np.sort(rng.uniform(-3, 3, (200, 1)))
    y = np.sin(2.0 * x) + 0.01 * x
    F = x / 1.0
    r = gp_fit_predict(F, F, y, want_var=True, refine=True)
    out["interp_rmse"] = float(np.sqrt(np.mean((r["mu"] - y) ** 2)))
    out["selected_n2"] = float(r["n2"])

    # (b) reversion far from data
    Far = np.array([[50.0]])
    r2 = gp_fit_predict(F, Far, y, want_var=True)
    out["far_mean_dev"] = float(abs(r2["mu"][0, 0] - y.mean()))
    out["far_var_over_s2"] = float(r2["var"][0, 0] / r["s2"])

    # (c) lengthscale recovery on a clean 1-d GP draw
    n = 120
    xtr = np.sort(rng.uniform(-4, 4, (n, 1)))
    D = sqdist(xtr / 1.0, xtr / 1.0)
    ell_true = 0.9
    K = np.exp(-D / (2 * ell_true ** 2)) + 1e-6 * np.eye(n)
    L = np.linalg.cholesky(K)
    yv = (L @ rng.normal(size=(n, 1))) * 1.0
    best = _search(D, (yv - yv.mean(0)) / (yv.std(0) + 1e-12), refine=True)
    out["ell_selected"] = float(best["ell"])
    out["ell_true"] = ell_true
    out["ell_ratio"] = float(best["ell"] / ell_true)

    out["PASS_interp"] = bool(out["interp_rmse"] < 1e-2)
    out["PASS_revert"] = bool(out["far_mean_dev"] < 0.05 and 0.5 < out["far_var_over_s2"] < 2.0)
    out["PASS_lengthscale"] = bool(0.5 <= out["ell_ratio"] <= 2.0)
    out["PASS"] = bool(out["PASS_interp"] and out["PASS_revert"] and out["PASS_lengthscale"])
    return out


if __name__ == "__main__":
    r = run()
    for k, v in r.items():
        print(f"{k}: {v}")
