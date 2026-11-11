"""Throw-away: is the marginal-likelihood grid behaving sensibly?  Inspect the MLL
surface and the train/test RMSE at each cell for the treatment at n=160."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feeder import Physics                                          # noqa: E402
from data import load_ensemble                                     # noqa: E402
from folds import make_fold                                        # noqa: E402
from kernels import gram_dist, sqdist, _mll_one                    # noqa: E402
from operators import build_operator                               # noqa: E402

ph = Physics()
far, s15 = ph.sheet(15)
ens = load_ensemble("2.0")
X, Y = ens["X"], ens["Y"]
S = build_operator(ph, s15, ph.phase, "pu_correct", "noload")

tr, te = make_fold(160, 0, 0)
vsd = Y[tr].std(0).mean()
mx = X[tr].mean(0)
Ftr = (X[tr] - mx) @ S.T
Fte = (X[te] - mx) @ S.T
sc = np.sqrt(np.einsum("ij,ij->i", Ftr, Ftr).mean())
Ftr, Fte = Ftr / sc, Fte / sc
Ytr, Yte = Y[tr], Y[te]
Ybar, Ysd = Ytr.mean(0), Ytr.std(0)
Yc = (Ytr - Ybar) / Ysd
D = gram_dist(Ftr)
Dst = sqdist(Fte, Ftr)

print(f"{'ell':>6} {'s2':>5} {'n2':>7} {'mll':>10} {'train':>8} {'test':>8}")
rows = []
for ell in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32):
    for s2 in (0.3, 1.0, 3.0, 10.0):
        for n2 in (1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            r = _mll_one(D, Yc, ell, s2, n2)
            if r is None:
                continue
            Ks = s2 * np.exp(-Dst / (2 * ell * ell))
            a = np.linalg.solve(r["cf"][0].T, np.linalg.solve(r["cf"][0], Ks.T))
            mu = (Ks @ r["alpha"]) * Ysd + Ybar
            trmu = (s2 * np.exp(-D / (2 * ell * ell)) @ r["alpha"]) * Ysd + Ybar
            e_te = np.sqrt(np.mean((mu - Yte) ** 2)) / vsd
            e_tr = np.sqrt(np.mean((trmu - Ytr) ** 2)) / vsd
            rows.append((r["mll"], ell, s2, n2, e_tr, e_te))
rows.sort(reverse=True, key=lambda r: r[0])
print("TOP 12 BY MLL")
for r in rows[:12]:
    print(f"{r[1]:6.2f} {r[2]:5.1f} {r[3]:7.0e} {r[0]:10.1f} {r[4]:8.4f} {r[5]:8.4f}")
rows.sort(key=lambda r: r[5])
print("TOP 8 BY TEST RMSE")
for r in rows[:8]:
    print(f"{r[1]:6.2f} {r[2]:5.1f} {r[3]:7.0e} {r[0]:10.1f} {r[4]:8.4f} {r[5]:8.4f}")
