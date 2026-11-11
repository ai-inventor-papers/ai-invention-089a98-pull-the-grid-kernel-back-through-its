"""Throw-away: MLL vs train-only CV hyperparameter selection, per candidate, and a
ridge-on-S-features control (how much of the treatment's edge is just the linear
map in the pulled-back coordinates?)."""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feeder import Physics                                          # noqa: E402
from data import load_ensemble                                     # noqa: E402
from folds import make_fold                                        # noqa: E402
from kernels import gp_fit_predict, sqdist                         # noqa: E402
from operators import build_operator                               # noqa: E402
from baselines import standardise_train                            # noqa: E402

ph = Physics()
far, s15 = ph.sheet(15)
ens = load_ensemble("2.0")
X, Y = ens["X"], ens["Y"]
S = build_operator(ph, s15, ph.phase, "pu_correct", "noload")


def feats(Sop, Xtr, Xte):
    mx = Xtr.mean(0)
    Ftr = (Xtr - mx) @ Sop.T
    Fte = (Xte - mx) @ Sop.T
    sc = np.sqrt(np.einsum("ij,ij->i", Ftr, Ftr).mean()) + 1e-12
    return Ftr / sc, Fte / sc


def ridge_S(Ftr, Fte, Ytr):
    n, p = Ftr.shape
    G = Ftr.T @ Ftr
    lam_max = float(np.linalg.eigvalsh(G)[-1])
    grid = [f * lam_max for f in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)]
    k = max(2, n // 4)
    folds = [f for f in np.array_split(np.arange(n), k) if len(f) > 0]
    best = None
    for lam in grid:
        s = 0.0
        for f in folds:
            tri = np.setdiff1d(np.arange(n), f)
            if len(tri) < 3:
                continue
            G1 = Ftr[tri].T @ Ftr[tri]
            c = np.linalg.solve(G1 + lam * np.eye(p), Ftr[tri].T @ Ytr[tri])
            s += float(np.mean((Ftr[f] @ c - Ytr[f]) ** 2)) / len(folds)
        if best is None or s < best[0]:
            best = (s, lam)
    lam = best[1]
    c = np.linalg.solve(G + lam * np.eye(p), Ftr.T @ Ytr)
    return Fte @ c, lam


Xtr_all, Xte_all = X, Y
print(f"{'n':>4} {'seed':>4} {'sch':>3} | {'treat_mll':>9} {'treat_cv':>8} "
      f"{'rbf_mll':>8} {'rbf_cv':>7} {'ridgeS':>7} {'mean':>6}")
for n in (10, 20, 40, 80, 160):
    for seed in (0, 5):
        for sch in (0, 1):
            if n >= 160 and sch == 1:
                continue
            tr, te = make_fold(n, seed, sch)
            vsd = Y[tr].std(0).mean()
            Ytr, Yte = Y[tr], Y[te]
            Ftr, Fte = feats(S, X[tr], X[te])
            rm = np.sqrt(np.mean((gp_fit_predict(Ftr, Fte, Ytr, select="mll")["mu"] - Yte) ** 2)) / vsd
            rc = np.sqrt(np.mean((gp_fit_predict(Ftr, Fte, Ytr, select="cv")["mu"] - Yte) ** 2)) / vsd
            Gm, Gt, _, _ = standardise_train(X[tr], X[te])
            bm = np.sqrt(np.mean((gp_fit_predict(Gm, Gt, Ytr, select="mll")["mu"] - Yte) ** 2)) / vsd
            bc = np.sqrt(np.mean((gp_fit_predict(Gm, Gt, Ytr, select="cv")["mu"] - Yte) ** 2)) / vsd
            rs, lam = ridge_S(Ftr, Fte, Ytr)
            er = np.sqrt(np.mean((rs - Yte) ** 2)) / vsd
            em = np.sqrt(np.mean((np.tile(Ytr.mean(0), (len(te), 1)) - Yte) ** 2)) / vsd
            print(f"{n:4d} {seed:4d} {sch:3d} | {rm:9.4f} {rc:8.4f} {bm:8.4f} {bc:7.4f} "
                  f"{er:7.4f} {em:6.4f}")
