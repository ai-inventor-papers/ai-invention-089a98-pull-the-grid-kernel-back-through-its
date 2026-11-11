# NOTE (published copy): this file names server paths this repository
# does not publish (a stage it does not ship, or another run's workspace),
# so the steps that read them will not run from a clone as written:
#   /ai-inventor/aii_data/runs/run_RlJ6BnxHoG2N/iter_2/gen_hypo/claude_agent/pilot
"""Throw-away: why is this implementation's treatment 4.5x worse than the archived one
at n=160 on nominally the same folds?  Sweep the operating choices one at a time."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feeder import Physics                                          # noqa: E402
from data import load_ensemble                                     # noqa: E402
from folds import make_fold                                        # noqa: E402
from kernels import gp_fit_predict                                 # noqa: E402
from operators import build_operator                               # noqa: E402

PILOT = "/ai-inventor/aii_data/runs/run_RlJ6BnxHoG2N/iter_2/gen_hypo/claude_agent/pilot"
ph = Physics()
far, s15 = ph.sheet(15)
ens = load_ensemble("2.0")
X, Y = ens["X"], ens["Y"]


def feat(S, Xtr, Xte):
    mx = Xtr.mean(0)
    Ftr = (Xtr - mx) @ S.T
    Fte = (Xte - mx) @ S.T
    sc = np.sqrt(np.einsum("ij,ij->i", Ftr, Ftr).mean()) + 1e-12
    return Ftr / sc, Fte / sc


def ev(S, n, seed, scheme, refine, vsd):
    tr, te = make_fold(n, seed, scheme)
    Ftr, Fte = feat(S, X[tr], X[te])
    r = gp_fit_predict(Ftr, Fte, Y[tr], refine=refine)
    return np.sqrt(np.mean((r["mu"] - Y[te]) ** 2)) / vsd, r["ell"], r["n2"], r["s2"]


S15 = np.load(f"{PILOT}/S15.npy")
cands = {"archived S15.npy": S15}
for tr_mode in ("none", "pu_correct", "legacy"):
    for v0m in ("flat", "noload"):
        cands[f"{tr_mode}/{v0m}"] = build_operator(ph, s15, ph.phase, tr_mode, v0m)

for n, seed, scheme in [(20, 0, 0), (160, 0, 0), (80, 0, 0)]:
    tr, te = make_fold(n, seed, scheme)
    vsd = Y[tr].std(0).mean()
    vsd_all = Y.std(0).mean()
    print(f"\n--- n={n} seed={seed} scheme={scheme} (train vsd {vsd:.5f}, all {vsd_all:.5f})")
    for name, S in cands.items():
        for refine in (True, False):
            e, ell, n2, s2 = ev(S, n, seed, scheme, refine, vsd)
            print(f"  {name:22s} refine={int(refine)}  rmse {e:.4f}  "
                  f"ell={ell:.2f} s2={s2:.1f} n2={n2:.0e}")

# does the ARCHIVE really get 0.022 at n=160? reproduce its split rule:
rng = np.random.default_rng(31337 + 0)
idx = rng.permutation(220)
tr, te = idx[:160], idx[160:]
Ftr, Fte = feat(S15, X[tr], X[te])
r = gp_fit_predict(Ftr, Fte, Y[tr], refine=False)
print("\narchive split rule n=160: rmse",
      np.sqrt(np.mean((r["mu"] - Y[te]) ** 2)) / Y.std(0).mean(), "ell", r["ell"], "n2", r["n2"])
