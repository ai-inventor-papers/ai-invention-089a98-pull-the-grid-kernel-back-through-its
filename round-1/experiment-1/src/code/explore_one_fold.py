"""Throw-away: one fold, every candidate side by side, and the T5 baseline sanity checks."""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feeder import Physics                                          # noqa: E402
from data import load_ensemble                                     # noqa: E402
from folds import make_fold                                        # noqa: E402
from screen import run_fold, norm_rmse                             # noqa: E402
from operators import build_operator, spectral_facts               # noqa: E402

t0 = time.time()
ph = Physics()
far, s15 = ph.sheet(15)
ens = load_ensemble("2.0")
X, Y = ens["X"], ens["Y"]
vsd = Y.std(0).mean()
print("load", time.time() - t0, ens["checks"])

for n, seed, scheme in [(20, 0, 0), (160, 0, 0), (20, 1, 1)]:
    tr, te = make_fold(n, seed, scheme)
    t1 = time.time()
    rec = run_fold(X, Y, tr, te, ph, s15, ph.phase, vsd)
    print(f"\nn={n} seed={seed} scheme={scheme}  ({time.time()-t1:.1f}s)")
    for k in ("mean", "treatment", "b1_rbf", "b2_hop", "b3_adm", "b4_ridge",
              "b5_lindistflow", "b5_legacy"):
        if k in rec:
            r = rec[k]
            extra = ""
            if "ell" in r:
                extra += f" ell={r['ell']:.3f} n2={r['n2']:.0e}"
            if "cell" in r:
                extra += f" [{r['cell']}]"
            if "form" in r:
                extra += f" [{r['form']}]"
            if "nlpd" in r:
                extra += f" nlpd={r['nlpd']:.2f} c68={r['coverage68']:.2f}"
            print(f"   {k:16s} {r['rmse']:.4f}{extra}")
    print(f"   best baseline: {rec['best_baseline_name']} "
          f"margin {100*rec['margin_vs_best']:+.1f}%  vs_b5 "
          f"{100*rec.get('margin_vs_b5', float('nan')):+.1f}%")
    if "b5_lindistflow" in rec:
        cells = sorted(rec["b5_lindistflow"]["all_cells_rmse"].items(), key=lambda kv: kv[1])
        print("   b5 cells (best 4 by test rmse):",
              [(k, round(v, 4)) for k, v in cells[:4]])

S = build_operator(ph, s15, ph.phase, "pu_correct", "noload")
sf = spectral_facts(S)
print("\nS_far spectrum:", {k: v for k, v in sf.items() if "eff_rank" in k or k == "stable_rank"})
