"""Guard g3 diagnostic: does an extended lengthscale grid recover the collapsed arms?

Guard g1 (ratio to the mean predictor >= 1.0) fired for b2, b3 and b3p on the T4 fold.
The diagnostic order the plan prescribes is (i) check whether the selected lengthscale
sits at a grid endpoint and extend the grid if so. This script measures, on the same
fold, what the distance scale actually is and what each arm does across a decade-wide
extension.
"""
import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import mahalanobis_sqd, standardise  # noqa: E402
from engine import project_injections  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import gp_fit_predict, normalise_distance, sqd  # noqa: E402
from method_core import prepare  # noqa: E402

WIDE = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)

if __name__ == "__main__":
    prep = prepare()
    X, Y = prep["X"], prep["Y"]
    tr, te = fold_indices(80, 0, 0)
    Ytr, Yte = Y[tr], Y[te]
    ysd = Ytr.std(0).mean()
    mean_e = np.sqrt(((np.tile(Ytr.mean(0), (len(te), 1)) - Yte) ** 2).mean()) / ysd
    print("mean predictor %.4f" % mean_e)

    def report(tag, Dtr, Dst, dim):
        off = Dtr[~np.eye(Dtr.shape[0], dtype=bool)]
        raw = off.mean()
        A, B = normalise_distance(Dtr, Dst, dim)
        aoff = A[~np.eye(A.shape[0], dtype=bool)]
        for lg, lab in ((None, "plan-grid"), (WIDE, "wide")):
            mu, pv, hp = gp_fit_predict(A, Ytr, B, lgrid=lg if lg else (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8))
            e = np.sqrt(((mu - Yte) ** 2).mean()) / ysd
            print(f"  {tag:<14} {lab:<10} rawD {raw:9.2f} normD {aoff.mean():7.2f} "
                  f"l={hp[0]:<6} nrmse {e:.4f} ratio {e/mean_e:.3f}")

    Ztr, Zte, _ = standardise(X[tr], X[te])
    report("b1_rbf", sqd(Ztr, Ztr), sqd(Zte, Ztr), 110)
    report("b3_M110", mahalanobis_sqd(Ztr, Ztr, prep["M110"]),
           mahalanobis_sqd(Zte, Ztr, prep["M110"]), 110)
    Ptr, Pte, _ = standardise(X[tr, :55], X[te, :55])
    report("b2_hop", mahalanobis_sqd(Ptr, Ptr, prep["Mhop"]),
           mahalanobis_sqd(Pte, Ptr, prep["Mhop"]), 55)
    report("b3p_M55", mahalanobis_sqd(Ptr, Ptr, prep["M55"]),
           mahalanobis_sqd(Pte, Ptr, prep["M55"]), 55)
    for k in (5, 11, 22):
        Fk = prep["ops"].features("R1", k, "real")
        Atr, Ate = project_injections(X[tr], X[te], Fk)
        Atr, Ate, _ = standardise(Atr, Ate)
        report(f"SP-k{k}", sqd(Atr, Atr), sqd(Ate, Atr), Atr.shape[1])
