"""Diagnose the small-n baselines. At n = 10 several baselines exceed the mean predictor.

The artifact's central warning is that a weakened baseline manufactures the candidate's
margin. The raw per-fold records show that at n = 10, b4_ridge reaches ~3e3 and
b5_lindist ~1.86 while the mean predictor is 1.37 — so at the smallest training size the
margin is measured against baselines that are failing outright. That is a real result about
those baselines at n = 10, but it must be characterised rather than left to flatter the
candidate. This script asks, for the SAME folds:

  (a) what is b5's error under each of its five calibrations at n = 10?
  (b) what does the literature-honest per-output affine fit give (the reviewer measured
      0.1665 at n = 10)?
  (c) is b4_ridge's blow-up the unpenalised-intercept-plus-110-columns problem?

If a calibration that a practitioner would actually use beats the mean predictor at n = 10,
then b5-as-scored is the weakened one and must be fixed before any margin at n = 10 is
quoted.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import b5_calibrations, b5_lindist  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import ridge_fit_predict  # noqa: E402
from method_core import prepare  # noqa: E402

NS = (10, 20, 40, 80)


def main() -> None:
    prep = prepare()
    Y, Vp1, Vp2 = prep["Y"], prep["Vp1"], prep["Vp2"]

    print("\n=== b5 LinDistFlow: error of each calibration (mean over 24 folds) ===")
    print(f"{'n':>4} {'meanPred':>9} " + " ".join(
        f"{k:>20}" for k in ("p17_joint46_trace", "joint46_ols", "joint46_ridge_free",
                             "per_output_ols", "per_output_gain", "per_output_ridge_cv")))
    for n in NS:
        rows = []
        for sch in (0, 1):
            for s in range(12):
                f = fold_indices(n, s, sch)
                if f is None:
                    continue
                tr, te = f
                ysd = Y[tr].std(0).mean()
                meanpred = np.sqrt(((np.tile(Y[tr].mean(0), (len(te), 1)) - Y[te]) ** 2).mean()) / ysd
                mu, _, hp, order = b5_lindist(Vp1[tr], Vp2[tr], Y[tr], Vp1[te], Vp2[te])
                cal = b5_calibrations(Vp1[tr], Vp2[tr], Y[tr], Vp1[te], Vp2[te], order)
                rec = {"meanPred": meanpred,
                       "per_output_ridge_cv": np.sqrt(((mu - Y[te]) ** 2).mean()) / ysd,
                       "order": order}
                for k, v in cal.items():
                    rec[k] = np.sqrt(((v - Y[te]) ** 2).mean()) / ysd
                rows.append(rec)
        keys = ("p17_joint46_trace", "joint46_ols", "joint46_ridge_free",
                "per_output_ols", "per_output_gain", "per_output_ridge_cv")
        print(f"{n:>4} {np.mean([r['meanPred'] for r in rows]):9.4f} "
              + " ".join(f"{np.mean([r[k] for r in rows]):>20.4f}" for k in keys))

    # ---- why does b4_ridge explode at n = 10?
    print("\n=== b4_ridge: penalty actually selected, and the intercept's behaviour ===")
    from baselines import b4_ridge, standardise
    for n in (10, 20):
        f = fold_indices(n, 0, 0)
        tr, te = f
        ysd = Y[tr].std(0).mean()
        mu, _, hp, _ = b4_ridge(prep["X"][tr], Y[tr], prep["X"][te])
        e = np.sqrt(((mu - Y[te]) ** 2).mean()) / ysd
        Ztr, Zte, _ = standardise(prep["X"][tr], prep["X"][te])
        lmax = np.linalg.eigvalsh(np.hstack([Ztr, np.ones((len(Ztr), 1))]).T
                                  @ np.hstack([Ztr, np.ones((len(Ztr), 1))])).max()
        print(f"  n={n}: nrmse {e:.4f}  lam={hp[1]:.3e}  lam/lam_max={hp[1]/lmax:.2e}  "
              f"p={Ztr.shape[1]}")
    print("\nA ridge that is underdetermined at p >> n needs a penalty near lam_max to")
    print("shrink the null-space directions; any grid that stops below that leaves the fit")
    print("with an enormous unpenalised-intercept-driven excursion.")


if __name__ == "__main__":
    main()
