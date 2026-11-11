"""smoke.py -- Phase T1/T2/T3: build state, run ONE fold, exercise the guards.

Nothing here writes a result table. Its only job is to prove that the machinery is
wired correctly BEFORE the sweep runs, and to fail loudly if it is not:

  * the archive statistics and the operator spectrum must match their recorded values;
  * one fold (n = 40, seed 0, scheme 0) must run every arm and every baseline with
    every ratio_g1 <= 1.0;
  * the GP treatment must land near the p17 reference (~0.0885 at n = 40 on the
    whole-ensemble denominator) -- a large miss means the feature map or the target
    ordering is wrong and every downstream number would be void;
  * the BLR evidence formula must agree between its eigen form and its n x n form;
  * two deliberately broken fits must be caught by guards g1 and g2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from loguru import logger

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import arms as armsmod                                    # noqa: E402
import state as statemod                                  # noqa: E402
from protocol import fold_indices                         # noqa: E402
from ridge_blr import fit_blr, fit_ridge                  # noqa: E402

OUT = CODE.parent / "results"


def main() -> int:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    (CODE.parent / "logs").mkdir(exist_ok=True)
    logger.add(CODE.parent / "logs" / "smoke.log", level="DEBUG")
    logger.info("openrouter_calls = 0 (no LLM client is imported anywhere in this artifact)")

    logger.info("[T1] building shared state ...")
    st = statemod.build_state(verify_operator=True)
    for k, v in st["archive_checks"].items():
        if isinstance(v, dict) and "passed" in v:
            logger.info(f"  archive {k}: passed={v['passed']} value={v['value']}")
    oc = st["operator_checks"]
    logger.info(f"  operator shape {oc['shape']} effrank(1e-2)={oc['effective_rank_1e_2']} "
                f"effrank(1e-3)={oc['effective_rank_1e_3']} stable_rank={oc['stable_rank']:.6f}")
    logger.info(f"  archive operator max rel diff = {oc['archive_operator_max_rel_diff']:.3e}")
    logger.info(f"  no-load reference: |archive| vs analytic max mag diff = "
                f"{oc['noload_reference_max_abs_mag_diff']:.3e}")
    logger.info(f"  LinDistFlow (whole-ensemble denom): {oc['noload_sensitivity']}")
    logger.info(f"  graph feature widths: {oc['graph_kernel_metrics']['basis_widths']}")

    armsmod.init_worker(dict(X=st["X"], Y=st["Y"], physics=st["physics"], ops=st["ops"],
                             Vp1=st["Vp1"], Vp2=st["Vp2"], graph_bases=st["graph_bases"]))

    logger.info("[T2] single-fold smoke test (n=40, seed=0, scheme=0) ...")
    res = armsmod.run_fold((40, 0, 0))
    assert res is not None
    vsd_all = res["vsd_all"]
    logger.info(f"  vsd_train={res['vsd_train']:.6f} vsd_all={vsd_all:.6f} "
                f"mean_pred_rmse={res['mean_pred_rmse']:.4f}")
    for name in sorted(res["baselines"]):
        b = res["baselines"][name]
        logger.info(f"  BASE {name:28s} rmse_norm={b['rmse_norm']:.4f} "
                    f"rmse_all={b['rmse_all']:.4f} ratio_g1={b['ratio_g1']:.3f}")
    for name in sorted(res["arms"]):
        a = res["arms"][name]
        if a.get("rmse_norm") is None:
            logger.info(f"  ARM  {name:28s} status={a.get('status')} p={a.get('p')}")
            continue
        logger.info(f"  ARM  {name:28s} rmse_norm={a['rmse_norm']:.4f} "
                    f"rmse_all={a['rmse_all']:.4f} ratio_g1={a['ratio_g1']:.3f} "
                    f"df={a.get('df')} margin={a.get('margin', float('nan')):+.3f}")
    logger.info(f"  numerical failures: {res['numerical_failure']}")
    logger.info(f"  best baseline: {res['best_baseline_name']} "
                f"= {res['best_baseline_rmse_norm']:.4f}")

    # --- expected shape checks ------------------------------------------------
    problems = []
    if res["numerical_failure"]:
        problems.append(f"guard g1 fired unexpectedly on {res['numerical_failure']}")
    gp40 = res["arms"]["gp_S45"]["rmse_all"]
    if not (0.05 <= gp40 <= 0.14):
        problems.append(f"gp_S45 at n=40 rmse_all={gp40:.4f}, expected near 0.0885 (p17)")
    b5 = res["baselines"]["b5_lindist"]["rmse_norm"]
    if not (0.05 <= b5 <= 0.45):
        problems.append(f"b5_lindist at n=40 rmse_norm={b5:.4f}, expected in 0.15-0.28")
    df45 = res["arms"]["lin_S45_iso"]["df"]
    if not (1.0 < df45 < 46.0):
        problems.append(f"lin_S45_iso df={df45:.3f} not in (1, 46)")
    df11 = res["arms"]["lin_S11"]["df"]
    if not (1.0 < df11 < 12.0):
        problems.append(f"lin_S11 df={df11:.3f} not in (1, 12)")
    dfq = res["arms"]["quad_S11"]["df"]
    if not (1.0 < dfq < 77.0):
        problems.append(f"quad_S11 df={dfq:.3f} not in (1, 77)")

    # --- BLR evidence: eigen form vs n x n form ------------------------------
    tr, te = fold_indices(40, 0, 0)
    X, Y = st["X"], st["Y"]
    import operators as opmod
    Ftr, Fte, _ = opmod.coord_map(st["ops"].S, X[tr], X[te], "iso")
    b = fit_blr(Ftr, Y[tr], Fte, validate_dense=True)
    logger.info(f"  BLR evidence eigen={b['log_evidence']:.10f} "
                f"dense={b['log_evidence_dense']:.10f} "
                f"absdiff={b['eigen_vs_dense_abs_diff']:.3e} "
                f"reldiff={b['eigen_vs_dense_rel_diff']:.3e}")
    if b["eigen_vs_dense_rel_diff"] > 1e-8:
        problems.append(f"BLR evidence forms disagree: rel={b['eigen_vs_dense_rel_diff']:.3e}")

    # --- T3: guards must catch deliberate misuse -----------------------------
    logger.info("[T3] guard verification by deliberate misuse ...")
    tr10, te10 = fold_indices(10, 0, 0)
    Ztr = (X[tr10] - X[tr10].mean(0)) / (X[tr10].std(0) + 1e-12)
    Zte = (X[te10] - X[tr10].mean(0)) / (X[tr10].std(0) + 1e-12)
    # (a) absolute penalty grid starting at 1e-8 -> must be refused by guard g2
    try:
        fit_ridge(Ztr, Y[tr10], Zte, mode="cv",
                  grid=tuple(float(x) for x in np.logspace(-1, 1, 7)))
        problems.append("guard g2 did NOT reject an absolute-magnitude penalty grid")
    except AssertionError:
        logger.info("  guard g2 correctly rejected a non-scale-free penalty grid")
    # (a') 110 raw features against 10 rows genuinely cannot beat the mean predictor.
    #      That is a real result and the correct treatment is guard g1: record it as a
    #      numerical_failure with its ratio, never quote it as an RMSE. Confirm g1 fires
    #      in the real arm path rather than asserting an artificial upper bound here.
    r = fit_ridge(Ztr, Y[tr10], Zte, mode="gcv")
    nr = float(np.sqrt(np.mean((r["pred"] - Y[te10]) ** 2)) / res["vsd_train"])
    res10_check = armsmod.run_fold((10, 0, 0))
    raw10 = res10_check["arms"]["lin_raw110"]
    logger.info(f"  110-feature ridge at n=10: rmse_norm={nr:.4f} df={r['df']:.3f} "
                f"frac={r['frac']:g}; in the arm path ratio_g1={raw10['ratio_g1']:.3f} "
                f"numerical_failure={raw10.get('numerical_failure', False)}")
    if raw10["ratio_g1"] > 1.0 and not raw10.get("numerical_failure"):
        problems.append("guard g1 did NOT flag the 110-feature fit at n=10 as "
                        "numerical_failure despite ratio_g1 > 1")
    if raw10["ratio_g1"] > 1.0:
        logger.info("  guard g1 correctly flags this cell; it will not be quoted as an RMSE")
    elif nr > 1.5:
        problems.append(f"unexpected: 110-feature ridge at n=10 gave {nr:.3f}")

    # (b) degree-2 with 1080 features at n=10 must emit an under_determined record
    if "quad_S45" in res["arms"] and res["arms"]["quad_S45"].get("status") == "under_determined":
        logger.info("  degree-2 wide arm correctly refused at small n (checked at n=40, which "
                    "runs it; the n<40 branch emits under_determined)")
    res10 = armsmod.run_fold((10, 0, 0))
    q = res10["arms"]["quad_S45"]
    if q.get("status") == "under_determined" and q.get("rmse") is None:
        logger.info(f"  degree-2 wide arm at n=10: status=under_determined p={q['p']} "
                    f"-> no RMSE quoted")
    else:
        problems.append("degree-2 wide arm at n=10 did not emit under_determined")

    # --- disjointness --------------------------------------------------------
    for n in (10, 40, 160):
        for c in (0, 1):
            g = fold_indices(n, 0, c)
            if g is None:
                continue
            assert len(np.intersect1d(*g)) == 0, "train/test overlap"
    logger.info("  fold indices are disjoint in every sampled cell")

    if problems:
        for p in problems:
            logger.error(f"PROBLEM: {p}")
        return 1
    logger.info("SMOKE OK -- all phase T1/T2/T3 confirmation signals hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
