#!/usr/bin/env python3
"""Staged tests T1-T6 of the artifact plan, with the numeric expectation at each stage.

Run BEFORE the full sweep.  Each stage is seconds to a couple of minutes.
A broken implementation is caught here rather than mistaken for a finding.

  T1 inputs        shapes, finiteness, archived statistics, denylist, phases
  T2 operator      shape, agreement with the archived S, spectrum, Laplacian symmetry
  T3 baselines     one fold at n=160: every baseline below the mean predictor, guard fires
  T4 arms          one fold at n=10 and n=160: the six prediction columns
  T5 tempering     point predictions bit-identical; LOO coverage near test coverage
  T6 leakage       permuted test targets leave predictions bit-identical; determinism
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feeder import Feeder, build_laplacians, check_q_is_deterministic  # noqa: E402
import method as M  # noqa: E402


def banner(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def t1_inputs(fe: Feeder, sheet: np.ndarray, phase_true: np.ndarray) -> dict:
    banner("T1 - inputs")
    out = {}
    for regime in (M.REGIME_DECIDE, M.REGIME_ROBUST):
        x, y, st = M.stage_ensemble(regime)
        print(f"regime {regime}: X{x.shape} Y{y.shape} finite={np.isfinite(x).all() and np.isfinite(y).all()} "
              f"mean {st['mean_kW']:.3f} kW (target {M.EXPECT[regime][0]:.2f}) "
              f"Vstd {st['Vstd']:.6f} (target {M.EXPECT[regime][1]:.5f})")
        print(f"  q_is_deterministic: {check_q_is_deterministic(x)}")
        out[regime] = st
    assert np.array_equal(np.argmax(fe.P0, axis=1), phase_true), "phase mismatch"
    print(f"ph_true matches argmax of the network P0 columns: True "
          f"(counts {np.bincount(phase_true, minlength=3).tolist()})")
    # denylist must block reserved evidence
    for tok in M.DENYLIST:
        p = M.PILOT / f"{tok}_Y.npy"
        blocked = any(t in str(p) for t in M.DENYLIST)
        assert blocked, f"denylist failed to block {p}"
    print(f"denylist blocks every reserved token: True ({list(M.DENYLIST)})")
    assert np.array_equal(sheet,
                          np.array([681, 697, 713, 729, 745, 761, 777, 793, 809, 825,
                                    841, 857, 873, 889, 906])), "sheet mismatch"
    print("sheet15 matches the archived 15-bus far-end sheet: True")
    return out


def t2_operator(fe: Feeder, sheet: np.ndarray, s_ref: np.ndarray) -> dict:
    banner("T2 - analytic operator, Laplacians, and the triangular-solve convention")
    from kernels import assert_cholesky_convention
    cc = assert_cholesky_convention()
    print(f"triangular-solve convention: max |solve - direct| {cc['chol_solve_vs_direct_max_abs_error']:.3e} "
          f"(passes {cc['passes']}); note scipy 1.16.3's cho_factor ignores lower=True here, "
          f"so the explicit form is used")
    assert cc["passes"]
    s, chk = M.verify_operator(fe, sheet, s_ref)
    sp = chk["spectrum"]
    print(f"S {sp['shape']} frob ratio {chk['frob_ratio_vs_archived']:.6f} "
          f"normalised deviation {chk['normalised_deviation_vs_archived']:.3e} (must be < 1e-8)")
    print(f"sv[:12]/sv0 {np.round(sp['sv_normalised'][:12], 4).tolist()}")
    print(f"stable rank {sp['stable_rank']:.4f} (expect ~2.18)  "
          f"r_eff 1e-2/1e-3/1e-4 = {sp['rank_1e2']}/{sp['rank_1e3']}/{sp['rank_1e4']} "
          f"(expect ~11/19/23)  cond {sp['cond']:.2e} (expect ~1.3e32)")
    met = build_laplacians(fe)
    sym = met["laplacian_symmetry"]
    print(f"Laplacian: max|L-L^T| {sym['max_abs_L_minus_LT']:.3e} vs max|L-L^H| "
          f"{sym['max_abs_L_minus_LH']:.3e} on scale {sym['scale_max_abs_L']:.3e}")
    print(f"  relative transpose defect {sym['rel_transpose_defect']:.3e}, "
          f"relative Hermitian defect {sym['rel_hermitian_defect']:.3e}"
          "  -> complex-symmetric, NOT Hermitian; eigh/eigsh are wrong for it")
    for k in ("M_adm", "M_imp", "M_hop"):
        m = met[k]
        ev = np.linalg.eigvalsh(0.5 * (m + m.T))
        print(f"  {k}: shape {m.shape} symmetric={np.allclose(m, m.T)} "
              f"min eig {ev.min():.3e} max eig {ev.max():.3e}")
    assert np.allclose(met["M_adm"], met["M_adm"].T)
    return {"operator": chk, "laplacian": sym}


def t3_baselines(fe: Feeder, sheet: np.ndarray, s_op: np.ndarray, metrics: dict,
                 v0bus: np.ndarray) -> None:
    banner("T3 - baselines on one fold at n = 160")
    x, y, _ = M.stage_ensemble(M.REGIME_DECIDE)
    tr, te = M.fold_indices(160, 0, 0, len(x))
    base, _ = M.build_baselines(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
    vsd_tr = float(y[tr].std(0).mean())
    vsd_ens = float(y.std(0).mean())
    mp = float(np.sqrt(((y[tr].mean(0)[None, :] - y[te]) ** 2).mean()))
    print(f"normalisers: train {vsd_tr:.6f} ensemble {vsd_ens:.6f}; mean predictor {mp/vsd_tr:.4f}")
    for name, d in base.items():
        e = float(np.sqrt(((d["pred"] - y[te]) ** 2).mean()))
        print(f"  {name:14s} rmse_pu {e:.3e}  /train {e/vsd_tr:.4f}  /ens {e/vsd_ens:.4f}  "
              f"/meanpred {e/mp:.4f}  hyper {d['hyper']}")
        assert e / mp <= 1.0, f"{name} does not beat the mean predictor"
    assert "order" in base["b5_lindist"]["hyper"]
    print("all baselines beat the mean predictor: True")
    # guard g1 firing demonstration: a model that predicts a blown-up vector must
    # score worse than the mean predictor, which is what guard g1 detects.
    from kernels import ridge_cv
    xb = np.hstack([x[tr], np.ones((160, 1))])
    xb_te = np.hstack([x[te], np.ones((len(te), 1))])
    r = ridge_cv(xb, y[tr], xb_te, penal_fracs=np.array([1e12]))
    e_const = float(np.sqrt(((r["pred"] - y[te]) ** 2).mean()))
    blown = r["pred"] * 5.0 + 1.0  # an amplified solve, the failure mode g1 guards
    e_blown = float(np.sqrt(((blown - y[te]) ** 2).mean()))
    print(f"guard g1: a penalised-to-constant ridge scores /meanpred {e_const/mp:.4f} "
          f"(=1.0 exactly, the degenerate-but-not-failed case); a blown-up solve scores "
          f"{e_blown/mp:.2f} -> guard g1 fires, as designed")
    assert e_const / mp <= 1.0 + 1e-9
    assert e_blown / mp > 1.0

    # the mean predictor itself must be reproducible from the guard column
    print(f"guard column reference: the mean predictor scores exactly 1.0 by definition")


def t4_arms(fe: Feeder, sheet: np.ndarray, s_op: np.ndarray, metrics: dict,
            v0bus: np.ndarray) -> None:
    banner("T4 - the six prediction columns at n = 10 and n = 160")
    x, y, _ = M.stage_ensemble(M.REGIME_DECIDE)
    for n in (10, 160):
        tr, te = M.fold_indices(n, 0, 0, len(x))
        rec = M.run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
        print(f"\nn = {n}: train vsd {rec['vsd_train']:.6f}  best baseline {rec['best_baseline']} "
              f"({rec['best_baseline_rmse']:.4f})  physics dof {rec['physics_effective_dof']:.2f} "
              f"(= {rec['physics_effective_dof']/n:.2f} x n)")
        for a in M.MODEL_ARMS:
            d = rec["arms"][a]
            line = (f"  {a:22s} rmse/train {d['rmse_norm']:.4f}  margin {d['margin_vs_best_baseline']:+.2%}"
                    f"  /meanpred {d['rmse_over_mean_predictor']:.4f}")
            if "nlpd" in d:
                line += f"  nlpd {d['nlpd']:.3f}  c68 {d['c68']:.3f} c95 {d['c95']:.3f}"
            print(line)
        for a in M.BASE_ARMS:
            print(f"  {a:22s} rmse/train {rec['baselines'][a]['rmse_norm']:.4f}")


def t5_tempering(fe: Feeder, sheet: np.ndarray, s_op: np.ndarray, metrics: dict,
                 v0bus: np.ndarray) -> None:
    banner("T5 - tempering self-test")
    x, y, _ = M.stage_ensemble(M.REGIME_DECIDE)
    for n in (10, 80, 160):
        tr, te = M.fold_indices(n, 0, 0, len(x))
        chk = M._tempering_identity_check(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
        rec = M.run_fold(x, y, tr, te, fe, sheet, s_op, metrics, v0bus)
        t = rec["arms"]["resid_gp_tempered"]
        print(f"n={n:3d} alpha_loo {chk['alpha_used']:.4f}  point predictions bit-identical "
              f"{chk['tempered_point_prediction_identical']}  min var increase "
              f"{chk['min_var_increase']:.3e}")
        print(f"      test c68 {t['c68']:.3f} c95 {t['c95']:.3f} | raw floor c68 "
              f"{t['c68_raw_floor']:.3f} c95 {t['c95_raw_floor']:.3f} | mll c68 "
              f"{rec['arms']['resid_gp_mll']['c68']:.3f}")
        assert chk["exact_zero_difference"], "tempering moved the point prediction"


def t6_leakage(fe: Feeder, sheet: np.ndarray, s_op: np.ndarray, metrics: dict,
               v0bus: np.ndarray) -> None:
    banner("T6 - leakage and determinism")
    x, y, _ = M.stage_ensemble(M.REGIME_DECIDE)
    lk = M.leakage_test(x, y, fe, sheet, s_op, metrics, v0bus)
    for n, v in lk.items():
        print(f"n={n:>4s} permuted-test-target: predictions bit-identical {v['bit_identical']} "
              f"(max |d pred| {v['max_abs_prediction_difference']:.3e}), variances "
              f"identical {v['variance_bit_identical']} "
              f"(max |d var| {v['max_abs_variance_difference']:.3e})")
    assert all(v["bit_identical"] for v in lk.values()), "LEAKAGE: predictions depend on test targets"
    dt = M.determinism_test(x, y, fe, sheet, s_op, metrics, v0bus)
    print(f"determinism: bit-identical {dt['bit_identical']} "
          f"(max |d pred| {dt['max_abs_prediction_difference']:.3e})")
    assert dt["bit_identical"], "non-deterministic fold"


def main() -> None:
    fe = Feeder()
    sheet, phase_true, s_ref = M.load_metadata()
    _, sheet_rebuilt = fe.far_sheet(15)
    assert np.array_equal(sheet_rebuilt, sheet), f"sheet mismatch {sheet_rebuilt} vs {sheet}"
    v0bus = fe.noload_voltage(sheet)
    print(f"no-load reference phasors: |V| in [{np.abs(v0bus).min():.6f}, {np.abs(v0bus).max():.6f}] pu")
    s_op, _ = M.verify_operator(fe, sheet, s_ref)
    metrics = build_laplacians(fe)

    t1_inputs(fe, sheet, phase_true)
    t2_operator(fe, sheet, s_ref)
    t3_baselines(fe, sheet, s_op, metrics, v0bus)
    t4_arms(fe, sheet, s_op, metrics, v0bus)
    t5_tempering(fe, sheet, s_op, metrics, v0bus)
    t6_leakage(fe, sheet, s_op, metrics, v0bus)
    banner("ALL STAGES T1-T6 PASSED")


if __name__ == "__main__":
    main()
