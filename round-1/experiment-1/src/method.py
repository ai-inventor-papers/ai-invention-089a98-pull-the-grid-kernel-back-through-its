#!/usr/bin/env python3
"""Output-restricted sensitivity-pullback kernel for unbalanced LV distribution
power flow: the shared screen protocol, its mechanism factorial, and its controls.

WHAT THIS SCRIPT IS
    A complete experimental system for the hypothesis

        k(x, x') = exp( -||S (x - x')||^2 / (2 ell^2) ),   S = d|V_far|/dx   (45 x 110)

    built analytically from the feeder's line sequence impedances and the 55
    single-phase loads' phase labels, with ZERO AC solve in the operator build.
    It is run on the IEEE PES European LV test feeder (pandapower
    ieee_european_lv_asymmetric('on_peak_566')) with ground truth from runpp_3ph.

WHAT IT COMPARES AGAINST (the canonical pool that decides the verdicts)
    b1  isotropic RBF Gaussian process on per-column-standardised injections
    b2  hop-count graph-Laplacian kernel (two constructions, better kept)
    b3  admittance-weighted graph-Laplacian kernel
    b4  ridge on standardised injections, scale-free penalty grid, train-only CV
    b5  multiphase LinDistFlow order 2 with the Z-bus current correction, per-output
        ridge calibration, 16-cell factorial chosen by train-only cross-validation
    plus, in an extended pool that does not decide the verdicts: a small MLP, an
    ARD-RBF with per-dimension lengthscales, the impedance-weighted Laplacian and
    the unrestricted 55-dimensional Z-bus metric.

STAGES (run in this order; a truncated run still produces a reportable result)
    0  inputs, feeder, operator build and the identity / cross-check suite
    1  the screen at n = 80 on the 2.0x regime (the pre-registered ranking point)
    2  the full learning curve on the 2.0x regime
    3  the 1.0x secondary regime
    4  the mechanism factorial (metric x target sheet)
    5  the attribution controls
    6  the extended-pool baselines
    7  the train-only-CV hyperparameter arm (a selection-rule robustness check)
    8  aggregation into results/*.json and method_out.json

Everything that can leak test information is computed from training rows only:
centring, feature scaling, target standardisation, the reference point, every
hyperparameter, and the LinDistFlow calibration cell.  The leakage check recomputes
folds with the test block replaced by a disjoint dummy block and requires the
selected hyperparameters and the training predictions to be identical.

USAGE
    uv run method.py                 # everything
    uv run method.py --stages 0,1    # a prefix, for staged scaling
    uv run method.py --workers 2     # fold-level process parallelism
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                  # noqa: E402
from loguru import logger                                           # noqa: E402

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
sys.path.insert(0, str(CODE))

RESULTS = HERE / "results"
LOGS = HERE / "logs"
FIGS = HERE / "figures"
for d in (RESULTS, LOGS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "run.log", rotation="30 MB", level="DEBUG")

REGIME_PRIMARY = "2.0"
REGIME_SECONDARY = "1.0"
NS_FULL = (10, 20, 40, 80, 160)
NS_RANKING = (80,)


# ------------------------------------------------------------------ utilities
def _dump(name: str, obj) -> None:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, default=_json_default))
    logger.info(f"wrote {p.relative_to(HERE)} ({p.stat().st_size/1024:.0f} KiB)")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _pool(workers: int):
    import multiprocessing as mp
    from worker import init_worker
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=mp.get_context("spawn"),
                               initializer=init_worker)


def _run_batch(tasks, fn, workers, label):
    t0 = time.time()
    out = []
    if workers <= 1:
        from worker import init_worker
        init_worker()
        for i, t in enumerate(tasks):
            out.append(fn(t))
            if (i + 1) % 10 == 0:
                logger.info(f"  {label}: {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")
    else:
        with _pool(workers) as pool:
            for i, r in enumerate(pool.map(fn, tasks, chunksize=1)):
                out.append(r)
                if (i + 1) % 10 == 0:
                    logger.info(f"  {label}: {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")
    logger.info(f"  {label}: done {len(out)} tasks in {time.time()-t0:.0f}s")
    return out


def _fold_tasks(regime, ns, seeds, schemes, **extra):
    from folds import make_fold, fold_fingerprint
    out = []
    for n in ns:
        for seed in seeds:
            for scheme in schemes:
                try:
                    tr, te = make_fold(n, seed, scheme)
                except AssertionError:
                    continue
                out.append(dict(regime=regime, n=int(n), seed=int(seed),
                                scheme=int(scheme), train=tr, test=te,
                                fingerprint=fold_fingerprint(tr, te), **extra))
    return out


# ---------------------------------------------------------------- stage 0
def stage0_inputs(workers: int) -> dict:
    """Feeder facts, operator builds, the identity checks, the FD cross-check."""
    from feeder import Physics
    from data import load_ensemble, load_sheet_and_phases, load_exact_jacobian, \
        load_near_sheet
    from operators import (build_operator, spectral_facts, structural_cross_check,
                           data_whitened_diagnostics, differential_mode_share,
                           laplacian_is_hermitian, shuffle_effectiveness,
                           shuffled_phase_operator)
    from env_checks import (identity_checks, transformer_convention_sweep,
                            near_operator, all90_operator, all55_operator)

    ph = Physics()
    facts = ph.structural_facts()
    logger.info(f"feeder facts: {facts}")

    sheet = load_sheet_and_phases(ph)
    logger.info(f"sheet15 {sheet['sheet15'].tolist()} source={sheet['sheet_source']}")

    ens2 = load_ensemble(REGIME_PRIMARY)
    ens1 = load_ensemble(REGIME_SECONDARY)
    logger.info(f"2.0x provenance={ens2['provenance']} {ens2['checks']}")
    logger.info(f"1.0x provenance={ens1['provenance']} {ens1['checks']}")

    near = load_near_sheet()
    jac = load_exact_jacobian()

    # ---- operators
    s15 = sheet["sheet15"]
    S_far = build_operator(ph, s15, ph.phase, "pu_correct", "noload")
    S_near = near_operator(ph, near if near["available"] else None)
    S_full = build_operator(ph, ph.buses, ph.phase, "pu_correct", "noload")
    S_all90 = all90_operator(ph)
    S_all55 = all55_operator(ph)
    logger.info(f"operators: far {S_far.shape} near {None if S_near is None else S_near.shape} "
                f"full {S_full.shape} all90 {S_all90.shape} all55 {S_all55.shape}")

    spec = dict(far=spectral_facts(S_far), full=spectral_facts(S_full),
                all90=spectral_facts(S_all90), all55=spectral_facts(S_all55))
    if S_near is not None:
        spec["near"] = spectral_facts(S_near)
    for k, v in spec.items():
        logger.info(f"spectrum {k}: eff ranks {[v[f'eff_rank_{t:g}'] for t in (1e-2,1e-3,1e-4)]} "
                    f"stable {v['stable_rank']:.2f}")

    # ---- checks
    ident = identity_checks(ph, s15)
    _id_keys = ("scalar_rel", "block_rel_none", "block_rel_pu_correct", "block_rel_legacy")
    logger.info("identity: " + json.dumps({k: ident[k] for k in _id_keys if k in ident}))
    conv = transformer_convention_sweep(ph, s15, jac) if jac["available"] else {
        "available": False}
    if jac["available"]:
        logger.info(f"FD cross-check best cell: {conv['best_cell']} rel "
                    f"{conv['cells'][conv['best_cell']]['rel_rescaled']:.4f}")
    arch = None
    from data import PILOT
    ap = PILOT / "S_sheet15.npy"
    if ap.exists():
        from operators import structural_cross_check as scc
        arch = scc(S_far, np.load(ap))
        logger.info(f"structural cross-check vs archived: support_match="
                    f"{arch['support_pattern_match']} corr={arch['pearson']:.3f}")

    # ---- diagnostics and PC7 corrections
    diag = {reg: data_whitened_diagnostics(S_far, e["X"])
            for reg, e in ((REGIME_PRIMARY, ens2), (REGIME_SECONDARY, ens1))}
    dmode = {reg: differential_mode_share(e["Y"])
             for reg, e in ((REGIME_PRIMARY, ens2), (REGIME_SECONDARY, ens1))}
    lap = laplacian_is_hermitian(ph, ph.phase)

    # ---- the shuffled-phase operator and its effectiveness
    S_shuf, phase_shuf = shuffled_phase_operator(ph, s15, ph.phase,
                                                 "pu_correct", "noload", seed=0)
    shuf = shuffle_effectiveness(S_far, S_shuf)
    logger.info(f"phase shuffle: changed {shuf['n_columns_changed']}/110 columns, "
                f"frob rel {shuf['frobenius_rel_diff']:.3f}")
    # the archived attempt permuted ROWS; show that it is a no-op
    from operators import random_operator_like
    perm_rows = np.random.default_rng(0).permutation(S_far.shape[0])
    row_perm_effect = float(np.linalg.norm(S_far[perm_rows] - S_far) /
                            np.linalg.norm(S_far))
    from screen import metric_features
    Xt = ens2["X"][:40]
    F1, _, _, _ = metric_features(S_far, Xt, Xt)
    F2, _, _, _ = metric_features(S_far[perm_rows], Xt, Xt)
    d_row = float(np.sqrt(np.mean((np.einsum("ij,ij->i", F1, F1) -
                                   np.einsum("ij,ij->i", F2, F2)) ** 2)))
    np.save(RESULTS / "phase_shuffled_assignment.npy", phase_shuf)
    np.save(RESULTS / "S_far.npy", S_far)
    np.save(RESULTS / "S_full.npy", S_full)
    np.save(RESULTS / "S_all90.npy", S_all90)
    np.save(RESULTS / "S_all55.npy", S_all55)
    if S_near is not None:
        np.save(RESULTS / "S_near.npy", S_near)

    out = dict(
        feeder_facts=facts,
        sheet15=[int(b) for b in s15],
        sheet_source=sheet["sheet_source"],
        phase_source=sheet["phase_source"],
        archived_sheet_agrees=sheet["archived_sheet_agrees"],
        archived_phase_agrees=sheet["archived_phase_agrees"],
        ensemble_2_0=dict(provenance=ens2["provenance"], **ens2["checks"]),
        ensemble_1_0=dict(provenance=ens1["provenance"], **ens1["checks"]),
        near_sheet=dict(available=near["available"], shape=near.get("shape"),
                        n_finite=near.get("n_finite"),
                        buses=None if not near["available"] else [int(b) for b in near["buses"]]),
        exact_jacobian=dict(available=jac["available"]),
        spectra=spec,
        identity=ident,
        transformer_convention=conv,
        structural_cross_check_vs_archive=arch,
        operator_row_permutation_is_noop=dict(
            frobenius_rel_diff=float(row_perm_effect),
            note="permuting the ROWS of S cannot change ||S(x-x')||; the max change in "
                 "the pairwise feature norms is machine zero, which is why the archived "
                 "row-permutation control returned an identical number",
            max_abs_change_in_feature_sqnorm=d_row),
        shuffled_phase_assignment=shuf,
        data_whitened=diag,
        differential_mode=dmode,
        laplacian=lap,
    )
    _dump("stage0_inputs.json", out)
    return dict(S_far=S_far, S_near=S_near, S_full=S_full, S_all90=S_all90,
                S_all55=S_all55, s15=s15, ph=ph, ens2=ens2, ens1=ens1,
                near=near, jac=jac, stage0=out)


# ---------------------------------------------------------------- stage 1/2/3
def run_screen(regime, ns, workers, select="mll", tag=""):
    tasks = _fold_tasks(regime, ns, range(12), (0, 1), select=select,
                        do_b5=True, do_graph=True)
    logger.info(f"screen {regime}x select={select} {tag}: {len(tasks)} folds")
    from worker import screen_task
    recs = _run_batch(tasks, screen_task, workers, f"screen {regime}x")
    return recs


def aggregate_screen(recs, regime) -> dict:
    from screen import summarise_folds, bootstrap_ci, wilcoxon_p, crossover_curve
    by_n = {}
    for r in recs:
        by_n.setdefault(int(r["n"]), []).append(r)
    summary = {}
    for n, folds in sorted(by_n.items()):
        s = summarise_folds(folds)
        s["bootstrap_ci_margin"] = bootstrap_ci(np.array(s["per_fold_margin"]))
        s["wilcoxon_margin"] = wilcoxon_p(np.array(s["per_fold_margin"]))
        if s["per_fold_margin_vs_b5"]:
            s["bootstrap_ci_margin_vs_b5"] = bootstrap_ci(
                np.array(s["per_fold_margin_vs_b5"]))
            s["wilcoxon_margin_vs_b5"] = wilcoxon_p(
                np.array(s["per_fold_margin_vs_b5"]))
        summary[n] = s
        logger.info(
            f"[{regime}x] n={n:3d} treat {s['mean_far_end_rmse']:.4f} "
            f"margin {100*s['mean_margin_vs_best_baseline']:+6.1f}% "
            f"worst {100*s['worst_fold_margin']:+6.1f}% "
            f"frac>=20% {s['frac_folds_clearing_20pct']:.2f} "
            f"| vs b5 {100*s.get('mean_margin_vs_lindistflow', float('nan')):+6.1f}% "
            f"| c68 {s['mean_coverage68']:.2f} c95 {s['mean_coverage95']:.2f} "
            f"nlpd {s['mean_nlpd']:.2f}")
        for k in ("b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow"):
            if f"mean_rmse_{k}" in s:
                logger.info(f"        {k:16s} {s[f'mean_rmse_{k}']:.4f} "
                            f"(/mean {s[f'guard_g1_{k}']:.3f})")
    return dict(regime=regime, n_summary={int(k): v for k, v in summary.items()},
                crossovers=crossover_curve(summary),
                per_fold=recs)


# ---------------------------------------------------------------- stage 4
def run_mechanism(workers, ctx) -> dict:
    from operators import truncate_rank, partial_whiten, spectral_facts
    from fold_utils import mechanism_cells
    S_far, S_full, S_near = ctx["S_far"], ctx["S_full"], ctx["S_near"]
    S_all90, S_all55 = ctx["S_all90"], ctx["S_all55"]
    k_far = spectral_facts(S_far)["eff_rank_0.01"]
    cells = []
    metrics = {"S_far": S_far, "S_full": S_full,
               "S_full_r": truncate_rank(S_full, k_far),
               "S_all90": S_all90, "S_all55": S_all55}
    if S_near is not None:
        metrics["S_near"] = S_near
    for name, S in metrics.items():
        cells += mechanism_cells(name, REGIME_PRIMARY, NS_FULL, ("far",), S)
    if S_near is not None:
        # the near-end target sheet is its OWN 120-draw AC ensemble, so its folds are
        # drawn from its own pool with the same seed/scheme recipe
        from data import load_near_sheet
        n_rows = len(load_near_sheet()["Y"])
        for name in ("S_far", "S_near"):
            cells += mechanism_cells(name, REGIME_PRIMARY, (10, 20, 40, 80),
                                     ("near",), metrics[name], kind="near_targets",
                                     n_draws=n_rows)
    logger.info(f"mechanism: {len(cells)} fold cells (rank-matched k={k_far})")
    # GUARD: refuse to run a factorial in which different metrics carry the same
    # operator array.  An earlier version of this pipeline dropped the operator from
    # the task payload and every mechanism row came out numerically identical --
    # the 'feature map reused by reference' signature the artifact plan names.
    from fold_utils import assert_metrics_differ
    dup = assert_metrics_differ(cells)
    if dup["collisions"]:
        raise AssertionError(f"mechanism metrics share an operator array: {dup}")
    logger.info(f"mechanism operator guard: {dup['n_distinct_operator_shapes_and_norms']} "
                f"distinct operators, {len(cells)} cells")
    from worker import mechanism_task
    recs = _run_batch(cells, mechanism_task, workers, "mechanism")

    # rank truncation and partial whitening of the primary treatment
    trunc = []
    for k in (3, 5, 8, 10, 11, 15, 20, 45):
        Sk = truncate_rank(S_far, k)
        trunc += mechanism_cells(f"S_far_k{k}", REGIME_PRIMARY, (20, 40, 80, 160),
                                 ("far",), Sk)
    whiten = []
    for g in (0.5, 1.0):
        Sg = partial_whiten(S_far, g)
        whiten += mechanism_cells(f"S_far_gamma{g}", REGIME_PRIMARY, (20, 40, 80, 160),
                                  ("far",), Sg)
    logger.info(f"truncation {len(trunc)} + whitening {len(whiten)} fold cells")
    recs += _run_batch(trunc, mechanism_task, workers, "truncation")
    recs += _run_batch(whiten, mechanism_task, workers, "whitening")

    return dict(rank_matched_k=int(k_far), cells=recs)


# ---------------------------------------------------------------- stage 5
def run_controls(workers, ctx) -> dict:
    from fold_utils import control_cells, aggregate_controls
    cells = []
    for c in ("shuffled_phase", "random_operator", "rotated_injections"):
        cells += control_cells(c, REGIME_PRIMARY, (20, 40, 80, 160))
    logger.info(f"controls: {len(cells)} fold cells")
    from worker import control_task
    recs = _run_batch(cells, control_task, workers, "controls")

    # (d) per-phase aggregate features, (e) PCA / random projection of width 11
    from worker import mechanism_task
    agg = []
    for kind in ("phase_aggregate_6", "phase_aggregate_9", "pca_11", "randproj_11"):
        agg += control_cells(kind, REGIME_PRIMARY, (20, 40, 80, 160))
    logger.info(f"reduction controls: {len(agg)} fold cells")
    recs2 = _run_batch(agg, mechanism_task, workers, "reduction")
    return dict(cells=recs, reduction_cells=recs2,
                summary=aggregate_controls(recs + recs2))


# ---------------------------------------------------------------- stage 6
def run_extended(workers) -> dict:
    from worker import extra_baseline_task
    t_mlp = _fold_tasks(REGIME_PRIMARY, (80,), range(12), (0, 1), baseline="b6_mlp")
    t_ard = _fold_tasks(REGIME_PRIMARY, (20, 40, 80), range(12), (0, 1), baseline="b7_ard")
    logger.info(f"extended pool: {len(t_mlp)} MLP + {len(t_ard)} ARD folds")
    mlp = _run_batch(t_mlp, extra_baseline_task, workers, "mlp")
    ard = _run_batch(t_ard, extra_baseline_task, workers, "ard")
    return dict(mlp=mlp, ard=ard)


# ---------------------------------------------------------------- stage 8
def stage_checks(ctx, screen_p, screen_s) -> dict:
    from env_checks import (determinism_and_leakage_checks, reserved_evidence_audit,
                            baseline_sanity)
    ph, s15 = ctx["ph"], ctx["s15"]
    det = determinism_and_leakage_checks(ph, s15, ctx["ens2"])
    res = reserved_evidence_audit()
    san = baseline_sanity(ph, s15, ctx["ens2"])
    logger.info(f"leakage check: {det['identical_hyperparameters']} "
                f"determinism: {det['deterministic']}")
    return dict(determinism_and_leakage=det, reserved_evidence=res, baseline_sanity=san)


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-stage0", action="store_true")
    args = ap.parse_args()
    stages = {int(s) for s in args.stages.split(",")}
    logger.info(f"stages {sorted(stages)} workers {args.workers}")

    t_start = time.time()
    ctx = stage0_inputs(args.workers) if 0 in stages else _reload_ctx()
    ph, s15 = ctx["ph"], ctx["s15"]

    screen_p = screen_s = None
    path_p = RESULTS / "screen_2.0x.json"
    path_s = RESULTS / "screen_1.0x.json"

    if 1 in stages:
        recs = run_screen(REGIME_PRIMARY, NS_RANKING, args.workers)
        _dump("screen_2.0x_n80_only.json", aggregate_screen(recs, REGIME_PRIMARY))
    if 2 in stages:
        recs = run_screen(REGIME_PRIMARY, NS_FULL, args.workers)
        screen_p = aggregate_screen(recs, REGIME_PRIMARY)
        _dump("screen_2.0x.json", screen_p)
        gc.collect()
    elif path_p.exists():
        screen_p = json.loads(path_p.read_text())
    if 3 in stages:
        recs = run_screen(REGIME_SECONDARY, NS_FULL, args.workers)
        screen_s = aggregate_screen(recs, REGIME_SECONDARY)
        _dump("screen_1.0x.json", screen_s)
        gc.collect()
    elif path_s.exists():
        screen_s = json.loads(path_s.read_text())

    mech = ctrl = ext = sel = None
    if 4 in stages:
        mech = run_mechanism(args.workers, ctx)
        _dump("mechanism.json", mech)
        gc.collect()
    elif (RESULTS / "mechanism.json").exists():
        mech = json.loads((RESULTS / "mechanism.json").read_text())
    if 5 in stages:
        ctrl = run_controls(args.workers, ctx)
        _dump("controls.json", ctrl)
        gc.collect()
    elif (RESULTS / "controls.json").exists():
        ctrl = json.loads((RESULTS / "controls.json").read_text())
    if 6 in stages:
        ext = run_extended(args.workers)
        _dump("extended_pool.json", ext)
        gc.collect()
    elif (RESULTS / "extended_pool.json").exists():
        ext = json.loads((RESULTS / "extended_pool.json").read_text())
    if 7 in stages:
        recs = run_screen(REGIME_PRIMARY, NS_FULL, args.workers, select="cv",
                          tag="(hyperparameters by train-only CV)")
        sel = aggregate_screen(recs, REGIME_PRIMARY)
        _dump("screen_2.0x_cv_selected.json", sel)
        gc.collect()
    elif (RESULTS / "screen_2.0x_cv_selected.json").exists():
        sel = json.loads((RESULTS / "screen_2.0x_cv_selected.json").read_text())

    checks = stage_checks(ctx, screen_p, screen_s)
    _dump("protocol_checks.json", checks)
    _dump("operator_checks.json", dict(
        spectra=ctx["stage0"]["spectra"],
        identity=ctx["stage0"]["identity"],
        transformer_convention=ctx["stage0"]["transformer_convention"],
        structural_cross_check_vs_archive=ctx["stage0"]["structural_cross_check_vs_archive"],
        operator_row_permutation_is_noop=ctx["stage0"]["operator_row_permutation_is_noop"],
        shuffled_phase_assignment=ctx["stage0"]["shuffled_phase_assignment"],
        data_whitened=ctx["stage0"]["data_whitened"],
        differential_mode=ctx["stage0"]["differential_mode"],
        laplacian=ctx["stage0"]["laplacian"],
        feeder_facts=ctx["stage0"]["feeder_facts"],
    ))

    from make_out import build_method_out
    mo = build_method_out(here=HERE, stage0=ctx["stage0"], screen_p=screen_p,
                          screen_s=screen_s, mech=mech, ctrl=ctrl, ext=ext,
                          sel=sel, checks=checks)
    _dump("screen_A_output_restricted_pullback.json", mo["contract"])
    # method_out.json is the datasets-grouped `exp_gen_sol_out` document the pipeline
    # reads: one example per screen cell, per mechanism cell and per control cell, with
    # every candidate's prediction as a predict_* field.  The full nested record --
    # verdicts, mechanism verdict, operator checks, protocol checks, cost ledger -- is
    # carried in `metadata`, which the schema leaves open, so no detail is lost.
    (HERE / "method_out.json").write_text(json.dumps(
        dict(metadata=mo["metadata"], datasets=mo["datasets"]),
        indent=1, default=_json_default))
    (HERE / "method_detail.json").write_text(json.dumps(
        mo["method_out"], indent=1, default=_json_default))
    logger.info(f"wrote method_out.json ({len(mo['datasets'][0]['examples'])} examples) "
                f"and method_detail.json; total wall clock {time.time()-t_start:.0f}s")

    try:
        from figures import make_all
        make_all(HERE, mo)
        logger.info("figures written")
    except Exception as e:                                        # noqa: BLE001
        logger.error(f"figures failed: {e}")


def _reload_ctx() -> dict:
    """Rebuild the in-memory context from cached npy files when stage 0 is skipped."""
    from feeder import Physics
    from data import load_ensemble, load_near_sheet, load_exact_jacobian
    ph = Physics()
    far, s15 = ph.sheet(15)
    return dict(
        ph=ph, s15=s15,
        S_far=np.load(RESULTS / "S_far.npy"),
        S_full=np.load(RESULTS / "S_full.npy"),
        S_all90=np.load(RESULTS / "S_all90.npy"),
        S_all55=np.load(RESULTS / "S_all55.npy"),
        S_near=(np.load(RESULTS / "S_near.npy")
                if (RESULTS / "S_near.npy").exists() else None),
        ens2=load_ensemble(REGIME_PRIMARY), ens1=load_ensemble(REGIME_SECONDARY),
        near=load_near_sheet(), jac=load_exact_jacobian(),
        stage0=json.loads((RESULTS / "stage0_inputs.json").read_text()),
    )


if __name__ == "__main__":
    main()
