"""Aggregate the four results files into the artifact deliverable.

Two outputs:
  results/screen_A_output_restricted_pullback.json
      the shared screen contract: candidate_id, mechanism, and for every n a record
      with mean/worst far-end RMSE, mean RMSE over the mean predictor, mean paired
      margin against the best per-fold baseline, worst fold, the fraction of folds
      clearing 20 percent, NLPD and 68/95 coverage, the per-fold arrays behind each
      summary, and the per-fold table of every baseline's own RMSE
  method_out.json
      the deliverable: the contract above plus the crossovers, the pre-registered
      verdicts evaluated explicitly, the calibrated-expectation ledger, every
      departure, and the cost ledger
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

CANDIDATE_ID = "A_output_restricted_pullback"
MECHANISM_STR = ("output-restricted sensitivity-operator pullback: an isotropic RBF in "
                 "the coordinates S x, S = d|V_far|/dx of shape 45 x 110")

# stated BEFORE the run, so the outcome cannot be relabelled afterwards
CALIBRATED_EXPECTATION = {
    "n20_margin_vs_tuned_lindistflow_pct": 4.0,
    "n20_margin_vs_archived_reading_pct": 46.5,
    "crossover_mean_margin_interval": [20, 40],
    "crossover_everyfold20_runs": 40,
    "n80_margin_range_pct": [78.0, 88.0],
    "coverage_expected": "poor and degrading with n while NLPD improves",
    "note": "recorded before execution; the measured values are compared against it in "
            "calibrated_expectation_ledger below",
}

VERDICTS = {
    "S1_margin_at_least_20pct_every_split_at_n80": (
        "the treatment must beat the best per-fold canonical baseline by >= 20 percent "
        "in every one of the 24 folds at n = 80 on the 2.0x regime"),
    "S2_crossover_finite_on_the_grid": (
        "the mean paired margin must become positive at a finite n, reported as the "
        "interval between adjacent grid sizes, with the full fraction-clearing curve"),
    "S3_ranking_against_the_other_candidates": (
        "NOT EVALUABLE INSIDE THIS ARTIFACT: it requires the other four candidates' "
        "screen files, which this artifact must not read"),
}


def _nkey(d):
    return {int(k): v for k, v in d.items()}


def _summary_record(s: dict) -> dict:
    keep = ("n_folds", "mean_far_end_rmse", "worst_fold_far_end_rmse",
            "std_far_end_rmse", "mean_rmse_over_mean_predictor",
            "mean_margin_vs_best_baseline", "worst_fold_margin", "std_fold_margin",
            "frac_folds_clearing_20pct", "mean_nlpd", "mean_coverage68",
            "mean_coverage95", "mean_margin_vs_lindistflow",
            "worst_fold_margin_vs_lindistflow", "bootstrap_ci_margin",
            "wilcoxon_margin", "bootstrap_ci_margin_vs_b5", "wilcoxon_margin_vs_b5",
            "best_baseline_counts", "per_fold_rmse_treatment", "per_fold_margin",
            "per_fold_margin_vs_b5", "per_fold_rmse_baselines")
    out = {k: s[k] for k in keep if k in s}
    out["mean_rmse_each_baseline"] = {k[len("mean_rmse_"):]: s[k]
                                      for k in s if k.startswith("mean_rmse_")}
    out["guard_g1_each_baseline"] = {k[len("guard_g1_"):]: s[k]
                                     for k in s if k.startswith("guard_g1_")}
    return out


def build_contract(screen_p: dict, screen_s: dict | None) -> dict:
    rec = dict(candidate_id=CANDIDATE_ID, mechanism=MECHANISM_STR,
               regime_primary=screen_p["regime"], screen_metric_at_80=None,
               per_n={}, crossovers=screen_p["crossovers"],
               secondary_regime=None)
    for n, s in _nkey(screen_p["n_summary"]).items():
        rec["per_n"][n] = _summary_record(s)
    if 80 in rec["per_n"]:
        rec["screen_metric_at_80"] = rec["per_n"][80]["mean_margin_vs_best_baseline"]
    rec["primary_pool"] = {
        "canonical": ["b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow"],
        "note": "the canonical pool decides the pre-registered verdicts; the extended "
                "pool is reported separately as a conservative headline"}
    if screen_s is not None:
        sec = {}
        for n, s in _nkey(screen_s["n_summary"]).items():
            sec[n] = _summary_record(s)
        rec["secondary_regime"] = dict(regime=screen_s["regime"], per_n=sec,
                                       crossovers=screen_s["crossovers"])
    return rec


def _mechanism_verdict(mech_agg: dict | None) -> dict:
    """The hypothesis's own disconfirmation test, evaluated explicitly.

    The pre-registered primary mechanism test is: the output-restricted far-end operator
    must beat the full-state operator on far-end targets.  If it does not, output
    restriction is not the active ingredient and the claim reduces to 'admittance
    structure helps'.  The plan requires this to be reported as measured, with no hunt
    for a variant of the full-state metric that loses.
    """
    if not mech_agg:
        return dict(result="NOT RUN")
    far = {m: (v.get("far") or {}) for m, v in mech_agg.items()}
    def g(m, n):
        s = far.get(m, {}).get(n) or far.get(m, {}).get(str(n))
        return None if not s else s.get("mean_rmse")
    ns = [20, 40, 80, 160]
    tab = {}
    for m in ("S_far", "S_full", "S_full_r", "S_all90", "S_all55", "S_near"):
        row = {n: g(m, n) for n in ns}
        if any(v is not None for v in row.values()):
            tab[m] = row
    ratios = {}
    for n in ns:
        f, fl = g("S_far", n), g("S_full", n)
        if f and fl:
            ratios[n] = dict(far_over_full=float(f / fl),
                             far_wins=bool(f < fl))
    near_tab = {}
    for m in ("S_far", "S_near"):
        row = {}
        for n in (10, 20, 40, 80):
            s = (mech_agg.get(m, {}).get("near") or {}).get(n) or \
                (mech_agg.get(m, {}).get("near") or {}).get(str(n))
            if s:
                row[n] = s.get("mean_rmse")
        if row:
            near_tab[m] = row
    crossed = {}
    for n in (20, 40, 80):
        a = near_tab.get("S_far", {}).get(n)
        b = near_tab.get("S_near", {}).get(n)
        if a and b:
            crossed[n] = dict(far_metric_on_near_targets=a,
                              near_metric_on_near_targets=b,
                              near_metric_wins=bool(b < a),
                              ratio=float(a / b))
    n_win = sum(1 for v in ratios.values() if v["far_wins"])
    res = "NOT DISCONFIRMED" if n_win == len(ratios) and ratios else \
          ("DISCONFIRMED" if n_win == 0 else "MIXED")
    return dict(
        statement=("the output-restricted far-end operator must beat the full-state "
                   "operator on far-end targets; if it does not, restriction is not the "
                   "active ingredient"),
        far_end_rmse_by_metric=tab, far_over_full_ratio=ratios,
        n_sizes_where_far_wins=int(n_win), n_sizes=int(len(ratios)),
        result=res,
        crossed_operator_test=crossed,
        rank_matched_cell=dict(
            note="S_full_r is S_full truncated to the effective rank of S_far; if it "
                 "matches S_full's accuracy the full-state metric's advantage is not a "
                 "dimension artefact",
            far_end_rmse={n: g("S_full_r", n) for n in ns}),
        interpretation=("output restriction is NOT what carries the kernel: the "
                        "unrestricted full-state operator is at least as accurate on the "
                        "far-end targets from n = 40 upward, so the claim reduces to "
                        "'admittance structure helps'. This is the hypothesis's own "
                        "stated disconfirmation condition and is reported as measured."
                        if res == "DISCONFIRMED" else
                        "see far_over_full_ratio for the measured values"))


def _verdicts(contract: dict, screen_p: dict) -> dict:
    out = {}
    n80 = contract["per_n"].get(80) or contract["per_n"].get("80")
    if n80:
        frac = n80["frac_folds_clearing_20pct"]
        out["S1"] = dict(
            statement=VERDICTS["S1_margin_at_least_20pct_every_split_at_n80"],
            measured=dict(frac_folds_clearing_20pct=frac,
                          mean_margin=n80["mean_margin_vs_best_baseline"],
                          worst_fold_margin=n80["worst_fold_margin"],
                          n_folds=n80["n_folds"]),
            result=("PASS" if frac >= 1.0 else
                    ("PARTIAL" if frac > 0 else "FAIL")))
    co = screen_p["crossovers"]
    out["S2"] = dict(
        statement=VERDICTS["S2_crossover_finite_on_the_grid"],
        measured=dict(crossover_vs_best=co.get("crossover_vs_best"),
                      crossover_vs_lindistflow=co.get("crossover_vs_lindistflow"),
                      crossover_everyfold20=co.get("crossover_everyfold20"),
                      frac_clearing_curve=co.get("frac_clearing_20pct_curve")),
        result=("PASS" if co.get("crossover_vs_best", {}).get("interval") else "FAIL"))
    out["S3"] = dict(statement=VERDICTS["S3_ranking_against_the_other_candidates"],
                     result="NOT EVALUABLE INSIDE THIS ARTIFACT",
                     reason="needs the other candidates' screen files, which this "
                            "artifact must not read")
    return out


def _ledger(contract: dict) -> dict:
    out = {"stated_before_execution": CALIBRATED_EXPECTATION, "measured": {}}
    n20 = contract["per_n"].get(20) or contract["per_n"].get("20")
    n80 = contract["per_n"].get(80) or contract["per_n"].get("80")
    if n20:
        out["measured"]["n20_margin_vs_best_baseline_pct"] = \
            100 * n20["mean_margin_vs_best_baseline"]
        if "mean_margin_vs_lindistflow" in n20:
            out["measured"]["n20_margin_vs_lindistflow_pct"] = \
                100 * n20["mean_margin_vs_lindistflow"]
        out["measured"]["n20_coverage68"] = n20["mean_coverage68"]
        out["measured"]["n20_coverage95"] = n20["mean_coverage95"]
    if n80:
        out["measured"]["n80_margin_vs_best_baseline_pct"] = \
            100 * n80["mean_margin_vs_best_baseline"]
        if "mean_margin_vs_lindistflow" in n80:
            out["measured"]["n80_margin_vs_lindistflow_pct"] = \
                100 * n80["mean_margin_vs_lindistflow"]
        out["measured"]["n80_coverage68"] = n80["mean_coverage68"]
        out["measured"]["n80_coverage95"] = n80["mean_coverage95"]
        out["measured"]["n80_nlpd"] = n80["mean_nlpd"]
    out["measured"]["crossover_vs_best"] = contract["crossovers"].get("crossover_vs_best")
    out["measured"]["crossover_vs_lindistflow"] = \
        contract["crossovers"].get("crossover_vs_lindistflow")
    out["verdict"] = ("the quantitative expectation was NOT met" if
                      out["measured"].get("n20_margin_vs_best_baseline_pct", 0) < 0
                      else "the quantitative expectation was met on the sign")
    return out


def _cost_ledger(stage0: dict) -> dict:
    conv = stage0.get("transformer_convention") or {}
    adopted = conv.get("adopted", "n/a")
    n_solves = 0
    if isinstance(adopted, str) and adopted.startswith("noload"):
        n_solves += 1
    return dict(
        ac_solves_spent_by_this_artifact=n_solves,
        breakdown=dict(
            operator_construction="0 AC solves with a flat reference voltage; 1 AC "
                                  "solve (no-load) if the no-load reference is adopted",
            adopted_reference_voltage=adopted,
            ensembles="0 (archived 660-solve pool reused; 0 if regenerated, see "
                      "provenance in protocol_checks)"),
        ensembles_provenance=dict(primary=stage0["ensemble_2_0"]["provenance"],
                                  secondary=stage0["ensemble_1_0"]["provenance"]),
        identification_cost_the_previous_framing_charged_to_the_kernel=220,
        note="the kernel itself needs no AC solve; the 220-solve identification cost "
             "belongs to the ground-truth ensemble, not to the metric")


def _departures(checks: dict, stage0: dict) -> list[dict]:
    out = []
    conv = stage0.get("transformer_convention") or {}
    out.append(dict(id="PC1", title="the archived phase-label shuffle was a no-op",
                    status="CLOSED",
                    measured=stage0.get("shuffled_phase_assignment"),
                    detail="the archived control permuted the ROWS of the operator, and "
                           "||S(x-x')|| is invariant under row permutation, so that "
                           "control returned an identical number by construction.  This "
                           "artifact rebuilds the operator from a permuted LOAD->PHASE "
                           "assignment, leaving the data on the real phases."))
    out.append(dict(id="PC2", title="the fixed-half test block is infeasible",
                    status="DEVIATION, recorded",
                    detail="'train = n from one half, test = the other half' needs "
                           "n + 110 <= 220 and fails at n = 160.  Redefined as a fixed "
                           "60-row test block with the training rows drawn from the "
                           "remaining 160.  Scheme 1 folds here will NOT match another "
                           "artifact that implemented the literal wording; scheme 0 is "
                           "unaffected."))
    out.append(dict(id="PC3", title="normaliser undefined between two candidates",
                    status="CLOSED",
                    detail="train-row per-output standard deviation is used everywhere. "
                           "Paired MARGINS are invariant to this choice because the "
                           "numerator and denominator share it; absolute RMSE values "
                           "shift, so a margin comparison to an archived table is "
                           "legitimate and an absolute-value comparison is not."))
    out.append(dict(id="PC4", title="'effective rank at 2.6x' is vacuous and reserved",
                    status="CLOSED",
                    measured=stage0.get("data_whitened"),
                    detail="the operator takes no loading argument, so recomputing its "
                           "own rank at another regime is an identity; and 2.6x is "
                           "reserved evidence this artifact may not read.  Replaced "
                           "with the data-whitened rank of S Cov(X_train) S^T and the "
                           "injection-variance share captured by the top-6/11/20 row "
                           "space of S, at 1.0x and 2.0x."))
    out.append(dict(id="PC5", title="the archived S_sheet15.npy is not elementwise comparable",
                    status="REPORTED",
                    measured=stage0.get("structural_cross_check_vs_archive"),
                    detail="that array came from a builder without the per-unit divide, "
                           "without the |V| projection and with a real/imaginary split.  "
                           "Support-pattern equality was required to proceed; the "
                           "Frobenius difference, Pearson correlation and maximum "
                           "principal angle are informational."))
    out.append(dict(id="PC6", title="LinDistFlow self-disagrees across prior artifacts",
                    status="CLOSED",
                    measured="the 16-cell factorial test errors are in "
                             "results/protocol_checks.json::baseline_sanity and in the "
                             "per-fold table's b5_lindistflow.all_cells_rmse",
                    detail="order 1 vs 2 x flat vs no-load reference x target "
                           "standardisation on/off x absolute vs scale-free penalty "
                           "grid, selected by TRAIN-ONLY cross-validation; margins are "
                           "reported against the selected cell AND against the legacy "
                           "cell so the disagreement stays visible."))
    out.append(dict(id="PC7", title="two prior factual corrections not propagated",
                    status="CORRECTED",
                    measured=dict(differential_mode=stage0.get("differential_mode"),
                                  laplacian=stage0.get("laplacian")),
                    detail="(i) the differential-mode share of far-end variance is "
                           "re-measured here with the definition written out, and is "
                           "reported as an argument FOR phase awareness rather than "
                           "against it.  (ii) the phase-expanded admittance-weighted "
                           "Laplacian is complex-SYMMETRIC, not Hermitian, so eigh/eigsh "
                           "are invalid for it; both the Hermitian-part spectrum and the "
                           "general spectrum are reported."))
    out.append(dict(id="PC8", title="two inherited operator errors found and fixed",
                    status="CORRECTED",
                    measured=conv.get("cells"),
                    detail="(a) the Q-block sign: the archived builder wrote "
                           "dV/dQ = +j dV/dP, which returns +Im(Z)/(Sb|V0|); the "
                           "physical sign is -Im(Z)/(Sb|V0|).  (b) a missing factor 3 "
                           "from the per-phase vs three-phase power base, which left the "
                           "P block 3.15x low.  With both fixed, the P block matches the "
                           "finite-difference AC Jacobian at correlation 0.995 and the "
                           "implied effective resistance matches the path impedance to "
                           "about 8 percent.  The kernel is scale-invariant in S, so "
                           "(b) changes nothing for the treatment's metric but changes "
                           "the LinDistFlow baseline's second-order term by 9x."))
    out.append(dict(id="PC9", title="marginal likelihood over-smooths on this target",
                    status="REPORTED",
                    detail="the AC map is nearly linear in the pulled-back coordinates "
                           "and has no observation noise, so the marginal likelihood's "
                           "noise term absorbs model error and can prefer an "
                           "over-smoothing lengthscale.  Both selection rules (marginal "
                           "likelihood and train-only CV) are run over the SAME grid for "
                           "EVERY candidate, and both are reported."))
    out.append(dict(id="PC10", title="the ridge baseline's intercept is mandatory",
                    status="CORRECTED",
                    detail="standardised features are zero-mean per column while the "
                           "targets sit near 1.0 pu, so a no-intercept ridge returns a "
                           "zero-mean prediction and a normalised RMSE of order 50."))
    if checks.get("determinism_and_leakage"):
        out.append(dict(id="PC11", title="leakage and determinism",
                        status="PASS" if checks["determinism_and_leakage"].get("pass")
                        else "FAIL", measured=checks["determinism_and_leakage"]))
    if checks.get("reserved_evidence"):
        out.append(dict(id="PC12", title="reserved evidence audit",
                        status="PASS" if checks["reserved_evidence"].get("passed")
                        else "FAIL", measured=checks["reserved_evidence"]))
    return out


def build_method_out(here: Path, stage0, screen_p, screen_s, mech, ctrl, ext, sel,
                     checks) -> dict:
    contract = build_contract(screen_p, screen_s)
    mech_agg = None
    if mech:
        import sys
        sys.path.insert(0, str(here / "code"))
        from fold_utils import aggregate_mechanism
        mech_agg = aggregate_mechanism(mech["cells"])
    ctrl_agg = None
    if ctrl:
        import sys
        sys.path.insert(0, str(here / "code"))
        from fold_utils import aggregate_controls, relative_to_treatment
        ctrl_agg = ctrl.get("summary") or aggregate_controls(
            ctrl["cells"] + ctrl.get("reduction_cells", []))
        # `summary` may already carry the {absolute, relative_to_treatment} wrapper (a
        # cached file written by a previous aggregation); unwrap it rather than nesting
        # a second copy, which silently offsets every later traversal by one level.
        if isinstance(ctrl_agg, dict) and "absolute" in ctrl_agg:
            ctrl_agg = ctrl_agg["absolute"]
        ctrl_agg = dict(absolute=ctrl_agg,
                        relative_to_treatment=relative_to_treatment(
                            ctrl_agg, _nkey(screen_p["n_summary"])))

    method_out = dict(
        candidate_id=CANDIDATE_ID,
        artifact="screen_A_output_restricted_pullback",
        kernel_mechanism=MECHANISM_STR,
        hypothesis=("a Gaussian-process emulator whose kernel is an isotropic RBF in the "
                    "coordinates S x, with S the analytic sensitivity of the 15 far-end "
                    "bus phase-voltage magnitudes to the 55 single-phase load P/Q "
                    "injections, predicts far-end phase voltages of a three-phase "
                    "unbalanced LV feeder more accurately than a standard stationary "
                    "kernel, a graph-Laplacian kernel, and a tuned LinDistFlow baseline"),
        feeder=stage0["feeder_facts"],
        contract=contract,
        verdicts=_verdicts(contract, screen_p),
        calibrated_expectation_ledger=_ledger(contract),
        mechanism=mech_agg,
        mechanism_verdict=_mechanism_verdict(mech_agg),
        mechanism_rank_matched_k=None if not mech else mech.get("rank_matched_k"),
        controls=ctrl_agg,
        extended_pool=ext if ext else None,
        cv_selection_arm=None if not sel else dict(
            regime=sel["regime"],
            n_summary={int(k): _summary_record(v)
                       for k, v in sel["n_summary"].items()},
            crossovers=sel["crossovers"]),
        operator_checks=dict(
            identity=stage0["identity"],
            transformer_convention=stage0["transformer_convention"],
            structural_cross_check_vs_archive=stage0[
                "structural_cross_check_vs_archive"],
            row_permutation_is_noop=stage0["operator_row_permutation_is_noop"],
            shuffled_phase_assignment=stage0["shuffled_phase_assignment"],
            spectra=stage0["spectra"],
            data_whitened=stage0["data_whitened"],
            differential_mode=stage0["differential_mode"],
            laplacian=stage0["laplacian"]),
        protocol_checks=_departures(checks, stage0),
        cost_ledger=_cost_ledger(stage0),
        pools=dict(
            canonical=["b1_rbf", "b2_hop", "b3_adm", "b4_ridge", "b5_lindistflow"],
            extended=["canonical + b6_mlp, b7_ard, the impedance-weighted Laplacian "
                      "retained as a demonstration, and the unrestricted 55-dimension "
                      "admittance Z-bus metric"],
            deliberate_exclusion=("the unrestricted 55x55 Z-bus metric is NOT in the "
                                  "canonical pool because it is the full-state metric's "
                                  "load-restricted cousin; including it would let a "
                                  "same-object comparison suppress the treatment's "
                                  "margin.  It appears only in the mechanism section, "
                                  "where it is the point.")),
        headline_claim=dict(
            what_is_claimed=("an isotropic RBF in the coordinates S x, with S the "
                             "analytic sensitivity of the far-end phase-voltage "
                             "magnitudes, is far more sample-efficient than a standard "
                             "stationary kernel, a graph-Laplacian kernel, ridge, an MLP "
                             "and a tuned LinDistFlow baseline on this feeder"),
            what_is_NOT_claimed=("that the OUTPUT RESTRICTION is what carries the "
                                 "kernel -- the mechanism factorial finds the "
                                 "unrestricted full-state operator at least as accurate "
                                 "on far-end targets from n = 40 upward"),
            retained_claim=("admittance structure helps, and the analytic zero-AC-solve "
                            "operator is what supplies it")),
        limitations=[
            "the bootstrap interval is over SPLITS, not over data: all folds share one "
            "220-draw pool, so the genuinely independent evidence is the untouched "
            "regime and the untouched physically-moved-phase feeder, not the seed list",
            "the physically-moved-phase control is NOT run here: it is reserved "
            "confirmation evidence for the survivor.  This artifact therefore cannot "
            "distinguish 'phase identity matters' from 'phase consistency matters'; the "
            "shuffled-label control here is the INCONSISTENCY test, not the identity test",
            "the 2.6x stressed regime is reserved confirmation evidence and was never "
            "opened",
            "the paper cannot claim a data-efficiency advantage over neural surrogates "
            "across the curve: the MLP is run at n = 80 only",
            "S3 (ranking against the other candidates) is not evaluable inside this "
            "artifact",
        ],
        figures=[],
        total_wall_clock_s=None,
    )
    # ---- the pipeline contract: a datasets-grouped `exp_gen_sol_out` document.
    # One example per (regime, n) screen cell, with the treatment's prediction next to
    # every baseline's on the SAME folds.  The full nested record (mechanism, controls,
    # operator checks, protocol checks, verdicts, cost ledger) travels in `metadata`,
    # which the schema allows to carry arbitrary content, so nothing is lost.
    mo = dict(contract=contract, method_out=method_out,
              datasets=build_datasets(contract, method_out))
    mo["metadata"] = compact_metadata(contract, mo["method_out"])
    return mo


def _pred(v):
    return "n/a" if v is None else f"{v:.6g}"


def compact_metadata(contract: dict, mo: dict) -> dict:
    """A small `metadata` block for method_out.json.

    The schema leaves `metadata` open, so the temptation is to put the whole nested
    record there -- which makes the `mini_` and `preview_` variants hundreds of
    kilobytes, defeating their purpose.  Everything bulky (mechanism, controls,
    operator checks, per-fold arrays) lives in method_detail.json and in
    results/*.json instead; this block keeps only what a reader needs in order to
    interpret the `datasets` array.
    """
    c = contract
    return {
        "method_name": "output-restricted sensitivity-operator pullback kernel",
        "candidate_id": mo["candidate_id"],
        "mechanism": mo["kernel_mechanism"],
        "feeder": "IEEE PES European LV test feeder (pandapower "
                  "ieee_european_lv_asymmetric('on_peak_566')), 906 LV buses, 905 lines, "
                  "55 single-phase loads on their real phases (21/19/15)",
        "ground_truth": "pandapower runpp_3ph; 220-draw archived ensemble per regime",
        "folds": "12 seeds x 2 partition schemes = 24 folds per cell",
        "normaliser": "training-fold per-output voltage standard deviation",
        "regime_primary": c["regime_primary"],
        "regime_secondary": (c.get("secondary_regime") or {}).get("regime"),
        "screen_metric_at_80": c.get("screen_metric_at_80"),
        "crossovers": c.get("crossovers"),
        "verdicts": mo["verdicts"],
        "mechanism_verdict_summary": {
            "result": (mo.get("mechanism_verdict") or {}).get("result"),
            "far_over_full_ratio": (mo.get("mechanism_verdict") or {}).get(
                "far_over_full_ratio"),
            "interpretation": (mo.get("mechanism_verdict") or {}).get("interpretation"),
        },
        "headline_claim": mo.get("headline_claim"),
        "pools": mo.get("pools"),
        "cost_ledger": mo.get("cost_ledger"),
        "limitations": mo.get("limitations"),
        "detail_file": "method_detail.json",
        "results_files": ["results/screen_2.0x.json", "results/screen_1.0x.json",
                          "results/mechanism.json", "results/controls.json",
                          "results/operator_checks.json", "results/protocol_checks.json",
                          "results/extended_pool.json",
                          "results/screen_A_output_restricted_pullback.json"],
        "audit": "audit_result.json",
    }


def build_datasets(contract: dict, method_out: dict) -> list[dict]:
    """The `datasets` array: one example per (regime, n) screen cell.

    `predict_*` fields are the per-cell mean normalised test RMSE of every candidate
    on the identical 24 folds, so a reader can compare any two candidates cell by cell.
    `metadata_*` fields carry the paired statistics needed to check the claim: the
    margin, its worst fold, the fraction of folds clearing 20 percent, the bootstrap
    interval and the Wilcoxon p-value.
    """
    ex = []

    def cell(regime: str, n: int, rec: dict) -> dict:
        base = rec.get("mean_rmse_each_baseline", {}) or {}
        e = {
            "input": (f"Predict the far-end phase-voltage magnitudes of the IEEE PES European "
                      f"LV feeder at {regime} loading from n = {n} AC training runs, and "
                      f"report the normalised test RMSE of the output-restricted "
                      f"sensitivity-pullback kernel against every canonical baseline on "
                      f"identical folds."),
            "output": (f"Treatment {_pred(rec.get('mean_far_end_rmse'))}; best per-fold "
                       f"baseline margin {100*(rec.get('mean_margin_vs_best_baseline') or 0):+.1f} "
                       f"percent; worst fold {100*(rec.get('worst_fold_margin') or 0):+.1f} "
                       f"percent; {rec.get('frac_folds_clearing_20pct', 0)*100:.0f} percent of "
                       f"folds clear 20 percent."),
            "metadata_regime": regime,
            "metadata_n_train": str(n),
            "metadata_n_folds": str(rec.get("n_folds")),
            "metadata_mean_margin_vs_best_baseline": _pred(rec.get("mean_margin_vs_best_baseline")),
            "metadata_worst_fold_margin": _pred(rec.get("worst_fold_margin")),
            "metadata_frac_folds_clearing_20pct": _pred(rec.get("frac_folds_clearing_20pct")),
            "metadata_mean_margin_vs_lindistflow": _pred(rec.get("mean_margin_vs_lindistflow")),
            "metadata_mean_coverage68": _pred(rec.get("mean_coverage68")),
            "metadata_mean_coverage95": _pred(rec.get("mean_coverage95")),
            "metadata_mean_nlpd": _pred(rec.get("mean_nlpd")),
            "predict_output_restricted_pullback": _pred(rec.get("mean_far_end_rmse")),
        }
        for k, v in base.items():
            e[f"predict_{k}"] = _pred(v)
        return e

    for n, rec in sorted(contract["per_n"].items(), key=lambda kv: int(kv[0])):
        ex.append(cell(contract["regime_primary"], int(n), rec))
    sec = contract.get("secondary_regime")
    if sec:
        for n, rec in sorted(sec["per_n"].items(), key=lambda kv: int(kv[0])):
            ex.append(cell(sec["regime"], int(n), rec))

    # one example per mechanism cell, so the attribution evidence is queryable too
    for metric, by_t in (method_out.get("mechanism") or {}).items():
        for target, by_n in by_t.items():
            for n, s in sorted(by_n.items(), key=lambda kv: int(kv[0])):
                if not s.get("n_folds"):
                    continue
                ex.append({
                    "input": (f"Mechanism cell: metric {metric} on {target}-end targets at "
                              f"n = {n} AC training runs."),
                    "output": (f"Far-end/target test RMSE {_pred(s.get('mean_rmse'))} "
                               f"({_pred(s.get('mean_rmse_over_mean'))} of the mean predictor)."),
                    "metadata_regime": contract["regime_primary"],
                    "metadata_metric": metric,
                    "metadata_target_sheet": target,
                    "metadata_n_train": str(n),
                    "predict_metric_rmse": _pred(s.get("mean_rmse")),
                })

    # and one per attribution control
    cs = (method_out.get("controls") or {}).get("absolute") or {}
    for c, by_n in cs.items():
        for n, s in sorted(by_n.items(), key=lambda kv: int(kv[0])):
            if not s.get("n_folds"):
                continue
            ex.append({
                "input": (f"Attribution control {c} at n = {n} AC training runs, matched folds "
                          f"and matched tuning budget."),
                "output": (f"Test RMSE {_pred(s.get('mean_rmse'))}, "
                           f"{_pred(s.get('mean_rmse_over_mean'))} of the mean predictor."),
                "metadata_regime": contract["regime_primary"],
                "metadata_control": c,
                "metadata_n_train": str(n),
                "predict_control_rmse": _pred(s.get("mean_rmse")),
            })

    return [{
        "dataset": ("ieee_european_lv_asymmetric on_peak_566, pandapower runpp_3ph "
                    "(1.0x and 2.0x loading; 24 folds per cell)"),
        "examples": ex,
    }]
