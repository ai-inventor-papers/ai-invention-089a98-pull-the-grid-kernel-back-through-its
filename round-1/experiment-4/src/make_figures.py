#!/usr/bin/env python3
"""Render the artifact's figures from the measured numbers, via the aii-data-fig-gen skill.

Every figure is a JSON spec handed to the skill's generator, so the drawn bar is the
measured value and cannot disagree with the results file.  The raw values behind each
figure are written next to the figure as a chart-spec JSON, which is also the record
another reader can re-check.

Figures
-------
fig1_normalised_rmse_vs_n   normalised far-end RMSE against training size, all ten
                            prediction columns, with the mean-predictor guard line
fig2_paired_margin_forest    the paired margin over the best of the five named
                            baselines at each training size, mean with a 95% bootstrap
                            interval over the 24 folds
fig3_coverage               empirical 68% and 95% interval coverage against nominal,
                            for the four interval-producing columns
fig4_spectrum               the singular spectrum of the sensitivity operator and the
                            eigenvalue spectrum of the three graph metrics
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
SPEC = HERE / "figure_specs"
SPEC.mkdir(exist_ok=True)

GEN = Path("/ai-inventor/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py")

LABEL = {
    "b1_rbf": "RBF on injections",
    "b2_lap_hop": "unit-weight graph Laplacian",
    "b3_lap_adm": "admittance graph Laplacian",
    "b4_ridge": "ridge on injections",
    "b5_lindist": "LinDistFlow + ridge calibration",
    "physics_only": "physics mean alone",
    "zero_mean_gp": "zero-mean sensitivity kernel",
    "resid_gp_mll": "physics + residual GP (marginal likelihood)",
    "resid_gp_tempered": "physics + residual GP (tempered intervals)",
    "resid_ridge_linear": "physics + residual ridge",
}
NS = [10, 20, 40, 80, 160]


def render(spec: dict, name: str) -> None:
    sp = SPEC / f"{name}.json"
    sp.write_text(json.dumps(spec, indent=1))
    out = FIG / name
    cmd = [sys.executable, str(GEN), "--spec", str(sp), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {name}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        raise SystemExit(1)
    print(f"rendered {name}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'ok'}")


def fig1(dec: dict) -> None:
    series = []
    for arm in LABEL:
        series.append({"label": LABEL[arm], "x": NS,
                       "values": [dec[str(n)][arm]["mean_rmse_norm"] for n in NS]})
    guard = [dec[str(n)]["mean_rmse_over_mean_predictor"]["b5_lindist"] for n in NS]
    series.append({"label": "mean predictor (guard g1 = 1.0)", "x": NS,
                   "values": [1.0] * len(NS)})
    render({"type": "line",
            "title": "Normalised RMSE against training size",
            "xlabel": "AC power-flow solves for training",
            "ylabel": "Normalised far-end RMSE",
            "aspect": "16:9", "series": series}, "fig1_normalised_rmse_vs_n")
    _ = guard


def fig2(dec: dict) -> None:
    cats, vals, errs = [], [], []
    for n in NS:
        d = dec[str(n)]["resid_gp_tempered"]
        cats.append(f"n = {n}")
        vals.append(100.0 * d["mean_margin_vs_best_baseline"])
        lo, hi = d["margin_ci95"]
        # the forest renderer takes one half-width per row; use the LARGER side so
        # the drawn bar never understates the bootstrap interval
        errs.append(float(max(100.0 * (d["mean_margin_vs_best_baseline"] - lo),
                              100.0 * (hi - d["mean_margin_vs_best_baseline"]))))
    render({"type": "forest", "null_line": 0.0,
            "title": "Paired margin over the best baseline",
            "xlabel": "Paired margin (%), 95% bootstrap CI",
            "aspect": "4:3", "categories": cats,
            "series": [{"values": vals, "errors": errs}]},
           "fig2_paired_margin_forest")
    (SPEC / "fig2_values.json").write_text(json.dumps(
        {"categories": cats, "mean_pct": vals, "ci95_pct": [dec[str(n)]["resid_gp_tempered"]["margin_ci95"] for n in NS],
         "sign_test_p": [dec[str(n)]["resid_gp_tempered"]["margin_sign_test_p"] for n in NS],
         "worst_fold_pct": [100.0 * dec[str(n)]["resid_gp_tempered"]["worst_fold_margin"] for n in NS]},
        indent=1))


def fig3(dec: dict) -> None:
    cats = [f"n = {n}" for n in NS]
    series = []
    for arm, key in (("resid_gp_tempered", "mean_coverage68"), ("resid_gp_tempered", "mean_coverage95"),
                     ("resid_gp_mll", "mean_coverage68"), ("resid_gp_mll", "mean_coverage95"),
                     ("resid_ridge_linear", "mean_coverage68"), ("resid_ridge_linear", "mean_coverage95"),
                     ("physics_only", "mean_coverage68"), ("physics_only", "mean_coverage95")):
        label = f"{LABEL[arm]}, {key[-2:]}%"
        series.append({"label": label, "values": [dec[str(n)][arm][key] for n in NS]})
    series.append({"label": "nominal 68%", "values": [0.68] * len(NS)})
    series.append({"label": "nominal 95%", "values": [0.95] * len(NS)})
    render({"type": "line",
            "title": "Empirical coverage against the nominal level",
            "xlabel": "AC power-flow solves for training",
            "ylabel": "Empirical interval coverage",
            "aspect": "16:9", "series": series}, "fig3_coverage")
    (SPEC / "fig3_values.json").write_text(json.dumps(
        {arm: {k: [dec[str(n)][arm][k] for n in NS] for k in
               ("mean_coverage68", "mean_coverage95", "mean_nlpd")}
         for arm in ("resid_gp_tempered", "resid_gp_mll", "resid_ridge_linear", "physics_only")},
        indent=1))


def fig4(checks: dict, dec: dict) -> None:
    sv = np.array(checks["operator_vs_archived"]["spectrum"]["sv_normalised"])
    render({"type": "line",
            "title": "Spectrum of the sensitivity operator",
            "xlabel": "Singular-value index",
            "ylabel": "Singular value / largest",
            "aspect": "16:9",
            "series": [{"label": "S = d|V_far|/dx (45 x 110)", "x": list(range(1, len(sv) + 1)),
                        "values": [float(v) for v in sv]}]},
           "fig4a_operator_spectrum")
    cats, vals = [], []
    for key in ("M_hop", "M_adm", "M_imp"):
        spec = checks["graph_metric_spectra"][key]
        cats.append({"M_hop": "unit-weight Laplacian", "M_adm": "admittance Laplacian",
                     "M_imp": "impedance Laplacian (diagnostic)"}[key])
        vals.append(spec["participation_ratio"])
    render({"type": "bar",
            "title": "Effective rank of the graph metrics",
            "ylabel": "Participation ratio (max 55)",
            "aspect": "16:9", "categories": cats,
            "series": [{"label": "participation ratio", "values": vals}]},
           "fig4b_metric_effective_rank")
    (SPEC / "fig4_values.json").write_text(json.dumps(
        {"operator_sv_normalised": [float(v) for v in sv],
         "operator_rank_1e2": checks["operator_vs_archived"]["spectrum"]["rank_1e2"],
         "operator_rank_1e3": checks["operator_vs_archived"]["spectrum"]["rank_1e3"],
         "operator_stable_rank": checks["operator_vs_archived"]["spectrum"]["stable_rank"],
         "graph_metric_spectra": checks["graph_metric_spectra"],
         "hop_distance_diagnostic": checks["hop_distance_diagnostic"]}, indent=1))
    _ = dec


def main() -> None:
    screen = json.loads((RES / "screen_D.json").read_text())
    checks = json.loads((RES / "protocol_checks.json").read_text())
    dec = screen["panel_decision"]["cells"]
    fig1(dec)
    fig2(dec)
    fig3(dec)
    fig4(checks, dec)
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
