"""make_figures.py -- write chart specs from the result files and render them with the
aii-data-fig-gen house style (never ad-hoc matplotlib colours, fonts or sizes).

Every plotted number is read from results/*.json, so the picture cannot disagree with the
data. The underlying values are also written beside each spec (figures/*_values.json) so a
reader can re-check any claim from the raw numbers through a second view.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
GEN = Path("/ai-inventor/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py")

NS = ["10", "20", "40", "80", "160"]
NSI = [int(x) for x in NS]
STD_YL = "Normalised RMSE (train-rows denominator)"
STD_XL = "AC solves used for training (n)"


def vals(d, arm, key="mean_rmse_norm"):
    return [d["arms"][n][arm].get(key) for n in NS]


def _band(sd):
    """A scalar `band` is what the house-style line renderer accepts; use the largest sd."""
    clean = [v for v in sd if v is not None]
    return float(max(clean)) if clean else None


def render(spec_path: Path, out: Path) -> None:
    exe = str(GEN) if GEN.is_file() else "chart_gen.py"
    py = sys.executable if not GEN.is_file() else _pipeline_python()
    r = subprocess.run([py, exe, "--spec", str(spec_path), "--out", str(out)],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[:600])


def _pipeline_python() -> str:
    for cand in ("/usr/local/bin/python", "/usr/bin/python3"):
        if Path(cand).is_file():
            return cand
    return sys.executable


def main() -> int:
    FIG.mkdir(exist_ok=True)
    d = json.loads((RES / "screen_C.json").read_text())
    h = json.loads((RES / "headtohead_C.json").read_text())
    cal = json.loads((RES / "calibration_C.json").read_text())

    # ---- figure 1: error against the number of AC solves --------------------
    best_base = [float(np.mean([x["rmse_norm"] for x in d["baselines"][n]["_best_per_fold"]]))
                 for n in NS]
    b3 = [d["baselines"][n]["b3_admittance_laplacian_gp"]["rmse_norm"]["mean"] for n in NS]
    b5 = [d["baselines"][n]["b5_lindist"]["rmse_norm"]["mean"] for n in NS]
    b1 = [d["baselines"][n]["b1_rbf_gp"]["rmse_norm"]["mean"] for n in NS]
    # Per-series x, because the degree-2-in-Sx arm is only REPORTED at n >= 40: its
    # n = 10 / 20 cells are pre-registered `under_determined`, not zero, and plotting a
    # gap as a point would be a lie about the data.
    def ser(label, arm, full=True):
        if full:
            return {"label": label, "x": NSI, "values": vals(d, arm),
                    "band": _band(vals(d, arm, "sd_rmse_norm"))}
        pts = [(n, d["arms"][n][arm].get("mean_rmse_norm")) for n in NS]
        pts = [(int(n), v) for n, v in pts if v is not None]
        return {"label": label, "x": [p[0] for p in pts], "values": [p[1] for p in pts]}

    f1 = {
        "type": "line", "logx": True, "logy": True,
        "title": "Far-end voltage error against the number of AC solves",
        "xlabel": STD_XL, "ylabel": STD_YL, "aspect": "16:9",
        "series": [
            ser("degree-2 polynomial in Sx", "quad_S45", full=False),
            ser("GP on Sx (treatment)", "gp_S45"),
            ser("degree-2 polynomial in V11-x", "quad_S11"),
            ser("ridge on Sx", "lin_S45_iso"),
            ser("Bayesian linear regression on Sx", "blr_S45"),
            {"label": "best of five baselines", "x": NSI, "values": best_base},
        ],
    }
    (FIG / "fig1_spec.json").write_text(json.dumps(f1, indent=1))
    (FIG / "fig1_values.json").write_text(json.dumps(dict(
        x=NSI, series={s["label"]: s["values"] for s in f1["series"]},
        denominator="train rows' mean per-output sd (primary); the whole-ensemble value "
                    "is in results/screen_C.json under per_fold.rmse_all_denominator",
        note="quad_S45 is not reported at n=10 or n=20: 1080 features against <40 rows is "
             "not a reportable RMSE (see results/screen_C.json arms['10']['quad_S45'])"),
        indent=1))
    render(FIG / "fig1_spec.json", FIG / "fig1_learning_curve")

    # ---- figure 2: paired relative difference, explicit features vs the GP --
    lin_drel = [h[n]["lin_S45_iso"].get("mean_d_rel") for n in NS]
    q45_drel = [h[n].get("quad_S45", {}).get("mean_d_rel") for n in NS]
    q11_drel = [h[n].get("quad_S11", {}).get("mean_d_rel") for n in NS]
    f2 = {
        "type": "bar",
        "title": "Paired error difference: explicit features against the GP, same folds",
        "xlabel": STD_XL,
        "ylabel": "mean RMSE ratio minus 1 (negative = arm better)",
        "aspect": "16:9", "categories": NS,
        "series": [
            {"label": "ridge on Sx (purely linear)", "values": lin_drel},
            {"label": "degree-2 polynomial in V11-x", "values": q11_drel},
            {"label": "degree-2 polynomial in Sx", "values": q45_drel},
        ],
    }
    # a series with nulls is not plottable; the n<40 degree-2 cells are genuinely absent
    f2["series"] = [s for s in f2["series"] if all(v is not None for v in s["values"])]
    (FIG / "fig2_spec.json").write_text(json.dumps(f2, indent=1))
    fig2_vals = dict(
        x=NSI, series={s["label"]: s["values"] for s in f2["series"]},
        unplottable=dict(degree2_in_Sx=q45_drel,
                         reason="degree-2 in Sx is only reported at n>=40"),
        sign_tests={n: {a: dict(p=h[n][a].get("sign_test_p"),
                                ci=h[n][a].get("boot_ci_rel"),
                                n_folds=h[n][a].get("n_folds"),
                                min_attainable_p=h[n][a].get("min_attainable_p"))
                        for a in ("lin_S45_iso", "quad_S11", "quad_S45")
                        if a in h[n] and "sign_test_p" in h[n][a]}
                    for n in NS})
    (FIG / "fig2_values.json").write_text(json.dumps(fig2_vals, indent=1))
    render(FIG / "fig2_spec.json", FIG / "fig2_forecast")

    # ---- figure 3: calibration trajectories --------------------------------
    arms = [a for a in ("gp_S45", "blr_S45") if a in cal.get("80", {})]
    f3 = {
        "type": "line", "logx": True,
        "title": "Predictive-interval coverage against the number of AC solves",
        "xlabel": STD_XL, "ylabel": "empirical coverage of the nominal 68% interval",
        "aspect": "16:9",
        "series": [{"label": a, "x": NSI,
                    "values": [cal[n][a]["mean_coverage68"] for n in NS]} for a in arms]
                  + [{"label": "nominal 68%", "x": [NSI[0], NSI[-1]], "values": [0.68, 0.68]}],
    }
    (FIG / "fig3_spec.json").write_text(json.dumps(f3, indent=1))
    (FIG / "fig3_values.json").write_text(json.dumps(dict(
        x=NSI,
        coverage68={a: [cal[n][a]["mean_coverage68"] for n in NS] for a in arms},
        coverage95={a: [cal[n][a]["mean_coverage95"] for n in NS] for a in arms},
        nlpd={a: [cal[n][a]["mean_nlpd"] for n in NS] for a in arms},
        nominal=dict(c68=0.68, c95=0.95),
        note="on a deterministic AC ground truth with marginal-likelihood noise inference, "
             "NLPD and coverage can move in opposite directions; both are reported"),
        indent=1))
    render(FIG / "fig3_spec.json", FIG / "fig3_coverage")
    print("figures written to", FIG.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
