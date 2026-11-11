"""Figures, rendered from their numbers with the aii-data-fig-gen generator.

Two figures, each drawn from a JSON chart spec written out of the aggregated
results so the picture cannot disagree with the data:

  fig1_learning_curve   test RMSE (normalised by the train-fold voltage standard
                        deviation) against the number of AC training runs, every
                        candidate on identical folds, in both regimes
  fig2_mechanism_margin paired margin against the best per-fold baseline, by metric

Raw values behind every plotted point are written next to the figures as
results/figure_data.json, so a reader can re-check any curve without rerunning
the screen.
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

GEN = Path("/ai-inventor/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py")
LABELS = {
    "treatment": "Output-restricted pullback (S = d|V_far|/dx)",
    "b1_rbf": "RBF on standardised injections",
    "b2_hop": "Hop-count graph-Laplacian kernel",
    "b3_adm": "Admittance-weighted graph-Laplacian kernel",
    "b4_ridge": "Ridge on standardised injections",
    "b5_lindistflow": "LinDistFlow order 2 (Z-bus correction), train-tuned",
}
ORDER = ["treatment", "b5_lindistflow", "b3_adm", "b2_hop", "b4_ridge", "b1_rbf"]


def _series(per_n: dict, keys) -> list[dict]:
    ns = sorted(int(k) for k in per_n)
    out = []
    for k in keys:
        vals, xs = [], []
        for n in ns:
            s = per_n[str(n)] if str(n) in per_n else per_n[n]
            v = (s["mean_far_end_rmse"] if k == "treatment"
                 else s.get("mean_rmse_each_baseline", {}).get(k))
            if v is None:
                continue
            xs.append(n)
            vals.append(v)
        if xs:
            out.append(dict(label=LABELS.get(k, k), x=xs, values=vals))
    return out


def _write_spec(path: Path, spec: dict) -> None:
    path.write_text(json.dumps(spec, indent=1))


def _render(spec_path: Path, out_base: Path) -> None:
    subprocess.run([sys.executable, str(GEN), "--spec", str(spec_path),
                    "--out", str(out_base)], check=True)


def make_all(here: Path, mo: dict) -> None:
    import sys
    sys.path.insert(0, str(here / "code"))
    results = here / "results"
    figs = here / "figures"
    figs.mkdir(exist_ok=True)
    contract = mo["contract"]
    specs = here / "figures" / "specs"
    specs.mkdir(exist_ok=True)

    raw = {"fig1": {}, "fig2": {}}

    # ---- figure 1: learning curves, both regimes
    # Short series labels: six full descriptions do not fit a single 6.5 in canvas
    # inside a legend plus axes.  The full text is in LABELS and in the results file.
    SHORT = {"treatment": "Pullback (S=d|V_far|/dx)", "b5_lindistflow": "LinDistFlow",
             "b3_adm": "Admittance Laplacian", "b2_hop": "Hop-count Laplacian",
             "b4_ridge": "Ridge", "b1_rbf": "RBF on injections"}

    def series_short(per_n, keys):
        out = []
        for s in _series(per_n, keys):
            out.append(dict(s, label=SHORT.get(
                [k for k in keys if LABELS[k] == s["label"]][0], s["label"])))
        return out

    per_n = {str(k): v for k, v in contract["per_n"].items()}
    raw["fig1"]["2.0x"] = {k: {n: v.get("mean_far_end_rmse") if k == "treatment"
                              else v.get("mean_rmse_each_baseline", {}).get(k)
                              for n, v in per_n.items()} for k in ORDER}
    spec1 = dict(type="line", aspect="4:3",
                 title="Test error vs AC training runs (2.0x loading)",
                 xlabel="AC training runs n",
                 ylabel="Test RMSE / voltage s.d.",
                 series=series_short(per_n, ORDER))
    _write_spec(specs / "fig1.json", spec1)
    raw["fig1"]["_spec"] = spec1
    try:
        _render(specs / "fig1.json", figs / "fig1_learning_curve")
    except Exception as e:                                        # noqa: BLE001
        raw["fig1"]["render_error"] = str(e)

    sec = contract.get("secondary_regime")
    if sec:
        per_ns = {str(k): v for k, v in sec["per_n"].items()}
        spec1b = dict(type="line", aspect="4:3",
                      title="Test error vs AC training runs (1.0x loading)",
                      xlabel="AC training runs n",
                      ylabel="Test RMSE / voltage s.d.",
                      series=series_short(per_ns, ORDER))
        _write_spec(specs / "fig1b.json", spec1b)
        raw["fig1"]["1.0x_spec"] = spec1b
        try:
            _render(specs / "fig1b.json", figs / "fig1b_learning_curve_1p0x")
        except Exception as e:                                    # noqa: BLE001
            raw["fig1"]["render_error_1p0x"] = str(e)

    # ---- figure 2: mechanism, far-end error by metric at n = 80, grouped by n
    mech = mo["method_out"].get("mechanism")
    if mech:
        KEEP = ("S_far", "S_full", "S_full_r", "S_all90", "S_all55", "S_near")
        NS = (20, 40, 80, 160)
        series = []
        for m in KEEP:
            by_n = (mech.get(m) or {}).get("far") or {}
            xs, ys = [], []
            for n in NS:
                s = by_n.get(n) or by_n.get(str(n))
                if s and s.get("mean_rmse"):
                    xs.append(n)
                    ys.append(s["mean_rmse"])
            if xs:
                series.append(dict(label=m, x=xs, values=ys))
        if series:
            spec2 = dict(type="line", aspect="4:3",
                         title="Far-end error by metric (2.0x loading)",
                         xlabel="AC training runs n",
                         ylabel="Test RMSE / voltage s.d.",
                         series=series)
            _write_spec(specs / "fig2.json", spec2)
            raw["fig2"]["values"] = series
            try:
                _render(specs / "fig2.json", figs / "fig2_mechanism_by_metric")
            except Exception as e:                                # noqa: BLE001
                raw["fig2"]["render_error"] = str(e)

    (results / "figure_data.json").write_text(json.dumps(raw, indent=1, default=str))
