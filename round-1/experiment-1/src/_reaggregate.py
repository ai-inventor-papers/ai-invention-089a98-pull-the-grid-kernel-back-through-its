"""Re-aggregate method_out.json from the cached results files, without re-running.

Used after a fix to the aggregation code, so a reporting bug costs seconds rather
than a re-run.  Everything it reads was produced by method.py.
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from make_out import build_method_out                                 # noqa: E402
from fold_utils import aggregate_mechanism, aggregate_controls, \
    relative_to_treatment                                            # noqa: E402

RES = HERE / "results"


def _load(name, default=None):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else default


def _nkey(d):
    return {int(k): v for k, v in d.items()}


stage0 = _load("stage0_inputs.json")
screen_p = _load("screen_2.0x.json")
screen_s = _load("screen_1.0x.json")
mech = _load("mechanism.json")
ctrl = _load("controls.json")
ext = _load("extended_pool.json")
sel = _load("screen_2.0x_cv_selected.json")
checks = _load("protocol_checks.json")

# regenerate the controls summary through the SAME aggregator so the fix propagates
if ctrl:
    agg = aggregate_controls(ctrl["cells"] + ctrl.get("reduction_cells", []))
    ctrl = dict(ctrl)
    ctrl["summary"] = dict(
        absolute=agg,
        relative_to_treatment=relative_to_treatment(
            agg, _nkey(screen_p["n_summary"])))
    (RES / "controls.json").write_text(json.dumps(ctrl, indent=1))

mo = build_method_out(here=HERE, stage0=stage0, screen_p=screen_p,
                      screen_s=screen_s, mech=mech, ctrl=ctrl, ext=ext,
                      sel=sel, checks=checks)
(RES / "screen_A_output_restricted_pullback.json").write_text(
    json.dumps(mo["contract"], indent=1))
(HERE / "method_out.json").write_text(json.dumps(
    dict(metadata=mo["metadata"], datasets=mo["datasets"]), indent=1))
(HERE / "method_detail.json").write_text(json.dumps(mo["method_out"], indent=1))
print("examples:", len(mo["datasets"][0]["examples"]))
mv = mo["method_out"]["mechanism_verdict"]
print("mechanism verdict:", mv["result"])
print("far/full ratio:", json.dumps(mv["far_over_full_ratio"], indent=1))
print("crossed test:", json.dumps(mv["crossed_operator_test"], indent=1))
