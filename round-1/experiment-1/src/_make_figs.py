"""Standalone figure regeneration from the results files (no re-run of the screen)."""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from figures import make_all                                       # noqa: E402

contract = json.loads((HERE / "results/screen_A_output_restricted_pullback.json").read_text())
mo = json.loads((HERE / "method_out.json").read_text())
contract["per_n"] = {int(k): v for k, v in contract["per_n"].items()}
sec = contract.get("secondary_regime")
if sec:
    sec["per_n"] = {int(k): v for k, v in sec["per_n"].items()}
make_all(HERE, {"contract": contract, "method_out": mo})
print("figures done")
