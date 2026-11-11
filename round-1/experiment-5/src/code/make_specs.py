"""Build the two chart specs the artifact saves, straight from results/screen_E.json.

Figure 1 (learning_curve): normalised far-end RMSE against the number of AC training runs,
with the mean line and the fold-to-fold spread band, for the spectral candidate and the
best baselines. This is the crossover figure an operator reads.

Figure 2 (bar, diverging-free): the control comparison at n = 80 — the candidate and the
faithful controls, all on 22 coordinates with the identical GP protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
SPECS = ROOT / "figures" / "specs"

SERIES = [
    ("SP-R1a-GP-k11", "Laplacian spectral GP (candidate)"),
    ("SP-R1a-LIN-k11", "ridge on the same coordinates"),
]


def main() -> None:
    SPECS.mkdir(parents=True, exist_ok=True)
    scr = json.loads((ROOT / "results" / "screen_E.json").read_text())
    table = scr["table"]

    # ---- figure 1: RMSE vs training-set size, mean and worst over folds
    ns = sorted(int(n) for n in table)
    spec1 = {
        "type": "learning_curve",
        "logx": True,
        "title": "Far-end voltage error against the number of AC power-flow runs",
        "xlabel": "AC training runs",
        "ylabel": "normalised far-end RMSE",
        "aspect": "16:9",
        "train_sizes": ns,
        "series": [
            {"label": label,
             "values": [table[str(n)]["arms"][arm]["mean_far_end_rmse"] for n in ns],
             "std": [table[str(n)]["arms"][arm]["std_far_end_rmse"] for n in ns]}
            for arm, label in SERIES
            if all("mean_far_end_rmse" in table[str(n)]["arms"].get(arm, {}) for n in ns)
        ],
    }
    (SPECS / "fig1_learning_curve.json").write_text(json.dumps(spec1, indent=1))

    # ---- figure 2: control comparison at n = 80
    ctrl = json.loads((ROOT / "results" / "controls_n80.json").read_text())
    order = ["C1_laplacian", "C2_random", "C3_permuted_load_rows", "C4_shuffled_phase",
             "C5_pca_injection", "C6_identity11", "C7_hop_laplacian"]
    labels = {
        "C1_laplacian": "Laplacian eigenvectors (candidate)",
        "C2_random": "random 55x11 block",
        "C3_permuted_load_rows": "row-permuted eigenvectors",
        "C4_shuffled_phase": "shuffled phase labels",
        "C5_pca_injection": "injection PCA",
        "C6_identity11": "fixed indicator coordinates",
        "C7_hop_laplacian": "hop-count Laplacian",
    }
    spec2 = {
        "type": "barh",
        "title": "What produces the margin: identical GP, same 22 coordinates",
        "xlabel": "mean normalised far-end RMSE at 80 runs (lower is better)",
        "aspect": "4:3",
        "categories": [labels[k] for k in order],
        "series": [{"label": "mean over 24 folds",
                    "values": [sum(ctrl[k]) / len(ctrl[k]) for k in order]}],
    }
    (SPECS / "fig2_controls.json").write_text(json.dumps(spec2, indent=1))
    print("wrote", sorted(p.name for p in SPECS.glob("*.json")))


if __name__ == "__main__":
    main()
