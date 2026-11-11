"""T10 - report contract check on the produced artefacts.

Asserts the ten-key contract of results/screen_B.json, that every per_n cell carries all
arms, a fold count, the curves, the diagnostics, the baselines block and the crossover
block, that results/baselines_per_fold.csv has one row per fold-cell with no NaN, and
that results/method_out.json parses with the right number of examples.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

TEN_KEYS = {"candidate", "mechanism_one_line", "protocol", "structural_facts", "per_n",
            "screen_metric_at_80", "selection_optimism_at_80", "survival",
            "ranking_report", "corrections_to_prior_text"}
N_GRID = ("10", "20", "40", "80", "160")
EXPECTED_FOLDS = {"10": 24, "20": 24, "40": 24, "80": 24, "160": 12}
REQUIRED_ARMS = {"A0_raw", "A1_sel", "A2_fixed_0p5", "A3_fixed_0p5_k11", "A4_gamma1",
                 "A5_trunc_inert", "A6_floor", "A7_oracle", "A8_shuffled",
                 "A9_meanpred", "A10_tempered"}
REQUIRED_ARM_FIELDS = {"mean_far_end_rmse", "worst_fold_far_end_rmse",
                       "mean_rmse_over_mean_predictor", "mean_margin_vs_best_baseline",
                       "worst_fold_margin", "frac_folds_clearing_20pct", "mean_nlpd",
                       "mean_coverage68", "mean_coverage95", "per_fold", "folds"}

fails = []

screen = json.loads((RESULTS / "screen_B.json").read_text())
if set(screen) != TEN_KEYS:
    fails.append(f"screen_B top-level keys: extra={set(screen) - TEN_KEYS} "
                 f"missing={TEN_KEYS - set(screen)}")

if screen["candidate"] != "B_whitening_rank":
    fails.append(f"candidate id {screen['candidate']}")
if not isinstance(screen["mechanism_one_line"], str) or len(screen["mechanism_one_line"]) < 20:
    fails.append("mechanism_one_line missing")

proto = screen["protocol"]
for k in ("n_grid", "k_grid", "gamma_grid", "f_grid", "seed_base", "seeds_used",
          "schemes", "folds_per_n", "total_fold_cells", "reserved", "reserved_note",
          "denominator", "baseline_primary", "reused_ensembles"):
    if k not in proto:
        fails.append(f"protocol missing {k}")
if proto.get("seed_base") != 20261111:
    fails.append("seed_base")
if proto.get("total_fold_cells") != 108:
    fails.append("total_fold_cells")
if proto.get("folds_per_n") != EXPECTED_FOLDS:
    fails.append(f"folds_per_n {proto.get('folds_per_n')}")
if "SPLITS, not data" not in proto.get("reserved_note", ""):
    fails.append("reserved_note does not say reserving seeds reserves splits not data")

sf = screen["structural_facts"]
for k in ("shape", "singular_values_normalised_first20", "stable_rank", "rank_1e2",
          "rank_1e3", "rank_1e4", "condition_number",
          "admittance_laplacian_identity_max_rel_dev", "operator_crosscheck_vs_archive"):
    if k not in sf:
        fails.append(f"structural_facts missing {k}")
if not (sf["rank_1e2"] == 11 and sf["rank_1e3"] == 19 and sf["rank_1e4"] == 23):
    fails.append("effective ranks")
if abs(sf["stable_rank"] - 2.1768) > 0.01:
    fails.append("stable rank")

if set(screen["per_n"]) != set(N_GRID):
    fails.append(f"per_n keys {set(screen['per_n'])}")
for n in N_GRID:
    cell = screen["per_n"][n]
    if cell["folds"] != EXPECTED_FOLDS[n]:
        fails.append(f"n={n} folds {cell['folds']}")
    if set(cell["arms"]) != REQUIRED_ARMS:
        fails.append(f"n={n} arms extra={set(cell['arms']) - REQUIRED_ARMS} "
                     f"missing={REQUIRED_ARMS - set(cell['arms'])}")
    for a, rec in cell["arms"].items():
        miss = REQUIRED_ARM_FIELDS - set(rec)
        if miss:
            fails.append(f"n={n} arm {a} missing {miss}")
        if len(rec["per_fold"]["rmse"]) != cell["folds"]:
            fails.append(f"n={n} arm {a} per_fold length {len(rec['per_fold']['rmse'])} "
                         f"!= folds {cell['folds']}")
    sd = cell["selected_distribution"]
    if sum(sd["k_counts"].values()) != cell["folds"]:
        fails.append(f"n={n} selected k counts do not sum to folds")
    if sum(sd["gamma_counts"].values()) != cell["folds"]:
        fails.append(f"n={n} selected gamma counts do not sum to folds")
    cur = cell["curves"]
    for k in ("gamma_at_k45", "k_at_gamma0", "k_at_gamma0p5", "k_at_gamma1", "floor_f"):
        if k not in cur:
            fails.append(f"n={n} curves missing {k}")
    if len(cur["gamma_at_k45"]) != 7 or len(cur["k_at_gamma0"]) != 8 \
            or len(cur["floor_f"]) != 8:
        fails.append(f"n={n} curve lengths")
    diag = cell["diagnostics"]
    for tag in ("raw", "whitened_g0p5"):
        if tag not in diag:
            fails.append(f"n={n} diagnostics missing {tag}")
            continue
        for k in ("cond_K", "min_cholesky_pivot", "jitter_sweep"):
            if k not in diag[tag]:
                fails.append(f"n={n} diag {tag} missing {k}")
    bl = cell["baselines"]
    for k in ("b1a", "b1b", "b2", "b3", "b4", "b5_primary", "b5_p17secondary",
              "winner_counts"):
        if k not in bl:
            fails.append(f"n={n} baselines missing {k}")
    xo = cell["crossover"]
    for k in ("raw_mean_margin", "raw_every_fold", "selected_mean_margin",
              "selected_every_fold"):
        if k not in xo:
            fails.append(f"n={n} crossover missing {k}")

if not isinstance(screen["survival"]["survives"], bool):
    fails.append("survival.survives not boolean")
if "S3_no_candidate_beats_by_15pct" not in screen["survival"]:
    fails.append("survival missing S3")
if not isinstance(screen["ranking_report"], dict) or "ordering" not in screen["ranking_report"]:
    fails.append("ranking_report")
if "nothing_clears_20pct" not in screen["ranking_report"]:
    fails.append("ranking_report.nothing_clears_20pct")
if not isinstance(screen["corrections_to_prior_text"], list) \
        or len(screen["corrections_to_prior_text"]) < 4:
    fails.append("corrections_to_prior_text too short")

df = pd.read_csv(RESULTS / "baselines_per_fold.csv")
need_cols = {"n", "fold_id", "seed", "scheme", "b1a_rbf_median", "b1b_rbf_raw",
             "b2_hop_laplacian", "b3_admittance", "b4_ridge",
             "b5p_lindist_honest", "b5s_p17_jointmap", "best_baseline_name",
             "best_baseline_rmse", "mean_predictor"}
if not need_cols.issubset(df.columns):
    fails.append(f"csv missing {need_cols - set(df.columns)}")
if len(df) != 108:
    fails.append(f"csv rows {len(df)}")
if int(df.isna().sum().sum()) != 0:
    fails.append(f"csv NaNs {int(df.isna().sum().sum())}")
if len(df.drop_duplicates(["n", "fold_id"])) != len(df):
    fails.append("csv has duplicate (n, fold_id)")

mo = json.loads((ROOT / "method_out.json").read_text())
if "datasets" not in mo or len(mo["datasets"]) != 1:
    fails.append("method_out datasets")
elif len(mo["datasets"][0]["examples"]) != 108:
    fails.append(f"method_out examples {len(mo['datasets'][0]['examples'])}")
for ex in mo["datasets"][0]["examples"]:
    if not isinstance(ex["input"], str) or not isinstance(ex["output"], str):
        fails.append("method_out example input/output not strings")
        break
    for k, v in ex.items():
        if k.startswith("predict_") and not isinstance(v, str):
            fails.append(f"method_out {k} is not a string")
            break

for tag in ("gamma_curve", "k_curve", "jitter_sweep", "selected_hist", "rmse_vs_n",
            "margin_vs_n"):
    png = ROOT / "figures" / f"{tag}.png"
    if not png.exists() or png.stat().st_size < 5000:
        fails.append(f"figure {tag} missing or too small")

if fails:
    print(f"T10 CONTRACT FAILED ({len(fails)} problems):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("T10 CONTRACT PASSED: 10 top-level keys, 5 per_n cells, 11 arms each, "
      "108 csv rows with no NaN, 108 method_out examples, 6 figures.")
