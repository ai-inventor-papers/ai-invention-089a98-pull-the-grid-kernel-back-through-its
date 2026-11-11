# Reproducing this artifact

Everything below is what was actually run on this machine. Paths are relative to
this artifact's workspace folder (the folder holding `method.py`); a reader who
clones the published repository should `cd` into that folder first.

## 1. Getting this artifact

```bash
git clone <the published repository URL>
cd <repository>/round-1/experiment-1/src
```

Everything needed to rerun the screen is in this folder. One set of **inputs is
not published here** because it was produced by earlier artifacts in the same
run: the 220-draw AC ground-truth ensembles (and a few optional cross-check
arrays). They live — in the run's own tree — at

```
../../iter_2/gen_hypo/claude_agent/pilot/     (ens_s1.0_{X,Y}.npy, ens_s2.0_{X,Y}.npy,
                                               sheet15.npy, ph_true.npy, Aexact_2.0.npy,
                                               Vref_2.0.npy, xref_2.0.npy)
../../iter_3/gen_hypo/claude_agent/verify/    (ens_near_Y.npy, near_buses.npy,
                                               S_sheet15.npy)
```

The repository publishes those artifacts as **sibling folders**, so the relative
paths above resolve after a clone of the whole run repository. `code/data.py`
holds the single constant `RUN_ROOT` that anchors them; point it at the sibling
folders (or set it to your own copy) and everything else follows. **If the
archived ensembles are absent, `code/data.py::regenerate_ensemble` rebuilds them
from a documented deterministic recipe** — 220 `runpp_3ph` solves per regime,
lognormal multiplier with mean 1 and coefficient of variation 0.5, seed 11,
clipped to [0.02e-3, 60e-3] MW, Q = P tan(arccos 0.95). That route costs about
90 s per regime and the results file records `provenance = "regenerated"` plus
the mean total load and far-end standard deviation, which must land at
57.491 kW / 0.009686 pu (1.0x) and 114.693 kW / 0.021013 pu (2.0x).

No user-uploaded private data is used. No API keys, model downloads or
checkpoints are needed anywhere: the whole artifact is CPU-only and
self-contained once the ensembles exist.

## 2. System and environment

* Ubuntu / Debian GNU/Linux 12 (bookworm), kernel 6.8, x86-64
* CPU only — this run had **2 vCPU** and 14 GB RAM in a cgroup-limited container;
  no GPU, no CUDA, `torch` is never imported
* Python **3.12.14**

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml
```

`pyproject.toml` pins every version actually installed (`numpy==2.4.6`,
`scipy==1.18.1`, `pandas==2.3.3`, `networkx==3.7`, `loguru==0.7.3`,
`matplotlib==3.11.2`, `pandapower==3.5.6`, plus pandapower's own transitive
pins). `pandapower==3.5.6` matters: the feeder model and the `runpp_3ph`
solution change between minor versions, and 3.5.6 is the version the archived
ensembles were solved with.

`pip` is not present in this image — use `uv`.

## 3. Environmental variables

None are required. Do **not** set `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE`
inside this folder: no HuggingFace model or dataset is downloaded, and pointing
them into the workspace duplicates any shared cache. `OMP_NUM_THREADS` and the
other BLAS thread variables are pinned to 1 inside `method.py` and
`code/worker.py` so the fold-level process pool does not oversubscribe the box.

## 4. Exact commands, in order

```bash
# 0) smoke tests first, each with a numeric acceptance window
.venv/bin/python code/tests_kernel.py          # GP correctness (T4): all PASS
.venv/bin/python method.py --stages 0 --workers 1     # ~90 s
#   -> results/stage0_inputs.json, results/operator_checks.json,
#      results/S_far.npy, results/S_full.npy, results/S_all90.npy,
#      results/S_all55.npy, results/S_near.npy,
#      results/phase_shuffled_assignment.npy

# 1+2+3..8) the full experiment
.venv/bin/python method.py --stages 2,3,4,5,6,7,8 --workers 2
```

Or all at once: `.venv/bin/python method.py --workers 2`.

`--stages` accepts any subset of `0,1,2,3,4,5,6,7,8`; a later invocation reuses
the cached results files by name. `--workers 1` runs in-process, `--workers 2`
uses a `ProcessPoolExecutor` with the `spawn` start method (one task per fold).
On a 2-vCPU box, `--workers 2` is the right setting; more workers do not help
because each fold is a sequence of small dense linear-algebra calls.

Approximate runtimes measured here (2 vCPU, `--workers 2`):

| stage | what it does | wall clock |
|---|---|---|
| `code/tests_kernel.py` | GP unit test on synthetic data | 3 s |
| stage 0 | feeder facts, 5 operator builds, identity checks, finite-difference sweep, shuffle construction | 90 s |
| stage 1 | screen at n = 80 only, 24 folds | 100 s |
| stage 2 | full screen, 2.0x regime, 120 folds | 7 min |
| stage 3 | full screen, 1.0x regime, 120 folds | 7 min |
| stage 4 | mechanism factorial + truncation + whitening, ~900 fold cells | 40 min |
| stage 5 | attribution controls, ~500 fold cells | 20 min |
| stage 6 | extended pool (MLP, ARD-RBF) | 12 min |
| stage 7 | train-only-CV hyperparameter arm, 120 folds | 12 min |
| stage 8 | checks, aggregation, `method_out.json`, figures | 30 s |

**Total: 90–100 minutes** on 2 vCPU. Nineteen of those minutes are stage 0–3;
the rest is the mechanism and control factorial.

Every fold is fully determined: `code/folds.py` fixes the RNG as
`default_rng(20261111 + 7919*seed + 131*scheme)` and every hyperparameter search
is a deterministic grid. Re-running gives bit-identical numbers, which
`results/protocol_checks.json::determinism_and_leakage` asserts.

## 5. What a reader should get

| file | contents |
|---|---|
| `method_out.json` | the deliverable: contract, verdicts, mechanism, controls, protocol checks, cost ledger |
| `results/screen_A_output_restricted_pullback.json` | the shared screen contract in its exact key set |
| `results/screen_2.0x.json`, `results/screen_1.0x.json` | every per-fold record (both regimes) |
| `results/mechanism.json` | the metric x target factorial, rank truncation, partial whitening |
| `results/controls.json` | shuffled-phase, random-operator, rotated-injection, reduction controls |
| `results/operator_checks.json` | identity, finite-difference sweep, spectra, data-whitened rank |
| `results/protocol_checks.json` | determinism, leakage, reserved-evidence audit, baseline sanity |
| `results/extended_pool.json` | MLP and ARD-RBF |
| `results/screen_2.0x_cv_selected.json` | the train-only-CV hyperparameter arm |
| `results/figure_data.json` | raw values behind every plotted point |
| `figures/fig1_learning_curve.pdf/.png` | test error against n, 2.0x |
| `figures/fig1b_learning_curve_1p0x.pdf/.png` | test error against n, 1.0x |
| `figures/fig2_mechanism_by_metric.pdf/.png` | far-end error by metric at n = 80 |
| `audit_result.json` | the independent re-derivation of the headline numbers |

Numbers the paper quotes and where they appear:

* **mean paired margin at n = 80, 2.0x: +71.8 %**, worst fold +50.6 %, all 24
  folds clearing 20 % → `method_out.json::contract.per_n[80]`,
  `results/screen_A_output_restricted_pullback.json::screen_metric_at_80`.
  Independently re-derived in `audit_headline.json::headline_gate["2.0"]`.
* **crossover against the best per-fold baseline: between 10 and 20 runs**
  (mean margin −266 % at n = 10, +15.3 % at n = 20) →
  `...::crossovers.crossover_vs_best`. Against LinDistFlow alone the same
  bracket.
* **all 24 folds clear 20 % from n = 80 up** (fraction 1.00 at n = 80 and 160,
  0.875 at n = 40) → `...::crossovers.frac_clearing_20pct_curve`.
* **operator effective rank 11 at relative 1e-2, 19 at 1e-3, 23 at 1e-4, stable
  rank 2.11** → `results/operator_checks.json::spectra.far`.
* **the operator is a Z-bus object**: the grounded block-Laplacian inverse
  matches the BFS path accumulation at 7.2e-12 relative, with zero AC solves →
  `results/operator_checks.json::identity`.
* **the analytic operator's P-block correlation with the finite-difference AC
  Jacobian is 0.995** → `results/operator_checks.json::transformer_convention`.

## 6. Staged scaling (how the run reached the full grid)

The stage order is deliberate so that a truncated run still reports. Stage 1
finishes the pre-registered ranking point (n = 80) before anything else; stages
2 and 3 fill in the curve; stages 4–7 are the mechanism and controls. If you
have less than an hour, run `--stages 0,1` and you still get the headline
verdict; drop stages in the order 3 (1.0x), 6 (extended pool), 7 (CV arm),
4's truncation cells, 5's reduction controls — never drop the LinDistFlow
reconciliation, the full-state metric, or the shuffled-phase control.

## 7. Restoring removed files

Nothing in this artifact is deleted; every results file, figure and saved array
stays in the repository. The Python virtual environment `.venv/` is regenerable
and is the only heavy path:

```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r pyproject.toml
```
