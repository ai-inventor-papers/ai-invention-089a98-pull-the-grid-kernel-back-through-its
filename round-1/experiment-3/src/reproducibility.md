# Reproducing this artifact on Ubuntu

Every command below was actually run on this machine to produce `results/`.

## 1. Get the artifact

This workspace is published as **one folder of a public GitHub repository**. From a clone:

```bash
git clone <the repository URL> && cd <repo>/round-1/experiment-3/src
```

Everything from here on is relative to that folder — no absolute server path appears in the
code, the README, or `restore.sh`.

## 2. System packages and Python

Ubuntu 22.04 / 24.04, no GPU required (this artifact is pure CPU linear algebra).

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv 0.6.14 was used
```

Python **3.12.14**. The pinned numeric stack (`pyproject.toml`) is the one that produced the
archived ground truth's comparability, so do not float it:

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml
```

Key pins: `numpy==2.4.6`, `scipy==1.18.1`, `networkx==3.7`, **`pandapower==3.5.6`**,
`pandas==2.3.3`, `loguru==0.7.3`, `psutil==7.2.2`, `matplotlib==3.11.2`.
`pandapower` is pinned hard: a different version can change the archived ground truth's
comparability, and `ph.noload_voltage` (the single zero-load AC solve the physics baseline
uses) is called through it.

Verify:

```bash
.venv/bin/python -c "import numpy, scipy, networkx, pandapower; \
  print(numpy.__version__, scipy.__version__, networkx.__version__, pandapower.__version__)"
# expected: 2.4.6 1.18.1 3.7 3.5.6
```

## 3. Data and environment

**No downloads and no API keys.** This artifact makes zero LLM calls and no HTTP request
anywhere in its code path; there is no OpenRouter key to supply.

The inputs are four small `.npy` archives (145 KB total) produced by an earlier step of the
same run. They live in the run tree, not in this folder, so `code/state.py` locates them by
walking up from `__file__`:

```
iter_2/gen_hypo/claude_agent/pilot/ens_s2.0_X.npy   (220, 110) float64  MW / MVAr
iter_2/gen_hypo/claude_agent/pilot/ens_s2.0_Y.npy   (220,  45) float64  pu magnitudes
iter_2/gen_hypo/claude_agent/pilot/sheet15.npy      (15,)     int64    the 15 far-end buses
iter_2/gen_hypo/claude_agent/pilot/ph_true.npy      (55,)     int64    load -> phase
iter_3/gen_hypo/claude_agent/verify/S_sheet15.npy   (45, 110) float64  cross-check only
```

**If you have only the published repository**, either (a) point `PF_RUN_ROOT` at a tree that
contains those two directories,

```bash
export PF_RUN_ROOT=/path/to/your/run/tree
```

or (b) regenerate the ensembles from the documented recipe. This is the exact recipe the
earlier step ran (`ens.py`, reproduced here as the specification, not as code to import):

```python
import numpy as np, pandapower as pn_net, pandapower as pp
net = pp.networks.ieee_european_lv_asymmetric('on_peak_566')
al = net.asymmetric_load
P0 = al[['p_a_mw','p_b_mw','p_c_mw']].values          # nominal per-phase P
ph = np.argmax(P0, axis=1)                            # each load's own phase
rng = np.random.default_rng(11); cv = 0.5; scale = 2.0
for i in range(220):
    m = rng.lognormal(-0.5*np.log(1+cv**2), np.sqrt(np.log(1+cv**2)), 55)
    p = np.clip(P0.sum(1)*scale*m, 0.02e-3, 60e-3)     # MW, clipped
    q = p*np.tan(np.arccos(0.95))                      # pf 0.95
    for j,(c1,c2) in enumerate([('p_a_mw','q_a_mvar'),('p_b_mw','q_b_mvar'),
                                ('p_c_mw','q_c_mvar')]):
        al[c1] = np.where(ph==j, p, 0.0)
        al[c2] = np.where(ph==j, q, 0.0)
    pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
    # X[i] = concat(p, q);  Y[i] = res_bus_3ph[['vm_a_pu','vm_b_pu','vm_c_pu']].values[sheet15].ravel()
```

About 220 solves at ~90 ms each ≈ 20 s. Then assert the statistics the artifact re-verifies
on load: **mean total active load 114.69 kW**, **far-end mean per-output sd 0.02101 pu**,
220 finite rows, `sheet15 == [681,697,713,729,745,761,777,793,809,825,841,857,873,889,906]`,
`bincount(ph) == [21,19,15]`. If a regenerated array differs materially, stop — the archived
ensembles are the only ground truth this artifact has.

## 4. Run it

```bash
uv run method.py
```

Hardware used: **2 vCPU (container cgroup quota), 14 GB RAM, no GPU.** Approximate runtimes
actually observed:

| phase | time |
|---|---|
| build state + verify operator (T1) | ~24 s |
| partial sweep, n ∈ {20,80}, 8 folds each (T4) | ~25-70 s |
| full sweep, 120 jobs → 108 realised folds (T5) | ~50-55 s |
| determinism re-run of 4 folds (T6) | ~1 s |
| independent audit (`audit_headline.py`) | ~30 s |
| figures | ~10 s |
| **whole pipeline, pinned BLAS threads** | **~82 s** |
| `check_determinism.py` (two full pipelines + byte diff) | ~170 s |

Everything is seeded; no randomness is unseeded. To force bit-reproducibility, `run_screen.py`
sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to `1` **before importing numpy** — without that, multi-threaded BLAS
reductions drift by one ULP between runs and the canonical hash changes.

To reproduce the plan's Step 12E determinism requirement exactly:

```bash
uv run code/check_determinism.py
# expected: "DETERMINISTIC: all 7 result files byte-identical across two full pipeline runs"
```

Individual phases: `uv run code/smoke.py` (T1/T2/T3, guards verified by deliberate misuse),
`uv run code/run_screen.py` (T4/T5/T6), `uv run code/audit_headline.py` (independent audit),
`uv run code/make_figures.py`.

## 5. What you should get, and where it appears

| output | expected | where |
|---|---|---|
| verdict class | **V3** ("nonlinear, but not kernel-specific") | `results/method_out.json` → `verdict_class`; `results/screen_C.json` |
| GP on `Sx` at n = 80 | **0.0340** normalised RMSE | `results/method_out.json` → `key_numbers_at_80_normalised_rmse.gp_on_operator_coordinates` |
| degree-2 in `Sx` at n = 80 | **0.0223** | ...`.degree2_polynomial_in_operator_coordinates` |
| ridge on `Sx` at n = 80 | **0.0911** | ...`.linear_ridge_on_operator_coordinates` |
| best-of-five baseline at n = 80 | **0.0578** (mean over folds) | ...`.best_baseline_per_fold_mean` |
| paired GP-vs-ridge advantage | **+188.5%**, p = 1.19e-07, 24/24 folds | `results/headtohead_C.json` → `["80"]["lin_S45_iso"]` |
| crossover n* (GP beats ridge) | **40 runs** | `results/method_out.json` → `crossover_n_star` |
| survival rule | S1 pass (+53.0%), S2 fail (+2.5% worst), S3 clear | `results/method_out.json` → `survival` |
| baseline table hash | `396072e4d95b6bea...` | `results/baseline_table.json` → `sha256` |
| operator spectrum | rank 11 @1e-2, 19 @1e-3, stable rank 2.177, S vs archive 0.0 | `results/operator_checks.json` |
| audit | 28/28 checks pass | `results/audit_headline.json` |

Figures: `figures/fig1_learning_curve.{pdf,png}` (error versus number of AC solves),
`figures/fig2_forecast.{pdf,png}` (paired difference: explicit features versus the GP),
`figures/fig3_coverage.{pdf,png}` (68% interval coverage). The raw numbers behind each are in
`figures/fig*_values.json`.

In the paper these appear as: the learning-curve figure (error against AC-solve budget), the
paired-difference figure (the mechanism result), and the calibration figure.

## 6. Restoring removed files

See the "Restoring removed files" section of `README.md`; `.venv/` and `code/__pycache__/` are
the only removed paths, and both regenerate from `pyproject.toml` / a single import.

## 7. Known failure modes and what to do

* **Ridge returns RMSE ≈ 50** — the n = 10 the pilot's blow-up. Cause: a penalty grid that is
  not scale-free, or a criterion that rewards an interpolating fit. This artifact's guard g2
  rejects any non-fractional grid and its GCV criterion penalises `df → n`; `smoke.py` proves
  both by deliberately misusing them.
* **The GP arm misses 0.0340 at n = 80 by more than 15%** — the target ordering (bus-major,
  then phases a, b, c), the train-row rms feature scaling, or the per-output train
  standardisation has drifted. `smoke.py` asserts the value and stops.
* **A `.json.tmp` file is left behind** — a crash during `write_json`; the real file is intact.
