# Screen candidate B — whiten versus truncate the pullback sensitivity kernel

Does **partially whitening** the 45×110 pullback sensitivity operator `S` before the RBF
repair the raw Kirchhoff-sensitivity kernel's small-`n` failure, and does **rank
truncation** do anything at all?

This is one of five deliberately different mechanisms screened on one pre-registered
yardstick (slot B of the iteration-1 five-way screen). It is a mechanism screen, not a
regime-transfer study.

## Result in one line

**The candidate does not survive.** At the pre-registered ranking cell (n=80, 2.0×
regime) the train-selected whitened kernel has a mean paired margin of **−15.1 %**
against the best per-fold baseline, with a worst fold of −111.2 % and 8 % of folds
clearing +20 %. No arm in the screen clears the +20 % bar, so `survival.survives` is
`false`. That is reported as the result, not re-thresholded.

The two halves of the mechanism separate cleanly:

* **Truncation is inert at gamma = 0** (k = 8, 10, 11, 15, 20, 45 all score 0.112 at
  n=80): the raw operator already behaves like its rank-8 truncation, so `k` is not a
  design parameter in the raw family. This confirms the archived negative finding.
* **Partial whitening moves the result in the wrong direction.** At k=45, n=80 the gamma
  curve is monotone increasing from 0.112 (gamma=0) to 0.197 (gamma=1), so any whitening
  hurts. The archived pilot's apparent +29.4 % at n=20 was measured against the
  miscalibrated LinDistFlow baseline, not a real gain.
* **Full whitening (gamma=1) is catastrophic but not by the mechanism the direction
  predicted**: it is −107 % at n=80, and the jitter sweep does not rescue it, so the
  collapse is statistical (null-space amplification), as predicted.
* **The crossover with the physics baseline is at n=160 for the mean margin only**; the
  every-fold crossover is never reached in the screened grid. The honest per-output
  ridge on LinDistFlow wins all 24 folds at n=80, so the physics baseline is the margin
  reference throughout.

## What is measured

* **Ground truth**: archived 220-draw `runpp_3ph` ensemble of the IEEE PES European LV
  test feeder (`ieee_european_lv_asymmetric('on_peak_566')`), 2.0× stressed regime,
  55 single-phase loads on their real phases, 45 far-end outputs (15 far-end buses × a,b,c).
  **Reused, not re-solved** on the measurement path. Mean total load 114.6928 kW;
  far-end per-output sd 0.021013 pu.
* **Operator**: rebuilt analytically from line data, **zero AC solves**. Reproduces the
  archived operator exactly (elementwise max-abs deviation **0.0**) with effective ranks
  11 / 19 / 23 at 1e-2 / 1e-3 / 1e-4 and stable rank 2.1768.
* **Treatment**: 8 truncation ranks × 7 whitening exponents = 56 variants, plus an
  8-point singular-value-floor family = 64 variants per fold.
* **Baselines**: median-normalised RBF on injections (b1a, primary), raw-scale RBF
  (b1b, secondary), hop-count graph kernel induced to draw space (b2), 110-dimensional
  admittance-weighted multiphase Kirchhoff kernel (b3), scale-free ridge (b4), and
  per-output ridge with an unpenalised intercept on multiphase LinDistFlow order 2
  (b5-primary, the margin reference). The archived p17 joint-map calibration is kept as
  a labelled secondary row only.
* **Folds**: 12 seeds × 2 partition schemes, n ∈ {10, 20, 40, 80, 160} → 24 folds at
  n ≤ 80, 12 at n = 160, **108 fold-cells**.
* **Metric**: far-end test RMSE in pu voltage normalised by the mean per-output sd of
  each fold's **TRAINING** rows (leak-free), with NLPD and empirical 68 % / 95 % coverage.
* **Selection**: `(k, gamma)` chosen per fold by train-only held-out validation, 5 reps ×
  20 coarse configs; **the distribution of selected values is itself the finding**
  (pooled: k=8 in 71/108, gamma=0.3 in 64/108).

## Headline table

| n | folds | whitened (selected) | raw (k=45, γ=0) | best baseline | mean pred | margin | worst fold | frac ≥20 % | NLPD | c68 | c95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 24 | 0.9278 | 0.9147 | 0.1833 | 1.4018 | −425.5 % | −778.0 % | 0.00 | 3.09 | 0.76 | 0.92 |
| 20 | 24 | 0.4366 | 0.4485 | 0.1236 | 1.2262 | −248.9 % | −466.6 % | 0.00 | 3.87 | 0.88 | 0.98 |
| 40 | 24 | 0.2123 | 0.2355 | 0.1034 | 1.1723 | −101.8 % | −191.2 % | 0.00 | 4.47 | 0.90 | 0.98 |
| 80 | 24 | 0.1111 | 0.1118 | 0.0941 | 1.1500 | −15.1 % | −111.2 % | 0.08 | 4.91 | 0.94 | 1.00 |
| 160 | 12 | 0.0634 | 0.0637 | 0.0907 | 1.1157 | **+23.5 %** | −3.0 % | 0.58 | 5.26 | 0.95 | 1.00 |

Selection optimism at n=80 (train-selected minus per-fold test oracle): **0.0029**
normalised RMSE — the train-only rule is within 0.3 % of the per-fold oracle, so the
negative result is **not** a grid-search artefact.

The coverage deficit is a noise-inference artefact: the tempered-noise diagnostic row
(marginal-likelihood mean, predictive variance fixed at the train held-out residual
variance) moves coverage68 from 0.94 → 0.96 and NLPD from 4.91 → 4.41 at n=80.

## Layout

```
method.py                      the whole pipeline: checks → baselines → sweep → select → stats → figures
audit.py                       TODO-5 independent re-derivation of every headline number, with placebos
test_gp.py                     GP posterior + both hyperparameter-selection criteria vs a dense reference
test_leakage.py                T6 leakage checks (test-permutation invariance, train-only calibration)
test_cell.py                   T2 single-cell smoke test with timings
pyproject.toml                 pinned dependencies (exact versions from the run's .venv)
data/physics.py                the feeder physics (LCA path-impedance blocks, LinDistFlow, sheet)
data/ens_s2.0_X.npy            220×110 archived injection ensemble (2.0× regime)
data/ens_s2.0_Y.npy            220×45  archived runpp_3ph far-end voltages
data/sheet15.npy               the 15 far-end bus sheet
data/ph_true.npy               the 55 load phase labels
data/S_sheet15.npy             archived operator, used ONLY as a row-space cross-check
data/operators.npz             built by `--stage checks`: S, its SVD, the shuffled-phase S, Z_bus, hop
data/lindist_1_2.npy           built by `--stage baselines`: LinDistFlow order 1 and 2 for all 220 rows
results/screen_B.json          THE ten-key contract output
results/method_out.json        exp_gen_sol_out contract (108 fold examples)
results/baselines_per_fold.csv one row per fold-cell, no NaN — the cross-artifact agreement artefact
results/protocol_checks.json   input assertions, exponent orientation, T4/T7, leakage, audit
results/structural_facts.json  singular spectrum, effective ranks, admittance identity, cross-check
results/audit_rederivation.json  independent re-derivation + placebo controls
results/raw/                   per-cell checkpoints (cell_n{n}_f{fold}.json) + baselines + selection
results/figure_specs.json      the machine-readable numbers behind every plotted series
figures/*.png, *.pdf           gamma_curve, k_curve, jitter_sweep, rmse_vs_n, selected_hist, margin_vs_n
```

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r <(uv pip compile pyproject.toml)   # or: uv sync
.venv/bin/python method.py --stage all          # ~5 min on 2 CPUs, no GPU
.venv/bin/python audit.py                       # re-derive every headline number
.venv/bin/python test_gp.py test_leakage.py     # unit + leakage checks
```

Stages can be run separately and are all checkpointed:
`checks` → `baselines` → `sweep` → `select` → `stats` → `figures`. `--stage sweep --n 80`
restricts the sweep. A re-run skips any cell already on disk.

## Restoring removed files

Nothing in this artifact is deleted except the Python virtual environment, which is
regenerable:

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python numpy==2.4.6 scipy==1.18.1 networkx==3.7 \
    loguru==0.7.3 matplotlib==3.11.2 pandas==2.3.3 pandapower==3.5.6
```

`data/ens_s2.0_*.npy` are small enough to publish and are kept. If they are ever absent,
`method.py --stage checks` regenerates them from pandapower (220 `runpp_3ph` solves,
~30 s) and labels the results `REGENERATED`; the seed and injection law are recorded in
`results/protocol_checks.json`.

## Reproducibility notes

* The GP hyperparameter criterion is the **archived** one
  (`-0.5‖K⁻¹Yc‖²_F − 0.5 m log|K|`), kept so the arms are comparable with the earlier
  tables. It is not the marginal likelihood; the correct form is available as
  `gp_corr(..., mll_mode="quadratic")` and both are verified against a dense reference in
  `test_gp.py`. This is disclosed in `corrections_to_prior_text`.
* `cond(S)` is a round-off residue: `S` has 29 of 45 singular values above the
  double-precision floor, so the reported condition number depends on the LAPACK path
  (4.99e32 vs 1.13e16 on identical input). The reproducible statements are the effective
  ranks, the stable rank, and `condition_number_at_rank23` = 2230.
* Seeds 12–29 are reserved and were not evaluated. Reserving seeds reserves **splits,
  not data** — every fold draws from the same 220-row pool. The genuinely untouched
  evidence is the 2.6× regime and the moved-phase feeder.
* The 1.0× and 2.6× ensembles were not read for any number in this artifact.
