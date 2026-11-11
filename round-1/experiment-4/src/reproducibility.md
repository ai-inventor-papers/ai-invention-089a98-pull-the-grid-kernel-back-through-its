# Reproducing this artifact

Everything below is what was **actually run** on this machine, not an idealised
recipe. It reproduces every number in `results/screen_D.json` and `README.md`.

## 1. Getting the artifact

This workspace is published as one folder of a public GitHub repository. Clone
that repository and `cd` into this folder:

```bash
git clone <repository-url> aii-run
cd aii-run/round-1/experiment-4/src
```

All paths in this document, in `method.py`, `audit.py` and `make_figures.py` are
relative to this folder (anchored on `Path(__file__)`), so nothing depends on the
server this ran on.

## 2. System, Python and library versions

- **OS:** Debian GNU/Linux 12 (bookworm), kernel 6.8.0.
- **Python:** 3.12.14 (`uv venv .venv --python=3.12`).
- **Hardware actually used:** 2 vCPU (AMD EPYC 9654), 14 GB RAM, **no GPU**.
  The whole sweep is CPU-only and needs no CUDA.
- **System packages:** none beyond a C toolchain. `pandapower` was installed as a
  pure-Python wheel; `numba` is **not** required because every three-phase solve in
  the archived ground truth was recorded with `numba=True` upstream but this
  artifact never re-solves, and the one no-load solve uses `numba=False`.

Environment setup, exactly as run:

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

`requirements-lock.txt` (30 pinned packages) and `pyproject.toml` both carry the
exact versions. The ones that matter for numerical results:

| package | version | why it matters |
|---|---|---|
| numpy | 2.5.4 | all linear algebra and the Cholesky path |
| scipy | 1.16.3 | `solve_triangular`, sparse Laplacian solves. **See the warning in §4.** |
| pandapower | 3.3.3 | the European LV feeder and `runpp_3ph` |
| networkx | 3.7 | the BFS tree, hop distances, LCA |
| pandas | 2.3.3 | pandapower's tables |
| loguru | 0.7.3 | logging |

## 3. Data and models

**No downloads and no API keys.** Ground truth is a converged three-phase AC
power-flow ensemble that ships inside this workspace in `data/` (3.5 MB total):

| file | contents | shape |
|---|---|---|
| `ens_s2.0_X.npy` | household injections at the 2.0x loading regime | 220 x 110 |
| `ens_s2.0_Y.npy` | per-unit far-end phase-voltage magnitudes | 220 x 45 |
| `ens_s1.0_X.npy`, `ens_s1.0_Y.npy` | the 1.0x loading regime (robustness panel) | 220 x 110, 220 x 45 |
| `sheet15.npy` | the 15 far-end buses | 15 |
| `ph_true.npy` | each of the 55 loads' native phase | 55 |
| `S_sheet15.npy` | the reference sensitivity operator, for the cross-check | 45 x 110 |

These are read-only archives produced by the sibling artifact of this run that
generated the ensembles (`gen_art_experiment_1`, the pilot), and the reference
operator was produced by `gen_art_experiment_3`. The repository publishes them as
sibling folders, so `method.py` resolves them through one relative constant
anchored on `Path(__file__)` (`RUN_ROOT`), not an absolute path, and a reader only
needs `data/`, which is present. The `data/` copy is authoritative; the sibling
lookup is only a fallback and never fires when `data/` is complete.

The feeder itself is fetched by `pandapower.networks.ieee_european_lv_asymmetric('on_peak_566')`,
which `pandapower` ships with its package data — no network access.

Nothing in this artifact was uploaded by the user, so there is no private input a
reader has to supply.

## 4. The exact commands, in order

Run from this folder, with the venv active (or prefix each with `uv run`).

```bash
# (0) sanity: the environment and the pinned triangular-solve convention
uv run python -c "import numpy, scipy, pandapower, networkx; print(numpy.__version__, scipy.__version__, pandapower.__version__)"

# (1) staged tests T1-T6, with the numeric expectation asserted at each stage.
#     ~4 min. THIS MUST PASS BEFORE THE SWEEP: it is what caught the two
#     implementation bugs described below.
uv run test_stages.py

# (2) the full sweep: 2 regimes x 5 training sizes x 24 folds x 10 columns.
#     ~9 min wall clock. Writes a partial results file after every cell.
uv run method.py

# (3) independent re-derivation of every headline number from the raw per-fold
#     CSV, plus placebo tests that must fail. ~5 min.
uv run audit.py

# (4) the five figures, rendered from the measured numbers. ~20 s.
uv run make_figures.py
```

Seeds and config are fixed in `method.py` and are not command-line arguments:

- fold seed base `20261111`; fold `(seed, scheme)` RNG is
  `default_rng(20261111 + 7919*seed + 131*scheme)` with `seed` in 0..11 and
  `scheme` in {0, 1} — **24 folds per cell**.
- training sizes `n in {10, 20, 40, 80, 160}`.
- decision regime `2.0`, robustness panel `1.0`; the `2.6` regime is a denylist
  token and is never opened.
- GP grid `ell in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8)`, `s2 in (0.3, 1, 3)`,
  `n2 in (1e-8, 1e-5, 1e-3, 1e-1)`; marginal likelihood on training rows only.
- ridge penalty = `logspace(-6, 6, 25)` **fractions** of the design-Gram largest
  eigenvalue, with `max(2, n//4)` contiguous CV folds; the intercept is never
  penalised.
- tempering grid `logspace(0.05, 20, 13)`; the multiplier is chosen by matching
  LOO coverage on training rows only.
- RBF squared distance divided by 110.

To force a full recomputation instead of reusing `results/`:

```bash
RESUME_PANELS=0 uv run method.py
```

Measured wall clock of `method.py` on the 2-vCPU box: **521 s** for both panels
(36.8 s / 8.2 s / 12.1 s / 17.7 s / 186.8 s per `n` cell on the decision panel;
the n=160 cells dominate because the GP fit is O(n^3)).

## 5. Two implementation bugs the staged tests caught

These are recorded because a reader reproducing this must hit the same numbers,
and because both would have produced a wrong *conclusion*, not a crash.

**(a) `scipy.linalg.cho_factor(a, lower=True)` returns the UPPER factor in
scipy 1.16.3.** `cho_solve` then solves with the wrong triangle. On this kernel it
gave a log marginal likelihood wrong by ~1000x on ill-conditioned cells and drove
the hyperparameter search to an over-smooth corner, costing a factor 2.5 in test
RMSE at n=160. `kernels._chol_solve` therefore uses `numpy.linalg.cholesky` plus an
explicit `scipy.linalg.solve_triangular(..., lower=True)` and
`(..., lower=False)`, and `kernels.assert_cholesky_convention()` verifies it
against `np.linalg.solve` (error 3e-12). T2 asserts this.

**(b) The marginal-likelihood quadratic form must be `||L^-1 y||^2`, the FORWARD
triangular solve, not `y^T K^-1 y`.** Using the full `K^-1 y` squares the wrong
vector and again selects the wrong hyperparameters. Fixed in `kernels.gp_fit`, with
the reason recorded in the code.

Two further corrections to inherited constructions are documented in `README.md`
(the `F = P M` vs `F = P M^{1/2}` graph-metric feature map, and the penalised
intercept in the ridge calibrations).

## 6. What a reader should get

### Output files

| file | what to check |
|---|---|
| `results/screen_D.json` | 5 cells under `panel_decision.cells`; each has all 10 columns, each column with `mean_rmse_norm`, `worst_fold_rmse_norm` and a 24-element `per_fold_rmse_norm` |
| `results/screen_D_robustness.json` | the same for the 1.0x panel |
| `results/protocol_checks.json` | every `leakage_test.*.bit_identical` is `true`; `determinism_test.bit_identical` is `true`; `guard_g1_pass.passes` is `true`; `operator_vs_archived.normalised_deviation_vs_archived` is 6.6e-17 |
| `results/folds_D.csv` | 240 data rows (2 regimes x 5 sizes x 24 folds), one row per fold, full float64 precision |
| `results/raw_predictions_seed0_scheme0.json` | raw predictions at n = 10, 80, 160 for seed 0 / scheme 0, so a reader can re-score independently |
| `method_out.json` (+ full/mini/preview) | 50 examples in the `exp_gen_sol_out` schema |

### Numbers, and where they appear in the paper

| number | value | paper location |
|---|---|---|
| physics-plus-residual GP, n=10, normalised RMSE | 0.2591 | Results table 1, contribution figure 1 |
| physics mean alone, n=10 | 0.2748 | Results table 1 |
| zero-mean sensitivity kernel, n=10 | 0.4740 | Results table 1 |
| unit-weight graph Laplacian, n=160 | 0.0167 (best in the screen) | Results table 1 |
| paired margin at n=10, with 95% CI | +1.8%, [-1.5%, +4.5%] | Results, "the small-n claim" paragraph, figure 2 |
| paired margin at n=80 | -15.2% (fails the 20% screen rule) | Results, "screen survival" paragraph |
| crossover in AC solves | between n=40 and n=80 | Abstract and Conclusion |
| physics out-of-sample R^2 at n=160 | 99.35% | Mechanism section |
| sensitivity operator effective rank (1e-2) | 11 of 45, stable rank 2.18 | Mechanism section, figure 4a |
| graph-metric participation ratios | 3.19 / 2.35 / 1.47 | Mechanism section, figure 4b |
| tempered coverage c68 at n=160 vs nominal 0.68 | 0.897 (rule fails at every n) | Calibration paragraph, figure 3 |

### Figures

`figures/fig1_normalised_rmse_vs_n.{pdf,png}` — normalised RMSE against training
size, all ten columns plus the mean-predictor guard.
`figures/fig2_paired_margin_forest.{pdf,png}` — the paired margin at each n with
its bootstrap interval.
`figures/fig3_coverage.{pdf,png}` — empirical 68%/95% coverage against nominal.
`figures/fig4a_operator_spectrum.{pdf,png}` and
`figures/fig4b_metric_effective_rank.{pdf,png}` — the spectra that explain the
ranking.

The raw values behind each figure are in `figure_specs/`.

## 7. Known non-reproducible details

- **BLAS reduction order** can move the last 1-2 digits of a fold's RMSE between
  machines. The `audit.py` check `independent b5_lindist at n=160` reproduces the
  screen file to a relative 2.3e-5 through a completely different code path, which
  is the scale to expect.
- **The n=10 physics cell is grid-plateau sensitive**: with 47 parameters against
  10 rows the CV score is flat across neighbouring penalty fractions, so a coarser
  penalty grid lands on a different plateau and can move that one cell's mean by
  tens of percent (audit reports this explicitly rather than asserting equality).
  Every *per-fold* value still matches exactly (`audit.verify_b5_folds_exact`
  reports max |difference| = 0 across all 120 decision folds).
- **A regenerated ensemble is not the archived one.** Re-solving the 220 draws
  reproduces the load model's statistics to within 2% but not the same sample, so
  numbers from a regenerated ensemble are a different sample and must be labelled
  as such.
