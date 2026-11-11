# Screen of candidate E — the feeder's own Laplacian as a Gaussian-process kernel

Does a phase-aware graph kernel built from the **IEEE PES European LV feeder's
phase-expanded admittance-weighted block Laplacian** predict per-bus phase voltages of a
three-phase unbalanced LV feeder better than the standard kernels and the tuned physics
baseline an operator already has? This artifact answers that with one pre-registered
yardstick over 108 folds, and reports what produced the answer.

## Result in one paragraph

The Laplacian spectral kernel **passes the pre-registered criterion**: at 80 AC training
runs it beats the best per-fold baseline by **+76.7 % on the mean and +69.4 % in its worst
fold** (100 % of folds clear +20 %), and the crossover is **20 AC runs** on the mean
criterion and **40 AC runs** before *every* fold clears. The margin is **not** generic
compression: with the identical GP protocol on the same 22 coordinates, the candidate
beats a random block by 9.7×, a row-permutation of its own eigenvectors by 10.9×, the
shuffled-phase-label control by 12.8×, injection PCA by 42× and the hop-count graph
Laplacian by 43×. **The density claim fails**: empirical coverage is 0.8 % at the nominal
68 % level and 1.6 % at 95 %, because the tuning protocol drives the noise variance to the
grid floor on a deterministic AC ground truth. The point prediction is sound; the GP
variance must not be read as a confidence statement. Both facts are in
`results/screen_E.json` under `survival`.

**A baseline bug found and fixed mid-run, which moved every number.** At n = 10 the ridge
baselines were solving a 110-column normal-equation system with a free unpenalised
intercept, i.e. a singular Gram matrix; `b4_ridge` returned an error of 3093 and
`b5_lindist` 1.86 against a mean predictor of 1.40. That is a broken baseline manufacturing
a margin, so the ridge was rewritten as an augmented least-squares problem solved by
`lstsq` (minimum-norm on a rank-deficient design). With the fix the tuned LinDistFlow
reproduces the independent reviewer's measurements almost exactly — 0.1648 at n = 10
(reviewer 0.1665) and 0.0944 at n = 80 (reviewer 0.0974) — and the margins above are the
recomputed, smaller ones. The earlier, larger margins are not reported.

## Layout

| Path | What it is |
|---|---|
| `method.py` | the entry point: data certification, spectral build, the 108-fold sweep, mechanism measurement, decision, all outputs |
| `code/physics.py` | the feeder's BFS tree, sequence-impedance blocks, shared-path block Zbus, multiphase LinDistFlow operators |
| `code/feeder.py` | assembles the grounded 2715×2715 complex-symmetric block Laplacian, verifies the bus-impedance identity with its scope, and builds all five spectral routes with a recorded sign/phase convention |
| `code/baselines.py` | the seven baselines (RBF, hop kernel, admittance Kirchhoff metric at 110 and 55 dims, ridge, tuned LinDistFlow with five calibration variants, dose-matched PCA control) |
| `code/learners.py` | multi-output GP (one shared hyperparameter set, train-only CV *and* marginal-likelihood selection) and the scale-free ridge, with guards g1/g2/g3 |
| `code/folds.py` | the pre-registered 12-seed × 2-scheme fold schedule and the realised-count audit |
| `code/engine.py` | one fold: seven baselines computed once, then every spectral cell against them |
| `code/method_core.py` | prepare / sweep / aggregate |
| `code/mechanism.py` | principal angles against the operator `S`, its SVD spectrum, the Laplacian spectrum summary |
| `code/controls.py` | the shuffled-phase-label permutation |
| `code/checks.py`, `code/ens_regenerate.py` | data/feeder certification, the toy Laplacian unit test, and the rebuild-the-ensembles fallback |
| `code/t4_fold.py` … `code/t8_decomp.py` | the one-fold gate, the distance-scale and lengthscale diagnostics, the decisive controls, the linear-vs-nonlinear decomposition |
| `code/audit.py` | independent re-derivation of every headline number, plus the null control |
| `code/make_specs.py` | the two chart specs |
| `results/screen_E.json` | **the deliverable table**: 12 spectral cells + 7 baselines × 5 sizes × realised folds, per-fold arrays behind every summary, paired margins against each baseline, NLPD and coverage, the crossover pair, the decision |
| `results/protocol_checks.json` | data certification, fold-count audit, `\|L−Lᵀ\|` and `\|L−Lᴴ\|`, the two identity deviations with scopes, eigenpair residuals, the sign/phase record, guard-g3 rates, Cholesky-jitter count, versions |
| `results/spectra.npz` | eigenvalues per route, the 55×45 gathered blocks, `S`, its singular values, the principal-angle arrays, the three metrics |
| `results/controls_n80.json`, `results/decomposition_n80.json` | per-fold control and decomposition errors |
| `results/per_fold_raw.json` | the raw per-fold records (checkpointed during the sweep) |
| `figures/fig1_learning_curve.{pdf,png}`, `figures/fig2_controls.{pdf,png}` | the two figures; specs are in `figures/specs/` |
| `method_out.json`, `full_`, `mini_`, `preview_` | `exp_gen_sol_out`-shaped output, one record per (size, cell) |
| `reproducibility.md` | the complete step-by-step reproduction |

## What the artifact deliberately measures rather than asserts

* **The bus-impedance identity, with its scope.** `inv(Lg)` matches the *line-network*
  shared-path block Zbus to **3.7 × 10⁻¹²**, but deviates from the *transformer-inclusive*
  Zbus by **4.1 × 10⁻²** — exactly `|ztr| / |Zpath|`, because the transformer's series
  impedance is an additive shared term that no perturbation of the line Laplacian
  represents. Both numbers are reported so the claim cannot overreach.
* **The solver choice.** `|L−Lᵀ| = 4.1 × 10⁻¹²` but `|L−Lᴴ| = 2.5 × 10⁴` on a `|L|` scale of
  3.6 × 10⁴: the matrix is complex-**symmetric**, not Hermitian. `eigvalsh` on it is wrong
  by 9.3 × 10³ and `eigsh` never converges (bounded at 180 s and recorded).
* **The eigenvector convention.** Every vector is rotated so its largest-magnitude entry is
  real and positive, and that entry index is recorded for all 45 vectors — without it the
  artifact is not reproducible across LAPACK builds.
* **The circularity.** The admittance Kirchhoff metric is derived from the same Laplacian
  as the candidate's spectrum, so it is flagged as *not* an independent baseline;
  RBF, hop, ridge and LinDistFlow are all constructible without it, and the margins against
  those are reported separately.

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python -r pyproject.toml
.venv/bin/python code/checks.py            # smoke + unit tests, seconds
.venv/bin/python code/_build.py            # spectral build, ~265 s
.venv/bin/python method.py                 # the screen, ~510 s
.venv/bin/python method.py --reuse-sweep   # re-aggregate only, ~22 s
.venv/bin/python code/t6_controls.py       # controls
.venv/bin/python code/t8_decomp.py         # decomposition
.venv/bin/python code/audit.py             # independent re-derivation
```

Full instructions, seeds, hardware and expected numbers: **`reproducibility.md`**.

## Restoring removed files

Exactly one path is marked for deletion in `.aii/manifest.yaml`. Everything else in this
workspace — code, results, figures, logs and the spectral cache — is text or small arrays,
well under the 10 MB auto-keep floor, so it stays where it is on the run's storage volume
and is published with the repository.

| Removed path | Restore with |
|---|---|
| `.venv/` | `uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python -r pyproject.toml` |

Nothing else needs restoring. Two intermediate files created during the work
(`results/_Lg.npy`, the dense 2715-square Laplacian, and `results/_M110.npy`) were
transient probe output and have already been removed; `code/_build.py` recreates both in
under a second, and every authoritative spectral object they fed is stored in
`results/spectra_cache.npz` and `results/spectra.npz`.
