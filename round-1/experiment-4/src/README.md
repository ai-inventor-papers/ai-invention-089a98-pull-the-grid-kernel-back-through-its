# Physics guess first, kernel on what is left

A Gaussian-process emulator for three-phase unbalanced low-voltage distribution
power flow, screened as one of five candidates in a kernel-selection comparison.

**Question.** On the IEEE PES European LV test feeder (`ieee_european_lv_asymmetric('on_peak_566')`,
906 buses, 905 lines, 55 single-phase loads on their real phases), with ground
truth from converged three-phase AC power flow (`runpp_3ph`), can the
output-restricted sensitivity kernel `exp(-||S(x-x')||^2 / (2 l^2))` beat an RBF
kernel on injections and a plain graph-Laplacian kernel at 10-160 AC solves?

**Candidate D (this artifact).** Use a train-calibrated multiphase LinDistFlow
prediction as the Gaussian-process **prior mean**, and place the sensitivity
kernel on the **residual**. The physics does the mean; the kernel models what the
calibration misses.

## Headline results (2.0x loading regime, 24 folds per cell)

Normalised far-end RMSE, mean over folds, normalised by the training rows'
per-output standard deviation (lower is better). The mean predictor scores 1.0.

| method | n=10 | n=20 | n=40 | n=80 | n=160 |
|---|---|---|---|---|---|
| RBF on injections | 1.3996 | 1.1830 | 0.7468 | 0.3049 | 0.1046 |
| unit-weight graph Laplacian | 0.5258 | 0.1767 | **0.0689** | **0.0344** | **0.0167** |
| admittance graph Laplacian | 0.7526 | 0.2528 | 0.1199 | 0.0555 | 0.0230 |
| ridge on injections | 1.3664 | 1.0532 | 0.7000 | 0.1458 | 0.1043 |
| LinDistFlow + ridge calibration | 0.2748 | 0.1404 | 0.1019 | 0.0925 | 0.0891 |
| **physics mean alone** | 0.2748 | 0.1404 | 0.1019 | 0.0925 | 0.0891 |
| zero-mean sensitivity kernel | 0.4740 | 0.1719 | 0.0836 | 0.0341 | 0.0210 |
| **physics + residual GP** | **0.2591** | **0.1268** | 0.0575 | 0.0386 | 0.0250 |
| physics + residual GP, tempered intervals | **0.2591** | **0.1268** | 0.0575 | 0.0386 | 0.0250 |
| physics + residual ridge | 0.2696 | 0.1392 | 0.1019 | 0.0926 | 0.0891 |

Bold marks the best value in each column. The two physics-plus-residual-GP rows
are identical by construction: tempering changes the predictive **variance** only,
and the protocol asserts the point predictions are bit-identical.

Paired margin over the best of the five named baselines in the same fold:

| n | mean margin | 95% bootstrap CI | sign-test p | worst fold | folds clearing 20% |
|---|---|---|---|---|---|
| 10 | **+1.8%** | [-1.5%, +4.5%] | 3.0e-06 | -26.3% | 1/24 |
| 20 | **+4.1%** | [-1.1%, +9.0%] | 1.5e-03 | -24.5% | 4/24 |
| 40 | **+9.3%** | [+3.0%, +15.6%] | 2.3e-02 | -25.4% | 7/24 |
| 80 | -15.2% | [-23.6%, -6.1%] | 2.3e-02 | -49.2% | 2/24 |
| 160 | -74.1% | [-98.3%, -52.4%] | 3.6e-05 | -269.1% | 0/24 |

### What this means, stated as the pre-registered rules required

1. **The small-n crossing is NOT confirmed.** At n=10 and n=20 the mean margin is
   positive but its 95% bootstrap CI **includes zero**, so the pre-registered
   criterion (mean > 0 *and* CI excludes 0) fails at both. The sign test is
   significant because 23 of 24 margins are positive — but the one outlier at
   n=10 is -26%, and a bootstrap CI over 24 folds cannot exclude zero against
   that. This is exactly the outcome the plan predicted in advance: at n=10 the
   physics calibration has 47 parameters against 10 training rows (measured
   effective degrees of freedom 2.49, i.e. 0.25 x n), so the residual GP has
   almost nothing left to fit and the arm is approximately the physics baseline
   by construction. **A +1.8% margin at n=10 is evidence that the residual GP
   does no harm there, not evidence of a new mechanism.**

2. **Against the zero-mean sensitivity kernel, D wins up to n = 40 and loses
   above it.** At n=10 the residual GP scores 0.2591 against the zero-mean
   kernel's 0.4740 (+45%); at n=20 0.1268 vs 0.1719 (+26%); at n=40 0.0575 vs
   0.0836 (+31%). The zero-mean kernel then overtakes it: at n=80 it is **12%
   better** (0.0341 vs 0.0386) and at n=160 **16% better** (0.0210 vs 0.0250).
   The pilot reported the same reversal. So the honest statement is: **the physics
   prior mean buys accuracy while the residual is still data-poor and costs
   accuracy once the kernel has enough rows to model the map directly.** The
   reason is measurable: the physics mean already explains 99.35% of the
   out-of-sample far-end voltage variance at n=160, so the residual is a 0.65%
   correction and the prior mean's own small errors are the binding term.

3. **Coverage was NOT repaired.** The pre-registered rule required
   |c68 - 0.68| <= 0.05 and |c95 - 0.95| <= 0.05 at *every* n; the qualifying set
   is **empty**. The tempered intervals fix the marginal-likelihood arm's severe
   under-coverage (c68 0.493 -> 0.897 at n=160, against 0.68 nominal), but they
   overshoot at large n and undershoot at n=10. The selected tempering multiplier
   sits on the **lower grid boundary (0.05) at every n**, which says the LOO
   coverage-matching criterion wanted *less* widening than the fixed grid allows —
   reported, not re-tuned, because the pre-registered rule forbids widening the
   search after looking. The interval shape (a single Gaussian) is the limiting
   factor, not the width.

4. **Screen survival fails.** The shared rule needs a mean margin >= +20% and a
   worst fold >= +20% at n=80. D achieves **-15.2% mean and -49.2% worst fold**.
   So D **fails S1 and S2**, and S3 (no other candidate beats D by more than 15%)
   is not evaluable inside this artifact because the other four screened
   candidates run in parallel and reading them is forbidden.

**The one clean positive result.** Where D is useful is at n in {10, 20, 40}: it
is the best of the ten columns at all three sizes, beating both the physics
baseline and every kernel baseline (at n=40, 0.0575 against 0.0689 for the best
baseline, +17%). The crossover in AC solves is between **n = 40 and n = 80**:
below it the physics prior mean is the right thing to spend the budget on, above
it the residual has enough rows to be modelled directly and the prior mean becomes
a liability. A 17% margin at n=40 is not the 20% the screen demanded, so this is a
positive but sub-threshold result.

### Mechanism, measured not assumed

The ranking is explained by the rank each model can represent.

- The sensitivity operator `S = d|V_far|/dx` is 45 x 110 with **effective rank 11**
  at a 1e-2 threshold (19 at 1e-3, 23 at 1e-4) and stable rank 2.18. It is
  dominated by three singular values (1.00, 0.94, 0.50), so the operator kernel
  sees an essentially 3-dimensional feature space.
- The three 55x55 graph metrics are all effectively low-rank. Their participation
  ratios (of a maximum 55) are **3.19** for the unit-weight Laplacian, **2.35** for
  the impedance Laplacian and **1.47** for the admittance Laplacian. The unit-weight
  metric is the *least* degenerate of the three and is the strongest graph kernel
  here — its top-3 eigenvalues carry 96% of the spectrum, and its leading feature
  direction correlates 0.94 with total load.
- This is why the plain topology kernel wins: total load plus its phase split is
  almost the whole story of far-end voltage on a radial LV feeder, and a smooth
  graph metric on that one direction captures it with very few parameters. The
  admittance metric spends its leading eigenvalue (82% of its spectrum) on the same
  direction and so has less left for anything else — it is beaten by the unit-weight
  metric at every n.
- The physics mean already explains **99.35%** of the out-of-sample far-end voltage
  variance at n=160 (95.5% even at n=10). A residual model therefore has only 0.65%
  of the variance left to find, which is why the residual GP's large-n ceiling is
  low and why the zero-mean kernel, unconstrained by an imperfect prior, can beat it.

### One inherited construction corrected here, and one kept

**Corrected.** The `F = P M` graph-metric feature map with a `sqrt(mean p^T M p)`
scale used by the earlier code of this run is dimensionally inconsistent (the
feature is `P M` but the scale is the norm of `P M^{1/2}`). It made the unit-weight
and impedance-weight graph baselines degenerate to the constant mean predictor.
This artifact uses the standard PSD metric feature map `F = P M^{1/2}`, which makes
the squared row distance the Mahalanobis quantity `(p_i-p_j)^T M (p_i-p_j)`. With
the correction all five graph baselines beat the mean predictor, as they must.

**Kept, and recorded.** The hop-count **distance matrix** is not a kernel — it has
54 negative eigenvalues out of 55 — so the plain-topology baseline here is the
unit-weight Laplacian influence Gram matrix rather than the raw distance matrix.
Both are reported (`protocol_checks.json -> hop_distance_diagnostic`).

**Also corrected.** The ridge calibrations penalise no intercept. Penalising the
intercept drove the heavily-regularised calibration to predict zero instead of the
train mean, which made the physics baseline a straw man below n=40.

## Layout

```
method.py                      the experiment: 6 prediction columns, 5 named baselines,
                               24-fold sweep, readings, protocol checks, outputs
feeder.py                      the feeder: multiphase LinDistFlow (order 1 and 2), the
                               analytic sensitivity operator S, graph Laplacians and the
                               55x55 kernel metrics. Zero AC solves except one no-load solve
kernels.py                     one kernel implementation shared by every model: GP by
                               marginal likelihood, LOO moments, ridge with train-only CV,
                               coverage scoring
test_stages.py                 staged tests T1-T6 with the numeric expectation at each
                               stage (inputs, operator, baselines, prediction columns,
                               tempering, leakage and determinism). Run this first
audit.py                       independent re-derivation of every headline number from
                               the raw per-fold CSV, plus placebo tests that must fail
make_figures.py                renders the figures from the measured numbers via the
                               aii-data-fig-gen chart generator
data/                          the archived ground truth this artifact needs:
  ens_s2.0_X.npy, ens_s2.0_Y.npy   220 x 110 injections, 220 x 45 far-end phase voltages
                                    at the 2.0x loading regime (decision panel)
  ens_s1.0_X.npy, ens_s1.0_Y.npy   the same at the 1.0x loading regime (robustness panel)
  sheet15.npy, ph_true.npy         the 15 far-end buses and the 55 loads' native phases
  S_sheet15.npy                    the reference sensitivity operator, for the cross-check
results/
  screen_D.json                  the mandated table, diagnostics and pre-registered readings
  screen_D_robustness.json       the 1.0x panel under its own key
  protocol_checks.json           executable leakage, determinism, guard and spectrum checks
  folds_D.csv                    one row per fold (240 rows), full precision
  raw_predictions_seed0_scheme0.json  raw predictions at n = 10, 80, 160 for one fold
figures/                       fig1 RMSE vs n, fig2 paired margins, fig3 coverage,
                               fig4a operator spectrum, fig4b metric effective rank
figure_specs/                  the chart spec and raw values behind each figure
method_out.json                the pipeline result (exp_gen_sol_out schema)
full_method_out.json           identical to method_out.json, full variant
mini_method_out.json           first 3 examples, for quick inspection
preview_method_out.json        first 3 examples, strings truncated
```

## How to run

```bash
uv venv .venv --python=3.12
uv pip install -r requirements-lock.txt   # or: uv pip install -e .
uv run test_stages.py      # T1-T6 first, ~4 min, asserts the numeric expectations
uv run method.py           # the full sweep, ~9 min on 2 CPUs, no GPU
uv run audit.py            # independent re-derivation and placebo tests
uv run make_figures.py     # the five figures
```

`method.py` writes a partial results file after every `(regime, n)` cell, so a
crash loses at most one cell. Set `RESUME_PANELS=0` to force a full recomputation;
by default it reuses complete panels already in `results/`.

## Cost

- **AC power-flow solves charged to the operator:** 1 (a no-load solve for the
  reference phasors). Everything else in `feeder.py` is analytic.
- **AC power-flow solves charged to labels:** 220 per regime, all archived.
- **Wall clock:** 521 s for both panels (240 folds x 10 prediction columns) on 2
  AMD EPYC cores, no GPU.

## Departures from what the literature would want

- **No neural-network baseline.** At 10-160 AC solves a DNN is outside its data
  regime, and a fair hyperparameter search would dwarf the rest of the screen. The
  claim is about which kernel to use at small n, not about kernels beating networks.
- **One feeder, one 15-bus far-end sheet, one load model** (independent lognormal
  draws, CV 0.5, power factor 0.95). External validity is asserted for this feeder
  only; the crossover at n=40-80 is not generalised.
- **No measurement noise.** Ground truth is a deterministic function of the
  injection vector, so the GP noise term is a regulariser and calibration must be
  read off empirical coverage, never off the inferred noise level.
- **The 110-vector input is effectively 55-dimensional**: the ensemble generator
  set `Q = P tan(arccos 0.95)` entrywise, so the Q block is an exact multiple of the
  P block. Recorded in `protocol_checks.json -> q_is_deterministic_*`, not corrected,
  because every model sees the same inputs.
- **The graph metrics are 55x55 and act on the load-magnitude block**, so P and Q
  share one metric. This is the construction this run has used throughout; if
  anything it flatters the treatment's margin, and it is not claimed to be the
  strongest possible phase-aware graph kernel.
- **The 2.6x stressed regime and the moved-phase feeder are reserved evidence**
  and are hard-blocked by a denylist assertion: no result here speaks to regime
  transfer or to phase-label sensitivity.

## Restoring removed files

Nothing in this workspace is large enough to need removal: `data/` is 3.5 MB of
archived `.npy` ensembles, all figures are vector PDFs plus small PNGs, and the
largest single file is `results/raw_predictions_seed0_scheme0.json` at 2.5 MB. The
`.venv/` directory is a local environment and is not published; rebuild it with the
two commands under **How to run**.

If `data/` is ever missing, it can be rebuilt by re-running the archived-ensemble
generator (220 lognormal draws per regime, `runpp_3ph`, ~0.1 s per solve, ~30 s per
regime) and the sensitivity operator from `feeder.lin_operator(sheet)`. The
acceptance test is the archived statistics asserted in `method.stage_ensemble`:
mean total load within 2% of 114.69 kW (2.0x) and 57.49 kW (1.0x), and far-end
voltage standard deviation within 2% of 0.021013 pu and 0.009686 pu. The draws
would not be bit-identical, so a regenerated ensemble is a different sample of the
same load model and must be labelled as such.
