# Does restricting the grid operator to the predicted outputs carry the kernel?

An experimental artifact for one candidate mechanism in a five-way screen: a
Gaussian-process emulator of three-phase unbalanced low-voltage distribution
power flow whose kernel is an isotropic RBF **in the coordinates `S x`**, where
`S = d|V_far|/dx` is the sensitivity of 15 far-end bus phase-voltage magnitudes
to the 55 single-phase load P/Q injections — a 45 × 110 operator built from the
feeder's line sequence impedances and load phase labels, with **no AC solve in
the operator build**.

The question this artifact answers is not "does a sensitivity metric help" but
**which part of it is doing the work**: the output restriction (far-end only),
the phase coupling, the admittance structure, or merely having a low-dimensional
linear map.

Test bed: the IEEE PES European LV test feeder in pandapower
(`ieee_european_lv_asymmetric('on_peak_566')`) — 906 LV buses, 905 lines, 55
single-phase asymmetric loads on their real phases (21 / 19 / 15), one 0.8 MVA
0.416 kV transformer, 57.358 kW native load. Ground truth from `runpp_3ph`.

## Headline

On identical folds, with everything tuned on training rows only:

| n (AC runs) | treatment | best baseline | mean paired margin | worst fold | folds ≥ 20 % |
|---|---|---|---|---|---|
| 10 | 0.556 | 0.067 (LinDistFlow) | −266 % | — | 0.00 |
| 20 | 0.0974 | 0.1136 (LinDistFlow) | **+15.3 %** | −21.2 % | 0.38 |
| 40 | 0.0549 | 0.0988 | **+45.6 %** | +15.9 % | 0.88 |
| 80 | 0.0229 | 0.0912 | **+71.8 %** | +50.6 % | **1.00** |
| 160 | 0.0145 | 0.0863 | **+71.9 %** | +60.1 % | **1.00** |

2.0x loading regime, 24 folds per cell (12 seeds × 2 partition schemes), errors
normalised by the training-fold voltage standard deviation. Paired bootstrap 95 %
interval on the n = 80 margin: **[0.692, 0.740]**, Wilcoxon signed-rank
p = 6 × 10⁻⁸ (over splits, not over data — see Limitations).

The crossover where the treatment starts to win lies **between 10 and 20 AC
runs**; every fold clears 20 % from **n = 80**. The physics baseline is flat in n
(0.114 → 0.086), which is the signature of a correctly calibrated specification
error rather than a variance effect.

## Layout

```
method.py                       the whole experiment, staged 0..8
audit_headline.py               INDEPENDENT re-derivation of the headline numbers
                                + placebo gates; does not import any of code/
reproducibility.md              exact commands, versions, runtimes, expected numbers
pyproject.toml                  every dependency pinned to the installed version

code/feeder.py                  the feeder object: graph, BFS path impedances,
                                grounded multiphase block Laplacian, Z-bus solve,
                                LinDistFlow order 1/2, S = d|V|/dx
code/operators.py               the operator family, spectral facts, rank
                                truncation, partial whitening, the shuffled-phase
                                operator, the structural cross-check, PC7 corrections
code/kernels.py                 the multi-output GP, marginal-likelihood and
                                train-only-CV hyperparameter selection, NLPD +
                                coverage, the ARD-RBF
code/baselines.py               the canonical pool b1..b5 and the extended pool
                                (b5's 16-cell LinDistFlow reconciliation lives here)
code/folds.py                   the fold recipe, 12 seeds × 2 schemes, fingerprints
code/screen.py                  one fold through every candidate, aggregation,
                                bootstrap, Wilcoxon, crossover brackets
code/worker.py                  process-pool entry points (screen / mechanism /
                                control / extended-pool / reduction tasks)
code/fold_utils.py              fold-cell construction and aggregation
code/env_checks.py              the identity checks, the finite-difference sweep,
                                leakage + determinism, the reserved-evidence audit
code/make_out.py                aggregation into the contract + method_out.json
code/figures.py                 chart specs and rendering for both figures
code/data.py                    ensemble resolution: archived, else regenerated

results/                        every per-fold record and every check
figures/                        fig1 / fig1b / fig2 as PDF + PNG, specs included
logs/                           the run log
```

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml
.venv/bin/python code/tests_kernel.py                    # GP correctness, ~3 s
.venv/bin/python method.py --workers 2                   # ~95 min on 2 vCPU
```

`method.py --stages 0,1` gets the headline verdict in about 3 minutes. See
`reproducibility.md` for the per-stage timings and the staged-drop order.

## What is compared

**Canonical pool** (decides the verdicts): an isotropic RBF GP on
per-column-standardised injections; a hop-count graph-Laplacian kernel (two
constructions, better kept); an admittance-weighted graph-Laplacian kernel built
from line sequence impedances only; ridge on standardised injections with a
scale-free penalty grid and train-only CV; and multiphase LinDistFlow order 2
with the exact second-order Z-bus current correction, calibrated per output by
ridge, with a 16-cell factorial (order × reference voltage × target
standardisation × penalty grid) chosen by **train-only** cross-validation.

**Extended pool** (conservative headline only, does not decide verdicts): a
110-32-32-45 tanh MLP at n = 80; an ARD-RBF with genuine per-dimension
lengthscales; the impedance-weighted Laplacian; and the load-restricted 55-row
admittance Z-bus operator — deliberately excluded from the canonical pool
because it is the full-state metric's cousin, and appears only in the mechanism
section where it is the point.

## What the mechanism section decides — and it says NO to output restriction

A full factorial over **metric × target sheet** on the same folds, so that
output restriction is separated from feature dimension: the far-end operator, an
11-bus near-end operator, the full 2718-row state operator, the full-state
operator **rank-matched** to the far-end operator's effective rank, a 90-row
operator spanning the whole hop range, and the 55-row Z-bus operator. Plus rank
truncation (k = 3…45) and partial whitening (γ = 0.5, 1.0) of the primary
operator, and the honest loading-dependent dimension diagnostic (the rank of
`S Cov(X_tr) Sᵀ`, and the injection-variance share captured by the top-6/11/20
row space of `S`) at both regimes.

Far-end test RMSE, 2.0x, 24 folds per cell:

| metric | n=20 | n=40 | n=80 | n=160 |
|---|---|---|---|---|
| `S_far` (45 × 110, the treatment) | 0.0974 | 0.0549 | 0.0229 | 0.0145 |
| `S_full` (2718 × 110) | 0.1040 | **0.0457** | **0.0179** | **0.0109** |
| `S_full_r` (rank-matched to 11) | 0.1102 | 0.0500 | 0.0178 | 0.0111 |
| `S_all90` (30 buses, full hop range) | 0.1122 | 0.0474 | 0.0171 | 0.0101 |
| `S_all55` (55 load-bus rows) | 0.0986 | 0.0395 | 0.0200 | 0.0124 |
| `S_near` (33 × 110) | 0.6829 | 0.5637 | 0.3819 | 0.3253 |

**The pre-registered primary mechanism test is only partially supported — result
MIXED.** The far-end operator wins at n = 20 (far/full = 0.937), but the
unrestricted full-state operator is *more* accurate from n = 40 upward
(far/full = 1.20, 1.28, 1.33). The rank-matched cell `S_full_r` tracks `S_full`
exactly (0.0500 / 0.0178 / 0.0111 vs 0.0457 / 0.0179 / 0.0109), so the full-state
metric's advantage is **not** a dimension artefact either. `S_all90` — 30 buses
spanning the whole feeder — is equally good at n = 160 (0.0101), which is the
narrower claim that survives: **restrict to a modest output set, but not
specifically to the far end.**

The crossed-operator test confirms the metric *is* target-specific, which is what
keeps a restriction-flavoured claim alive at all: the 11-bus near-end operator
beats the far-end operator on near-end targets by 3.03× / 1.87× / 1.77× at
n = 20 / 40 / 80. So the metric encodes the target sheet, but on far-end targets a
*different* target sheet does the job at least as well.

What *does* carry the kernel is the **admittance structure**, and it is
near-linear: truncating `S_far` to 3 ranks costs 8× accuracy, to 5 ranks 4×, to
8 ranks nothing (k = 8, 10, 11, 15, 20, 45 are identical to four decimals), and
partial whitening at γ = 0.5 changes nothing above n = 40. The mechanism reduces
to "a low-rank admittance-derived linear map, not the far-end restriction".

This is reported as measured, not defused: the artifact's headline is the metric's
sample efficiency, not the restriction claim.

## Attribution controls

Shuffled **load-to-phase assignment** rebuilt into the operator (not a row
permutation — that is a no-op in the metric, and the archived attempt was
exactly that); a Frobenius-matched random operator; a random orthogonal rotation
of the injections inside the metric; per-phase aggregate feature sets (6 and 9
features); PCA-11 and random-projection-11 of the training injections.

The **physically-moved-phase feeder** is deliberately *not* run: it is reserved
confirmation evidence for the surviving candidate. This artifact therefore
cannot distinguish "phase identity matters" from "phase consistency matters" on
its own, and the shuffled-label control here is the *inconsistency* test.

## Findings that repair prior artifacts

Recorded in `method_out.json::protocol_checks`, each with its measured value:

* **The archived phase-label shuffle was a no-op.** It permuted the rows of `S`,
  and `‖S(x−x')‖` is invariant under row permutation. Rebuilding from a permuted
  load→phase assignment changes 76 of 110 columns.
* **The inherited operator had a sign error and a factor-3 error.** The Q block
  was built as `+j dV/dP` (returning `+Im(Z)/(Sb|V0|)`); the physical sign is
  negative. And `Z` was divided by `Z_base = V_LL²/S_base` while a single-phase
  load's P is that phase's whole power, so the coefficient needs a factor 3.
  With both fixed the P block correlates **0.995** with a finite-difference AC
  Jacobian and the implied effective resistance matches the path impedance to
  ~8 %. Both errors are reported with the measured six-cell sweep.
* **The LinDistFlow baseline self-disagrees by 1.7× across prior artifacts.**
  Closed by the 16-cell factorial, selected at train-only CV, with every cell's
  test error recorded and the legacy cell reported beside it.
* **The ridge baseline's intercept is mandatory** — standardised features are
  zero-mean while the targets sit near 1.0 pu, so a no-intercept ridge returns a
  normalised RMSE of order 50.
* **Marginal likelihood over-smooths on this target.** The AC map is nearly
  linear in the pulled-back coordinates and has no observation noise, so the
  likelihood's noise term absorbs model error. Both selection rules (marginal
  likelihood and train-only CV) run over the same widened grid for **every**
  candidate, and both are reported.
* **The ridge/injections path in this repo is a cautionary tale, not a bug**: an
  unscaled absolute penalty grid does return ~50, and the scale-free grid is
  what fixes it.

## Limitations

* The bootstrap resamples **splits, not data**: all folds share one 220-draw
  pool. The genuinely independent evidence is the untouched 1.0x regime and the
  reserved moved-phase feeder, not the seed list.
* The 2.6x stressed regime is reserved confirmation evidence and was never
  opened (guarded in `code/data.py::_guard`).
* The MLP is run at n = 80 only, so no data-efficiency claim over neural
  surrogates across the curve is made.
* Ranking against the other four candidates in the screen is **not evaluable
  inside this artifact** — it needs their screen files, which this artifact must
  not read.
* Coverage at n = 80 and above is 1.00 for both intervals: the GP's predictive
  variance is conservative on this near-deterministic target. NLPD and coverage
  are both reported rather than the more flattering one.

## Restoring removed files

Every result, figure, array and log is kept in this repository — nothing a later step
reads is removed. Two heavy paths are regenerable and are dropped after the round:

| removed path | why | restore with |
|---|---|---|
| `.venv/` | the Python environment, rebuilt from the pinned `pyproject.toml` | `uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r pyproject.toml` |
| `code/__pycache__/` | bytecode; created automatically on import | `uv run method.py --workers 2` (or any import of `code/*.py`) |

```bash
# one command to restore both
uv venv .venv --python=3.12 \
  && uv pip install --python=.venv/bin/python -r pyproject.toml \
  && uv run method.py --stages 8 --workers 1
```

`logs/` is kept: it is text, and it records the actual run.
