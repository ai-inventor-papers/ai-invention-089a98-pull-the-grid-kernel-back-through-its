# Is the Gaussian process doing the work?

A mechanism ablation of the phase-aware power-flow surrogate kernel on the IEEE European
low-voltage feeder. The question is narrow and deflationary: when a Gaussian-process emulator
on a Kirchhoff/Laplacian-derived set of coordinates predicts three-phase unbalanced
voltages well, **is the kernel the active ingredient, or is the analytic operator's
coordinate system doing all the work?**

**Answer (verdict class V3 — "nonlinear, but not kernel-specific").** At n = 80 the GP on the
operator's coordinates reaches 0.0340 normalised RMSE, plain ridge on the *same* coordinates
0.0911, and a **fixed degree-2 polynomial feature map in those same coordinates 0.0223** —
better than the GP, on 24/24 folds, at less than half its effective degrees of freedom. So:
the GP is emphatically not decoration (it beats every purely linear model by 188% with a
paired sign test at p = 1.2e-07 over 24/24 folds), but the kernel itself is substitutable by
explicit quadratic features. The deliverable is **explicit features, no GP**.

---

## Layout

| path | what it is |
|---|---|
| `method.py` | the one-command driver: sweep → independent audit → figures |
| `method_out.json` | pipeline-standard deliverable, `exp_gen_sol_out` shape, one example per fold |
| `full_method_out.json`, `mini_method_out.json`, `preview_method_out.json` | size variants of the above |
| `pyproject.toml` | every dependency pinned to the exact version installed |
| `reproducibility.md` | step-by-step reproduction on Ubuntu |
| `results/method_out.json` | **the rich analysis**: verdict, crossover, ranking, scope limits, provenance |
| `results/screen_C.json` | the mandated contract file: every arm × every n, per-fold arrays, hyperparameters |
| `results/baseline_table.json` | all ten LinDistFlow calibration members per fold + the canonical SHA-256 |
| `results/headtohead_C.json` | paired linear-vs-GP arrays, sign tests, Wilcoxon, bootstrap CIs, df ratios |
| `results/calibration_C.json` | NLPD and 68%/95% coverage trajectories |
| `results/operator_checks.json` | the analytic operator's spectrum, effective ranks, stable rank |
| `results/protocol_checks.json` | archive verification, the reserved-file list that was NOT read, provenance |
| `results/audit_headline.json` | the independent re-derivation: 28 checks, all passing |
| `figures/` | three house-style figures (PDF + PNG) plus the exact values behind each |
| `code/` | the modules: `vendor_physics.py`, `protocol.py`, `operators.py`, `ridge_blr.py`, `arms.py`, `state.py` |
| `logs/` | full run logs (`run_screen.log`, `smoke.log`, `determinism.out`) |

## How to run

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml
uv run method.py                 # everything, about 4 minutes on 2 vCPUs
uv run method.py --quick         # smoke test only (archive + operator + guards), ~1 min
uv run method.py --no-figures    # skip figure rendering
```

Individual phases can be run directly from `code/`: `smoke.py` (pre-flight checks and guard
verification), `run_screen.py` (the sweep and all result tables), `audit_headline.py`
(independent re-derivation), `check_determinism.py` (two full pipeline runs, byte-diffed),
`make_figures.py` (house-style figures).

**Zero LLM calls.** No OpenRouter client is imported anywhere; `openrouter_calls = 0` and
`openrouter_usd = 0.0` are recorded in every output file.

## What the artifact actually measures

* **Ground truth** — 220 archived converged `runpp_3ph` three-phase AC solves on the IEEE
  European LV asymmetric feeder (`on_peak_566`), 55 single-phase loads on their real phases,
  2.0× loading (114.69 kW mean active load), lognormal household draws at cv = 0.5 with P and
  Q tied at 0.95 power factor. **Never re-solved.**
* **The operator** — `S = d|V_far|/dx` of shape 45 × 110 (pu per MW/MVAr), rebuilt from the
  per-line sequence-impedance blocks and the phase labels at **zero AC solves**, verified
  against the archived matrix to 0.0 relative difference. Effective rank 11 at a 1e-2
  threshold, 19 at 1e-3, stable rank 2.177.
* **Folds** — sizes n ∈ {10, 20, 40, 80, 160}, 12 seeds × 2 partition schemes = 24 folds per
  size (12 at n = 160, because scheme 1's training block is 110 rows). 108 realised folds.
* **Five baselines**, all computed once per fold: RBF GP on standardised injections; hop-count
  graph-Laplacian GP; admittance-weighted graph-Laplacian GP built from line data only; ridge
  on the 110 standardised injections; and multiphase LinDistFlow with the exact second-order
  Z-bus correction, calibrated by a family of ten (order × route) members with the member
  chosen on a train-only inner CV. Every margin is scored against the **best** of the five in
  the same fold.
* **Guards** — g1: any fold-level RMSE worse than that fold's mean predictor is recorded as
  `numerical_failure` and never quoted. g2: every ridge/BLR penalty is a fraction of the
  largest Gram eigenvalue, never an absolute number. Both are verified by deliberate misuse
  in `smoke.py`.

## What is genuinely new here

Earlier probes in this line of work compared a GP against a ridge model and stopped. This
artifact adds the comparison that decides the mechanism question — a **fixed degree-2
polynomial feature map in the same coordinates** — and finds it *beats* the kernel. It also
publishes the full ten-member physics-baseline reconciliation, so the margins are not
measured against a weaker baseline than the field would use.

## Scope limits (stated, not hidden)

1. **One feeder.** Mechanism claim only, no generalisation claim.
2. **One loading regime (2.0×).** The 1.0× and 2.6× archives exist in the run tree and are
   deliberately **not read** by this artifact; they are reserved as the survivor's independent
   confirmation evidence. No regime-independence claim is made.
3. **No neural-network baseline.** A DNN introduces a second tuning dimension that is not
   budget-comparable to closed-form ridge. Deferred and named.
4. **`Sx` is still linear in `x`.** The claim is about conditioning and effective dimension at
   fixed n, never about a richer function class.

## Restoring removed files

Nothing a reader needs is deleted. Only two regenerable paths are removed after the round:

```bash
# .venv/  (1.3 GB of installed wheels, regenerated from the pins in pyproject.toml)
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -r pyproject.toml

# code/__pycache__/  (byte-compiled modules, recreated on first import)
uv run method.py            # importing the modules regenerates it
```

## Inputs

This artifact declares no dependency, so its inputs live in the run tree it was built in.
`code/state.py` locates them by walking up from `__file__`, so the code runs from a clone at
any depth; set `PF_RUN_ROOT` to point at a tree containing
`iter_2/gen_hypo/claude_agent/pilot/` and `iter_3/gen_hypo/claude_agent/verify/`. Those four
`.npy` archives (145 KB together) are the only data needed, and `reproducibility.md` gives the
exact recipe that regenerates them from pandapower if they are unavailable.
