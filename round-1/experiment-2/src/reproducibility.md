# Reproducing this artifact

Everything below was actually run on this machine. No step is idealised.

## 1. Getting the artifact

The workspace is published as one folder of a public GitHub repository. Start from a
clone of that repository, then `cd` into this artifact's folder (the directory holding
`method.py`). All paths in this document and in the code are **relative to that folder**:
`method.py` anchors every path on `Path(__file__).resolve().parent`.

## 2. System packages, Python, venv, library versions

Ubuntu (Debian bookworm base), no GPU. 2 CPUs, 14 GB container RAM, ~2.1 PB free disk.
No system packages beyond a working C toolchain are needed — every dependency ships a
manylinux wheel.

```bash
python3 --version          # Python 3.12.14
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python \
    numpy==2.4.6 scipy==1.18.1 networkx==3.7 loguru==0.7.3 \
    matplotlib==3.11.2 pandas==2.3.3 pandapower==3.5.6
```

These are the exact versions that were installed and used (`uv pip list` on the run's
`.venv`); `pyproject.toml` pins the same set. `pandapower==3.5.6` brings pandas 2.3.3,
pandera 0.36.0, pydantic 2.14.3 and tqdm 4.70.1 as transitive pins.

`uv sync` also works, or use a plain venv plus `pip install` with the same pins.

## 3. Data, models, checkpoints, env vars

**No downloads, no model weights, no checkpoints, no API keys.** Ground truth is the
archive of 220 converged `runpp_3ph` solves shipped in `data/`:

| file | contents |
|---|---|
| `data/ens_s2.0_X.npy` | 220×110 injection ensemble, 2.0× stressed regime (55 loads × P,Q) |
| `data/ens_s2.0_Y.npy` | 220×45 far-end phase voltages from `runpp_3ph` |
| `data/sheet15.npy` | the 15 far-end buses |
| `data/ph_true.npy` | the 55 load phase labels |
| `data/physics.py` | feeder physics: LCA path-impedance blocks, LinDistFlow, far-end sheet |
| `data/S_sheet15.npy` | archived operator, used only as a row-space cross-check |

These files were produced by an **earlier artifact of the same run**
(`iter_2/gen_hypo/claude_agent/pilot`), not by the user. They are small (≈350 kB total)
and are published with this folder, so a reader needs nothing else. If they are missing,
`method.py --stage checks` regenerates `ens_s2.0_{X,Y}.npy`, `sheet15.npy` and
`ph_true.npy` from pandapower (220 `runpp_3ph` solves, ≈30 s, seed 11, lognormal cv 0.5,
2.0× scale, P clipped to [0.02e-3, 60e-3] MW, Q = P·tan(arccos 0.95), own-phase
assignment) and records `ensemble_source: REGENERATED` in `results/protocol_checks.json`.

If the user supplied any private input for this artifact it is **not** published; none was
used here.

**Environment variables** — all optional:

* `AII_FIG_GEN` — path to the `chart_gen.py` of the run's `aii-data-fig-gen` skill. If
  unset, a sibling `skills/` path is tried and then the run image's default; if none
  exists, `--stage figures` prints a warning and every numeric output is still written.

## 4. Exact commands, in the order they were run

Hardware: 2 CPU cores, no GPU, 14 GB RAM. `method.py` caps itself at 12 GB
(`RLIMIT_AS`) and runs the sweep on 2 worker processes.

```bash
# full pipeline, from an empty results/ and figures/
rm -rf results figures && mkdir -p results/raw figures
.venv/bin/python method.py --stage all        # 312 s wall clock
```

Stage by stage (each is checkpointed and resumable):

```bash
.venv/bin/python method.py --stage checks     # ~41 s  inputs asserted, S rebuilt, protocol_checks
.venv/bin/python method.py --stage baselines  # ~22 s  108 fold-cells × 7 baselines
.venv/bin/python method.py --stage sweep      # ~118 s 64 variants × 108 fold-cells
.venv/bin/python method.py --stage select     # ~111 s train-only (k,gamma) + controls
.venv/bin/python method.py --stage stats      # ~10 s  aggregation, bootstrap, JSON products
.venv/bin/python method.py --stage figures    # ~35 s  6 figures via the aii-data-fig-gen skill
```

Checks and audit (not part of `--stage all`):

```bash
.venv/bin/python test_gp.py         # GP posterior + both selection criteria vs dense reference
.venv/bin/python test_leakage.py    # T6: test-permutation invariance, train-only calibration
.venv/bin/python test_cell.py       # T2 single-cell smoke test (n=40 and n=160) with timings
.venv/bin/python audit.py           # independent re-derivation of every headline number + placebos
```

Seeds and configs are all at the top of `method.py` and pre-registered:

```
SEED_BASE = 20261111   SEEDS = 0..11   SCHEMES = (0, 1)   N_GRID = (10, 20, 40, 80, 160)
K_GRID = (3, 5, 8, 10, 11, 15, 20, 45)
G_GRID = (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00)
F_GRID = (1e-1 .. 1e-8)   LGRID = (0.02 .. 16)   S2G = (0.3, 1.0, 3.0)   N2G = (1e-8 .. 1e-1)
```

Fold recipe: `default_rng(20261111 + 7919*seed + 131*scheme)` per seed × scheme. Scheme 0
is the complement split, scheme 1 the fixed-half split (train from the first 110 of the
permutation, test the other 110); scheme 1 cannot draw 160 rows from 110, so n=160 has
12 folds, not 24.

Note the seed base differs from `p17_table.py`'s 31337: these are different split
families and are never mixed within one table. This is recorded in
`results/protocol_checks.json`.

## 5. Expected output files and the numbers they should contain

| file | what a reader should see |
|---|---|
| `results/screen_B.json` | exactly ten top-level keys; `screen_metric_at_80 = −0.1514`; `selection_optimism_at_80 = 0.0029`; `survival.survives = false` |
| `results/method_out.json` | 108 examples, one per fold-cell, validated against `exp_gen_sol_out` |
| `results/baselines_per_fold.csv` | 108 rows, 0 NaN; at n=10 `b5p_lindist_honest ≈ 0.1833` and `b5s_p17_jointmap ≈ 0.4315` (T4: honest must beat the archived calibration) |
| `results/protocol_checks.json` | `T4_baseline_order.passes = true`; exponent orientation `cond(gamma=1) = 1.000000`; `operator_crosscheck_vs_archive.elementwise_max_abs_over_max_abs = 0.0` |
| `results/structural_facts.json` | `rank_1e2 = 11`, `rank_1e3 = 19`, `rank_1e4 = 23`, `stable_rank = 2.1768` |
| `results/audit_rederivation.json` | every headline number re-derived; all three placebos fail |
| `figures/{gamma_curve,k_curve,jitter_sweep,rmse_vs_n,selected_hist,margin_vs_n}.png` | the six figures, plus their specs in `results/figure_specs.json` |

Where these appear in the paper:

* `screen_metric_at_80 = −15.1 %` and `survival` — the screen's headline row and the
  candidate-B verdict.
* The n=10/20/40/80/160 margin column — the small-`n` table beside the ranking, where the
  crossover question lives.
* `selection_optimism_at_80` — the row that shows the negative is not grid-shopping.
* `structural_facts` (ranks 11/19/23, stable rank 2.1768) — the operator effective-dimension
  statement.
* The gamma curve and the k curve — the two halves of the whiten-versus-truncate mechanism.

## 6. Known deviations a re-run will see

* `results/screen_B.json` will not reproduce the archived `p17`/`final_table` numbers to
  the digit even where the model is identical, because the error is normalised by the
  **training** rows of each fold rather than the all-row sd. Both denominators are
  reported for the LinDistFlow row at every n.
* `cond(S)` differs by LAPACK path (see README); it is a round-off residue, not a
  reproducible number.
* A regenerated ensemble (archived `.npy` absent) gives slightly different levels — its
  220-draw sample mean and sd differ from 114.6928 kW / 0.021013 pu by sampling noise.
  `results/protocol_checks.json` states which one was used.
