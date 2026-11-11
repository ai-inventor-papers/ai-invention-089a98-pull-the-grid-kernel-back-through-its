# Reproducing the screen of candidate E

Everything here reproduces the numbers in `results/screen_E.json` and `method_out.json`
on Ubuntu from a clone of the published repository. Every path is relative to this
artifact's folder; the code anchors itself on `Path(__file__)`.

---

## 1. Get the artifact

The workspace is published as one folder of a public GitHub repository.

```bash
git clone <repository-url> repo
cd repo/round-1/experiment-5/src
```

## 2. System packages and Python

Ubuntu 22.04 or 24.04, `python3.12` and `uv`:

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Virtual environment and the exact pinned dependency set (`pyproject.toml` carries these
same versions; `uv` resolves them from there):

```bash
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python numpy==2.4.6 scipy==1.18.1 pandas==2.3.3 \
  networkx==3.7 pandapower==3.5.6 loguru==0.7.3 psutil==7.2.2 matplotlib==3.11.2 pillow==13.0.0
```

`pandapower==3.5.6` is not incidental: the feeder, its `on_peak_566` loading case and the
`runpp_3ph` solution all come from it, and a different version can shift the ground truth.

## 3. Data and inputs

**No download is needed and no API key is used.** The heavy inputs are the archived
`runpp_3ph` ensembles, which are an **input produced by a sibling artifact** of the same
run (`gen_hypo` iteration 2, artifact id `gen_hypo_iter2_pilot`). The repository publishes
that artifact as a sibling folder, so from this artifact's directory it is:

```bash
export PILOT_DIR=../../../../iter_2/gen_hypo/claude_agent/pilot
```

`code/method_core.py` holds that as the single constant `PILOT`; point it at your own copy
if your layout differs. The five files it reads are `ens_s2.0_X.npy` (220x110),
`ens_s2.0_Y.npy` (220x45), `sheet15.npy`, `ph_true.npy` — and, for the data certification
only, `ens_s1.0_*` and `ens_s2.6_*`. If the sibling folder is absent, `code/ens_regenerate.py`
rebuilds all three ensembles from scratch with `runpp_3ph` (about 640 s at N = 220 on 2 vCPU) and records in
`results/protocol_checks.json` that they were regenerated rather than reused. Measured
during this run, the rebuilt arrays are **bit-identical** to the archive (max |diff| = 0
for all three scales, both `X` and `Y`, plus `sheet15` and `ph_true`), so a reader who has
to regenerate gets exactly the numbers reported here.

The user uploaded no data for this artifact; `user_uploads/` is empty. **Nothing here is
private** — the feeder ships inside pandapower.

## 4. Commands actually run, in order

Each was run with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2` (2 vCPU pod, 16 GB).

```bash
# (a) smoke checks: data certification, feeder certification, toy Laplacian unit test
.venv/bin/python code/checks.py

# (b) build the Laplacian, verify the bus-impedance identity, compute every spectral route
#     ~265 s (a 2715-square general complex eigenproblem plus a bounded eigsh probe)
.venv/bin/python code/_build.py          # writes results/spectra_cache.npz + .json

# (c) one-fold end-to-end (guard g1/g3 gates) — fast, run before spending the sweep
.venv/bin/python code/t4_fold.py

# (d) THE SCREEN: 5 sizes x realized folds x 12 spectral cells + 7 baselines, ~510 s
.venv/bin/python method.py               # writes results/per_fold_raw.json incrementally

# (e) controls: random, row-permuted, shuffled-phase, PCA, hop-Laplacian, indicators
.venv/bin/python code/t6_controls.py

# (f) decomposition: OLS vs wide-RBF GP on spectral / PCA / random coordinates
.venv/bin/python code/t8_decomp.py

# (g) re-derive every headline number by a different code path, plus the null control
.venv/bin/python code/audit.py

# (h) figures
.venv/bin/python code/make_specs.py
.venv/bin/python ../.claude/skills/aii-data-fig-gen/scripts/chart_gen.py \
  --spec figures/specs/fig1_learning_curve.json --out figures/fig1_learning_curve
.venv/bin/python ../.claude/skills/aii-data-fig-gen/scripts/chart_gen.py \
  --spec figures/specs/fig2_controls.json --out figures/fig2_controls
```

To regenerate the outputs from an existing sweep without re-running the folds:

```bash
.venv/bin/python method.py --reuse-sweep      # ~22 s
```

**Seeds.** The fold schedule is fixed by the pre-registration and needs no seed argument:
`numpy.random.default_rng(20261111 + 7919*seed + 131*scheme)` with `seed` in 0..11 and
`scheme` in {0, 1}. The GP's internal CV split uses `numpy.random.default_rng(0)`. The
control permutations use seed 20261111 (phase labels) and 7 / 11 (random blocks).

**Hardware.** CPU only, 2 vCPU, 16 GB RAM, no GPU. Total wall clock about 15 minutes,
of which 265 s is the one-off spectral build and 510 s the sweep.

## 5. What a reader should get, and where it appears

| Output | What it holds | Where the paper uses it |
|---|---|---|
| `results/screen_E.json` | the full table: 12 spectral cells + 7 baselines x 5 sizes x realized folds, per-fold arrays, paired margins, NLPD/coverage, `screen_metric_at_80`, `crossover_ac_runs`, `survival` (S1/S2, the coverage arbiter, the controls and the decomposition readings) | headline result table, the crossover paragraph, the limitations section |
| `results/protocol_checks.json` | data certification, fold-count audit `{10:24,20:24,40:24,80:24,160:12}`, `|L-L^T|` and `|L-L^H|`, the two identity deviations with scopes, eigenpair residuals, sign/phase convention, guard-g3 rates, Cholesky-jitter count, versions | the "verification" subsection |
| `results/spectra.npz` | eigenvalues per route, the 55x45 gathered blocks, `S`, its singular values, principal-angle arrays, the three metrics | the mechanism figure and any re-check of the geometry |
| `results/controls_n80.json`, `results/decomposition_n80.json` | per-fold control and decomposition errors | the attribution subsection |
| `figures/fig1_learning_curve.{pdf,png}` | normalised RMSE against AC training runs | headline figure |
| `figures/fig2_controls.{pdf,png}` | the control comparison at 80 runs | attribution figure |
| `method_out.json` (+ `full_`, `mini_`, `preview_`) | one record per (size, cell) with the exact key names a downstream assertion looks for | machine-readable summary |

**Headline numbers to expect at n = 80** (24 folds): candidate `SP-R1a-GP-k11`
mean normalised far-end RMSE **0.0222**, worst fold 0.0319, paired margin **+76.7%**
(mean) and **+69.4%** (worst fold); `SP-R1a-GP-k11-S2` 0.0210 / +77.9% / +70.1%; best
baseline `b5_lindist` **0.0944**; `b1_rbf` 0.1449; `b4_ridge` 0.1482; mean predictor 1.1500.
Crossover: **20 AC runs** on the mean criterion, **40 AC runs** for every fold to clear
+20 %. Coverage at 68 % is **0.009**, i.e. the predictive density is not usable — see
`survival.coverage_arbiter`.

**One corrective note.** An earlier draft of these numbers used a ridge solved as a normal
equation system with a free unpenalised intercept, which is singular at n = 10 with 110
columns; it reported `b4_ridge` at 3093 and `b5_lindist` at 1.86 and inflated every margin.
The ridge is now an augmented least-squares problem solved by `lstsq`. The b5 values above
are the corrected ones and match the independent reviewer's 0.1665 (n = 10) and 0.0974
(n = 80) to within a few percent. Reproducing this file gives the corrected, smaller
margins; the superseded per-fold records are kept in
`results/per_fold_raw_before_ridgefix.json` for audit but are NOT used by any summary.

## 6. Restoring removed files

Nothing is deleted. `results/_Lg.npy` (118 MB) and `results/_M110.npy` are transient
probe scratch from step 2 and are marked `delete: regenerable` in `.aii/manifest.yaml`;
`results/spectra_cache.npz` is rebuilt by `code/_build.py`. Everything else — code,
results, figures, logs — stays on the run's storage volume and, being text or small
arrays, is published in the repository.
