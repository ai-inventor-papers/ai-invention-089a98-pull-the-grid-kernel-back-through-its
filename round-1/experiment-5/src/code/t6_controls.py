"""DECISIVE CONTROLS at n = 80 on the screen folds.

The screen's spectral cells reach ~0.021 normalised RMSE where a GP on the full 110-dim
injection vector reaches ~0.145. A factor-7 gain from strictly FEWER coordinates cannot
come from extra information; it must come from the geometry of the coordinates. This
script decides WHICH geometry does the work by holding dimension (2k = 22) and learner
(the fixed GP protocol) constant and changing only how the 55x11 block is chosen:

  C1 laplacian      k=11 Laplacian eigenvectors at each load's own phase (the candidate)
  C2 random         a random 55x11 Gaussian block (no network, no data)
  C3 permuted_load  a random permutation of C1's ROWS: same spectrum, same phase labels,
                    but each eigenvector value attached to the wrong load
  C4 shuffled_phase C1 gathered at SHUFFLED phase labels (the request-named control)
  C5 pca_injection  the top 11 principal components of the standardised injections
  C6 identity11     11 fixed indicator coordinates over the 55 loads
  C7 hop_laplacian  the k=11 eigenvectors of the hop-count graph Laplacian (structure
                    without impedance and without phase)

If C2 (random) matches C1, the Laplacian selects nothing and the margin is a conditioning
artefact of working in 22 dimensions. If C1 beats C2, C3 and C4, the feeder's own
spectrum and its phase labels are what the margin rests on.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "code")
from baselines import standardise  # noqa: E402
from controls import shuffled_phase_labels  # noqa: E402
from engine import project_injections  # noqa: E402
from folds import fold_indices  # noqa: E402
from learners import gp_fit_predict, normalise_distance, sqd  # noqa: E402
from method_core import prepare  # noqa: E402

K = 11


def gp_nrmse(Atr, Ate, Ytr, Yte):
    Dtr, Dst = normalise_distance(sqd(Atr, Atr), sqd(Ate, Atr), Atr.shape[1])
    mu, pv, hp = gp_fit_predict(Dtr, Ytr, Dst)
    return float(np.sqrt(((mu - Yte) ** 2).mean()) / Ytr.std(0).mean()), hp["hyper"]


def run_block(X, Y, tr, te, Fk):
    Atr, Ate = project_injections(X[tr], X[te], Fk)
    Atr, Ate, _ = standardise(Atr, Ate)
    return gp_nrmse(Atr, Ate, Y[tr], Y[te])


def gather_at(ops, V, phase_labels):
    """Regather a (2715,q) eigenvector block at the given per-load phase labels."""
    lidx = ops.lidx
    full = 3 * lidx + np.asarray(phase_labels)
    # full (3M) index -> grounded index, straight from the grounding map
    assert (ops.gmap[full] >= 0).all(), "shuffled gather touched a grounded-out index"
    g = ops.gmap[full]
    return V[g, :]


if __name__ == "__main__":
    prep = prepare()
    ops, X, Y = prep["ops"], prep["X"], prep["Y"]
    phase = prep["phase"]
    rng = np.random.default_rng(7)

    V1 = ops.routes["R1"].Vraw
    if V1 is None:
        raise SystemExit("cache has no raw eigenvectors; delete results/spectra_cache.npz "
                         "and rerun so the C4 shuffled-phase control can be built")
    F_cand = ops.features("R1", K, "real")
    F_shuf = gather_at(ops, V1[:, :K].real, shuffled_phase_labels(phase))

    F_rand = rng.normal(size=(55, K))
    F_permload = F_cand[rng.permutation(55), :]
    F_ind = np.zeros((55, K))
    F_ind[np.linspace(0, 54, K).astype(int), np.arange(K)] = 1.0
    Lh = prep["Mhop"].copy()
    np.fill_diagonal(Lh, 0.0)
    Lh = np.diag(Lh.sum(1)) - Lh
    _, Vh = np.linalg.eigh(Lh)
    F_hop = Vh[:, 1:K + 1]

    blocks = {
        "C1_laplacian": F_cand,
        "C2_random": F_rand,
        "C3_permuted_load_rows": F_permload,
        "C4_shuffled_phase": F_shuf,
        "C6_identity11": F_ind,
        "C7_hop_laplacian": F_hop,
    }
    res: dict = {k: [] for k in blocks}
    res["C5_pca_injection"] = []

    for sch in (0, 1):
        for s in range(12):
            f = fold_indices(80, s, sch)
            if f is None:
                continue
            tr, te = f
            for tag, Fb in blocks.items():
                res[tag].append(run_block(X, Y, tr, te, Fb))
            Ztr, Zte, _ = standardise(X[tr], X[te])
            _, _, Vt = np.linalg.svd(Ztr, full_matrices=False)
            B = Vt[:2 * K].T                  # 2k = 22 coordinates, dose-matched
            res["C5_pca_injection"].append(gp_nrmse(Ztr @ B, Zte @ B, Y[tr], Y[te]))

    base = np.mean([e for e, _ in res["C1_laplacian"]])
    print(f"\n=== n = 80, {len(res['C1_laplacian'])} folds, 2k = {2*K} coordinates ===")
    print(f"{'block':<24} {'mean nrmse':>11} {'worst':>8} {'best':>8} {'mean l':>7}  vs C1")
    for tag, vals in res.items():
        e = np.array([v[0] for v in vals])
        ls = np.array([v[1][0] for v in vals])
        print(f"{tag:<24} {e.mean():11.4f} {e.max():8.4f} {e.min():8.4f} "
              f"{ls.mean():7.1f}   {e.mean()/base:5.2f}x")
    json.dump({k: [float(v[0]) for v in vals] for k, vals in res.items()},
              open("results/controls_n80.json", "w"), indent=1)
    print("\nwrote results/controls_n80.json")
