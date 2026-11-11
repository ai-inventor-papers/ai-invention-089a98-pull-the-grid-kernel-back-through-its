"""Fallback F1: rebuild the three archived runpp_3ph ensembles from scratch.

Used only when the sibling artifact's pilot directory is absent (earlier workspaces can be
purged). It re-implements the archived generator exactly:

  per-load multiplier  lognormal(mean 1, coefficient of variation 0.5) on each load's OWN
                       real-phase P0
  clip                 p in [0.02e-3, 60e-3] MW
  reactive             q = p * tan(arccos 0.95)
  scale                1.0 / 2.0 / 2.6 applied to the load's nominal P0 before sampling
  seed 11, N = 220, numba = False
  targets              the 15-bus far-end sheet, from a 0.75 x max-hop cut of the BFS tree
  outputs              X (220 x 110) = per-load P then Q in MW/MVAr,
                       Y (220 x 45) = far-end phase-voltage magnitudes, sheet15, ph_true

The archived arrays were produced with numba=True; this script runs numba=False, so
agreement with the archive is checked and REPORTED rather than assumed. Nothing in the
artifact presents regenerated data as if it came from the archive: `checks.certify_data`
records `regenerated: true` when it falls back here.

Run:  .venv/bin/python code/ens_regenerate.py --out data
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn


def build_sheet(k: int = 15, frac: float = 0.75):
    """The 15-bus far-end sheet, from hop distances on the line graph."""
    import networkx as nx

    base = pn.ieee_european_lv_asymmetric("on_peak_566")
    G = nx.Graph()
    for _, r in base.line.iterrows():
        G.add_edge(int(r.from_bus), int(r.to_bus))
    hop = nx.single_source_shortest_path_length(G, 1)
    mh = max(hop.values())
    far = np.array(sorted([b for b, d in hop.items() if d >= frac * mh]))
    s15 = np.array(sorted(set(far[np.linspace(0, len(far) - 1, k).astype(int)])))
    return far, s15


def ensemble(scale: float, cv: float, n_draws: int, seed: int, targets: np.ndarray,
             ph: np.ndarray, p0: np.ndarray):
    net = pn.ieee_european_lv_asymmetric("on_peak_566")
    nload = len(p0)
    rng = np.random.default_rng(seed)
    X = np.zeros((n_draws, 2 * nload))
    Y = np.full((n_draws, 3 * len(targets)), np.nan)
    t0 = time.time()
    for i in range(n_draws):
        m = rng.lognormal(-0.5 * np.log(1 + cv ** 2), np.sqrt(np.log(1 + cv ** 2)), nload)
        p = p0.sum(1) * scale * m
        p = np.clip(p, 0.02e-3, 60e-3)
        q = p * np.tan(np.arccos(0.95))
        al = net.asymmetric_load
        for j, (c1, c2) in enumerate([("p_a_mw", "q_a_mvar"), ("p_b_mw", "q_b_mvar"),
                                      ("p_c_mw", "q_c_mvar")]):
            al[c1] = np.where(ph == j, p, 0.0)
            al[c2] = np.where(ph == j, q, 0.0)
        try:
            pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
            v = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].values
            if net.converged and not np.isnan(v).any():
                Y[i] = v[targets].ravel()
        except Exception:                                   # noqa: BLE001 - recorded as NaN
            pass
        X[i] = np.concatenate([p, q])
    return X, Y, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--cv", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = pn.ieee_european_lv_asymmetric("on_peak_566")
    al = base.asymmetric_load
    p0 = al[["p_a_mw", "p_b_mw", "p_c_mw"]].values.copy()
    ph = np.argmax(p0, axis=1)
    far, s15 = build_sheet(15)
    print(f"loads {len(p0)}, phases {np.bincount(ph, minlength=3)}, "
          f"far {len(far)}, sheet {len(s15)}")
    np.save(out / "sheet15.npy", s15)
    np.save(out / "ph_true.npy", ph)
    for scale in (1.0, 2.0, 2.6):
        X, Y, dt = ensemble(scale, args.cv, args.n, args.seed, s15, ph, p0)
        np.save(out / f"ens_s{scale}_X.npy", X)
        np.save(out / f"ens_s{scale}_Y.npy", Y)
        print(f"  scale {scale}: {dt:.0f}s, mean kW {1000*X[:, :len(p0)].sum(1).mean():.2f}, "
              f"far-end sd {np.nanstd(Y, 0).mean():.5f} pu, "
              f"failed rows {int(np.isnan(Y).any(1).sum())}")
    print(f"wrote ensembles to {out}/ -- protocol_checks must record regenerated=true")


if __name__ == "__main__":
    main()
