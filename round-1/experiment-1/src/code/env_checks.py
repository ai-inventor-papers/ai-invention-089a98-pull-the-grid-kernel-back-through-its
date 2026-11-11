"""Structural checks: the Laplacian / Z-bus identity, the finite-difference
cross-check of the analytic operator, the transformer convention sweep, and the
leakage and determinism tests.

Nothing in this module produces a headline number.  It produces the evidence that
the numbers mean what they claim: that the operator really is a Z-bus object, that
it agrees with a converged AC Jacobian to the accuracy a first-order LinDistFlow
model can, and that no test row enters any fitted quantity.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------- identity (zero AC solves)
def identity_checks(ph, targets) -> dict:
    """Z_bus via the grounded block Laplacian vs via BFS path accumulation.

    Acceptance, from the artifact plan: scalar identity <= 1e-13, block identity
    without transformer <= 1e-10 (the archive reported 3e-12).  Both are measured
    here rather than inherited.
    """
    out = {}
    out.update(_scalar_identity(ph))
    for tr in ("none", "pu_correct", "legacy"):
        Zs = ph.z_bus_solve(targets, ph.phase, tr)
        Zp = ph.zc_path_matrix(targets, ph.phase, tr)
        rel = float(np.abs(Zs - Zp).max() / np.abs(Zp).max())
        out[f"block_rel_{tr}"] = rel
        out[f"block_pass_{tr}"] = bool(rel <= 1e-10)
        if tr == "none":
            S_asm = ph.lin_operator(targets, ph.phase, "none", "flat")
            S_ref = _assembly_reference(ph, targets, "none", "flat")
            out["assembly_rel_none"] = float(np.abs(S_asm - S_ref).max() /
                                             np.abs(S_ref).max())
            out["assembly_pass_none"] = bool(out["assembly_rel_none"] <= 1e-8)
    out["acceptance"] = dict(scalar="<=1e-13", block="<=1e-10", assembly="<=1e-8")
    return out


def _scalar_identity(ph) -> dict:
    """Single-phase Laplacian identity: on a one-conductor model the impedance to a
    source-grounded resistive network must equal the path impedance exactly."""
    import networkx as nx
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    buses = list(ph.buses)
    idx = {int(b): i for i, b in enumerate(buses)}
    n = len(buses)
    rows, cols, vals = [], [], []
    for _, r in ph.net.line.iterrows():
        u, v = int(r.from_bus), int(r.to_bus)
        if u not in idx or v not in idx:
            continue
        z = (float(r.r_ohm_per_km) + 1j * float(r.x_ohm_per_km)) * float(r.length_km)
        y = 1.0 / z
        iu, iv = idx[u], idx[v]
        rows += [iu, iv, iu, iv]
        cols += [iu, iv, iv, iu]
        vals += [y, y, -y, -y]
    L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    i0 = idx[ph.source]
    keep = [k for k in range(n) if k != i0]
    Lg = L[keep][:, keep].tocsc()
    lu = spla.splu(Lg)
    cols_load = [idx[int(b)] for b in ph.lbus]
    E = np.zeros((len(keep), len(cols_load)))
    pos = {k: i for i, k in enumerate(keep)}
    for j, c in enumerate(cols_load):
        if c in pos:
            E[pos[c], j] = 1.0
    Zsol = lu.solve(E)
    Zpath_full = np.zeros((n, len(cols_load)), dtype=complex)
    Zs1 = {ph.source: 0.0 + 0.0j}
    for u, v in nx.bfs_edges(ph.tree, ph.source):
        r = ph._line_record(int(u), int(v))
        Zs1[int(v)] = Zs1[int(u)] + (float(r.r_ohm_per_km) +
                                     1j * float(r.x_ohm_per_km)) * float(r.length_km)
    if not hasattr(ph, "_depth"):
        ph._build_lifting()
    Zbus_arr = np.array([Zs1[int(b)] for b in buses], dtype=complex)
    tb_arr = np.arange(n, dtype=np.int32)
    for j, lb in enumerate(ph.lbus):
        A, B = np.meshgrid(tb_arr, np.int32(idx[int(lb)]), indexing="ij")
        lca = ph.lca_index(A.ravel(), B.ravel())
        Zpath_full[:, j] = Zbus_arr[lca]
    Zpath = Zpath_full[keep]
    rel = float(np.abs(Zsol - Zpath).max() / np.abs(Zpath).max())
    return dict(scalar_rel=rel, scalar_pass=bool(rel <= 1e-13))


def _assembly_reference(ph, targets, tr_mode, v0mode):
    """An independent re-statement of d|V|/dx from the (3m,55) complex operator."""
    Zt = ph.zc_path_matrix(targets, ph.phase, tr_mode) / ph.Zb
    V0t = ph.v0_at(targets, v0mode)
    V0l = ph.load_reference_voltage(ph.phase, v0mode)
    W = np.conj(V0t) / np.abs(V0t)
    dPd = -ph.k3 * Zt / (ph.Sb * np.conj(V0l))[None, :]
    S = np.zeros((dPd.shape[0], 2 * ph.nload), dtype=float)
    S[:, :ph.nload] = np.real(W[:, None] * dPd)
    S[:, ph.nload:] = np.real(W[:, None] * (-1j * dPd))
    return S


# ------------------------------------------------- finite-difference cross-check
def transformer_convention_sweep(ph, targets, jac) -> dict:
    """All six (V0 convention) x (transformer convention) cells against the archived
    finite-difference complex AC Jacobian at a 2.0x mid-range draw."""
    A = jac["A"]
    Vref = jac["Vref"]
    W = np.conj(Vref) / np.abs(Vref)
    S_fd = np.real(W[:, None] * A)
    cells = {}
    for v0m in ("flat", "noload"):
        for tr in ("none", "pu_correct", "legacy"):
            S = ph.lin_operator(targets, ph.phase, tr, v0m)
            alpha = float((S * S_fd).sum() / (S * S).sum())
            cells[f"{v0m}|{tr}"] = dict(
                alpha=alpha,
                rel_raw=float(np.linalg.norm(S - S_fd) / np.linalg.norm(S_fd)),
                rel_rescaled=float(np.linalg.norm(alpha * S - S_fd) /
                                   np.linalg.norm(S_fd)),
                corr=float(np.corrcoef(S.ravel(), S_fd.ravel())[0, 1]),
                per_output_corr=[float(np.corrcoef(S[i], S_fd[i])[0, 1])
                                 for i in range(S.shape[0])],
            )
    best = min(cells, key=lambda k: cells[k]["rel_rescaled"])
    spread = max(c["rel_rescaled"] for c in cells.values()) - \
        min(c["rel_rescaled"] for c in cells.values())
    return dict(available=True, cells=cells, best_cell=best,
                best_rel_rescaled=cells[best]["rel_rescaled"],
                spread_across_cells=float(spread),
                convention_matters=bool(spread > 1e-3),
                adopted=best,
                note="a first-order LinDistFlow operator cannot match a four-wire AC "
                     "Jacobian exactly; the residual is the neutral/earth-return and "
                     "second-order content, reported rather than hidden")


def near_operator(ph, near):
    if near is None or not near["available"]:
        return None
    return ph.lin_operator(list(near["buses"]), ph.phase, "pu_correct", "noload")


def all90_operator(ph) -> np.ndarray:
    """30 buses evenly spaced over the FULL hop range x 3 phases = 90 rows."""
    hop = ph.hop
    cand = sorted(b for b, d in hop.items() if d >= 4)
    hmax = max(hop[b] for b in cand)
    hmin = min(hop[b] for b in cand)
    chosen = set()
    for w in np.linspace(hmin, hmax, 30):
        b = min(cand, key=lambda x: (abs(hop[x] - w), x))
        chosen.add(int(b))
    for b in cand:
        if len(chosen) >= 30:
            break
        chosen.add(int(b))
    chosen = sorted(chosen)[:30]
    return ph.lin_operator(chosen, ph.phase, "pu_correct", "noload")


def all55_operator(ph) -> np.ndarray:
    """The load-restricted admittance Z-bus metric's own operator: 55 rows, one per
    load bus on the phase that bus's load sits on, with both P and Q columns.

    This is the 'unrestricted 55-dimensional Z-bus metric' of the mechanism section:
    it is the full-state metric's load-restricted cousin (it is unfiltered by
    distance, unlike S_far, but it does not cover every phase of every bus).
    """
    Sfull = ph.lin_operator(list(ph.lbus), ph.phase, "pu_correct", "noload")  # (165,110)
    rows = [3 * i + int(ph.phase[i]) for i in range(ph.nload)]
    return Sfull[rows]


# ------------------------------------------------- leakage and determinism
def determinism_and_leakage_checks(ph, targets, ens) -> dict:
    """Recompute folds twice and, separately, with the test block replaced by a
    disjoint dummy block."""
    from folds import make_fold
    from screen import run_fold_metric
    from operators import build_operator
    from kernels import gp_fit_predict, sqdist

    X, Y = ens["X"], ens["Y"]
    S = build_operator(ph, targets, ph.phase, "pu_correct", "noload")
    out = {"folds_checked": [], "deterministic": True, "identical_hyperparameters": True}
    for n, seed, scheme in ((20, 0, 0), (80, 3, 1), (160, 5, 0)):
        tr, te = make_fold(n, seed, scheme)
        mx = X[tr].mean(0)
        Ftr = (X[tr] - mx) @ S.T
        sc = np.sqrt(np.einsum("ij,ij->i", Ftr, Ftr).mean())
        Ftr = Ftr / sc
        Dtr = sqdist(Ftr, Ftr)
        Ytr = Y[tr]
        r1 = gp_fit_predict(Ftr, Ftr, Ytr, Dtr=Dtr, want_var=False, refine=False)
        r2 = gp_fit_predict(Ftr, Ftr, Ytr, Dtr=Dtr, want_var=False, refine=False)
        det = bool(r1["ell"] == r2["ell"] and r1["s2"] == r2["s2"] and
                   r1["n2"] == r2["n2"] and np.array_equal(r1["mu"], r2["mu"]))
        # Replace the test block with a DISJOINT dummy block.  Rows outside the union
        # are used where they exist; at n = 20 under the complement scheme the test
        # block IS every non-train row, so the dummy is a deterministic synthetic block
        # drawn from TRAIN statistics only.  Either way it shares no row with the data,
        # which makes this test strictly stronger than a row swap.
        avail = np.setdiff1d(np.arange(len(X)), np.union1d(tr, te))
        Xd = X.copy()
        Yd = Y.copy()
        if len(avail) >= len(te):
            dummy = avail[: len(te)]
            Xd[te] = X[dummy]
            Yd[te] = Y[dummy]
            dummy_kind = "real rows disjoint from train"
        else:
            rng = np.random.default_rng(20261111)
            Xd[te] = X[tr].mean(0) + rng.normal(size=(len(te), X.shape[1])) * X[tr].std(0)
            Yd[te] = Y[tr].mean(0) + rng.normal(size=(len(te), Y.shape[1])) * Y[tr].std(0)
            dummy_kind = "synthetic block from train statistics (no disjoint real row exists)"
        vsd = float(Y[tr].std(0).mean())
        rec_a = run_fold_metric(S, X, Y, tr, te, vsd, want_var=False)
        rec_b = run_fold_metric(S, Xd, Yd, tr, te, vsd, want_var=False)
        same_hp = bool(rec_a["ell"] == rec_b["ell"] and rec_a["s2"] == rec_b["s2"] and
                       rec_a["n2"] == rec_b["n2"])
        out["folds_checked"].append(dict(n=int(n), seed=int(seed), scheme=int(scheme),
                                         deterministic=det,
                                         identical_hyperparameters=same_hp,
                                         n_dummy_rows=int(len(te)),
                                         dummy_kind=dummy_kind))
        out["deterministic"] &= det
        out["identical_hyperparameters"] &= same_hp
    out["pass"] = bool(out["deterministic"] and out["identical_hyperparameters"])
    out["note"] = ("the test block is replaced by a disjoint dummy block of the same "
                   "size; if any fitted quantity changed, a test row was leaking into "
                   "the fit.  The TEST error is of course different -- only the fitted "
                   "quantities are required to be identical.")
    return out


def reserved_evidence_audit() -> dict:
    from folds import SIZES, SEEDS, SCHEMES, RESERVED_SIZES, RESERVED_SEED_MIN
    bad = []
    for n in SIZES:
        if n in RESERVED_SIZES:
            bad.append(f"size {n} requested but reserved")
    for s in SEEDS:
        if s >= RESERVED_SEED_MIN:
            bad.append(f"seed {s} requested but reserved")
    if len(SCHEMES) != 2:
        bad.append(f"{len(SCHEMES)} partition schemes coded, exactly 2 permitted")
    return dict(sizes=list(SIZES), seeds=list(SEEDS), schemes=list(SCHEMES),
                reserved_sizes=list(RESERVED_SIZES),
                reserved_seed_min=RESERVED_SEED_MIN,
                violations=bad, passed=bool(not bad),
                reserved_2_6x_regime="never opened on any code path; guarded in "
                                     "code/data.py::_guard")


def baseline_sanity(ph, targets, ens) -> dict:
    """T5: guard-g1 ratios and the LinDistFlow flatness signature."""
    from folds import make_fold
    from screen import run_fold
    X, Y = ens["X"], ens["Y"]
    out = {}
    for n in (10, 160):
        tr, te = make_fold(n, 0, 0)
        vsd = float(Y[tr].std(0).mean())
        rec = run_fold(X, Y, tr, te, ph, targets, ph.phase, vsd, want_var=False)
        out[f"n{n}"] = {k: dict(rmse=rec[k]["rmse"],
                                over_mean=float(rec[k]["rmse"] / rec["mean"]["rmse"]),
                                **({"cell": rec[k]["cell"]} if "cell" in rec[k] else {}))
                        for k in ("mean", "treatment", "b1_rbf", "b2_hop", "b3_adm",
                                  "b4_ridge", "b5_lindistflow", "b5_legacy") if k in rec}
    try:
        a = out["n10"]["b5_lindistflow"]["rmse"]
        b = out["n160"]["b5_lindistflow"]["rmse"]
        out["lindistflow_flat_ratio_n160_over_n10"] = float(b / a)
        out["lindistflow_flat_pass"] = bool(0.7 < b / a < 1.4)
    except KeyError:
        pass
    return out
