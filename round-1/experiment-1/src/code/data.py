"""Resolve the AC ground-truth ensembles: archived if present, regenerated if not.

Archived (preferred): iter_2/gen_hypo/claude_agent/pilot/ens_s{1.0,2.0}_{X,Y}.npy
  220 draws x 110 injections; 220 draws x 45 far-end phase-voltage magnitudes.
  These represent 660 runpp_3ph solves shared across the whole run tree.

Regeneration recipe (documented, deterministic, ~60-90 s for both regimes):
  pandapower 3.5.6, network ieee_european_lv_asymmetric('on_peak_566'),
  seed 11, per-load lognormal multiplier with mean 1 and coefficient of variation
  0.5, per-load base power = that load's native on-peak P scaled by the regime
  factor, clipped to [0.02e-3, 60e-3] MW, Q = P tan(arccos 0.95), numba=False,
  220 draws.  Verification on regeneration: mean total load, finite rows, and the
  per-output standard deviation of the far-end sheet.

The 2.6x ensemble is RESERVED confirmation evidence and is never opened: every load
path goes through a guard that refuses a path containing '2.6'.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[4]
PILOT = RUN_ROOT / "iter_2/gen_hypo/claude_agent/pilot"
VERIFY = RUN_ROOT / "iter_3/gen_hypo/claude_agent/verify"

N_DRAWS = 220
N_LOAD = 55
N_TARGET = 45
SEED_RECIPE = 11
CV_RECIPE = 0.5
CLIP_MW = (0.02e-3, 60e-3)
PF = 0.95
EXPECTED_MEAN_KW = {"1.0": 57.491, "2.0": 114.693}
EXPECTED_FAR_SD = {"1.0": 0.009686, "2.0": 0.021013}


class ReservedEvidenceViolation(RuntimeError):
    pass


def _guard(path: Path) -> Path:
    if "2.6" in str(path):
        raise ReservedEvidenceViolation(f"refusing to open reserved evidence: {path}")
    return path


def regenerate_ensemble(scale: str, n_draws: int = N_DRAWS, seed: int = SEED_RECIPE):
    """Deterministic regeneration of one regime's ensemble."""
    import pandapower as pp
    import pandapower.networks as pn

    base = pn.ieee_european_lv_asymmetric("on_peak_566")
    al0 = base.asymmetric_load
    P0 = al0[["p_a_mw", "p_b_mw", "p_c_mw"]].values
    ph = np.argmax(P0, axis=1)
    net = pn.ieee_european_lv_asymmetric("on_peak_566")
    al = net.asymmetric_load
    far, s15 = _sheet_from_net(base)
    rng = np.random.default_rng(seed)
    X = np.zeros((n_draws, 2 * len(P0)))
    Y = np.zeros((n_draws, 3 * len(s15)))
    for i in range(n_draws):
        m = rng.lognormal(-0.5 * np.log(1 + CV_RECIPE ** 2),
                          np.sqrt(np.log(1 + CV_RECIPE ** 2)), len(P0))
        p = np.clip(P0.sum(1) * float(scale) * m, *CLIP_MW)
        q = p * np.tan(np.arccos(PF))
        for j, (c1, c2) in enumerate([("p_a_mw", "q_a_mvar"), ("p_b_mw", "q_b_mvar"),
                                      ("p_c_mw", "q_c_mvar")]):
            al[c1] = np.where(ph == j, p, 0.0)
            al[c2] = np.where(ph == j, q, 0.0)
        pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
        v = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].values
        Y[i] = v[s15].ravel()
        X[i] = np.concatenate([p, q])
    return X, Y


def _sheet_from_net(net):
    import networkx as nx
    G = nx.Graph()
    for _, r in net.line.iterrows():
        G.add_edge(int(r.from_bus), int(r.to_bus))
    hop = nx.single_source_shortest_path_length(G, 1)
    mx = max(hop.values())
    far = np.array(sorted([b for b, d in hop.items() if d >= 0.75 * mx]))
    s15 = np.array(sorted({int(far[i]) for i in np.linspace(0, len(far) - 1, 15).astype(int)}))
    return far, s15


def load_ensemble(scale: str, allow_regenerate: bool = True) -> dict:
    """Return dict(X, Y, provenance, checks)."""
    assert scale in ("1.0", "2.0"), f"regime {scale} not permitted here"
    fx = _guard(PILOT / f"ens_s{scale}_X.npy")
    fy = _guard(PILOT / f"ens_s{scale}_Y.npy")
    checks = {}
    if fx.exists() and fy.exists():
        X = np.load(fx)
        Y = np.load(fy)
        provenance = "archived"
    elif allow_regenerate:
        X, Y = regenerate_ensemble(scale)
        provenance = "regenerated"
    else:
        raise FileNotFoundError(f"no ensemble for regime {scale} and regeneration disabled")
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    checks["n_rows"] = int(len(X))
    checks["n_finite_rows"] = int(ok.sum())
    checks["shape_X"] = list(X.shape)
    checks["shape_Y"] = list(Y.shape)
    checks["mean_total_load_kw"] = float(X[:, :N_LOAD].sum(1).mean() * 1e3)
    checks["far_end_sd_pu"] = float(Y.std(0).mean())
    checks["matches_archive_mean_load"] = bool(
        abs(checks["mean_total_load_kw"] - EXPECTED_MEAN_KW[scale]) < 0.05)
    checks["matches_archive_far_sd"] = bool(
        abs(checks["far_end_sd_pu"] - EXPECTED_FAR_SD[scale]) / EXPECTED_FAR_SD[scale] < 0.01)
    return dict(X=X, Y=Y, provenance=provenance, checks=checks, scale=scale)


def load_sheet_and_phases(physics) -> dict:
    """sheet15 and ph_true: recomputed locally, asserted against the archive if present."""
    far, s15_local = physics.sheet(15)
    ph_local = physics.phase.astype(int)
    out = dict(far=far, sheet15=s15_local, phase=ph_local,
               sheet_source="recomputed", phase_source="recomputed",
               archived_sheet_agrees=None, archived_phase_agrees=None)
    fs = _guard(PILOT / "sheet15.npy")
    fp = _guard(PILOT / "ph_true.npy")
    if fs.exists():
        a = np.load(fs).astype(int)
        out["archived_sheet_agrees"] = bool(np.array_equal(a, s15_local))
        out["sheet_source"] = "recomputed (archive present, agrees)"
        if not out["archived_sheet_agrees"]:
            raise AssertionError(f"recomputed sheet15 {s15_local} != archived {a}")
    if fp.exists():
        a = np.load(fp).astype(int)
        out["archived_phase_agrees"] = bool(np.array_equal(a, ph_local))
        out["phase_source"] = "recomputed (archive present, agrees)"
        if not out["archived_phase_agrees"]:
            raise AssertionError("recomputed phase labels != archived ph_true")
    return out


def load_near_sheet() -> dict:
    """Near-end target ensemble (120 draws x 33) and its 11 bus ids, if archived."""
    fy = _guard(VERIFY / "ens_near_Y.npy")
    fb = _guard(VERIFY / "near_buses.npy")
    if not (fy.exists() and fb.exists()):
        return dict(available=False, Y=None, buses=None)
    Y = np.load(fy)
    b = np.load(fb).astype(int)
    return dict(available=True, Y=Y, buses=b,
                n_finite=int(np.isfinite(Y).all(1).sum()), shape=list(Y.shape))


def load_exact_jacobian() -> dict:
    """Archived finite-difference complex Jacobian at 2.0x and its reference."""
    fa = _guard(PILOT / "Aexact_2.0.npy")
    fv = _guard(PILOT / "Vref_2.0.npy")
    fx = _guard(PILOT / "xref_2.0.npy")
    if not (fa.exists() and fv.exists() and fx.exists()):
        return dict(available=False)
    return dict(available=True, A=np.load(fa), Vref=np.load(fv), xref=np.load(fx))
