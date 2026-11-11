"""state.py -- build the shared, read-only state every fold reads.

Loads the archived AC ground truth, verifies every recorded statistic, rebuilds the
analytic operator (zero AC solves), computes the two LinDistFlow voltage vectors
(one zero-load AC solve for the reference) and the two graph-kernel feature maps.

Nothing here reads the reserved ensembles (1.0x, 2.6x) -- the 1.0x/2.6x regimes are
the survivor's confirmation evidence in a later iteration and must stay untouched.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

import operators as opmod
from vendor_physics import Physics

# --- portable input location -------------------------------------------------
# This artifact declares no dependency, so its inputs live in the run tree it was built
# in rather than in a sibling artifact folder. Paths are therefore anchored on this
# file and kept RELATIVE (never an absolute server path) so the code runs from a clone:
#   workspace = <run>/3_invention_loop/iter_1/gen_art/gen_art_experiment_3
#   code/     = workspace/code        -> parents[4] is <run>
# Override the run root with PF_RUN_ROOT when the inputs live elsewhere (e.g. a reader
# who has only the published repository and supplies the archived arrays themselves).
PILOT_REL = "iter_2/gen_hypo/claude_agent/pilot"
VERIFY_REL = "iter_3/gen_hypo/claude_agent/verify"


def _find_run_root() -> Path:
    """Walk up from this file until the directory holding the archived ensembles is found.

    Anchored on __file__, so the same code runs from a clone at any depth. `PF_RUN_ROOT`
    overrides it when the inputs are supplied from somewhere else.
    """
    env = os.environ.get("PF_RUN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / PILOT_REL / "ens_s2.0_X.npy").is_file():
            return parent
    return Path(__file__).resolve().parents[1]


RUN_ROOT = _find_run_root()
PILOT = RUN_ROOT / PILOT_REL
VERIFY = RUN_ROOT / VERIFY_REL

RESERVED = ["ens_s1.0_X.npy", "ens_s1.0_Y.npy", "ens_s2.6_X.npy", "ens_s2.6_Y.npy",
            "ens_perm_Y.npy", "phase_perm.npy", "ens_near_Y.npy"]

EXPECT_SHEET15 = [681, 697, 713, 729, 745, 761, 777, 793, 809, 825, 841, 857, 873, 889, 906]
EXPECT_PHASE_COUNTS = [21, 19, 15]
EXPECT_MEAN_KW = (114.60, 114.80)
EXPECT_FAR_SD = (0.02098, 0.02104)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_archives() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X = np.load(PILOT / "ens_s2.0_X.npy")
    Y = np.load(PILOT / "ens_s2.0_Y.npy")
    sheet15 = np.load(PILOT / "sheet15.npy")
    ph_true = np.load(PILOT / "ph_true.npy")

    checks: dict[str, Any] = {}

    def note(k: str, ok: bool, val: Any) -> None:
        checks[k] = dict(passed=bool(ok), value=val)
        assert ok, f"archive check failed: {k} = {val}"

    note("X_shape_220x110", X.shape == (220, 110), list(X.shape))
    note("Y_shape_220x45", Y.shape == (220, 45), list(Y.shape))
    note("X_all_finite", bool(np.isfinite(X).all()), bool(np.isfinite(X).all()))
    note("Y_all_finite", bool(np.isfinite(Y).all()), bool(np.isfinite(Y).all()))
    note("sheet15_exact", sheet15.tolist() == EXPECT_SHEET15, sheet15.tolist())
    note("phase_counts", np.bincount(ph_true, minlength=3).tolist() == EXPECT_PHASE_COUNTS,
         np.bincount(ph_true, minlength=3).tolist())
    mkw = float(X[:, :55].sum(1).mean() * 1e3)
    note("mean_total_active_load_kw", EXPECT_MEAN_KW[0] <= mkw <= EXPECT_MEAN_KW[1], mkw)
    fsd = float(Y.std(0).mean())
    note("far_end_mean_per_output_sd_pu", EXPECT_FAR_SD[0] <= fsd <= EXPECT_FAR_SD[1], fsd)
    note("n_rows_220", X.shape[0] == 220, int(X.shape[0]))

    checks["reserved_not_read"] = RESERVED
    checks["archive_files_sha256"] = {
        "ens_s2.0_X.npy": sha256_file(PILOT / "ens_s2.0_X.npy"),
        "ens_s2.0_Y.npy": sha256_file(PILOT / "ens_s2.0_Y.npy"),
        "sheet15.npy": sha256_file(PILOT / "sheet15.npy"),
        "ph_true.npy": sha256_file(PILOT / "ph_true.npy"),
    }
    checks["openrouter_calls"] = 0
    checks["openrouter_usd"] = 0.0
    return X, Y, sheet15, ph_true, checks


def build_state(verify_operator: bool = True) -> dict[str, Any]:
    X, Y, sheet15, ph_true, checks = load_archives()

    ph = Physics()
    far, s15 = ph.sheet(15)
    assert np.array_equal(s15, sheet15), "rebuilt target sheet differs from the archive"

    ops = opmod.OperatorBundle.from_physics(ph).build()

    op_checks = dict(ops.diagnostics)
    op_checks["sheet15_rebuilt_equals_archive"] = True
    if verify_operator:
        S_arch = np.load(VERIFY / "S_sheet15.npy")
        rel = float(np.abs(ops.S - S_arch).max() / np.abs(S_arch).max())
        op_checks["archive_operator_max_rel_diff"] = rel
        assert ops.S.shape == (45, 110)
        assert rel <= 1e-8, f"rebuilt operator disagrees with the archive by {rel}"
    op_checks["sensitivity_operator_ac_solves"] = 0
    op_checks["physics_baseline_ac_solves"] = 1
    op_checks["ac_solve_note"] = ("S costs zero AC solves. The physics BASELINE uses one "
                                  "zero-load solve for the no-load reference voltage; the "
                                  "treatment does not.")

    # no-load reference: the archived route (one AC solve) and an analytic substitute
    V0b = ph.noload_voltage(s15)
    V0_analytic = 1.05 * np.exp(1j * np.tile(opmod.ANGS, 15))
    op_checks["noload_reference_max_abs_diff"] = float(np.abs(V0b - V0_analytic).max())
    op_checks["noload_reference_max_abs_mag_diff"] = float(
        np.abs(np.abs(V0b) - np.abs(V0_analytic)).max())

    Vp1 = ph.lindist(X, s15, order=1, v0bus=V0b)
    Vp2 = ph.lindist(X, s15, order=2, v0bus=V0b)
    Vp1_an = ph.lindist(X, s15, order=1, v0bus=V0_analytic)
    Vp2_an = ph.lindist(X, s15, order=2, v0bus=V0_analytic)
    vsd_all = float(Y.std(0).mean())
    op_checks["noload_sensitivity"] = {
        "order1_rmse_norm_archive_ref": float(np.sqrt(np.mean((Vp1 - Y) ** 2)) / vsd_all),
        "order1_rmse_norm_analytic_ref": float(np.sqrt(np.mean((Vp1_an - Y) ** 2)) / vsd_all),
        "order2_rmse_norm_archive_ref": float(np.sqrt(np.mean((Vp2 - Y) ** 2)) / vsd_all),
        "order2_rmse_norm_analytic_ref": float(np.sqrt(np.mean((Vp2_an - Y) ** 2)) / vsd_all),
        "used": "archive (noload_voltage)",
    }

    # graph-kernel bases (baselines b2, b3) -- line data only, no sensitivity Jacobian
    Mh = opmod.hop_metric(ph)
    Phi_h, hop_info = opmod.mds_features(Mh)
    G_adm, adm_info = opmod.admittance_metric(ph)
    Phi_a, adm_rank_info = opmod.gram_sqrt_features(G_adm)
    P = X[:, :ph.nload]                                    # raw load magnitudes (MW)
    graph_bases = {
        "b2_hop_laplacian_gp": dict(Phi=Phi_h, P=P, info=hop_info),
        "b3_admittance_laplacian_gp": dict(Phi=Phi_a, P=P, info=adm_rank_info),
    }
    op_checks["graph_kernel_metrics"] = {
        "hop": hop_info, "admittance_metric": adm_info, "admittance_basis": adm_rank_info,
        "basis_widths": {k: int(v["Phi"].shape[1]) for k, v in graph_bases.items()},
        "hop_metric_shape": list(Mh.shape),
        "standardisation": "per-fold, TRAIN rows only (applied inside run_fold)",
        "built_from": "line sequence-impedance blocks and phase labels only; "
                      "no sensitivity Jacobian",
    }

    return dict(X=X, Y=Y, sheet15=s15, ph_true=ph_true, physics=ph, ops=ops,
                Vp1=Vp1, Vp2=Vp2, Vp1_an=Vp1_an, Vp2_an=Vp2_an,
                graph_bases=graph_bases,
                archive_checks=checks, operator_checks=op_checks)
