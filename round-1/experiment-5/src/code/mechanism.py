"""Step 13: the mechanism measurement.

(a) Principal angles between the k=11 Laplacian-selected injection-space subspace and
    the column space of the target-restricted operator S, both expressed in the SAME
    ambient space R^110.
    - S's subspace: the row space of S (45 x 110), i.e. span of the rows.
    - The Laplacian embedding's induced injection-space subspace for a gathered block
      Fk (55 x q) is the column space of the block-diagonal 110 x 2q matrix
      [[Fk, 0], [0, Fk]] — the same Fk acting on the P block and the Q block.
    Principal angles via arccos of the singular values of Q1^T Q2.
(b) The SVD spectrum of S, effective rank at 1e-2/1e-3/1e-4, stable rank, condition.
(c) The Laplacian spectrum summary.

The comparison is deliberately between 2k = 22 dimensions and the operator's effective
rank of 11 at a 1e-2 threshold, so the number is interpretable.
"""
from __future__ import annotations

import numpy as np


def angular_table(S: np.ndarray, ops, ks=(3, 11, 22, 45), route: str = "R1") -> dict:
    """Principal angles (degrees) between the spectral subspace and S's row space."""
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    tol = 1e-2 * sv[0]
    r_eff = int((sv > tol).sum())
    Qs = Vt[:r_eff].T                          # (110, r_eff) orthonormal row-space basis
    out: dict = {"s_effective_rank_at_1e-2": r_eff, "per_k": {}}
    for k in ks:
        Fk = ops.features(route, k, "real")     # (55, k) real part, the headline variant
        E = np.zeros((110, 2 * k))
        E[:55, :k] = Fk
        E[55:, k:] = Fk
        q = np.linalg.qr(E)[0]                  # (110, 2k)
        M = q.T @ Qs
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        ang = np.degrees(np.arccos(s))
        # the operator's subspace is r_eff-dimensional; the embedding's is 2k-dimensional.
        # For 2k > r_eff only r_eff angles are informative (the rest are structural zeros).
        n_inf = min(2 * k, r_eff)
        a = ang[:n_inf]
        out["per_k"][str(k)] = {
            "dim_embedding": int(2 * k), "dim_operator": int(r_eff),
            "n_angles_informative": int(n_inf),
            "min_deg": float(a.min()), "mean_deg": float(a.mean()), "max_deg": float(a.max()),
            "angles_deg": a.tolist(),
            "overlap_fraction": float(np.mean(np.cos(np.radians(a)) ** 2)),
            "frac_angles_under_10deg": float(np.mean(a < 10.0)),
        }
    return out


def s_spectrum(S: np.ndarray) -> dict:
    sv = np.linalg.svd(S, compute_uv=False)
    return {
        "singular_values": sv.tolist(),
        "rank_1e-2": int((sv > 1e-2 * sv[0]).sum()),
        "rank_1e-3": int((sv > 1e-3 * sv[0]).sum()),
        "rank_1e-4": int((sv > 1e-4 * sv[0]).sum()),
        "stable_rank": float((sv ** 2).sum() / sv[0] ** 2),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
    }


def laplacian_spectrum_summary(ops) -> dict:
    c = ops.checks
    return {
        "re_lambda_45_smallest": c.get("re_lam_smallest_45"),
        "im_over_lambda_45_smallest": c.get("im_lam_frac_smallest_45"),
        "frac_im_over_1e-6": c.get("frac_im_over_rel_gt_1e6_45"),
        "r1_residual_max": c.get("r1_residual_max"),
        "r1_residual_mean": c.get("r1_residual_mean"),
        "r2_residual_max": c.get("r2_residual_max"),
        "r2_residual_mean": c.get("r2_residual_mean"),
        "hermiticity_gap_abs_max": c.get("lap_nonherm_max"),
        "complex_symmetry_gap_abs_max": c.get("lap_asym_max"),
        "abs_scale": c.get("lap_abs_max"),
        "r3_vs_r1_angle_max_deg": c.get("r3_vs_r1_angle_max_deg"),
        "r5_available": c.get("r5_available"),
        "r5_im_diag_ratio": c.get("r5_im_diag_ratio"),
    }
