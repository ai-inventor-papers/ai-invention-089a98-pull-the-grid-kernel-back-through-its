"""The sensitivity operators, their spectral facts, and the structural checks.

Primary treatment
  S_far      45 x 110   = d|V_far|/dx on the 15 far-end buses x 3 phases
Mechanism metrics
  S_near     33 x 110   = 11 near buses x 3 phases
  S_full   2718 x 110   = all 906 buses x 3 phases
  S_full_r  rank-matched truncation of S_full to the effective rank of S_far
  S_all90    90 x 110   = 30 buses evenly spaced over the full hop range x 3
  S_all55    55 x 110   = the unrestricted 55x55 admittance Z-bus metric's own operator
                          (P and Q blocks both, load-bus outputs, all phases)
  S_far_k    rank-k truncation of S_far
  S_far_g    partial whitening U diag((sv/sv1)^gamma) V^T

Controls
  S_shuffled the operator rebuilt from a permuted LOAD->PHASE assignment (not a
             row permutation, which is a no-op in the metric)
  S_random   iid Gaussian rescaled to the Frobenius norm of S_far
  S_rotated  the injections rotated by a fixed random orthogonal Q inside the metric
"""
from __future__ import annotations

import numpy as np

N_LOAD = 55


def build_operator(physics, targets, phase_labels, tr_mode="pu_correct",
                   v0mode="noload") -> np.ndarray:
    return physics.lin_operator(targets, phase_labels, tr_mode, v0mode)


def spectral_facts(S: np.ndarray, thresholds=(1e-2, 1e-3, 1e-4), n_terms: int = 20) -> dict:
    sv = np.linalg.svd(S, compute_uv=False)
    sv = np.sort(np.maximum(sv, 0.0))[::-1]
    tot = float((sv ** 2).sum())
    eff = {}
    for t in thresholds:
        r = int((sv > t * sv[0]).sum()) if sv[0] > 0 else 0
        eff[f"eff_rank_{t:g}"] = r
    stable = float((sv ** 2).sum() / (sv[0] ** 2)) if sv[0] > 0 else 0.0
    cum = np.cumsum(sv ** 2) / tot if tot > 0 else np.zeros_like(sv)
    return dict(
        shape=list(S.shape),
        rank_full=int(np.linalg.matrix_rank(S)) if min(S.shape) <= 300 else int(sv.size),
        sv_top=[float(v) for v in sv[:n_terms]],
        sv_norm_top=[float(v / sv[0]) for v in sv[:n_terms]],
        cum_energy_top=[float(v) for v in cum[:n_terms]],
        stable_rank=stable,
        condition_number=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        fro_norm=float(np.linalg.norm(S)),
        **eff,
    )


def truncate_rank(S: np.ndarray, k: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    k = int(min(k, len(s)))
    return (U[:, :k] * s[:k]) @ Vt[:k]


def partial_whiten(S: np.ndarray, gamma: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    scale = (s / s[0]) ** gamma * s[0]
    return (U * scale) @ Vt


def random_operator_like(S: np.ndarray, seed: int = 777) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=S.shape)
    G *= np.linalg.norm(S) / np.linalg.norm(G)
    return G


def shuffled_phase_operator(physics, targets, phase_true, tr_mode="pu_correct",
                            v0mode="noload", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild S from a permuted LOAD->PHASE assignment.

    The archived attempt permuted the ROWS of S.  That is a no-op: ||S(x-x')|| is
    invariant under row permutation, so it returns an identical number by
    construction.  The correct control permutes which phase each load sits on in
    the OPERATOR BUILD, leaving the data on the real phases.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(phase_true))
    phase_shuf = np.asarray(phase_true)[perm].astype(int)
    S = build_operator(physics, targets, phase_shuf, tr_mode, v0mode)
    return S, phase_shuf


def shuffle_effectiveness(S_true: np.ndarray, S_shuf: np.ndarray,
                          min_cols: int = 40) -> dict:
    """Frobenius relative difference and how many of the 110 columns changed direction."""
    denom = np.linalg.norm(S_true)
    frob_rel = float(np.linalg.norm(S_true - S_shuf) / denom) if denom > 0 else 0.0
    changed = 0
    cos = []
    for j in range(S_true.shape[1]):
        a, b = S_true[:, j], S_shuf[:, j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        c = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        cos.append(c)
        if abs(c) < 0.9999:
            changed += 1
    cos = np.array(cos)
    return dict(frobenius_rel_diff=frob_rel, n_columns_changed=int(changed),
                n_columns=int(S_true.shape[1]),
                min_abs_cos=float(np.abs(cos).min()),
                median_abs_cos=float(np.median(np.abs(cos))),
                effective=bool(changed >= min_cols))


def rotate_injections(X: np.ndarray, seed: int = 31415) -> np.ndarray:
    """Random orthogonal rotation of the injections inside each 55-d block."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    for sl in (slice(0, N_LOAD), slice(N_LOAD, 2 * N_LOAD)):
        Q, _ = np.linalg.qr(rng.normal(size=(N_LOAD, N_LOAD)))
        out[:, sl] = X[:, sl] @ Q
    return out


def structural_cross_check(S_new: np.ndarray, S_arch: np.ndarray) -> dict:
    """Compare with the archived 45x110 operator.  Elementwise equality is impossible
    (that array came from a builder without the per-unit divide, without the |V|
    projection, and with a real/imaginary split rather than Re(conj(V0)/|V0| dV/dx)).
    Support pattern is required to proceed; the rest is informational.
    """
    mask_new = np.abs(S_new) > 0
    mask_arch = np.abs(S_arch) > 0
    support_match = bool(mask_new.shape == mask_arch.shape and np.array_equal(mask_new, mask_arch))
    frob_rel = float(np.linalg.norm(S_new - S_arch) / np.linalg.norm(S_arch))
    pear = float(np.corrcoef(S_new.ravel(), S_arch.ravel())[0, 1])
    Un, _, _ = np.linalg.svd(S_new, full_matrices=False)
    Ua, _, _ = np.linalg.svd(S_arch, full_matrices=False)
    k = min(11, Un.shape[1], Ua.shape[1])
    sv = np.linalg.svd(Un[:, :k].T @ Ua[:, :k], compute_uv=False)
    max_angle = float(np.degrees(np.arccos(np.clip(sv.min(), -1.0, 1.0))))
    return dict(support_pattern_match=support_match,
                frobenius_rel_diff=frob_rel, pearson=pear,
                max_principal_angle_top11_deg=max_angle,
                required_for_proceeding="support_pattern_match",
                informational="frobenius_rel_diff, pearson, max_principal_angle")


def data_whitened_diagnostics(S: np.ndarray, Xtr: np.ndarray, ks=(6, 11, 20)) -> dict:
    """The honest, loading-DEPENDENT replacement for 'effective rank at 2.6x'.

    The operator itself takes no loading argument, so recomputing ITS rank at
    another regime is a vacuous identity.  What does depend on loading is
      (a) the rank of the data-whitened operator  S Cov(X_tr) S^T,  and
      (b) the fraction of injection variance captured by the top-k row space of S.
    """
    C = np.cov(Xtr, rowvar=False)
    M = S @ C @ S.T
    sv = np.sort(np.linalg.svd(M, compute_uv=False))[::-1]
    out = dict(rank_data_whitened=int((sv > 1e-10 * sv[0]).sum()),
               stable_rank_data_whitened=float((sv ** 2).sum() / (sv[0] ** 2)))
    # top-k row space of S: fraction of injection variance inside it
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    total = float(np.trace(C))
    for k in ks:
        kk = min(k, Vt.shape[0])
        Pk = Vt[:kk].T
        captured = float(np.trace(Pk.T @ C @ Pk))
        out[f"injection_var_share_top{kk}_rowspace"] = captured / total if total > 0 else 0.0
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    out["injection_pc_energy_top11"] = float(ev[:11].sum() / ev.sum()) if ev.sum() > 0 else 0.0
    return out


def differential_mode_share(Y: np.ndarray) -> dict:
    """Share of far-end voltage variance that is differential-mode (per-bus 3-phase mean
    removed), pooled and after removing per-(bus,phase) draw means.

    PC7(i): the archived note gives 0.983/0.984/0.984 pooled and 0.900/0.906/0.908 after
    removing per-(bus,phase) draw means.  Re-measured here with the definition written out.
    """
    V = Y.reshape(len(Y), -1, 3)
    dm = V - V.mean(2, keepdims=True)
    pooled = float(np.var(dm) / np.var(V))
    Vc = V - V.mean(0, keepdims=True)
    after = float(np.var(dm - dm.mean(0, keepdims=True)) / np.var(Vc))
    phase_means = V.mean(0).mean(0)
    return dict(definition="differential mode = per-bus 3-phase mean removed",
                pooled_share=pooled, after_removing_draw_means=after,
                phase_mean_magnitudes_pu=[float(v) for v in phase_means])


def laplacian_is_hermitian(physics, phase_labels) -> dict:
    """PC7(ii): the phase-expanded admittance-weighted Laplacian is complex-SYMMETRIC,
    not Hermitian, so eigh/eigsh are invalid for it."""
    L = physics.block_laplacian("none").toarray()
    asym_T = float(np.abs(L - L.T).max())
    asym_H = float(np.abs(L - L.conj().T).max())
    scale = float(np.abs(L).max())
    Lh = 0.5 * (L + L.conj().T)
    ev_h = np.linalg.eigvalsh(Lh)
    ev_g = np.linalg.eigvals(L)
    return dict(max_abs_L_minus_LT=asym_T, max_abs_L_minus_LH=asym_H, scale=scale,
                complex_symmetric=bool(asym_T < 1e-6 * scale),
                hermitian=bool(asym_H < 1e-6 * scale),
                method="scipy.linalg.eig (general) and eigvalsh on (L+L^H)/2",
                hermitian_part_eig_min=float(ev_h.min()),
                general_eig_real_min=float(ev_g.real.min()),
                general_eig_abs_min=float(np.abs(ev_g).min()),
                general_eig_max_abs=float(np.abs(ev_g).max()))
