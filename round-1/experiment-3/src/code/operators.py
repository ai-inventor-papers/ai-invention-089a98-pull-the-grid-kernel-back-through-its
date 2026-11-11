"""operators.py -- rebuild and verify the analytic sensitivity operator, and build
every feature map / kernel metric used by the arms and controls.

The operator
------------
S = d|V_far| / dx  of shape (45, 110), pu per MW/MVAr, from the sequence-impedance
line data and the phase labels at ZERO AC solves.  It is the multiphase LinDistFlow
sensitivity, i.e. the Kirchhoff / Z-bus map restricted to the 15 far-end buses.

Targets: 15 buses x phases (a, b, c) in that order -> 45 outputs.
Inputs : 55 loads' P then the same 55 loads' Q, never reordered -> 110 coordinates.

Feature map scalings (both carried everywhere, because the difference between them
is a live confound in this run's own pilot logs: 0.1118 vs 0.1264 at n=20):
    'iso' : divide by the TRAIN-row rms of ||Sx||, which reproduces the isotropic
            kernel metric ||S(x-x')||^2 exactly.
    'std' : per-coordinate train mean/sd of Sx.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vendor_physics import Physics, zmat

ANGS = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
EPS = 1e-12


def _lin_operator_with_phase(ph: Physics, targets: np.ndarray,
                             phase: np.ndarray) -> np.ndarray:
    """S = d|V_far|/dx with a GIVEN load->phase assignment (zero AC solves).

    Identical algebra to `Physics.lin_operator`, but the phase used for each load is
    passed in, so the phase-label-shuffle control is built from the same line data with a
    permuted assignment.
    """
    Zt = np.zeros((3 * len(targets), ph.nload), dtype=complex)
    for i, tb in enumerate(targets):
        for li in range(ph.nload):
            Zt[3 * i:3 * i + 3, li] = ph.shared(ph.lbus[li], int(tb))[:, int(phase[li])]
    Zt /= ph.Zb
    V0t = np.tile(ph.V0, len(targets))
    V0l = ph.V0[np.asarray(phase)]
    Wt = np.conj(V0t) / np.abs(V0t)
    dPd = -Zt / (ph.Sb * np.conj(V0l))[None, :]
    S = np.zeros((Zt.shape[0], 2 * ph.nload))
    S[:, :ph.nload] = np.real(Wt[:, None] * dPd)
    S[:, ph.nload:] = np.real(Wt[:, None] * (1j * dPd))
    return S


class OperatorBundle:
    """Everything derived from the feeder's line data, computed once."""

    def __init__(self, targets: np.ndarray, n_load: int, phase: np.ndarray) -> None:
        self.targets = np.asarray(targets)
        self.n_load = int(n_load)
        self.phase = np.asarray(phase)
        # filled by build()
        self.S: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.sv: np.ndarray | None = None
        self.Vt: np.ndarray | None = None
        self.S11: np.ndarray | None = None
        self.V11: np.ndarray | None = None
        self.Sr: np.ndarray | None = None
        self.R11: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_physics(cls, ph: Physics) -> "OperatorBundle":
        far, s15 = ph.sheet(15)
        b = cls(s15, ph.nload, ph.phase)
        b.physics = ph
        b.far = far
        return b

    def build(self) -> "OperatorBundle":
        ph = self.physics
        S = ph.lin_operator(self.targets)
        assert S.shape == (45, 110), f"unexpected operator shape {S.shape}"
        assert np.isfinite(S).all(), "operator has non-finite entries"
        self.S = S

        # PHASE-LABEL SHUFFLE control on the SAME archived ground truth. The phase
        # assignment is permuted (seed 4242, a fixed permutation), the operator rebuilt
        # from the SAME line data with those permuted labels, and the same linear arm run
        # on its coordinates. This asks whether the operator's phase-aware load->bus/phase
        # assignment is doing the work, or whether any assignment would do. It is NOT the
        # reserved `ens_perm_Y.npy` / `phase_perm.npy` pair: those need fresh AC solves at
        # physically moved phases and are the survivor's confirmation evidence in a later
        # iteration, so they are not read here.
        rng_p = np.random.default_rng(4242)
        self.phase_perm = rng_p.permutation(np.asarray(ph.phase))
        self.S_shuf = _lin_operator_with_phase(ph, self.targets, self.phase_perm)

        U, sv, Vt = np.linalg.svd(S, full_matrices=False)
        self.U, self.sv, self.Vt = U, sv, Vt
        self.S11 = (U[:, :11] * sv[:11]) @ Vt[:11]           # rank-11 truncation of S
        self.V11 = Vt[:11]                                    # operator row space (11, 110)

        rng = np.random.default_rng(555)
        Sr = rng.standard_normal(S.shape)
        Sr *= np.linalg.norm(S) / np.linalg.norm(Sr)          # Frobenius- AND shape-matched
        self.Sr = Sr

        rng = np.random.default_rng(556)
        Q, R = np.linalg.qr(rng.standard_normal((110, 11)))
        R11 = Q[:, :11].T
        # fix signs deterministically so the control is reproducible to the bit
        R11 = R11 * np.sign(np.sum(R11, axis=1, keepdims=True))
        self.R11 = R11

        rel = sv / sv[0]
        self.diagnostics = dict(
            shape=list(S.shape),
            singular_values=[float(x) for x in sv],
            normalised_spectrum=[float(x) for x in rel],
            effective_rank_1e_2=int((rel > 1e-2).sum()),
            effective_rank_1e_3=int((rel > 1e-3).sum()),
            effective_rank_1e_6=int((rel > 1e-6).sum()),
            stable_rank=float((sv ** 2).sum() / sv[0] ** 2),
            condition_number=float(sv[0] / max(sv[-1], 1e-300)),
            frobenius_norm=float(np.linalg.norm(S)),
            nonzero_singular_values=int((rel > 1e-12).sum()),
        )
        return self


# ---------------------------------------------------------------------------
# feature maps
# ---------------------------------------------------------------------------
def coord_map(Op: np.ndarray, Xtr: np.ndarray, Xte: np.ndarray,
              mode: str = "iso") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Linear feature map x -> (Op x) under one of two train-only scalings."""
    Ftr = Xtr @ Op.T
    Fte = Xte @ Op.T
    if mode == "iso":
        g = float(np.sqrt(np.mean(np.sum(Ftr ** 2, axis=1))) + EPS)
        info = dict(mode="iso", scale=g)
        return Ftr / g, Fte / g, info
    if mode == "std":
        mu = Ftr.mean(0)
        sd = Ftr.std(0) + EPS
        info = dict(mode="std", n_zero_sd=int((Ftr.std(0) < EPS).sum()))
        return (Ftr - mu) / sd, (Fte - mu) / sd, info
    raise ValueError(f"unknown scaling mode {mode!r}")


def quad_features(Ftr: np.ndarray, Fte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degree-2 polynomial expansion: [linear, all squares and cross terms]."""
    def _q(F: np.ndarray) -> np.ndarray:
        n, d = F.shape
        iu = np.triu_indices(d)
        return np.hstack([F, (F[:, :, None] * F[:, None, :])[:, iu[0], iu[1]]])
    return _q(Ftr), _q(Fte)


def standardised_injections(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + EPS
    return (Xtr - mu) / sd, (Xte - mu) / sd


def pca_features(Xtr: np.ndarray, Xte: np.ndarray, k: int = 45
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train-only PCA of the injections; width capped at the TRAIN design rank."""
    Ztr, Zte = standardised_injections(Xtr, Xte)
    U, s, Vt = np.linalg.svd(Ztr, full_matrices=False)
    rank = int((s > s[0] * 1e-10).sum()) if s.size else 0
    kk = int(min(k, rank))
    Vk = Vt[:kk]
    explained = float((s[:kk] ** 2).sum() / max((s ** 2).sum(), EPS))
    return Ztr @ Vk.T, Zte @ Vk.T, dict(k_requested=int(k), k_actual=kk,
                                       train_rank=rank, explained_variance=explained)


# ---------------------------------------------------------------------------
# graph metrics on the 55 loads (baselines b2 and b3) -- line data only
# ---------------------------------------------------------------------------
def hop_metric(ph: Physics) -> np.ndarray:
    """Hop-count graph distance between each pair of load buses."""
    import networkx as nx
    n = ph.nload
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            bi, bj = int(ph.lbus[i]), int(ph.lbus[j])
            lca = nx.lowest_common_ancestor(ph.tree, bi, bj)
            v = ph.hop[bi] + ph.hop[bj] - 2 * ph.hop[lca]
            M[i, j] = M[j, i] = v
    return M


def admittance_metric(ph: Physics) -> tuple[np.ndarray, dict[str, Any]]:
    """Grounded multiphase block admittance Laplacian -> Z_bus -> 55x55 real load Gram.

    Built ONLY from the per-line sequence impedance blocks and the length data
    (never from any sensitivity Jacobian -- that was the prior iteration's fault).
    """
    buses = ph.buses
    idx = {int(b): i for i, b in enumerate(buses)}
    M = len(buses)
    Lb = np.zeros((3 * M, 3 * M), dtype=complex)
    for _, r in ph.net.line.iterrows():
        u, v = int(r.from_bus), int(r.to_bus)
        Mk = zmat(r.r_ohm_per_km * r.length_km, r.x_ohm_per_km * r.length_km,
                  r.r0_ohm_per_km * r.length_km, r.x0_ohm_per_km * r.length_km)
        Y = np.linalg.inv(Mk)
        iu = slice(3 * idx[u], 3 * idx[u] + 3)
        iv = slice(3 * idx[v], 3 * idx[v] + 3)
        Lb[iu, iu] += Y
        Lb[iv, iv] += Y
        Lb[iu, iv] -= Y
        Lb[iv, iu] -= Y
    src = idx[1]
    drop = {3 * src, 3 * src + 1, 3 * src + 2}
    keep = np.array([i for i in range(3 * M) if i not in drop])
    Zb = np.linalg.inv(Lb[np.ix_(keep, keep)])

    def blk(b: int) -> int:
        return 3 * idx[b] - 3 if idx[b] > src else 3 * idx[b]

    n = ph.nload
    li = [blk(int(b)) for b in ph.lbus]
    Zl = np.zeros((3 * n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            Zl[3 * i:3 * i + 3, j] = Zb[li[i]:li[i] + 3, li[j] + int(ph.phase[j])]
    G = 0.5 * (Zl.conj().T @ Zl).real
    sym = float(np.abs(G - G.T).max())
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    return G, dict(generator="per-line sequence-impedance admittance blocks only",
                   symmetry_max_abs_diff=sym,
                   min_eigenvalue=float(ev[0]), max_eigenvalue=float(ev[-1]),
                   condition_number=float(ev[-1] / max(ev[0], 1e-300)),
                   n_negative_eigenvalues=int((ev < -1e-12).sum()))


def _eigen_sqrt(B: np.ndarray, kind: str, n_keep: int | None
                ) -> tuple[np.ndarray, dict[str, Any]]:
    ev, V = np.linalg.eigh(0.5 * (B + B.T))
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    scale = max(abs(float(ev[0])), EPS)
    pos = ev > scale * 1e-10
    k = int(pos.sum())
    if n_keep is not None:
        k = min(k, n_keep)
    Phi = V[:, :k] * np.sqrt(np.maximum(ev[:k], 0.0))[None, :]
    recon = float(np.abs(Phi @ Phi.T - B).max() / max(np.abs(B).max(), EPS))
    return Phi, dict(kind=kind, rank=k,
                     negative_eigenvalues=int((ev < -scale * 1e-10).sum()),
                     reconstruction_rel_error=recon,
                     top_eigenvalues=[float(x) for x in ev[:min(6, ev.size)]])


def mds_features(M: np.ndarray, n_keep: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Classical MDS on a SQUARED-DISTANCE matrix: Phi (n_load x r) with Phi Phi^T ~= B.

    For a row x the feature is f = Phi^T p, so with p one-hot this gives
    ||f_i - f_j||^2 = (p_i - p_j)^T B (p_i - p_j) = Mh_ij for the hop-count metric
    Mh_ij = hop_i + hop_j - 2 hop_lca.  That is exactly the graph-Laplacian kernel on the
    load magnitudes.
    """
    n = M.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ M @ J)
    return _eigen_sqrt(B, "hop_mds", n_keep)


def gram_sqrt_features(G: np.ndarray, n_keep: int | None = None
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Eigen-square-root of a load-space GRAM (not a distance matrix): Phi Phi^T ~= G.

    Used for the admittance-weighted graph kernel, whose 55x55 real Gram of the grounded
    multiphase Z-bus is already the kernel matrix -- double-centring it would be wrong.
    """
    return _eigen_sqrt(G, "admittance_gram", n_keep)


def graph_features(Phi: np.ndarray, Xtr: np.ndarray, Xte: np.ndarray,
                   n_load: int, mode: str = "iso") -> tuple[np.ndarray, np.ndarray]:
    """Apply a graph-kernel load-space basis to the 55 load MAGNITUDES (P only).

    `mode='iso'` divides by the TRAIN-row rms of ||f||, exactly as the operator's
    coordinate map does, so the squared distances fed to the GP are O(1) and the shared
    lengthscale grid is meaningful for this arm too. Without this the graph arms sit at
    squared distances of thousands and every kernel entry underflows to zero, which would
    report the graph baselines as the mean predictor -- a straw man, not a baseline.
    """
    Ptr = Xtr[:, :n_load]
    Pte = Xte[:, :n_load]
    Ftr = Ptr @ Phi
    Fte = Pte @ Phi
    if mode == "iso":
        g = float(np.sqrt(np.mean(np.sum(Ftr ** 2, axis=1))) + EPS)
        return Ftr / g, Fte / g
    if mode == "std":
        mu = Ftr.mean(0)
        sd = Ftr.std(0) + EPS
        return (Ftr - mu) / sd, (Fte - mu) / sd
    raise ValueError(f"unknown graph scaling {mode!r}")
