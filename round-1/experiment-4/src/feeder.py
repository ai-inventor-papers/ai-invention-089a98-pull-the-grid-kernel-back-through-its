"""IEEE PES European LV test feeder: multiphase LinDistFlow, sensitivity operator, graph metrics.

Everything in this module is ANALYTIC and costs ZERO three-phase AC power-flow
solves except `noload_voltage`, which costs exactly one (a zero-injection solve
used only to obtain the no-load reference phasors).

Bases: S_base = transformer rating (0.8 MVA), V_base = transformer LV voltage
(0.416 kV) => Z_base = 0.2163 ohm.  All impedances and injections are in per unit.

Objects produced here
---------------------
Physics.far_sheet(k)      : the 15 far-end buses x 3 phases = 45 prediction columns
Physics.lindist(x, ...)   : multiphase LinDistFlow, order 1 or order 2 (exact
                            second-order Z-bus current correction)
Physics.lin_operator(...) : S = d|V_far|/dx, the output-restricted sensitivity
                            operator used as the treatment's feature map
build_laplacians(...)     : the grounded phase-expanded admittance Laplacian and
                            the three 55x55 graph metrics (admittance, impedance,
                            hop count) used by the graph-kernel baselines
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ANGS = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])


def zmat(r1: float, x1: float, r0: float, x0: float) -> np.ndarray:
    """3x3 sequence-impedance block for one line, in ohm."""
    z1 = r1 + 1j * x1
    z0 = r0 + 1j * x0
    zs = (z0 + 2.0 * z1) / 3.0
    zm = (z0 - z1) / 3.0
    m = np.full((3, 3), zm, dtype=complex)
    np.fill_diagonal(m, zs)
    return m


class Feeder:
    """The feeder's electrical structure, cached. Read-only after construction."""

    def __init__(self, name: str = "on_peak_566", v0: float = 1.05) -> None:
        net = pn.ieee_european_lv_asymmetric(name)
        self.net = net
        al = net.asymmetric_load
        self.nload = int(len(al))
        self.cols = ["p_a_mw", "p_b_mw", "p_c_mw"]
        self.qcols = ["q_a_mvar", "q_b_mvar", "q_c_mvar"]
        self.P0 = al[self.cols].to_numpy(dtype=float).copy()
        self.Q0 = al[self.qcols].to_numpy(dtype=float).copy()
        # each load sits on exactly one phase: its native (real) phase
        self.phase = np.argmax(self.P0, axis=1).astype(int)
        self.lbus = al.bus.to_numpy().astype(int)
        self.v0 = float(v0)
        self.V0 = self.v0 * np.exp(1j * ANGS)
        self.Vb = float(net.trafo.vn_lv_kv.to_numpy()[0])
        self.Sb = float(net.trafo.sn_mva.to_numpy()[0])
        self.Zb = self.Vb ** 2 / self.Sb
        tr = net.trafo.iloc[0]
        zk = float(tr.vk_percent) / 100.0
        rk = float(tr.vkr_percent) / 100.0
        self.ztr = rk + 1j * np.sqrt(max(zk ** 2 - rk ** 2, 0.0))

        g = nx.Graph()
        for _, r in net.line.iterrows():
            g.add_edge(int(r.from_bus), int(r.to_bus))
        self.G = g
        self.tree = nx.bfs_tree(g, 1)
        self.hop = nx.single_source_shortest_path_length(g, 1)
        self.buses = np.array(sorted(self.tree.nodes()))
        self.bidx = {int(b): i for i, b in enumerate(self.buses)}
        self.lbus_hop = np.array([self.hop[int(b)] for b in self.lbus])
        self.Zp = self._path_impedance()
        self._shared_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._zc_cache: Dict[Tuple[int, ...], np.ndarray] = {}

    # ---------------------------------------------------------------- graph
    def _path_impedance(self) -> Dict[int, np.ndarray]:
        """Cumulative 3x3 sequence impedance from the source to each bus."""
        zto: Dict[int, np.ndarray] = {1: np.zeros((3, 3), dtype=complex)}
        for u, v in nx.bfs_edges(self.tree, 1):
            row = self.net.line[((self.net.line.from_bus == u) & (self.net.line.to_bus == v)) |
                                ((self.net.line.from_bus == v) & (self.net.line.to_bus == u))].iloc[0]
            length = float(row.length_km)
            zto[int(v)] = zto[int(u)] + zmat(float(row.r_ohm_per_km) * length,
                                             float(row.x_ohm_per_km) * length,
                                             float(row.r0_ohm_per_km) * length,
                                             float(row.x0_ohm_per_km) * length)
        return zto

    def shared(self, load_bus: int, target_bus: int) -> np.ndarray:
        """Impedance of the path shared by a load and a target, plus the transformer."""
        key = (int(load_bus), int(target_bus))
        hit = self._shared_cache.get(key)
        if hit is not None:
            return hit
        z = self.Zp[nx.lowest_common_ancestor(self.tree, int(load_bus), int(target_bus))]
        m = np.zeros((3, 3), dtype=complex)
        np.fill_diagonal(m, self.ztr)
        out = z + m
        self._shared_cache[key] = out
        return out

    def far_buses(self, frac: float = 0.75) -> np.ndarray:
        mh = max(self.hop.values())
        return np.array(sorted(b for b, d in self.hop.items() if d >= frac * mh))

    def far_sheet(self, k: int = 15, frac: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        far = self.far_buses(frac)
        sel = np.array(sorted(set(far[np.linspace(0, len(far) - 1, k).astype(int)])))
        return far, sel

    def hop_distance(self, i: int, j: int) -> int:
        a, b = int(self.lbus[i]), int(self.lbus[j])
        return int(self.hop[a] + self.hop[b] - 2 * self.hop[nx.lowest_common_ancestor(self.tree, a, b)])

    # ------------------------------------------------------------ operators
    def Zc(self, targets: np.ndarray) -> np.ndarray:
        """Complex (3*n_target, n_load) per-unit voltage/current operator."""
        key = tuple(int(t) for t in targets)
        hit = self._zc_cache.get(key)
        if hit is not None:
            return hit
        zt = np.zeros((3 * len(targets), self.nload), dtype=complex)
        for i, tb in enumerate(targets):
            for ell in range(self.nload):
                zt[3 * i:3 * i + 3, ell] = self.shared(self.lbus[ell], int(tb))[:, self.phase[ell]]
        zt = zt / self.Zb
        self._zc_cache[key] = zt
        return zt

    def s_of(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return (x[:, : self.nload] + 1j * x[:, self.nload:]) / self.Sb

    def lindist(self, x: np.ndarray, targets: np.ndarray, order: int = 1,
                v0bus: np.ndarray | None = None, phase: np.ndarray | None = None) -> np.ndarray:
        """Multiphase LinDistFlow: |V| at `targets`, shape (n_rows, 3*len(targets)).

        order 1:  dV = -Zc @ conj(s)/conj(V0_load)
        order 2:  adds the exact second-order Z-bus current correction
                  dI2_l = -conj(s_l) conj(dV1_l)/|V0_l|^2
        `phase` lets a control rebuild the same operator with permuted phase labels.
        """
        phase = self.phase if phase is None else np.asarray(phase, dtype=int)
        s = self.s_of(x)
        zt = self._Zc_with_phase(targets, phase)
        m = len(targets)
        v0t = np.tile(self.V0, m) if v0bus is None else np.asarray(v0bus, dtype=complex)
        v0l = self.V0[phase]
        i1 = np.conj(s) / np.conj(v0l)[None, :]
        dv1 = -(i1 @ zt.T)
        if order == 1:
            return np.abs(v0t[None, :] + dv1)
        zall = self._Zc_with_phase(self.buses, phase)
        dv1a = -((np.conj(s) / np.conj(v0l)[None, :]) @ zall.T)
        lidx = np.array([self.bidx[int(b)] for b in self.lbus])
        pidx = lidx * 3 + phase
        dv1_load = dv1a[:, pidx]
        i2 = np.conj(s) * np.conj(dv1_load) / (np.abs(v0l) ** 2)[None, :]
        dv2 = -(i2 @ zt.T)
        return np.abs(v0t[None, :] + dv1 + dv2)

    def _Zc_with_phase(self, targets: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Zc for an arbitrary load->phase assignment (cached on the label vector)."""
        if phase is self.phase or np.array_equal(phase, self.phase):
            return self.Zc(targets)
        key = (tuple(int(t) for t in targets), tuple(int(p) for p in phase))
        hit = self._zc_cache.get(key)
        if hit is not None:
            return hit
        zt = np.zeros((3 * len(targets), self.nload), dtype=complex)
        for i, tb in enumerate(targets):
            for ell in range(self.nload):
                zt[3 * i:3 * i + 3, ell] = self.shared(self.lbus[ell], int(tb))[:, phase[ell]]
        zt = zt / self.Zb
        self._zc_cache[key] = zt
        return zt

    def lin_operator(self, targets: np.ndarray, phase: np.ndarray | None = None) -> np.ndarray:
        """S = d|V|/dx, shape (3*n_target, 2*n_load), per unit per MW/MVAr.

        Complex LinDistFlow gives dV_q/dP_l = -Zc[q,l]/(Sb conj(V0_l)) and
        dV_q/dQ_l = +j times that; then d|V|/dP = Re(conj(V0)/|V0| dV/dP).
        Built from line data only -- no AC solve, no Jacobian of any AC solver.
        """
        phase = self.phase if phase is None else np.asarray(phase, dtype=int)
        zt = self._Zc_with_phase(targets, phase)
        v0t = np.tile(self.V0, len(targets))
        v0l = self.V0[phase]
        wt = np.conj(v0t) / np.abs(v0t)
        dpd = -zt / (self.Sb * np.conj(v0l))[None, :]
        s = np.zeros((zt.shape[0], 2 * self.nload))
        s[:, : self.nload] = np.real(wt[:, None] * dpd)
        s[:, self.nload:] = np.real(wt[:, None] * (1j * dpd))
        return s

    def noload_voltage(self, targets: np.ndarray) -> np.ndarray:
        """No-load complex phasors at `targets`. Costs ONE three-phase AC solve."""
        net = pn.ieee_european_lv_asymmetric("on_peak_566")
        al = net.asymmetric_load
        for c in self.cols + self.qcols:
            al[c] = 0.0
        pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
        if not bool(net.converged):
            raise RuntimeError("no-load three-phase solve did not converge")
        va = net.res_bus_3ph[["va_a_degree", "va_b_degree", "va_c_degree"]].to_numpy()[targets]
        vm = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].to_numpy()[targets]
        out = (vm * np.exp(1j * np.deg2rad(va))).ravel()
        if not np.isfinite(out).all():
            raise RuntimeError("no-load solve returned non-finite voltages")
        return out


# ---------------------------------------------------------------------------
# graph Laplacians and the 55x55 kernel metrics
# ---------------------------------------------------------------------------
def assemble_laplacian(fe: Feeder, weights: str = "admittance") -> sp.csc_matrix:
    """Grounded phase-expanded block Laplacian (source dofs removed).

    weights = 'admittance' -> off-diagonal block -inv(Zline)  (the Kirchhoff object)
    weights = 'impedance'  -> off-diagonal block -Zline       (a diagnostic only:
                              the prior round found this is the wrong object)
    weights = 'unit'       -> off-diagonal block -I           (pure topology, no
                              electrical data: the plain graph-Laplacian baseline)
    """
    nb = len(fe.buses)
    pos = {int(b): i for i, b in enumerate(fe.buses)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for _, r in fe.net.line.iterrows():
        u, v = int(r.from_bus), int(r.to_bus)
        length = float(r.length_km)
        zl = zmat(float(r.r_ohm_per_km) * length, float(r.x_ohm_per_km) * length,
                  float(r.r0_ohm_per_km) * length, float(r.x0_ohm_per_km) * length)
        if weights == "admittance":
            b = np.linalg.inv(zl)
        elif weights == "impedance":
            b = zl
        elif weights == "unit":
            b = np.eye(3, dtype=complex)
        else:
            raise ValueError(f"unknown Laplacian weighting {weights!r}")
        iu, iv = 3 * pos[u], 3 * pos[v]
        for i in range(3):
            for j in range(3):
                # diagonal blocks +B, off-diagonal blocks -B
                rows += [iu + i, iv + i, iu + i, iv + i]
                cols += [iu + j, iv + j, iv + j, iu + j]
                vals += [b[i, j], b[i, j], -b[i, j], -b[i, j]]
    L = sp.coo_matrix((vals, (rows, cols)), shape=(3 * nb, 3 * nb)).tocsc()
    # ground the source bus: delete its three rows and columns
    src = 3 * pos[1]
    keep = np.setdiff1d(np.arange(3 * nb), np.array([src, src + 1, src + 2]))
    return L[keep][:, keep].tocsc()


def build_laplacians(fe: Feeder) -> dict:
    """The three 55x55 metrics used by the graph-kernel baselines, from line data only.

    admittance : M[i,j] = 0.5 Re(sum_dof conj(Zl[dof,i]) Zl[dof,j]) with Zl the
                 grounded admittance-Laplacian inverse gathered at each load's own
                 phase -- the Kirchhoff influence Gram matrix.
    hop        : the SAME construction with UNIT conductance on every line, i.e. the
                 plain graph-Laplacian (pure topology, hop counts, no electrical data).
                 Kept as a PSD Gram so that "unit weights" and "admittance weights"
                 are one nested family and the difference between them is exactly
                 the electrical information.
    impedance  : identical construction with impedance weights (DIAGNOSTIC ONLY).

    `hop_distance_matrix` is retained as a DIAGNOSTIC: used directly as a smoothing
    operator it is not a kernel, because a distance matrix has negative eigenvalues
    (54 of 55 here).  That is recorded rather than quietly clipped, and it is why
    the hop baseline is the unit-weight Laplacian and not the raw distance matrix.
    """
    nb = len(fe.buses)
    pos = {int(b): i for i, b in enumerate(fe.buses)}
    src = 3 * pos[1]
    shift = np.where(np.arange(3 * nb) < src, np.arange(3 * nb), np.arange(3 * nb) - 3)

    def influence_gram(weights: str) -> np.ndarray:
        L = assemble_laplacian(fe, weights)
        lu = spla.splu(L.tocsc())
        zl = np.zeros((3 * nb - 3, fe.nload), dtype=complex)
        for i in range(fe.nload):
            dof = int(shift[3 * pos[int(fe.lbus[i])] + int(fe.phase[i])])
            e = np.zeros(L.shape[0], dtype=complex)
            e[dof] = 1.0
            zl[:, i] = lu.solve(e)
        return 0.5 * np.real(zl.conj().T @ zl)

    m_adm = influence_gram("admittance")
    m_imp = influence_gram("impedance")
    m_uni = influence_gram("unit")

    m_hopd = np.zeros((fe.nload, fe.nload))
    for i in range(fe.nload):
        for j in range(fe.nload):
            m_hopd[i, j] = fe.hop_distance(i, j)

    L = assemble_laplacian(fe, "admittance")
    dense = L.toarray()
    scale = float(np.max(np.abs(dense)))
    sym_defect = {
        "max_abs_L_minus_LT": float(np.max(np.abs(dense - dense.T))),
        "max_abs_L_minus_LH": float(np.max(np.abs(dense - dense.conj().T))),
        "scale_max_abs_L": scale,
        "rel_transpose_defect": float(np.max(np.abs(dense - dense.T)) / scale),
        "rel_hermitian_defect": float(np.max(np.abs(dense - dense.conj().T)) / scale),
        "eigensolver_used": "scipy.linalg.eig on L and eigh on 0.5*(L+L^H); never eigh/eigsh on L",
    }
    ev_hopd = np.linalg.eigvalsh(0.5 * (m_hopd + m_hopd.T))
    hopd_diag = {"n_negative_eigenvalues": int((ev_hopd < -1e-9 * max(abs(ev_hopd))).sum()),
                 "n_eigenvalues": int(len(ev_hopd)),
                 "min_eigenvalue": float(ev_hopd.min()), "max_eigenvalue": float(ev_hopd.max()),
                 "why_not_used_as_a_metric": ("a distance matrix is not a kernel: it has negative "
                                              "eigenvalues, so clipping them (the earlier construction) "
                                              "leaves an object that is worse than the mean predictor")}
    return {"M_adm": m_adm, "M_imp": m_imp, "M_hop": m_uni,
            "M_hop_distance_diagnostic": m_hopd,
            "hop_distance_diagnostic": hopd_diag,
            "laplacian_symmetry": sym_defect}


def singular_spectrum(S: np.ndarray) -> dict:
    """Singular spectrum and effective rank of the sensitivity operator."""
    sv = np.linalg.svd(np.asarray(S, dtype=float), compute_uv=False)
    sv0 = float(sv[0]) if sv[0] > 0 else 1.0
    return {
        "sv_normalised": [float(v) for v in sv / sv0],
        "stable_rank": float((sv ** 2).sum() / sv0 ** 2),
        "rank_1e2": int((sv > 1e-2 * sv0).sum()),
        "rank_1e3": int((sv > 1e-3 * sv0).sum()),
        "rank_1e4": int((sv > 1e-4 * sv0).sum()),
        "cond": float(sv0 / max(sv[-1], 1e-300)),
        "shape": list(S.shape),
    }


def regime_label(x: np.ndarray) -> dict:
    """Load statistics of an injection ensemble."""
    p = np.asarray(x, dtype=float)[:, :55]
    return {"mean_total_kW": float(1000.0 * p.sum(1).mean()),
            "cv_total_load": float(p.sum(1).std() / p.sum(1).mean())}


def check_q_is_deterministic(x: np.ndarray, pf: float = 0.95) -> dict:
    """Report whether the Q block is a fixed multiple of the P block.

    The ensemble generator set Q = P tan(arccos pf) entrywise, so the 110-dimensional
    input is effectively 55-dimensional.  This is recorded, not corrected: every
    baseline and every arm sees the same inputs.
    """
    p, q = np.asarray(x)[:, :55], np.asarray(x)[:, 55:]
    ratio = np.tan(np.arccos(pf))
    return {"Q_over_P_exact": bool(np.allclose(q, p * ratio, rtol=1e-6, atol=0.0)),
            "Q_over_P_ratio_median": float(np.median(q[np.abs(p) > 0] / p[np.abs(p) > 0])),
            "nominal_ratio": float(ratio),
            "effective_input_dim": 55}
