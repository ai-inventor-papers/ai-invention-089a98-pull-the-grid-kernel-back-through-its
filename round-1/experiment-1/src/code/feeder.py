"""IEEE PES European LV test feeder (on_peak_566), its admittance structure, and the
Kirchhoff / sensitivity operators built from line data only.

Everything in this module is derived from the feeder's own line table (sequence
impedances r1,x1,r0,x0 and length), the transformer nameplate table, and the
55 load records' phase labels.  No AC solve enters any operator build.  The one
AC solve that can enter is the *no-load* reference voltage, which is a single
solve independent of the training set and of n; both conventions are reported.

Contents
--------
zmat(r1,x1,r0,x0)               3x3 phase-domain sequence-impedance block (ohms)
Physics                          the feeder object: graph, hops, path impedances,
                                 Z-bus (block Laplacian inverse), sensitivity
                                 operators S = d|V|/dx, LinDistFlow predictors
Later in method.py the following are used:
    Physics.lin_operator(...)    S = d|V|/dx, shape (3*n_targets, 110)
    Physics.zc_path(...)         the complex (3m, 55) pu operator
    Physics.block_laplacian()    grounded multiphase block Laplacian, 2718 x 2718
    Physics.z_bus_via_solve()    Z-bus columns for a chosen output set
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ANGS = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
TR_MODES = ("none", "pu_correct", "legacy")
V0_MODES = ("flat", "noload")


def zmat(r1: float, x1: float, r0: float, x0: float) -> np.ndarray:
    """3x3 phase-domain impedance block from sequence impedances (ohms)."""
    Z1 = r1 + 1j * x1
    Z0 = r0 + 1j * x0
    Zs = (Z0 + 2.0 * Z1) / 3.0
    Zm = (Z0 - Z1) / 3.0
    M = np.full((3, 3), Zm, dtype=complex)
    np.fill_diagonal(M, Zs)
    return M


class Physics:
    """Feeder structural object for ieee_european_lv_asymmetric('on_peak_566')."""

    def __init__(self, name: str = "on_peak_566"):
        self.name = name
        self.net = pn.ieee_european_lv_asymmetric(name)
        al = self.net.asymmetric_load
        self.nload = int(len(al))
        self.cols = ["p_a_mw", "p_b_mw", "p_c_mw"]
        self.P0 = al[self.cols].values.copy()
        self.phase = np.argmax(self.P0, axis=1).astype(int)
        self.lbus = al.bus.values.astype(int)
        self.qcols = ["q_a_mvar", "q_b_mvar", "q_c_mvar"]

        self.Vb = float(self.net.trafo.vn_lv_kv.values[0])
        self.Sb = float(self.net.trafo.sn_mva.values[0])
        self.Zb = self.Vb ** 2 / self.Sb
        tr = self.net.trafo.iloc[0]
        self.vk = float(tr.vk_percent) / 100.0
        self.vkr = float(tr.vkr_percent) / 100.0
        self.ztr = self.vkr + 1j * np.sqrt(max(self.vk ** 2 - self.vkr ** 2, 0.0))

        # --- graph over the LV buses (bus 0 is the HV ext_grid, excluded) ---
        G = nx.Graph()
        for _, r in self.net.line.iterrows():
            G.add_edge(int(r.from_bus), int(r.to_bus))
        self.G = G
        self.source = 1
        self.components = nx.number_connected_components(G)
        self.tree = nx.bfs_tree(G, self.source)
        self.hop = nx.single_source_shortest_path_length(G, self.source)
        self.buses = np.array(sorted(self.tree.nodes()))          # 906 LV buses
        self.bidx = {int(b): i for i, b in enumerate(self.buses)}
        self.maxhop = max(self.hop.values())
        self.Zto = self._path_impedances()                        # bus -> 3x3 ohms
        self._nl_voltage = None
        self.CACHE: dict = {}
        # per-phase vs three-phase base: see lin_operator docstring
        self.k3 = 3.0

    # ------------------------------------------------------------------ paths
    def _line_record(self, u: int, v: int):
        ln = self.net.line
        sel = ln[((ln.from_bus == u) & (ln.to_bus == v)) | ((ln.from_bus == v) & (ln.to_bus == u))]
        return sel.iloc[0]

    def _path_impedances(self) -> dict[int, np.ndarray]:
        """Accumulated series impedance from the source to every LV bus (ohms)."""
        Zto: dict[int, np.ndarray] = {self.source: np.zeros((3, 3), dtype=complex)}
        for u, v in nx.bfs_edges(self.tree, self.source):
            row = self._line_record(int(u), int(v))
            L = float(row.length_km)
            Zto[int(v)] = Zto[int(u)] + zmat(
                float(row.r_ohm_per_km) * L, float(row.x_ohm_per_km) * L,
                float(row.r0_ohm_per_km) * L, float(row.x0_ohm_per_km) * L,
            )
        return Zto

    def tr_ohm(self, tr_mode: str) -> np.ndarray:
        """Transformer series impedance as a 3x3 phase block, in OHMS.

        'none'       no transformer term at all
        'pu_correct' the nameplate per-unit z is the true per-unit impedance, so the
                     ohmic value is z_pu * Z_base
        'legacy'     the convention of the archived physics.Physics, whose
                     shared() adds the raw nameplate number to an OHMS sum and then
                     divides by Z_base, i.e. it puts z_pu / Z_base into per unit --
                     a factor Z_base too small.  Reproduced here deliberately.
        """
        M = np.zeros((3, 3), dtype=complex)
        if tr_mode == "none":
            return M
        if tr_mode == "pu_correct":
            np.fill_diagonal(M, self.ztr * self.Zb)
            return M
        if tr_mode == "legacy":
            np.fill_diagonal(M, self.ztr)
            return M
        raise ValueError(f"unknown tr_mode {tr_mode!r}")

    # ------------------------------------------------------------ fast LCA
    def _build_lifting(self) -> None:
        """Binary-lifting LCA so the 906 x 55 pair matrix costs milliseconds, not minutes."""
        n = len(self.buses)
        depth = np.array([self.hop[int(b)] for b in self.buses], dtype=np.int32)
        parent = np.zeros(n, dtype=np.int32)
        for b in self.buses:
            par = [p for p in self.tree.predecessors(int(b))]
            parent[self.bidx[int(b)]] = self.bidx[int(par[0])] if par else self.bidx[int(b)]
        LOG = int(np.ceil(np.log2(max(n, 2)))) + 1
        up = np.zeros((LOG, n), dtype=np.int32)
        up[0] = parent
        for j in range(1, LOG):
            up[j] = up[j - 1][up[j - 1]]
        self._depth, self._up, self._LOG = depth, up, LOG

    def lca_index(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """LCA bus index for arrays of bus indices (vectorised binary lifting)."""
        a = np.asarray(a, dtype=np.int32).copy()
        b = np.asarray(b, dtype=np.int32).copy()
        d = self._depth
        swap = d[a] < d[b]
        a[swap], b[swap] = b[swap], a[swap]
        diff = d[a] - d[b]
        for j in range(self._LOG):
            m = ((diff >> j) & 1).astype(bool)
            if m.any():
                a[m] = self._up[j][a[m]]
        eq = a == b
        out = a.copy()
        for j in range(self._LOG - 1, -1, -1):
            ua, ub = self._up[j][a], self._up[j][b]
            m = (~eq) & (ua != ub)
            a[m] = ua[m]
            b[m] = ub[m]
        m = ~eq
        out[m] = self._up[0][a[m]]
        out[eq] = a[eq]
        return out

    def lca_impedance(self, load_bus: int, target_bus: int, tr_mode: str) -> np.ndarray:
        """Path impedance from source to the lowest common ancestor + transformer."""
        Z = self.Zto[int(nx.lowest_common_ancestor(self.tree, int(load_bus), int(target_bus)))]
        return Z + self.tr_ohm(tr_mode)

    def impedance_pairs(self, tr_mode: str) -> np.ndarray:
        """(906, 55, 3, 3) complex path impedances in OHMS, cached per transformer mode.

        Entry [b, l, q, p] = Z_path(LCA(bus b, load l))[q, p], i.e. already including
        the transformer convention.  Every operator in this module indexes this array,
        so the 906 x 55 pair matrix is built once per mode instead of once per call.
        """
        key = ("pairs", tr_mode)
        if key in self.CACHE:
            return self.CACHE[key]
        if not hasattr(self, "_depth"):
            self._build_lifting()
        lb = np.array([self.bidx[int(b)] for b in self.lbus], dtype=np.int32)   # (55,)
        tb = np.arange(len(self.buses), dtype=np.int32)                        # (906,)
        A, B = np.meshgrid(tb, lb, indexing="ij")                              # (906,55)
        lca = self.lca_index(A.ravel(), B.ravel()).reshape(A.shape)
        Z = np.zeros((len(self.buses), self.nload, 3, 3), dtype=complex)
        tr = self.tr_ohm(tr_mode)
        uniq = np.unique(lca)
        zt = {int(u): self.Zto[int(self.buses[int(u)])] + tr for u in uniq}
        for u in uniq:
            m = lca == u
            Z[m] = zt[int(u)]
        self.CACHE[key] = Z
        return Z

    # -------------------------------------------------------- reference voltage
    def noload_voltage_full(self) -> np.ndarray:
        """Complex no-load voltage at every LV bus x phase, shape (906, 3).

        One AC solve, on a zero-injection copy of the feeder.  Independent of the
        training data and of n.
        """
        if self._nl_voltage is not None:
            return self._nl_voltage
        net = pn.ieee_european_lv_asymmetric(self.name)
        al = net.asymmetric_load
        for c in self.cols + self.qcols:
            al[c] = 0.0
        pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
        vm = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].values[self.buses]
        va = net.res_bus_3ph[["va_a_degree", "va_b_degree", "va_c_degree"]].values[self.buses]
        V = vm * np.exp(1j * np.deg2rad(va))
        self._nl_voltage = V
        return V

    def V0_of(self, v0mode: str, v0flat: float = 1.05) -> np.ndarray:
        """Reference voltage matrix (906, 3) for the chosen convention."""
        if v0mode == "flat":
            return np.tile(v0flat * np.exp(1j * ANGS), (len(self.buses), 1))
        if v0mode == "noload":
            return self.noload_voltage_full()
        raise ValueError(f"unknown v0mode {v0mode!r}")

    def v0_at(self, buses, v0mode: str, v0flat: float = 1.05) -> np.ndarray:
        """(3m,) complex reference voltage for the flattened (bus, phase) outputs."""
        V0 = self.V0_of(v0mode, v0flat)
        rows = np.array([self.bidx[int(b)] for b in buses])
        return V0[rows].ravel()

    # -------------------------------------------- complex sensitivity (3m, 55)
    def zc_path(self, targets, phase_labels, tr_mode: str, v0mode: str = "flat",
                v0flat: float = 1.05, Sb: float | None = None) -> np.ndarray:
        """Complex operator Zt with dV_pu = -Zt @ conj(s), shape (3m, nload).

        Zt[3i+q, l] = Z_path(lca(load_l, target_i))[q, phase_l] / Z_base
        """
        Sb = self.Sb if Sb is None else Sb
        Z = self.impedance_pairs(tr_mode)[[self.bidx[int(t)] for t in targets]]  # (m,55,3,3)
        Zsel = Z[np.arange(len(targets))[:, None], np.arange(self.nload)[None, :],
                 :, np.asarray(phase_labels, dtype=int)[None, :]]                  # (m,55,3)
        Zt = Zsel.transpose(0, 2, 1).reshape(3 * len(list(targets)), self.nload)
        return Zt / self.Zb

    def lin_operator(self, targets, phase_labels, tr_mode: str, v0mode: str,
                     v0flat: float = 1.05, Sb: float | None = None) -> np.ndarray:
        """S = d|V|/dx, shape (3m, 2*nload), per-unit volts per MW / MVAr.

        Complex LinDistFlow.  With s_l = (P_l + j Q_l)/Sb and I_l = conj(s_l)/conj(V_l):
            dV_q/dP_l = -Zt[q,l] / (Sb * conj(V0_load_l))      =: dPd[q,l]
            dV_q/dQ_l = -j * dV_q/dP_l                          (the -j is the sign that
                                                                 the archived code had as +j)
            d|V_q|/dx_l = Re( conj(V0_q)/|V0_q| * dV_q/dx_l )
        so d|V|/dP = -Re(Z)/(Sb|V0|) and d|V|/dQ = -Im(Z)/(Sb|V0|) -- both negative for a
        load, which is the physical sign.  The archived convention wrote dV/dQ = +j dV/dP,
        which returns +Im(Z)/(Sb|V0|) and is a sign error; protocol check PC7b corrects it.

        The k3 = 3 factor closes a second inherited error.  In this model Zt is divided by
        Z_base = V_LL^2/S_base while a single-phase asymmetric load's power P_l is the
        whole of that phase's power, not a third of a three-phase total; the per-phase
        relation is dV_pu = -Z_ohm * conj(P+jQ) / (V_base_phase^2 * conj(V0)) with
        V_base_phase = V_LL/sqrt(3).  Since Sb*Zb = V_LL^2 = 3 V_base_phase^2, the correct
        coefficient is 3/(Sb*Zb), i.e. exactly 3x the archived one.  Measured against the
        finite-difference AC Jacobian, the archived operator is low by 3.15x on the P block
        and has the Q block inverted; with both fixed the P block matches at corr 0.995
        and the implied effective resistance matches the path impedance to ~8 percent.
        The kernel is scale invariant in S, so the factor 3 changes nothing for the
        treatment's metric; it matters for the LinDistFlow baseline's order-2 term, which
        scales as dV1^2 and is therefore off by 9x without it.
        """
        Sb = self.Sb if Sb is None else Sb
        targets = list(targets)
        Zt = self.zc_path(targets, phase_labels, tr_mode, v0mode, v0flat, Sb)
        V0t = self.v0_at(targets, v0mode, v0flat)                  # (3m,)
        V0l = self.load_reference_voltage(phase_labels, v0mode, v0flat)
        Wt = np.conj(V0t) / np.abs(V0t)
        dPd = -self.k3 * Zt / (Sb * np.conj(V0l))[None, :]
        S = np.zeros((Zt.shape[0], 2 * self.nload))
        S[:, :self.nload] = np.real(Wt[:, None] * dPd)
        S[:, self.nload:] = np.real(Wt[:, None] * (-1j * dPd))
        return S

    def load_reference_voltage(self, phase_labels, v0mode: str,
                               v0flat: float = 1.05) -> np.ndarray:
        """Complex reference voltage at each load's own bus and phase, shape (55,)."""
        V0 = self.V0_of(v0mode, v0flat)
        rows = np.array([self.bidx[int(b)] for b in self.lbus])
        return V0[rows, np.asarray(phase_labels, dtype=int)]

    # ----------------------------------------------- grounded block Laplacian
    def block_laplacian(self, tr_mode: str = "none") -> sp.csc_matrix:
        """Multiphase grounded block Laplacian over the bus-1 component (2718 nodes).

        Node index = 3 * bus_index + phase.  Each line contributes
            L_ii += Y_e,  L_ij -= Y_e,  Y_e = inv(Zblock)   (siemens)
        with Zblock the phase-domain sequence-impedance block of that line.

        The transformer is handled by an extra ground node (index 3*906) that
        every phase of the source bus connects to through 1/z_tr.  Grounding that
        node (deleting its row and column) then places z_tr in series with the
        source, so every Z-bus entry picks up z_tr exactly once -- which is what
        the path construction does.
        """
        nb = len(self.buses)
        nn = 3 * nb
        gnode = nn
        N = nn + 1
        rows, cols, vals = [], [], []
        for _, r in self.net.line.iterrows():
            u, v = int(r.from_bus), int(r.to_bus)
            if u not in self.bidx or v not in self.bidx:
                continue
            L = float(r.length_km)
            Zb_ = zmat(float(r.r_ohm_per_km) * L, float(r.x_ohm_per_km) * L,
                       float(r.r0_ohm_per_km) * L, float(r.x0_ohm_per_km) * L)
            Ye = np.linalg.inv(Zb_)
            iu, iv = 3 * self.bidx[u], 3 * self.bidx[v]
            for a in range(3):
                for b in range(3):
                    val = Ye[a, b]
                    rows += [iu + a, iv + a, iu + a, iv + a]
                    cols += [iu + b, iv + b, iv + b, iu + b]
                    vals += [val, val, -val, -val]
        if tr_mode != "none":
            zt = self.tr_ohm(tr_mode)[0, 0]
            ys = 1.0 / zt
            i0 = 3 * self.bidx[self.source]
            for a in range(3):
                rows += [i0 + a, gnode, i0 + a, gnode]
                cols += [i0 + a, gnode, gnode, i0 + a]
                vals += [ys, ys, -ys, -ys]
        Lm = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
        return Lm

    def z_bus_solve(self, targets, phase_labels, tr_mode: str, ground_bus: int | None = None):
        """Z-bus entries at the (target, phase) rows x 55 load (bus, phase) columns.

        Returns a complex (3m, 55) array in OHMS.  Solves the grounded Laplacian
        for the 55 load columns instead of inverting 2718 x 2718.
        """
        Lm = self.block_laplacian(tr_mode)
        nb = len(self.buses)
        nn = 3 * nb
        gnode = nn
        keep = list(range(nn))
        if tr_mode == "none":
            i0 = 3 * self.bidx[self.source]
            keep = [k for k in range(nn) if k not in (i0, i0 + 1, i0 + 2)]
        else:
            keep = list(range(nn))          # gnode removed => Dirichlet at ground
        Lg = Lm[keep][:, keep].tocsc()
        lu = spla.splu(Lg)
        cols = [3 * self.bidx[int(self.lbus[l])] + int(phase_labels[l]) for l in range(self.nload)]
        E = np.zeros((len(keep), self.nload), dtype=complex)
        pos = {k: i for i, k in enumerate(keep)}
        for j, c in enumerate(cols):
            E[pos[c], j] = 1.0
        Zsol = lu.solve(E)                                     # (len(keep), 55) ohms
        Zrows = []
        for tb in targets:
            for a in range(3):
                k = 3 * self.bidx[int(tb)] + a
                Zrows.append(Zsol[pos[k], :] if k in pos else np.zeros(self.nload, dtype=complex))
        return np.array(Zrows)

    def zc_path_matrix(self, targets, phase_labels, tr_mode: str) -> np.ndarray:
        """The same (3m, 55) complex Z matrix built from BFS path accumulation (ohms)."""
        Zp = np.zeros((3 * len(targets), self.nload), dtype=complex)
        for i, tb in enumerate(targets):
            for l in range(self.nload):
                Zp[3 * i:3 * i + 3, l] = self.lca_impedance(
                    self.lbus[l], int(tb), tr_mode)[:, int(phase_labels[l])]
        return Zp

    # ------------------------------------------------------------- LinDistFlow
    def s_of(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (x[:, : self.nload] + 1j * x[:, self.nload:]) / self.Sb

    def lindist(self, x, targets, phase_labels, order: int = 1, v0mode: str = "flat",
                tr_mode: str = "none", v0flat: float = 1.05,
                i2_form: str = "conj2") -> np.ndarray:
        """|V| at the flattened (bus, phase) outputs, (N, 3m).

        order = 1 : dV = -k3 Zt I1,           I1 = conj(s)/conj(V0)
        order = 2 : dV = -k3 Zt (I1 + I2),    I2 = conj(s) conj(dV1_load) / conj(V0)^2
                    (the exact second-order Z-bus current correction; i2_form='abs2'
                     reproduces the archived form which used |V0|^2 instead of conj(V0)^2)
        """
        targets = list(targets)
        s = self.s_of(x)
        Zt = self.zc_path(targets, phase_labels, tr_mode, v0mode, v0flat)
        V0t = self.v0_at(targets, v0mode, v0flat)
        V0l = self.load_reference_voltage(phase_labels, v0mode, v0flat)
        I1 = np.conj(s) / np.conj(V0l)[None, :]
        dV1 = -self.k3 * (I1 @ Zt.T)
        V = V0t[None, :] + dV1
        if order == 1:
            return np.abs(V)
        # second-order Z-bus current correction, evaluated at the load buses
        key = ("zall", tr_mode)
        if key in self.CACHE:
            Zall = self.CACHE[key]
        else:
            Zall = self.zc_path(self.buses, phase_labels, tr_mode, v0mode, v0flat)
            self.CACHE[key] = Zall
        dV1a = -self.k3 * (I1 @ Zall.T)
        lidx = np.array([self.bidx[int(b)] for b in self.lbus])
        pidx = lidx * 3 + np.asarray(phase_labels, dtype=int)
        dV1_load = dV1a[:, pidx]
        if i2_form == "conj2":
            den = np.conj(V0l) ** 2
        elif i2_form == "abs2":
            den = (np.abs(V0l) ** 2).astype(complex)
        else:
            raise ValueError(i2_form)
        I2 = np.conj(s) * np.conj(dV1_load) / den[None, :]
        dV2 = -self.k3 * (I2 @ Zt.T)
        return np.abs(V0t[None, :] + dV1 + dV2)

    # ---------------------------------------------------------------- helpers
    def far_buses(self, frac: float = 0.75) -> np.ndarray:
        return np.array(sorted([b for b, d in self.hop.items() if d >= frac * self.maxhop]))

    def sheet(self, k: int = 15, frac: float = 0.75):
        far = self.far_buses(frac)
        sel = np.array(sorted({int(far[i]) for i in np.linspace(0, len(far) - 1, k).astype(int)}))
        return far, sel

    def structural_facts(self) -> dict:
        nb = len(self.net.bus)
        comp = nx.node_connected_component(self.G, self.source)
        far, s15 = self.sheet(15)
        return dict(
            n_bus_table=int(nb),
            n_bus_after_source=int(nb - 1),
            n_lines=int(len(self.net.line)),
            n_asym_load=int(self.nload),
            n_load_buses=int(len(np.unique(self.lbus))),
            phase_counts=[int(v) for v in np.bincount(self.phase, minlength=3)],
            native_total_kw=float(self.P0.sum() * 1e3),
            single_component_of_source=int(len(comp)) == int(len(self.buses)),
            n_components=int(self.components),
            max_hop=int(self.maxhop),
            n_far_buses=int(len(far)),
            far_hop_range=[int(min(self.hop[b] for b in far)), int(max(self.hop[b] for b in far))],
            sheet15=[int(b) for b in s15],
            Z_base_ohm=float(self.Zb),
            trafo_vk_percent=float(self.vk * 100),
            trafo_vkr_percent=float(self.vkr * 100),
            ztr_pu=float(abs(self.ztr)),
        )
