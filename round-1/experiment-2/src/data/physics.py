# NOTE (published copy): this file names server paths this repository
# does not publish (a stage it does not ship, or another run's workspace),
# so the steps that read them will not run from a clone as written:
#   /ai-inventor/aii_data/runs/run_RlJ6BnxHoG2N/iter_2/gen_hypo/claude_agent/pilot
"""Exact, vectorised multi-phase DistFlow operators for the European LV feeder.

Bases: S_base = trafo rating (0.8 MVA), V_base = 0.416 kV  =>  Z_base = 0.2163 ohm.
All impedances and injections are converted to per unit.

  lindist(order=1) : dV = -Zc_pu @ s_pu
  lindist(order=2) : dV += -Zc_pu @ dI2,  dI2_l = -conj(s_l conj(dV1_l)/|V0_l|^2)
                     the exact second-order Z-bus term.
"""
import numpy as np
import networkx as nx
import pandapower as pp
import pandapower.networks as pn

ANGS = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])


def zmat(r1, x1, r0, x0):
    Z1 = r1 + 1j * x1
    Z0 = r0 + 1j * x0
    Zs = (Z0 + 2 * Z1) / 3
    Zm = (Z0 - Z1) / 3
    M = np.full((3, 3), Zm, dtype=complex)
    np.fill_diagonal(M, Zs)
    return M


class Physics:
    def __init__(self, name='on_peak_566', v0=1.05):
        self.net = pn.ieee_european_lv_asymmetric(name)
        al = self.net.asymmetric_load
        self.nload = len(al)
        self.cols = ['p_a_mw', 'p_b_mw', 'p_c_mw']
        self.P0 = al[self.cols].values.copy()
        self.phase = np.argmax(self.P0, axis=1)
        self.lbus = al.bus.values.astype(int)
        self.v0 = v0
        self.V0 = v0 * np.exp(1j * ANGS)
        self.Vb = float(self.net.trafo.vn_lv_kv.values[0])
        self.Sb = float(self.net.trafo.sn_mva.values[0])
        self.Zb = self.Vb ** 2 / self.Sb
        tr = self.net.trafo.iloc[0]
        zk = tr.vk_percent / 100.0
        rk = tr.vkr_percent / 100.0
        self.ztr = rk + 1j * np.sqrt(max(zk ** 2 - rk ** 2, 0.0))
        G = nx.Graph()
        for _, r in self.net.line.iterrows():
            G.add_edge(int(r.from_bus), int(r.to_bus))
        self.G = G
        self.tree = nx.bfs_tree(G, 1)
        self.hop = nx.single_source_shortest_path_length(G, 1)
        self.buses = np.array(sorted(self.tree.nodes()))
        self.bidx = {b: i for i, b in enumerate(self.buses)}
        self.Zp = self._path_imp()
        self._cache = {}

    def _path_imp(self):
        Zto = {1: np.zeros((3, 3), dtype=complex)}
        for u, v in nx.bfs_edges(self.tree, 1):
            row = self.net.line[((self.net.line.from_bus == u) & (self.net.line.to_bus == v)) |
                                ((self.net.line.from_bus == v) & (self.net.line.to_bus == u))].iloc[0]
            L = row.length_km
            Zto[v] = Zto[u] + zmat(row.r_ohm_per_km * L, row.x_ohm_per_km * L,
                                   row.r0_ohm_per_km * L, row.x0_ohm_per_km * L)
        return Zto

    def shared(self, load_bus, target_bus):
        """Path impedance from source to the lowest common ancestor, plus the
        transformer series impedance (shared by every load/target pair)."""
        Z = self.Zp[nx.lowest_common_ancestor(self.tree, int(load_bus), int(target_bus))]
        M = np.zeros((3, 3), dtype=complex)
        np.fill_diagonal(M, self.ztr)
        return Z + M

    def Zc(self, targets):
        """Complex operator (3*n_targets, n_load) in pu."""
        key = tuple(int(t) for t in targets)
        if key in self._cache:
            return self._cache[key]
        Zc = np.zeros((3 * len(targets), self.nload), dtype=complex)
        for i, tb in enumerate(targets):
            for l in range(self.nload):
                Zc[3 * i:3 * i + 3, l] = self.shared(self.lbus[l], tb)[:, self.phase[l]]
        Zc = Zc / self.Zb
        self._cache[key] = Zc
        return Zc

    def s_of(self, x):
        x = np.atleast_2d(x)
        return (x[:, :self.nload] + 1j * x[:, self.nload:]) / self.Sb

    def lindist(self, x, targets, order=1, v0bus=None):
        """Complex-power -> voltage via the Z-bus expansion.

        Load current:  I_l = conj(s_l / V_l)
        First order:   I_l ~ conj(s_l)/conj(V0_pl),   dV = -Zc @ I
        Second order:  I_l ~ conj(s_l)/conj(V0_pl) - conj(s_l) conj(dV_l)/|V0_pl|^2
        """
        s = self.s_of(x)                                   # (N, nl) load power, pu
        Zt = self.Zc(targets)                              # (3m, nl)
        m = len(targets)
        V0t = np.tile(self.V0, m) if v0bus is None else v0bus
        V0_load = self.V0[self.phase]                      # (nl,)
        I1 = np.conj(s) / np.conj(V0_load)[None, :]
        dV1 = -(I1 @ Zt.T)
        if order == 1:
            return np.abs(V0t[None, :] + dV1)
        # dV1 at each load's own bus and phase
        Zall = self.Zc(self.buses)
        dV1a = -((np.conj(s) / np.conj(V0_load)[None, :]) @ Zall.T)
        lidx = np.array([self.bidx[b] for b in self.lbus])
        pidx = lidx * 3 + self.phase
        dV1_load = dV1a[:, pidx]
        I2 = np.conj(s) * np.conj(dV1_load) / (np.abs(V0_load) ** 2)[None, :]
        dV2 = -(I2 @ Zt.T)
        return np.abs(V0t[None, :] + dV1 + dV2)

    def noload_voltage(self, targets):
        net = pn.ieee_european_lv_asymmetric('on_peak_566')
        al = net.asymmetric_load
        for c in ['p_a_mw', 'p_b_mw', 'p_c_mw', 'q_a_mvar', 'q_b_mvar', 'q_c_mvar']:
            al[c] = 0.0
        pp.runpp_3ph(net, calculate_voltage_angles=True, max_iter=50, numba=False)
        va = net.res_bus_3ph[['va_a_degree', 'va_b_degree', 'va_c_degree']].values[targets]
        vm = net.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']].values[targets]
        return (vm * np.exp(1j * np.deg2rad(va))).ravel()

    def far_buses(self, frac=0.75):
        mh = max(self.hop.values())
        return np.array(sorted([b for b, d in self.hop.items() if d >= frac * mh]))

    def sheet(self, k=15, frac=0.75):
        far = self.far_buses(frac)
        return far, np.array(sorted(set(far[np.linspace(0, len(far) - 1, k).astype(int)])))

    def lin_operator(self, targets):
        """S = d|V|/dx, shape (3*n_targets, 2*n_load), pu per MW/MVAr.

        Complex LinDistFlow:  dV_q = -sum_l Zc[q,l] * I_l,
        I_l = conj(s_l)/conj(V0_pl),  s_l = (P_l + j Q_l)/Sb.
        So   dV/dP_l = -Zc[:,l] / (Sb * conj(V0_pl))
             dV/dQ_l = +j Zc[:,l] / (Sb * conj(V0_pl))
        and  d|V|/dP_l = Re( conj(V0_q)/|V0_q| * dV_q/dP_l )
                       = -Re( Zc[:,l] / (Sb * |V0_q| * V0_pl/|V0_pl| ) ) ...
        which we evaluate directly rather than simplifying.
        """
        Zt = self.Zc(targets)                          # (3m, nl) pu
        V0t = np.tile(self.V0, len(targets))
        V0l = self.V0[self.phase]                      # (nl,)
        Wt = np.conj(V0t) / np.abs(V0t)                # (3m,)
        # dV/dP_MW = -Zc / (Sb * conj(V0l));  dV/dQ_MW = +j * that
        dPd = -Zt / (self.Sb * np.conj(V0l))[None, :]
        S = np.zeros((Zt.shape[0], 2 * self.nload))
        S[:, :self.nload] = np.real(Wt[:, None] * dPd)
        S[:, self.nload:] = np.real(Wt[:, None] * (1j * dPd))
        return S


if __name__ == '__main__':
    ph = Physics()
    far, s15 = ph.sheet(15)
    print('loads', ph.nload, 'phase counts', np.bincount(ph.phase, minlength=3))
    print('Zb ohm', ph.Zb, 'ztr', ph.ztr, 'far', len(far), 'sheet', len(s15))
    V0b = ph.noload_voltage(s15)
    print('no-load |V| range', np.abs(V0b).min(), np.abs(V0b).max())
    S = ph.lin_operator(s15)
    sv = np.linalg.svd(S, compute_uv=False)
    print('S', S.shape, 'rank rel1e-6', int((sv > 1e-6 * sv[0]).sum()), 'cap', min(S.shape))
    OUT = '/ai-inventor/aii_data/runs/run_RlJ6BnxHoG2N/iter_2/gen_hypo/claude_agent/pilot'
    for sc in ['1.0', '2.0', '2.6']:
        try:
            X = np.load(f'{OUT}/ens_s{sc}_X.npy')
            Y = np.load(f'{OUT}/ens_s{sc}_Y.npy')
            ok = ~np.isnan(Y).any(1)
            X, Y = X[ok], Y[ok]
            ysd = Y.std(0).mean()
            for order in (1, 2):
                Vh = ph.lindist(X, s15, order=order, v0bus=V0b)
                e = np.sqrt(((Vh - Y) ** 2).mean())
                print(f'  scale {sc} order {order}: RMSE {e:.3e} pu ({100*e/ysd:.1f}% of Vstd)')
            for order in (1, 2):
                Vtr = ph.lindist(X[:40], s15, order=order, v0bus=V0b)
                A = np.hstack([Vtr, np.ones((40, 1))])
                c = np.linalg.lstsq(A, Y[:40], rcond=None)[0]
                Vte = ph.lindist(X[40:], s15, order=order, v0bus=V0b)
                pr = np.hstack([Vte, np.ones((len(X) - 40, 1))]) @ c
                e = np.sqrt(((pr - Y[40:]) ** 2).mean())
                print(f'  scale {sc} order{order}+calib: RMSE {e:.3e} pu ({100*e/ysd:.1f}% of Vstd)')
        except FileNotFoundError:
            pass
