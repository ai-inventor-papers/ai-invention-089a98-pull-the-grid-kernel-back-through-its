"""Phase-expanded admittance-weighted block Laplacian of the IEEE PES European LV feeder.

Builds the grounded 2715 x 2715 complex-symmetric block Laplacian from LINE DATA
ONLY (zero AC solves), verifies the line-network bus-impedance identity with its
scope stated, and produces the spectral feature blocks of the five constructions
the artifact screens (R1a/R1b/R2/R3/R4/R5).

Conventions fixed before any fit (recorded in results/protocol_checks.json):
  (a) every selected eigenvector is scaled to unit L2 norm over the 2715 entries;
  (b) SIGN/PHASE: general-eig eigenvectors are rotated so that their largest-magnitude
      entry is real and positive (LAPACK returns an arbitrary complex scalar, so
      without this step the artifact is not reproducible across BLAS builds);
  (c) ORDERING: by Re(lambda) ascending (R3: by the largest eigenvalues of the
      Hermitian part of the inverse, i.e. the smallest of the Laplacian);
  (d) GATHERING: a load's own phase entry, through an explicit full->grounded index map.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.linalg as sla

from physics import Physics, zmat

ANGS = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])


def _lca_depth_table(ph: Physics) -> np.ndarray:
    """(M,M) int32 table of hop distance from the root to the LCA of each bus pair.

    Vectorised replacement for a 906x906 python loop of nx.lowest_common_ancestor
    (340 s -> 0.1 s), spot-checked against networkx in checks.py.
    """
    buses = ph.buses
    M = len(buses)
    parent = {v: u for u, v in nx.bfs_edges(ph.tree, 1)}
    Lmax = max(ph.hop.values()) + 1
    anc = np.full((M, Lmax), -1, dtype=np.int64)
    for i, b in enumerate(buses):
        chain = [b]
        while chain[-1] in parent:
            chain.append(parent[chain[-1]])
        chain = chain[::-1]
        anc[i, :len(chain)] = chain
        anc[i, len(chain):] = -(i + 2)          # unique pad, never a common ancestor
    Dl = np.zeros((M, M), dtype=np.int32)
    for s in range(0, M, 256):
        blk = anc[s:s + 256]
        same = (blk[:, None, :] == anc[None, :, :]) & (blk[:, None, :] >= 0) & (anc[None, :, :] >= 0)
        Dl[s:s + 256] = same.sum(2)
    return Dl - 1                               # chain prefix length -> LCA depth in hops


@dataclass
class Route:
    name: str
    F: np.ndarray | None                      # (55, q) gathered block, None if unavailable
    lam: np.ndarray | None                    # selected eigenvalues
    resid: np.ndarray | None                  # eigenpair residuals
    imfrac: float | None
    note: str = ""
    Vraw: np.ndarray | None = None            # (2715, 45) unit-norm, phase-fixed vectors


@dataclass
class FeederOps:
    ph: Physics
    Lg: np.ndarray
    Zk: np.ndarray                            # line-network block bus impedance, kept indices
    Dl: np.ndarray                            # LCA hop-depth table (M,M)
    gmap: np.ndarray                          # full 3M -> grounded index, -1 if removed
    lidx: np.ndarray                          # bus index of each load
    gather: np.ndarray                        # grounded index of each load's own phase
    Zll: np.ndarray                           # (55,55) load-to-load own-phase block Zbus
    checks: dict = field(default_factory=dict)
    eig_full: tuple | None = None
    routes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ assembly
    @staticmethod
    def assemble(ph: Physics, reverse: bool = False) -> np.ndarray:
        """Dense (3M x 3M) complex block Laplacian in per unit, line data only."""
        buses = ph.buses
        idx = {b: i for i, b in enumerate(buses)}
        M = len(buses)
        Lb = np.zeros((3 * M, 3 * M), dtype=complex)
        rows = list(ph.net.line.itertuples(index=False))
        if reverse:
            rows = rows[::-1]
        for r in rows:
            u, v = int(r.from_bus), int(r.to_bus)
            if u not in idx or v not in idx:
                continue
            L = r.length_km
            Zblk = zmat(r.r_ohm_per_km * L, r.x_ohm_per_km * L,
                        r.r0_ohm_per_km * L, r.x0_ohm_per_km * L) / ph.Zb
            Y = np.linalg.inv(Zblk)
            iu = slice(3 * idx[u], 3 * idx[u] + 3)
            iv = slice(3 * idx[v], 3 * idx[v] + 3)
            Lb[iu, iu] += Y
            Lb[iv, iv] += Y
            Lb[iu, iv] -= Y
            Lb[iv, iu] -= Y
        return Lb

    @classmethod
    def build(cls, ph: Physics, verbose: bool = True) -> "FeederOps":
        t_all = time.time()
        buses = ph.buses
        idx = {b: i for i, b in enumerate(buses)}
        M = len(buses)
        src = idx[1]
        keep = np.array([i for i in range(3 * M) if i not in (3 * src, 3 * src + 1, 3 * src + 2)])
        gmap = np.full(3 * M, -1, dtype=np.int64)
        gmap[keep] = np.arange(len(keep))

        t0 = time.time()
        Lb = cls.assemble(ph)
        t_asm = time.time() - t0
        Lg = Lb[np.ix_(keep, keep)]
        del Lb
        assert Lg.shape == (3 * M - 3, 3 * M - 3), Lg.shape

        # ---- LCA table and shared-path block bus impedance (no inversion anywhere)
        t0 = time.time()
        Dl = _lca_depth_table(ph)
        t_lca = time.time() - t0
        parent = {v: u for u, v in nx.bfs_edges(ph.tree, 1)}
        A3 = np.zeros((M, Dl.max() + 1, 3, 3), dtype=complex)
        for i, b in enumerate(buses):
            chain = [b]
            while chain[-1] in parent:
                chain.append(parent[chain[-1]])
            for k, node in enumerate(chain[::-1]):
                A3[i, k] = ph.Zp[node] / ph.Zb

        t0 = time.time()
        Mk = len(keep)
        rows = np.arange(M)[:, None]
        Zpath_full = A3[rows, Dl].transpose(0, 2, 1, 3).reshape(3 * M, 3 * M)
        Zk = Zpath_full[np.ix_(keep, keep)]
        del Zpath_full
        t_zp = time.time() - t0

        lidx = np.array([idx[int(b)] for b in ph.lbus])
        gather = gmap[3 * lidx + ph.phase]
        assert (gather >= 0).all()
        Zll = A3[lidx, Dl[np.ix_(lidx, lidx)]]              # (55,55,3,3) load-to-load block Zbus
        # own-phase column entry: Zbus[i,j] at (phase_i, phase_j)
        Zll_own = Zll[np.arange(55)[:, None], np.arange(55)[None, :],
                      ph.phase[:, None], ph.phase[None, :]]
        # the block Zbus is complex SYMMETRIC (transposed lines), not Hermitian
        assert np.allclose(Zll_own, Zll_own.T), "own-phase load block must be complex symmetric"

        checks: dict = {}
        checks["lap_shape"] = list(Lg.shape)
        checks["lap_abs_max"] = float(np.abs(Lg).max())
        checks["lap_asym_max"] = float(np.abs(Lg - Lg.T).max())
        checks["lap_nonherm_max"] = float(np.abs(Lg - Lg.conj().T).max())
        checks["lap_im_diag_ratio"] = float(np.abs(Lg.diagonal().imag).max() / np.abs(Lg.diagonal()).max())
        checks["lap_cond"] = None
        checks["assembly_seconds"] = t_asm
        checks["lca_table_seconds"] = t_lca
        checks["zpath_seconds"] = t_zp

        # ---- identity: inv(Lg) vs line-network Zpath, then vs transformer-inclusive Zbus
        t0 = time.time()
        Li = np.linalg.inv(Lg)
        checks["inv_seconds"] = time.time() - t0
        # condition number estimated from the general spectrum (a full SVD of a 2715-square
        # complex matrix costs minutes and buys nothing here)
        _w = sla.eigvals(Lg, check_finite=False)
        checks["lap_cond"] = float(np.abs(_w).max() / np.abs(_w).min())
        checks["lap_cond_estimator"] = "max|lambda|/min|lambda| from zgeev eigenvalues"
        del _w
        checks["identity_line_network_max_rel"] = float(np.abs(Li - Zk).max() / np.abs(Zk).max())
        checks["identity_line_network_mean_rel"] = float(np.abs(Li - Zk).mean() / np.abs(Zk).mean())
        Mtr = np.zeros((3, 3), dtype=complex)
        np.fill_diagonal(Mtr, ph.ztr)
        nb = Mk // 3
        Ztr = Zk + np.kron(np.ones((nb, nb)), Mtr)
        checks["identity_transformer_inclusive_max_rel"] = float(np.abs(Li - Ztr).max() / np.abs(Ztr).max())
        checks["ztr_over_zpath_max"] = float(np.abs(ph.ztr) / np.abs(Zk).max())
        # independent check of the gather offset AND the identity at once:
        # inv(Lg) restricted to the loads' own-phase entries must equal the 55x55
        # own-phase block Zbus built by the shared-path rule.
        sub = Li[np.ix_(gather, gather)]
        checks["gather_identity_max_rel"] = float(np.abs(sub - Zll_own).max() / np.abs(Zll_own).max())
        checks["gather_index_min"] = int(gather.min())
        checks["gather_index_max"] = int(gather.max())
        checks["gather_unique"] = int(len(set(gather.tolist())))
        del Ztr

        # ---- reverse-order rebuild (determinism / order-independence, step 14(i))
        Lb_rev = cls.assemble(ph, reverse=True)
        Lg_rev = Lb_rev[np.ix_(keep, keep)]
        checks["reverse_order_max_diff"] = float(np.abs(Lg_rev - Lg).max())
        del Lb_rev, Lg_rev

        ops = cls(ph=ph, Lg=Lg, Zk=Zk, Dl=Dl, gmap=gmap, lidx=lidx, gather=gather,
                  Zll=Zll_own, checks=checks)

        # ---- spectral routes
        ops.eig_full = ops._general_eig()
        checks["eig_seconds"] = ops.eig_full[2]
        ops.routes = ops._build_routes()
        # Step 14(ii): eigh/eigvalsh/eigsh on a non-Hermitian matrix are WRONG, measured
        checks["hermiticity_probe"] = ops.hermiticity_probe()
        checks["total_seconds"] = time.time() - t_all
        if verbose:
            print(f"[feeder] built in {checks['total_seconds']:.1f}s; "
                  f"identity line-net {checks['identity_line_network_max_rel']:.2e}, "
                  f"transformer-incl {checks['identity_transformer_inclusive_max_rel']:.2e}, "
                  f"gather {checks['gather_identity_max_rel']:.2e}")
        return ops

    # ------------------------------------------------------------------ spectra
    def _general_eig(self):
        t0 = time.time()
        w, V = sla.eig(self.Lg, right=True, left=False, check_finite=False, overwrite_a=False)
        return w, V, time.time() - t0

    @staticmethod
    def _fix_phase(v: np.ndarray) -> tuple[np.ndarray, int]:
        """Rotate v so its largest-magnitude entry is real and positive."""
        j = int(np.argmax(np.abs(v)))
        if np.abs(v[j]) > 0:
            v = v * np.exp(-1j * np.angle(v[j]))
        return v, j

    def _gather(self, vecs: np.ndarray) -> np.ndarray:
        return vecs[self.gather, :]

    def _build_routes(self) -> dict[str, Route]:
        routes: dict[str, Route] = {}
        w, V, _ = self.eig_full
        order = np.argsort(w.real)
        self.checks["re_lam_smallest_45"] = [float(x) for x in w.real[order[:45]]]
        self.checks["im_lam_frac_smallest_45"] = [float(x) for x in
                                                  (np.abs(w.imag[order[:45]]) / np.abs(w[order[:45]]))]
        self.checks["frac_im_over_rel_gt_1e6_45"] = float(
            np.mean(np.abs(w.imag[order[:45]]) / np.abs(w[order[:45]]) > 1e-6))
        self.checks["eig_argmax_entries_45"] = []

        # --- R1: general complex eigenvectors, real part (R1a) and re+im (R1b)
        Vs = np.empty_like(V[:, order[:45]])
        res_R1 = np.zeros(45)
        for c in range(45):
            k = order[c]
            v, j = self._fix_phase(V[:, k].copy())
            Vs[:, c] = v
            self.checks["eig_argmax_entries_45"].append(j)
            res_R1[c] = np.linalg.norm(self.Lg @ v - w[k] * v) / (np.abs(w[k]) * np.linalg.norm(v))
        G1 = self._gather(Vs)
        self.checks["r1_residual_max"] = float(res_R1.max())
        self.checks["r1_residual_mean"] = float(res_R1.mean())
        routes["R1"] = Route("R1", G1, w[order[:45]], res_R1,
                             float(np.mean(np.abs(w.imag[order[:45]]) / np.abs(w[order[:45]]) > 1e-6)),
                             "general complex eigenvectors (complex-symmetric, non-Hermitian)",
                             Vraw=Vs)

        # --- R2: Hermitian part
        H = 0.5 * (self.Lg + self.Lg.conj().T)
        t0 = time.time()
        w2, V2 = sla.eigh(H)
        self.checks["eigh_seconds"] = time.time() - t0
        V2 = V2[:, :45]
        res_R2 = np.array([np.linalg.norm(self.Lg @ V2[:, c] - w2[c] * V2[:, c])
                           / (np.abs(w2[c]) * np.linalg.norm(V2[:, c])) for c in range(45)])
        self.checks["r2_residual_max"] = float(res_R2.max())
        self.checks["r2_residual_mean"] = float(res_R2.mean())
        self.checks["herm_vs_general_re_lam_max_dev_45"] = float(np.abs(w2[:45] - w.real[order[:45]]).max())
        routes["R2"] = Route("R2", self._gather(V2), w2[:45], res_R2, 0.0,
                             "Hermitian part 0.5(L+L^H)")

        # --- R3: Hermitian part of the inverse (bus-impedance spectrum)
        Li = np.linalg.inv(self.Lg)
        Zh = 0.5 * (Li + Li.conj().T)
        w3, V3 = sla.eigh(Zh)
        V3 = V3[:, ::-1][:, :45]                            # k LARGEST eigenvalues
        w3 = w3[::-1]
        res_R3 = np.array([np.linalg.norm(Zh @ V3[:, c] - w3[c] * V3[:, c])
                           / (np.abs(w3[c]) * np.linalg.norm(V3[:, c])) for c in range(45)])
        for c in range(45):
            V3[:, c], _ = self._fix_phase(V3[:, c].copy())
        G3 = self._gather(V3)
        # consistency of R3 with R1: principal angles between the two 11-dim spans, as |cos|
        q1 = np.linalg.qr(G1[:, :11])[0]
        q3 = np.linalg.qr(G3[:, :11])[0]
        sv = np.linalg.svd(q1.conj().T @ q3, compute_uv=False)
        self.checks["r3_vs_r1_angle_min_deg"] = float(np.degrees(np.arccos(np.clip(sv.max(), -1, 1))))
        self.checks["r3_vs_r1_angle_max_deg"] = float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
        routes["R3"] = Route("R3", G3, w3, res_R3, 0.0,
                             "Hermitian part of inv(Lg); for complex-symmetric Lg the exact "
                             "eigenvectors coincide with R1, so R3-as-implemented is the "
                             "Hermitian-part variant, not a new kernel")
        del Li, Zh, V3

        # --- R4: impedance-weighted Laplacian (weights = Zblock, not inv(Zblock))
        buses = self.ph.buses
        idx = {b: i for i, b in enumerate(buses)}
        M = len(buses)
        Limp = np.zeros((3 * M, 3 * M), dtype=complex)
        for r in self.ph.net.line.itertuples(index=False):
            u, v = int(r.from_bus), int(r.to_bus)
            if u not in idx or v not in idx:
                continue
            L = r.length_km
            Zblk = zmat(r.r_ohm_per_km * L, r.x_ohm_per_km * L,
                        r.r0_ohm_per_km * L, r.x0_ohm_per_km * L) / self.ph.Zb
            iu = slice(3 * idx[u], 3 * idx[u] + 3)
            iv = slice(3 * idx[v], 3 * idx[v] + 3)
            Limp[iu, iu] += Zblk
            Limp[iv, iv] += Zblk
            Limp[iu, iv] -= Zblk
            Limp[iv, iu] -= Zblk
        src = idx[1]
        keep = np.array([i for i in range(3 * M) if i not in (3 * src, 3 * src + 1, 3 * src + 2)])
        Lzg = Limp[np.ix_(keep, keep)]
        del Limp
        t0 = time.time()
        w4, V4 = sla.eig(Lzg, right=True, left=False, check_finite=False, overwrite_a=False)
        self.checks["eig_r4_seconds"] = time.time() - t0
        o4 = np.argsort(w4.real)
        V4s = np.empty_like(V4[:, o4[:45]])
        for c in range(45):
            V4s[:, c], _ = self._fix_phase(V4[:, o4[c]].copy())
        res_R4 = np.array([np.linalg.norm(Lzg @ V4s[:, c] - w4[o4[c]] * V4s[:, c])
                           / (np.abs(w4[o4[c]]) * np.linalg.norm(V4s[:, c])) for c in range(45)])
        # how wrong is this construction as a bus-impedance operator?
        Li4 = np.linalg.inv(Lzg)
        self.checks["r4_impedance_weighted_max_rel_vs_zpath"] = float(
            np.abs(Li4 - self.Zk).max() / np.abs(self.Zk).max())
        self.checks["r4_impedance_weighted_corr_vs_zpath"] = float(
            np.corrcoef(Li4.ravel().real, self.Zk.ravel().real)[0, 1])
        del Li4, Lzg
        routes["R4"] = Route("R4", self._gather(V4s), w4[o4[:45]], res_R4,
                             float(np.mean(np.abs(w4.imag[o4[:45]]) / np.abs(w4[o4[:45]]) > 1e-6)),
                             "impedance-weighted (not admittance-weighted) Laplacian")

        # --- R5: diagonal-similarity symmetrisation, availability test
        diag = self.Lg.diagonal()
        im_diag_ratio = float(np.abs(diag.imag).max() / np.abs(diag).max())
        # phase-consistency condition: exists real d>0 with d_i L_ij / d_j real for all i,j.
        # On the diagonal this already forces Im L_ii = 0, so the route is unavailable
        # whenever im_diag_ratio is not ~0. Record both numbers.
        self.checks["r5_im_diag_ratio"] = im_diag_ratio
        self.checks["r5_available"] = bool(im_diag_ratio < 1e-12)
        if self.checks["r5_available"]:
            s = np.sqrt(diag.real)
            A = self.Lg / (s[:, None] * s[None, :])
            wa, Va = sla.eigh(0.5 * (A + A.conj().T))
            Va = (Va[:, :45] / s[:, None])
            Va /= np.linalg.norm(Va, axis=0, keepdims=True)
            routes["R5"] = Route("R5", self._gather(Va), wa[:45], None, 0.0,
                                 "diagonal-similarity symmetrised")
        else:
            routes["R5"] = Route("R5", None, None, None, None,
                                 f"UNAVAILABLE: max|Im diag|/|diag| = {im_diag_ratio:.3f} != 0, so no "
                                 f"real diagonal D makes D Lg D^-1 Hermitian")
        return routes

    # ------------------------------------------------------------------ helpers
    def features(self, route: str, k: int, variant: str = "real") -> np.ndarray:
        """(55, q) gathered feature block for a route and truncation k."""
        r = self.routes[route]
        if r.F is None:
            raise ValueError(f"route {route} unavailable: {r.note}")
        G = r.F[:, :k]
        if variant == "real":
            return np.real(G)
        if variant == "reim":
            return np.hstack([np.real(G), np.imag(G)])
        raise ValueError(variant)

    # ------------------------------------------------------------------ caching
    def save_cache(self, path) -> None:
        """Cache the spectra so downstream steps never re-run a 2715-square zgeev."""
        import pathlib

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob: dict[str, np.ndarray] = {}
        for name, r in self.routes.items():
            if r.F is not None:
                blob[f"F_{name}"] = r.F
            if r.lam is not None:
                blob[f"lam_{name}"] = np.asarray(r.lam, dtype=complex)
            if r.resid is not None:
                blob[f"resid_{name}"] = np.asarray(r.resid, dtype=float)
            if r.Vraw is not None:
                blob[f"Vraw_{name}"] = r.Vraw
        blob["gather"] = self.gather
        blob["Zll"] = self.Zll
        np.savez_compressed(p, **blob)
        import json

        def _ser(v):
            if isinstance(v, (int, float, str, bool)) or v is None:
                return True
            if isinstance(v, (list, tuple)):
                return all(_ser(x) for x in v)
            if isinstance(v, dict):
                return all(isinstance(k, str) and _ser(x) for k, x in v.items())
            return False

        p.with_suffix(".json").write_text(json.dumps(
            {k: v for k, v in self.checks.items() if _ser(v)}, indent=1))

    def hermiticity_probe(self) -> dict:
        """Step 14(ii): show that eigh/eigvalsh on Lg is WRONG, measurably.

        On a cache load `eig_full` holds no vectors, so the eigenvalues are recomputed
        from `eigvals` (cheap, no vectors) and the eigsh attempt is skipped: its outcome
        is already a recorded fact in the cached protocol checks.
        """
        out = {}
        w = self.eig_full[0]
        if w is None or self.Lg.shape[0] < 4:
            return {"status": "not run on cache load; recorded in the cached protocol checks"}
        try:
            wv = np.linalg.eigvalsh(self.Lg)
            order = np.argsort(w.real)
            out["eigvalsh_vs_eig_max_dev_45"] = float(np.abs(np.sort(wv)[:45] - w.real[order[:45]]).max())
        except np.linalg.LinAlgError as e:                       # pragma: no cover
            out["eigvalsh_vs_eig_max_dev_45"] = None
            out["eigvalsh_error"] = repr(e)
        # eigsh on a non-Hermitian matrix is invalid AND can spin for many minutes; bound it
        # with a wall-clock alarm so the probe can never hang the run (measured fact, bounded).
        try:
            import scipy.sparse.linalg as spla
            import signal
            import warnings

            def _alarm(signum, frame):
                raise TimeoutError("eigsh exceeded its 180 s wall-clock allowance")

            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(180)
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vals = spla.eigsh(self.Lg, k=14, which="SM", return_eigenvectors=False)
                out["eigsh_outcome"] = "converged"
                out["eigsh_vals_smallest"] = [complex(v) for v in np.sort_complex(vals)[:5]]
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            out["eigsh_seconds"] = time.time() - t0
        except Exception as e:                                   # noqa: BLE001 - recorded, not raised
            out["eigsh_outcome"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
