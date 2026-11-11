"""T2 single-cell end-to-end smoke test plus timing (n=40, fold 0) and n=160 cell."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
import method as M

M._init_worker()
M._G["V1"], M._G["V2"] = M._lindist_all()
bases = {(r["n"], r["fold_id"]): r for r in M.jload(M.RAW / "baselines.json")}

for n in (40, 160):
    f = M.folds_for(n)[0]
    t0 = time.time()
    cell = M._sweep_cell((n, f, bases[(n, f["fold_id"])]))
    dt = time.time() - t0
    b = bases[(n, f["fold_id"])]
    print(f'=== n={n} fold {f["fold_id"]} ({dt:.1f}s) ===')
    print(f'  vsd_train {cell["vsd_train"]:.5f}  mean-predictor {cell["mean_predictor_rmse"]:.4f}')
    for k in ("b1a_rbf_median", "b2_hop_laplacian", "b3_admittance", "b4_ridge",
              "b5p_lindist_honest", "b5s_p17_jointmap"):
        print(f'  {k:24s} {b[k]["rmse"]:.4f}')
    v = cell["variants"]
    print(f'  A0_raw (k45,g0)          {v["k45_g0.0"]["rmse"]:.4f}')
    print(f'  A4_gamma1 (k45,g1)       {v["k45_g1.0"]["rmse"]:.4f}')
    print(f'  k=3 g=0                  {v["k3_g0.0"]["rmse"]:.4f}')
    print(f'  k=11 g=0.5               {v["k11_g0.5"]["rmse"]:.4f}')
    print(f'  floor 1e-3               {v["floor_0.001"]["rmse"]:.4f} '
          f'(dim {v["floor_0.001"]["dim"]})')
    print(f'  diagnostics raw cond_K {cell["diagnostics"]["raw"]["cond_K"]:.3e} '
          f'pivot {cell["diagnostics"]["raw"]["min_cholesky_pivot"]:.3e}')
    print('  jitter sweep raw', cell["diagnostics"]["raw"]["jitter_sweep"])
