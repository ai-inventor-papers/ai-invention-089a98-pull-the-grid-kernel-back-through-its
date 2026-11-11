"""Pre-registered fold schedule, exactly as the artifact plan specifies.

12 seeds x 2 partition schemes, indices from
    numpy.random.default_rng(20261111 + 7919*seed + 131*scheme)  over the 220 rows,
seed 0..11.

Scheme 0: idx = permutation(220); train = idx[:n], test = idx[n:]
Scheme 1: idx = permutation(220); h = 110; train = idx[:h][:n], test = idx[h:]
          skipped if len(train) < n or len(test) < 3

Consequence, counted not asserted: scheme 1 yields 12 folds at n <= 80 and ZERO at
n = 160, because the train set is drawn from a 110-row half. Realized counts are
therefore 24 at n <= 80 and 12 at n = 160.
"""
from __future__ import annotations

import numpy as np

N_ROWS = 220
SEEDS = tuple(range(12))
SCHEMES = (0, 1)
SIZES = (10, 20, 40, 80, 160)
EXPECTED_COUNTS = {10: 24, 20: 24, 40: 24, 80: 24, 160: 12}


def fold_indices(n: int, seed: int, scheme: int) -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng(20261111 + 7919 * seed + 131 * scheme)
    idx = rng.permutation(N_ROWS)
    if scheme == 0:
        tr, te = idx[:n], idx[n:]
    else:
        h = N_ROWS // 2
        tr, te = idx[:h][:n], idx[h:]
    if len(tr) < n or len(te) < 3:
        return None
    assert len(np.intersect1d(tr, te)) == 0, "train and test overlap"
    return np.sort(tr), np.sort(te)


def enumerate_folds(sizes=SIZES, seeds=SEEDS, schemes=SCHEMES):
    """List of (n, seed, scheme, train_idx, test_idx), plus the realized counts."""
    out = []
    counts: dict[int, int] = {n: 0 for n in sizes}
    per_scheme: dict[tuple[int, int], int] = {}
    for n in sizes:
        for scheme in schemes:
            for seed in seeds:
                f = fold_indices(n, seed, scheme)
                key = (n, scheme)
                per_scheme.setdefault(key, 0)
                if f is None:
                    continue
                per_scheme[key] += 1
                counts[n] += 1
                out.append((n, seed, scheme, f[0], f[1]))
    return out, counts, {f"{k[0]}_{k[1]}": v for k, v in per_scheme.items()}
