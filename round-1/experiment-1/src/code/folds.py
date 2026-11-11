"""The shared fold recipe: 12 seeds x 2 partition schemes = 24 folds per cell.

Scheme 0 (complement split):  a fresh permutation of the 220 draws, train = idx[:n],
                              test = idx[n:].  Test grows as n shrinks.
Scheme 1 (fixed test block):  a fresh permutation, test = idx[:60] FIXED across n,
                              train = idx[60:60+n].

DEVIATION (protocol check PC2): the direction's second scheme was worded as
"fixed-half test block, train = n drawn from one half, test = the other half".
That needs n + 110 <= 220 and is infeasible at n = 160, so it is redefined here as
a fixed 60-row test block with the training rows drawn from the remaining 160.  The
rule is stated exactly so a later evaluator can detect that scheme-1 folds here will
NOT match another artifact that implemented the literal wording.

RESERVED, and asserted against: seeds 12..29, sizes 60 and 120, and any third
partition scheme.  The 2.6x ensemble is reserved as confirmation evidence.
"""
from __future__ import annotations

import hashlib
import numpy as np

N_DRAWS = 220
SEEDS = tuple(range(12))
SCHEMES = (0, 1)
SIZES = (10, 20, 40, 80, 160)
FIXED_TEST = 60
RESERVED_SIZES = (60, 120)
RESERVED_SEED_MIN = 12
N_TEST_SCHEME1 = FIXED_TEST


def assert_not_reserved(n: int, seed: int, scheme: int) -> None:
    if n in RESERVED_SIZES:
        raise AssertionError(f"n={n} is a reserved size")
    if seed >= RESERVED_SEED_MIN:
        raise AssertionError(f"seed={seed} is reserved (>= {RESERVED_SEED_MIN})")
    if scheme not in SCHEMES:
        raise AssertionError(f"scheme={scheme} is not one of {SCHEMES}")


def fold_rng(seed: int, scheme: int) -> np.random.Generator:
    return np.random.default_rng(20261111 + 7919 * seed + 131 * scheme)


def make_fold(n: int, seed: int, scheme: int, n_draws: int = N_DRAWS):
    """Return (train_idx, test_idx) for one fold, both sorted arrays."""
    assert_not_reserved(n, seed, scheme)
    rng = fold_rng(seed, scheme)
    idx = rng.permutation(n_draws)
    if scheme == 0:
        tr, te = idx[:n], idx[n:]
    else:
        te = idx[:FIXED_TEST]
        tr = idx[FIXED_TEST:FIXED_TEST + n]
    tr = np.sort(tr)
    te = np.sort(te)
    assert len(tr) == n, (len(tr), n)
    assert len(te) >= 40, len(te)
    assert len(np.intersect1d(tr, te)) == 0
    return tr, te


def fold_fingerprint(tr: np.ndarray, te: np.ndarray) -> str:
    """8-hex-digit fingerprint of sorted(train) || sorted(test)."""
    payload = ",".join(str(int(v)) for v in np.sort(tr)) + "|" + \
              ",".join(str(int(v)) for v in np.sort(te))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def all_folds(n_draws: int = N_DRAWS) -> list[dict]:
    """Every (n, seed, scheme) fold with its fingerprint, in a stable order."""
    out = []
    for n in SIZES:
        for seed in SEEDS:
            for scheme in SCHEMES:
                if scheme == 1 and n > n_draws - FIXED_TEST:
                    continue
                tr, te = make_fold(n, seed, scheme, n_draws)
                out.append(dict(n=int(n), seed=int(seed), scheme=int(scheme),
                                n_train=int(len(tr)), n_test=int(len(te)),
                                train=tr, test=te,
                                fingerprint=fold_fingerprint(tr, te)))
    return out


def fold_key(n: int, seed: int, scheme: int) -> str:
    return f"n{n}_s{seed}_p{scheme}"
