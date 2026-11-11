"""Controls that decide what a margin would MEAN, run at n = 80 on the screen folds.

Two controls, both requested by the original ask and both independent of the candidate's
own tuning:

SHUFFLED-PHASE-LABEL CONTROL. The base paper's control, and the one the original request
names explicitly. The phase labels attached to the 55 loads are permuted (a fixed
derangement-like permutation, recorded) and the whole spectral construction is rebuilt
with the fake labels: same Laplacian, same eigenpairs, same k, same learner, same folds.
Whatever accuracy survives is accuracy that does NOT depend on knowing which phase each
load sits on. If the candidate scores the same with shuffled labels, the phase labels are
doing no work and the "phase-aware" claim is unsupported.

DOSE-MATCHED GENERIC-COMPRESSION CONTROL. b6: a GP on the top-11 principal components of
the standardised injections — the same number of coordinates the spectral embedder
produces, chosen without any network information. This separates "the Laplacian selects
informative directions" from "any 22-dimensional projection helps".
"""
from __future__ import annotations

import numpy as np

PERM_SEED = 20261111


def load_permutation(n: int = 55, seed: int = PERM_SEED) -> np.ndarray:
    """A fixed, recorded permutation of the load->phase labels (no fixed points)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p.astype(int)


def shuffled_phase_labels(phase: np.ndarray, seed: int = PERM_SEED) -> np.ndarray:
    """Permute which phase each load's vector is gathered at."""
    return phase[load_permutation(len(phase), seed)]
