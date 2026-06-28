"""
Baseline coreset methods: uniform and leverage-score sampling.

Non-greedy alternatives to herding. They select all atoms at
once (not sequentially), so they lack the adaptive error
correction of Frank-Wolfe.
"""

import numpy as np
from typing import List

from .gram import residual_norm_sq


def uniform_sampling(
    K: np.ndarray,
    G: np.ndarray,
    max_atoms: int,
    rng: np.random.Generator,
    tau: float = 1.0,
) -> List[float]:
    """Pick T keys uniformly at random with equal weights.

    Expected convergence: O(1/sqrt(T)) by CLT.
    """
    n = K.shape[0]
    residuals: List[float] = []
    for t in range(1, max_atoms + 1):
        idx = rng.choice(n, size=t, replace=False)
        w = np.ones(t) / t
        r_sq = residual_norm_sq(K, G, K[idx], w, tau)
        residuals.append(r_sq ** 0.5)
    return residuals


def leverage_sampling(
    K: np.ndarray,
    G: np.ndarray,
    max_atoms: int,
    rng: np.random.Generator,
    tau: float = 1.0,
) -> List[float]:
    """Sample proportional to kernel diagonal G_ii.

    G_ii = exp(tau * ||k_i||^2) is a cheap proxy for
    leverage scores. Nystrom / WildCat-style selection.
    """
    n = K.shape[0]
    diag = np.diag(G)
    probs = diag / diag.sum()
    residuals: List[float] = []
    for t in range(1, max_atoms + 1):
        idx = rng.choice(n, size=t, replace=False, p=probs)
        w = np.ones(t) / t
        r_sq = residual_norm_sq(K, G, K[idx], w, tau)
        residuals.append(r_sq ** 0.5)
    return residuals
