"""
Frank-Wolfe / kernel herding for attention coresets.

Two LMO strategies:

Subset LMO (Section 1.2, cheap baseline):
  Scan existing keys {k_i} and pick the one most aligned
  with the residual in feature space.

Synthetic LMO (Section 1.2, gradient-ascent version):
  Maximize <r, psi(k)> over ||k|| <= 1 via projected
  gradient ascent from centroid-seeded restarts.

The LMO score for candidate key k is:

  <r, psi(k)> = (1/n) sum_i exp(tau * k_i^T k)
              - sum_j w_j exp(tau * k'_j^T k)

where tau is the kernel temperature.
"""

import numpy as np
from typing import List, Tuple
from scipy.cluster.vq import kmeans2

from .gram import residual_norm_sq


def _lmo_scores(
    K: np.ndarray,
    G: np.ndarray,
    atom_keys: List[np.ndarray],
    atom_weights: List[float],
    tau: float,
) -> np.ndarray:
    """LMO scores for all existing keys.

    score(l) = (1/n) sum_i G[i,l] - sum_j w_j G'[j,l]
    """
    n = K.shape[0]
    target = G.sum(axis=0) / n
    if len(atom_keys) == 0:
        return target
    w = np.array(atom_weights)
    aK = np.array(atom_keys)
    cross = np.exp(tau * (aK @ K.T))
    return target - w @ cross


def _lmo_gradient(
    K: np.ndarray,
    atom_keys: List[np.ndarray],
    atom_weights: List[float],
    k: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Gradient of <r, psi(k)> w.r.t. k.

    grad = tau * [(1/n) sum_i exp(tau k_i^T k) k_i
                  - sum_j w_j exp(tau k'_j^T k) k'_j]
    """
    n = K.shape[0]
    dots_K = np.clip(tau * (K @ k), -50, 50)
    exp_K = np.exp(dots_K)
    grad = tau * (exp_K / n) @ K

    if len(atom_keys) > 0:
        w = np.array(atom_weights)
        aK = np.array(atom_keys)
        dots_a = np.clip(tau * (aK @ k), -50, 50)
        exp_a = np.exp(dots_a)
        grad -= tau * ((w * exp_a) @ aK)

    return grad


def _lmo_value(
    K: np.ndarray,
    atom_keys: List[np.ndarray],
    atom_weights: List[float],
    k: np.ndarray,
    tau: float,
) -> float:
    """Objective <r, psi(k)> at candidate k."""
    n = K.shape[0]
    dots = np.clip(tau * (K @ k), -50, 50)
    val = np.sum(np.exp(dots)) / n
    if len(atom_keys) > 0:
        w = np.array(atom_weights)
        aK = np.array(atom_keys)
        dots_a = np.clip(tau * (aK @ k), -50, 50)
        val -= float(w @ np.exp(dots_a))
    return float(val)


def _herding_update(
    atom_keys: List[np.ndarray],
    atom_weights: List[float],
    new_key: np.ndarray,
    t: int,
) -> None:
    """Herding update: gamma_t = 1/(t+1), equal weights.

    After T atoms, every atom has weight 1/T. This is the
    standard kernel herding schedule (Chen, Welling, Smola
    2010) which gives O(1/sqrt(T)) in RKHS norm.
    """
    gamma = 1.0 / (t + 1)
    for i in range(len(atom_weights)):
        atom_weights[i] *= (1.0 - gamma)
    atom_weights.append(gamma)
    atom_keys.append(new_key.copy())


def _record_residual(
    K, G, atom_keys, atom_weights, residuals, tau
):
    r_sq = residual_norm_sq(
        K, G,
        np.array(atom_keys),
        np.array(atom_weights),
        tau,
    )
    residuals.append(r_sq ** 0.5)


def herding_subset(
    K: np.ndarray,
    G: np.ndarray,
    max_atoms: int,
    tau: float = 1.0,
) -> Tuple[List[float], List[int]]:
    """Subset herding: best existing key per step.

    Returns:
        (residuals, selected_indices)
    """
    atom_keys: List[np.ndarray] = []
    atom_weights: List[float] = []
    residuals: List[float] = []
    indices: List[int] = []

    for t in range(max_atoms):
        scores = _lmo_scores(
            K, G, atom_keys, atom_weights, tau
        )
        best = int(np.argmax(scores))
        indices.append(best)
        _herding_update(atom_keys, atom_weights, K[best], t)
        _record_residual(
            K, G, atom_keys, atom_weights, residuals, tau
        )

    return residuals, indices


def herding_synthetic(
    K: np.ndarray,
    G: np.ndarray,
    max_atoms: int,
    tau: float = 1.0,
    n_restarts: int = 8,
    n_steps: int = 60,
    lr: float = 0.3,
    rng: np.random.Generator = None,
) -> List[float]:
    """Synthetic herding: gradient ascent LMO on ||k||<=1.

    Restarts seeded at k-means centroids plus the best
    existing key (plan recipe iii + warm start).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n, dim = K.shape
    atom_keys: List[np.ndarray] = []
    atom_weights: List[float] = []
    residuals: List[float] = []

    n_clust = min(n_restarts - 1, n)
    centroids, _ = kmeans2(
        K.astype(np.float32), n_clust, minit="points"
    )
    centroids = centroids.astype(np.float64)

    for t in range(max_atoms):
        # Collect starting points: centroids + best subset key
        starts = list(centroids.copy())
        sub_scores = _lmo_scores(
            K, G, atom_keys, atom_weights, tau
        )
        starts.append(K[np.argmax(sub_scores)].copy())

        best_k = None
        best_val = -np.inf

        for start in starts:
            k = start.copy()
            nk = np.linalg.norm(k)
            if nk > 1.0:
                k /= nk

            for _ in range(n_steps):
                grad = _lmo_gradient(
                    K, atom_keys, atom_weights, k, tau
                )
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-12:
                    break
                k += lr * grad / gnorm
                nk = np.linalg.norm(k)
                if nk > 1.0:
                    k /= nk

            val = _lmo_value(
                K, atom_keys, atom_weights, k, tau
            )
            if val > best_val:
                best_val = val
                best_k = k.copy()

        _herding_update(atom_keys, atom_weights, best_k, t)
        _record_residual(
            K, G, atom_keys, atom_weights, residuals, tau
        )

    return residuals
