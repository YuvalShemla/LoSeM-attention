"""
Exponential kernel Gram matrix and effective dimension.

The Gram matrix G_ij = exp(tau * k_i^T k_j) is the central
object. The temperature tau controls how much the kernel
amplifies inner-product differences. In attention,
tau = 1/sqrt(d) is the standard scaling.

All herding computations go through G via the kernel trick,
avoiding the infinite-dimensional feature map psi.

The effective dimension d_eff = sum_j mu_j / (mu_j + lambda)
is the complexity measure that should predict coreset size
(Conjecture 5, Question 6 in the research plan).
"""

import numpy as np
from typing import Tuple


def compute_gram(
    K: np.ndarray, tau: float = 1.0
) -> np.ndarray:
    """Exponential kernel Gram matrix.

    G_ij = exp(tau * k_i^T k_j)

    tau controls the kernel's sensitivity to inner-product
    differences. Larger tau makes the kernel more peaked
    and increases d_eff, especially for structured data.

    Args:
        K: [n, d] key matrix.
        tau: temperature (default 1.0).

    Returns:
        G: [n, n] positive-definite Gram matrix.
    """
    return np.exp(tau * (K @ K.T))


def effective_dimension(
    G: np.ndarray, lam: float = None
) -> Tuple[float, np.ndarray]:
    """Ridge-leverage count of the kernel operator.

    d_eff(lambda) = sum_j mu_j / (mu_j + lambda)

    where mu_j are eigenvalues of G. This is Eqn (5).

    If lam is None, it is set to the mean eigenvalue
    (trace(G)/n), making d_eff scale-invariant and
    sensitive to the spectral shape rather than the
    overall magnitude.

    Returns:
        d_eff: scalar effective dimension.
        eigs: sorted eigenvalues of G (ascending).
    """
    n = G.shape[0]
    eigs = np.linalg.eigvalsh(G)
    eigs = np.maximum(eigs, 0.0)
    if lam is None:
        lam = float(np.sum(eigs)) / n
    d_eff = float(np.sum(eigs / (eigs + lam)))
    return d_eff, eigs


def residual_norm_sq(
    K: np.ndarray,
    G: np.ndarray,
    atom_keys: np.ndarray,
    atom_weights: np.ndarray,
    tau: float = 1.0,
) -> float:
    """Squared residual ||sigma/n - sum w_j psi(k'_j)||^2.

    Computed entirely through kernel evaluations:
      term1 = (1/n^2) sum_{i,l} G[i,l]
      term2 = -(2/n) sum_i sum_j w_j exp(tau * k_i^T k'_j)
      term3 = sum_{j,l} w_j w_l exp(tau * k'_j^T k'_l)

    Args:
        K: [n, d] original keys.
        G: [n, n] precomputed Gram matrix of K.
        atom_keys: [T, d] coreset atom positions.
        atom_weights: [T] atom weights.
        tau: temperature (must match G).

    Returns:
        Squared residual norm (non-negative).
    """
    n = K.shape[0]
    w = np.asarray(atom_weights)
    aK = np.asarray(atom_keys)

    term1 = np.sum(G) / (n * n)
    cross = np.exp(tau * (K @ aK.T))
    term2 = -2.0 / n * np.sum(cross @ w)
    atom_G = np.exp(tau * (aK @ aK.T))
    term3 = float(w @ atom_G @ w)

    return max(0.0, term1 + term2 + term3)
