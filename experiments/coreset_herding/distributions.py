"""
Key distributions for the coreset herding experiment.

Four regimes from the research plan (Section 1.2):
  (a) Isotropic Gaussian — no structure, worst-case-like
  (b) Clustered — 8 tight groups, synthetic coresets shine
  (c) Spherical — uniform on S^{d-1}
  (d) Real cache — pre-extracted from Llama-3.1-8B
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


def _normalize(K: np.ndarray) -> np.ndarray:
    """Center and project to the unit sphere."""
    K = K - K.mean(axis=0)
    norms = np.linalg.norm(K, axis=1, keepdims=True)
    return K / np.maximum(norms, 1e-10)


def gaussian_keys(
    n: int, d: int, rng: np.random.Generator
) -> np.ndarray:
    """Isotropic Gaussian, then normalized to unit sphere."""
    return _normalize(rng.standard_normal((n, d)))


def clustered_keys(
    n: int,
    d: int,
    rng: np.random.Generator,
    n_clusters: int = 8,
    cluster_std: float = 0.12,
) -> np.ndarray:
    """Keys in tight clusters around random unit-sphere centers."""
    centers = rng.standard_normal((n_clusters, d))
    centers /= np.linalg.norm(
        centers, axis=1, keepdims=True
    )
    per_c = n // n_clusters
    blocks = []
    for i in range(n_clusters):
        count = (
            per_c
            if i < n_clusters - 1
            else n - per_c * (n_clusters - 1)
        )
        noise = rng.standard_normal((count, d))
        blocks.append(centers[i] + noise * cluster_std)
    return _normalize(np.vstack(blocks))


def spherical_keys(
    n: int, d: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform on the unit sphere S^{d-1}."""
    return _normalize(rng.standard_normal((n, d)))


def load_real_keys(
    n: int,
    rng: np.random.Generator,
    data_dir: str = "data/notebook_data",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load pre-extracted Llama-3.1-8B keys and subsample.

    Returns (K, V, Q) or None if the file is missing.
    """
    path = Path(data_dir) / "code_run_p50_L8H24.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    K = npz["K"].astype(np.float64)
    V = npz["V"].astype(np.float64)
    Q = npz["Q"].astype(np.float64)
    idx = rng.choice(len(K), size=min(n, len(K)), replace=False)
    return _normalize(K[idx]), V[idx], Q[idx]


def build_distributions(
    n: int,
    d: int,
    rng: np.random.Generator,
    include_real: bool = True,
    data_dir: str = "data/notebook_data",
) -> Tuple[
    Dict[str, np.ndarray],
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Build all key distributions for the experiment.

    Returns:
        dists: dict mapping name -> K [n, d].
        real_data: (K, V, Q) tuple if real data is loaded,
            None otherwise. K here is already in dists.
    """
    dists: Dict[str, np.ndarray] = {
        "Gaussian": gaussian_keys(n, d, rng),
        "Clustered": clustered_keys(n, d, rng),
        "Spherical": spherical_keys(n, d, rng),
    }
    real_data = None
    if include_real:
        result = load_real_keys(n, rng, data_dir)
        if result is not None:
            K, V, Q = result
            dists["Real (Llama)"] = K
            real_data = (K, V, Q)
    return dists, real_data
