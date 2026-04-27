"""
KMeans Value/Key Clustering — two complementary methods:

KMeansValue: Cluster V, replace each V[i] with cluster mean.
  All individual keys used for scoring. Tests value
  space compressibility.

KMeansKey: Cluster K, replace each K[i] with cluster mean.
  All individual values used for output. Tests key
  space compressibility.

Both keep special tokens (sink + local window) exact.
Plotted as horizontal dashed lines across the budget axis.
"""

import numpy as np
from typing import Dict, List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, cached_flat_kmeans


class KMeansValueClustering(AttentionAlgorithm):
    """
    Cluster V, quantize values, keep all keys exact.
    Special keys keep exact values.
    """

    def __init__(self, n_clusters: int = 256):
        self.n_clusters = n_clusters
        self._seed = 42

    @property
    def name(self) -> str:
        return f"KMeansValue-{self.n_clusters}"

    @property
    def kind(self) -> str:
        return "algorithm"

    @property
    def sweeps_budget(self) -> bool:
        return False

    @property
    def point_label(self) -> str:
        return str(self.n_clusters)

    @property
    def horizontal_line(self) -> bool:
        return True

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        self._seed = seed

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        q = problem.query
        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        candidate_idx = problem.candidate_idx
        sqrt_d = np.sqrt(head_dim)

        n_clusters = max(
            1, min(self.n_clusters, len(candidate_idx)),
        )

        # K-Means on candidate values only
        cand_values = values[candidate_idx]
        _, labels = cached_flat_kmeans(
            cand_values, n_clusters, seed=self._seed,
        )

        # Mean value per cluster
        d_v = values.shape[1]
        avg_vals = np.zeros(
            (n_clusters, d_v), dtype=np.float32,
        )
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                avg_vals[c] = np.mean(
                    cand_values[mask], axis=0,
                )

        # Exact special + quantized candidates
        v_mixed = values.copy()
        v_mixed[candidate_idx] = avg_vals[labels]

        # Full attention with all original keys
        logits = q @ keys.T / sqrt_d
        w = softmax(logits)
        output = w @ v_mixed

        return AttentionOutput(
            output=output,
            actual_budget=n_clusters,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters_list = cfg.get(
            "n_clusters_sweep",
            [128, 512, 1024, 2048, 4096],
        )
        return [
            KMeansValueClustering(n_clusters=n_c)
            for n_c in clusters_list
        ]


class KMeansKeyClustering(AttentionAlgorithm):
    """
    Cluster K, quantize keys, keep all values exact.
    Special keys keep exact key vectors.

    logits[i] = Q @ K_quantized[i] / sqrt(d)
    weights = softmax(logits)
    output = weights @ V  (actual values)
    """

    def __init__(self, n_clusters: int = 256):
        self.n_clusters = n_clusters
        self._seed = 42

    @property
    def name(self) -> str:
        return f"KMeansKey-{self.n_clusters}"

    @property
    def kind(self) -> str:
        return "algorithm"

    @property
    def sweeps_budget(self) -> bool:
        return False

    @property
    def point_label(self) -> str:
        return str(self.n_clusters)

    @property
    def horizontal_line(self) -> bool:
        return True

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        self._seed = seed

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        q = problem.query
        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        candidate_idx = problem.candidate_idx
        sqrt_d = np.sqrt(head_dim)

        n_clusters = max(
            1, min(self.n_clusters, len(candidate_idx)),
        )

        # K-Means on candidate keys only
        cand_keys = keys[candidate_idx]
        _, labels = cached_flat_kmeans(
            cand_keys, n_clusters, seed=self._seed,
        )

        # Mean key per cluster
        d = keys.shape[1]
        avg_keys = np.zeros(
            (n_clusters, d), dtype=np.float32,
        )
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                avg_keys[c] = np.mean(
                    cand_keys[mask], axis=0,
                )

        # Exact special + quantized candidates
        k_mixed = keys.copy()
        k_mixed[candidate_idx] = avg_keys[labels]

        # Attention with quantized keys, actual values
        logits = q @ k_mixed.T / sqrt_d
        w = softmax(logits)
        output = w @ values

        return AttentionOutput(
            output=output,
            actual_budget=n_clusters,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters_list = cfg.get(
            "n_clusters_sweep",
            [128, 512, 1024, 2048, 4096],
        )
        return [
            KMeansKeyClustering(n_clusters=n_c)
            for n_c in clusters_list
        ]
