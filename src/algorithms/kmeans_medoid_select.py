"""
KMeans Medoid Select: pick the medoid (nearest key to
causal centroid) from each cluster, then run a plain
subset softmax over those selected keys — no count
weighting.

Each cluster contributes exactly one real key/value pair
to the softmax, scored by its actual logit q^T k / √d.

Offline cost: KMeans on N keys.
Per-query cost: O(N) centroid + nearest lookup per cluster.
"""

import numpy as np
from typing import List

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, cached_flat_kmeans


class KMeansMedoidSelect(AttentionAlgorithm):
    """
    Flat KMeans on keys, select the medoid from each
    cluster's causal window, then plain subset softmax
    over special tokens + all medoids (no count weight).
    """

    def __init__(self, n_clusters: int = 256):
        self.n_clusters = n_clusters
        self._member_indices = None

    @property
    def name(self) -> str:
        return f"KMeansMedoidSelect-{self.n_clusters}"

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        _, labels = cached_flat_kmeans(
            keys, self.n_clusters, seed=seed,
        )
        C = self.n_clusters
        self._member_indices = [None] * C
        for c in range(C):
            mask = labels == c
            if np.any(mask):
                self._member_indices[c] = (
                    np.where(mask)[0]
                )
            else:
                self._member_indices[c] = np.array(
                    [], dtype=np.int64,
                )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._member_indices is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)

        C = self.n_clusters
        # Find medoid per cluster from causal window
        selected = []
        for c in range(C):
            idx = self._member_indices[c]
            if len(idx) == 0:
                continue
            idx = idx[idx < n_causal]
            if special_set:
                keep = np.ones(len(idx), dtype=bool)
                for s in special_set:
                    keep &= (idx != s)
                idx = idx[keep]
            if len(idx) == 0:
                continue
            cluster_keys = keys[idx]
            centroid = np.mean(cluster_keys, axis=0)
            dists = np.sum(
                (cluster_keys - centroid) ** 2, axis=1,
            )
            nearest_local = np.argmin(dists)
            selected.append(idx[nearest_local])

        # Build subset: special tokens + medoids
        if special_idx is not None and len(special_idx):
            all_idx = np.concatenate([
                special_idx,
                np.array(selected, dtype=np.int64),
            ])
        else:
            all_idx = np.array(selected, dtype=np.int64)

        if len(all_idx) == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        # Plain subset softmax — no count correction
        sub_logits = logits[all_idx]
        w = softmax(sub_logits)
        output = w @ values[all_idx]

        return AttentionOutput(
            output=output.astype(np.float32),
            actual_budget=len(all_idx),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters_list = cfg.get(
            "n_clusters_sweep",
            [cfg.get("n_clusters", 256)],
        )
        return [
            KMeansMedoidSelect(n_clusters=n_c)
            for n_c in clusters_list
        ]
