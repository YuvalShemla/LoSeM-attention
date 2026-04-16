"""
KMeans Nearest-Center: like KMeansClustering but uses the
actual key closest to the cluster centroid as the
representative (and its corresponding value), instead of
the cluster mean.

Representatives are recomputed per-query from the causal
window only, so no future information leaks.

Each cluster is one softmax item: score = q^T rep_key / √d
+ log(count), value = rep_value.

Offline cost: KMeans on N keys.
Per-query cost: O(N) centroid + nearest lookup per cluster.
"""

import numpy as np
from typing import List, Dict

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import (
    softmax, cached_flat_kmeans,
)


class KMeansNearestCenter(AttentionAlgorithm):
    """
    Flat KMeans on keys, represent each cluster by the
    key nearest to its causal centroid (and that key's
    value). Pure grouped attention — every cluster is a
    single softmax item, no top-k expansion.

    Representatives are recomputed at query time from
    the causal window to avoid future leakage.
    """

    def __init__(self, n_clusters: int = 256):
        self.n_clusters = n_clusters
        self._member_indices = None

    @property
    def name(self) -> str:
        return f"KMeansMedoid-{self.n_clusters}"

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        """Run KMeans, store only cluster assignments."""
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
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)
        sqrt_d = np.sqrt(head_dim)

        C = self.n_clusters
        # Build causal members per cluster
        valid = []
        causal_members = [None] * C
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
            if len(idx) > 0:
                causal_members[c] = idx
                valid.append(c)

        n_special = (
            len(special_idx)
            if special_idx is not None else 0
        )
        n_total = n_special + len(valid)
        if n_total == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        all_scores = np.empty(n_total)
        out_values = np.empty((n_total, head_dim))

        # Special tokens (exact)
        if n_special > 0:
            all_scores[:n_special] = (
                problem.logits[special_idx]
            )
            out_values[:n_special] = (
                values[special_idx]
            )

        # Per-cluster: centroid from causal keys,
        # then nearest causal key to that centroid.
        off = n_special
        for fi, c in enumerate(valid):
            idx = causal_members[c]
            count = len(idx)
            cluster_keys = keys[idx]
            centroid = np.mean(cluster_keys, axis=0)
            dists = np.sum(
                (cluster_keys - centroid) ** 2, axis=1,
            )
            nearest_local = np.argmin(dists)
            nearest_global = idx[nearest_local]

            rep_key = keys[nearest_global]
            rep_val = values[nearest_global]

            all_scores[off + fi] = (
                q @ rep_key / sqrt_d
                + np.log(count)
            )
            out_values[off + fi] = rep_val

        w = softmax(all_scores)
        output = w @ out_values

        return AttentionOutput(
            output=output.astype(np.float32),
            actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters_list = cfg.get(
            "n_clusters_sweep",
            [cfg.get("n_clusters", 256)],
        )
        return [
            KMeansNearestCenter(n_clusters=n_c)
            for n_c in clusters_list
        ]
