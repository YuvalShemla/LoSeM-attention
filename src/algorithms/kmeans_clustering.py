"""
KMeans Clustering: flat KMeans on keys, score clusters
per query, then hybrid/topk attention over sorted clusters.

Offline cost: KMeans on N keys.
Per-query cost: O(C) cluster scoring + O(C log C) sort.
"""

import numpy as np
from typing import List, Dict, Tuple

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import (
    softmax, hybrid_attention, flat_kmeans,
)


def precompute_cluster_stats(
    keys: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    C: int,
) -> Dict:
    """Per-cluster avg_keys, avg_values, counts, members."""
    d = keys.shape[1]
    d_v = values.shape[1]

    avg_keys = np.zeros((C, d), dtype=np.float32)
    avg_values = np.zeros((C, d_v), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int32)
    member_indices = [None] * C

    for c in range(C):
        mask = labels == c
        count = int(np.sum(mask))
        counts[c] = count
        if count > 0:
            avg_keys[c] = np.mean(keys[mask], axis=0)
            avg_values[c] = np.mean(
                values[mask], axis=0
            )
            member_indices[c] = np.where(mask)[0]
        else:
            member_indices[c] = np.array(
                [], dtype=np.int64
            )

    return {
        "avg_keys": avg_keys,
        "avg_values": avg_values,
        "counts": counts,
        "member_indices": member_indices,
    }


def _filter_cluster_members(
    cluster_stats: Dict,
    n_causal: int,
    special_set: set,
):
    """
    Filter members to causal window, remove special keys.

    Returns (causal_members, causal_counts, valid_mask).
    """
    members = cluster_stats["member_indices"]
    C = len(cluster_stats["counts"])

    causal_members = [None] * C
    causal_counts = np.zeros(C, dtype=np.int32)
    valid_mask = np.zeros(C, dtype=bool)

    for c in range(C):
        idx = members[c]
        if len(idx) == 0:
            causal_members[c] = np.array(
                [], dtype=np.int64
            )
            continue
        idx = idx[idx < n_causal]
        if special_set:
            keep = np.ones(len(idx), dtype=bool)
            for s in special_set:
                keep &= (idx != s)
            idx = idx[keep]
        causal_members[c] = idx
        causal_counts[c] = len(idx)
        if len(idx) > 0:
            valid_mask[c] = True

    return causal_members, causal_counts, valid_mask


class KMeansClustering(AttentionAlgorithm):
    """
    Flat KMeans on keys, score clusters per query,
    hybrid/topk attention over sorted clusters.
    """

    def __init__(
        self,
        n_clusters: int = 256,
        mode: str = "hybrid",
        top_k: int = 5,
    ):
        self.n_clusters = n_clusters
        self.mode = mode
        self.top_k = top_k
        self._cluster_stats = None

    @property
    def name(self) -> str:
        return (
            f"KMeans-C{self.n_clusters}"
            f"-{self.mode}-k{self.top_k}"
        )

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        """Run KMeans on keys and precompute stats."""
        _, labels = flat_kmeans(
            keys, self.n_clusters, seed=seed,
        )
        self._cluster_stats = precompute_cluster_stats(
            keys, values, labels, self.n_clusters,
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._cluster_stats is None:
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
        sqrt_d = np.sqrt(head_dim)

        cs = self._cluster_stats
        avg_keys = cs["avg_keys"]

        cm, cc, vm = _filter_cluster_members(
            cs, n_causal, special_set,
        )

        valid_clusters = np.where(vm)[0]
        if len(valid_clusters) == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        # Score each valid cluster
        scores = np.array([
            float(
                q @ avg_keys[c] / sqrt_d
                + np.log(cc[c])
            )
            for c in valid_clusters
        ])

        # Sort clusters by score descending
        order = valid_clusters[
            np.argsort(scores)[::-1]
        ]
        groups = [cm[c] for c in order]

        output, eff_budget = hybrid_attention(
            q, keys, values, logits, groups,
            self.top_k, head_dim, special_idx,
            self.mode,
        )

        return AttentionOutput(
            output=output,
            actual_budget=eff_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        for mode in cfg.get("modes", ["hybrid"]):
            for k in cfg.get("top_k_sweep", [5]):
                instances.append(KMeansClustering(
                    n_clusters=cfg.get(
                        "n_clusters", 256
                    ),
                    mode=mode,
                    top_k=k,
                ))
        return instances
