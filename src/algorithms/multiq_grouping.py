"""
MultiQ Grouping: KMeans on queries to find representative
centroids, sort keys by each centroid, route queries to
nearest centroid's ordering.

Offline cost: KMeans on Q + C matrix-vector products.
Per-query cost: O(C) centroid matching + O(G) group scoring.
"""

import numpy as np
from typing import List

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import (
    make_equal_groups, hybrid_attention, flat_kmeans,
)


class MultiQGrouping(AttentionAlgorithm):
    """
    KMeans on queries -> C centroid queries, each with
    its own key ordering. Per query: route to nearest
    centroid, use that ordering's equal-group split.
    """

    def __init__(
        self,
        n_query_clusters: int = 32,
        n_groups: int = 32,
        mode: str = "hybrid",
        top_k: int = 5,
    ):
        self.n_query_clusters = n_query_clusters
        self.n_groups = n_groups
        self.mode = mode
        self.top_k = top_k
        self._centroids = None
        self._sorted_orders = None

    @property
    def name(self) -> str:
        return (
            f"MultiQ-Q{self.n_query_clusters}"
            f"-G{self.n_groups}"
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
        """KMeans on queries, sort keys per centroid."""
        if queries is None:
            raise ValueError(
                "MultiQGrouping.prepare requires queries"
            )
        sqrt_d = np.sqrt(head_dim)

        centroids, _ = flat_kmeans(
            queries, self.n_query_clusters, seed=seed,
        )
        self._centroids = centroids.astype(np.float32)

        # Sort keys by each centroid query
        self._sorted_orders = []
        for i in range(self.n_query_clusters):
            scores = (
                keys @ self._centroids[i]
            ) / sqrt_d
            self._sorted_orders.append(
                np.argsort(scores)[::-1]
            )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._centroids is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query
        n_causal = len(problem.keys)
        special_set = problem.special_set
        special_idx = problem.special_idx

        # Route to nearest centroid
        best = int(np.argmax(self._centroids @ q))
        sorted_order = self._sorted_orders[best]

        valid = np.array([
            i for i in sorted_order
            if i < n_causal and i not in special_set
        ])

        groups = make_equal_groups(valid, self.n_groups)
        if not groups:
            return AttentionOutput(
                output=np.zeros(problem.head_dim),
                actual_budget=0,
            )

        output, eff_budget = hybrid_attention(
            q, problem.keys, problem.values,
            problem.logits, groups, self.top_k,
            problem.head_dim, special_idx, self.mode,
        )

        return AttentionOutput(
            output=output,
            actual_budget=eff_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        nqc_list = cfg.get(
            "n_query_clusters", [32]
        )
        if isinstance(nqc_list, int):
            nqc_list = [nqc_list]
        for nqc in nqc_list:
            for mode in cfg.get("modes", ["hybrid"]):
                for k in cfg.get("top_k_sweep", [5]):
                    instances.append(MultiQGrouping(
                        n_query_clusters=nqc,
                        n_groups=cfg.get(
                            "n_groups", 32
                        ),
                        mode=mode,
                        top_k=k,
                    ))
        return instances
