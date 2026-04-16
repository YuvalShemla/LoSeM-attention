"""
LSH Cross-Polytope Hybrid: reduced-dim CP hashing with
hybrid attention over buckets.

For the top-K most relevant buckets, all individual keys
are enumerated.  Remaining buckets are represented by their
count-corrected means.  This mirrors the KMeans hybrid
approach but uses CP-LSH hashing instead of KMeans.

Budget is controlled by top_k (number of individually
enumerated buckets), while bucket granularity is fixed
by k_dim.
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .lsh_crosspoly_multiprobe import (
    _random_orthogonal, crosspolytope_bucket_labels,
)
from .kmeans_clustering import (
    precompute_cluster_stats, _filter_cluster_members,
)
from ..core import hybrid_attention


class LSHCrossPolytopeHybrid(AttentionAlgorithm):
    """
    Hybrid attention with CP-LSH buckets.

    prepare() hashes all keys into (2k)^2+1 buckets via
    two reduced-dim CP hashes.  run() ranks buckets by
    query-mean score, enumerates individual keys for the
    top_k buckets, and uses count-corrected means for the
    rest.
    """

    def __init__(self, k_dim: int = 11, top_k: int = 0):
        self._k = k_dim
        self._top_k = top_k
        self._cluster_stats = None

    # ── naming ──────────────────────────────────────────

    @property
    def name(self) -> str:
        n_buckets = (2 * self._k) ** 2 + 1
        return (
            f"LSH-CrossPoly-Hybrid"
            f"-C{n_buckets}-hybrid-k{self._top_k}"
        )

    @property
    def sweeps_budget(self) -> bool:
        return False

    # ── offline ─────────────────────────────────────────

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: Optional[np.ndarray] = None,
        query_positions: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        d = head_dim
        k = self._k
        n_cp = 2 * k
        n_buckets = n_cp * n_cp + 1
        rng = np.random.default_rng(seed)

        if len(keys) == 0:
            self._cluster_stats = precompute_cluster_stats(
                keys, values,
                np.zeros(0, dtype=np.int32), n_buckets,
            )
            return

        # Center keys for hashing only
        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        x_c = keys.astype(np.float32) - key_mean
        x64 = x_c.astype(np.float64)

        R1 = _random_orthogonal(d, rng)
        R2 = _random_orthogonal(d, rng)
        R1_k = R1[:k].astype(np.float64)
        R2_k = R2[:k].astype(np.float64)

        z1 = (x64 @ R1_k.T).astype(np.float32)
        z2 = (x64 @ R2_k.T).astype(np.float32)

        b1 = crosspolytope_bucket_labels(z1)
        b2 = crosspolytope_bucket_labels(z2)
        labels = (b1 * n_cp + b2).astype(np.int32)

        self._cluster_stats = precompute_cluster_stats(
            keys, values, labels, n_buckets,
        )

    # ── online ──────────────────────────────────────────

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

        # Score each valid bucket by mean-key dot + log(count)
        scores = np.array([
            float(
                q @ avg_keys[c] / sqrt_d
                + np.log(cc[c])
            )
            for c in valid_clusters
        ])

        # Sort buckets by score descending
        order = valid_clusters[
            np.argsort(scores)[::-1]
        ]
        groups = [cm[c] for c in order]

        output, eff_budget = hybrid_attention(
            q, keys, values, logits, groups,
            self._top_k, head_dim, special_idx,
            "hybrid",
        )

        return AttentionOutput(
            output=output,
            actual_budget=eff_budget,
        )

    # ── config expansion ────────────────────────────────

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        k_dim = cfg.get("k_dim", 11)
        top_k_sweep = cfg.get(
            "top_k_sweep", [0, 1, 2, 3, 5, 7, 10, 16],
        )
        return [
            LSHCrossPolytopeHybrid(
                k_dim=k_dim, top_k=k,
            )
            for k in top_k_sweep
        ]
