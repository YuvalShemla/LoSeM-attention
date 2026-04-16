"""
KMeans-KK: cluster into K groups, then take K individual keys
from the closest cluster(s), reconstructing split clusters.

Budget = 2K + special:
  - K cluster representatives (mean key / mean value, count-weighted)
  - K individual keys from the nearest cluster(s) to the query

If the closest cluster has fewer than K members, spill to the
next closest, and so on. When a cluster is partially drained
(some keys taken as individuals), its representative is
reconstructed from the remaining keys only.
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, cached_flat_kmeans


class KMeansKK(AttentionAlgorithm):
    """
    KMeans with K clusters and K exact keys from closest clusters.

    For each value of K:
      - Offline: cluster all keys into K groups
      - Per-query: score clusters, greedily take K individual
        keys from the closest cluster(s), reconstruct any
        partially-drained cluster reps from remaining members
      - Joint softmax over special + K individuals + cluster reps
      - Budget = K individuals + cluster reps + special
    """

    def __init__(self, K: int = 256):
        self._K = K
        self._member_indices = None

    @property
    def name(self) -> str:
        return f"KMeans-KK-{self._K}"

    @property
    def sweeps_budget(self) -> bool:
        return False

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: Optional[np.ndarray] = None,
        query_positions: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        C = min(self._K, len(keys))
        _, labels = cached_flat_kmeans(
            keys, C, seed=seed,
        )
        self._member_indices = [None] * C
        for c in range(C):
            mask = labels == c
            if np.any(mask):
                self._member_indices[c] = np.where(mask)[0]
            else:
                self._member_indices[c] = np.array(
                    [], dtype=np.int64,
                )
        self._n_clusters_actual = C

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._member_indices is None:
            raise RuntimeError("Call prepare() before run()")

        q = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)
        sqrt_d = np.sqrt(head_dim)
        C = self._n_clusters_actual

        # Filter clusters to causal window, exclude special keys
        causal_members = [None] * C
        cluster_scores = np.full(C, -np.inf)
        for c in range(C):
            idx = self._member_indices[c]
            if len(idx) == 0:
                causal_members[c] = np.array(
                    [], dtype=np.int64,
                )
                continue
            idx = idx[idx < n_causal]
            if special_set:
                keep = np.ones(len(idx), dtype=bool)
                for s in special_set:
                    keep &= (idx != s)
                idx = idx[keep]
            causal_members[c] = idx
            if len(idx) > 0:
                mean_key = np.mean(keys[idx], axis=0)
                cluster_scores[c] = float(
                    q @ mean_key / sqrt_d
                )

        valid_mask = cluster_scores > -np.inf
        if not np.any(valid_mask):
            from ..core import subset_attention
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        # Sort clusters by score (descending = closest first)
        sorted_clusters = np.argsort(cluster_scores)[::-1]
        sorted_clusters = sorted_clusters[
            cluster_scores[sorted_clusters] > -np.inf
        ]

        # Greedily collect K individual keys from closest clusters
        n_exact_target = self._K
        remaining_need = n_exact_target
        exact_list = []
        remaining_members = {}  # cluster -> remaining members

        for c in sorted_clusters:
            members = causal_members[c]
            n_m = len(members)
            if remaining_need <= 0:
                remaining_members[c] = members
                continue

            if n_m <= remaining_need:
                # Take all members as individuals
                exact_list.append(members)
                remaining_need -= n_m
                # This cluster is fully drained, no rep needed
            else:
                # Take remaining_need members, keep rest for rep
                chosen = rng.choice(
                    n_m, size=remaining_need, replace=False,
                )
                exact_list.append(members[chosen])
                mask = np.ones(n_m, dtype=bool)
                mask[chosen] = False
                remaining_members[c] = members[mask]
                remaining_need = 0

        if exact_list:
            exact_idx = np.concatenate(exact_list)
        else:
            exact_idx = np.array([], dtype=np.int64)

        # Build cluster reps from remaining members
        rep_clusters = [
            c for c in sorted_clusters
            if c in remaining_members
            and len(remaining_members[c]) > 0
        ]

        # Assemble final softmax
        n_special = len(special_idx)
        n_exact = len(exact_idx)
        n_rep = len(rep_clusters)
        n_total = n_special + n_exact + n_rep

        if n_total == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        scores_arr = np.empty(n_total)
        out_values = np.empty((n_total, head_dim))

        # Special keys (exact)
        off = 0
        if n_special > 0:
            scores_arr[:n_special] = logits[special_idx]
            out_values[:n_special] = values[special_idx]
            off = n_special

        # Individual keys (exact)
        if n_exact > 0:
            scores_arr[off:off + n_exact] = logits[exact_idx]
            out_values[off:off + n_exact] = values[exact_idx]
            off += n_exact

        # Cluster reps (count-weighted mean key/value)
        for fi, c in enumerate(rep_clusters):
            idx = remaining_members[c]
            count = len(idx)
            avg_key = np.mean(keys[idx], axis=0)
            avg_val = np.mean(values[idx], axis=0)
            scores_arr[off + fi] = (
                q @ avg_key / sqrt_d + np.log(count)
            )
            out_values[off + fi] = avg_val

        w = softmax(scores_arr)
        output = w @ out_values

        return AttentionOutput(
            output=output.astype(np.float32),
            actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        K_values = cfg.get("K_sweep", [
            16, 32, 64, 128, 256, 512, 1024, 2048,
        ])
        return [KMeansKK(K=k) for k in K_values]
