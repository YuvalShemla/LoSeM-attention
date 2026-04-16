"""
KMeans Hybrid TopK: score clusters by query affinity,
then either:

  hybrid — take n_exact actual keys from closest clusters
           + centroid reps for remaining clusters
  topk   — take the same total budget as hybrid, but all
           as exact keys from closest clusters (no reps)

Both modes share the same KMeans clustering (cached).

Offline cost: KMeans on N keys.
Per-query cost: O(C) cluster scoring + O(n_exact) selection.
"""

import numpy as np
from typing import List

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, cached_flat_kmeans, subset_attention


class KMeansHybridTopK(AttentionAlgorithm):
    """
    KMeans clustering + exact top keys from closest
    clusters.

    mode="hybrid": n_exact exact + centroid reps for rest.
    mode="topk":   same total budget, all exact keys.
    """

    def __init__(
        self,
        n_clusters: int = 256,
        n_exact: int = 1024,
        mode: str = "hybrid",
    ):
        self.n_clusters = n_clusters
        self.n_exact = n_exact
        self.mode = mode
        self._member_indices = None

    @property
    def name(self) -> str:
        if self.mode == "hybrid":
            return (
                f"Hybrid{self.n_exact}"
                f"-{self.n_clusters}"
            )
        return (
            f"TopK{self.n_exact}"
            f"-{self.n_clusters}"
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

    def _filter_and_score(self, problem):
        """Filter clusters to causal window, score them."""
        q = problem.query
        keys = problem.keys
        special_set = problem.special_set
        n_causal = len(keys)
        sqrt_d = np.sqrt(problem.head_dim)
        C = self.n_clusters

        causal_members = [None] * C
        valid = []
        scores = []
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
                valid.append(c)
                mean_key = np.mean(keys[idx], axis=0)
                scores.append(
                    float(q @ mean_key / sqrt_d)
                )

        if len(valid) == 0:
            return causal_members, np.array([]), np.array([])

        valid = np.array(valid)
        scores = np.array(scores)
        order = np.argsort(scores)[::-1]
        return causal_members, valid[order], scores[order]

    def _count_remaining_reps(
        self, sorted_clusters, causal_members,
    ):
        """
        Count how many cluster reps hybrid mode would
        produce (deterministic, no rng needed).
        """
        acc = 0
        n_reps = 0
        for c in sorted_clusters:
            n_m = len(causal_members[c])
            if acc >= self.n_exact:
                n_reps += 1
            elif acc + n_m <= self.n_exact:
                acc += n_m
            else:
                remainder = n_m - (self.n_exact - acc)
                if remainder > 0:
                    n_reps += 1
                acc = self.n_exact
        return n_reps

    def _greedy_collect(
        self, sorted_clusters, causal_members,
        total_exact, rng,
    ):
        """
        Greedily collect total_exact keys from sorted
        clusters. Returns (exact_idx, remaining_members).
        """
        remaining_need = total_exact
        exact_list = []
        remaining = {}

        for c in sorted_clusters:
            members = causal_members[c]
            n_m = len(members)

            if remaining_need <= 0:
                remaining[c] = members
                continue

            if n_m <= remaining_need:
                exact_list.append(members)
                remaining_need -= n_m
            else:
                chosen = rng.choice(
                    n_m, size=remaining_need,
                    replace=False,
                )
                exact_list.append(members[chosen])
                mask = np.ones(n_m, dtype=bool)
                mask[chosen] = False
                remaining[c] = members[mask]
                remaining_need = 0

        if exact_list:
            exact_idx = np.concatenate(exact_list)
        else:
            exact_idx = np.array([], dtype=np.int64)
        return exact_idx, remaining

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
        sqrt_d = np.sqrt(head_dim)

        causal_members, sorted_clusters, _ = (
            self._filter_and_score(problem)
        )
        if len(sorted_clusters) == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        n_reps = self._count_remaining_reps(
            sorted_clusters, causal_members,
        )

        if self.mode == "topk":
            # Match hybrid budget: all exact keys
            total_exact = self.n_exact + n_reps
            exact_idx, _ = self._greedy_collect(
                sorted_clusters, causal_members,
                total_exact, rng,
            )
            n_special = (
                len(special_idx)
                if special_idx is not None else 0
            )
            if n_special > 0:
                all_idx = np.concatenate([
                    special_idx, exact_idx,
                ]).astype(np.int64)
            else:
                all_idx = exact_idx.astype(np.int64)

            if len(all_idx) == 0:
                return AttentionOutput(
                    output=np.zeros(head_dim),
                    actual_budget=0,
                )
            output = subset_attention(
                logits, values, all_idx,
            )
            return AttentionOutput(
                output=output.astype(np.float32),
                actual_budget=len(all_idx),
            )

        # ── Hybrid mode: n_exact exact + centroid reps ──
        exact_idx, remaining = self._greedy_collect(
            sorted_clusters, causal_members,
            self.n_exact, rng,
        )
        rep_clusters = [
            c for c in sorted_clusters
            if c in remaining
            and len(remaining[c]) > 0
        ]

        n_special = (
            len(special_idx)
            if special_idx is not None else 0
        )
        n_exact_actual = len(exact_idx)
        n_rep = len(rep_clusters)
        n_total = n_special + n_exact_actual + n_rep

        if n_total == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        scores_arr = np.empty(n_total)
        out_values = np.empty((n_total, head_dim))

        off = 0
        if n_special > 0:
            scores_arr[:n_special] = logits[special_idx]
            out_values[:n_special] = values[special_idx]
            off = n_special

        if n_exact_actual > 0:
            scores_arr[off:off + n_exact_actual] = (
                logits[exact_idx]
            )
            out_values[off:off + n_exact_actual] = (
                values[exact_idx]
            )
            off += n_exact_actual

        for fi, c in enumerate(rep_clusters):
            idx = remaining[c]
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
        clusters_list = cfg.get(
            "n_clusters_sweep",
            [cfg.get("n_clusters", 256)],
        )
        n_exact = cfg.get("n_exact", 1024)
        modes = cfg.get("modes", ["hybrid"])
        instances = []
        for n_c in clusters_list:
            for mode in modes:
                instances.append(KMeansHybridTopK(
                    n_clusters=n_c,
                    n_exact=n_exact,
                    mode=mode,
                ))
        return instances
