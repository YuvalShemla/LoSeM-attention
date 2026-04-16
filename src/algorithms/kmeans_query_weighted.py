"""
KMeans Query-Weighted: cluster values are weighted by
their key's score against the mean query, instead of
uniform averaging.

Within each cluster:
  w_i = softmax(q̄ · k_i / √d)   (within-cluster)
  weighted_val = Σ w_i * v_i

alpha controls interpolation:
  final_val = (1 - alpha) * mean_val + alpha * weighted_val

alpha=1.0 → full query-weighted, alpha=0.5 → half blend.

Representatives recomputed per-query from causal window.
"""

import numpy as np
from typing import List

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, cached_flat_kmeans


class KMeansQueryWeighted(AttentionAlgorithm):
    """
    Flat KMeans on keys, value representatives are
    weighted by key scores against the mean query.
    Count-weighted grouped softmax.
    """

    def __init__(
        self,
        n_clusters: int = 256,
        alpha: float = 1.0,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self._member_indices = None
        self._mean_query = None

    @property
    def name(self) -> str:
        a = int(self.alpha * 100)
        return f"KMeansQW{a}-{self.n_clusters}"

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

        if queries is not None:
            self._mean_query = np.mean(
                queries, axis=0,
            ).astype(np.float32)
        else:
            self._mean_query = np.mean(
                keys, axis=0,
            ).astype(np.float32)

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
        q_bar = self._mean_query

        C = self.n_clusters
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

        if n_special > 0:
            all_scores[:n_special] = (
                problem.logits[special_idx]
            )
            out_values[:n_special] = (
                values[special_idx]
            )

        off = n_special
        alpha = self.alpha
        for fi, c in enumerate(valid):
            idx = causal_members[c]
            count = len(idx)
            cluster_keys = keys[idx]
            cluster_vals = values[idx]

            # Cluster score: mean key dot query + log count
            mean_key = np.mean(cluster_keys, axis=0)
            all_scores[off + fi] = (
                q @ mean_key / sqrt_d
                + np.log(count)
            )

            # Value representative
            if alpha == 0.0 or count == 1:
                out_values[off + fi] = np.mean(
                    cluster_vals, axis=0,
                )
            else:
                # Within-cluster softmax weights
                # using mean query
                inner_scores = (
                    cluster_keys @ q_bar / sqrt_d
                )
                w = softmax(inner_scores)
                qw_val = w @ cluster_vals

                if alpha == 1.0:
                    out_values[off + fi] = qw_val
                else:
                    mean_val = np.mean(
                        cluster_vals, axis=0,
                    )
                    out_values[off + fi] = (
                        (1 - alpha) * mean_val
                        + alpha * qw_val
                    )

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
        alphas = cfg.get("alpha_sweep", [1.0])
        instances = []
        for a in alphas:
            for n_c in clusters_list:
                instances.append(
                    KMeansQueryWeighted(
                        n_clusters=n_c, alpha=a,
                    )
                )
        return instances
