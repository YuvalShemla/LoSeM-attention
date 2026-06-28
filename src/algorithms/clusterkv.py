"""
ClusterKV — Liu et al., 2024.

Cluster keys in cosine-similarity space, then at query time select the
top-scoring clusters by query-centroid inner product and do exact
attention over all member tokens of the selected clusters.

Paper: "ClusterKV: Manipulating LLM KV Cache in Semantic Space for
        Recallable Compression"

Algorithm:
  1. (prepare) Normalize keys to unit length.  Run K-means with
     C = N/80 clusters (cosine K-means via Euclidean on unit sphere).
  2. (run) Score each centroid: s_c = q @ centroid_c / sqrt(d).
     Greedily select top clusters until budget B tokens are collected.
     Exact subset attention over special + selected cluster members.

Key difference from MQBeta:
  - ClusterKV does EXACT attention on selected member tokens.
  - MQBeta uses cluster-MEAN approximation but covers ALL keys.
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionOutput
from ..core import softmax, subset_attention


class ClusterKVEviction(AttentionAlgorithm):
    """ClusterKV: cosine-KMeans + cluster-level retrieval."""

    def __init__(self, cluster_ratio=80):
        self.cluster_ratio = cluster_ratio
        self._centroids = None
        self._all_labels = None
        self._n_clusters = None

    @property
    def name(self) -> str:
        return "ClusterKV"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        from ..core import flat_kmeans
        N = keys.shape[0]

        test_positions = set(query_positions) if query_positions else set()
        k_train_mask = np.array(
            [i not in test_positions for i in range(N)], bool)
        train_key_idx = np.where(k_train_mask)[0]
        K_train = keys[train_key_idx].astype(np.float64)

        # Cosine K-means: normalize then Euclidean
        norms = np.linalg.norm(K_train, axis=1, keepdims=True)
        K_norm = (K_train / np.maximum(norms, 1e-12)).astype(np.float32)

        n_clust = max(1, len(train_key_idx) // self.cluster_ratio)
        self._n_clusters = n_clust
        centroids, train_labels = flat_kmeans(
            K_norm, n_clust, seed=seed, n_iter=50)

        # Re-normalize centroids (cosine K-means convention)
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self._centroids = (centroids / np.maximum(c_norms, 1e-12)
                           ).astype(np.float64)

        # Store per-cluster member counts for fast lookup
        self._cluster_sizes = np.bincount(
            train_labels, minlength=n_clust)

        # All-key labels
        self._all_labels = np.full(N, -1, dtype=np.int32)
        self._all_labels[train_key_idx] = train_labels

        # Assign test keys to nearest centroid
        test_key_idx = np.where(~k_train_mask)[0]
        if len(test_key_idx) > 0:
            K_test = keys[test_key_idx].astype(np.float64)
            tn = np.linalg.norm(K_test, axis=1, keepdims=True)
            K_test_norm = K_test / np.maximum(tn, 1e-12)
            sims = K_test_norm @ self._centroids.T
            self._all_labels[test_key_idx] = np.argmax(
                sims, axis=1).astype(np.int32)

    def run(self, problem, budget, rng):
        q = problem.query
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        head_dim = problem.head_dim
        n_cand = len(candidate_idx)

        if n_cand == 0:
            output = subset_attention(logits, values, special_idx)
            return AttentionOutput(output=output,
                                   actual_budget=len(special_idx))

        sqrt_d = np.sqrt(head_dim)
        q64 = q.astype(np.float64)

        # Score centroids by q @ centroid / sqrt_d
        centroid_scores = (q64 @ self._centroids.T) / sqrt_d

        # Sort clusters by score descending
        sorted_clusters = np.argsort(centroid_scores)[::-1]

        # Build candidate label lookup
        cand_labels = self._all_labels[candidate_idx]

        # Greedily select top clusters until budget is reached
        selected_cand_pos = []
        total = 0
        for c in sorted_clusters:
            members = np.where(cand_labels == c)[0]
            if len(members) == 0:
                continue
            selected_cand_pos.append(members)
            total += len(members)
            if total >= budget:
                break

        if len(selected_cand_pos) == 0:
            output = subset_attention(logits, values, special_idx)
            return AttentionOutput(output=output,
                                   actual_budget=len(special_idx))

        all_sel = np.concatenate(selected_cand_pos)

        # Trim last cluster if over budget (keep highest-logit members)
        if len(all_sel) > budget:
            excess = len(all_sel) - budget
            last = selected_cand_pos[-1]
            last_logits = logits[candidate_idx[last]]
            drop = np.argsort(last_logits)[:excess]
            keep_mask = np.ones(len(last), dtype=bool)
            keep_mask[drop] = False
            selected_cand_pos[-1] = last[keep_mask]
            all_sel = np.concatenate(selected_cand_pos)

        topk_idx = candidate_idx[all_sel]
        all_idx = np.concatenate([special_idx, topk_idx]).astype(np.int64)
        output = subset_attention(logits, values, all_idx)
        return AttentionOutput(output=output,
                               actual_budget=len(all_idx),
                               selected_indices=all_idx)

    @staticmethod
    def expand_from_config(cfg):
        cr = cfg.get('cluster_ratio', 80)
        return [ClusterKVEviction(cr)]
