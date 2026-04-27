"""
TopK + Cluster comparison methods.

Budget split: half for oracle topK, half for clusters.
Number of clusters = budget / 2 (scales with budget).
TopK keys removed from cluster stats.

Variants differ only in how clusters are formed:
  1. KeyClust+TopK:    KMeans on candidate KEYS
  2. OracleClust+TopK: equal-size groups sorted by true logit
  3. ValClust+TopK:    KMeans on candidate VALUES
"""

import numpy as np

from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax, flat_kmeans


def _run_topk_plus_clusters(
    q, keys, values, logits, special_idx, candidate_idx,
    head_dim, budget, labels, n_clust, seed,
):
    """
    Shared logic: half budget topK + half budget clusters.
    labels: cluster assignment for candidates (length n_cand).
    """
    sqrt_d = np.sqrt(head_dim)
    q64 = q.astype(np.float64)
    n_cand = len(candidate_idx)
    d = values.shape[1]

    if n_cand == 0:
        out = softmax(logits[special_idx]) @ values[special_idx]
        return AttentionOutput(output=out, actual_budget=len(special_idx))

    cand_keys = keys[candidate_idx]
    cand_vals = values[candidate_idx]

    # Budget split
    b_topk = min(budget // 2, n_cand)
    # n_clust is already set by caller

    # Oracle topK
    cand_logits = logits[candidate_idx]
    if b_topk < n_cand:
        topk_local = np.argpartition(cand_logits, -b_topk)[-b_topk:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = candidate_idx[topk_local]

    # Cluster stats
    cand_keys_f = cand_keys.astype(np.float64)
    cand_vals_f = cand_vals.astype(np.float64)
    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    v_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(labels, weights=cand_keys_f[:, j], minlength=n_clust)
        v_sums[:, j] = np.bincount(labels, weights=cand_vals_f[:, j], minlength=n_clust)
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)

    # Remove topK from clusters
    sel_labels = labels[topk_local]
    sel_k = cand_keys_f[topk_local]
    sel_v = cand_vals_f[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_clust)
    sel_counts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
    cnts_r = counts - sel_counts

    # Build joint softmax: special + topK + all residual clusters
    active = np.where(cnts_r > 0)[0]
    n_sp = len(special_idx)
    n_topk = len(topk_local)
    n_active = len(active)
    n_total = n_sp + n_topk + n_active

    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float32)

    scores[:n_sp] = logits[special_idx].astype(np.float64)
    out_vals[:n_sp] = values[special_idx]
    off = n_sp
    scores[off:off + n_topk] = logits[topk_global].astype(np.float64)
    out_vals[off:off + n_topk] = values[topk_global]
    off = n_sp + n_topk
    for i, c in enumerate(active):
        nc = cnts_r[c]
        mk = k_sums[c] / nc
        mv = (v_sums[c] / nc).astype(np.float32)
        scores[off + i] = float(q64 @ mk) / sqrt_d + np.log(nc)
        out_vals[off + i] = mv

    w = softmax(scores).astype(np.float32)
    output = w @ out_vals
    return AttentionOutput(output=output, actual_budget=n_total)


class TopKKeyClusters(AttentionAlgorithm):
    """KMeans on keys. n_clusters = budget/2."""

    def __init__(self):
        self._seed = 42

    @property
    def name(self):
        return "KeyClust+TopK"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed

    def run(self, problem, budget, rng):
        n_cand = len(problem.candidate_idx)
        b_cluster = budget - budget // 2
        n_clust = min(b_cluster, n_cand)
        if n_clust < 1:
            n_clust = 1

        cand_keys = problem.keys[problem.candidate_idx]
        _, labels = flat_kmeans(cand_keys, n_clust, seed=self._seed, n_iter=50)

        return _run_topk_plus_clusters(
            problem.query, problem.keys, problem.values,
            problem.logits, problem.special_idx, problem.candidate_idx,
            problem.head_dim, budget, labels, n_clust, self._seed,
        )

    @staticmethod
    def expand_from_config(cfg):
        return [TopKKeyClusters()]


class TopKOracleClusters(AttentionAlgorithm):
    """Oracle equal-size groups by logit rank. n_groups = budget/2."""

    @property
    def name(self):
        return "OracleClust+TopK"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        pass

    def run(self, problem, budget, rng):
        n_cand = len(problem.candidate_idx)
        b_cluster = budget - budget // 2
        n_clust = min(b_cluster, n_cand)
        if n_clust < 1:
            n_clust = 1

        cand_logits = problem.logits[problem.candidate_idx]
        sort_order = np.argsort(cand_logits)[::-1]

        labels = np.zeros(n_cand, dtype=np.int32)
        gs = n_cand // n_clust
        rem = n_cand % n_clust
        pos = 0
        for c in range(n_clust):
            sz = gs + (1 if c < rem else 0)
            labels[sort_order[pos:pos + sz]] = c
            pos += sz

        return _run_topk_plus_clusters(
            problem.query, problem.keys, problem.values,
            problem.logits, problem.special_idx, problem.candidate_idx,
            problem.head_dim, budget, labels, n_clust, 42,
        )

    @staticmethod
    def expand_from_config(cfg):
        return [TopKOracleClusters()]


class TopKValueClusters(AttentionAlgorithm):
    """KMeans on values. n_clusters = budget/2."""

    def __init__(self):
        self._seed = 42

    @property
    def name(self):
        return "ValClust+TopK"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed

    def run(self, problem, budget, rng):
        n_cand = len(problem.candidate_idx)
        b_cluster = budget - budget // 2
        n_clust = min(b_cluster, n_cand)
        if n_clust < 1:
            n_clust = 1

        cand_vals = problem.values[problem.candidate_idx]
        _, labels = flat_kmeans(cand_vals, n_clust, seed=self._seed, n_iter=50)

        return _run_topk_plus_clusters(
            problem.query, problem.keys, problem.values,
            problem.logits, problem.special_idx, problem.candidate_idx,
            problem.head_dim, budget, labels, n_clust, self._seed,
        )

    @staticmethod
    def expand_from_config(cfg):
        return [TopKValueClusters()]
