"""
Two-stage clustering: key clusters → value sub-clusters.

Stage 1: Cluster keys (standard KMeans or QClust prototypes)
Stage 2: Within each key cluster, sub-cluster values into C_v groups

Total sub-clusters = C_k × C_v = budget/2
Half budget for oracle topK (removed from sub-clusters).

Each sub-cluster has its own mean_key (for scoring) and mean_value (for output).
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax, flat_kmeans


def _two_stage_cluster(cand_keys, cand_vals, first_stage_labels, n_key_clusters,
                       n_val_clusters, seed):
    """
    Given first-stage key cluster labels, sub-cluster values within each.
    Returns flat labels [0 .. n_key_clusters * n_val_clusters).
    """
    n_cand = len(cand_keys)
    flat_labels = np.full(n_cand, -1, dtype=np.int32)
    sub_id = 0

    for kc in range(n_key_clusters):
        members = np.where(first_stage_labels == kc)[0]
        if len(members) == 0:
            sub_id += n_val_clusters
            continue

        if len(members) <= n_val_clusters or n_val_clusters <= 1:
            # Too few members to sub-cluster, keep as one group
            flat_labels[members] = sub_id
            sub_id += n_val_clusters
            continue

        # KMeans on values within this key cluster
        vals_sub = cand_vals[members]
        n_vc = min(n_val_clusters, len(members))

        # Check for degenerate values (all identical)
        n_unique = len(np.unique(vals_sub, axis=0))
        if n_unique < n_vc:
            n_vc = max(1, n_unique)

        if n_vc <= 1:
            flat_labels[members] = sub_id
            sub_id += n_val_clusters
            continue

        _, sub_labels = flat_kmeans(
            vals_sub, n_vc,
            seed=seed + kc, n_iter=30,
        )
        for vc in range(n_vc):
            mask = sub_labels == vc
            if mask.any():
                flat_labels[members[mask]] = sub_id + vc
        sub_id += n_val_clusters

    # Compact labels (remove gaps from empty sub-clusters)
    used = np.unique(flat_labels[flat_labels >= 0])
    remap = np.full(flat_labels.max() + 1, -1, dtype=np.int32)
    for new_id, old_id in enumerate(used):
        remap[old_id] = new_id
    flat_labels = remap[flat_labels]
    n_total = len(used)
    return flat_labels, n_total


def _run_twostage(q, keys, values, logits, special_idx, candidate_idx,
                  head_dim, b_topk, flat_labels, n_sub_clusters):
    """Build output: topK + all sub-clusters. b_topk = number of oracle topK keys."""
    sqrt_d = np.sqrt(head_dim)
    q64 = q.astype(np.float64)
    n_cand = len(candidate_idx)
    d = values.shape[1]

    if n_cand == 0:
        out = softmax(logits[special_idx]) @ values[special_idx]
        return AttentionOutput(output=out, actual_budget=len(special_idx))

    cand_keys = keys[candidate_idx]
    cand_vals = values[candidate_idx]
    b_topk = min(b_topk, n_cand)

    # Oracle topK
    cand_logits = logits[candidate_idx]
    if b_topk < n_cand:
        topk_local = np.argpartition(cand_logits, -b_topk)[-b_topk:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = candidate_idx[topk_local]

    # Sub-cluster stats
    cand_keys_f = cand_keys.astype(np.float64)
    cand_vals_f = cand_vals.astype(np.float64)
    k_sums = np.zeros((n_sub_clusters, d), dtype=np.float64)
    v_sums = np.zeros((n_sub_clusters, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(flat_labels, weights=cand_keys_f[:, j], minlength=n_sub_clusters)
        v_sums[:, j] = np.bincount(flat_labels, weights=cand_vals_f[:, j], minlength=n_sub_clusters)
    counts = np.bincount(flat_labels, minlength=n_sub_clusters).astype(np.float64)

    # Remove topK from sub-clusters
    sel_labels = flat_labels[topk_local]
    sel_k = cand_keys_f[topk_local]
    sel_v = cand_vals_f[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_sub_clusters)
        v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_sub_clusters)
    cnts_r = counts - np.bincount(sel_labels, minlength=n_sub_clusters).astype(np.float64)

    # Joint softmax
    active = np.where(cnts_r > 0)[0]
    n_sp = len(special_idx)
    n_topk = len(topk_local)
    n_total = n_sp + n_topk + len(active)
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
    return AttentionOutput(output=w @ out_vals, actual_budget=n_total)


# ═══════════════════════════════════════════════
# KeyClust + Value sub-clusters
# ═══════════════════════════════════════════════

class KeyClustValSub(AttentionAlgorithm):
    """Two-stage: KMeans on keys, then KMeans on values within each cluster."""

    def __init__(self, n_val_clusters: int = 2):
        self.n_val_clusters = n_val_clusters
        self._seed = 42

    @property
    def name(self):
        return f"KeyClust-V{self.n_val_clusters}+TopK"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed

    def run(self, problem, budget, rng):
        """
        topK = budget/2 (same for all methods).
        Cluster budget = budget/2 = C_k × C_v.
        So C_k = budget / (2 × C_v).
        """
        n_cand = len(problem.candidate_idx)
        b_topk = budget // 2
        b_cluster = budget - b_topk
        n_key_clusters = max(1, b_cluster // self.n_val_clusters)
        n_key_clusters = min(n_key_clusters, n_cand)

        cand_keys = problem.keys[problem.candidate_idx]
        cand_vals = problem.values[problem.candidate_idx]

        _, key_labels = flat_kmeans(cand_keys, n_key_clusters,
                                    seed=self._seed, n_iter=50)

        flat_labels, n_sub = _two_stage_cluster(
            cand_keys, cand_vals, key_labels,
            n_key_clusters, self.n_val_clusters, self._seed,
        )

        return _run_twostage(
            problem.query, problem.keys, problem.values,
            problem.logits, problem.special_idx, problem.candidate_idx,
            problem.head_dim, b_topk, flat_labels, n_sub,
        )

    @staticmethod
    def expand_from_config(cfg):
        v_clusters = cfg.get("n_val_clusters", [4, 8])
        if isinstance(v_clusters, int):
            v_clusters = [v_clusters]
        return [KeyClustValSub(n_val_clusters=v) for v in v_clusters]


# ═══════════════════════════════════════════════
# QClust + Value sub-clusters
# ═══════════════════════════════════════════════

class QClustValSub(AttentionAlgorithm):
    """Two-stage: QClust on keys (query prototypes), then KMeans on values."""

    def __init__(self, n_val_clusters: int = 2, n_proto: int = 8,
                 n_train_queries: int = 100):
        self.n_val_clusters = n_val_clusters
        self.n_proto = n_proto
        self.n_train_queries = n_train_queries
        self._q_centers = None
        self._seed = 42

    @property
    def name(self):
        if self.n_train_queries >= 100000:
            q_label = "All"
        else:
            q_label = f"L{self.n_train_queries}"
        return f"QClust{self.n_proto}-{q_label}-V{self.n_val_clusters}+TopK"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed
        self._q_centers = None
        if queries is not None:
            n_q = len(queries)
            n_use = min(self.n_train_queries, n_q - 1)
            Q_train = queries[-(n_use + 1):-1].astype(np.float64)
            if len(Q_train) >= self.n_proto:
                centers, _ = flat_kmeans(
                    Q_train.astype(np.float32),
                    self.n_proto, seed=seed, n_iter=30,
                )
                self._q_centers = centers.astype(np.float64)

    def run(self, problem, budget, rng):
        """
        topK = budget/2 (same for all methods).
        Cluster budget = budget/2 = C_k × C_v.
        So C_k = budget / (2 × C_v).
        """
        n_cand = len(problem.candidate_idx)
        sqrt_d = np.sqrt(problem.head_dim)
        b_topk = budget // 2
        b_cluster = budget - b_topk
        n_key_clusters = max(1, b_cluster // self.n_val_clusters)
        n_key_clusters = min(n_key_clusters, n_cand)

        cand_keys = problem.keys[problem.candidate_idx]
        cand_vals = problem.values[problem.candidate_idx]

        # First stage: QClust on keys
        if self._q_centers is not None:
            proto_logits = (
                cand_keys.astype(np.float64)
                @ self._q_centers.T
            ) / sqrt_d
            _, key_labels = flat_kmeans(
                proto_logits.astype(np.float32),
                n_key_clusters, seed=self._seed, n_iter=50,
            )
        else:
            _, key_labels = flat_kmeans(
                cand_keys, n_key_clusters,
                seed=self._seed, n_iter=50,
            )

        # Second stage: value sub-clusters within each key cluster
        flat_labels, n_sub = _two_stage_cluster(
            cand_keys, cand_vals, key_labels,
            n_key_clusters, self.n_val_clusters, self._seed,
        )

        return _run_twostage(
            problem.query, problem.keys, problem.values,
            problem.logits, problem.special_idx, problem.candidate_idx,
            problem.head_dim, b_topk, flat_labels, n_sub,
        )

    @staticmethod
    def expand_from_config(cfg):
        v_clusters = cfg.get("n_val_clusters", [2, 4, 8, 16])
        if isinstance(v_clusters, int):
            v_clusters = [v_clusters]
        n_proto = cfg.get("n_proto", 8)
        n_train_list = cfg.get("n_train_queries", [100])
        if isinstance(n_train_list, int):
            n_train_list = [n_train_list]
        instances = []
        for n_train in n_train_list:
            for v in v_clusters:
                instances.append(QClustValSub(
                    n_val_clusters=v, n_proto=n_proto,
                    n_train_queries=n_train,
                ))
        return instances
