"""
QClust: Query-adapted key clustering.

Cluster training queries into P prototypes. For each candidate key,
compute its logit against each prototype → P-dimensional feature.
KMeans on these features. n_clusters = budget/2.
Half budget topK + half budget clusters. TopK removed from clusters.
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax, flat_kmeans


class QClustTopK(AttentionAlgorithm):
    """
    QClust-P + oracle topK. n_clusters = budget/2.

    In prepare(): cluster last N training queries into P prototypes.
    In run(): compute each key's P-dim logit profile against prototypes,
    KMeans on those features, oracle topK removed from clusters.
    """

    def __init__(self, n_proto: int = 8, n_train_queries: int = 1000):
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
        return f"QClust{self.n_proto}-{q_label}+TopK"

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
                    self.n_proto,
                    seed=seed, n_iter=30,
                )
                self._q_centers = centers.astype(np.float64)

    def run(self, problem, budget, rng):
        q = problem.query
        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        logits = problem.logits
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        sqrt_d = np.sqrt(head_dim)
        n_cand = len(candidate_idx)
        d = values.shape[1]

        if n_cand == 0:
            out = softmax(logits[special_idx]) @ values[special_idx]
            return AttentionOutput(output=out, actual_budget=len(special_idx))

        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]

        # Budget split
        b_topk = min(budget // 2, n_cand)
        b_cluster = budget - b_topk
        n_clust = min(b_cluster, n_cand)
        if n_clust < 1:
            n_clust = 1

        # Cluster keys by logit profile against query prototypes
        if self._q_centers is not None:
            proto_logits = (
                cand_keys.astype(np.float64)
                @ self._q_centers.T
            ) / sqrt_d
            _, labels = flat_kmeans(
                proto_logits.astype(np.float32),
                n_clust, seed=self._seed, n_iter=50,
            )
        else:
            _, labels = flat_kmeans(
                cand_keys, n_clust,
                seed=self._seed, n_iter=50,
            )

        # Cluster stats
        cand_keys_f = cand_keys.astype(np.float64)
        cand_vals_f = cand_vals.astype(np.float64)
        k_sums = np.zeros((n_clust, d), dtype=np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            k_sums[:, j] = np.bincount(labels, weights=cand_keys_f[:, j], minlength=n_clust)
            v_sums[:, j] = np.bincount(labels, weights=cand_vals_f[:, j], minlength=n_clust)
        counts = np.bincount(labels, minlength=n_clust).astype(np.float64)

        # Oracle topK
        cand_logits = logits[candidate_idx]
        if b_topk < n_cand:
            topk_local = np.argpartition(cand_logits, -b_topk)[-b_topk:]
        else:
            topk_local = np.arange(n_cand)
        topk_global = candidate_idx[topk_local]

        # Remove topK from clusters
        sel_labels = labels[topk_local]
        sel_k = cand_keys_f[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_clust)
            v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_clust)
        sel_counts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
        cnts_r = counts - sel_counts

        # Build joint softmax
        q64 = q.astype(np.float64)
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

    @staticmethod
    def expand_from_config(cfg):
        n_proto = cfg.get("n_proto", 8)
        n_train_list = cfg.get("n_train_queries", [1000])
        if isinstance(n_train_list, int):
            n_train_list = [n_train_list]
        return [
            QClustTopK(n_proto=n_proto, n_train_queries=n)
            for n in n_train_list
        ]
