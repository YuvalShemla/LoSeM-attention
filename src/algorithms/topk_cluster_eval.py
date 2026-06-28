"""
TopK + Cluster evaluation: test second-order correction.

Budget split: topk_frac (default 7/8) for topK, rest for clusters.
Number of clusters = B * (1 - topk_frac), scaling with budget.

Four variants from two axes:
  TopK method:     Oracle (exact logits) vs PQ (approximate)
  Cluster method:  Oracle (equal-size by logit rank) vs KMeans (on keys)
  Scoring order:   1st (mean-key) vs 2nd (+var/2 correction)

Produces:
  1. OracleTopK+OracleClust   — oracle topK, oracle clusters, 1st order
  2. PQTopK+OracleClust       — PQ topK, oracle clusters, 1st order
  3. PQTopK+KeyClust           — PQ topK, KMeans clusters, 1st order
  4. PQTopK+KeyClust-2nd       — PQ topK, KMeans clusters, 2nd order
"""

import numpy as np

from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from .pq_topk import PQIndex
from ..core import softmax, flat_kmeans


class TopKClusterEval(AttentionAlgorithm):
    """
    Configurable topK + cluster method for evaluation.

    topk_method:    "oracle" or "pq"
    cluster_method: "oracle" (equal-size by logit rank) or "kmeans"
    order:          0 (1st order) or 2 (2nd order with var/2 correction)
    topk_frac:      fraction of budget for topK (default 7/8)
    """

    def __init__(self, topk_method="oracle", cluster_method="oracle",
                 order=0, topk_frac=7/8, m_pq=8):
        self.topk_method = topk_method
        self.cluster_method = cluster_method
        self.order = order
        self.topk_frac = topk_frac
        self.m_pq = m_pq
        self._pq = None
        self._seed = 42

    @property
    def name(self):
        topk = "Oracle" if self.topk_method == "oracle" else "PQ"
        clust = "OracleClust" if self.cluster_method == "oracle" else "KeyClust"
        ord_str = "-2nd" if self.order == 2 else ""
        return f"{topk}TopK+{clust}{ord_str}"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed
        if self.topk_method == "pq":
            self._pq = PQIndex(m=self.m_pq, n_codes=256, seed=seed)
            self._pq.fit(keys)

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
        n = len(keys)
        d = values.shape[1]

        if n_cand == 0:
            out = softmax(logits[special_idx]) @ values[special_idx]
            return AttentionOutput(output=out, actual_budget=len(special_idx))

        # Budget split: topk_frac for topK, rest for cluster count
        b_topk = min(int(budget * self.topk_frac), n_cand)
        n_clust = max(budget - b_topk, 1)
        n_clust = min(n_clust, n_cand)

        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]
        cand_logits = logits[candidate_idx]
        cand_keys_f = cand_keys.astype(np.float64)
        cand_vals_f = cand_vals.astype(np.float64)
        cand_logits_f = cand_logits.astype(np.float64)

        # --- Cluster candidates ---
        if self.cluster_method == "oracle":
            sort_order = np.argsort(cand_logits)[::-1]
            labels = np.zeros(n_cand, dtype=np.int32)
            gs = n_cand // n_clust
            rem = n_cand % n_clust
            pos = 0
            for c in range(n_clust):
                sz = gs + (1 if c < rem else 0)
                labels[sort_order[pos:pos + sz]] = c
                pos += sz
        else:
            _, labels = flat_kmeans(cand_keys, n_clust,
                                    seed=self._seed, n_iter=50)

        # --- Cluster stats: key_sum, val_sum, counts ---
        k_sums = np.zeros((n_clust, d), dtype=np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            k_sums[:, j] = np.bincount(
                labels, weights=cand_keys_f[:, j], minlength=n_clust)
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j], minlength=n_clust)
        cnts = np.bincount(labels, minlength=n_clust).astype(np.float64)

        # For 2nd order: precompute M[c] = Σ k_i k_i^T per cluster
        # This is query-independent and can be precomputed ahead of time.
        if self.order == 2:
            M = np.zeros((n_clust, d, d), dtype=np.float64)
            for c in range(n_clust):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                ck = cand_keys_f[mask]
                M[c] = ck.T @ ck

        # --- Select topK ---
        if self.topk_method == "oracle":
            if b_topk < n_cand:
                topk_local = np.argpartition(
                    cand_logits, -b_topk)[-b_topk:]
            else:
                topk_local = np.arange(n_cand)
            topk_global = candidate_idx[topk_local]
        else:
            n_pq = len(self._pq.codes)
            cand_mask = np.zeros(n_pq, dtype=bool)
            cand_mask[candidate_idx] = True
            topk_global = self._pq.approximate_topk(
                q, b_topk, candidate_mask=cand_mask)
            g2l = np.full(n_pq, -1, dtype=np.int64)
            g2l[candidate_idx] = np.arange(n_cand)
            topk_local = g2l[topk_global]
            valid = topk_local >= 0
            topk_local = topk_local[valid]
            topk_global = topk_global[valid]

        # --- Remove topK from cluster stats ---
        sel_labels = labels[topk_local]
        sel_k = cand_keys_f[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            k_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_k[:, j], minlength=n_clust)
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j], minlength=n_clust)
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust).astype(np.float64)
        cnts_r = cnts - sel_cnts

        if self.order == 2:
            M_r = M.copy()
            for idx in range(len(topk_local)):
                c = sel_labels[idx]
                M_r[c] -= np.outer(sel_k[idx], sel_k[idx])

        # --- Build joint softmax: special + topK + active clusters ---
        active = np.where(cnts_r > 0)[0]
        n_sp = len(special_idx)
        n_topk = len(topk_local)
        n_active = len(active)
        n_total = n_sp + n_topk + n_active

        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)
        q64 = q.astype(np.float64)

        # Special tokens
        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]

        # TopK individuals (use TRUE logits)
        off = n_sp
        scores[off:off + n_topk] = logits[topk_global].astype(np.float64)
        out_vals[off:off + n_topk] = values[topk_global]

        # Residual cluster representatives
        off = n_sp + n_topk
        for i, c in enumerate(active):
            nc = cnts_r[c]
            mk = k_sums[c] / nc
            mv = (v_sums[c] / nc).astype(np.float32)
            ml = float(q64 @ mk) / sqrt_d

            if self.order == 2:
                qMq = float(q64 @ M_r[c] @ q64)
                var_l = max(qMq / (nc * d) - ml ** 2, 0.0)
                scores[off + i] = ml + var_l / 2 + np.log(nc)
            else:
                scores[off + i] = ml + np.log(nc)

            out_vals[off + i] = mv

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals

        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        topk_frac = cfg.get("topk_frac", 7 / 8)
        m_pq = cfg.get("m_pq", 8)
        variants = cfg.get("variants", [
            {"topk": "oracle", "cluster": "oracle", "order": 0},
            {"topk": "pq", "cluster": "oracle", "order": 0},
            {"topk": "pq", "cluster": "kmeans", "order": 0},
            {"topk": "pq", "cluster": "kmeans", "order": 2},
        ])
        return [
            TopKClusterEval(
                topk_method=v["topk"],
                cluster_method=v["cluster"],
                order=v.get("order", 0),
                topk_frac=topk_frac,
                m_pq=m_pq,
            )
            for v in variants
        ]
