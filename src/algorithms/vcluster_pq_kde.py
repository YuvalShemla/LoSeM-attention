"""
Value-Cluster PQ-KDE attention.

Full attention over ALL keys using PQ hybrid logits (top-K exact,
rest PQ approximate). Values replaced by value-cluster centers.

Two variants:
  A) VClustPQ-KDE: all values = cluster centers
  B) VClustPQ-KDE-exV: top-K get exact values, rest cluster centers
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from .pq_topk import PQIndex
from ..core import softmax, flat_kmeans


class VClusterPQKDE(AttentionAlgorithm):

    def __init__(self, m_pq=8, topk=1024, exact_topk_vals=False):
        self.m_pq = m_pq
        self.topk = topk
        self.exact_topk_vals = exact_topk_vals
        self._pq = None
        self._seed = 42

    @property
    def name(self):
        base = f"PQFullAttn+VClust-top{self.topk}"
        if self.exact_topk_vals:
            base += "+exV"
        return base

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed
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
            return AttentionOutput(output=out,
                                   actual_budget=len(special_idx))

        n_clust = min(budget, n_cand)

        # Cluster candidate values
        cand_vals = values[candidate_idx]
        _, labels = flat_kmeans(cand_vals, n_clust,
                                seed=self._seed, n_iter=50)

        cand_vals_f = cand_vals.astype(np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust)
        cnts = np.bincount(labels, minlength=n_clust).astype(np.float64)
        mean_vals = np.zeros((n_clust, d), dtype=np.float64)
        for c in range(n_clust):
            if cnts[c] > 0:
                mean_vals[c] = v_sums[c] / cnts[c]

        replaced_vals = mean_vals[labels]  # [n_cand, d]

        # PQ approximate logits for all candidates
        pq = self._pq
        m = pq.m; dsub = pq.dsub
        q_f32 = q.astype(np.float32)
        lut = np.empty((m, pq.n_codes), dtype=np.float32)
        for i in range(m):
            lut[i] = pq.codebooks[i] @ q_f32[i * dsub:(i + 1) * dsub]
        approx_ip = np.zeros(len(pq.codes), dtype=np.float32)
        for i in range(m):
            approx_ip += lut[i][pq.codes[:, i]]
        cand_lg = approx_ip[candidate_idx].astype(np.float64) / sqrt_d

        # Top-K: exact logits (and optionally exact values)
        k = min(self.topk, n_cand)
        topk_l = np.argpartition(cand_lg, -k)[-k:]
        cand_lg[topk_l] = logits[candidate_idx[topk_l]].astype(np.float64)

        if self.exact_topk_vals:
            replaced_vals[topk_l] = cand_vals_f[topk_l]

        # Full softmax: special + all candidates
        all_lg = np.concatenate([
            logits[special_idx].astype(np.float64), cand_lg])
        all_v = np.vstack([
            values[special_idx].astype(np.float64), replaced_vals])
        w = softmax(all_lg)
        output = (w @ all_v).astype(np.float32)

        return AttentionOutput(output=output,
                               actual_budget=len(special_idx) + n_cand)

    @staticmethod
    def expand_from_config(cfg):
        m_pq = cfg.get("m_pq", 8)
        topk = cfg.get("topk", 1024)
        variants = cfg.get("variants", ["vclust", "vclust_exv"])
        out = []
        for v in variants:
            exv = v == "vclust_exv"
            out.append(VClusterPQKDE(m_pq=m_pq, topk=topk,
                                     exact_topk_vals=exv))
        return out
