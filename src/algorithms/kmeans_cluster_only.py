"""
Simple KMeans key clustering (cluster-only, no topK).

Plain KMeans on raw keys with log(n_c) bias. No M_Q transform,
no importance weighting, no beta fitting. Clusters at C=budget
for each budget level (cached after first call).
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionOutput
from ..core import softmax, flat_kmeans


class KMeansClusterOnly(AttentionAlgorithm):
    """Plain KMeans clustering + sink + local. No topK.

    Clusters at C=budget for each budget level. Results are cached.
    """

    def __init__(self):
        self._keys = None
        self._seed = 42
        self._cache = {}  # budget -> labels

    @property
    def name(self):
        return "KMeans"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._keys = keys
        self._seed = seed
        self._cache = {}

    def _get_labels(self, budget):
        if budget in self._cache:
            return self._cache[budget]
        import time as _time
        n_keys = len(self._keys)
        n_clust = min(budget, n_keys)
        print(f"[KMeans] clustering {n_keys} keys -> {n_clust} "
              f"clusters...", flush=True)
        t0 = _time.time()
        _, labels = flat_kmeans(
            self._keys.astype(np.float32), n_clust,
            seed=self._seed, n_iter=50)
        print(f"[KMeans] done ({_time.time()-t0:.0f}s)", flush=True)
        self._cache[budget] = labels
        return labels

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
            return AttentionOutput(output=out,
                                   actual_budget=len(special_idx))

        labels_all = self._get_labels(budget)
        n_clust = int(labels_all.max()) + 1
        q64 = q.astype(np.float64)

        cand_labels = labels_all[candidate_idx]
        cand_keys_f = keys[candidate_idx].astype(np.float64)
        cand_vals_f = values[candidate_idx].astype(np.float64)

        cnts = np.bincount(
            cand_labels, minlength=n_clust).astype(np.float64)
        k_sums = np.zeros((n_clust, d), np.float64)
        v_sums = np.zeros((n_clust, d), np.float64)
        for j in range(d):
            k_sums[:, j] = np.bincount(
                cand_labels, weights=cand_keys_f[:, j],
                minlength=n_clust)
            v_sums[:, j] = np.bincount(
                cand_labels, weights=cand_vals_f[:, j],
                minlength=n_clust)

        active = np.where(cnts > 0)[0]
        n_sp = len(special_idx)
        n_cl = len(active)
        n_total = n_sp + n_cl

        scores = np.empty(n_total, np.float64)
        out_vals = np.empty((n_total, d), np.float64)

        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx].astype(np.float64)

        for i, c in enumerate(active):
            nc = cnts[c]
            mk = k_sums[c] / nc
            mv = v_sums[c] / nc
            scores[n_sp + i] = float(q64 @ mk) / sqrt_d + np.log(nc)
            out_vals[n_sp + i] = mv

        w = softmax(scores).astype(np.float32)
        output = (w @ out_vals.astype(np.float32))

        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        return [KMeansClusterOnly()]
