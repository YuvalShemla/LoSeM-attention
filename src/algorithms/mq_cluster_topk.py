"""
Importance-Weighted M_Q Key Clustering + PQ TopK.

Transform keys by M_Q^{1/2} (query covariance), then run weighted
KMeans where each key is weighted by its attention importance
rho_i = E_q[exp(q.k_i/sqrt_d - s(q))]. This is the 2nd-order
correction to the linearized loss: high-attention keys get
proportionally more influence on centroid placement.

Budget split: topk_frac (7/8) for PQ top-K, rest = cluster count.
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from .pq_topk import PQIndex
from ..core import softmax, flat_kmeans


def _weighted_kmeans(data, weights, n_clust, seed=42, n_iter=50):
    """KMeans with per-sample importance weights."""
    rng = np.random.default_rng(seed)
    N, d = data.shape
    n_clust = min(n_clust, N)
    data_f = data.astype(np.float64)
    w = weights.astype(np.float64)

    # KMeans++ init (weighted sampling)
    centroids = np.empty((n_clust, d), np.float64)
    centroids[0] = data_f[rng.integers(N)]
    min_dists = np.full(N, np.inf, np.float64)
    data_sq = np.sum(data_f ** 2, axis=1)
    for ci in range(1, n_clust):
        c_sq = float(np.sum(centroids[ci - 1] ** 2))
        new_d = data_sq + c_sq - 2.0 * (data_f @ centroids[ci - 1])
        np.maximum(new_d, 0.0, out=new_d)
        np.minimum(min_dists, new_d, out=min_dists)
        probs = w * min_dists
        probs /= probs.sum() + 1e-30
        centroids[ci] = data_f[rng.choice(N, p=probs)]

    # Lloyd iterations with weighted centroids
    labels = np.zeros(N, np.int32)
    for _ in range(n_iter):
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True)
        dists = data_sq[:, None] - 2.0 * (data_f @ centroids.T) + c_sq.T
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clust):
            mask = labels == c
            if mask.sum() > 0:
                wc = w[mask]
                centroids[c] = ((wc[:, None] * data_f[mask]).sum(0)
                                / (wc.sum() + 1e-30))
    return centroids.astype(np.float32), labels


class MQClusterTopK(AttentionAlgorithm):

    def __init__(self, topk_frac=7/8, m_pq=8):
        self.topk_frac = topk_frac
        self.m_pq = m_pq
        self._pq = None
        self._seed = 42
        self._eigvecs = None
        self._sqrt_eig = None
        self._rho = None  # attention importance weights

    @property
    def name(self):
        return "PQTopK+MQCluster"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._seed = seed

        self._pq = PQIndex(m=self.m_pq, n_codes=256, seed=seed)
        self._pq.fit(keys)

        if queries is not None and len(queries) > 1:
            Q_f = queries.astype(np.float64)
            d = keys.shape[1]

            # M_Q transform
            M_Q = Q_f.T @ Q_f + 1e-6 * np.eye(d)
            eigvals, eigvecs = np.linalg.eigh(M_Q)
            eigvals = np.maximum(eigvals, 0.0)
            self._sqrt_eig = np.sqrt(eigvals)
            self._eigvecs = eigvecs

            # Attention importance weights: rho_i = E_q[exp(q.k/sqrt_d - s)]
            # Use a subsample of queries for efficiency
            n_q = len(Q_f)
            n_sub = min(500, n_q)
            Q_sub = Q_f[-n_sub:]
            K_f = keys.astype(np.float64)
            sqrt_d = np.sqrt(d)
            logits = (Q_sub @ K_f.T) / sqrt_d
            s_max = logits.max(axis=1, keepdims=True)
            self._rho = np.mean(
                np.exp(logits - s_max), axis=0
            ).astype(np.float64)
        else:
            self._eigvecs = None
            self._sqrt_eig = None
            self._rho = None

    def _transform(self, keys):
        if self._eigvecs is None:
            return keys.astype(np.float32)
        return (keys.astype(np.float64)
                @ self._eigvecs
                * self._sqrt_eig[None, :]).astype(np.float32)

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

        # Budget split
        b_topk = min(int(budget * self.topk_frac), n_cand)
        n_clust = max(budget - b_topk, 1)
        n_clust = min(n_clust, n_cand)

        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]

        # Importance-weighted M_Q KMeans
        K_z = self._transform(cand_keys)
        if self._rho is not None:
            rho_cand = self._rho[candidate_idx]
        else:
            rho_cand = np.ones(n_cand)
        _, labels = _weighted_kmeans(
            K_z, rho_cand, n_clust,
            seed=self._seed, n_iter=50,
        )

        # Cluster stats (original key space)
        cand_keys_f = cand_keys.astype(np.float64)
        cand_vals_f = cand_vals.astype(np.float64)
        k_sums = np.zeros((n_clust, d), dtype=np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            k_sums[:, j] = np.bincount(
                labels, weights=cand_keys_f[:, j],
                minlength=n_clust)
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust)
        cnts = np.bincount(
            labels, minlength=n_clust).astype(np.float64)

        # PQ top-K
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

        # Remove top-K from cluster stats
        sel_labels = labels[topk_local]
        sel_k = cand_keys_f[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            k_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_k[:, j],
                minlength=n_clust)
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=n_clust)
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust).astype(np.float64)
        cnts_r = cnts - sel_cnts

        # Build joint softmax
        active_idx = np.where(cnts_r > 0)[0]
        n_sp = len(special_idx)
        n_topk = len(topk_local)
        n_total = n_sp + n_topk + len(active_idx)

        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)
        q64 = q.astype(np.float64)

        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]

        off = n_sp
        scores[off:off + n_topk] = logits[topk_global].astype(
            np.float64)
        out_vals[off:off + n_topk] = values[topk_global]

        off = n_sp + n_topk
        for i, c in enumerate(active_idx):
            nc = cnts_r[c]
            mean_k = k_sums[c] / nc
            mean_v = (v_sums[c] / nc).astype(np.float32)
            ml = float(q64 @ mean_k) / sqrt_d
            scores[off + i] = ml + np.log(nc)
            out_vals[off + i] = mean_v

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals

        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        topk_frac = cfg.get("topk_frac", 7 / 8)
        m_pq = cfg.get("m_pq", 8)
        return [MQClusterTopK(topk_frac=topk_frac, m_pq=m_pq)]


class OracleTopKMQCluster(MQClusterTopK):
    """Oracle (exact logit) top-K + importance-weighted M_Q clusters."""

    def __init__(self, topk_frac=7/8, m_pq=8):
        super().__init__(topk_frac=topk_frac, m_pq=m_pq)

    @property
    def name(self):
        return "OracleTopK+MQCluster"

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

        b_topk = min(int(budget * self.topk_frac), n_cand)
        n_clust = max(budget - b_topk, 1)
        n_clust = min(n_clust, n_cand)

        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]

        # Importance-weighted M_Q KMeans (same as parent)
        K_z = self._transform(cand_keys)
        rho_cand = (self._rho[candidate_idx]
                    if self._rho is not None
                    else np.ones(n_cand))
        _, labels = _weighted_kmeans(
            K_z, rho_cand, n_clust,
            seed=self._seed, n_iter=50)

        cand_keys_f = cand_keys.astype(np.float64)
        cand_vals_f = cand_vals.astype(np.float64)
        k_sums = np.zeros((n_clust, d), dtype=np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            k_sums[:, j] = np.bincount(
                labels, weights=cand_keys_f[:, j],
                minlength=n_clust)
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust)
        cnts = np.bincount(
            labels, minlength=n_clust).astype(np.float64)

        # Oracle top-K: exact logits
        cand_logits = logits[candidate_idx]
        if b_topk < n_cand:
            topk_local = np.argpartition(
                cand_logits, -b_topk)[-b_topk:]
        else:
            topk_local = np.arange(n_cand)
        topk_global = candidate_idx[topk_local]

        # Remove top-K from clusters
        sel_labels = labels[topk_local]
        sel_k = cand_keys_f[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            k_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_k[:, j],
                minlength=n_clust)
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=n_clust)
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust).astype(np.float64)
        cnts_r = cnts - sel_cnts

        # Joint softmax
        active_idx = np.where(cnts_r > 0)[0]
        n_sp = len(special_idx)
        n_topk = len(topk_local)
        n_total = n_sp + n_topk + len(active_idx)
        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)
        q64 = q.astype(np.float64)

        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]
        off = n_sp
        scores[off:off + n_topk] = logits[topk_global].astype(
            np.float64)
        out_vals[off:off + n_topk] = values[topk_global]
        off = n_sp + n_topk
        for i, c in enumerate(active_idx):
            nc = cnts_r[c]
            mean_k = k_sums[c] / nc
            mean_v = (v_sums[c] / nc).astype(np.float32)
            ml = float(q64 @ mean_k) / sqrt_d
            scores[off + i] = ml + np.log(nc)
            out_vals[off + i] = mean_v

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals
        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        topk_frac = cfg.get("topk_frac", 7 / 8)
        m_pq = cfg.get("m_pq", 8)
        return [OracleTopKMQCluster(topk_frac=topk_frac, m_pq=m_pq)]


class MQClusterTrueZ(MQClusterTopK):
    """PQ top-K + MQ clusters scored by true Z_c (exact logits per cluster)."""

    def __init__(self, topk_frac=7/8, m_pq=8):
        super().__init__(topk_frac=topk_frac, m_pq=m_pq)

    @property
    def name(self):
        return "PQTopK+MQClust-trueZ"

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

        b_topk = min(int(budget * self.topk_frac), n_cand)
        n_clust = max(budget - b_topk, 1)
        n_clust = min(n_clust, n_cand)

        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]

        K_z = self._transform(cand_keys)
        rho_cand = (self._rho[candidate_idx]
                    if self._rho is not None
                    else np.ones(n_cand))
        _, labels = _weighted_kmeans(
            K_z, rho_cand, n_clust,
            seed=self._seed, n_iter=50)

        cand_vals_f = cand_vals.astype(np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust)
        cnts = np.bincount(
            labels, minlength=n_clust).astype(np.float64)

        # PQ top-K
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

        # Remove top-K from cluster stats
        sel_labels = labels[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=n_clust)
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust).astype(np.float64)
        cnts_r = cnts - sel_cnts

        # True Z_c: exact logits for residual keys per cluster
        exact_cand_logits = logits[candidate_idx].astype(np.float64)
        topk_set = set(topk_local.tolist())
        z_clusters = np.zeros(n_clust, dtype=np.float64)
        for c in range(n_clust):
            members = np.where(labels == c)[0]
            residual = np.array([i for i in members if i not in topk_set])
            if len(residual) > 0:
                lg = exact_cand_logits[residual]
                z_clusters[c] = np.sum(np.exp(lg - lg.max())) * np.exp(lg.max())

        # Build: special + topK (exact) + cluster reps (true Z_c weight)
        active = np.where(cnts_r > 0)[0]
        n_sp = len(special_idx)
        n_topk = len(topk_local)
        n_total = n_sp + n_topk + len(active)

        # Use exp-domain weights directly
        sp_logits = logits[special_idx].astype(np.float64)
        topk_logits = logits[topk_global].astype(np.float64)
        s_max = max(
            sp_logits.max() if n_sp > 0 else -np.inf,
            topk_logits.max() if n_topk > 0 else -np.inf)

        exp_weights = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float64)

        exp_weights[:n_sp] = np.exp(sp_logits - s_max)
        out_vals[:n_sp] = values[special_idx].astype(np.float64)

        off = n_sp
        exp_weights[off:off + n_topk] = np.exp(topk_logits - s_max)
        out_vals[off:off + n_topk] = values[topk_global].astype(np.float64)

        off = n_sp + n_topk
        for i, c in enumerate(active):
            nc = cnts_r[c]
            exp_weights[off + i] = z_clusters[c] * np.exp(-s_max)
            out_vals[off + i] = v_sums[c] / nc

        total = exp_weights.sum()
        w = exp_weights / (total + 1e-30)
        output = (w @ out_vals).astype(np.float32)
        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        topk_frac = cfg.get("topk_frac", 7 / 8)
        m_pq = cfg.get("m_pq", 8)
        return [MQClusterTrueZ(topk_frac=topk_frac, m_pq=m_pq)]


class MQClusterPQZ(MQClusterTopK):
    """PQ top-K + MQ clusters scored by PQ-estimated Z_c."""

    def __init__(self, topk_frac=7/8, m_pq=8):
        super().__init__(topk_frac=topk_frac, m_pq=m_pq)

    @property
    def name(self):
        return "PQTopK+MQClust-pqZ"

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

        b_topk = min(int(budget * self.topk_frac), n_cand)
        n_clust = max(budget - b_topk, 1)
        n_clust = min(n_clust, n_cand)

        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]

        K_z = self._transform(cand_keys)
        rho_cand = (self._rho[candidate_idx]
                    if self._rho is not None
                    else np.ones(n_cand))
        _, labels = _weighted_kmeans(
            K_z, rho_cand, n_clust,
            seed=self._seed, n_iter=50)

        cand_vals_f = cand_vals.astype(np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust)
        cnts = np.bincount(
            labels, minlength=n_clust).astype(np.float64)

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
        pq_cand_logits = approx_ip[candidate_idx].astype(np.float64) / sqrt_d

        # PQ top-K
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

        # Remove top-K from cluster stats
        sel_labels = labels[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=n_clust)
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust).astype(np.float64)
        cnts_r = cnts - sel_cnts

        # PQ Z_c for residual keys per cluster
        topk_set = set(topk_local.tolist())
        z_clusters = np.zeros(n_clust, dtype=np.float64)
        for c in range(n_clust):
            members = np.where(labels == c)[0]
            residual = np.array([i for i in members if i not in topk_set])
            if len(residual) > 0:
                lg = pq_cand_logits[residual]
                z_clusters[c] = np.sum(np.exp(lg - lg.max())) * np.exp(lg.max())

        # Build output
        active = np.where(cnts_r > 0)[0]
        n_sp = len(special_idx)
        n_topk = len(topk_local)
        n_total = n_sp + n_topk + len(active)

        sp_logits = logits[special_idx].astype(np.float64)
        topk_logits = logits[topk_global].astype(np.float64)
        s_max = max(
            sp_logits.max() if n_sp > 0 else -np.inf,
            topk_logits.max() if n_topk > 0 else -np.inf)

        exp_weights = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float64)

        exp_weights[:n_sp] = np.exp(sp_logits - s_max)
        out_vals[:n_sp] = values[special_idx].astype(np.float64)

        off = n_sp
        exp_weights[off:off + n_topk] = np.exp(topk_logits - s_max)
        out_vals[off:off + n_topk] = values[topk_global].astype(np.float64)

        off = n_sp + n_topk
        for i, c in enumerate(active):
            nc = cnts_r[c]
            exp_weights[off + i] = z_clusters[c] * np.exp(-s_max)
            out_vals[off + i] = v_sums[c] / nc

        total = exp_weights.sum()
        w = exp_weights / (total + 1e-30)
        output = (w @ out_vals).astype(np.float32)
        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        topk_frac = cfg.get("topk_frac", 7 / 8)
        m_pq = cfg.get("m_pq", 8)
        return [MQClusterPQZ(topk_frac=topk_frac, m_pq=m_pq)]


class PQFullAttn(AttentionAlgorithm):
    """Full attention over all keys: budget-many exact logits + PQ rest.

    Budget controls how many candidate logits are computed exactly
    (the rest use PQ approximations). All keys participate in softmax.
    """

    def __init__(self, m_pq=8):
        self.m_pq = m_pq
        self._pq = None

    @property
    def name(self):
        return "PQFullAttn"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
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

        # Replace top-budget with exact logits
        k = min(budget, n_cand)
        if k > 0:
            topk_l = np.argpartition(cand_lg, -k)[-k:]
            cand_lg[topk_l] = logits[
                candidate_idx[topk_l]].astype(np.float64)

        # Full softmax: special + all candidates, exact values
        all_lg = np.concatenate([
            logits[special_idx].astype(np.float64), cand_lg])
        all_v = np.vstack([
            values[special_idx].astype(np.float64),
            values[candidate_idx].astype(np.float64)])
        w = softmax(all_lg)
        output = (w @ all_v).astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=len(special_idx) + k)

    @staticmethod
    def expand_from_config(cfg):
        m_pq = cfg.get("m_pq", 8)
        return [PQFullAttn(m_pq=m_pq)]


class PQTopKOnly(AttentionAlgorithm):
    """Pure PQ top-K: no clusters, just subset attention on top-K keys."""

    def __init__(self, m_pq=8):
        self.m_pq = m_pq
        self._pq = None

    @property
    def name(self):
        return "PQTopK-Only"

    @property
    def sweeps_budget(self):
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        self._pq = PQIndex(m=self.m_pq, n_codes=256, seed=seed)
        self._pq.fit(keys)

    def run(self, problem, budget, rng):
        q = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)
        n = len(keys)
        d = values.shape[1]

        if n_cand == 0:
            out = softmax(logits[special_idx]) @ values[special_idx]
            return AttentionOutput(output=out,
                                   actual_budget=len(special_idx))

        b_topk = min(budget, n_cand)
        n_pq = len(self._pq.codes)
        cand_mask = np.zeros(n_pq, dtype=bool)
        cand_mask[candidate_idx] = True
        topk_global = self._pq.approximate_topk(
            q, b_topk, candidate_mask=cand_mask)
        # Filter valid
        g2l = np.full(n_pq, -1, dtype=np.int64)
        g2l[candidate_idx] = np.arange(n_cand)
        topk_local = g2l[topk_global]
        valid = topk_local >= 0
        topk_global = topk_global[valid]

        # Subset attention: special + topK
        all_idx = np.unique(np.concatenate(
            [special_idx, topk_global])).astype(np.int64)
        w = softmax(logits[all_idx].astype(np.float64))
        output = (w @ values[all_idx].astype(np.float64)
                  ).astype(np.float32)
        return AttentionOutput(output=output,
                               actual_budget=len(all_idx))

    @staticmethod
    def expand_from_config(cfg):
        m_pq = cfg.get("m_pq", 8)
        return [PQTopKOnly(m_pq=m_pq)]
