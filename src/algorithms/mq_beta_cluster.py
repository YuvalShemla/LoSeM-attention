"""
MQ-Weighted Key Clustering with Fitted beta Bias + TopK.

Improvement over MQClusterTopK:
1. Pre-clusters ALL keys in prepare()
2. Uses ALL train queries for M_Q, rho, and beta fitting
3. Fitted beta_c replaces log(n_c) for better cluster scoring
4. delta_c = beta_c - log(n_c) decomposition handles topK subtraction

At query time, topK keys are subtracted from cluster key/value sums
and the cluster score uses log(n_residual) + delta_c.
"""

import logging
import numpy as np
from .base import AttentionAlgorithm, AttentionOutput
from .pq_topk import PQIndex
from ..core import softmax

log = logging.getLogger(__name__)


def _logsumexp(x, axis=1):
    """Numerically stable logsumexp without scipy."""
    x_max = x.max(axis=axis, keepdims=True)
    return x_max.squeeze(axis) + np.log(
        np.sum(np.exp(x - x_max), axis=axis))


def _weighted_kmeans(data, weights, n_clust, seed=42, n_iter=50):
    """KMeans with per-sample importance weights (KMeans++ init)."""
    rng = np.random.default_rng(seed)
    N, d = data.shape
    n_clust = min(n_clust, N)
    data_f = data.astype(np.float64)
    w = weights.astype(np.float64)

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


class MQBetaCluster(AttentionAlgorithm):
    """MQ-weighted clustering with fitted beta bias + PQ/Oracle TopK."""

    def __init__(self, topk_frac=7/8, m_pq=8, n_clusters=512,
                 oracle_topk=False):
        self.topk_frac = topk_frac
        self.m_pq = m_pq
        self.n_clusters = n_clusters
        self.oracle_topk = oracle_topk

        self._pq = None
        self._eigvecs = None
        self._sqrt_eig = None
        self._all_labels = None
        self._centroids_z = None
        self._mean_keys = None
        self._mean_vals = None
        self._counts = None
        self._beta = None
        self._delta = None
        self._seed = 42

    @property
    def name(self):
        prefix = "OracleTopK" if self.oracle_topk else "PQTopK"
        if self.topk_frac == 0:
            return f"MQBeta-C{self.n_clusters}"
        return f"{prefix}+MQBeta"

    @property
    def sweeps_budget(self):
        return True

    def _transform(self, keys):
        if self._eigvecs is None:
            return keys.astype(np.float32)
        return (keys.astype(np.float64)
                @ self._eigvecs
                * self._sqrt_eig[None, :]).astype(np.float32)

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42,
                train_queries=None):
        self._seed = seed
        N, d = keys.shape
        sqrt_d = np.sqrt(d)
        n_clust = min(self.n_clusters, N)

        if not self.oracle_topk:
            self._pq = PQIndex(m=self.m_pq, n_codes=256, seed=seed)
            self._pq.fit(keys)

        if queries is None or len(queries) < 2:
            self._eigvecs = None
            self._sqrt_eig = None
            self._all_labels = (np.arange(N) % max(n_clust, 1)
                                ).astype(np.int32)
            self._counts = np.bincount(
                self._all_labels, minlength=n_clust
            ).astype(np.float64)
            K_f = keys.astype(np.float64)
            V_f = values.astype(np.float64)
            self._mean_keys = np.zeros((n_clust, d), np.float64)
            self._mean_vals = np.zeros((n_clust, d), np.float64)
            for j in range(d):
                self._mean_keys[:, j] = np.bincount(
                    self._all_labels, weights=K_f[:, j],
                    minlength=n_clust)
                self._mean_vals[:, j] = np.bincount(
                    self._all_labels, weights=V_f[:, j],
                    minlength=n_clust)
            active = self._counts > 0
            self._mean_keys[active] /= self._counts[active, None]
            self._mean_vals[active] /= self._counts[active, None]
            self._beta = np.log(np.maximum(self._counts, 1))
            self._delta = np.zeros(n_clust, np.float64)
            return

        # --- Train/test split for both queries AND keys ---
        test_positions = set(query_positions) if query_positions else set()
        if train_queries is not None:
            Q_train = train_queries.astype(np.float64)
        else:
            q_train_mask = np.array(
                [i not in test_positions for i in range(len(queries))],
                dtype=bool)
            Q_train = queries[q_train_mask].astype(np.float64)
        n_train = len(Q_train)
        k_train_mask = np.array(
            [i not in test_positions for i in range(N)], dtype=bool)
        train_key_idx = np.where(k_train_mask)[0]
        test_key_idx = np.where(~k_train_mask)[0]
        n_train_keys = len(train_key_idx)

        K_f = keys.astype(np.float64)
        V_f = values.astype(np.float64)
        K_train_f = K_f[train_key_idx]
        V_train_f = V_f[train_key_idx]
        BATCH = 500

        import time as _time
        import sys as _sys
        def _prog(msg):
            print(f"[MQBeta] {msg}", flush=True)
        _t0 = _time.time()
        _prog(f"prepare: {n_train} train queries, {n_train_keys} "
              f"train keys ({len(test_positions)} test excl), C={n_clust}")

        # --- 1. M_Q transform from train queries ---
        _prog("[1/5] M_Q eigdecomp...")
        M_Q = Q_train.T @ Q_train + 1e-6 * np.eye(d)
        eigvals, eigvecs = np.linalg.eigh(M_Q)
        eigvals = np.maximum(eigvals, 0.0)
        self._sqrt_eig = np.sqrt(eigvals)
        self._eigvecs = eigvecs
        _prog(f"[1/5] done ({_time.time()-_t0:.0f}s)")

        # --- 2. Importance weights rho from train queries over train keys ---
        _prog(f"[2/5] rho from {n_train} queries...")
        _t1 = _time.time()
        rho_sum = np.zeros(n_train_keys, np.float64)
        for b0 in range(0, n_train, BATCH):
            b1 = min(b0 + BATCH, n_train)
            logits_b = (Q_train[b0:b1] @ K_train_f.T) / sqrt_d
            s_max = logits_b.max(axis=1, keepdims=True)
            rho_sum += np.sum(np.exp(logits_b - s_max), axis=0)
            if (b0 // BATCH) % 30 == 0 and b0 > 0:
                _prog(f"  rho: {b1}/{n_train} ({_time.time()-_t1:.0f}s)")
        _prog(f"[2/5] rho done ({_time.time()-_t1:.0f}s)")

        # --- 3. Weighted KMeans on train keys in M_Q space ---
        _prog(f"[3/5] KMeans {n_train_keys} keys -> {n_clust} clusters...")
        _t2 = _time.time()
        K_z_train = self._transform(keys[train_key_idx])
        self._centroids_z, train_labels = _weighted_kmeans(
            K_z_train, rho_sum, n_clust, seed=seed, n_iter=50)
        _prog(f"[3/5] KMeans done ({_time.time()-_t2:.0f}s)")

        # Assign ALL keys: train from clustering, test to nearest centroid
        self._all_labels = np.empty(N, dtype=np.int32)
        self._all_labels[train_key_idx] = train_labels
        if len(test_key_idx) > 0:
            K_z_test = self._transform(keys[test_key_idx])
            cents_f = self._centroids_z.astype(np.float64)
            test_f = K_z_test.astype(np.float64)
            dists = (np.sum(test_f ** 2, axis=1, keepdims=True)
                     + np.sum(cents_f ** 2, axis=1)[None, :]
                     - 2.0 * (test_f @ cents_f.T))
            self._all_labels[test_key_idx] = np.argmin(
                dists, axis=1).astype(np.int32)

        # --- 4. Cluster stats from train keys only ---
        _prog("[4/5] cluster stats...")
        self._counts = np.bincount(
            train_labels, minlength=n_clust).astype(np.float64)
        self._mean_keys = np.zeros((n_clust, d), np.float64)
        self._mean_vals = np.zeros((n_clust, d), np.float64)
        for j in range(d):
            self._mean_keys[:, j] = np.bincount(
                train_labels, weights=K_train_f[:, j],
                minlength=n_clust)
            self._mean_vals[:, j] = np.bincount(
                train_labels, weights=V_train_f[:, j],
                minlength=n_clust)
        active = self._counts > 0
        self._mean_keys[active] /= self._counts[active, None]
        self._mean_vals[active] /= self._counts[active, None]

        # --- 5. Fit beta_c (no-mask) from train queries over train keys ---
        _prog(f"[5/5] fitting beta ({n_train} queries x "
              f"{n_train_keys} keys, C={n_clust})...")
        _t3 = _time.time()
        sort_idx = np.argsort(train_labels)
        sorted_labels = train_labels[sort_idx]
        boundaries = np.searchsorted(
            sorted_labels, np.arange(n_clust + 1))

        beta_sum = np.zeros(n_clust, np.float64)
        beta_cnt = np.zeros(n_clust, np.float64)

        n_batches = (n_train + BATCH - 1) // BATCH
        for bi, b0 in enumerate(range(0, n_train, BATCH)):
            b1 = min(b0 + BATCH, n_train)
            Q_b = Q_train[b0:b1]
            B = len(Q_b)
            logits_b = (Q_b @ K_train_f.T) / sqrt_d
            pred_b = (Q_b @ self._mean_keys.T) / sqrt_d
            logits_sorted = logits_b[:, sort_idx]
            for c in range(n_clust):
                s, e = int(boundaries[c]), int(boundaries[c + 1])
                if s == e:
                    continue
                cl_logits = logits_sorted[:, s:e]
                y_c = _logsumexp(cl_logits, axis=1)
                beta_sum[c] += np.sum(y_c - pred_b[:, c])
                beta_cnt[c] += B
            if bi % 10 == 0:
                _prog(f"  beta: {b1}/{n_train} queries "
                      f"({_time.time()-_t3:.0f}s)")

        self._beta = np.full(n_clust, -1e30, np.float64)
        for c in range(n_clust):
            if beta_cnt[c] > 0:
                self._beta[c] = beta_sum[c] / beta_cnt[c]
        _prog(f"[5/5] beta done ({_time.time()-_t3:.0f}s)")

        # --- 6. delta_c = beta_c - log(n_c) ---
        log_counts = np.log(np.maximum(self._counts, 1))
        self._delta = self._beta - log_counts

        total_time = _time.time() - _t0
        _prog(f"DONE in {total_time:.0f}s. C={n_clust}, "
              f"delta mean={self._delta[active].mean():.4f} "
              f"std={self._delta[active].std():.4f}")

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

        n_clust = len(self._counts)
        q64 = q.astype(np.float64)

        cand_labels = self._all_labels[candidate_idx]

        # Causal counts and sums per cluster
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

        all_active = np.where(cnts > 0)[0]
        n_all_active = len(all_active)

        # Budget split: clusters first, remainder to topK
        # When topk_frac=0, never use topK (pure clustering mode)
        if self.topk_frac == 0:
            n_topk = 0
            use_clusters = all_active
        elif budget > n_all_active:
            n_topk = min(budget - n_all_active, n_cand)
            use_clusters = all_active
        else:
            n_topk = 0
            prelim = np.full(n_clust, -np.inf, np.float64)
            for c in all_active:
                prelim[c] = (float(q64 @ self._mean_keys[c]) / sqrt_d
                             + self._beta[c])
            top_c = np.argsort(prelim)[-budget:]
            use_clusters = top_c[prelim[top_c] > -np.inf]

        # --- TopK selection ---
        if n_topk > 0:
            if self.oracle_topk:
                cand_logits = logits[candidate_idx]
                if n_topk < n_cand:
                    topk_local = np.argpartition(
                        cand_logits, -n_topk)[-n_topk:]
                else:
                    topk_local = np.arange(n_cand)
                topk_global = candidate_idx[topk_local]
            else:
                n_pq = len(self._pq.codes)
                cand_mask = np.zeros(n_pq, dtype=bool)
                cand_mask[candidate_idx] = True
                topk_global = self._pq.approximate_topk(
                    q, n_topk, candidate_mask=cand_mask)
                g2l = np.full(n_pq, -1, dtype=np.int64)
                g2l[candidate_idx] = np.arange(n_cand)
                topk_local = g2l[topk_global]
                valid = topk_local >= 0
                topk_local = topk_local[valid]
                topk_global = topk_global[valid]
        else:
            topk_local = np.array([], dtype=np.int64)
            topk_global = np.array([], dtype=np.int64)

        # --- Subtract topK from cluster key/value sums and counts ---
        if len(topk_local) > 0:
            sel_labels = cand_labels[topk_local]
            sel_cnts = np.bincount(
                sel_labels, minlength=n_clust).astype(np.float64)
            sel_k = cand_keys_f[topk_local]
            sel_v = cand_vals_f[topk_local]
            for j in range(d):
                k_sums[:, j] -= np.bincount(
                    sel_labels, weights=sel_k[:, j],
                    minlength=n_clust)
                v_sums[:, j] -= np.bincount(
                    sel_labels, weights=sel_v[:, j],
                    minlength=n_clust)
        else:
            sel_cnts = np.zeros(n_clust, np.float64)
        cnts_r = cnts - sel_cnts

        final_active = use_clusters[cnts_r[use_clusters] > 0]

        # --- Build joint softmax ---
        n_sp = len(special_idx)
        n_tk = len(topk_local)
        n_cl = len(final_active)
        n_total = n_sp + n_tk + n_cl

        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)

        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]

        off = n_sp
        if n_tk > 0:
            scores[off:off + n_tk] = logits[topk_global].astype(
                np.float64)
            out_vals[off:off + n_tk] = values[topk_global]

        off = n_sp + n_tk
        for i, c in enumerate(final_active):
            nc_r = cnts_r[c]
            mean_k = k_sums[c] / nc_r
            mean_v = (v_sums[c] / nc_r).astype(np.float32)
            ml = float(q64 @ mean_k) / sqrt_d
            scores[off + i] = ml + np.log(nc_r) + self._delta[c]
            out_vals[off + i] = mean_v

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals

        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        variants = cfg.get("variants")
        if variants:
            instances = []
            for v in variants:
                tf = v.get("topk_frac", cfg.get("topk_frac", 7/8))
                mp = v.get("m_pq", cfg.get("m_pq", 8))
                nc = v.get("n_clusters", cfg.get("n_clusters", 512))
                oracle = v.get("oracle_topk", False)
                instances.append(MQBetaCluster(
                    topk_frac=tf, m_pq=mp,
                    n_clusters=nc, oracle_topk=oracle))
            return instances
        topk_frac = cfg.get("topk_frac", 7 / 8)
        m_pq = cfg.get("m_pq", 8)
        n_clusters = cfg.get("n_clusters", 512)
        return [
            MQBetaCluster(topk_frac=topk_frac, m_pq=m_pq,
                          n_clusters=n_clusters, oracle_topk=False),
            MQBetaCluster(topk_frac=topk_frac, m_pq=m_pq,
                          n_clusters=n_clusters, oracle_topk=True),
        ]


class MQBetaClusterOnly(AttentionAlgorithm):
    """MQBeta in cluster-only mode (no topK). C = budget at each level.

    Computes M_Q and rho once in prepare(), then clusters + fits beta
    per budget on demand (cached).
    """

    def __init__(self):
        self._eigvecs = None
        self._sqrt_eig = None
        self._rho = None
        self._K_train_f = None
        self._V_train_f = None
        self._train_key_idx = None
        self._test_key_idx = None
        self._keys = None
        self._seed = 42
        self._cache = {}  # budget -> (labels_all, delta)

    @property
    def name(self):
        return "MQBeta"

    @property
    def sweeps_budget(self):
        return True

    def _transform(self, keys):
        if self._eigvecs is None:
            return keys.astype(np.float32)
        return (keys.astype(np.float64)
                @ self._eigvecs
                * self._sqrt_eig[None, :]).astype(np.float32)

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        import time as _time
        self._seed = seed
        self._keys = keys
        self._values = values
        self._cache = {}
        N, d = keys.shape
        sqrt_d = np.sqrt(d)

        if queries is None or len(queries) < 2:
            self._eigvecs = None
            self._sqrt_eig = None
            self._rho = None
            self._K_train_f = keys.astype(np.float64)
            self._V_train_f = values.astype(np.float64)
            self._train_key_idx = np.arange(N)
            self._test_key_idx = np.array([], dtype=np.int64)
            return

        _t0 = _time.time()
        test_positions = set(query_positions) if query_positions else set()
        q_train_mask = np.array(
            [i not in test_positions for i in range(len(queries))],
            dtype=bool)
        k_train_mask = np.array(
            [i not in test_positions for i in range(N)], dtype=bool)
        Q_train = queries[q_train_mask].astype(np.float64)
        n_train = len(Q_train)
        self._train_key_idx = np.where(k_train_mask)[0]
        self._test_key_idx = np.where(~k_train_mask)[0]
        K_f = keys.astype(np.float64)
        self._K_train_f = K_f[self._train_key_idx]
        self._V_train_f = values[self._train_key_idx].astype(np.float64)
        BATCH = 500

        def _prog(msg):
            print(f"[MQBeta] {msg}", flush=True)
        _prog(f"prepare: {n_train} train queries, "
              f"{len(self._train_key_idx)} train keys")

        # M_Q
        M_Q = Q_train.T @ Q_train + 1e-6 * np.eye(d)
        eigvals, eigvecs = np.linalg.eigh(M_Q)
        eigvals = np.maximum(eigvals, 0.0)
        self._sqrt_eig = np.sqrt(eigvals)
        self._eigvecs = eigvecs

        # Rho
        _prog(f"rho from {n_train} queries...")
        _t1 = _time.time()
        rho_sum = np.zeros(len(self._train_key_idx), np.float64)
        for b0 in range(0, n_train, BATCH):
            b1 = min(b0 + BATCH, n_train)
            logits_b = (Q_train[b0:b1] @ self._K_train_f.T) / sqrt_d
            s_max = logits_b.max(axis=1, keepdims=True)
            rho_sum += np.sum(np.exp(logits_b - s_max), axis=0)
        self._rho = rho_sum
        # Pre-transform train keys
        self._K_z_train = self._transform(keys[self._train_key_idx])
        # Store Q_train for beta fitting
        self._Q_train = Q_train
        _prog(f"prepare done ({_time.time()-_t0:.0f}s). "
              f"Clustering deferred to run().")

    def _build_for_budget(self, budget, n_keys):
        """Cluster at C=budget, fit beta, cache results."""
        import time as _time
        n_clust = min(budget, len(self._train_key_idx))
        N = n_keys
        d = self._K_train_f.shape[1]
        sqrt_d = np.sqrt(d)

        def _prog(msg):
            print(f"[MQBeta C={n_clust}] {msg}", flush=True)

        _t0 = _time.time()
        _prog("KMeans...")
        centroids_z, train_labels = _weighted_kmeans(
            self._K_z_train, self._rho, n_clust,
            seed=self._seed, n_iter=50)
        _prog(f"KMeans done ({_time.time()-_t0:.0f}s)")

        # Assign all keys: train from clustering, test to nearest centroid
        all_labels = np.empty(N, dtype=np.int32)
        all_labels[self._train_key_idx] = train_labels
        if len(self._test_key_idx) > 0:
            K_z_test = self._transform(
                self._keys[self._test_key_idx])
            cents_f = centroids_z.astype(np.float64)
            test_f = K_z_test.astype(np.float64)
            dists = (np.sum(test_f ** 2, axis=1, keepdims=True)
                     + np.sum(cents_f ** 2, axis=1)[None, :]
                     - 2.0 * (test_f @ cents_f.T))
            all_labels[self._test_key_idx] = np.argmin(
                dists, axis=1).astype(np.int32)

        # Cluster stats from train keys
        counts = np.bincount(
            train_labels, minlength=n_clust).astype(np.float64)
        mean_keys = np.zeros((n_clust, d), np.float64)
        for j in range(d):
            mean_keys[:, j] = np.bincount(
                train_labels, weights=self._K_train_f[:, j],
                minlength=n_clust)
        active = counts > 0
        mean_keys[active] /= counts[active, None]

        # Beta fitting
        _prog("fitting beta...")
        _t1 = _time.time()
        sort_idx = np.argsort(train_labels)
        sorted_labels = train_labels[sort_idx]
        boundaries = np.searchsorted(
            sorted_labels, np.arange(n_clust + 1))
        beta_sum = np.zeros(n_clust, np.float64)
        beta_cnt = np.zeros(n_clust, np.float64)
        BATCH = 500
        n_train = len(self._Q_train)
        for bi, b0 in enumerate(range(0, n_train, BATCH)):
            b1 = min(b0 + BATCH, n_train)
            Q_b = self._Q_train[b0:b1]
            B = len(Q_b)
            logits_b = (Q_b @ self._K_train_f.T) / sqrt_d
            pred_b = (Q_b @ mean_keys.T) / sqrt_d
            logits_sorted = logits_b[:, sort_idx]
            for c in range(n_clust):
                s, e = int(boundaries[c]), int(boundaries[c + 1])
                if s == e:
                    continue
                y_c = _logsumexp(logits_sorted[:, s:e], axis=1)
                beta_sum[c] += np.sum(y_c - pred_b[:, c])
                beta_cnt[c] += B
            if bi % 20 == 0 and bi > 0:
                _prog(f"  beta: {b1}/{n_train} "
                      f"({_time.time()-_t1:.0f}s)")

        beta = np.full(n_clust, -1e30, np.float64)
        for c in range(n_clust):
            if beta_cnt[c] > 0:
                beta[c] = beta_sum[c] / beta_cnt[c]
        log_counts = np.log(np.maximum(counts, 1))
        delta = beta - log_counts
        _prog(f"done ({_time.time()-_t0:.0f}s total)")

        self._cache[budget] = (all_labels, delta)
        return all_labels, delta

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

        if budget in self._cache:
            all_labels, delta = self._cache[budget]
        else:
            all_labels, delta = self._build_for_budget(
                budget, len(self._keys))

        n_clust = int(all_labels.max()) + 1
        q64 = q.astype(np.float64)
        cand_labels = all_labels[candidate_idx]
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

        off = n_sp
        for i, c in enumerate(active):
            nc = cnts[c]
            mk = k_sums[c] / nc
            mv = v_sums[c] / nc
            scores[off + i] = (float(q64 @ mk) / sqrt_d
                                + np.log(nc) + delta[c])
            out_vals[off + i] = mv

        w = softmax(scores).astype(np.float32)
        output = (w @ out_vals.astype(np.float32))

        return AttentionOutput(output=output, actual_budget=n_total)

    @staticmethod
    def expand_from_config(cfg):
        return [MQBetaClusterOnly()]
