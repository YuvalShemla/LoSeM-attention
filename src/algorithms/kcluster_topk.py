"""
Key-Cluster + TopK hybrid methods.

KMeans on keys → C clusters. Select top-K individual keys
(oracle or PQ approximate), remove from clusters, represent
residual clusters with mean-key scoring (1st order) or with
second-order variance correction.

Four variants from two axes:
  TopK method:  Oracle (exact logits) vs PQ (approximate)
  Scoring:      1st order (mean-key) vs 2nd order (+var/2)
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .pq_topk import PQIndex
from ..core import softmax, cached_flat_kmeans


def _full_index_candidate_mask(index_len: int, candidate_idx: np.ndarray) -> np.ndarray:
    """Mask over a full prepared PQ index, allowing only current causal candidates."""
    mask = np.zeros(index_len, dtype=bool)
    valid = candidate_idx[candidate_idx < index_len]
    mask[valid] = True
    return mask


class KClusterTopK(AttentionAlgorithm):
    """
    KMeans on keys + top-K selection + cluster residuals.

    Parameters:
        n_clusters: number of key clusters
        topk_method: "oracle" (exact logits) or "pq"
        order: 0 (mean-key scoring) or 2 (+ variance/2)
        m_pq: PQ subspaces (only used if topk_method="pq")
    """

    def __init__(self, n_clusters: int = 1024,
                 topk_method: str = "oracle",
                 order: int = 0,
                 m_pq: int = 8,
                 n_codes_pq: int = 256):
        self.n_clusters = n_clusters
        self.topk_method = topk_method
        self.order = order
        self.m_pq = m_pq
        self.n_codes_pq = n_codes_pq
        self._pq = None
        self._seed = 42

        # Precomputed in prepare()
        self._M = None  # [C, d, d] sum of outer products

    @property
    def name(self) -> str:
        topk = "Oracle" if self.topk_method == "oracle" else f"PQ"
        ord_str = "" if self.order == 0 else "-2nd"
        return (
            f"{topk}TopK+KCluster{ord_str}"
            f"-C{self.n_clusters}"
        )

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None,
                seed=42):
        self._seed = seed
        if self.topk_method == "pq":
            self._pq = PQIndex(
                m=self.m_pq, n_codes=self.n_codes_pq, seed=seed,
            )
            self._pq.fit(keys)

    def run(self, problem: AttentionInput, budget: int,
            rng: np.random.Generator) -> AttentionOutput:
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
            return AttentionOutput(
                output=out, actual_budget=len(special_idx),
            )

        # Cluster candidate keys
        n_clust = min(self.n_clusters, n_cand)
        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]

        centroids, labels = cached_flat_kmeans(
            cand_keys, n_clust, seed=self._seed,
        )

        # Precompute cluster stats
        k_sums = np.zeros((n_clust, d), dtype=np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        cnts = np.zeros(n_clust, dtype=np.float64)
        cand_keys_f = cand_keys.astype(np.float64)
        cand_vals_f = cand_vals.astype(np.float64)

        for j in range(d):
            k_sums[:, j] = np.bincount(
                labels, weights=cand_keys_f[:, j],
                minlength=n_clust,
            )
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust,
            )
        cnts = np.bincount(
            labels, minlength=n_clust,
        ).astype(np.float64)

        # For 2nd order: precompute M = Σ k_i k_i^T
        if self.order == 2:
            M = np.zeros((n_clust, d, d), dtype=np.float64)
            for c in range(n_clust):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                ck = cand_keys_f[mask]
                M[c] = ck.T @ ck

        # Select top-K keys
        buse = min(budget, n_cand)

        if self.topk_method == "oracle":
            cand_logits = logits[candidate_idx]
            if buse < n_cand:
                topk_local = np.argpartition(
                    cand_logits, -buse,
                )[-buse:]
            else:
                topk_local = np.arange(n_cand)
            topk_global = candidate_idx[topk_local]
        else:
            cand_mask = _full_index_candidate_mask(
                len(self._pq.codes), candidate_idx,
            )
            topk_global = self._pq.approximate_topk(
                q, buse, candidate_mask=cand_mask,
            )
            g2l = np.full(n_pq, -1, dtype=np.int64)
            g2l[candidate_idx] = np.arange(n_cand)
            topk_local = g2l[topk_global]
            valid = topk_local >= 0
            topk_local = topk_local[valid]
            topk_global = topk_global[valid]
            buse = len(topk_local)

        # Remove selected keys from cluster stats
        sel_labels = labels[topk_local]
        sel_k = cand_keys_f[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            k_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_k[:, j],
                minlength=n_clust,
            )
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=n_clust,
            )
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust,
        ).astype(np.float64)
        cnts_r = cnts - sel_cnts

        if self.order == 2:
            M_r = M.copy()
            for idx in range(buse):
                c = sel_labels[idx]
                M_r[c] -= np.outer(sel_k[idx], sel_k[idx])

        # Build joint softmax
        active = cnts_r > 0
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        n_sp = len(special_idx)
        n_total = n_sp + buse + n_active

        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)
        q64 = q.astype(np.float64)

        # Special
        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]

        # TopK individuals (TRUE logits for attention)
        off = n_sp
        scores[off:off + buse] = (
            logits[topk_global].astype(np.float64)
        )
        out_vals[off:off + buse] = values[topk_global]

        # Residual cluster reps
        off = n_sp + buse
        for i, c in enumerate(active_idx):
            nc = cnts_r[c]
            mean_k = k_sums[c] / nc
            mean_v = (v_sums[c] / nc).astype(np.float32)
            ml = float(q64 @ mean_k) / sqrt_d

            if self.order == 2:
                qMq = float(q64 @ M_r[c] @ q64)
                var_l = qMq / (nc * d) - ml ** 2
                var_l = max(var_l, 0.0)
                scores[off + i] = ml + var_l / 2 + np.log(nc)
            else:
                scores[off + i] = ml + np.log(nc)

            out_vals[off + i] = mean_v

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals

        return AttentionOutput(
            output=output, actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters = cfg.get("n_clusters_sweep", [1024])
        if isinstance(clusters, int):
            clusters = [clusters]
        methods = cfg.get("methods", [
            "oracle", "oracle_2nd", "pq", "pq_2nd",
        ])
        m_pq = cfg.get("m_pq", 8)
        n_codes_pq = int(cfg.get("n_codes_pq", 256))

        instances = []
        for nc in clusters:
            for method in methods:
                if method == "oracle":
                    instances.append(KClusterTopK(
                        n_clusters=nc, topk_method="oracle",
                        order=0,
                    ))
                elif method == "oracle_2nd":
                    instances.append(KClusterTopK(
                        n_clusters=nc, topk_method="oracle",
                        order=2,
                    ))
                elif method == "pq":
                    instances.append(KClusterTopK(
                        n_clusters=nc, topk_method="pq",
                        order=0, m_pq=m_pq,
                        n_codes_pq=n_codes_pq,
                    ))
                elif method == "pq_2nd":
                    instances.append(KClusterTopK(
                        n_clusters=nc, topk_method="pq",
                        order=2, m_pq=m_pq,
                        n_codes_pq=n_codes_pq,
                    ))
        return instances


class OracleClusterPQTopK(AttentionAlgorithm):
    """
    Ablation: Oracle clusters + PQ approximate top-k.

    Sort candidates by TRUE logit, split into C equal-sized
    groups (like IdealEqualSplits). Then use PQ to find
    approximate top-K keys, remove them from their oracle
    groups, and recompute residual mean-key + mean-value.

    This isolates the clustering error: since clusters are
    oracle-optimal, any remaining error comes from the PQ
    top-k approximation and the mean-key representation.
    """

    def __init__(self, n_clusters: int = 1024,
                 m_pq: int = 8,
                 n_codes_pq: int = 256):
        self.n_clusters = n_clusters
        self.m_pq = m_pq
        self.n_codes_pq = n_codes_pq
        self._pq = None
        self._seed = 42

    @property
    def name(self) -> str:
        return f"PQTopK+OracleCluster-C{self.n_clusters}"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None,
                seed=42):
        self._seed = seed
        self._pq = PQIndex(
            m=self.m_pq, n_codes=self.n_codes_pq, seed=seed,
        )
        self._pq.fit(keys)

    def run(self, problem: AttentionInput, budget: int,
            rng: np.random.Generator) -> AttentionOutput:
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
            return AttentionOutput(
                output=out, actual_budget=len(special_idx),
            )

        # Oracle clustering: sort candidates by true logit,
        # split into C equal-sized groups
        n_clust = min(self.n_clusters, n_cand)
        cand_logits = logits[candidate_idx]
        sort_order = np.argsort(cand_logits)[::-1]
        sorted_cand = candidate_idx[sort_order]

        # Assign group labels (0..n_clust-1) to sorted cands
        labels = np.zeros(n_cand, dtype=np.int32)
        group_size = n_cand // n_clust
        remainder = n_cand % n_clust
        pos = 0
        for c in range(n_clust):
            sz = group_size + (1 if c < remainder else 0)
            # Map back from sorted to original candidate order
            sorted_positions = sort_order[pos:pos + sz]
            labels[sorted_positions] = c
            pos += sz

        # Compute per-group stats
        cand_keys = keys[candidate_idx]
        cand_vals = values[candidate_idx]
        cand_keys_f = cand_keys.astype(np.float64)
        cand_vals_f = cand_vals.astype(np.float64)

        k_sums = np.zeros((n_clust, d), dtype=np.float64)
        v_sums = np.zeros((n_clust, d), dtype=np.float64)
        for j in range(d):
            k_sums[:, j] = np.bincount(
                labels, weights=cand_keys_f[:, j],
                minlength=n_clust,
            )
            v_sums[:, j] = np.bincount(
                labels, weights=cand_vals_f[:, j],
                minlength=n_clust,
            )
        cnts = np.bincount(
            labels, minlength=n_clust,
        ).astype(np.float64)

        # PQ approximate top-k
        buse = min(budget, n_cand)
        cand_mask = _full_index_candidate_mask(
            len(self._pq.codes), candidate_idx,
        )
        topk_global = self._pq.approximate_topk(
            q, buse, candidate_mask=cand_mask,
        )
        g2l = np.full(n_pq, -1, dtype=np.int64)
        g2l[candidate_idx] = np.arange(n_cand)
        topk_local = g2l[topk_global]
        valid = topk_local >= 0
        topk_local = topk_local[valid]
        topk_global = topk_global[valid]
        buse = len(topk_local)

        # Remove selected from group stats
        sel_labels = labels[topk_local]
        sel_k = cand_keys_f[topk_local]
        sel_v = cand_vals_f[topk_local]
        for j in range(d):
            k_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_k[:, j],
                minlength=n_clust,
            )
            v_sums[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=n_clust,
            )
        sel_cnts = np.bincount(
            sel_labels, minlength=n_clust,
        ).astype(np.float64)
        cnts_r = cnts - sel_cnts

        # Build joint softmax
        active = cnts_r > 0
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        n_sp = len(special_idx)
        n_total = n_sp + buse + n_active
        q64 = q.astype(np.float64)

        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)

        # Special
        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]

        # TopK individuals
        off = n_sp
        scores[off:off + buse] = (
            logits[topk_global].astype(np.float64)
        )
        out_vals[off:off + buse] = values[topk_global]

        # Residual group reps (1st order only)
        off = n_sp + buse
        for i, c in enumerate(active_idx):
            nc = cnts_r[c]
            mean_k = k_sums[c] / nc
            mean_v = (v_sums[c] / nc).astype(np.float32)
            ml = float(q64 @ mean_k) / sqrt_d
            scores[off + i] = ml + np.log(nc)
            out_vals[off + i] = mean_v

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals

        return AttentionOutput(
            output=output, actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters = cfg.get("n_clusters_sweep", [1024])
        if isinstance(clusters, int):
            clusters = [clusters]
        m_pq = cfg.get("m_pq", 8)
        n_codes_pq = int(cfg.get("n_codes_pq", 256))
        return [
            OracleClusterPQTopK(
                n_clusters=nc,
                m_pq=m_pq,
                n_codes_pq=n_codes_pq,
            )
            for nc in clusters
        ]
