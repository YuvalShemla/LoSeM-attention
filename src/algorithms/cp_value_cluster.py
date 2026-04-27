"""
Cross-Polytope LSH + Value Cluster Compensation.

Two methods compared to baseline CP-SNIS:

A) "snis_residual": Standard SNIS output, then add
   residual cluster value sums (cluster_sum - retrieved
   values) weighted by the minimum SNIS weight among
   retrieved keys. Conservative: the weakest retrieved
   key sets the scale for missed contributions.

B) "subset_residual": Subset softmax (no IS correction)
   over retrieved + special keys, then add residual
   cluster value sums weighted by the minimum subset
   weight. Like TopK + cluster correction.

Both only add clusters where we actually retrieved at
least one key. Special tokens are never clustered.
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .lsh_crosspoly_snis import LSHCrossPolySNIS
from ..core import (
    softmax, cached_flat_kmeans, snis_attention,
    subset_attention,
)


class CPValueCluster(AttentionAlgorithm):
    """
    CP-LSH + value cluster compensation for missed keys.
    """

    _next_id = 0

    def __init__(
        self,
        n_rotations: int = 1,
        L: int = 100,
        min_hits: int = 2,
        center_keys: bool = True,
        n_clusters: int = 1024,
        variant: str = "snis_residual",
    ):
        self._m = n_rotations
        self._L = L
        self._min_hits = min_hits
        self._center_keys = center_keys
        self.n_clusters = n_clusters
        self.variant = variant
        self._id = CPValueCluster._next_id
        CPValueCluster._next_id += 1
        self._lsh = None
        self._seed = 42

    @property
    def name(self) -> str:
        tag = "SR" if self.variant == "snis_residual" else "SubR"
        return f"CPVC-{tag}-{self._id}"

    @property
    def point_label(self) -> str:
        d = 128
        if self._lsh is not None and self._lsh._d:
            d = self._lsh._d
        C = 2 * self._m * d
        return f"C{C}/L{self._L}"

    @property
    def sweeps_budget(self) -> bool:
        return False

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: Optional[np.ndarray] = None,
        query_positions: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        self._seed = seed
        self._lsh = LSHCrossPolySNIS(
            n_rotations=self._m,
            L=self._L,
            min_hits=self._min_hits,
            center_keys=self._center_keys,
        )
        self._lsh.prepare(
            keys, values, head_dim,
            queries=queries,
            query_positions=query_positions,
            seed=seed,
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        query = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)
        d_v = values.shape[1]

        if n_cand == 0:
            out = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=out,
                actual_budget=len(special_idx),
            )

        # ── CP-LSH retrieval ──
        lsh = self._lsh
        q_labels = lsh._hash_query(query)
        cand_hash = lsh._key_labels[candidate_idx]
        matches = cand_hash == q_labels[np.newaxis, :]
        match_counts = np.sum(matches, axis=1)
        retrieved_mask = match_counts >= self._min_hits
        retrieved_local = np.where(retrieved_mask)[0]

        if len(retrieved_local) == 0:
            out = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=out,
                actual_budget=len(special_idx),
            )

        retrieved_idx = candidate_idx[retrieved_local]
        n_ret = len(retrieved_idx)
        n_sp = len(special_idx)

        # ── Value clusters (offline, cached) ──
        n_clust = max(1, min(self.n_clusters, n_cand))
        cand_v = values[candidate_idx].astype(np.float64)
        _, labels = cached_flat_kmeans(
            values[candidate_idx], n_clust,
            seed=self._seed,
        )

        # Per-cluster value sum
        cluster_v_sum = np.zeros(
            (n_clust, d_v), dtype=np.float64,
        )
        for j in range(d_v):
            cluster_v_sum[:, j] = np.bincount(
                labels, weights=cand_v[:, j],
                minlength=n_clust,
            )

        # Subtract retrieved values from their clusters
        ret_labels = labels[retrieved_local]
        ret_v = cand_v[retrieved_local]
        residual_v_sum = cluster_v_sum.copy()
        for j in range(d_v):
            residual_v_sum[:, j] -= np.bincount(
                ret_labels, weights=ret_v[:, j],
                minlength=n_clust,
            )

        if self.variant == "snis_residual":
            # ── A: SNIS + per-cluster residual ──
            if self._center_keys:
                q_c = (
                    query.astype(np.float64)
                    - lsh._key_mean.astype(np.float64)
                )
                k_c = (
                    keys[retrieved_idx].astype(np.float64)
                    - lsh._key_mean.astype(np.float64)
                )
            else:
                q_c = query.astype(np.float64)
                k_c = keys[retrieved_idx].astype(np.float64)

            q_norm = np.linalg.norm(q_c)
            k_norms = np.linalg.norm(k_c, axis=1)
            cos_sims = (
                (k_c @ q_c)
                / (q_norm * k_norms + 1e-10)
            )
            u = lsh._compute_inclusion_prob(
                cos_sims,
            ).astype(np.float64)

            # SNIS weights
            scores = np.empty(
                n_sp + n_ret, dtype=np.float64,
            )
            scores[:n_sp] = logits[special_idx].astype(
                np.float64,
            )
            log_u = np.log(np.maximum(u, 1e-30))
            scores[n_sp:] = (
                logits[retrieved_idx].astype(np.float64)
                - log_u
            )
            w = softmax(scores)
            w_ret = w[n_sp:]

            # SNIS output
            snis_out = snis_attention(
                logits=logits[retrieved_idx],
                values=values[retrieved_idx],
                inclusion_probs=u,
                special_logits=logits[special_idx],
                special_values=values[special_idx],
            ).astype(np.float64)

            # Global min weight for all residual clusters
            min_w = w_ret.min()
            correction = (
                min_w * residual_v_sum.sum(axis=0)
            )

            output = snis_out + correction.astype(
                np.float32,
            )

        elif self.variant == "subset_residual":
            # ── B: Subset attention + per-cluster residual ──
            all_idx = np.concatenate([
                special_idx, retrieved_idx,
            ]).astype(np.int64)

            sub_logits = logits[all_idx]
            w = softmax(sub_logits.astype(np.float64))
            w_ret = w[n_sp:]

            sub_out = (
                w[:, None]
                * values[all_idx].astype(np.float64)
            ).sum(axis=0)

            global_min_w = w_ret.min()
            correction = np.zeros(d_v, dtype=np.float64)
            for c in range(n_clust):
                mask_c = ret_labels == c
                if mask_c.any():
                    min_w_c = w_ret[mask_c].min()
                else:
                    min_w_c = global_min_w
                correction += min_w_c * residual_v_sum[c]

            output = (
                sub_out + correction
            ).astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=n_sp + n_ret,
            selected_indices=np.concatenate([
                special_idx, retrieved_idx,
            ]),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        center = cfg.get("center_keys", True)
        min_hits = cfg.get("min_hits", 2)
        n_clusters = cfg.get("n_clusters", 1024)
        m_values = cfg.get("n_rotations", [1])
        L_sweep = cfg.get(
            "L_sweep", [50, 100, 150, 200],
        )
        variants = cfg.get(
            "variants",
            ["snis_residual", "subset_residual"],
        )

        for m in m_values:
            for L in L_sweep:
                for var in variants:
                    instances.append(CPValueCluster(
                        n_rotations=m, L=L,
                        min_hits=min_hits,
                        center_keys=center,
                        n_clusters=n_clusters,
                        variant=var,
                    ))
        return instances
