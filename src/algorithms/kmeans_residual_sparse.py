"""
Sparse top-k KMeans attention with residual correction.

Unlike the full coarse-to-fine version (kmeans_residual.py)
which includes ALL clusters (top-k refined, rest mean-only),
this variant includes ONLY the top-k scoring clusters.

As k increases, more clusters (and more attention mass)
enter the approximation. This is closer to sparse attention
/ top-k enumeration, but with each selected cluster
represented as a residual mixture rather than individual
tokens.

Offline: same as KMeansResidualClustering.
Query-time:
1. Score all coarse clusters: q^T mu_c / √d + log(n_c).
2. Select top-k clusters.
3. For each selected cluster, include its full residual
   mixture (or mean-only if r=0).
4. Softmax over: special tokens + selected cluster items.
5. Non-selected clusters are excluded entirely.

Cost: O(C·d) scoring + O(k·R·d) refinement.
Memory per query: O(k·R) items in softmax.
"""

import numpy as np
from typing import List, Dict, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, flat_kmeans
from .kmeans_residual import (
    _precompute_coarse_stats,
    _precompute_residual_stats,
    _causal_filter,
)


class KMeansResidualSparse(AttentionAlgorithm):
    """
    Sparse top-k clustered attention with residual
    correction.

    Only the top-k highest-scoring coarse clusters are
    included in the approximation. Each selected cluster
    is represented by its residual subcluster mixture
    (or a single count-corrected mean if n_residuals=0).

    As top_k increases, more attention mass enters the
    approximation — the budget grows from special-only
    to special + k*R items.

    Comparison with KMeansResidualClustering:
    - That method: all C clusters contribute (top-k
      refined, rest as mean). Budget ~ C + k*(R-1).
    - This method: only top-k contribute. Budget ~ k*R.
      Sparser, but misses tail mass from excluded
      clusters.
    """

    def __init__(
        self,
        n_clusters: int = 512,
        n_residuals: int = 8,
        top_k: int = 16,
    ):
        self.n_clusters = n_clusters
        self.n_residuals = n_residuals
        self.top_k = top_k
        self._coarse = None
        self._residuals = None

    @property
    def name(self) -> str:
        return (
            f"KMeansResSp-C{self.n_clusters}"
            f"-r{self.n_residuals}"
            f"-{self.top_k}"
        )

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Coarse KMeans + per-cluster residual KMeans.

        Position 0 (sink token) excluded from clustering.
        """
        k_nosink = keys[1:]
        v_nosink = values[1:]
        self._key_mean = np.mean(k_nosink, axis=0)
        k_centered = k_nosink - self._key_mean

        _, labels = flat_kmeans(
            k_centered, self.n_clusters, seed=seed,
        )

        coarse = _precompute_coarse_stats(
            k_centered, v_nosink, labels,
            self.n_clusters,
        )
        for c in range(self.n_clusters):
            coarse["member_indices"][c] = (
                coarse["member_indices"][c] + 1
            )

        self._coarse = coarse
        keys_centered = keys.copy()
        keys_centered -= self._key_mean
        self._residuals = _precompute_residual_stats(
            keys_centered, values, self._coarse,
            self.n_residuals, seed=seed,
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._coarse is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query
        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)
        sqrt_d = np.sqrt(head_dim)

        # Center keys to match prepare()
        keys_c = keys - self._key_mean
        d_v = values.shape[1]

        # ── Causal filtering (recomputes active stats) ──
        cc, c_mk, c_sv, vm, fres = _causal_filter(
            self._coarse, self._residuals,
            n_causal, special_set, keys_c, values,
        )

        valid = np.where(vm)[0]
        n_special = (
            len(special_idx)
            if special_idx is not None else 0
        )

        if len(valid) == 0 and n_special == 0:
            return AttentionOutput(
                output=np.zeros(d_v, dtype=np.float32),
                actual_budget=0,
            )

        # ── Build (score, value) items for softmax ──
        all_scores = []
        all_values = []

        # Special tokens: exact
        if n_special > 0:
            sp_logits = (
                q @ keys_c[special_idx].T / sqrt_d
            )
            all_scores.append(sp_logits)
            all_values.append(
                values[special_idx].astype(np.float64)
            )

        # Score, rank, and take ONLY top-k clusters
        if len(valid) > 0:
            mu_logits = (
                c_mk[valid] @ q / sqrt_d
            )
            rank_scores = mu_logits + np.log(
                cc[valid].astype(np.float64),
            )
            order = np.argsort(rank_scores)[::-1]
            sorted_valid = valid[order]
            sorted_mu = mu_logits[order]

            top_k = min(
                self.top_k, len(sorted_valid),
            )

            for i in range(top_k):
                c = sorted_valid[i]
                mu_l = sorted_mu[i]

                # Use residual subclusters if available
                if fres[c] is not None:
                    rho = fres[c]["centroids"]
                    r_cnt = fres[c]["counts"]
                    r_sv = fres[c]["sum_values"]

                    active = r_cnt > 0
                    if np.any(active):
                        rho_a = rho[active]
                        cnt_a = r_cnt[active].astype(
                            np.float64,
                        )
                        sv_a = r_sv[active]

                        r_logits = (
                            mu_l
                            + rho_a @ q / sqrt_d
                            + np.log(cnt_a)
                        )
                        avg_v = sv_a / cnt_a[:, None]

                        all_scores.append(r_logits)
                        all_values.append(avg_v)
                        continue

                # Mean-only for this cluster
                n_c = float(cc[c])
                if n_c > 0:
                    score = mu_l + np.log(n_c)
                    avg_v = c_sv[c] / n_c
                    all_scores.append(
                        np.array([score]),
                    )
                    all_values.append(
                        avg_v[None, :],
                    )

        if not all_scores:
            return AttentionOutput(
                output=np.zeros(d_v, dtype=np.float32),
                actual_budget=0,
            )

        scores = np.concatenate(all_scores)
        vals = np.vstack(all_values)

        w = softmax(scores)
        output = (w @ vals).astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=len(scores),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        C = cfg.get("n_clusters", 512)
        for r in cfg.get(
            "n_residuals_sweep", [0, 2, 8, 16],
        ):
            for k in cfg.get(
                "top_k_sweep",
                [1, 2, 4, 8, 16, 32, 64, 128, 256],
            ):
                instances.append(
                    KMeansResidualSparse(
                        n_clusters=C,
                        n_residuals=r,
                        top_k=k,
                    )
                )
        return instances
