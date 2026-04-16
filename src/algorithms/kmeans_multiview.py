"""
Multi-view mean-only KMeans attention approximation.

Run M independent KMeans clusterings on the same keys
(each with a different seed and low iteration count for
diversity).  Each view gives a mean-only approximation
where every cluster participates in a single softmax
weighted by its count:

    N^(m)(q) = sum_c  n_c^(m) exp(q^T mu_c^(m) / sqrt d) v_bar_c^(m)
    Z^(m)(q) = sum_c  n_c^(m) exp(q^T mu_c^(m) / sqrt d)

Combine across views by averaging numerators and
denominators, then normalise:

    N_multi = (1/M) sum_m  N^(m)
    Z_multi = (1/M) sum_m  Z^(m)
    y       = N_multi / Z_multi

Why multiple views help:
    A single clustering may badly represent some keys
    with their cluster mean.  Under a different partition
    those same keys may land in a tighter bucket with a
    better mean.  Averaging N/Z across views smooths
    over partition-specific errors.

Why numerator/denominator averaging (not output averaging):
    Attention is fundamentally a ratio N/Z.  Averaging at
    that level respects the softmax structure.  Averaging
    final normalised outputs is weaker because
    mean(N^(m)/Z^(m)) != mean(N^(m)) / mean(Z^(m)).

Why NOT one giant softmax over all views' means:
    Every key would be represented M times (once per view)
    inside one softmax, overcounting mass.  The cluster
    means from different views are overlapping summaries
    of the same keys, not disjoint support points.

Offline cost:  M independent KMeans runs on N keys.
Per-query cost: O(M * C * d) scoring, no sorting needed.
"""

import numpy as np
from typing import List, Dict, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import flat_kmeans


# ─── helpers ──────────────────────────────────────────

def _precompute_view_stats(
    keys: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    C: int,
) -> Dict:
    """Per-cluster avg_keys, avg_values, counts, members."""
    d = keys.shape[1]
    d_v = values.shape[1]

    avg_keys = np.zeros((C, d), dtype=np.float32)
    avg_values = np.zeros((C, d_v), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int32)
    member_indices = [None] * C

    for c in range(C):
        mask = labels == c
        count = int(np.sum(mask))
        counts[c] = count
        if count > 0:
            avg_keys[c] = np.mean(keys[mask], axis=0)
            avg_values[c] = np.mean(
                values[mask], axis=0
            )
            member_indices[c] = np.where(mask)[0]
        else:
            member_indices[c] = np.array(
                [], dtype=np.int64
            )

    return {
        "avg_keys": avg_keys,
        "avg_values": avg_values,
        "counts": counts,
        "member_indices": member_indices,
    }


def _causal_counts(
    view_stats: Dict,
    n_causal: int,
    special_set: set,
) -> np.ndarray:
    """
    Recount each cluster after restricting to the causal
    window and removing special tokens.

    Returns int array [C] of causal counts.
    """
    members = view_stats["member_indices"]
    C = len(view_stats["counts"])
    cc = np.zeros(C, dtype=np.int32)

    for c in range(C):
        idx = members[c]
        if len(idx) == 0:
            continue
        idx = idx[idx < n_causal]
        if special_set:
            keep = np.ones(len(idx), dtype=bool)
            for s in special_set:
                keep &= (idx != s)
            idx = idx[keep]
        cc[c] = len(idx)

    return cc


def _mean_only_numden(
    query: np.ndarray,
    head_dim: int,
    avg_keys: np.ndarray,
    avg_values: np.ndarray,
    counts: np.ndarray,
    logits: np.ndarray,
    values: np.ndarray,
    special_idx: np.ndarray,
) -> tuple:
    """
    Compute the mean-only approximate numerator, denominator,
    and log-scale for one view.

    Cluster contribution:
        score_c = q^T mu_c / sqrt(d)
        weight_c = n_c * exp(score_c)
        N += weight_c * v_bar_c
        Z += weight_c

    Special tokens are added with exact logits and count 1.

    Returns (numer, denom, log_scale) where numer and denom
    are computed relative to exp(log_scale) for numerical
    stability.
    """
    sqrt_d = np.sqrt(head_dim)
    d_v = avg_values.shape[1]

    valid = counts > 0
    n_valid = int(np.sum(valid))
    n_special = len(special_idx)

    if n_valid == 0 and n_special == 0:
        return np.zeros(d_v), 0.0, 0.0

    # Build unified score / value / count arrays
    n_items = n_valid + n_special
    scores = np.empty(n_items)
    item_values = np.empty((n_items, d_v))
    item_counts = np.empty(n_items)

    # Cluster items
    if n_valid > 0:
        valid_idx = np.where(valid)[0]
        cluster_scores = (
            avg_keys[valid_idx] @ query / sqrt_d
        )
        scores[:n_valid] = cluster_scores
        item_values[:n_valid] = avg_values[valid_idx]
        item_counts[:n_valid] = counts[valid_idx]

    # Special tokens (exact logit, count = 1)
    if n_special > 0:
        scores[n_valid:] = logits[special_idx]
        item_values[n_valid:] = values[special_idx]
        item_counts[n_valid:] = 1.0

    # Numerically stable weighted sum
    # weight_i = count_i * exp(score_i)
    #          = exp(score_i + log(count_i))
    log_weights = scores + np.log(item_counts)
    log_scale = np.max(log_weights)
    exp_w = np.exp(log_weights - log_scale)

    denom = float(np.sum(exp_w))
    numer = exp_w @ item_values

    return numer, denom, log_scale


# ─── main class ───────────────────────────────────────

class KMeansMultiViewClustering(AttentionAlgorithm):
    """
    Multi-view mean-only KMeans attention.

    Every cluster participates in every view's softmax
    (no top-k selection, no sorting).  Views are combined
    by averaging unnormalised numerators and denominators.

    Parameters
    ----------
    n_views : int
        Number of independent KMeans partitions (M).
    n_clusters : int
        Clusters per view (C).
    n_kmeans_iter : int
        KMeans iterations per view.  Low values (3-5)
        give diverse, under-converged partitions.
    """

    def __init__(
        self,
        n_views: int = 50,
        n_clusters: int = 512,
        n_kmeans_iter: int = 5,
    ):
        self.n_views = n_views
        self.n_clusters = n_clusters
        self.n_kmeans_iter = n_kmeans_iter
        self._view_stats: Optional[List[Dict]] = None

    @property
    def name(self) -> str:
        return (
            f"KMeansMV-V{self.n_views}"
            f"-C{self.n_clusters}"
            f"-it{self.n_kmeans_iter}"
        )

    # ── offline ────────────────────────────────────

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
        Run M independent KMeans on keys.

        Each view uses seed + m and a low iteration count
        to produce a diverse, under-converged partition.
        """
        self._view_stats = []
        for m in range(self.n_views):
            _, labels = flat_kmeans(
                keys, self.n_clusters,
                seed=seed + m,
                n_iter=self.n_kmeans_iter,
            )
            stats = _precompute_view_stats(
                keys, values, labels, self.n_clusters,
            )
            self._view_stats.append(stats)

    # ── online ─────────────────────────────────────

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._view_stats is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)
        d_v = values.shape[1]

        # Collect per-view (numer, denom, log_scale)
        view_nd = []
        for m in range(self.n_views):
            vs = self._view_stats[m]
            cc = _causal_counts(
                vs, n_causal, special_set,
            )
            numer, denom, log_scale = _mean_only_numden(
                q, head_dim,
                vs["avg_keys"], vs["avg_values"], cc,
                logits, values, special_idx,
            )
            view_nd.append((numer, denom, log_scale))

        # Combine across views in a numerically stable way.
        # Each view's numer/denom are relative to
        # exp(log_scale_m).  Align to a global max.
        log_scales = np.array([s for _, _, s in view_nd])
        global_max = np.max(log_scales)

        total_numer = np.zeros(d_v)
        total_denom = 0.0
        for numer, denom, ls in view_nd:
            rescale = np.exp(ls - global_max)
            total_numer += rescale * numer
            total_denom += rescale * denom

        if total_denom < 1e-30:
            output = np.zeros(d_v)
        else:
            output = total_numer / total_denom

        # Budget: C items scored per view * M views
        # + specials (counted once per view)
        n_special = len(special_idx)
        total_budget = self.n_views * (
            self.n_clusters + n_special
        )

        return AttentionOutput(
            output=output,
            actual_budget=total_budget,
        )

    # ── config ─────────────────────────────────────

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        views_list = cfg.get("n_views", [50])
        if isinstance(views_list, int):
            views_list = [views_list]
        clusters_list = cfg.get("n_clusters", [512])
        if isinstance(clusters_list, int):
            clusters_list = [clusters_list]
        iters_list = cfg.get("n_kmeans_iter", [5])
        if isinstance(iters_list, int):
            iters_list = [iters_list]
        for nv in views_list:
            for nc in clusters_list:
                for ni in iters_list:
                    instances.append(
                        KMeansMultiViewClustering(
                            n_views=nv,
                            n_clusters=nc,
                            n_kmeans_iter=ni,
                        )
                    )
        return instances
