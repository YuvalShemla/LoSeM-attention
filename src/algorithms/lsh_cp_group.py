"""
Cross-polytope grouping variant that enumerates all non-empty buckets.
"""

import warnings

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .lsh_crosspoly_multiprobe import (
    _random_orthogonal, crosspolytope_bucket_labels, _cp_vertex_scores,
)
from ..core import softmax


def _bucket_stats(keys, values, labels, corr, n_buckets):
    """Per-bucket means and correction mass."""
    d_k = keys.shape[1]
    d_v = values.shape[1]
    counts = np.bincount(
        labels.astype(np.intp), minlength=n_buckets,
    ).astype(np.int32)[:n_buckets]

    sum_k = np.zeros((n_buckets, d_k), dtype=np.float64)
    sum_v_corr = np.zeros((n_buckets, d_v), dtype=np.float64)
    corr_mass = np.zeros(n_buckets, dtype=np.float64)
    np.add.at(sum_k, labels, keys.astype(np.float64))
    np.add.at(
        sum_v_corr, labels,
        values.astype(np.float64) * corr[:, None],
    )
    np.add.at(corr_mass, labels, corr.astype(np.float64))

    nonempty = counts > 0
    sum_k[nonempty] /= counts[nonempty, np.newaxis]
    safe_mass = np.maximum(corr_mass, 1e-30)
    sum_v_corr[nonempty] /= safe_mass[nonempty, np.newaxis]

    return (
        sum_k.astype(np.float32),
        sum_v_corr.astype(np.float32),
        corr_mass.astype(np.float32),
        counts,
    )


class LSHCPGroup(AttentionAlgorithm):
    """All-buckets CP grouping variant (your version)."""

    def __init__(
        self,
        importance_sampling: bool = False,
        random_bucketing: bool = False,
        random_bucket_size: int = 1,
        name_suffix: str = "",
    ):
        self._importance_sampling = importance_sampling
        self._random_bucketing = random_bucketing
        self._random_bucket_size = max(1, int(random_bucket_size))
        self._name_suffix = name_suffix
        self._avg_keys = None
        self._avg_values = None
        self._counts = None
        self._R1 = None
        self._R2 = None
        self._key_mean = None
        self._query_mean = None
        self._d = None
        self._n_cp = None
        self._sink_b = None
        self._n_buckets = None
        self._labels = None

    @property
    def name(self) -> str:
        base = "LSH-CP-Group"
        parts = ["allB"]
        if not self._importance_sampling:
            parts.append("noIS")
        if self._random_bucketing:
            parts.append(f"rand{self._random_bucket_size}")
        tag = "-".join(parts)
        if self._name_suffix:
            tag = f"{tag}-{self._name_suffix}"
        return f"{base}-{tag}"

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
        d = head_dim
        self._d = d
        n_cp = 2 * d
        self._n_cp = n_cp
        if self._random_bucketing:
            n_non_sink = max(0, len(keys) - 1)
            n_rand = int(np.ceil(
                n_non_sink / self._random_bucket_size
            ))
            n_buckets = n_rand + 1
            sink_b = n_rand
        else:
            n_buckets = n_cp * n_cp + 1
            sink_b = n_cp * n_cp
        self._sink_b = sink_b
        self._n_buckets = n_buckets
        rng = np.random.default_rng(seed)
        self._R1 = _random_orthogonal(d, rng)
        self._R2 = _random_orthogonal(d, rng)

        if queries is None or len(queries) == 0:
            raise ValueError(
                "LSHCPGroup.prepare requires non-empty queries."
            )
        self._query_mean = np.zeros(d, dtype=np.float32)

        if len(keys) == 0:
            self._avg_keys = np.zeros((n_buckets, d), dtype=np.float32)
            self._avg_values = np.zeros(
                (n_buckets, values.shape[1]), dtype=np.float32,
            )
            self._counts = np.zeros(n_buckets, dtype=np.int32)
            self._key_mean = np.zeros(d, dtype=np.float32)
            self._labels = np.array([], dtype=np.int64)
            return

        key_mean = np.mean(keys, axis=0, dtype=np.float64).astype(np.float32)
        self._key_mean = key_mean
        x64 = (keys.astype(np.float32) - key_mean).astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z1 = (x64 @ self._R1.astype(np.float64).T).astype(np.float32)
            z2 = (x64 @ self._R2.astype(np.float64).T).astype(np.float32)

        n = len(keys)
        labels = np.empty(n, dtype=np.int64)
        labels[0] = sink_b
        if n > 1:
            if self._random_bucketing:
                ns = np.arange(1, n, dtype=np.int64)
                perm = rng.permutation(ns)
                rb = np.arange(len(ns), dtype=np.int64)
                rb //= self._random_bucket_size
                labels[perm] = rb
            else:
                b1 = crosspolytope_bucket_labels(z1[1:])
                b2 = crosspolytope_bucket_labels(z2[1:])
                labels[1:] = b1 * n_cp + b2
        self._labels = labels.copy()

        corr = np.exp(
            (
                keys.astype(np.float64)
                @ self._query_mean.astype(np.float64)
            ) / np.sqrt(d, dtype=np.float64)
        ).astype(np.float32)
        self._avg_keys, self._avg_values, _, self._counts = _bucket_stats(
            keys, values, labels, corr, n_buckets,
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._avg_keys is None:
            raise RuntimeError("Call prepare() before run()")

        q = problem.query.astype(np.float32)
        q_centered = q - self._query_mean
        d = self._d
        n_cp = self._n_cp
        sqrt_d = np.sqrt(d, dtype=np.float64)
        n_causal = len(problem.keys)
        special_idx = (
            problem.special_idx
            if problem.special_idx is not None
            else np.array([], dtype=np.int64)
        )
        exact_idx = special_idx.astype(np.int64)
        if exact_idx.size > 0:
            exact_idx = np.unique(exact_idx[exact_idx < n_causal])

        labels = self._labels[:n_causal]
        keep = np.ones(n_causal, dtype=bool)
        if exact_idx.size > 0:
            keep[exact_idx] = False

        grouped_causal_idx = np.where(keep)[0].astype(np.int64)
        g_keys = problem.keys[keep]
        g_vals = problem.values[keep]
        g_labels = labels[keep]

        if len(g_keys) > 0:
            g_corr = np.exp(
                (
                    g_keys.astype(np.float64)
                    @ self._query_mean.astype(np.float64)
                ) / np.sqrt(d, dtype=np.float64)
            ).astype(np.float32)
            g_avg_k, g_avg_v, g_mass, g_counts = _bucket_stats(
                g_keys, g_vals, g_labels, g_corr, self._n_buckets
            )
            bucket_ne = np.where(g_counts > 0)[0]
        else:
            g_avg_k = np.zeros((self._n_buckets, d), dtype=np.float32)
            g_avg_v = np.zeros(
                (self._n_buckets, problem.values.shape[1]),
                dtype=np.float32,
            )
            g_mass = np.zeros(self._n_buckets, dtype=np.float32)
            bucket_ne = np.array([], dtype=np.int64)

        if len(bucket_ne) > 0:
            if self._random_bucketing:
                pi_comb = np.full(
                    len(bucket_ne), 1.0 / len(bucket_ne), dtype=np.float64
                )
            else:
                q_c = q_centered - self._key_mean
                z1_q = (self._R1 @ q_c).astype(np.float64)
                z2_q = (self._R2 @ q_c).astype(np.float64)
                pi1 = softmax(_cp_vertex_scores(z1_q))
                pi2 = softmax(_cp_vertex_scores(z2_q))
                valid_cp = bucket_ne < (n_cp * n_cp)
                cp_ids = bucket_ne[valid_cp]
                cp_pi = np.zeros(len(bucket_ne), dtype=np.float64)
                if len(cp_ids) > 0:
                    b1_idx = cp_ids // n_cp
                    b2_idx = cp_ids % n_cp
                    cp_pi[valid_cp] = pi1[b1_idx] * pi2[b2_idx]
                cp_pi[~valid_cp] = 1e-30
                order = np.argsort(cp_pi)[::-1]
                bucket_ne = bucket_ne[order]
                pi_comb = cp_pi[order]
        else:
            pi_comb = np.empty(0, dtype=np.float64)

        # Enumerate all non-empty buckets.
        probed = bucket_ne
        pi_probed = pi_comb
        n_probe = len(probed)

        if exact_idx.size == 0 and n_probe == 0:
            return AttentionOutput(
                output=np.zeros(problem.values.shape[1], dtype=np.float32),
                actual_budget=0,
            )

        exact_scores = np.array([], dtype=np.float64)
        exact_vals = np.zeros((0, problem.values.shape[1]), dtype=np.float32)
        if exact_idx.size > 0:
            if problem.logits is not None:
                exact_scores = problem.logits[exact_idx].astype(np.float64)
            else:
                exact_scores = (
                    problem.keys[exact_idx] @ q
                ).astype(np.float64) / sqrt_d
            exact_vals = problem.values[exact_idx]

        grp_scores = np.array([], dtype=np.float64)
        grp_vals = np.zeros((0, problem.values.shape[1]), dtype=np.float32)
        grp_log_pi = np.zeros(0, dtype=np.float64)
        if n_probe > 0:
            mk = g_avg_k[probed]
            mv = g_avg_v[probed]
            mass = np.maximum(g_mass[probed].astype(np.float64), 1e-30)
            grp_scores = (
                (mk @ q_centered).astype(np.float64) / sqrt_d
            ) + np.log(mass)
            grp_vals = mv
            if self._importance_sampling:
                grp_log_pi = np.log(np.maximum(pi_probed, 1e-30))
            else:
                grp_log_pi = np.zeros(n_probe, dtype=np.float64)

        all_scores = np.concatenate([exact_scores, grp_scores - grp_log_pi])
        all_vals = np.concatenate([exact_vals, grp_vals], axis=0)
        w = softmax(all_scores).astype(np.float32)
        output = w @ all_vals

        grouped_member_indices = []
        if n_probe > 0 and len(grouped_causal_idx) > 0:
            for b in probed:
                members = grouped_causal_idx[g_labels == b]
                if len(members) > 0:
                    grouped_member_indices.append(members)

        return AttentionOutput(
            output=output,
            actual_budget=int(len(exact_idx) + n_probe),
            grouped_member_indices=grouped_member_indices,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        imp = cfg.get("importance_sampling", False)
        rand_b = cfg.get("random_bucketing", False)
        rand_sz = cfg.get("random_bucket_size", 1)
        return [
            LSHCPGroup(
                importance_sampling=imp,
                random_bucketing=rand_b,
                random_bucket_size=rand_sz,
            ),
        ]
