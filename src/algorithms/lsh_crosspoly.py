"""
LSH Cross-Polytope: 2 independent CP hashes, multi-probe query.

Offline: center keys, apply two random rotations R1/R2,
cross-polytope hash giving (2d)^2 non-sink buckets + 1 sink.
Compute mean key/value per bucket.

Query: hash q, rank buckets by collision probability
(multi-probe LSH), probe top `budget`, importance-weighted
softmax for unbiased estimation.
"""

import warnings

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax


def _random_orthogonal(
    d: int, rng: np.random.Generator,
) -> np.ndarray:
    """Haar-distributed orthogonal matrix via QR."""
    a = rng.standard_normal((d, d)).astype(np.float32)
    q, _ = np.linalg.qr(a, mode="reduced")
    return q.astype(np.float32)


def crosspolytope_bucket_labels(Z: np.ndarray) -> np.ndarray:
    """
    Z: [n, d] -> bucket ids in {0, ..., 2d-1}.
    Assigns each row to the nearest cross-polytope vertex.
    """
    abs_z = np.abs(Z)
    i = np.argmax(abs_z, axis=1).astype(np.int64)
    n = len(Z)
    zi = Z[np.arange(n, dtype=np.int64), i]
    sign_bit = (zi < 0).astype(np.int64)
    return (2 * i + sign_bit).astype(np.int64)


def _cp_vertex_scores(z: np.ndarray) -> np.ndarray:
    """
    Cross-polytope vertex dot-products for z in R^d.
    Returns 2d values: score[2j] = z[j], score[2j+1] = -z[j].
    """
    d = len(z)
    s = np.empty(2 * d, dtype=z.dtype)
    s[0::2] = z
    s[1::2] = -z
    return s


def _bucket_stats(
    keys, values, labels, corr, n_buckets,
):
    """
    Per-bucket stats with query-mean correction.

    - avg_keys[b] = mean of raw keys in bucket b
    - avg_values[b] = corr-weighted mean value in bucket b
    - corr_mass[b] = sum_i corr_i in bucket b
    - counts[b] = number of keys in bucket b
    """
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

    # corr-weighted mean values: sum(c_i v_i) / sum(c_i)
    safe_mass = np.maximum(corr_mass, 1e-30)
    sum_v_corr[nonempty] /= safe_mass[nonempty, np.newaxis]

    return (
        sum_k.astype(np.float32),
        sum_v_corr.astype(np.float32),
        corr_mass.astype(np.float32),
        counts,
    )


class LSHCrossPolytope(AttentionAlgorithm):
    """
    2 iid cross-polytope hashes -> (2d)^2 + 1 buckets.
    Sink (position 0) is always its own bucket.
    Multi-probe query ordering with importance-weighted softmax.
    """

    def __init__(self, name_suffix: str = ""):
        self._name_suffix = name_suffix
        self._avg_keys = None
        self._avg_values = None
        self._corr_mass = None
        self._counts = None
        self._R1 = None
        self._R2 = None
        self._key_mean = None
        self._query_mean = None
        self._d = None
        self._n_cp = None

    @property
    def name(self) -> str:
        base = "LSH-CrossPoly"
        if self._name_suffix:
            return f"{base}-{self._name_suffix}"
        return base

    @property
    def sweeps_budget(self) -> bool:
        return True

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
        n_buckets = n_cp * n_cp + 1
        sink_b = n_cp * n_cp
        rng = np.random.default_rng(seed)

        self._R1 = _random_orthogonal(d, rng)
        self._R2 = _random_orthogonal(d, rng)

        if queries is None or len(queries) == 0:
            raise ValueError(
                "LSHCrossPolytope.prepare requires non-empty queries "
                "to compute mean_q"
            )
        self._query_mean = np.mean(
            queries, axis=0, dtype=np.float64,
        ).astype(np.float32)

        if len(keys) == 0:
            self._avg_keys = np.zeros(
                (n_buckets, d), dtype=np.float32,
            )
            self._avg_values = np.zeros(
                (n_buckets, values.shape[1]),
                dtype=np.float32,
            )
            self._corr_mass = np.zeros(
                n_buckets, dtype=np.float32,
            )
            self._counts = np.zeros(
                n_buckets, dtype=np.int32,
            )
            self._key_mean = np.zeros(
                d, dtype=np.float32,
            )
            self._query_mean = np.zeros(
                d, dtype=np.float32,
            )
            return

        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        self._key_mean = key_mean
        x_c = keys.astype(np.float32) - key_mean
        R1 = self._R1.astype(np.float64)
        R2 = self._R2.astype(np.float64)
        x64 = x_c.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z1 = (x64 @ R1.T).astype(np.float32)
            z2 = (x64 @ R2.T).astype(np.float32)

        n = len(keys)
        labels = np.empty(n, dtype=np.int64)
        labels[0] = sink_b
        if n > 1:
            b1 = crosspolytope_bucket_labels(z1[1:])
            b2 = crosspolytope_bucket_labels(z2[1:])
            labels[1:] = b1 * n_cp + b2

        # Query-centering correction factor:
        # exp(q·k/sqrt(d)) = exp((q-mean_q)·k/sqrt(d))
        #                   * exp(mean_q·k/sqrt(d)).
        # We absorb exp(mean_q·k/sqrt(d)) into values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.exp(
                (
                    keys.astype(np.float64)
                    @ self._query_mean.astype(np.float64)
                ) / np.sqrt(d, dtype=np.float64)
            ).astype(np.float32)
        (self._avg_keys,
         self._avg_values,
         self._corr_mass,
         self._counts) = _bucket_stats(
            keys, values, labels, corr, n_buckets,
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._avg_keys is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query.astype(np.float32)
        q_centered = q - self._query_mean
        d = self._d
        n_cp = self._n_cp
        sqrt_d = np.sqrt(d, dtype=np.float64)
        sink_b = n_cp * n_cp
        counts = self._counts
        corr_mass = self._corr_mass
        has_sink = counts[sink_b] > 0

        # Hash query with same centering + rotations
        q_c = q_centered - self._key_mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z1_q = (self._R1 @ q_c).astype(np.float64)
            z2_q = (self._R2 @ q_c).astype(np.float64)

        # Per-CP collision probabilities
        pi1 = softmax(_cp_vertex_scores(z1_q))
        pi2 = softmax(_cp_vertex_scores(z2_q))

        # Nonempty non-sink buckets
        nonsink_ne = np.where(counts[:sink_b] > 0)[0]

        if len(nonsink_ne) == 0 and not has_sink:
            return AttentionOutput(
                output=np.zeros(
                    problem.values.shape[1],
                    dtype=np.float32,
                ),
                actual_budget=0,
            )

        # Multi-probe order: sort by pi1[b1]*pi2[b2] desc
        if len(nonsink_ne) > 0:
            b1_idx = nonsink_ne // n_cp
            b2_idx = nonsink_ne % n_cp
            pi_comb = pi1[b1_idx] * pi2[b2_idx]
            order = np.argsort(pi_comb)[::-1]
            nonsink_ne = nonsink_ne[order]
            pi_comb = pi_comb[order]
        else:
            pi_comb = np.empty(0, dtype=np.float64)

        # Budget: sink always included
        cp_budget = max(
            0, budget - (1 if has_sink else 0),
        )
        n_probe = min(cp_budget, len(nonsink_ne))

        probed = nonsink_ne[:n_probe]
        pi_probed = pi_comb[:n_probe]

        # Assemble probed set
        if has_sink and n_probe > 0:
            all_idx = np.concatenate(
                [[sink_b], probed],
            )
        elif has_sink:
            all_idx = np.array(
                [sink_b], dtype=np.int64,
            )
        elif n_probe > 0:
            all_idx = probed
        else:
            return AttentionOutput(
                output=np.zeros(
                    problem.values.shape[1],
                    dtype=np.float32,
                ),
                actual_budget=0,
            )

        mk = self._avg_keys[all_idx]
        mv = self._avg_values[all_idx]
        mass = np.maximum(
            corr_mass[all_idx].astype(np.float64),
            1e-30,
        )

        # Importance weights: pi_sink = 1, others from CP
        log_pi = np.zeros(
            len(all_idx), dtype=np.float64,
        )
        start = 1 if has_sink else 0
        if n_probe > 0:
            log_pi[start:] = np.log(
                np.maximum(pi_probed, 1e-30),
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = (
                (mk @ q_centered).astype(np.float64) / sqrt_d
            )
            # Denominator + numerator correction:
            # add log(sum_i exp(mean_q·k_i/sqrt(d))) per bucket.
            # w_b proportional to exp(score_b + log(mass_b) - log(pi_b))
            adjusted = scores + np.log(mass) - log_pi
            w = softmax(adjusted).astype(np.float32)
            output = w @ mv

        return AttentionOutput(
            output=output,
            actual_budget=int(len(all_idx)),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [LSHCrossPolytope()]
