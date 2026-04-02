"""
LSH Cross-Polytope Clustered: compressed attention via
reduced-dimension cross-polytope hashing.

Unlike the probing variant (LSHCrossPolytope), this method:
  - uses only k <= d dimensions for CP hashing, giving
    (2k)^2 + 1 coarser buckets
  - attends over ALL non-empty bucket means (no multi-probe)
  - applies count correction: score_b + log(n_b)
  - optionally averages R independent repetitions to reduce
    partition sensitivity (bias is not removed by averaging)

Budget is controlled by k_dim, not by probing depth.
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .lsh_crosspoly_multiprobe import (
    _random_orthogonal, crosspolytope_bucket_labels,
    _bucket_means,
)
from ..core import softmax


class LSHCrossPolytopeClustered(AttentionAlgorithm):
    """
    Clustered attention via reduced cross-polytope LSH.

    Each repetition builds an independent partition using
    two random orthogonal rotations truncated to the first
    k_dim coordinates before CP hashing.  The final output
    is the arithmetic mean across repetitions.

    This is a compressed clustering approximation, not an
    importance-weighted probing estimator.
    """

    def __init__(
        self,
        k_dim: int,
        n_repetitions: int = 1,
        name_suffix: str = "",
    ):
        self._k = k_dim
        self._n_rep = n_repetitions
        self._name_suffix = name_suffix

        # Per-repetition state, populated by prepare().
        # Each entry: (avg_keys, avg_values, counts)
        self._reps: Optional[List[tuple]] = None
        self._key_mean: Optional[np.ndarray] = None
        self._d: Optional[int] = None

    # ── naming ──────────────────────────────────────────

    @property
    def name(self) -> str:
        n_buckets = (2 * self._k) ** 2 + 1
        if self._name_suffix:
            return (
                f"LSH-CrossPoly-Clustered"
                f"-{self._name_suffix}-{n_buckets}"
            )
        if self._n_rep > 1:
            return (
                f"LSH-CrossPoly-Clustered"
                f"-R{self._n_rep}-{n_buckets}"
            )
        return f"LSH-CrossPoly-Clustered-{n_buckets}"

    @property
    def sweeps_budget(self) -> bool:
        # Each instance is a single operating point;
        # expand_from_config produces multiple instances.
        return False

    # ── offline ─────────────────────────────────────────

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
        k = self._k
        n_cp = 2 * k
        n_buckets = n_cp * n_cp + 1
        sink_b = n_cp * n_cp

        rng = np.random.default_rng(seed)

        if len(keys) == 0:
            self._key_mean = np.zeros(
                d, dtype=np.float32,
            )
            empty = (
                np.zeros((0, d), dtype=np.float32),
                np.zeros(
                    (0, values.shape[1]),
                    dtype=np.float32,
                ),
                np.zeros(0, dtype=np.int32),
            )
            self._reps = [empty] * self._n_rep
            return

        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        self._key_mean = key_mean
        x_c = keys.astype(np.float32) - key_mean
        x64 = x_c.astype(np.float64)
        n = len(keys)

        self._reps = []
        for _ in range(self._n_rep):
            R1 = _random_orthogonal(d, rng)
            R2 = _random_orthogonal(d, rng)

            # Only compute first k columns of the rotation
            # (the rest are unused for hashing)
            R1_k = R1[:k].astype(np.float64)
            R2_k = R2[:k].astype(np.float64)
            z1_k = (x64 @ R1_k.T).astype(np.float32)
            z2_k = (x64 @ R2_k.T).astype(np.float32)

            labels = np.empty(n, dtype=np.int64)
            labels[0] = sink_b
            if n > 1:
                b1 = crosspolytope_bucket_labels(z1_k[1:])
                b2 = crosspolytope_bucket_labels(z2_k[1:])
                labels[1:] = b1 * n_cp + b2

            avg_k, avg_v, counts = _bucket_means(
                keys, values, labels, n_buckets,
            )

            # Keep only non-empty buckets
            ne = np.where(counts > 0)[0]
            self._reps.append((
                avg_k[ne],
                avg_v[ne],
                counts[ne],
            ))

    # ── online ──────────────────────────────────────────

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._reps is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query.astype(np.float32)
        d = self._d
        sqrt_d = np.sqrt(d, dtype=np.float64)
        q_c = q - self._key_mean
        d_v = problem.values.shape[1]

        accum = np.zeros(d_v, dtype=np.float64)
        total_budget = 0

        for mk, mv, counts in self._reps:
            if len(mk) == 0:
                continue

            # Count-corrected attention over all non-empty
            # bucket means:  α_b ∝ n_b · exp(q·k̄_b / √d)
            log_n = np.log(
                counts.astype(np.float64),
            )

            scores = (
                (mk @ q).astype(np.float64) / sqrt_d
            )
            # + log(n_b) so larger buckets contribute
            # proportionally to their token count
            adjusted = scores + log_n
            w = softmax(adjusted).astype(np.float32)
            accum += (w @ mv).astype(np.float64)

            total_budget += len(mk)

        output = (accum / self._n_rep).astype(np.float32)
        avg_budget = total_budget // max(self._n_rep, 1)

        return AttentionOutput(
            output=output,
            actual_budget=avg_budget,
        )

    # ── config expansion ────────────────────────────────

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        k_dims = cfg.get("k_dims", [4, 8, 16, 32, 64])
        n_reps = cfg.get("n_repetitions", [1])
        if isinstance(n_reps, int):
            n_reps = [n_reps]

        instances = []
        for r in n_reps:
            for k in k_dims:
                instances.append(
                    LSHCrossPolytopeClustered(
                        k_dim=k, n_repetitions=r,
                    )
                )
        return instances
