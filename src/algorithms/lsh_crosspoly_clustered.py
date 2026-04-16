"""
LSH Cross-Polytope Clustered: compressed attention via
reduced-dimension cross-polytope hashing.

Unlike the probing variant (LSHCrossPolytope), this method:
  - uses only k <= d dimensions for CP hashing, giving
    (2k)^2 + 1 coarser buckets
  - attends over ALL non-empty bucket means (no multi-probe)
  - applies count correction: score_b + log(n_b)
  - properly handles causal window and special indices
    (sink + local window always attended exactly)
  - optionally averages R independent repetitions to reduce
    partition sensitivity

Budget is controlled by k_dim, not by probing depth.
"""

import numpy as np
from typing import Dict, List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .lsh_crosspoly_multiprobe import (
    _random_orthogonal, crosspolytope_bucket_labels,
    _bucket_means,
)
from ..core import softmax, hybrid_attention


def _bucket_cosine_similarity(
    keys: np.ndarray,
    labels: np.ndarray,
    avg_keys: np.ndarray,
    n_buckets: int,
) -> float:
    """
    Weighted average cosine similarity of keys to their
    assigned bucket mean key.
    """
    total_sim = 0.0
    total_n = 0
    for b in range(n_buckets):
        mask = labels == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        k_b = keys[mask]
        mu = avg_keys[b]
        mu_norm = np.linalg.norm(mu)
        if mu_norm < 1e-10:
            continue
        k_norms = np.linalg.norm(k_b, axis=1)
        valid = k_norms > 1e-10
        if not np.any(valid):
            continue
        dots = k_b[valid] @ mu
        cos = dots / (k_norms[valid] * mu_norm)
        total_sim += float(np.sum(cos))
        total_n += int(np.sum(valid))
    if total_n == 0:
        return 0.0
    return total_sim / total_n


class LSHCrossPolytopeClustered(AttentionAlgorithm):
    """
    Clustered attention via reduced cross-polytope LSH.

    Each repetition builds an independent partition using
    two random orthogonal rotations truncated to the first
    k_dim coordinates before CP hashing.  The final output
    is the arithmetic mean across repetitions.

    run() filters to causal window, excludes special
    indices, recomputes bucket means from valid keys,
    and uses hybrid_attention (matching KMeans flow).
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

        self._rep_labels: Optional[List[np.ndarray]] = None
        self._n_buckets: int = 0
        self._d: Optional[int] = None
        self._quality: Optional[Dict[str, float]] = None

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
        self._n_buckets = n_buckets

        rng = np.random.default_rng(seed)

        if len(keys) == 0:
            self._rep_labels = [
                np.array([], dtype=np.int64),
            ] * self._n_rep
            return

        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        x_c = keys.astype(np.float32) - key_mean
        x64 = x_c.astype(np.float64)
        n = len(keys)

        self._rep_labels = []
        for _ in range(self._n_rep):
            R1 = _random_orthogonal(d, rng)
            R2 = _random_orthogonal(d, rng)

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

            self._rep_labels.append(labels)

        # Cluster quality from first repetition
        labels0 = self._rep_labels[0]
        avg_k, _, counts = _bucket_means(
            keys, values, labels0, n_buckets,
        )
        n_nonempty = int(np.sum(counts > 0))
        self._quality = {
            "avg_cosine_sim": _bucket_cosine_similarity(
                keys, labels0, avg_k, n_buckets,
            ),
            "n_groups": n_nonempty,
        }

    def cluster_quality(self) -> Optional[Dict[str, float]]:
        return self._quality

    # ── online ──────────────────────────────────────────

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._rep_labels is None:
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

        accum = np.zeros(d_v, dtype=np.float64)
        total_budget = 0

        for labels in self._rep_labels:
            # Slice labels to causal window
            causal_labels = labels[:n_causal]

            # Build group indices excluding special keys
            groups = []
            for b in range(self._n_buckets):
                idx = np.where(causal_labels == b)[0]
                if len(idx) == 0:
                    continue
                if special_set:
                    keep = np.ones(
                        len(idx), dtype=bool,
                    )
                    for s in special_set:
                        keep &= (idx != s)
                    idx = idx[keep]
                if len(idx) > 0:
                    groups.append(idx)

            if not groups:
                continue

            output, eff_budget = hybrid_attention(
                q, keys, values, logits, groups,
                0, head_dim, special_idx, "hybrid",
            )
            accum += output.astype(np.float64)
            total_budget += eff_budget

        n_reps_used = max(self._n_rep, 1)
        output = (accum / n_reps_used).astype(
            np.float32,
        )
        avg_budget = total_budget // n_reps_used

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
