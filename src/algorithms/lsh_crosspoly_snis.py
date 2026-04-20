"""
Cross-Polytope LSH with SNIS correction.

Concatenation of m cross-polytope hashes per table:
  - Apply m independent random d×d rotations R_1,...,R_m
  - Concatenate rotated vectors: z = [R_1 x, ..., R_m x]
  - Label = argmax(|z|) with sign → C = 2·m·d buckets

This allows matching SimHash bucket counts exactly:
  m=1  → C=256  (matches SimHash K=8)
  m=2  → C=512  (matches SimHash K=9)
  m=4  → C=1024 (matches SimHash K=10)
  m=8  → C=2048 (matches SimHash K=11)
  m=16 → C=4096 (matches SimHash K=12)

Collision probabilities are measured via Monte Carlo with
5M synthetic rotation pairs (scripts/cp_concat_collision_tables.py)
and stored in data/cp_concat_collision_table_m{m}.npz.

Key centering: before hashing, all keys and the query are
centered by subtracting the mean key vector. The CP hash is
scale-invariant (argmax|Rx| doesn't change with scaling), so
the collision probability depends only on the angle between
centered vectors.

Each (m, L) combo produces a fixed emergent budget.
sweeps_budget=False; each instance is one dot on the plot.
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .lsh_crosspoly_multiprobe import _random_orthogonal
from ..core import snis_attention


class LSHCrossPolySNIS(AttentionAlgorithm):
    """
    Concatenation-CP LSH with SNIS correction.

    Parameters:
        n_rotations: number of concatenated rotations per table
                     (bucket count = 2 * n_rotations * head_dim)
        L: number of independent hash tables
        min_hits: minimum table collisions to retrieve a key
        center_keys: whether to center keys before hashing
    """

    _next_id = 0

    def __init__(
        self,
        n_rotations: int = 2,
        L: int = 50,
        min_hits: int = 2,
        center_keys: bool = True,
    ):
        self._m = n_rotations
        self._L = L
        self._min_hits = min_hits
        self._center_keys = center_keys
        self._id = LSHCrossPolySNIS._next_id
        LSHCrossPolySNIS._next_id += 1

        # Populated by prepare()
        self._rotations = None  # [L][m] list of [d, d]
        self._key_labels = None  # [n, L]
        self._key_mean = None
        self._d = None

    @property
    def name(self) -> str:
        return f"CP-SNIS-{self._id}"

    @property
    def point_label(self) -> str:
        C = 2 * self._m * (self._d or 128)
        return f"C{C}/L{self._L}"

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
        m = self._m
        L = self._L

        rng = np.random.default_rng(seed)

        n = len(keys)
        if n == 0:
            self._key_labels = np.empty(
                (0, L), dtype=np.int32,
            )
            self._key_mean = np.zeros(
                d, dtype=np.float32,
            )
            return

        # Center keys
        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        self._key_mean = key_mean

        if self._center_keys:
            keys_c = keys.astype(np.float32) - key_mean
        else:
            keys_c = keys.astype(np.float32)

        x64 = keys_c.astype(np.float64)

        # Build L tables, each with m rotations
        all_rotations = []
        labels = np.empty((n, L), dtype=np.int32)

        for l in range(L):
            Rs = []
            z_parts = []
            for _ in range(m):
                Q = _random_orthogonal(d, rng)
                Rs.append(Q.astype(np.float32))
                z = (x64 @ Q.T).astype(np.float32)
                z_parts.append(z)  # [n, d]

            all_rotations.append(Rs)

            # Concatenate and take argmax
            z_concat = np.concatenate(
                z_parts, axis=1,
            )  # [n, m*d]
            abs_z = np.abs(z_concat)
            idx = np.argmax(abs_z, axis=1)
            signs = z_concat[np.arange(n), idx]
            labels[:, l] = (
                2 * idx
                + (signs < 0).astype(np.int32)
            )

        self._rotations = all_rotations
        self._key_labels = labels

    # ── query hashing ──────────────────────────────────

    def _hash_query(
        self, query: np.ndarray,
    ) -> np.ndarray:
        """Hash query, returns [L] bucket labels."""
        L = self._L
        m = self._m

        if self._center_keys:
            q_c = (
                query.astype(np.float64)
                - self._key_mean.astype(np.float64)
            )
        else:
            q_c = query.astype(np.float64)

        labels = np.empty(L, dtype=np.int32)
        for l in range(L):
            z_parts = []
            for R in self._rotations[l]:
                z = (R.astype(np.float64) @ q_c).astype(
                    np.float32,
                )
                z_parts.append(z)
            z_concat = np.concatenate(z_parts)  # [m*d]
            abs_z = np.abs(z_concat)
            idx = int(np.argmax(abs_z))
            sign_bit = 1 if z_concat[idx] < 0 else 0
            labels[l] = 2 * idx + sign_bit

        return labels

    # ── inclusion probability ──────────────────────────
    #
    # Exact collision probability lookup tables measured
    # via Monte Carlo with 5M synthetic rotation pairs at
    # 0.001 cosine-similarity resolution over [-1, 1].
    #
    # One table per n_rotations value, stored in
    # data/cp_concat_collision_table_m{m}.npz.

    _collision_tables = {}  # m → (cos_bins, p_table)

    @classmethod
    def _load_collision_table(cls, m):
        if m in cls._collision_tables:
            return cls._collision_tables[m]
        import os
        fname = f"cp_concat_collision_table_m{m}.npz"
        path = os.path.join(
            os.path.dirname(__file__),
            "../../data", fname,
        )
        path = os.path.normpath(path)
        d = np.load(path)
        table = (
            d["cos_bins"].astype(np.float64),
            d["p_table"].astype(np.float64),
        )
        cls._collision_tables[m] = table
        return table

    def _compute_inclusion_prob(
        self, cos_sims: np.ndarray,
    ) -> np.ndarray:
        """
        Inclusion probability from exact lookup table.

        Per-table collision probability is looked up via
        linear interpolation from a 5M-table Monte Carlo
        measurement.

        min_hits=1: P = 1 - (1-p)^L
        min_hits=2: P = 1 - (1-p)^L - L*p*(1-p)^(L-1)
        """
        L = self._L
        min_hits = self._min_hits
        cos_bins, p_tab = self._load_collision_table(
            self._m,
        )

        cos_sims = np.clip(
            cos_sims, cos_bins[0], cos_bins[-1],
        )
        p = np.interp(cos_sims, cos_bins, p_tab)
        p = np.clip(p, 1e-30, 1.0)
        q = 1.0 - p

        if min_hits <= 1:
            prob = 1.0 - np.power(q, L)
        else:
            from scipy.special import comb
            prob = 1.0 - np.power(q, L)
            for h in range(1, min_hits):
                prob -= (
                    comb(L, h)
                    * np.power(p, h)
                    * np.power(q, L - h)
                )

        return np.clip(prob, 1e-30, 1.0)

    # ── online ─────────────────────────────────────────

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._key_labels is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        query = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx

        q_labels = self._hash_query(query)

        if len(candidate_idx) == 0:
            from ..core import subset_attention
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        cand_labels = self._key_labels[candidate_idx]
        matches = (
            cand_labels == q_labels[np.newaxis, :]
        )
        match_counts = np.sum(matches, axis=1)

        retrieved_mask = match_counts >= self._min_hits
        retrieved_local = np.where(retrieved_mask)[0]

        if len(retrieved_local) == 0:
            from ..core import subset_attention
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        retrieved_idx = candidate_idx[retrieved_local]

        if self._center_keys:
            q_c = (
                query.astype(np.float64)
                - self._key_mean.astype(np.float64)
            )
            k_c = (
                keys[retrieved_idx].astype(np.float64)
                - self._key_mean.astype(np.float64)
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

        inclusion_probs = self._compute_inclusion_prob(
            cos_sims,
        )

        output = snis_attention(
            logits=logits[retrieved_idx],
            values=values[retrieved_idx],
            inclusion_probs=inclusion_probs,
            special_logits=logits[special_idx],
            special_values=values[special_idx],
        )

        actual_budget = (
            len(special_idx) + len(retrieved_idx)
        )

        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
            selected_indices=np.concatenate([
                special_idx, retrieved_idx,
            ]),
        )

    # ── config expansion ───────────────────────────────

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        center = cfg.get("center_keys", True)
        min_hits = cfg.get("min_hits", 2)
        L_sweep = cfg.get(
            "L_sweep", [50, 100, 150, 200, 300],
        )
        # n_rotations values — each gives C = 2*m*128
        m_values = cfg.get("n_rotations", [1, 2, 4])

        for m in m_values:
            for L in L_sweep:
                instances.append(LSHCrossPolySNIS(
                    n_rotations=m, L=L,
                    min_hits=min_hits,
                    center_keys=center,
                ))
        return instances
