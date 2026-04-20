"""
SimHash LSH with importance sampling correction.

Two variants:
  - SNIS (Z'): Self-Normalized IS -- softmax over retrieved keys
    with logit correction -log(u_i). The normalizer Z' is estimated
    from the sample itself.
  - IS (Z): Standard IS with the TRUE softmax normalizer Z computed
    over ALL keys. Unbiased but potentially higher variance.

Offline: center keys, build L SimHash tables with K random
hyperplanes each, store per-key hash codes.

Query: hash query, retrieve all keys colliding in >= min_hits
tables, compute angle-based inclusion probabilities, apply
importance sampling for attention estimation.

Each (K, L) combo produces a fixed (emergent) budget -- the
number of retrieved keys depends on the data, not a budget
parameter. sweeps_budget=False; each instance is one dot.

Reference: MagicPIG (Chen et al., 2024) -- LSH Sampling for
Efficient LLM Generation.
"""

import numpy as np
from typing import List, Optional, Dict

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import snis_attention


class LSHSimHashSNIS(AttentionAlgorithm):
    """
    SimHash LSH with SNIS correction.

    Parameters:
        K: number of hash bits per table (hash depth)
        L: number of independent hash tables
        min_hits: minimum number of table collisions to retrieve a key
        center_keys: whether to center keys before hashing (critical)
    """

    _next_id = 0

    def __init__(
        self,
        K: int = 9,
        L: int = 120,
        min_hits: int = 2,
        center_keys: bool = True,
    ):
        self._K = K
        self._L = L
        self._min_hits = min_hits
        self._center_keys = center_keys
        self._id = LSHSimHashSNIS._next_id
        LSHSimHashSNIS._next_id += 1

        # Populated by prepare()
        self._hyperplanes = None   # [L*K, d] random hyperplanes
        self._key_codes = None     # [n_keys, L] packed hash codes (uint32)
        self._key_mean = None      # [d] key centroid
        self._d = None

    @property
    def name(self) -> str:
        return f"SimHash-SNIS-{self._id}"

    @property
    def point_label(self) -> str:
        C = 2 ** self._K
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
        self._d = head_dim
        K, L = self._K, self._L
        rng = np.random.default_rng(seed)

        # Generate L*K random hyperplanes
        hyperplanes = rng.standard_normal(
            (L * K, head_dim),
        ).astype(np.float32)
        # Normalize rows for cleaner angle computation
        norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        hyperplanes /= np.maximum(norms, 1e-10)
        self._hyperplanes = hyperplanes

        n = len(keys)
        if n == 0:
            self._key_codes = np.empty((0, L), dtype=np.uint32)
            self._key_mean = np.zeros(head_dim, dtype=np.float32)
            return

        # Center keys (critical per MagicPIG paper)
        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        self._key_mean = key_mean

        if self._center_keys:
            keys_c = keys.astype(np.float32) - key_mean
        else:
            keys_c = keys.astype(np.float32)

        # Compute hash codes: sign of projection onto hyperplanes
        projections = keys_c @ hyperplanes.T  # [n, L*K]
        sign_bits = (projections >= 0)  # [n, L*K] bool

        # Pack K bits per table into uint32 hash codes
        sign_bits = sign_bits.reshape(n, L, K)
        powers = (1 << np.arange(K, dtype=np.uint32))[np.newaxis, np.newaxis, :]
        self._key_codes = np.sum(
            sign_bits.astype(np.uint32) * powers, axis=2,
        ).astype(np.uint32)  # [n, L]

    def _hash_query(
        self, query: np.ndarray,
    ) -> np.ndarray:
        """Hash a single query vector. Returns [L] uint32 codes."""
        K, L = self._K, self._L

        if self._center_keys:
            q_c = query.astype(np.float32) - self._key_mean
        else:
            q_c = query.astype(np.float32)

        proj = q_c @ self._hyperplanes.T  # [L*K]
        sign_bits = (proj >= 0).reshape(L, K)
        powers = (1 << np.arange(K, dtype=np.uint32))[np.newaxis, :]
        return np.sum(
            sign_bits.astype(np.uint32) * powers, axis=1,
        ).astype(np.uint32)  # [L]

    def _compute_inclusion_prob(
        self, cos_sims: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SimHash inclusion probability for min_hits collisions.

        For SimHash: p_bit = 1 - theta/pi where theta = arccos(cos_sim).
        P(collision in one table) = p_bit^K
        P(retrieved) = 1 - P(0 hits) - P(1 hit) - ... - P(min_hits-1 hits)

        For min_hits=2:
          P = 1 - (1-p)^L - L*p*(1-p)^(L-1)
        """
        K, L = self._K, self._L
        min_hits = self._min_hits

        cos_sims = np.clip(cos_sims, -1.0 + 1e-7, 1.0 - 1e-7)
        thetas = np.arccos(cos_sims)
        p_bit = 1.0 - thetas / np.pi
        p_table = np.power(p_bit, K)

        if min_hits == 1:
            return np.clip(1.0 - np.power(1.0 - p_table, L), 1e-30, 1.0)

        # min_hits == 2 (standard MagicPIG)
        q = 1.0 - p_table
        prob = 1.0 - np.power(q, L) - L * p_table * np.power(q, L - 1)

        if min_hits > 2:
            from scipy.special import comb
            for h in range(2, min_hits):
                prob -= comb(L, h) * np.power(p_table, h) * np.power(q, L - h)

        return np.clip(prob, 1e-30, 1.0)

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._key_codes is None:
            raise RuntimeError("Call prepare() before run()")

        query = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx

        # Hash the query
        q_codes = self._hash_query(query)  # [L]

        # Only search among candidate keys (exclude special)
        if len(candidate_idx) == 0:
            from ..core import subset_attention
            output = subset_attention(logits, values, special_idx)
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        # Get hash codes for candidate keys only
        cand_codes = self._key_codes[candidate_idx]  # [n_cand, L]

        # Count collisions
        matches = (cand_codes == q_codes[np.newaxis, :])
        match_counts = np.sum(matches, axis=1)

        # Retrieve candidates with >= min_hits collisions
        retrieved_mask = match_counts >= self._min_hits
        retrieved_local = np.where(retrieved_mask)[0]

        if len(retrieved_local) == 0:
            from ..core import subset_attention
            output = subset_attention(logits, values, special_idx)
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        # Map back to global indices
        retrieved_idx = candidate_idx[retrieved_local]

        # Compute cosine similarities for inclusion probabilities
        if self._center_keys:
            q_c = query.astype(np.float64) - self._key_mean.astype(np.float64)
            k_c = keys[retrieved_idx].astype(np.float64) - self._key_mean.astype(np.float64)
        else:
            q_c = query.astype(np.float64)
            k_c = keys[retrieved_idx].astype(np.float64)

        q_norm = np.linalg.norm(q_c)
        k_norms = np.linalg.norm(k_c, axis=1)
        cos_sims = (k_c @ q_c) / (q_norm * k_norms + 1e-10)

        inclusion_probs = self._compute_inclusion_prob(cos_sims)

        output = snis_attention(
            logits=logits[retrieved_idx],
            values=values[retrieved_idx],
            inclusion_probs=inclusion_probs,
            special_logits=logits[special_idx],
            special_values=values[special_idx],
        )

        actual_budget = len(special_idx) + len(retrieved_idx)

        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
            selected_indices=np.concatenate([
                special_idx, retrieved_idx,
            ]),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        K_values = cfg.get("K_sweep", [8, 9, 10, 11])
        L_values = cfg.get("L_sweep", [50, 75, 100, 120, 150, 200, 300])
        min_hits = cfg.get("min_hits", 2)
        center_keys = cfg.get("center_keys", True)
        for K in K_values:
            for L_val in L_values:
                instances.append(LSHSimHashSNIS(
                    K=K, L=L_val,
                    min_hits=min_hits,
                    center_keys=center_keys,
                ))
        return instances
