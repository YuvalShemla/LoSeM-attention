"""
Cross-Polytope LSH with SNIS correction.

Product of two cross-polytope hashes per table with min_hits=1
(any single table collision retrieves a key). Sweeping k_dim
controls the partition granularity:

  k_dim=128 (full):  (2*128)^2 = 65536 buckets/table — finest
  k_dim=64:          (2*64)^2  = 16384 buckets/table
  k_dim=32:          (2*32)^2  =  4096 buckets/table
  k_dim=16:          (2*16)^2  =  1024 buckets/table — coarsest


We tested several other configurations and found this one best:

  - CP1 (single CP/table, min_hits 2-3): too coarse — only
    2d=256 buckets, most keys collide regardless of distance.
  - CP2 with min_hits=2: too selective at low L — requiring 2+
    collisions across tables with 65K buckets filters out too
    many relevant keys.
  - Stacking n_cp>2 CPs: bucket count grows as (2k)^n_cp,
    quickly making collisions extremely rare.

CP2 with min_hits=1 is what was chosen: the product partition
is fine enough that each collision is meaningful, and a single
hit suffices.

Collision probability (Theorem 1, Andoni et al. 2015,
"Practical and Optimal LSH for Angular Distance"):

  For a single cross-polytope hash in k dimensions applied to
  unit vectors with Euclidean distance tau:

    ln(1/Pr[h(p)=h(q)]) = tau^2/(4-tau^2) * ln(k) + O(ln ln k)

  Using cosine similarity cos = 1 - tau^2/2:

    rho = (1 - cos) / (1 + cos)
    p_single = k^(-rho)

  For our product of two independent k-dim CP hashes:

    p_table = k^(-2 * rho)

  Inclusion probability (min_hits=1 across L tables):

    P(retrieved) = 1 - (1 - p_table)^L

Key centering: before hashing, all keys and the query are
centered by subtracting the mean key vector. This is critical
(as in MagicPIG/SimHash) because raw key vectors in
transformer attention are not zero-mean — centering removes
the shared component so the hash is sensitive to the angular
differences between keys. The CP hash is scale-invariant
(argmax|Rx| doesn't change with scaling), so the collision
probability depends only on the angle between centered vectors.

Each (k_dim, L) combo produces a fixed emergent budget.
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
    Product-of-two-CP LSH with SNIS, min_hits=1.

    Parameters:
        k_dim: CP hash dimension (uses first k_dim rows of a
               random d×d orthogonal rotation; k_dim=d for full)
        L: number of independent hash tables
        center_keys: whether to center keys before hashing
    """

    _next_id = 0

    def __init__(
        self,
        k_dim: int = 128,
        L: int = 50,
        center_keys: bool = True,
    ):
        self._k = k_dim
        self._L = L
        self._center_keys = center_keys
        self._id = LSHCrossPolySNIS._next_id
        LSHCrossPolySNIS._next_id += 1

        # Populated by prepare()
        self._all_R1 = None    # [L, k, d]
        self._all_R2 = None    # [L, k, d]
        self._key_labels = None  # [n, L]
        self._key_mean = None
        self._d = None

    @property
    def name(self) -> str:
        return f"CP-SNIS-{self._id}"

    @property
    def point_label(self) -> str:
        return f"{self._k}/{self._L}"

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
        L = self._L
        n_verts = 2 * k  # vertices per single CP hash

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

        # Build L tables, each with two k-dim rotations
        all_R1 = []
        all_R2 = []
        labels = np.empty((n, L), dtype=np.int32)

        for l in range(L):
            Q1 = _random_orthogonal(d, rng)
            Q2 = _random_orthogonal(d, rng)
            R1 = Q1[:k].astype(np.float64)
            R2 = Q2[:k].astype(np.float64)
            all_R1.append(Q1[:k].astype(np.float32))
            all_R2.append(Q2[:k].astype(np.float32))

            z1 = (x64 @ R1.T).astype(np.float32)
            z2 = (x64 @ R2.T).astype(np.float32)
            b1 = self._cp_labels_2d(z1)
            b2 = self._cp_labels_2d(z2)
            labels[:, l] = b1 * n_verts + b2

        self._all_R1 = np.stack(all_R1)  # [L, k, d]
        self._all_R2 = np.stack(all_R2)
        self._key_labels = labels

    @staticmethod
    def _cp_labels_2d(
        proj: np.ndarray,
    ) -> np.ndarray:
        """
        CP bucket labels for [n, k] matrix.
        Returns [n] labels in {0, ..., 2k-1}.
        """
        abs_proj = np.abs(proj)
        idx = np.argmax(abs_proj, axis=1)
        signs = proj[np.arange(len(proj)), idx]
        sign_bit = (signs < 0).astype(np.int32)
        return 2 * idx + sign_bit

    @staticmethod
    def _cp_label_1d(proj: np.ndarray) -> int:
        """CP bucket label for a single [k] vector."""
        abs_proj = np.abs(proj)
        idx = int(np.argmax(abs_proj))
        sign_bit = 1 if proj[idx] < 0 else 0
        return 2 * idx + sign_bit

    # ── query hashing ──────────────────────────────────

    def _hash_query(
        self, query: np.ndarray,
    ) -> np.ndarray:
        """Hash query, returns [L] product bucket labels."""
        k = self._k
        L = self._L
        n_verts = 2 * k

        if self._center_keys:
            q_c = (
                query.astype(np.float64)
                - self._key_mean.astype(np.float64)
            )
        else:
            q_c = query.astype(np.float64)

        labels = np.empty(L, dtype=np.int32)
        for l in range(L):
            R1 = self._all_R1[l].astype(np.float64)
            R2 = self._all_R2[l].astype(np.float64)
            z1 = (R1 @ q_c).astype(np.float32)
            z2 = (R2 @ q_c).astype(np.float32)
            b1 = self._cp_label_1d(z1)
            b2 = self._cp_label_1d(z2)
            labels[l] = b1 * n_verts + b2

        return labels

    # ── inclusion probability ──────────────────────────
    #
    # Exact CP2 collision probability lookup table,
    # measured via Monte Carlo with 5M synthetic rotation
    # pairs at 0.001 cosine-similarity resolution.
    #
    # Loaded from data/cp2_collision_table.npz (generated
    # by scripts/cp2_exact_collision_table.py). This gives
    # precise per-table collision rates as a function of
    # the angle between centered query and key vectors.

    # Per-k_dim collision table cache (class-level).
    _collision_tables = {}  # k_dim → (cos_bins, p_table)

    @classmethod
    def _load_collision_table(cls, k_dim):
        if k_dim in cls._collision_tables:
            return cls._collision_tables[k_dim]
        import os
        # k_dim=128 uses the base filename for backwards compat
        if k_dim == 128:
            fname = "cp2_collision_table.npz"
        else:
            fname = f"cp2_collision_table_k{k_dim}.npz"
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
        cls._collision_tables[k_dim] = table
        return table

    def _compute_inclusion_prob(
        self, cos_sims: np.ndarray,
    ) -> np.ndarray:
        """
        Inclusion probability from exact lookup table.

        Per-table collision probability is looked up via
        linear interpolation from a 5M-table Monte Carlo
        measurement (data/cp2_collision_table[_kN].npz).

        Inclusion with min_hits=1:
          P(retrieved) = 1 - (1 - p_table)^L
        """
        L = self._L
        cos_bins, p_tab = self._load_collision_table(
            self._k,
        )

        cos_sims = np.clip(
            cos_sims, cos_bins[0], cos_bins[-1],
        )
        p_table = np.interp(cos_sims, cos_bins, p_tab)
        p_table = np.clip(p_table, 1e-30, 1.0)

        prob = 1.0 - np.power(1.0 - p_table, L)
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

        retrieved_mask = match_counts >= 1
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

        # Default: per-k_dim L ranges — finer partitions
        # need more tables to retrieve enough keys.
        default_sweep = {
            64:  [10, 20, 50, 100, 150, 200],
            96:  [20, 50, 100, 150, 200, 250],
            128: [50, 100, 150, 200, 250, 300],
        }
        sweep = cfg.get("sweep", default_sweep)

        for k, L_values in sweep.items():
            k = int(k)
            for L in L_values:
                instances.append(LSHCrossPolySNIS(
                    k_dim=k, L=L, center_keys=center,
                ))
        return instances
