"""
ExactZ-SampledKeys: exact softmax denominator, clustered values.

Denominator Z = sum_i exp(logit_i) is computed exactly from
all precomputed logits.

For the numerator:
  1. Sample S candidate keys randomly.
  2. Use their values as Voronoi centers — assign every
     candidate value to its nearest sampled-value center
     (1-NN in value space).
  3. Sum values in each Voronoi cell.
  4. numerator = sum_special [exp(logit_j) * v_j]
               + sum_sampled [exp(logit_i) * cell_sum_i]
  5. output = numerator / Z

This guarantees:
  - Every value contributes exactly once (no duplication)
  - Each sampled key "owns" a partition of value space
  - Special keys contribute exactly
  - Z is exact
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import subset_attention


class ExactZSampledKeys(AttentionAlgorithm):
    """
    Exact Z + sampled keys with value-space Voronoi clustering.

    Parameters:
        n_samples: number of candidate keys to sample (S).
                   Budget = S + n_special.
    """

    def __init__(self, n_samples: int = 256):
        self._n_samples = n_samples

    @property
    def name(self) -> str:
        return f"ExactZ-SK-{self._n_samples}"

    @property
    def sweeps_budget(self) -> bool:
        return False

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        logits = problem.logits          # [N]
        values = problem.values          # [N, d]
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx

        n_cand = len(candidate_idx)
        S = min(self._n_samples, n_cand)

        if S == 0 or n_cand == 0:
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        # ── Exact Z (numerically stable) ──
        max_logit = float(np.max(logits))
        exp_logits = np.exp(
            logits.astype(np.float64) - max_logit
        )
        Z = np.sum(exp_logits)

        # ── Special keys: exact contribution ──
        sp_exp = exp_logits[special_idx]        # [n_sp]
        sp_vals = values[special_idx].astype(np.float64)
        sp_numer = sp_exp @ sp_vals             # [d]

        # ── Sample S candidate keys ──
        sampled_local = rng.choice(
            n_cand, size=S, replace=False,
        )
        sampled_idx = candidate_idx[sampled_local]
        sampled_vals = values[sampled_idx].astype(np.float32)

        # ── 1-NN: assign all candidate values to nearest
        #    sampled-value center (Voronoi partition) ──
        cand_vals = values[candidate_idx].astype(np.float32)
        sv_sq = np.sum(sampled_vals ** 2, axis=1)  # [S]

        # Chunked to keep memory bounded
        chunk_size = 4096
        assignments = np.empty(n_cand, dtype=np.int32)
        for start in range(0, n_cand, chunk_size):
            end = min(start + chunk_size, n_cand)
            chunk = cand_vals[start:end]
            chunk_sq = np.sum(chunk ** 2, axis=1)
            dots = chunk @ sampled_vals.T          # [chunk, S]
            dists = (
                chunk_sq[:, None] + sv_sq[None, :]
                - 2.0 * dots
            )
            assignments[start:end] = np.argmin(
                dists, axis=1,
            )

        # ── Sum candidate values per Voronoi cell ──
        cell_sums = np.zeros(
            (S, head_dim), dtype=np.float64,
        )
        np.add.at(
            cell_sums, assignments,
            cand_vals.astype(np.float64),
        )

        # ── Approximate candidate contribution ──
        sampled_exp = exp_logits[sampled_idx]   # [S]
        approx_numer = sampled_exp @ cell_sums  # [d]

        # ── Combine ──
        output = (
            (sp_numer + approx_numer) / Z
        ).astype(np.float32)

        actual_budget = len(special_idx) + S
        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        S_values = cfg.get("n_samples_sweep", [
            16, 32, 64, 128, 256, 512, 1024, 2048,
        ])
        return [ExactZSampledKeys(n_samples=s) for s in S_values]
