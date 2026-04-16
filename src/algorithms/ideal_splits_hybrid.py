"""
Ideal Equal Splits Hybrid: sort keys by true logit,
split into fixed equal-sized groups, then apply hybrid
attention (top-K groups enumerated, rest as means).

This is an idealized/oracle method — it uses the true
per-query logits to sort keys. Useful as an upper bound
for how well any fixed-group-count hybrid approach can do.

Parameters:
  n_groups  — number of equal-sized groups (fixed)
  top_k     — groups to expand into individual keys
              (0 = pure grouped, higher = more individual)
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .idealized_methods import _equal_size_split
from ..core import hybrid_attention


class IdealSplitsHybrid(AttentionAlgorithm):
    """
    Oracle-sorted equal splits with hybrid attention.

    Sort candidate keys by true logit (descending), split
    into n_groups balanced groups, then run hybrid_attention:
    top-K groups get individual keys, the rest use
    count-corrected means.
    """

    def __init__(self, n_groups: int = 512, top_k: int = 0):
        self._n_groups = n_groups
        self._top_k = top_k

    @property
    def name(self) -> str:
        return (
            f"IdealSplits-G{self._n_groups}"
            f"-hybrid-k{self._top_k}"
        )

    @property
    def kind(self) -> str:
        return "algorithm"

    @property
    def sweeps_budget(self) -> bool:
        return False

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        q = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx

        if len(candidate_idx) == 0:
            from ..core import subset_attention
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        # Sort candidates by logit descending
        cand_logits = logits[candidate_idx]
        sort_order = np.argsort(cand_logits)[::-1]
        sorted_idx = candidate_idx[sort_order]

        n = len(sorted_idx)
        num_groups = min(self._n_groups, n)
        groups = _equal_size_split(sorted_idx, num_groups)

        output, eff_budget = hybrid_attention(
            q, keys, values, logits, groups,
            self._top_k, head_dim, special_idx,
            "hybrid",
        )

        return AttentionOutput(
            output=output,
            actual_budget=eff_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        n_groups = cfg.get("n_groups", 512)
        top_k_sweep = cfg.get(
            "top_k_sweep",
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
        )
        return [
            IdealSplitsHybrid(
                n_groups=n_groups, top_k=k,
            )
            for k in top_k_sweep
        ]
