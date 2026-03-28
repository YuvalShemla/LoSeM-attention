"""
MeanQ Grouping: sort keys by mean-query logit, G equal
groups, then hybrid/topk attention over sorted groups.

Offline cost: one matrix-vector product (K @ mean_Q).
Per-query cost: O(G) group scoring.
"""

import numpy as np
from typing import List

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..math_utils import (
    make_equal_groups, hybrid_attention,
)


class MeanQGrouping(AttentionAlgorithm):
    """Sort keys by mean-query logit, G equal groups."""

    def __init__(
        self,
        n_groups: int = 32,
        mode: str = "hybrid",
        top_k: int = 5,
    ):
        self.n_groups = n_groups
        self.mode = mode
        self.top_k = top_k
        self._sorted_order = None

    @property
    def name(self) -> str:
        return (
            f"MeanQ-G{self.n_groups}"
            f"-{self.mode}-k{self.top_k}"
        )

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        """Compute mean query and sort keys once."""
        if queries is None or query_positions is None:
            raise ValueError(
                "MeanQGrouping.prepare requires "
                "queries and query_positions"
            )
        sqrt_d = np.sqrt(head_dim)
        # Mean of query vectors at evaluation positions
        mean_Q = np.mean(queries[query_positions], axis=0)
        # Sort all keys by projection onto mean query
        scores = (keys @ mean_Q) / sqrt_d
        self._sorted_order = np.argsort(scores)[::-1]

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._sorted_order is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        n_causal = len(problem.keys)
        special_set = problem.special_set
        special_idx = problem.special_idx

        # Filter sorted order to causal + non-special
        valid = np.array([
            i for i in self._sorted_order
            if i < n_causal and i not in special_set
        ])

        groups = make_equal_groups(valid, self.n_groups)
        if not groups:
            return AttentionOutput(
                output=np.zeros(problem.head_dim),
                actual_budget=0,
            )

        output, eff_budget = hybrid_attention(
            problem.query, problem.keys, problem.values,
            problem.logits, groups, self.top_k,
            problem.head_dim, special_idx, self.mode,
        )

        return AttentionOutput(
            output=output,
            actual_budget=eff_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        for mode in cfg.get("modes", ["hybrid"]):
            for k in cfg.get("top_k_sweep", [5]):
                instances.append(MeanQGrouping(
                    n_groups=cfg.get("n_groups", 32),
                    mode=mode,
                    top_k=k,
                ))
        return instances
