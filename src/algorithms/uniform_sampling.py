"""
Uniform Sampling: sample `budget` keys uniformly at random
from candidates, then plain subset softmax.

Baseline for how much structure-aware methods improve over
random selection.
"""

import numpy as np

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, subset_attention


class UniformSampling(AttentionAlgorithm):
    """
    Uniformly sample budget keys from candidates,
    run subset softmax over special + sampled keys.
    """

    @property
    def name(self) -> str:
        return "UniformSampling"

    @property
    def kind(self) -> str:
        return "idealized"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)

        if n_cand == 0:
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
                selected_indices=special_idx,
            )

        buse = min(budget, n_cand)
        chosen = rng.choice(
            n_cand, size=buse, replace=False,
        )
        sampled_idx = candidate_idx[chosen]

        all_idx = np.concatenate(
            [special_idx, sampled_idx],
        ).astype(np.int64)
        output = subset_attention(
            logits, values, all_idx,
        )

        return AttentionOutput(
            output=output,
            actual_budget=len(all_idx),
            selected_indices=all_idx,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [UniformSampling()]
