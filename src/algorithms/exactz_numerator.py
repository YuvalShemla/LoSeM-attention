"""
ExactZ numerator-only variants: full softmax denominator Z,
select keys for the numerator only (no value clustering).

output = (1/Z) * sum_{i in selected + special} exp(logit_i) * v_i

Two variants:
  ExactZTopK    — select top-budget candidates by logit
  ExactZUniform — sample budget candidates uniformly at random
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax


class ExactZTopK(AttentionAlgorithm):
    """
    Exact Z + top-K keys for numerator.

    Denominator = sum_all exp(logit_i).
    Numerator = sum_{special + topK} exp(logit_i) * v_i.
    """

    @property
    def name(self) -> str:
        return "ExactZ-TopK"

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

        # Exact Z
        max_l = float(np.max(logits))
        exp_l = np.exp(logits.astype(np.float64) - max_l)
        Z = np.sum(exp_l)

        # Top-budget candidates by logit
        buse = min(budget, n_cand)
        if buse > 0 and buse < n_cand:
            cand_logits = logits[candidate_idx]
            top_pos = np.argpartition(
                cand_logits, -buse,
            )[-buse:]
            sel_idx = candidate_idx[top_pos]
        elif buse > 0:
            sel_idx = candidate_idx
        else:
            sel_idx = np.array([], dtype=np.int64)

        all_idx = np.concatenate(
            [special_idx, sel_idx],
        ).astype(np.int64)

        numer = (
            exp_l[all_idx]
            @ values[all_idx].astype(np.float64)
        )
        output = (numer / Z).astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=len(all_idx),
            selected_indices=all_idx,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [ExactZTopK()]


class ExactZUniform(AttentionAlgorithm):
    """
    Exact Z + uniformly sampled keys for numerator.

    Denominator = sum_all exp(logit_i).
    Numerator = sum_{special + uniform_sample} exp(logit_i) * v_i.
    """

    @property
    def name(self) -> str:
        return "ExactZ-Uniform"

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

        # Exact Z
        max_l = float(np.max(logits))
        exp_l = np.exp(logits.astype(np.float64) - max_l)
        Z = np.sum(exp_l)

        # Uniform sample
        buse = min(budget, n_cand)
        if buse > 0:
            chosen = rng.choice(
                n_cand, size=buse, replace=False,
            )
            sel_idx = candidate_idx[chosen]
        else:
            sel_idx = np.array([], dtype=np.int64)

        all_idx = np.concatenate(
            [special_idx, sel_idx],
        ).astype(np.int64)

        numer = (
            exp_l[all_idx]
            @ values[all_idx].astype(np.float64)
        )
        output = (numer / Z).astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=len(all_idx),
            selected_indices=all_idx,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [ExactZUniform()]
