"""
Baseline attention methods: OracleTopK, OracleSampling,
OracleGrouping.

Always included in every experiment. Baselines use the
'budget' parameter directly (unlike grouping algorithms
which derive budget from their structural parameters).
"""

import numpy as np

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import (
    softmax, subset_attention,
    make_doubling_boundaries,
)


class OracleTopK(AttentionAlgorithm):
    """
    Top-B keys by logit + sink/local.

    Biased — discards the tail entirely, then
    renormalizes softmax over the selected subset.
    """

    @property
    def name(self) -> str:
        return "OracleTopK"

    @property
    def kind(self) -> str:
        return "baseline"

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

        cand_logits = logits[candidate_idx]
        if buse < n_cand:
            top_pos = np.argpartition(
                cand_logits, -buse
            )[-buse:]
        else:
            top_pos = np.arange(n_cand)
        topk_idx = candidate_idx[top_pos]

        all_idx = np.unique(
            np.concatenate([special_idx, topk_idx])
            .astype(np.int64)
        )
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
        return [OracleTopK()]


class OracleSampling(AttentionAlgorithm):
    """
    Sample B keys proportional to true attention weights,
    simple average of sampled values (unbiased estimator).

    Privileged baseline (needs true weights). Sampling from
    the true distribution and averaging gives E[output] =
    exact attention output.
    """

    @property
    def name(self) -> str:
        return "OracleSampling"

    @property
    def kind(self) -> str:
        return "baseline"

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
        n_keys = len(logits)

        true_weights = softmax(logits)
        buse = min(budget, n_keys)

        sampled = rng.choice(
            n_keys, size=buse,
            p=true_weights, replace=True,
        )
        output = np.mean(values[sampled], axis=0)

        return AttentionOutput(
            output=output,
            actual_budget=len(np.unique(sampled)),
            selected_indices=np.unique(sampled),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [OracleSampling()]


class OracleGrouping(AttentionAlgorithm):
    """
    Sort non-special keys by true logit, create doubling
    groups (sizes 1, 1, 2, 4, 8, ...), represent each
    group by its mean key/value with count-weighted softmax.

    Single fixed-budget point per query. Budget is ~log2(N).
    """

    def __init__(self, drop_last: bool = False):
        self._drop_last = drop_last

    @property
    def name(self) -> str:
        if self._drop_last:
            return "Oracle Grouping (no last)"
        return "Oracle Grouping"

    @property
    def kind(self) -> str:
        return "baseline"

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
        sqrt_d = np.sqrt(head_dim)

        if len(candidate_idx) == 0:
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        cand_logits = logits[candidate_idx]
        sort_order = np.argsort(cand_logits)[::-1]
        sorted_idx = candidate_idx[sort_order]

        bounds = make_doubling_boundaries(len(sorted_idx))
        groups = [sorted_idx[s:e] for s, e in bounds]

        if self._drop_last and len(groups) > 1:
            groups = groups[:-1]

        n_special = len(special_idx)
        n_total = n_special + len(groups)
        scores = np.empty(n_total)
        out_vals = np.empty((n_total, head_dim))

        scores[:n_special] = logits[special_idx]
        out_vals[:n_special] = values[special_idx]

        for gi, idx in enumerate(groups):
            avg_k = np.mean(keys[idx], axis=0)
            avg_v = np.mean(values[idx], axis=0)
            scores[n_special + gi] = (
                q @ avg_k / sqrt_d + np.log(len(idx))
            )
            out_vals[n_special + gi] = avg_v

        w = softmax(scores)
        output = w @ out_vals

        return AttentionOutput(
            output=output,
            actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [
            OracleGrouping(drop_last=False),
            OracleGrouping(drop_last=True),
        ]
