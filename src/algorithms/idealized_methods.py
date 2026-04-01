"""
Idealized attention methods: IdealTopK, IdealSampling,
IdealEqualSplits, IdealEqualWeightSplits.

Always included in every experiment. These represent the
best achievable accuracy at a given budget because they
use oracle knowledge (true logits) and spend per-query
computation on grouping. Any new algorithm should be
compared against these idealized methods.
"""

import numpy as np

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, subset_attention


class IdealTopK(AttentionAlgorithm):
    """
    Top-B keys by logit + sink/local.

    Biased -- discards the tail entirely, then
    renormalizes softmax over the selected subset.
    """

    @property
    def name(self) -> str:
        return "IdealTopK"

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
        return [IdealTopK()]


class IdealSampling(AttentionAlgorithm):
    """
    Always include special keys (sink + local window),
    then sample `budget` additional candidates
    proportional to the candidate-only attention
    distribution. Renormalize softmax over the union.

    Consistent with IdealTopK: both always include
    special keys and select `budget` candidates on top.
    TopK picks the highest-logit candidates; Sampling
    draws from the renormalized candidate distribution.
    """

    @property
    def name(self) -> str:
        return "IdealSampling"

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

        # Sample with replacement until we collect
        # exactly `buse` unique candidates
        cand_logits = logits[candidate_idx]
        cand_w = softmax(cand_logits)
        unique_pos = set()
        while len(unique_pos) < buse:
            batch = min(
                buse * 2, n_cand * 4,
            )
            drawn = rng.choice(
                n_cand, size=batch,
                p=cand_w, replace=True,
            )
            unique_pos.update(drawn.tolist())
        # Trim to exact budget
        unique_pos = list(unique_pos)[:buse]
        sampled_idx = candidate_idx[unique_pos]

        all_idx = np.unique(
            np.concatenate([special_idx, sampled_idx])
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
        return [IdealSampling()]


class IdealEqualSplits(AttentionAlgorithm):
    """
    Sort non-special keys by true logit, split into
    `budget` equal-sized groups, represent each group
    by its mean key/value with count-weighted softmax.

    Budget scales with the number of groups (= budget
    parameter). This is the simplest per-query grouping
    strategy: uniform partitioning of the sorted keys.
    """

    @property
    def name(self) -> str:
        return "IdealEqualSplits"

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

        n = len(sorted_idx)
        num_groups = min(budget, n)
        if num_groups <= 0:
            num_groups = 1

        # Equal-sized splits
        group_size = max(1, n // num_groups)
        groups = []
        for i in range(num_groups):
            start = i * group_size
            if i < num_groups - 1:
                end = (i + 1) * group_size
            else:
                end = n
            if start >= n:
                break
            groups.append(sorted_idx[start:end])

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
        return [IdealEqualSplits()]


class IdealEqualWeightSplits(AttentionAlgorithm):
    """
    Sort non-special keys by true logit, split into
    `budget` groups so each group captures approximately
    equal total attention weight mass. High-weight keys
    get more groups (finer resolution where it matters).

    Budget scales with the number of groups (= budget
    parameter).
    """

    @property
    def name(self) -> str:
        return "IdealEqualWeightSplits"

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
        cand_weights = softmax(cand_logits)
        sort_order = np.argsort(cand_weights)[::-1]
        sorted_idx = candidate_idx[sort_order]
        sorted_weights = cand_weights[sort_order]

        groups = self._equal_weight_groups(
            sorted_idx, sorted_weights, budget,
        )

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
    def _equal_weight_groups(
        sorted_idx: np.ndarray,
        sorted_weights: np.ndarray,
        num_groups: int,
    ) -> list:
        """
        Split so each group captures ~equal total weight
        mass. High-weight keys get more groups (finer
        resolution where it matters).
        """
        n = len(sorted_idx)
        num_groups = min(num_groups, n)
        if num_groups >= n:
            return [
                sorted_idx[i:i + 1] for i in range(n)
            ]

        cumsum = np.cumsum(sorted_weights)
        total = cumsum[-1]
        if total < 1e-12:
            # Fallback to equal-sized splits
            group_size = max(1, n // num_groups)
            groups = []
            for i in range(num_groups):
                start = i * group_size
                end = (
                    (i + 1) * group_size
                    if i < num_groups - 1
                    else n
                )
                if start >= n:
                    break
                groups.append(sorted_idx[start:end])
            return groups

        # Target cumulative weight boundaries
        targets = np.linspace(
            0, total, num_groups + 1,
        )[1:-1]
        split_indices = np.searchsorted(
            cumsum, targets,
        )
        split_indices = np.unique(
            np.clip(split_indices, 1, n - 1)
        )

        groups = []
        prev = 0
        for sp in split_indices:
            groups.append(sorted_idx[prev:sp])
            prev = sp
        groups.append(sorted_idx[prev:])

        return groups

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [IdealEqualWeightSplits()]
