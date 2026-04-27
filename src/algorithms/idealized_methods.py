"""
Idealized attention methods: IdealTopK, IdealSampling,
IdealEqualSplits, IdealEqualWeightSplits.

Always included in every evaluation. These represent the
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


def _group_scores_and_values(
    groups, keys, values, query, sqrt_d,
):
    """
    Vectorized mean-key scores and mean-values for a
    list of index groups.  Avoids a Python loop with
    per-group np.mean calls by concatenating all indices,
    computing a single scatter-based mean, then gathering.
    """
    n_groups = len(groups)
    d = keys.shape[1]

    # Build flat labels array
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    flat_idx = np.concatenate(groups)
    labels = np.repeat(np.arange(n_groups), sizes)

    # Per-group sums via bincount (one call per column)
    sum_k = np.empty((n_groups, d), dtype=np.float64)
    sum_v = np.empty((n_groups, d), dtype=np.float64)
    k_flat = keys[flat_idx]
    v_flat = values[flat_idx]
    for j in range(d):
        sum_k[:, j] = np.bincount(
            labels, weights=k_flat[:, j].astype(np.float64),
            minlength=n_groups,
        )
        sum_v[:, j] = np.bincount(
            labels, weights=v_flat[:, j].astype(np.float64),
            minlength=n_groups,
        )

    # Mean = sum / count
    sizes_f = sizes.astype(np.float64)[:, np.newaxis]
    avg_k = (sum_k / sizes_f).astype(np.float32)
    avg_v = (sum_v / sizes_f).astype(np.float32)

    scores = (
        avg_k @ query / sqrt_d
        + np.log(sizes.astype(np.float64))
    )
    return scores, avg_v


def _equal_size_split(indices, n_groups):
    """Split indices into n_groups balanced groups.

    Sizes differ by at most 1: the first (n % n_groups)
    groups get one extra element.
    """
    return [
        np.asarray(g)
        for g in np.array_split(indices, n_groups)
        if len(g) > 0
    ]


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

        all_idx = np.concatenate(
            [special_idx, topk_idx],
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
        return [IdealTopK()]


class IdealSamplingSubset(AttentionAlgorithm):
    """
    Sample B candidates proportional to true attention
    weights, then renormalize softmax over the selected
    subset (special + sampled). Classic subset attention.
    """

    @property
    def name(self) -> str:
        return "IdealSampling-Subset"

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
        cand_w = softmax(cand_logits)
        chosen = rng.choice(
            n_cand, size=buse,
            p=cand_w, replace=False,
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
        return [IdealSamplingSubset()]


class IdealSamplingIS(AttentionAlgorithm):
    """
    Sample B candidates proportional to true attention
    weights. Because p_i = w_i, the IS weights cancel:
      candidate contribution = W_cand · (1/B) Σ v_j

    Special keys get exact w_i · v_i contribution.
    Zero-variance estimator for the numerator; only
    sampling noise in the value average remains.
    """

    @property
    def name(self) -> str:
        return "IdealSampling-IS"

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

        # Full attention weights (over all keys)
        all_w = softmax(logits).astype(np.float64)

        # Sample candidates proportional to their weights
        cand_w = all_w[candidate_idx]
        cand_w_sum = cand_w.sum()
        if cand_w_sum < 1e-30:
            probs = np.ones(n_cand) / n_cand
        else:
            probs = cand_w / cand_w_sum

        chosen = rng.choice(
            n_cand, size=buse,
            p=probs, replace=False,
        )
        sampled_idx = candidate_idx[chosen]

        # IS with weight cancellation:
        # Special: exact w_i · v_i
        # Candidates: W_cand · (1/B) · Σ v_j
        d = values.shape[1]
        output = np.zeros(d, dtype=np.float64)

        # Special contribution (exact)
        output += (
            all_w[special_idx][:, None]
            * values[special_idx].astype(np.float64)
        ).sum(axis=0)

        # Candidate contribution (IS, weights cancel)
        avg_v = np.mean(
            values[sampled_idx].astype(np.float64),
            axis=0,
        )
        output += cand_w_sum * avg_v

        return AttentionOutput(
            output=output.astype(np.float32),
            actual_budget=len(special_idx) + buse,
            selected_indices=np.concatenate([
                special_idx, sampled_idx,
            ]).astype(np.int64),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [IdealSamplingIS()]


class IdealTopKPlusUniform(AttentionAlgorithm):
    """
    Half budget → top-B/2 keys by logit.
    Other half → B/2 keys sampled uniformly at random
    from the remaining candidates.
    Subset attention over the union + special.
    """

    @property
    def name(self) -> str:
        return "IdealTopK+Uniform"

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
        b_topk = buse // 2
        b_uniform = buse - b_topk

        # Top-K half
        cand_logits = logits[candidate_idx]
        if b_topk > 0 and b_topk < n_cand:
            top_pos = np.argpartition(
                cand_logits, -b_topk,
            )[-b_topk:]
        elif b_topk >= n_cand:
            top_pos = np.arange(n_cand)
        else:
            top_pos = np.array([], dtype=np.int64)

        # Uniform half from remaining
        remaining_mask = np.ones(n_cand, dtype=bool)
        remaining_mask[top_pos] = False
        remaining_pos = np.where(remaining_mask)[0]

        if b_uniform > 0 and len(remaining_pos) > 0:
            n_sample = min(b_uniform, len(remaining_pos))
            uniform_pos = rng.choice(
                remaining_pos, size=n_sample,
                replace=False,
            )
        else:
            uniform_pos = np.array([], dtype=np.int64)

        selected_local = np.concatenate(
            [top_pos, uniform_pos],
        ).astype(np.int64)
        selected_global = candidate_idx[selected_local]

        all_idx = np.concatenate(
            [special_idx, selected_global],
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
        return [IdealTopKPlusUniform()]


class VAttentionOracle(AttentionAlgorithm):
    """
    vAttention (Desai et al. 2025) with oracle top-k.

    Splits the budget in half:
      - Half for oracle top-k (highest logit candidates)
      - Half for uniform random sampling from the rest

    Uses IS-corrected attention (Eq. 5 from the paper):
      N = Σ_{fixed} exp(s_i)·V[i] + (n_s/b)·Σ_{sampled} exp(s_j)·V[j]
      D = Σ_{fixed} exp(s_i)       + (n_s/b)·Σ_{sampled} exp(s_j)
      Output = N / D

    This estimates the FULL denominator (not renormalized
    subset softmax), reducing bias from missing tokens.
    Fixed tokens = special (sink + local) + top-k.
    """

    @property
    def name(self) -> str:
        return "vAttention(oracle)"

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
        b_topk = buse // 2
        b_sample = buse - b_topk

        # Oracle top-k half
        cand_logits = logits[candidate_idx]
        if b_topk > 0 and b_topk < n_cand:
            top_pos = np.argpartition(
                cand_logits, -b_topk,
            )[-b_topk:]
        elif b_topk >= n_cand:
            top_pos = np.arange(n_cand)
        else:
            top_pos = np.array([], dtype=np.int64)

        # Uniform sample from remaining
        remaining_mask = np.ones(n_cand, dtype=bool)
        if len(top_pos) > 0:
            remaining_mask[top_pos] = False
        remaining_pos = np.where(remaining_mask)[0]
        n_s = len(remaining_pos)  # residual pool size

        if b_sample > 0 and n_s > 0:
            n_sample = min(b_sample, n_s)
            sampled_pos = rng.choice(
                remaining_pos, size=n_sample,
                replace=False,
            )
        else:
            sampled_pos = np.array([], dtype=np.int64)
            n_sample = 0

        # Global indices
        topk_global = candidate_idx[top_pos]
        sampled_global = candidate_idx[sampled_pos]
        fixed_idx = np.concatenate(
            [special_idx, topk_global],
        ).astype(np.int64)

        # IS-corrected attention (vAttention Eq. 5)
        # Numerical stability: subtract max logit
        all_logits = np.concatenate([
            logits[fixed_idx],
            logits[sampled_global],
        ]).astype(np.float64)
        s_max = np.max(all_logits) if len(all_logits) > 0 else 0.0

        d = values.shape[1]

        # Fixed part: exp(s_i - s_max)
        fixed_s = logits[fixed_idx].astype(np.float64)
        fixed_exp = np.exp(fixed_s - s_max)
        N_f = (
            fixed_exp[:, None]
            * values[fixed_idx].astype(np.float64)
        ).sum(axis=0)
        D_f = fixed_exp.sum()

        # Sampled part: (n_s / b) * exp(s_j - s_max)
        if n_sample > 0 and n_s > 0:
            samp_s = logits[sampled_global].astype(np.float64)
            samp_exp = np.exp(samp_s - s_max)
            weight = float(n_s) / float(n_sample)
            N_dyn = weight * (
                samp_exp[:, None]
                * values[sampled_global].astype(np.float64)
            ).sum(axis=0)
            D_dyn = weight * samp_exp.sum()
        else:
            N_dyn = np.zeros(d, dtype=np.float64)
            D_dyn = 0.0

        D_total = D_f + D_dyn
        if D_total < 1e-30:
            output = np.zeros(d, dtype=np.float32)
        else:
            output = ((N_f + N_dyn) / D_total).astype(
                np.float32,
            )

        actual_budget = len(fixed_idx) + n_sample
        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [VAttentionOracle()]


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

        groups = _equal_size_split(sorted_idx, num_groups)

        n_special = len(special_idx)
        n_groups = len(groups)
        n_total = n_special + n_groups
        scores = np.empty(n_total)
        out_vals = np.empty((n_total, head_dim))

        scores[:n_special] = logits[special_idx]
        out_vals[:n_special] = values[special_idx]

        scores[n_special:], out_vals[n_special:] = (
            _group_scores_and_values(
                groups, keys, values, q, sqrt_d,
            )
        )

        w = softmax(scores)
        output = w @ out_vals

        return AttentionOutput(
            output=output,
            actual_budget=n_total,
            grouped_member_indices=groups,
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
        n_groups = len(groups)
        n_total = n_special + n_groups
        scores = np.empty(n_total)
        out_vals = np.empty((n_total, head_dim))

        scores[:n_special] = logits[special_idx]
        out_vals[:n_special] = values[special_idx]

        scores[n_special:], out_vals[n_special:] = (
            _group_scores_and_values(
                groups, keys, values, q, sqrt_d,
            )
        )

        w = softmax(scores)
        output = w @ out_vals

        return AttentionOutput(
            output=output,
            actual_budget=n_total,
            grouped_member_indices=groups,
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

        Always produces exactly num_groups groups (or n
        groups if n < num_groups) by falling back to
        equal-sized splits for any segment that the
        weight-based splitting cannot subdivide further.
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
            return _equal_size_split(
                sorted_idx, num_groups,
            )

        # Target cumulative weight boundaries
        targets = np.linspace(
            0, total, num_groups + 1,
        )[1:-1]
        split_indices = np.searchsorted(
            cumsum, targets,
        )
        split_indices = np.clip(split_indices, 1, n - 1)

        # Build initial segments from weight boundaries
        # (may be fewer than num_groups due to duplicates)
        boundaries = list(
            dict.fromkeys(split_indices.tolist())
        )
        segments = []
        prev = 0
        for sp in boundaries:
            if sp > prev:
                segments.append((prev, sp))
            prev = sp
        if prev < n:
            segments.append((prev, n))

        # Subdivide segments until we have num_groups
        while len(segments) < num_groups:
            # Find the largest segment to split
            best = max(
                range(len(segments)),
                key=lambda i: segments[i][1] - segments[i][0],
            )
            s, e = segments[best]
            if e - s < 2:
                break  # can't split single-element
            mid = (s + e) // 2
            segments[best:best + 1] = [
                (s, mid), (mid, e),
            ]

        groups = [
            sorted_idx[s:e] for s, e in segments
        ]
        return groups

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [IdealEqualWeightSplits()]
