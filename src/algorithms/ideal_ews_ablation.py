"""
Ablation variants of IdealEqualWeightSplits.

Same oracle grouping (equal softmax mass), but with
injected noise to isolate error sources.

Noise dimensions: key, value, or both.
Noise scope: all groups, first group only, first half,
second half.
"""

import numpy as np

from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax, subset_attention
from .idealized_methods import (
    IdealEqualWeightSplits, _group_scores_and_values,
)


def _noisy_group_scores_and_values(
    groups, keys, values, query, sqrt_d,
    candidate_idx, rng,
    noise_key: bool = True,
    noise_value: bool = True,
    noise_scope: str = "all",
):
    """
    Vectorized group scoring with optional noise injection.

    noise_scope:
      "all"         — noise every group
      "first"       — noise only group 0
      "first_half"  — noise groups 0..n//2-1
      "second_half" — noise groups n//2..n-1
    """
    n_groups = len(groups)
    d = keys.shape[1]

    sizes = np.array(
        [len(g) for g in groups], dtype=np.int64,
    )
    flat_idx = np.concatenate(groups)
    labels = np.repeat(np.arange(n_groups), sizes)

    k_flat = keys[flat_idx].astype(np.float64)
    v_flat = values[flat_idx].astype(np.float64)

    sum_k = np.empty((n_groups, d), dtype=np.float64)
    sum_v = np.empty((n_groups, d), dtype=np.float64)
    for j in range(d):
        sum_k[:, j] = np.bincount(
            labels, weights=k_flat[:, j],
            minlength=n_groups,
        )
        sum_v[:, j] = np.bincount(
            labels, weights=v_flat[:, j],
            minlength=n_groups,
        )

    sizes_f = sizes.astype(np.float64)

    # Build noise mask
    mask = np.zeros(n_groups, dtype=bool)
    if noise_scope == "all":
        mask[:] = True
    elif noise_scope == "first":
        mask[0] = True
    elif noise_scope == "first_half":
        mask[:n_groups // 2] = True
    elif noise_scope == "second_half":
        mask[n_groups // 2:] = True

    n_noisy = mask.sum()

    if n_noisy > 0:
        noise_indices = rng.choice(
            candidate_idx, size=n_noisy,
        )
        nk = keys[noise_indices].astype(np.float64)
        nv = values[noise_indices].astype(np.float64)

        if noise_key:
            sum_k[mask] += nk
            denom_k = sizes_f.copy()
            denom_k[mask] += 1
            avg_k = sum_k / denom_k[:, None]
        else:
            avg_k = sum_k / sizes_f[:, None]

        if noise_value:
            sum_v[mask] += nv
            denom_v = sizes_f.copy()
            denom_v[mask] += 1
            avg_v = sum_v / denom_v[:, None]
        else:
            avg_v = sum_v / sizes_f[:, None]
    else:
        avg_k = sum_k / sizes_f[:, None]
        avg_v = sum_v / sizes_f[:, None]

    scores = (
        avg_k.astype(np.float32) @ query / sqrt_d
        + np.log(sizes_f)
    )
    return scores, avg_v


class IdealEWSAblation(AttentionAlgorithm):
    """
    Parameterized EWS ablation.

    noise_key/noise_value: which means get corrupted
    noise_scope: which groups get noise
    """

    def __init__(
        self,
        label: str,
        noise_key: bool = True,
        noise_value: bool = True,
        noise_scope: str = "all",
    ):
        self._label = label
        self._noise_key = noise_key
        self._noise_value = noise_value
        self._noise_scope = noise_scope

    @property
    def name(self) -> str:
        return self._label

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

        groups = IdealEqualWeightSplits._equal_weight_groups(
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
            _noisy_group_scores_and_values(
                groups, keys, values, q, sqrt_d,
                candidate_idx, rng,
                noise_key=self._noise_key,
                noise_value=self._noise_value,
                noise_scope=self._noise_scope,
            )
        )

        w = softmax(scores)
        output = w @ out_vals

        return AttentionOutput(
            output=output,
            actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        variants = [
            ("EWS+NoiseK",
             True, False, "all"),
            ("EWS+NoiseV",
             False, True, "all"),
            ("EWS+NoiseKV",
             True, True, "all"),
            ("EWS+NoiseKV-G0",
             True, True, "first"),
            ("EWS+NoiseKV-Hi",
             True, True, "first_half"),
            ("EWS+NoiseKV-Lo",
             True, True, "second_half"),
        ]
        return [
            IdealEWSAblation(label, nk, nv, scope)
            for label, nk, nv, scope in variants
        ]
