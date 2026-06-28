"""
Attention Matching (AM) — best-case: train-query-averaged TopK.

Selects the top-B keys by RMS attention importance averaged over
train queries.  The selected set is fixed for all test queries.
At query time, runs subset attention (exact logits + values) over
special tokens + selected keys.

This is AM's best configuration (no beta, no Cv fitting) which
reduces to token eviction with a stale, query-averaged selection.
"""

import numpy as np
from typing import List, Optional

from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax, subset_attention


class AttentionMatchingTopK(AttentionAlgorithm):
    """AM best-case: select top-B keys by train-query RMS attention."""

    def __init__(self):
        self._importance = None  # [N] RMS attention per key

    @property
    def name(self) -> str:
        return "AM-TopK"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        N = keys.shape[0]
        sqrt_d = np.sqrt(head_dim)

        if queries is None or len(queries) < 2:
            self._importance = np.zeros(N, np.float64)
            return

        # Train/test split: exclude test positions
        test_positions = set(query_positions) if query_positions else set()
        q_train_mask = np.array(
            [i not in test_positions for i in range(len(queries))],
            dtype=bool)
        Q_train = queries[q_train_mask].astype(np.float64)
        K_f = keys.astype(np.float64)
        n_train = len(Q_train)

        # Compute RMS attention weight per key across train queries
        # For each train query, compute softmax attention weights over
        # causal keys, accumulate squared weights
        BATCH = 500
        sq_sum = np.zeros(N, np.float64)
        count = np.zeros(N, np.float64)

        for b0 in range(0, n_train, BATCH):
            b1 = min(b0 + BATCH, n_train)
            Q_b = Q_train[b0:b1]
            for i in range(len(Q_b)):
                # This query's position in the original sequence
                qi_global = np.where(q_train_mask)[0][b0 + i]
                n_causal = qi_global + 1
                logits = (Q_b[i] @ K_f[:n_causal].T) / sqrt_d
                attn_w = softmax(logits)
                sq_sum[:n_causal] += attn_w ** 2
                count[:n_causal] += 1.0

        safe_count = np.maximum(count, 1.0)
        self._importance = np.sqrt(sq_sum / safe_count)

    def run(self, problem, budget, rng):
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)

        if n_cand == 0:
            output = subset_attention(logits, values, special_idx)
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
                selected_indices=special_idx,
            )

        buse = min(budget, n_cand)

        # Select top-B candidates by train-query-averaged importance
        cand_importance = self._importance[candidate_idx]
        if buse < n_cand:
            top_pos = np.argpartition(
                cand_importance, -buse)[-buse:]
        else:
            top_pos = np.arange(n_cand)
        topk_idx = candidate_idx[top_pos]

        all_idx = np.concatenate(
            [special_idx, topk_idx]).astype(np.int64)
        output = subset_attention(logits, values, all_idx)

        return AttentionOutput(
            output=output,
            actual_budget=len(all_idx),
            selected_indices=all_idx,
        )

    @staticmethod
    def expand_from_config(cfg):
        return [AttentionMatchingTopK()]
