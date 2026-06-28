"""
H2O (Heavy-Hitter Oracle) — Zhang et al., NeurIPS 2023.

Token eviction policy: keep sink + local window (special tokens) plus
the top-B candidate keys ranked by accumulated attention score across
train queries.  At query time, runs exact subset attention over the
selected keys.

Paper: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
        of Large Language Models"

Algorithm:
  1. (prepare) For each train query q_i, compute softmax attention
     weights over all train keys.  Accumulate the sum of weights per
     key: score[j] = sum_i softmax(q_i @ K^T / sqrt_d)[j].
  2. (run) Given budget B, select the top-B candidates by accumulated
     score.  Compute exact subset attention over special + selected.

Note: the paper applies eviction ONLINE (one token at a time with
cascading cache compression).  Our offline evaluation gives H2O an
advantage by using the full key set for scoring.
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionOutput
from ..core import softmax, subset_attention


class H2OEviction(AttentionAlgorithm):
    """H2O: select top-B keys by accumulated attention weight."""

    def __init__(self):
        self._accumulated_score = None

    @property
    def name(self) -> str:
        return "H2O"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        N = keys.shape[0]
        sqrt_d = np.sqrt(head_dim)

        if queries is None or len(queries) < 2:
            self._accumulated_score = np.ones(N, np.float64) / N
            return

        test_positions = set(query_positions) if query_positions else set()
        q_train_mask = np.array(
            [i not in test_positions for i in range(len(queries))], bool)
        k_train_mask = np.array(
            [i not in test_positions for i in range(N)], bool)
        Q_train = queries[q_train_mask].astype(np.float64)
        train_key_idx = np.where(k_train_mask)[0]
        K_train = keys[train_key_idx].astype(np.float64)

        # Accumulated attention: sum of softmax weights per key
        n_train = len(Q_train)
        score = np.zeros(len(train_key_idx), np.float64)
        BATCH = 500
        for b0 in range(0, n_train, BATCH):
            b1 = min(b0 + BATCH, n_train)
            logits = (Q_train[b0:b1] @ K_train.T) / sqrt_d
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            score += weights.sum(axis=0)

        self._accumulated_score = np.zeros(N, np.float64)
        self._accumulated_score[train_key_idx] = score

    def run(self, problem, budget, rng):
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)

        if n_cand == 0:
            output = subset_attention(logits, values, special_idx)
            return AttentionOutput(output=output,
                                   actual_budget=len(special_idx))

        buse = min(budget, n_cand)
        cand_scores = self._accumulated_score[candidate_idx]
        if buse < n_cand:
            top_pos = np.argpartition(cand_scores, -buse)[-buse:]
        else:
            top_pos = np.arange(n_cand)
        topk_idx = candidate_idx[top_pos]

        all_idx = np.concatenate([special_idx, topk_idx]).astype(np.int64)
        output = subset_attention(logits, values, all_idx)
        return AttentionOutput(output=output,
                               actual_budget=len(all_idx),
                               selected_indices=all_idx)

    @staticmethod
    def expand_from_config(cfg):
        return [H2OEviction()]
