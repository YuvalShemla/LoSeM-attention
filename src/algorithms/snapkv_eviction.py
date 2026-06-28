"""
SnapKV — Li et al., 2024.

Token selection using an observation window of recent queries.
Attention from the window queries votes for which earlier keys to keep.
Scores are smoothed with average pooling before selection.

Paper: "SnapKV: LLM Knows What You are Looking for Before Generation"

Algorithm:
  1. (prepare) Take the last `window_size` train queries as the
     observation window.  Compute softmax attention from window queries
     to all train keys.  Sum across window queries to get per-key vote
     scores.  Smooth with 1-D average pooling (kernel_size=5).
  2. (run) Given budget B, select the top-B candidates by smoothed
     score.  Compute exact subset attention over special + selected.

Key differences from H2O:
  - Uses only RECENT queries (observation window), not all history.
  - Applies pooling to encourage selection of contiguous key regions.
  - Per-head selection (each head has its own scores).
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionOutput
from ..core import softmax, subset_attention


def _avg_pool_1d(x, kernel_size):
    """1-D average pooling with same-size output (edge padding)."""
    if kernel_size <= 1:
        return x.copy()
    pad = kernel_size // 2
    padded = np.pad(x, (pad, pad), mode='edge')
    cumsum = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumsum[kernel_size:] - cumsum[:-kernel_size]) / kernel_size


class SnapKVEviction(AttentionAlgorithm):
    """SnapKV: observation-window voting + pooling for key selection."""

    def __init__(self, window_size=64, kernel_size=5):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self._importance = None

    @property
    def name(self) -> str:
        k_str = f"-k{self.kernel_size}" if self.kernel_size != 5 else ""
        return f"SnapKV-w{self.window_size}{k_str}"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None, seed=42):
        N = keys.shape[0]
        sqrt_d = np.sqrt(head_dim)

        if queries is None or len(queries) < 2:
            self._importance = np.ones(N, np.float64) / N
            return

        test_positions = set(query_positions) if query_positions else set()
        q_train_mask = np.array(
            [i not in test_positions for i in range(len(queries))], bool)
        k_train_mask = np.array(
            [i not in test_positions for i in range(N)], bool)
        Q_train = queries[q_train_mask].astype(np.float64)
        train_key_idx = np.where(k_train_mask)[0]
        K_train = keys[train_key_idx].astype(np.float64)

        # Observation window: last window_size train queries
        ws = min(self.window_size, len(Q_train))
        Q_window = Q_train[-ws:]

        # Attention from window queries to all train keys
        logits = (Q_window @ K_train.T) / sqrt_d
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        vote_score = weights.sum(axis=0)

        # Smooth with average pooling (in sequence order)
        vote_score = _avg_pool_1d(vote_score, self.kernel_size)

        # Map to full key space
        self._importance = np.zeros(N, np.float64)
        self._importance[train_key_idx] = vote_score

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
        cand_scores = self._importance[candidate_idx]
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
        ws_list = cfg.get('window_size', [64])
        if not isinstance(ws_list, list):
            ws_list = [ws_list]
        ks = cfg.get('kernel_size', 5)
        return [SnapKVEviction(ws, ks) for ws in ws_list]
