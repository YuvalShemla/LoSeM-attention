"""
CAKE (Cascading Adaptive KV Cache Eviction) — Qin et al., ICLR 2025.

Per-token importance: mean(attn) + gamma * var(attn) over an observation
window. The main paper contribution (layer-adaptive budgeting) applies
across layers; within a single head the algorithm is a variance-aware
token eviction policy.

Paper: "CAKE: Cascading and Adaptive KV Cache Eviction for Long Context
        LLM Inference"

Algorithm (per head):
  1. (prepare) Compute attention weights from the last Sw=32 train
     queries to all keys.
  2. For each key j: score[j] = mean(w[:,j]) + gamma * var(w[:,j])
     where gamma=200 heavily weights tokens whose importance fluctuates.
  3. (run) Select top-B candidates by score, exact subset attention.
"""

import numpy as np
from .base import AttentionAlgorithm, AttentionOutput
from ..core import softmax, subset_attention


class CAKEEviction(AttentionAlgorithm):
    """CAKE: mean + gamma*var importance scoring."""

    def __init__(self, window_size=32, gamma=200):
        self.window_size = window_size
        self.gamma = gamma
        self._importance = None

    @property
    def name(self) -> str:
        return f"CAKE-w{self.window_size}"

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

        ws = min(self.window_size, len(Q_train))
        Q_window = Q_train[-ws:]

        # Per-query attention weights: [ws, n_keys]
        logits = (Q_window @ K_train.T) / sqrt_d
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)

        # CAKE scoring: mean + gamma * var
        mean_w = weights.mean(axis=0)
        var_w = weights.var(axis=0)
        score = mean_w + self.gamma * var_w

        self._importance = np.zeros(N, np.float64)
        self._importance[train_key_idx] = score

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
        ws = cfg.get('window_size', 32)
        gamma = cfg.get('gamma', 200)
        return [CAKEEviction(ws, gamma)]
