from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
import numpy as np
from ..core import softmax
class _Stub(AttentionAlgorithm):
    @property
    def name(self): return "tree_attention"
    @property
    def sweeps_budget(self): return False
    def run(self, p, b, r):
        o = softmax(p.logits[p.special_idx]) @ p.values[p.special_idx]
        return AttentionOutput(output=o, actual_budget=len(p.special_idx))
    @staticmethod
    def expand_from_config(cfg): return []
TreeAttention = _Stub
