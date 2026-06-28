"""Stub — ValueClusterIS was removed but __init__.py still references it."""
from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
import numpy as np
from ..core import softmax

class ValueClusterIS(AttentionAlgorithm):
    def __init__(self, **kwargs): pass
    @property
    def name(self): return "ValueClusterIS"
    @property
    def sweeps_budget(self): return False
    def run(self, problem, budget, rng):
        out = softmax(problem.logits[problem.special_idx]) @ problem.values[problem.special_idx]
        return AttentionOutput(output=out, actual_budget=len(problem.special_idx))
    @staticmethod
    def expand_from_config(cfg): return []
