"""Tensor FC Frank-Wolfe under the query-weighted lq norm."""

from .algorithm import TensorFCFWLq
from .select_lq import (
    build_attention_profiles,
    build_lq_design_and_target,
    lq_objective,
    select_lq_coreset,
)

__all__ = [
    "TensorFCFWLq",
    "build_attention_profiles",
    "build_lq_design_and_target",
    "lq_objective",
    "select_lq_coreset",
]
