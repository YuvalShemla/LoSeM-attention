"""
Base classes and data structures for attention algorithms.

AttentionInput/Output dataclasses define the algorithm I/O.
AttentionAlgorithm ABC defines the interface every method
must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np


@dataclass
class MethodTiming:
    """Per-example timing accumulated between ``prepare()`` calls."""

    fit_by_budget: Dict[int, float] = field(default_factory=dict)
    inference_seconds: float = 0.0
    inference_calls: int = 0


@dataclass
class AttentionInput:
    """Everything an algorithm needs for one query."""
    query: np.ndarray              # [head_dim]
    keys: np.ndarray               # [n_causal, head_dim]
    values: np.ndarray             # [n_causal, head_dim]
    head_dim: int
    logits: Optional[np.ndarray] = None   # [n_causal]
    special_idx: Optional[np.ndarray] = None
    candidate_idx: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.special_idx is not None:
            self._special_set = set(
                self.special_idx.tolist()
            )
        else:
            self._special_set = set()

    @property
    def special_set(self) -> set:
        """O(1) membership lookup for special indices."""
        return self._special_set


@dataclass
class AttentionOutput:
    """Everything an algorithm returns."""
    output: np.ndarray             # [head_dim]
    actual_budget: int
    selected_indices: Optional[np.ndarray] = None
    # Optional debug payload for analysis:
    # list of arrays of member key indices, one per
    # grouped representative/bucket used by the method.
    grouped_member_indices: Optional[List[np.ndarray]] = None
    # Optional method-specific debug payload.
    debug_payload: Optional[Dict] = None


class AttentionAlgorithm(ABC):
    """Base class for all attention methods."""

    def reset_method_timing(self) -> None:
        """Clear per-example timing (call at the start of ``prepare()``)."""
        self._method_timing = MethodTiming()

    def record_coreset_fit(self, budget: int, seconds: float) -> None:
        """Record one-time coreset / training cost for ``budget``."""
        if not hasattr(self, "_method_timing"):
            self.reset_method_timing()
        if budget not in self._method_timing.fit_by_budget:
            self._method_timing.fit_by_budget[budget] = float(seconds)

    def record_inference_seconds(self, seconds: float) -> None:
        """Accumulate ``run()`` wall time (including cached forwards)."""
        if not hasattr(self, "_method_timing"):
            self.reset_method_timing()
        self._method_timing.inference_seconds += float(seconds)
        self._method_timing.inference_calls += 1

    def snapshot_method_timing(self) -> MethodTiming:
        """Return timing accumulated since the last ``prepare()``."""
        if not hasattr(self, "_method_timing"):
            return MethodTiming()
        return MethodTiming(
            fit_by_budget=dict(self._method_timing.fit_by_budget),
            inference_seconds=self._method_timing.inference_seconds,
            inference_calls=self._method_timing.inference_calls,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name for plots and logs."""
        ...

    @abstractmethod
    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        """Compute approximate attention output."""
        ...

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: Optional[np.ndarray] = None,
        query_positions: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        """
        Called once per example for precomputation.

        Override in subclasses that need offline setup
        (clustering, sorting, etc.). Default: no-op.
        """
        pass

    @property
    def kind(self) -> str:
        """'idealized' or 'algorithm'."""
        return "algorithm"

    @property
    def sweeps_budget(self) -> bool:
        """True if runner should sweep budget values."""
        return False

    @property
    def point_label(self) -> str:
        """Per-instance label shown next to dot in plots."""
        return ""

    def cluster_quality(self) -> Optional[Dict[str, float]]:
        """
        Grouping quality metrics computed during prepare().

        Returns dict with 'avg_cosine_sim' (weighted avg
        cosine similarity of keys to their group mean) and
        'n_groups' (number of non-empty groups), or None
        if not applicable.
        """
        return None

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        """Generate all param combos from config."""
        raise NotImplementedError
