"""
Base classes and data structures for attention algorithms.

AttentionInput/Output dataclasses define the algorithm I/O.
AttentionAlgorithm ABC defines the interface every method
must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


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


class AttentionAlgorithm(ABC):
    """Base class for all attention methods."""

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
        """'baseline' or 'algorithm'."""
        return "algorithm"

    @property
    def sweeps_budget(self) -> bool:
        """True if runner should sweep budget values."""
        return False

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        """Generate all param combos from config."""
        raise NotImplementedError
