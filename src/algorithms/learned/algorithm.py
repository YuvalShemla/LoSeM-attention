"""
Learned coreset: gradient-descent synthetic (K', V', w') for the residual region.

Training matches the exact evaluation pipeline (sink + local window exact,
learned pairs for the candidate region, normalized as ``weighted_attention``)
against true full attention over the fixed reference (test) context. Only the
residual pairs are learned; the special tokens are added at evaluation.

The budget sweep is **nested/monotone**: a larger budget freezes the smaller
budget's trained coreset and trains only the new pairs (initialized at near-zero
mass), so error is non-increasing in budget. Initialization is "pure" (k-means
or random over candidates), not FCFW.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import softmax
from ..wildcat2._device import resolve_device
from ..wildcat2.weighted_attention import weighted_attention
from .learn_coreset import (
    build_probe_queries,
    learn_kv_coreset,
    reference_position,
)


class LearnedCoreset(AttentionAlgorithm):
    """Learn B synthetic (K', V', w') on the residual; special tokens at eval only."""

    def __init__(
        self,
        n_train_queries: int = 5000,
        init: str = "kmeans",
        lr: float = 0.05,
        n_steps: int = 500,
        batch_size: int = 128,
        loss: str = "relative_l2",
        rel_l2_floor: float = 0.01,
        val_fraction: float = 0.1,
        early_stop_patience: int = 50,
        lr_decay_step: int = 200,
        lr_decay_gamma: float = 0.5,
        nested_budget: bool = True,
        kmeans_subsample: int = 8192,
        exact_denominator: bool = True,
        n_sink: int = 1,
        local_window: int = 1024,
        device: Optional[str] = None,
    ):
        self.n_train_queries = int(n_train_queries)
        if init not in ("kmeans", "random"):
            raise ValueError(f"init must be 'kmeans' or 'random'; got {init!r}")
        self.init = init
        self.lr = float(lr)
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        if loss not in ("mse", "relative_l2"):
            raise ValueError(
                f"loss must be 'mse' or 'relative_l2'; got {loss!r}",
            )
        self.loss = loss
        self.rel_l2_floor = float(rel_l2_floor)
        self.val_fraction = float(val_fraction)
        self.early_stop_patience = int(early_stop_patience)
        self.lr_decay_step = int(lr_decay_step)
        self.lr_decay_gamma = float(lr_decay_gamma)
        self.nested_budget = bool(nested_budget)
        self.kmeans_subsample = int(kmeans_subsample)
        self.exact_denominator = bool(exact_denominator)
        self.n_sink = int(n_sink)
        self.local_window = int(local_window)
        self._device = resolve_device(device)

        self._keys: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._head_dim: Optional[int] = None
        self._probe_queries: Optional[np.ndarray] = None
        self._ref_pos: Optional[int] = None
        self._seed = 42
        # budget -> (K', V', w')
        self._learned_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def name(self) -> str:
        return f"Learned-{self.init}"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: Optional[np.ndarray] = None,
        query_positions: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        self._keys = keys
        self._values = values
        self._head_dim = head_dim
        self._seed = int(seed)
        self._learned_cache = {}

        if queries is None:
            self._probe_queries = None
            self._ref_pos = None
            return

        self._ref_pos = reference_position(len(queries), query_positions)
        self._probe_queries = build_probe_queries(
            queries, query_positions, self._ref_pos, self.n_train_queries,
        )

    def _frozen_for_budget(
        self, budget: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Largest already-learned coreset smaller than ``budget`` (nested base)."""
        if not self.nested_budget or not self._learned_cache:
            return None
        smaller = [b for b in self._learned_cache if b < budget]
        if not smaller:
            return None
        return self._learned_cache[max(smaller)]

    def _get_learned(
        self,
        budget: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if budget in self._learned_cache:
            return self._learned_cache[budget]
        if self._keys is None or self._values is None:
            raise RuntimeError("Call prepare() before run()")
        if self._probe_queries is None or self._probe_queries.shape[0] == 0:
            raise RuntimeError(
                "prepare() received no queries; cannot learn coreset",
            )
        k_prime, v_prime, w_prime = learn_kv_coreset(
            self._keys,
            self._values,
            self._head_dim,
            self._probe_queries,
            self._ref_pos,
            budget,
            self.n_sink,
            self.local_window,
            init=self.init,
            exact_denominator=self.exact_denominator,
            lr=self.lr,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            loss=self.loss,
            rel_l2_floor=self.rel_l2_floor,
            val_fraction=self.val_fraction,
            early_stop_patience=self.early_stop_patience,
            lr_decay_step=self.lr_decay_step,
            lr_decay_gamma=self.lr_decay_gamma,
            frozen=self._frozen_for_budget(budget),
            kmeans_subsample=self.kmeans_subsample,
            device=self._device,
            seed=self._seed + budget,
        )
        self._learned_cache[budget] = (k_prime, v_prime, w_prime)
        return k_prime, v_prime, w_prime

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        del rng
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)
        n_sp = len(special_idx)

        if n_cand == 0:
            out = softmax(problem.logits[special_idx]) @ problem.values[special_idx]
            return AttentionOutput(
                output=out,
                actual_budget=n_sp,
                selected_indices=special_idx,
            )

        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        scale = 1.0 / np.sqrt(head_dim)
        device = self._device

        k_prime_np, v_prime_np, w_prime_np = self._get_learned(budget)
        k_prime = torch.as_tensor(
            k_prime_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        v_prime = torch.as_tensor(
            v_prime_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        w_prime = torch.as_tensor(
            w_prime_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        # Fold mass into the numerator (weighted_attention ignores core_one in the
        # exact-denominator path); core_one carries it for the coreset-mass path.
        v_prime_eff = v_prime * w_prime.unsqueeze(-1)
        actual_budget_pairs = int(k_prime.shape[1])

        keys_all = torch.as_tensor(
            keys, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        values_all = torch.as_tensor(
            values, dtype=torch.float32, device=device,
        ).unsqueeze(0)

        if n_sp > 0:
            sp_keys = keys_all[:, special_idx, :]
            sp_vals = values_all[:, special_idx, :]
            sp_one = torch.ones(
                (1, n_sp), dtype=torch.float32, device=device,
            )
            core_keys = torch.cat([sp_keys, k_prime], dim=1)
            core_values = torch.cat([sp_vals, v_prime_eff], dim=1)
            core_one = torch.cat([sp_one, w_prime], dim=-1)
        else:
            core_keys = k_prime
            core_values = v_prime_eff
            core_one = w_prime

        q = torch.as_tensor(
            problem.query, dtype=torch.float32, device=device,
        ).unsqueeze(0).unsqueeze(0)

        vmin = values_all.amin(dim=-2, keepdim=True)
        vmax = values_all.amax(dim=-2, keepdim=True)

        all_logits = None
        if self.exact_denominator:
            if problem.logits is None:
                raise ValueError(
                    "exact_denominator requires AttentionInput.logits",
                )
            all_logits = torch.as_tensor(
                problem.logits, dtype=torch.float32, device=device,
            )

        out_t = weighted_attention(
            q, core_keys, core_values, core_one, scale, vmin, vmax,
            all_logits=all_logits,
        )
        output = out_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=n_sp + actual_budget_pairs,
            selected_indices=None,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        n_train_list = cfg.get("n_train_queries", [5000])
        if isinstance(n_train_list, int):
            n_train_list = [n_train_list]
        init = cfg.get("init", "kmeans")
        instances = []
        for n_train in n_train_list:
            instances.append(
                LearnedCoreset(
                    n_train_queries=int(n_train),
                    init=init,
                    lr=float(cfg.get("lr", 0.05)),
                    n_steps=int(cfg.get("n_steps", 500)),
                    batch_size=int(cfg.get("batch_size", 128)),
                    loss=cfg.get("loss", "relative_l2"),
                    rel_l2_floor=float(cfg.get("rel_l2_floor", 0.01)),
                    val_fraction=float(cfg.get("val_fraction", 0.1)),
                    early_stop_patience=int(cfg.get("early_stop_patience", 50)),
                    lr_decay_step=int(cfg.get("lr_decay_step", 200)),
                    lr_decay_gamma=float(cfg.get("lr_decay_gamma", 0.5)),
                    nested_budget=bool(cfg.get("nested_budget", True)),
                    kmeans_subsample=int(cfg.get("kmeans_subsample", 8192)),
                    exact_denominator=bool(cfg.get("exact_denominator", True)),
                    n_sink=int(cfg.get("n_sink", 1)),
                    local_window=int(cfg.get("local_window", 1024)),
                    device=cfg.get("device"),
                ),
            )
        return instances
