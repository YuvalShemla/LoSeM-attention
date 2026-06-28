"""
KVSculpt: distill the candidate KV region into unconstrained pairs (L-BFGS + lstsq).

At evaluation, compressed pairs are concatenated with exact special (sink + local
window) tokens and attention is computed with the same ``weighted_attention`` path
as other coreset methods (unit weights).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import time

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import softmax
from ..probe_queries import (
    DEFAULT_N_SYNTHETIC,
    DEFAULT_N_TRAIN_QUERIES,
    DEFAULT_ROPE_THETA,
    n_train_queries_list,
    prepare_probe_queries,
    validate_train_q_strategy,
)
from ..wildcat2._device import resolve_device
from ..wildcat2.weighted_attention import weighted_attention
from .kvsculpt_distill import (
    distill_kv_cache,
)


class KVSculpt(AttentionAlgorithm):
    """KV cache compression as distillation (arXiv:2603.27819)."""

    def __init__(
        self,
        n_synthetic: int = 128,
        n_k_steps: int = 100,
        v_solve_every: int = 5,
        lbfgs_lr: float = 0.5,
        lbfgs_inner_iter: int = 10,
        ridge_lambda: float = 1e-3,
        rope_theta: float = DEFAULT_ROPE_THETA,
        exact_denominator: bool = True,
        n_sink: int = 1,
        local_window: int = 1024,
        n_train_queries: int = DEFAULT_N_TRAIN_QUERIES,
        train_q_strategy: str = "kvsculpt",
        device: Optional[str] = None,
    ):
        self.n_synthetic = int(n_synthetic)
        self.n_k_steps = int(n_k_steps)
        self.v_solve_every = int(v_solve_every)
        self.lbfgs_lr = float(lbfgs_lr)
        self.lbfgs_inner_iter = int(lbfgs_inner_iter)
        self.ridge_lambda = float(ridge_lambda)
        self.rope_theta = float(rope_theta)
        self.exact_denominator = bool(exact_denominator)
        self.n_sink = int(n_sink)
        self.local_window = int(local_window)
        self.n_train_queries = int(n_train_queries)
        self.train_q_strategy = validate_train_q_strategy(train_q_strategy)
        self._device = resolve_device(device)

        self._keys: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._head_dim: Optional[int] = None
        self._probe_queries: Optional[np.ndarray] = None
        self._ref_pos: Optional[int] = None
        self._seed = 42
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def name(self) -> str:
        return "KVSculpt"

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
        self.reset_method_timing()
        self._keys = keys
        self._values = values
        self._head_dim = head_dim
        self._seed = int(seed)
        self._cache = {}

        if queries is None:
            self._probe_queries = None
            self._ref_pos = None
            return

        self._ref_pos, self._probe_queries = prepare_probe_queries(
            queries,
            query_positions,
            head_dim,
            self.n_sink,
            self.local_window,
            self.train_q_strategy,
            self.n_train_queries,
            self.n_synthetic,
            self.rope_theta,
            self._seed,
        )

    def _get_distilled(self, budget: int) -> Tuple[np.ndarray, np.ndarray]:
        if budget in self._cache:
            return self._cache[budget]
        if self._keys is None or self._values is None or self._head_dim is None:
            raise RuntimeError("Call prepare() before run()")
        if self._probe_queries is None or self._probe_queries.shape[0] == 0:
            raise RuntimeError(
                "prepare() received no queries; cannot distill KV cache",
            )
        if self._ref_pos is None:
            raise RuntimeError("prepare() missing reference position")

        t0 = time.perf_counter()
        k_c, v_c = distill_kv_cache(
            self._keys,
            self._values,
            self._head_dim,
            self._probe_queries,
            self._ref_pos,
            budget,
            self.n_sink,
            self.local_window,
            n_k_steps=self.n_k_steps,
            v_solve_every=self.v_solve_every,
            lbfgs_lr=self.lbfgs_lr,
            lbfgs_inner_iter=self.lbfgs_inner_iter,
            ridge_lambda=self.ridge_lambda,
            device=self._device,
            seed=self._seed + budget,
        )
        self.record_coreset_fit(budget, time.perf_counter() - t0)
        self._cache[budget] = (k_c, v_c)
        return k_c, v_c

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

        k_c_np, v_c_np = self._get_distilled(budget)
        k_c = torch.as_tensor(
            k_c_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        v_c = torch.as_tensor(
            v_c_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        n_comp = int(k_c.shape[1])

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
            core_keys = torch.cat([sp_keys, k_c], dim=1)
            core_values = torch.cat([sp_vals, v_c], dim=1)
            core_one = torch.cat([sp_one, torch.ones((1, n_comp), device=device)], dim=-1)
        else:
            core_keys = k_c
            core_values = v_c
            core_one = torch.ones((1, n_comp), dtype=torch.float32, device=device)

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
            actual_budget=n_sp + n_comp,
            selected_indices=None,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        for n_train in n_train_queries_list(cfg):
            instances.append(
                KVSculpt(
                    n_synthetic=int(cfg.get("n_synthetic", 128)),
                    n_k_steps=int(cfg.get("n_k_steps", 100)),
                    v_solve_every=int(cfg.get("v_solve_every", 5)),
                    lbfgs_lr=float(cfg.get("lbfgs_lr", 0.5)),
                    lbfgs_inner_iter=int(cfg.get("lbfgs_inner_iter", 10)),
                    ridge_lambda=float(cfg.get("ridge_lambda", 1e-3)),
                    rope_theta=float(cfg.get("rope_theta", DEFAULT_ROPE_THETA)),
                    exact_denominator=bool(cfg.get("exact_denominator", True)),
                    n_sink=int(cfg.get("n_sink", 1)),
                    local_window=int(cfg.get("local_window", 1024)),
                    n_train_queries=int(n_train),
                    train_q_strategy=cfg.get("train_q_strategy", "kvsculpt"),
                    device=cfg.get("device"),
                ),
            )
        return instances
