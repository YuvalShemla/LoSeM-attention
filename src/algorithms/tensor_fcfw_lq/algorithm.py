"""
Tensor FC Frank-Wolfe under the query-weighted ``lq`` norm.

Like ``tensor_fcfw_l2`` (value-aware coreset of the candidate region, sink +
local window handled exactly via ``weighted_attention``), but the coreset is
selected and corrected to minimize a *query-weighted* error instead of the
RKHS-l2 (Frobenius) error:

    ||sigma - sigma'||_*  =  E_{q in Q} | (sigma - sigma') . psi(q) | ,

with ``Q`` the set of earlier-context queries (the same probe set the
``learned`` method uses). See ``select_lq`` for the full derivation.

**Lifecycle (matches ``learned``).** The coreset is built **once** per
example/head at the reference (last test) position in ``prepare`` / the first
``run()`` for each budget. At evaluation only the exact sink + **local window**
change with the query position; the compressed middle pairs are reused verbatim.
This matches the intended regime where all test queries lie in the trailing
local window (``n_queries`` << ``local_window``), so every coreset key remains
causal for every query.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import compute_special_indices, softmax
from ..wildcat2._device import resolve_device
from ..wildcat2.weighted_attention import weighted_attention
from ..probe_queries import (
    DEFAULT_N_SYNTHETIC,
    DEFAULT_N_TRAIN_QUERIES,
    DEFAULT_ROPE_THETA,
    DEFAULT_TRAIN_Q_STRATEGY,
    n_train_queries_int,
    prepare_probe_queries,
    validate_train_q_strategy,
)
from .lbfgs_refine import refine_coreset_lbfgs
from .select_lq import select_lq_coreset


class TensorFCFWLq(AttentionAlgorithm):
    """Tensor FCFW coreset under the query-weighted lq norm over context queries."""

    def __init__(
        self,
        n_train_queries: int = DEFAULT_N_TRAIN_QUERIES,
        oracle: str = "fw",
        irls_iters: int = 5,
        rcond: float = 1e-3,
        exact_denominator: bool = True,
        n_sink: int = 1,
        local_window: int = 1024,
        train_q_strategy: str = DEFAULT_TRAIN_Q_STRATEGY,
        n_synthetic: int = DEFAULT_N_SYNTHETIC,
        rope_theta: float = DEFAULT_ROPE_THETA,
        device: Optional[str] = None,
        show_progress: Optional[bool] = None,
        scoring_irls_iters: Optional[int] = None,
        correction_irls_iters: Optional[int] = None,
        correction_period: int = 400,
        lbfgs_steps: int = 0,
        lbfgs_lr: float = 0.5,
        lbfgs_inner_iter: int = 10,
    ):
        if oracle not in ("fw", "omp", "fc_lq", "residual_lq", "residual_lq_deflated"):
            raise ValueError(
                f"oracle must be 'fw', 'omp', 'fc_lq', 'residual_lq', or "
                f"'residual_lq_deflated'; got {oracle!r}",
            )
        self.n_train_queries = int(n_train_queries)
        self.oracle = oracle
        self.irls_iters = int(irls_iters)
        self.scoring_irls_iters = (
            int(scoring_irls_iters)
            if scoring_irls_iters is not None
            else self.irls_iters
        )
        self.correction_irls_iters = (
            int(correction_irls_iters)
            if correction_irls_iters is not None
            else self.irls_iters
        )
        self.correction_period = int(correction_period)
        self.lbfgs_steps = max(int(lbfgs_steps), 0)
        self.lbfgs_lr = float(lbfgs_lr)
        self.lbfgs_inner_iter = int(lbfgs_inner_iter)
        self.rcond = float(rcond)
        self.exact_denominator = bool(exact_denominator)
        self.n_sink = int(n_sink)
        self.local_window = int(local_window)
        self.train_q_strategy = validate_train_q_strategy(train_q_strategy)
        self.n_synthetic = int(n_synthetic)
        self.rope_theta = float(rope_theta)
        self._device = resolve_device(device)
        self.show_progress = show_progress

        self._keys: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._head_dim: Optional[int] = None
        self._probe_queries: Optional[np.ndarray] = None
        self._ref_pos: Optional[int] = None
        self._seed = 42
        # budget -> (K_core, V_core, w, global_idx)
        self._coreset_cache: Dict[
            int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._select_state: Optional[dict] = None

    @property
    def name(self) -> str:
        return f"TFCFW-lq-{self.oracle}"

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
        self._coreset_cache = {}
        self._select_state = None

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

    def _get_coreset(
        self,
        budget: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build or return cached coreset at the reference position."""
        if budget in self._coreset_cache:
            return self._coreset_cache[budget]

        if self._keys is None or self._values is None or self._head_dim is None:
            raise RuntimeError("Call prepare() before run()")
        if self._probe_queries is None or self._probe_queries.shape[0] == 0:
            raise RuntimeError(
                "prepare() received no queries; cannot build lq coreset",
            )
        if self._ref_pos is None:
            raise RuntimeError("reference position not set in prepare()")

        t0 = time.perf_counter()
        keys = self._keys
        values = self._values
        head_dim = self._head_dim
        ref_pos = self._ref_pos
        device = self._device
        scale = 1.0 / np.sqrt(head_dim)

        n_causal = ref_pos + 1
        sp_idx, cand_idx = compute_special_indices(
            n_causal, self.n_sink, self.local_window,
        )
        if len(cand_idx) == 0:
            empty = (
                np.zeros((0, keys.shape[1]), dtype=np.float32),
                np.zeros((0, values.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
            self.record_coreset_fit(budget, time.perf_counter() - t0)
            self._coreset_cache[budget] = empty
            return empty

        cand_keys = torch.as_tensor(
            keys[cand_idx], dtype=torch.float32, device=device,
        )
        cand_vals = torch.as_tensor(
            values[cand_idx], dtype=torch.float32, device=device,
        )
        probes = torch.as_tensor(
            self._probe_queries, dtype=torch.float32, device=device,
        )
        ref_keys = ref_vals = sp_t = None
        if self.exact_denominator:
            ref_keys = torch.as_tensor(
                keys[:n_causal], dtype=torch.float32, device=device,
            )
            ref_vals = torch.as_tensor(
                values[:n_causal], dtype=torch.float32, device=device,
            )
            sp_t = torch.as_tensor(sp_idx, dtype=torch.long, device=device)

        core_local, cmpd_values, w, new_state = select_lq_coreset(
            probes,
            cand_keys,
            cand_vals,
            budget,
            scale,
            oracle=self.oracle,
            irls_iters=self.irls_iters,
            rcond=self.rcond,
            state=self._select_state,
            numerator_only=self.exact_denominator,
            ref_keys=ref_keys,
            ref_values=ref_vals,
            sp_idx=sp_t,
            show_progress=self.show_progress,
            scoring_irls_iters=self.scoring_irls_iters,
            correction_irls_iters=self.correction_irls_iters,
            correction_period=self.correction_period,
        )
        self._select_state = new_state

        core_local_np = core_local.cpu().numpy()
        k_core = keys[cand_idx[core_local_np]].astype(np.float32)
        v_core = cmpd_values.cpu().numpy().astype(np.float32)
        w_np = w.cpu().numpy().astype(np.float32)
        global_idx = cand_idx[core_local_np].astype(np.int64)

        if self.lbfgs_steps > 0 and k_core.shape[0] > 0:
            keys_ref_t = torch.as_tensor(
                keys[:n_causal], dtype=torch.float32, device=device,
            )
            values_ref_t = torch.as_tensor(
                values[:n_causal], dtype=torch.float32, device=device,
            )
            sp_t = torch.as_tensor(sp_idx, dtype=torch.long, device=device)
            k_t = torch.as_tensor(k_core, dtype=torch.float32, device=device)
            v_t = torch.as_tensor(v_core, dtype=torch.float32, device=device)
            w_t = torch.as_tensor(w_np, dtype=torch.float32, device=device)
            k_t, v_t, w_t = refine_coreset_lbfgs(
                k_t,
                v_t,
                w_t,
                probes,
                keys_ref_t,
                values_ref_t,
                sp_t,
                scale,
                n_steps=self.lbfgs_steps,
                lbfgs_lr=self.lbfgs_lr,
                lbfgs_inner_iter=self.lbfgs_inner_iter,
                seed=self._seed + budget,
            )
            k_core = k_t.cpu().numpy().astype(np.float32)
            v_core = v_t.cpu().numpy().astype(np.float32)
            w_np = w_t.cpu().numpy().astype(np.float32)

        self.record_coreset_fit(budget, time.perf_counter() - t0)
        self._coreset_cache[budget] = (k_core, v_core, w_np, global_idx)
        return k_core, v_core, w_np, global_idx

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        del rng  # deterministic
        special_idx = problem.special_idx
        n_sp = len(special_idx)

        if self._keys is None:
            raise RuntimeError("Call prepare() before run()")

        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        scale = 1.0 / np.sqrt(head_dim)
        device = self._device

        k_core_np, v_core_np, w_np, global_idx = self._get_coreset(budget)
        n_core = int(k_core_np.shape[0])

        if n_core == 0 and n_sp == 0:
            return AttentionOutput(
                output=np.zeros(head_dim, dtype=np.float32),
                actual_budget=0,
                selected_indices=np.array([], dtype=np.int64),
            )

        if n_core == 0:
            out = softmax(problem.logits[special_idx]) @ problem.values[special_idx]
            return AttentionOutput(
                output=out,
                actual_budget=n_sp,
                selected_indices=special_idx,
            )

        k_core = torch.as_tensor(
            k_core_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        v_core = torch.as_tensor(
            v_core_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        w_core = torch.as_tensor(
            w_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)

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
            core_keys = torch.cat([sp_keys, k_core], dim=1)
            core_values = torch.cat([sp_vals, v_core], dim=1)
            core_one = torch.cat([sp_one, w_core], dim=-1)
        else:
            core_keys = k_core
            core_values = v_core
            core_one = w_core

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

        selected = (
            np.concatenate([special_idx, global_idx]).astype(np.int64)
            if n_sp > 0 else global_idx
        )

        return AttentionOutput(
            output=output,
            actual_budget=n_sp + n_core,
            selected_indices=selected,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [
            TensorFCFWLq(
                n_train_queries=n_train_queries_int(cfg),
                oracle=cfg.get("oracle", "fw"),
                irls_iters=int(cfg.get("irls_iters", 5)),
                rcond=float(cfg.get("rcond", 1e-3)),
                exact_denominator=bool(cfg.get("exact_denominator", True)),
                n_sink=int(cfg.get("n_sink", 1)),
                local_window=int(cfg.get("local_window", 1024)),
                train_q_strategy=cfg.get(
                    "train_q_strategy", DEFAULT_TRAIN_Q_STRATEGY,
                ),
                n_synthetic=int(cfg.get("n_synthetic", DEFAULT_N_SYNTHETIC)),
                rope_theta=float(cfg.get("rope_theta", DEFAULT_ROPE_THETA)),
                device=cfg.get("device"),
                show_progress=cfg.get("show_progress"),
                scoring_irls_iters=cfg.get("scoring_irls_iters"),
                correction_irls_iters=cfg.get("correction_irls_iters"),
                correction_period=int(cfg.get("correction_period", 400)),
                lbfgs_steps=int(cfg.get("lbfgs_steps", 0)),
                lbfgs_lr=float(cfg.get("lbfgs_lr", 0.5)),
                lbfgs_inner_iter=int(cfg.get("lbfgs_inner_iter", 10)),
            ),
        ]
