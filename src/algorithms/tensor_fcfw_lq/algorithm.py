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

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import compute_special_indices, softmax
from ..wildcat2._device import resolve_device
from ..wildcat2.weighted_attention import weighted_attention
from ..learned.learn_coreset import build_probe_queries, reference_position
from ..probe_queries import DEFAULT_N_TRAIN_QUERIES, n_train_queries_int
from .select_lq import select_lq_coreset


class TensorFCFWLq(AttentionAlgorithm):
    """Tensor FCFW coreset under the query-weighted lq norm over context queries."""

    def __init__(
        self,
        n_train_queries: int = DEFAULT_N_TRAIN_QUERIES,
        oracle: str = "fw",
        irls_iters: int = 5,
        rcond: float = 1e-3,
        exact_denominator: bool = False,
        n_sink: int = 1,
        local_window: int = 1024,
        device: Optional[str] = None,
    ):
        if oracle not in ("fw", "omp"):
            raise ValueError(f"oracle must be 'fw' or 'omp'; got {oracle!r}")
        self.n_train_queries = int(n_train_queries)
        self.oracle = oracle
        self.irls_iters = int(irls_iters)
        self.rcond = float(rcond)
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
        # budget -> (K_core, V_core, w, global_idx)
        self._coreset_cache: Dict[
            int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._select_state: Optional[dict] = None

    @property
    def name(self) -> str:
        return "TFCFW-lq"

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
        train_queries: Optional[np.ndarray] = None,
    ) -> None:
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

        self._ref_pos = reference_position(len(queries), query_positions)
        if train_queries is not None:
            self._probe_queries = train_queries.astype(np.float32)
        else:
            self._probe_queries = build_probe_queries(
                queries, query_positions, self._ref_pos, self.n_train_queries,
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

        keys = self._keys
        values = self._values
        head_dim = self._head_dim
        ref_pos = self._ref_pos
        device = self._device
        scale = 1.0 / np.sqrt(head_dim)

        n_causal = ref_pos + 1
        _, cand_idx = compute_special_indices(
            n_causal, self.n_sink, self.local_window,
        )
        if len(cand_idx) == 0:
            empty = (
                np.zeros((0, keys.shape[1]), dtype=np.float32),
                np.zeros((0, values.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
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
        )
        self._select_state = new_state

        core_local_np = core_local.cpu().numpy()
        k_core = keys[cand_idx[core_local_np]].astype(np.float32)
        v_core = cmpd_values.cpu().numpy().astype(np.float32)
        w_np = w.cpu().numpy().astype(np.float32)
        global_idx = cand_idx[core_local_np].astype(np.int64)

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
                exact_denominator=bool(cfg.get("exact_denominator", False)),
                n_sink=int(cfg.get("n_sink", 1)),
                local_window=int(cfg.get("local_window", 1024)),
                device=cfg.get("device"),
            ),
        ]
