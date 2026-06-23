"""
Tensor FC Frank-Wolfe under the query-weighted ``lq`` norm.

Like ``tensor_fcfw_l2`` (value-aware coreset of the candidate region, sink +
local window handled exactly via ``weighted_attention``), but the coreset is
selected and corrected to minimize a *query-weighted* error instead of the
RKHS-l2 (Frobenius) error:

    ||sigma - sigma'||_*  =  E_{q in Q} | (sigma - sigma') . psi(q) | ,

with ``Q`` the set of earlier-context queries (the same probe set the
``learned`` method uses). See ``select_lq`` for the full derivation.

Unlike ``tensor_fcfw_l2`` there is no temperature rescaling or key centering:
the objective is defined directly through the real attention kernel
``exp(scale q.k)`` evaluated at the probe queries, so the selected candidate
keys are used verbatim and the synthetic ``(value, mass)`` solve is calibrated
for the kernel ``weighted_attention`` consumes. Deterministic and nested.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import softmax
from ..wildcat2._device import resolve_device
from ..wildcat2.weighted_attention import weighted_attention
from ..learned.learn_coreset import build_probe_queries, reference_position
from .select_lq import select_lq_coreset


class TensorFCFWLq(AttentionAlgorithm):
    """Tensor FCFW coreset under the query-weighted lq norm over context queries."""

    def __init__(
        self,
        n_train_queries: int = 1280,
        oracle: str = "fw",
        irls_iters: int = 5,
        rcond: float = 1e-3,
        exact_denominator: bool = False,
        device: Optional[str] = None,
    ):
        if oracle not in ("fw", "omp"):
            raise ValueError(f"oracle must be 'fw' or 'omp'; got {oracle!r}")
        self.n_train_queries = int(n_train_queries)
        self.oracle = oracle
        self.irls_iters = int(irls_iters)
        self.rcond = float(rcond)
        self.exact_denominator = bool(exact_denominator)
        self._device = resolve_device(device)

        self._probe_queries: Optional[np.ndarray] = None
        self._ref_pos: Optional[int] = None
        self._cache_fp: Optional[tuple] = None
        self._cache_state: Optional[dict] = None

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
    ) -> None:
        self._cache_fp = None
        self._cache_state = None
        if queries is None:
            self._probe_queries = None
            self._ref_pos = None
            return
        self._ref_pos = reference_position(len(queries), query_positions)
        self._probe_queries = build_probe_queries(
            queries, query_positions, self._ref_pos, self.n_train_queries,
        )

    def _fingerprint(self, problem: AttentionInput, n_cand: int) -> tuple:
        cand = problem.candidate_idx
        return (
            int(n_cand),
            int(cand[0]),
            int(cand[-1]),
            int(len(problem.keys)),
            self.oracle,
            self.irls_iters,
        )

    def _probe_tensor(
        self,
        problem: AttentionInput,
        device: torch.device,
    ) -> torch.Tensor:
        """Probe queries Q; fall back to candidate keys if none were prepared."""
        if self._probe_queries is not None and self._probe_queries.shape[0] > 0:
            return torch.as_tensor(
                self._probe_queries, dtype=torch.float32, device=device,
            )
        cand = problem.keys[problem.candidate_idx]
        return torch.as_tensor(cand, dtype=torch.float32, device=device)

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        del rng  # deterministic
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
        n_causal = len(keys)
        head_dim = problem.head_dim
        scale = 1.0 / np.sqrt(head_dim)
        device = self._device

        keys_all = torch.as_tensor(
            keys[:n_causal], dtype=torch.float32, device=device,
        )
        values_all = torch.as_tensor(
            values[:n_causal], dtype=torch.float32, device=device,
        )
        cand_keys = keys_all[candidate_idx]
        cand_vals = values_all[candidate_idx]
        probes = self._probe_tensor(problem, device)

        fp = self._fingerprint(problem, n_cand)
        if fp != self._cache_fp:
            self._cache_fp = fp
            self._cache_state = None

        core_local, cmpd_values, w, new_state = select_lq_coreset(
            probes,
            cand_keys,
            cand_vals,
            budget,
            scale,
            oracle=self.oracle,
            irls_iters=self.irls_iters,
            rcond=self.rcond,
            state=self._cache_state,
        )
        self._cache_state = new_state

        cmpd_keys = cand_keys[core_local].unsqueeze(0)
        cmpd_values = cmpd_values.unsqueeze(0)
        w = w.unsqueeze(0)

        keys_all = keys_all.unsqueeze(0)
        values_all = values_all.unsqueeze(0)

        if n_sp > 0:
            sp_keys = keys_all[:, special_idx, :]
            sp_vals = values_all[:, special_idx, :]
            sp_one = torch.ones(
                (1, n_sp), dtype=torch.float32, device=device,
            )
            core_keys = torch.cat([sp_keys, cmpd_keys], dim=1)
            core_values = torch.cat([sp_vals, cmpd_values], dim=1)
            core_one = torch.cat([sp_one, w], dim=-1)
        else:
            core_keys = cmpd_keys
            core_values = cmpd_values
            core_one = w

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

        core_global = candidate_idx[core_local.cpu().numpy()]
        selected = np.concatenate([special_idx, core_global]).astype(np.int64)

        return AttentionOutput(
            output=output,
            actual_budget=n_sp + int(w.shape[-1]),
            selected_indices=selected,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [
            TensorFCFWLq(
                n_train_queries=int(cfg.get("n_train_queries", 1280)),
                oracle=cfg.get("oracle", "fw"),
                irls_iters=int(cfg.get("irls_iters", 5)),
                rcond=float(cfg.get("rcond", 1e-3)),
                exact_denominator=bool(cfg.get("exact_denominator", False)),
                device=cfg.get("device"),
            ),
        ]
