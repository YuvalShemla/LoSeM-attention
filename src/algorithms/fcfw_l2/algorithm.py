"""
FC Frank-Wolfe l2: coreset attention with fully-corrective Frank-Wolfe selection.

Same pipeline as WildCat2 (full-sequence key centering, sink + local window via
WtdAttn with weight 1, identical temperature and value aggregation), but the
candidate coreset is chosen by fully-corrective Frank-Wolfe in Gaussian kernel
space instead of randomly-pivoted Nystrom. Selection is deterministic and
nested, so error is monotone non-increasing in budget.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import softmax
from ..wildcat2._device import resolve_device
from .compress_kv import compress_kv_fcfw


class FCFrankWolfeL2(AttentionAlgorithm):
    """FCFW (l2) coreset on candidate keys; sink + local window via WtdAttn."""

    def __init__(
        self,
        num_bins: int = 1,
        bin_r: Optional[int] = None,
        device: Optional[str] = None,
        phi: Optional[float] = None,
        q_scale_mode: str = "key_max",
        exact_denominator: bool = False,
    ):
        self.num_bins = max(1, int(num_bins))
        self.bin_r = int(bin_r) if bin_r is not None else None
        self.phi = phi
        self._device = resolve_device(device)
        if q_scale_mode not in ("query", "key_max"):
            raise ValueError(
                f"q_scale_mode must be 'query' or 'key_max'; got {q_scale_mode!r}",
            )
        self.q_scale_mode = q_scale_mode
        self.exact_denominator = bool(exact_denominator)
        self._cache_fp: Optional[tuple] = None
        self._cache_state: Optional[dict] = None

    @property
    def name(self) -> str:
        if self.bin_r is not None:
            return f"FCFW-l2-bin{self.bin_r}"
        if self.num_bins > 1:
            return f"FCFW-l2-C{self.num_bins}"
        return "FCFW-l2"

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
        # New example: drop any nested-sweep cache.
        self._cache_fp = None
        self._cache_state = None

    def _fingerprint(self, problem: AttentionInput, n_cand: int) -> tuple:
        cand = problem.candidate_idx
        return (
            hash(problem.query.tobytes()),
            int(n_cand),
            int(cand[0]),
            int(cand[-1]),
            int(len(problem.keys)),
            self.q_scale_mode,
        )

    def _compress_single(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        r: int,
        scale: float,
        q_scale: Optional[torch.Tensor],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._cache_state if use_cache else None
        (ck, cv, w, loc), new_state = compress_kv_fcfw(
            keys, values, r, scale=scale, q_scale=q_scale,
            phi=self.phi, state=state,
        )
        if use_cache:
            self._cache_state = new_state
        return ck, cv, w, loc

    def _compress_binned(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        r_total: int,
        scale: float,
        q_scale: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = keys.shape[-2]
        r_total = min(max(int(r_total), 1), n)

        if self.bin_r is not None and int(self.bin_r) > 0:
            per_r = int(self.bin_r)
            n_bins = max(1, r_total // per_r)
        else:
            n_bins = int(self.num_bins)
            r_padded = r_total + (-r_total) % n_bins
            per_r = max(1, r_padded // n_bins)

        remainder = n % n_bins
        n_mid = n - remainder if remainder > 0 else n
        bin_n = n_mid // n_bins

        ck_parts: List[torch.Tensor] = []
        cv_parts: List[torch.Tensor] = []
        w_parts: List[torch.Tensor] = []
        loc_parts: List[torch.Tensor] = []

        pos = 0
        for _ in range(n_bins):
            sl = slice(pos, pos + bin_n)
            (ck, cv, w, loc), _ = compress_kv_fcfw(
                keys[..., sl, :], values[..., sl, :], per_r,
                scale=scale, q_scale=q_scale, phi=self.phi, state=None,
            )
            ck_parts.append(ck)
            cv_parts.append(cv)
            w_parts.append(w)
            loc_parts.append(loc + pos)
            pos += bin_n

        if remainder > 0:
            (ck, cv, w, loc), _ = compress_kv_fcfw(
                keys[..., pos:, :], values[..., pos:, :], per_r,
                scale=scale, q_scale=q_scale, phi=self.phi, state=None,
            )
            ck_parts.append(ck)
            cv_parts.append(cv)
            w_parts.append(w)
            loc_parts.append(loc + pos)

        return (
            torch.cat(ck_parts, dim=-2),
            torch.cat(cv_parts, dim=-2),
            torch.cat(w_parts, dim=-1),
            torch.cat(loc_parts, dim=-1),
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        del rng  # FCFW is deterministic
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

        kbar_np = keys[:n_causal].mean(axis=0, keepdims=True)
        kbar = torch.as_tensor(
            kbar_np, dtype=torch.float32, device=device,
        ).unsqueeze(0)

        keys_all = torch.as_tensor(
            keys[:n_causal], dtype=torch.float32, device=device,
        ).unsqueeze(0)
        values_all = torch.as_tensor(
            values[:n_causal], dtype=torch.float32, device=device,
        ).unsqueeze(0)
        keys_c = keys_all - kbar

        cand_keys = keys_c[:, candidate_idx, :]
        cand_vals = values_all[:, candidate_idx, :]

        if self.q_scale_mode == "query":
            q_scale = torch.tensor(
                [[max(float(np.linalg.norm(problem.query)), 1e-12)]],
                dtype=torch.float32,
                device=device,
            )
        else:
            q_scale = None

        use_cache = self.num_bins == 1 and self.bin_r is None
        if use_cache:
            fp = self._fingerprint(problem, n_cand)
            if fp != self._cache_fp:
                self._cache_fp = fp
                self._cache_state = None
            cmpd_keys, cmpd_values, w, core_local = self._compress_single(
                cand_keys, cand_vals, budget, scale, q_scale, use_cache=True,
            )
        else:
            cmpd_keys, cmpd_values, w, core_local = self._compress_binned(
                cand_keys, cand_vals, budget, scale, q_scale,
            )

        cmpd_keys = cmpd_keys + kbar

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

        from ..wildcat2.weighted_attention import weighted_attention

        out_t = weighted_attention(
            q, core_keys, core_values, core_one, scale, vmin, vmax,
            all_logits=all_logits,
        )
        output = out_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        core_global = candidate_idx[core_local.squeeze(0).cpu().numpy()]
        selected = np.concatenate([special_idx, core_global]).astype(np.int64)

        return AttentionOutput(
            output=output,
            actual_budget=n_sp + int(w.shape[-1]),
            selected_indices=selected,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [
            FCFrankWolfeL2(
                num_bins=int(cfg.get("num_bins", 1)),
                bin_r=cfg.get("bin_r"),
                device=cfg.get("device"),
                phi=cfg.get("phi"),
                q_scale_mode=cfg.get("q_scale_mode", "key_max"),
                exact_denominator=bool(cfg.get("exact_denominator", False)),
            ),
        ]
