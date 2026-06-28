"""
WildCat2: faithful port of microsoft/wildcat for loco-attention evaluation.

Compresses candidate keys only (sink + local window handled via special_idx).
Uses full-sequence key centering, reference compress_kv + weighted_attention.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ...core import softmax
from ._device import resolve_device
from .compress_kv import compress_kv
from .weighted_attention import weighted_attention


def _pivot_generator(
    eval_seed: int,
    budget: int,
    device: torch.device,
) -> torch.Generator:
    """Deterministic RPC pivots per (eval seed, budget); independent of call order."""
    ss = np.random.SeedSequence([int(eval_seed), int(budget)])
    seed = int(ss.generate_state(1)[0])
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _compress_candidates_binned(
    keys: torch.Tensor,
    values: torch.Tensor,
    r_total: int,
    scale: float,
    q_scale: Optional[torch.Tensor],
    num_bins: int,
    bin_r: Optional[int],
    generator: Optional[torch.Generator],
    phi: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binned CompressKV on candidate segment (reference WildCat.forward)."""
    n = keys.shape[-2]
    r_total = min(max(int(r_total), 1), n)

    if bin_r is not None and int(bin_r) > 0:
        per_r = int(bin_r)
        n_bins = max(1, r_total // per_r)
    elif num_bins > 1:
        n_bins = int(num_bins)
        r_padded = r_total + (-r_total) % n_bins
        per_r = max(1, r_padded // n_bins)
    else:
        return compress_kv(
            keys, values, r_total, scale=scale, q_scale=q_scale,
            phi=phi, generator=generator,
        )

    remainder = n % n_bins
    n_mid = n - remainder if remainder > 0 else n
    bin_n = n_mid // n_bins

    cmpd_k_parts: List[torch.Tensor] = []
    cmpd_v_parts: List[torch.Tensor] = []
    w_parts: List[torch.Tensor] = []
    local_parts: List[torch.Tensor] = []

    pos = 0
    for _ in range(n_bins):
        sl = slice(pos, pos + bin_n)
        ck, cv, w, loc = compress_kv(
            keys[..., sl, :], values[..., sl, :], per_r,
            scale=scale, q_scale=q_scale, phi=phi, generator=generator,
        )
        cmpd_k_parts.append(ck)
        cmpd_v_parts.append(cv)
        w_parts.append(w)
        local_parts.append(loc + pos)
        pos += bin_n

    if remainder > 0:
        ck, cv, w, loc = compress_kv(
            keys[..., pos:, :], values[..., pos:, :], per_r,
            scale=scale, q_scale=q_scale, phi=phi, generator=generator,
        )
        cmpd_k_parts.append(ck)
        cmpd_v_parts.append(cv)
        w_parts.append(w)
        local_parts.append(loc + pos)

    return (
        torch.cat(cmpd_k_parts, dim=-2),
        torch.cat(cmpd_v_parts, dim=-2),
        torch.cat(w_parts, dim=-1),
        torch.cat(local_parts, dim=-1),
    )


class WildCat2(AttentionAlgorithm):
    """
    WildCat2 on candidate keys; sink + local window via WtdAttn (weight=1).

    Each budget runs a fresh RPC coreset of that size (stochastic; error vs
    budget is not guaranteed monotone). Pivots are deterministic per
    (evaluation.seed, budget).
    """

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
        self._eval_seed = 42

    @property
    def name(self) -> str:
        if self.bin_r is not None:
            return f"WildCat2-bin{self.bin_r}"
        if self.num_bins > 1:
            return f"WildCat2-C{self.num_bins}"
        return "WildCat2"

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
        self._eval_seed = int(seed)

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        del rng  # pivots use evaluation seed + budget only
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
        gen = _pivot_generator(self._eval_seed, int(budget), device)

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

        cmpd_keys, cmpd_values, w, core_local = _compress_candidates_binned(
            cand_keys,
            cand_vals,
            budget,
            scale=scale,
            q_scale=q_scale,
            num_bins=self.num_bins,
            bin_r=self.bin_r,
            generator=gen,
            phi=self.phi,
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
            WildCat2(
                num_bins=int(cfg.get("num_bins", 1)),
                bin_r=cfg.get("bin_r"),
                device=cfg.get("device"),
                phi=cfg.get("phi"),
                q_scale_mode=cfg.get("q_scale_mode", "key_max"),
                exact_denominator=bool(cfg.get("exact_denominator", False)),
            ),
        ]
