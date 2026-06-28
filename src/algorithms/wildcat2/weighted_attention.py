"""Weighted coreset attention (WtdAttn, Alg. 3), ported from microsoft/wildcat."""

from __future__ import annotations

from typing import Optional

import torch


def weighted_attention(
    queries: torch.Tensor,
    core_keys: torch.Tensor,
    core_values: torch.Tensor,
    core_one: torch.Tensor,
    scale: float,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    all_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Coreset attention with optional exact causal KDE normalization.

    When ``all_logits`` is provided (shape ``[n_causal]``), the numerator still
    comes from the weighted coreset but the denominator is the exact partition
    function ``Z_exact = sum_i exp(all_logits[i])``, stabilized with the global
    max logit over all causal keys.
    """
    qk = scale * torch.einsum("...te, ...re -> ...tr", queries, core_keys)

    eps = 1e-20
    if all_logits is not None:
        logits = all_logits.reshape(-1).to(dtype=qk.dtype, device=qk.device)
        max_l = logits.max()
        qk_exp = (qk - max_l).exp()
        num = torch.einsum("...tr, ...rd -> ...td", qk_exp, core_values)
        z_exact = (logits - max_l).exp().sum()
        out = torch.where(
            z_exact > eps,
            num / z_exact,
            torch.zeros_like(num),
        )
    else:
        qk = qk - qk.amax(-1, keepdim=True)
        qk = qk.exp()

        qk1 = torch.einsum("...tr, ...r -> ...t", qk, core_one).unsqueeze(-1)

        out = torch.where(
            qk1 > eps,
            torch.einsum("...tr, ...rd -> ...td", qk, core_values) / qk1,
            torch.zeros_like(qk1),
        )
    return out.clamp(min=min_val, max=max_val)
