"""L-BFGS post-refinement for TFCFW-lq coresets (fixed Z_exact, learnable core numerator)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _per_query_logit_shift(
    queries: torch.Tensor,
    ref_keys: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-probe max logit over the full reference cache, shape ``[m, 1]``."""
    logits = scale * (queries @ ref_keys.T)
    return logits.amax(dim=-1, keepdim=True)


def core_unnorm_numerator_target(
    queries: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    sp_idx: torch.Tensor,
    scale: float,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Unnormalized compressible-region numerator from the full reference cache."""
    logits = scale * (queries @ ref_keys.T)
    e = (logits - shift).exp()
    num_total = e @ ref_values
    if sp_idx.numel() > 0:
        num_sp = e[:, sp_idx] @ ref_values[sp_idx]
    else:
        num_sp = torch.zeros_like(num_total)
    return num_total - num_sp


def core_unnorm_numerator_pred(
    queries: torch.Tensor,
    k_core: torch.Tensor,
    v_core: torch.Tensor,
    scale: float,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Coreset unnormalized numerator: ``sum_j exp(l_j - shift_q) v_j``."""
    logits = scale * (queries @ k_core.T)
    e = (logits - shift).exp()
    return e @ v_core


def relative_unnorm_numerator_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """``mean_q ||pred_q - target_q||_2^2 / (||target_q||_2^2 + eps)``."""
    sq_err = (pred - target).pow(2).sum(dim=-1)
    denom = target.pow(2).sum(dim=-1).clamp_min(eps)
    return (sq_err / denom).mean()


def refine_coreset_lbfgs(
    k_init: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    queries: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    sp_idx: torch.Tensor,
    scale: float,
    *,
    n_steps: int = 100,
    lbfgs_lr: float = 0.5,
    lbfgs_inner_iter: int = 10,
    rel_eps: float = 1e-12,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine ``(K, V)`` to match full-cache core numerator on ``queries``.

    Any initial per-key mass in ``w_init`` is folded into ``v_init``; eval uses
    unit ``core_one`` weights.  Minimizes mean relative squared error
    ``||N_new - N_old||_2^2 / (||N_old||_2^2 + eps)`` with per-query logit
    shift from the full reference cache.  ``Z_exact`` is fixed at eval.
    """
    k = int(k_init.shape[0])
    if k == 0 or int(n_steps) <= 0:
        return k_init.detach(), v_init.detach(), w_init.detach()

    v_eff = v_init * w_init.unsqueeze(-1)

    with torch.no_grad():
        shift = _per_query_logit_shift(queries, ref_keys, scale)
        target = core_unnorm_numerator_target(
            queries, ref_keys, ref_values, sp_idx, scale, shift,
        )

    k_c = nn.Parameter(k_init.clone())
    v_c = nn.Parameter(v_eff.clone())

    optimizer = torch.optim.LBFGS(
        [k_c, v_c],
        lr=float(lbfgs_lr),
        max_iter=max(int(lbfgs_inner_iter), 1),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
    torch.manual_seed(seed)

    for _ in range(max(int(n_steps), 0)):

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            pred = core_unnorm_numerator_pred(
                queries, k_c, v_c, scale, shift,
            )
            loss = relative_unnorm_numerator_loss(pred, target, rel_eps)
            loss.backward()
            return loss

        optimizer.step(closure)

    ones_w = torch.ones(k, dtype=v_init.dtype, device=v_init.device)
    return k_c.detach(), v_c.detach(), ones_w
