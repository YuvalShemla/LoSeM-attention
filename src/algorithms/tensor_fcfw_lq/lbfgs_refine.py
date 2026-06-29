"""L-BFGS post-refinement and denominator-only weight fitting for TFCFW-lq."""

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


def full_attention_targets(
    queries: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Full-cache softmax attention outputs for each probe, shape ``[m, d_v]``."""
    logits = scale * (queries @ ref_keys.T)
    shift = logits.amax(dim=-1, keepdim=True)
    e = (logits - shift).exp()
    z = e.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return (e @ ref_values) / z


def split_denominator_attention(
    queries: torch.Tensor,
    sp_keys: torch.Tensor,
    sp_values: torch.Tensor,
    k_core: torch.Tensor,
    v_core: torch.Tensor,
    w_den_core: torch.Tensor,
    scale: float,
    global_shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local softmax: unit core weights in the numerator, ``w`` in the denominator.

    When ``global_shift`` is set (per-probe max logit over the full reference
    cache), the numerator uses that shift so ``V`` matches the OMP+L-BFGS
    unnormalized-numerator objective.  The denominator still uses a coreset-local
    max over special + core logits.
    """
    if sp_keys.numel() > 0:
        lg_sp = scale * (queries @ sp_keys.T)
    else:
        lg_sp = queries.new_zeros((queries.shape[0], 0))
    lg_c = scale * (queries @ k_core.T)
    logits = torch.cat([lg_sp, lg_c], dim=-1)
    local_shift = logits.amax(dim=-1, keepdim=True)
    w = w_den_core.clamp(min=1e-20)

    if global_shift is None:
        e_sp = (lg_sp - local_shift).exp()
        e_c = (lg_c - local_shift).exp()
    else:
        e_sp = (lg_sp - global_shift).exp()
        e_c = (lg_c - global_shift).exp()
    num = e_sp @ sp_values + e_c @ v_core

    e_sp_d = (lg_sp - local_shift).exp()
    e_c_d = (lg_c - local_shift).exp()
    den = e_sp_d.sum(dim=-1) + (e_c_d * w.unsqueeze(0)).sum(dim=-1)
    return num / den.unsqueeze(-1).clamp_min(1e-20)


def relative_attention_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    sq_err = (pred - target).pow(2).sum(dim=-1)
    denom = target.pow(2).sum(dim=-1).clamp_min(eps)
    return (sq_err / denom).mean()


def fit_denominator_only_weights(
    queries: torch.Tensor,
    k_core: torch.Tensor,
    v_core: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    sp_idx: torch.Tensor,
    scale: float,
    *,
    n_steps: int = 30,
    lbfgs_lr: float = 0.5,
    lbfgs_inner_iter: int = 10,
    rel_eps: float = 1e-12,
    seed: int = 42,
    w_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit positive core weights for the **denominator only** (fixed ``K, V``).

    The numerator uses unit weights on core keys; ``(K, V)`` should already
    approximate the unnormalized attention numerator (from OMP + L-BFGS).
    """
    k = int(k_core.shape[0])
    if k == 0 or int(n_steps) <= 0:
        return torch.ones(k, dtype=k_core.dtype, device=k_core.device)

    k_fixed = k_core.detach()
    v_fixed = v_core.detach()
    sp_keys = (
        ref_keys[sp_idx]
        if sp_idx.numel() > 0
        else ref_keys.new_zeros((0, ref_keys.shape[-1]))
    )
    sp_values = (
        ref_values[sp_idx]
        if sp_idx.numel() > 0
        else ref_values.new_zeros((0, ref_values.shape[-1]))
    )

    with torch.no_grad():
        target = full_attention_targets(
            queries, ref_keys, ref_values, scale,
        )
        global_shift = (
            scale * (queries @ ref_keys.T)
        ).amax(dim=-1, keepdim=True)

    if w_init is None:
        log_w = nn.Parameter(torch.zeros(k, dtype=k_core.dtype, device=k_core.device))
    else:
        log_w = nn.Parameter(
            w_init.clamp(min=1e-20).log().to(dtype=k_core.dtype, device=k_core.device),
        )

    optimizer = torch.optim.LBFGS(
        [log_w],
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
            w = log_w.exp()
            pred = split_denominator_attention(
                queries, sp_keys, sp_values, k_fixed, v_fixed, w, scale,
                global_shift=global_shift,
            )
            loss = relative_attention_loss(pred, target, rel_eps)
            loss.backward()
            return loss

        optimizer.step(closure)

    return log_w.exp().detach()


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
    """Refine ``(K, V)`` to match the unnormalized core numerator on ``queries``.

    Per-key mass in ``w_init`` is folded into ``V``; eval uses unit core weights
    in the numerator (and exact ``Z`` when ``all_logits`` is supplied).
    """
    k = int(k_init.shape[0])
    if k == 0 or int(n_steps) <= 0:
        return k_init.detach(), v_init.detach(), w_init.detach()

    return _refine_exact_denominator(
        k_init, v_init, w_init, queries, ref_keys, ref_values, sp_idx,
        scale, n_steps=n_steps, lbfgs_lr=lbfgs_lr,
        lbfgs_inner_iter=lbfgs_inner_iter, rel_eps=rel_eps, seed=seed,
    )


def _refine_exact_denominator(
    k_init: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    queries: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    sp_idx: torch.Tensor,
    scale: float,
    *,
    n_steps: int,
    lbfgs_lr: float,
    lbfgs_inner_iter: int,
    rel_eps: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = int(k_init.shape[0])
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
