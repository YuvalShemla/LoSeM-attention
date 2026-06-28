"""
KVSculpt distillation: optimize unconstrained (K_c, V_c) to match full-cache attention.

Implements the core optimizer from Jiang & Jin, "KVSculpt: KV Cache Compression as
Distillation" (arXiv:2603.27819): L-BFGS on keys, ridge-regression value solves every
few steps, output-MSE + log-sum-exp matching loss, top-k attention init, and synthetic
future queries via De-RoPE.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...core import compute_special_indices
from ..learned.learn_coreset import reference_position
from ..probe_queries import (
    apply_rope,
    build_kvsculpt_train_queries,
    build_probe_queries,
    inverse_rope,
)

# Backward-compatible alias used by tests.
build_training_queries = build_kvsculpt_train_queries

def _full_targets(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = scale * (queries @ keys.T)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    target = attn @ values
    return target, lse


def _compressed_pred(
    queries: torch.Tensor,
    k_c: torch.Tensor,
    k_ret: torch.Tensor,
    v_c: torch.Tensor,
    v_ret: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_cat = torch.cat([k_c, k_ret], dim=0)
    v_cat = torch.cat([v_c, v_ret], dim=0)
    scores = scale * (queries @ k_cat.T)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    pred = attn @ v_cat
    return pred, lse


def _solve_v_ridge(
    queries: torch.Tensor,
    k_c: torch.Tensor,
    k_ret: torch.Tensor,
    v_ret: torch.Tensor,
    target_y: torch.Tensor,
    scale: float,
    ridge_lambda: float,
) -> torch.Tensor:
    """Closed-form ridge regression for V_c given fixed keys."""
    k_cat = torch.cat([k_c.detach(), k_ret], dim=0)
    scores = scale * (queries @ k_cat.T)
    attn = torch.softmax(scores, dim=-1)
    k = k_c.shape[0]
    a_c = attn[:, :k]
    a_r = attn[:, k:]
    residual = target_y - a_r @ v_ret
    ata = a_c.T @ a_c + ridge_lambda * torch.eye(
        k, device=a_c.device, dtype=a_c.dtype,
    )
    atb = a_c.T @ residual
    return torch.linalg.solve(ata, atb)


def _topk_init(
    training_queries: torch.Tensor,
    cand_keys: torch.Tensor,
    cand_values: torch.Tensor,
    keys_full: torch.Tensor,
    cand_idx: np.ndarray,
    budget: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warm-start from top-k candidate keys by accumulated attention mass."""
    n_cand = cand_keys.shape[0]
    k = min(max(int(budget), 1), n_cand)
    scores = scale * (training_queries @ keys_full.T)
    attn = torch.softmax(scores, dim=-1)
    cand_t = torch.as_tensor(cand_idx, dtype=torch.long, device=attn.device)
    importance = attn.index_select(dim=1, index=cand_t).sum(dim=0)
    top_local = torch.topk(importance, k=k, largest=True).indices
    return cand_keys[top_local].clone(), cand_values[top_local].clone()


def refine_kv_pairs(
    k_init: torch.Tensor,
    v_init: torch.Tensor,
    queries: torch.Tensor,
    k_ret: torch.Tensor,
    v_ret: torch.Tensor,
    keys_full: torch.Tensor,
    values_full: torch.Tensor,
    scale: float,
    *,
    n_k_steps: int = 10,
    v_solve_every: int = 5,
    lbfgs_lr: float = 0.5,
    lbfgs_inner_iter: int = 10,
    ridge_lambda: float = 1e-3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KVSculpt-style L-BFGS on keys with periodic ridge value solves.

    Refines an existing compressed coreset ``(k_init, v_init)`` against the
    full-cache output + log-sum-exp targets on ``queries``.  Returns detached
    ``(K_c, V_c)`` tensors.
    """
    k = int(k_init.shape[0])
    if k == 0 or int(n_k_steps) <= 0:
        return k_init.detach(), v_init.detach()

    with torch.no_grad():
        target_y, target_lse = _full_targets(
            queries, keys_full, values_full, scale,
        )

    k_c = nn.Parameter(k_init.clone())
    v_c = v_init.clone()

    with torch.no_grad():
        v_c = _solve_v_ridge(
            queries, k_c, k_ret, v_ret, target_y, scale, ridge_lambda,
        )

    optimizer = torch.optim.LBFGS(
        [k_c],
        lr=float(lbfgs_lr),
        max_iter=max(int(lbfgs_inner_iter), 1),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
    torch.manual_seed(seed)

    n_steps = max(int(n_k_steps), 0)
    v_every = max(int(v_solve_every), 1)

    for step in range(n_steps):

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            pred_y, pred_lse = _compressed_pred(
                queries, k_c, k_ret, v_c.detach(), v_ret, scale,
            )
            loss = (
                (pred_y - target_y).pow(2).mean()
                + (pred_lse - target_lse).pow(2).mean()
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        if (step + 1) % v_every == 0:
            with torch.no_grad():
                v_c = _solve_v_ridge(
                    queries, k_c, k_ret, v_ret, target_y, scale, ridge_lambda,
                )

    return k_c.detach(), v_c.detach()


def distill_kv_cache(
    keys: np.ndarray,
    values: np.ndarray,
    head_dim: int,
    training_queries: np.ndarray,
    ref_pos: int,
    budget: int,
    n_sink: int,
    local_window: int,
    *,
    n_k_steps: int = 100,
    v_solve_every: int = 5,
    lbfgs_lr: float = 0.5,
    lbfgs_inner_iter: int = 10,
    ridge_lambda: float = 1e-3,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distill the candidate (compress) zone into ``budget`` unconstrained KV pairs.

    Returns (K_c, V_c) float32 arrays of shape (budget, d).
    """
    if device is None:
        device = torch.device("cpu")
    budget = max(int(budget), 0)
    if budget == 0:
        d = keys.shape[1]
        return (
            np.zeros((0, d), dtype=np.float32),
            np.zeros((0, d), dtype=np.float32),
        )

    training_queries = np.asarray(training_queries, dtype=np.float32)
    if training_queries.ndim != 2 or training_queries.shape[0] == 0:
        raise ValueError("training_queries must be a non-empty [m, d] array")

    n_causal = ref_pos + 1
    sp_idx, cand_idx = compute_special_indices(n_causal, n_sink, local_window)
    if len(cand_idx) == 0:
        d = keys.shape[1]
        return (
            np.zeros((0, d), dtype=np.float32),
            np.zeros((0, d), dtype=np.float32),
        )

    scale = 1.0 / np.sqrt(head_dim)
    keys_ref = torch.as_tensor(
        keys[:n_causal], dtype=torch.float32, device=device,
    )
    values_ref = torch.as_tensor(
        values[:n_causal], dtype=torch.float32, device=device,
    )
    queries_t = torch.as_tensor(
        training_queries, dtype=torch.float32, device=device,
    )
    k_ret = keys_ref[sp_idx]
    v_ret = values_ref[sp_idx]

    cand_keys = keys_ref[cand_idx]
    cand_values = values_ref[cand_idx]

    with torch.no_grad():
        k0, v0 = _topk_init(
            queries_t, cand_keys, cand_values, keys_ref, cand_idx, budget, scale,
        )

    k_c, v_c = refine_kv_pairs(
        k0,
        v0,
        queries_t,
        k_ret,
        v_ret,
        keys_ref,
        values_ref,
        scale,
        n_k_steps=n_k_steps,
        v_solve_every=v_solve_every,
        lbfgs_lr=lbfgs_lr,
        lbfgs_inner_iter=lbfgs_inner_iter,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )

    return (
        k_c.cpu().numpy().astype(np.float32),
        v_c.cpu().numpy().astype(np.float32),
    )


def pilot_mse(
    keys: np.ndarray,
    values: np.ndarray,
    head_dim: int,
    training_queries: np.ndarray,
    ref_pos: int,
    budget: int,
    n_sink: int,
    local_window: int,
    *,
    pilot_steps: int = 30,
    device: Optional[torch.device] = None,
    seed: int = 42,
    **kwargs,
) -> float:
    """Cheap pilot compression MSE for adaptive budget allocation signals."""
    k_c, v_c = distill_kv_cache(
        keys,
        values,
        head_dim,
        training_queries,
        ref_pos,
        budget,
        n_sink,
        local_window,
        n_k_steps=pilot_steps,
        device=device,
        seed=seed,
        **kwargs,
    )
    if k_c.shape[0] == 0:
        return 0.0

    n_causal = ref_pos + 1
    sp_idx, _ = compute_special_indices(n_causal, n_sink, local_window)
    scale = 1.0 / np.sqrt(head_dim)
    queries_t = torch.as_tensor(training_queries, dtype=torch.float32, device=device)
    keys_ref = torch.as_tensor(keys[:n_causal], dtype=torch.float32, device=device)
    values_ref = torch.as_tensor(values[:n_causal], dtype=torch.float32, device=device)
    k_ret = keys_ref[sp_idx]
    v_ret = values_ref[sp_idx]
    k_c_t = torch.as_tensor(k_c, dtype=torch.float32, device=device)
    v_c_t = torch.as_tensor(v_c, dtype=torch.float32, device=device)

    with torch.no_grad():
        target_y, _ = _full_targets(queries_t, keys_ref, values_ref, scale)
        pred_y, _ = _compressed_pred(
            queries_t, k_c_t, k_ret, v_c_t, v_ret, scale,
        )
        return float((pred_y - target_y).pow(2).mean().cpu())


__all__ = [
    "apply_rope",
    "inverse_rope",
    "build_training_queries",
    "distill_kv_cache",
    "pilot_mse",
    "refine_kv_pairs",
    "reference_position",
    "build_probe_queries",
]
