"""
Tensor fully-corrective Frank-Wolfe (l2) coreset selection.

Greedily selects keys to approximate the *tensor* target

    T = sum_i phi(k_i) (x) m_i,    m_i = [1, v_i] in R^{1+d}

(an operator H (x) R^{1+d}) by a sparse set of selected key-features with free
(synthetic) augmented values:

    minimize_{S, U}  || sum_i phi(k_i) m_i^T - sum_{j in S} phi(k_j) u_j^T ||_F^2 .

For a fixed support S the best U is K_SS^{-1} K_{S,:} M, so the residual after
choosing S is tr(M^T E_S M), E_S = K - K_{:,S} K_SS^{-1} K_{S,:}. The FW linear
oracle adds the key whose residual-correlation row has the largest norm:

    G = E_S M  in R^{n x (1+d)},   i* = argmax_i || G[i, :] ||_2^2 .

Plain FCFW-l2 is exactly the mass channel (M = ones) of this objective; the
value channels make selection value-aware.

This is computed by a pivoted-Cholesky-with-labels recursion: maintain the
Cholesky factor columns ``L`` of the Gaussian kernel, the residual diagonal,
and the label residual ``G``; each step is a rank-1 update, keeping the total
cost O(n r^2 + n r (1+d)) -- the same leading term as plain FCFW / RP-Nystrom,
not (1+d) times more. The kernel is the unit-diagonal Gaussian on the
temperature-rescaled keys (``rp_nystrom.gsn_kernel``), identical to wildcat2.

Selection is deterministic and order-independent of ``r`` (nested), and a
``state`` dict allows warm-starting a larger budget on the same keys.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..wildcat2.rp_nystrom import gsn_kernel


def _kernel_matmul(
    keys: torch.Tensor,
    hsqd: torch.Tensor,
    mat: torch.Tensor,
    col_block: int,
) -> torch.Tensor:
    """G0 = K @ mat with K(i,j) = exp(<ki,kj> - |ki|^2/2 - |kj|^2/2), clamp<=1.

    Computed in column blocks to bound the O(n^2) kernel memory.
    """
    b_n, n = keys.shape[0], keys.shape[-2]
    p = mat.shape[-1]
    out = torch.zeros((b_n, n, p), dtype=keys.dtype, device=keys.device)
    for start in range(0, n, col_block):
        end = min(start + col_block, n)
        key_term = torch.einsum(
            "...ne, ...be -> ...nb", keys, keys[:, start:end, :],
        )
        block = torch.exp(
            key_term
            - hsqd.unsqueeze(-1)
            - hsqd[:, start:end].unsqueeze(-2),
        ).clamp(max=1.0)
        out = out + torch.einsum(
            "...nb, ...bp -> ...np", block, mat[:, start:end, :],
        )
    return out


def tensor_fcfw_select(
    keys: torch.Tensor,
    sqd_knorm: torch.Tensor,
    mat: torch.Tensor,
    r: int,
    oracle: str = "fw",
    state: Optional[Dict] = None,
    jitter: float = 1e-6,
    col_block: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Select ``r`` keys by tensor fully-corrective Frank-Wolfe.

    Args:
        keys: (batch, n, e) temperature-rescaled keys.
        sqd_knorm: (batch, n) squared norms of the rescaled keys.
        mat: (batch, n, 1+d) augmented target rows ``[1, v_i]``.
        r: target coreset size.
        oracle: "fw" (argmax ||G[i,:]||^2) or "omp" (||G[i,:]||^2 / E[i,i]).
        state: optional warm-start state from a previous call (same keys/mat).
        jitter: floor for residual-diagonal normalization.
        col_block: column block size for the one-time ``K @ mat`` pass.

    Returns:
        coreset: (batch, r) selected local indices, in selection order.
        kernel_inv: (batch, r, r) inverse Gram K_SS^{-1}.
        kernel_core: (batch, r, n) kernel rows K_{S,:}.
        new_state: dict for warm-starting a larger budget.
    """
    keys_dtype, device = keys.dtype, keys.device
    work_dtype = (
        torch.float32
        if keys_dtype in (torch.bfloat16, torch.float16)
        else keys_dtype
    )
    keys = keys.to(work_dtype)
    sqd_knorm = sqd_knorm.to(work_dtype)
    mat = mat.to(work_dtype)
    hsqd = sqd_knorm / 2.0

    batch = keys.shape[0]
    n = keys.shape[-2]
    r = min(int(r), n)

    reuse = (
        state is not None
        and int(state.get("n", -1)) == n
        and int(state.get("batch", -1)) == batch
    )
    if reuse:
        g_res = state["G"]
        res_diag = state["res_diag"]
        factor = state["L"]
        kernel_core = state["kernel_core"]
        selected = state["selected"]
        m = int(state["m"])
    else:
        g_res = _kernel_matmul(keys, hsqd, mat, col_block)
        res_diag = torch.ones((batch, n), dtype=work_dtype, device=device)
        factor = torch.empty((batch, n, 0), dtype=work_dtype, device=device)
        kernel_core = torch.empty(
            (batch, 0, n), dtype=work_dtype, device=device,
        )
        selected = torch.empty((batch, 0), dtype=torch.long, device=device)
        m = 0

    if r > m:
        new_factor = torch.zeros(
            (batch, n, r), dtype=work_dtype, device=device,
        )
        new_kernel_core = torch.empty(
            (batch, r, n), dtype=work_dtype, device=device,
        )
        new_selected = torch.empty((batch, r), dtype=torch.long, device=device)
        if m > 0:
            new_factor[:, :, :m] = factor
            new_kernel_core[:, :m, :] = kernel_core
            new_selected[:, :m] = selected
        factor, kernel_core, selected = (
            new_factor, new_kernel_core, new_selected,
        )

        neg_inf = torch.finfo(work_dtype).min
        for t in range(m, r):
            score = (g_res * g_res).sum(dim=-1)
            if oracle == "omp":
                score = score / res_diag.clamp(min=jitter)
            if t > 0:
                score = score.scatter(-1, selected[:, :t], neg_inf)
            idx = torch.argmax(score, dim=-1, keepdim=True)
            selected[:, t:t + 1] = idx

            kcol = gsn_kernel(keys, idx, hsqd).clamp(max=1.0).squeeze(-2)
            kernel_core[:, t, :] = kcol

            diag = res_diag.gather(-1, idx).squeeze(-1).clamp(min=jitter)

            if t > 0:
                lp = factor[:, :, :t].gather(
                    1, idx.unsqueeze(-1).expand(batch, 1, t),
                ).squeeze(1)
                proj = torch.einsum(
                    "...ns, ...s -> ...n", factor[:, :, :t], lp,
                )
                ecol = kcol - proj
            else:
                ecol = kcol

            ell = ecol / diag.sqrt().unsqueeze(-1)
            factor[:, :, t] = ell

            ell_mat = torch.einsum("...n, ...np -> ...p", ell, mat)
            g_res = g_res - ell.unsqueeze(-1) * ell_mat.unsqueeze(-2)

            res_diag = res_diag - ell * ell
            res_diag = res_diag.scatter(-1, idx, 0.0).clamp(min=0.0)
        m = r

    new_state = {
        "G": g_res,
        "res_diag": res_diag,
        "L": factor,
        "kernel_core": kernel_core,
        "selected": selected,
        "m": m,
        "n": n,
        "batch": batch,
    }

    coreset = selected[:, :r]
    kernel_core_r = kernel_core[:, :r, :]
    factor_s = factor[:, :, :r].gather(
        1, coreset.unsqueeze(-1).expand(batch, r, r),
    )
    kernel_inv = torch.cholesky_inverse(factor_s)
    return (
        coreset,
        kernel_inv.to(keys_dtype),
        kernel_core_r.to(keys_dtype),
        new_state,
    )
