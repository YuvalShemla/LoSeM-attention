"""
Fully-corrective Frank-Wolfe (l2) coreset selection in Gaussian kernel space.

Greedily selects keys that maximally reduce the residual norm of approximating
the kernel-mean target ``mu = sum_n phi(k_n)`` by a weighted set of selected
feature maps:

    minimize_{S, c}  || mu - sum_{j in S} c_j phi(k_j) ||^2_H .

Each step picks the atom most correlated with the current residual
(Frank-Wolfe linear oracle, ``argmax_i g_i`` with ``g = b - K_{:,S} c``), then
re-optimizes *all* selected weights by unconstrained least squares
(``c = K_SS^{-1} b_S``, the fully-corrective step). The kernel is the unit-
diagonal Gaussian/RBF kernel on the temperature-rescaled keys, identical to
``wildcat2`` (see ``rp_nystrom.gsn_kernel``).

Selection is deterministic and order-independent of the target size ``r``: the
first ``r`` picks are the same regardless of how large ``r`` is, so budget
sweeps are naturally nested and the error is monotone non-increasing. A
``state`` dict allows warm-starting (extending a previous selection) so a whole
budget sweep costs a single run to the largest budget.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..wildcat2.rp_nystrom import gsn_kernel


def _kernel_row_sums(
    keys: torch.Tensor,
    hsqd: torch.Tensor,
    col_block: int,
) -> torch.Tensor:
    """b_i = sum_j K(i, j) with K(i,j) = exp(<ki,kj> - |ki|^2/2 - |kj|^2/2).

    Computed in column blocks to bound the O(n^2) memory of the full kernel.
    """
    b_n, n = keys.shape[0], keys.shape[-2]
    out = torch.zeros((b_n, n), dtype=keys.dtype, device=keys.device)
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
        out = out + block.sum(dim=-1)
    return out


def fcfw_select(
    keys: torch.Tensor,
    sqd_knorm: torch.Tensor,
    r: int,
    state: Optional[Dict] = None,
    jitter: float = 1e-6,
    col_block: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Select ``r`` keys by fully-corrective Frank-Wolfe.

    Args:
        keys: (batch, n, e) temperature-rescaled keys.
        sqd_knorm: (batch, n) squared norms of the rescaled keys.
        r: target coreset size.
        state: optional warm-start state from a previous call (same keys).
        jitter: diagonal floor for the incremental Cholesky factor.
        col_block: column block size for the one-time ``b = K @ 1`` pass.

    Returns:
        coreset: (batch, r) selected local indices, in selection order.
        kernel_inv: (batch, r, r) inverse of the selected Gram matrix K_SS.
        kernel_core: (batch, r, n) kernel rows K_{S,:} for the selected atoms.
        new_state: dict for warm-starting a larger budget on the same keys.
    """
    keys_dtype, device = keys.dtype, keys.device
    work_dtype = (
        torch.float32
        if keys_dtype in (torch.bfloat16, torch.float16)
        else keys_dtype
    )
    keys = keys.to(work_dtype)
    sqd_knorm = sqd_knorm.to(work_dtype)
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
        b = state["b"]
        m = int(state["m"])
        selected = state["selected"]
        kernel_core = state["kernel_core"]
        chol = state["L"]
    else:
        b = _kernel_row_sums(keys, hsqd, col_block)
        m = 0
        selected = torch.empty((batch, 0), dtype=torch.long, device=device)
        kernel_core = torch.empty(
            (batch, 0, n), dtype=work_dtype, device=device,
        )
        chol = torch.empty((batch, 0, 0), dtype=work_dtype, device=device)

    if r > m:
        new_selected = torch.empty((batch, r), dtype=torch.long, device=device)
        new_kernel_core = torch.empty(
            (batch, r, n), dtype=work_dtype, device=device,
        )
        new_chol = torch.zeros((batch, r, r), dtype=work_dtype, device=device)
        if m > 0:
            new_selected[:, :m] = selected
            new_kernel_core[:, :m, :] = kernel_core
            new_chol[:, :m, :m] = chol
        selected, kernel_core, chol = new_selected, new_kernel_core, new_chol

        if m > 0:
            b_sel = b.gather(-1, selected[:, :m])
            c = torch.cholesky_solve(
                b_sel.unsqueeze(-1), chol[:, :m, :m],
            ).squeeze(-1)
            g = b - torch.einsum(
                "...mn, ...m -> ...n", kernel_core[:, :m, :], c,
            )
        else:
            g = b.clone()

        neg_inf = torch.finfo(work_dtype).min
        for i in range(m, r):
            masked = g.clone()
            if i > 0:
                masked.scatter_(-1, selected[:, :i], neg_inf)
            idx = torch.argmax(masked, dim=-1, keepdim=True)
            selected[:, i:i + 1] = idx

            row = gsn_kernel(keys, idx, hsqd).clamp(max=1.0).squeeze(-2)
            kernel_core[:, i, :] = row
            diag = row.gather(-1, idx).squeeze(-1)

            if i > 0:
                k_vec = row.gather(-1, selected[:, :i])
                solved = torch.linalg.solve_triangular(
                    chol[:, :i, :i], k_vec.unsqueeze(-1), upper=False,
                ).squeeze(-1)
                pivot = torch.sqrt(
                    torch.clamp(diag - (solved * solved).sum(-1), min=jitter),
                )
                chol[:, i, :i] = solved
                chol[:, i, i] = pivot
            else:
                chol[:, 0, 0] = torch.sqrt(torch.clamp(diag, min=jitter))

            b_sel = b.gather(-1, selected[:, :i + 1])
            c = torch.cholesky_solve(
                b_sel.unsqueeze(-1), chol[:, :i + 1, :i + 1],
            ).squeeze(-1)
            g = b - torch.einsum(
                "...mn, ...m -> ...n", kernel_core[:, :i + 1, :], c,
            )
        m = r

    new_state = {
        "b": b,
        "selected": selected,
        "kernel_core": kernel_core,
        "L": chol,
        "m": m,
        "n": n,
        "batch": batch,
    }

    coreset = selected[:, :r]
    chol_r = chol[:, :r, :r]
    kernel_core_r = kernel_core[:, :r, :]
    kernel_inv = torch.cholesky_inverse(chol_r)
    return (
        coreset,
        kernel_inv.to(keys_dtype),
        kernel_core_r.to(keys_dtype),
        new_state,
    )
