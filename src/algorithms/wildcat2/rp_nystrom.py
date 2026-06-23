"""Randomly pivoted Nyström (RPNys), ported from microsoft/wildcat."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def rp_nystrom(
    keys: torch.Tensor,
    sqd_knorm: torch.Tensor,
    r: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys_dtype, device = keys.dtype, keys.device
    dtype = (
        torch.float32
        if keys_dtype in (torch.bfloat16, torch.float16)
        else keys_dtype
    )

    keys = keys.to(dtype)
    sqd_knorm = sqd_knorm.to(dtype)
    hsqd_knorm = sqd_knorm / 2.0

    n = keys.shape[-2]
    batch_shape = keys.shape[:-2]

    kernel_core = torch.zeros((*batch_shape, r, n), dtype=dtype, device=device)
    kernel_core_dim = kernel_core.shape[0]
    kernel_inv = torch.zeros((*batch_shape, r, r), dtype=dtype, device=device)
    res_diagonal = torch.ones((*batch_shape, n), dtype=dtype, device=device)

    coreset_list: list[torch.Tensor] = []
    uniform = torch.empty((*batch_shape, n), dtype=dtype, device=device)
    g = torch.full((*batch_shape, r), -1.0, dtype=dtype, device=device)

    for i in range(r):
        if generator is not None:
            uniform.uniform_(generator=generator)
        else:
            uniform.uniform_()
        scores = (
            torch.log(res_diagonal) + sqd_knorm
            - torch.log(-torch.log(uniform))
        )
        ids = torch.argmax(scores, dim=-1, keepdim=True)
        coreset_list.append(ids)

        if i > 0:
            a = torch.gather(
                kernel_core[:, :i, :], -1,
                ids[..., None].expand(kernel_core_dim, i, 1),
            ).squeeze(2)
            g[..., :i] = torch.bmm(
                kernel_inv[..., :i, :i], a.unsqueeze(-1),
            ).squeeze(-1)
            g[..., :i + 1] *= torch.rsqrt(res_diagonal.gather(-1, ids))

        kernel_inv[..., :i + 1, :i + 1] += (
            g[..., :i + 1].unsqueeze(-1) * g[..., :i + 1].unsqueeze(-2)
        )

        kernel_row = gsn_kernel(keys, ids, hsqd_knorm).clamp(max=1.0)
        kernel_core[..., i, :] = kernel_row.squeeze(-2)

        if i < r - 1:
            y = torch.einsum(
                "...si, ...s -> ...i",
                kernel_core[..., :i + 1, :],
                g[..., :i + 1],
            )
            res_diagonal -= y.square()
            res_diagonal.scatter_(-1, ids, 0.0)
            res_diagonal.clamp_(min=0.0)

    coreset = torch.cat(coreset_list, dim=-1)
    return coreset, kernel_inv.to(keys_dtype), kernel_core.to(keys_dtype)


def gsn_kernel(
    keys: torch.Tensor,
    ids: torch.Tensor,
    halfsqdkeynorms: torch.Tensor,
) -> torch.Tensor:
    e = keys.shape[-1]
    key_term = torch.einsum(
        "...re, ...ne -> ...rn",
        keys.gather(-2, ids.unsqueeze(-1).expand(*ids.shape, e)),
        keys,
    )
    return torch.exp(
        key_term
        - halfsqdkeynorms.gather(-1, ids).unsqueeze(-1)
        - halfsqdkeynorms.unsqueeze(-2),
    )
