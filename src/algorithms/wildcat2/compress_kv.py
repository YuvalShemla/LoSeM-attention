"""CompressKV (Alg. 2), ported from microsoft/wildcat."""

from __future__ import annotations

import math
from math import sqrt
from typing import Optional, Tuple

import torch

from .math_utils import lambert_w_circ_exp
from .rp_nystrom import rp_nystrom

TWO_RHO_0 = 6.383202050647408


def find_kernel_temperature(
    scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    n: int,
    phi: Optional[float] = None,
) -> torch.Tensor:
    if phi is not None:
        n = int(n * phi ** 2)
    b = math.log(n) / (scale * q_scale * k_scale) + 2.0
    upper = b / (2.0 * lambert_w_circ_exp((b / TWO_RHO_0).log()))
    return torch.sqrt(k_scale / q_scale * upper)


def _finish_compress_kv(
    keys_scaled: torch.Tensor,
    values: torch.Tensor,
    sqd_knorm: torch.Tensor,
    key_multiplier: torch.Tensor,
    coreset: torch.Tensor,
    kernel_inv: torch.Tensor,
    kernel_core: torch.Tensor,
    r: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nyström weights and aggregated values for the first ``r`` pivots."""
    e = keys_scaled.shape[-1]
    coreset = coreset[..., :r]
    kernel_inv = kernel_inv[..., :r, :r]
    kernel_core = kernel_core[..., :r, :]

    cmpd_keys = keys_scaled.gather(
        -2, coreset.unsqueeze(-1).expand(*coreset.shape, e),
    )
    cmpd_sqd_knorms = sqd_knorm.gather(-1, coreset)
    cmpd_keys = cmpd_keys / key_multiplier.unsqueeze(-1)

    w_mat = torch.einsum("...rs, ...sl -> ...rl", kernel_inv, kernel_core)
    scaling = -cmpd_sqd_knorms.unsqueeze(-1) + sqd_knorm.unsqueeze(-2)
    w_mat = w_mat * torch.exp(scaling / 2.0)

    cmpd_values = torch.einsum("...rn, ...nd -> ...rd", w_mat, values)
    w = w_mat.sum(dim=-1)
    return cmpd_keys, cmpd_values, w, coreset


def compress_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    r: int,
    scale: Optional[float] = None,
    q_scale: Optional[torch.Tensor] = None,
    phi: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduce key-value pairs to a weighted coreset of size r.

    Args:
        keys: (batch, seq_len, dim)
        values: (batch, seq_len, v_dim)
        r: target coreset size
        scale: attention scale (default 1/sqrt(dim))
        q_scale: (batch, 1) query norm bound; defaults to max key norm
        phi: optional temperature adjustment
        generator: optional RNG for rp_nystrom pivots

    Returns:
        cmpd_keys, cmpd_values, w, coreset_local indices (batch, r)
    """
    e = keys.shape[-1]
    n = keys.shape[-2]

    if n <= r:
        w = torch.ones(*keys.shape[:2], device=keys.device, dtype=keys.dtype)
        local = torch.arange(n, device=keys.device).expand(keys.shape[0], n)
        return keys, values, w, local

    scale = scale or 1.0 / sqrt(e)
    sqd_knorm = keys.square().sum(dim=-1)
    k_scale = sqd_knorm.sqrt().amax(dim=-1, keepdim=True)

    if q_scale is None:
        q_scale = k_scale
    else:
        b = q_scale.shape[0]
        c = k_scale.shape[0] // b
        q_scale = q_scale.reshape(b, 1).expand(b, c).reshape(b * c, 1)

    tau = find_kernel_temperature(scale, q_scale, k_scale, n, phi=phi)
    key_multiplier = sqrt(scale) / tau
    keys = keys * key_multiplier.unsqueeze(-1)
    sqd_knorm = sqd_knorm * (key_multiplier ** 2)

    coreset, kernel_core_inv, kernel_rows = rp_nystrom(
        keys=keys, sqd_knorm=sqd_knorm, r=r, generator=generator,
    )

    return _finish_compress_kv(
        keys, values, sqd_knorm, key_multiplier,
        coreset, kernel_core_inv, kernel_rows, r,
    )
