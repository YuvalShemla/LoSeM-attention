"""
CompressKV via tensor fully-corrective Frank-Wolfe.

Reuses wildcat2's temperature + key rescaling, but selects the coreset by
value-aware tensor FCFW and computes the synthetic coreset values by the
algorithm's *own* fully-corrective tensor solve

    U = (K_SS^{-1} K_{S,:}  (.)  exp(scaling/2)) @ M,   M = [1, V],

rather than calling wildcat2's ``_finish_compress_kv``. The ``exp(scaling/2)``
diagonal factor is the attention kernel's own Gram expressed via the stable
normalized-Gaussian features (so the synthetic values are the true minimizer in
the kernel that ``weighted_attention`` consumes). ``U[..., 0]`` is the synthetic
weight (mass channel), ``U[..., 1:]`` the synthetic values.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Optional, Tuple

import torch

from ..wildcat2.compress_kv import find_kernel_temperature
from .tensor_fcfw_select import tensor_fcfw_select


def compress_kv_tensor_fcfw(
    keys: torch.Tensor,
    values: torch.Tensor,
    r: int,
    scale: Optional[float] = None,
    q_scale: Optional[torch.Tensor] = None,
    phi: Optional[float] = None,
    oracle: str = "fw",
    state: Optional[Dict] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Optional[Dict]]:
    """Reduce key-value pairs to a weighted coreset of size ``r`` via tensor FCFW.

    Returns ``((cmpd_keys, cmpd_values, w, coreset_local), new_state)``.
    """
    e = keys.shape[-1]
    n = keys.shape[-2]

    if n <= r:
        w = torch.ones(*keys.shape[:2], device=keys.device, dtype=keys.dtype)
        local = torch.arange(n, device=keys.device).expand(keys.shape[0], n)
        return (keys, values, w, local), state

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
    keys_s = keys * key_multiplier.unsqueeze(-1)
    sqd_knorm = sqd_knorm * (key_multiplier ** 2)

    # Augmented target rows m_i = [1, v_i].
    ones_col = torch.ones(
        (*values.shape[:-1], 1), dtype=values.dtype, device=values.device,
    )
    aug = torch.cat([ones_col, values], dim=-1)

    coreset, kernel_inv, kernel_core, new_state = tensor_fcfw_select(
        keys_s, sqd_knorm, aug, r, oracle=oracle, state=state,
    )

    # Own fully-corrective tensor value solve (attention-kernel corrected).
    cmpd_sqd_knorms = sqd_knorm.gather(-1, coreset)
    w_mat = torch.einsum("...rs, ...sn -> ...rn", kernel_inv, kernel_core)
    scaling = -cmpd_sqd_knorms.unsqueeze(-1) + sqd_knorm.unsqueeze(-2)
    w_mat = w_mat * torch.exp(scaling / 2.0)

    u_aug = torch.einsum("...rn, ...np -> ...rp", w_mat, aug)
    w = u_aug[..., 0]
    cmpd_values = u_aug[..., 1:]

    cmpd_keys = keys_s.gather(
        -2, coreset.unsqueeze(-1).expand(*coreset.shape, e),
    ) / key_multiplier.unsqueeze(-1)

    return (cmpd_keys, cmpd_values, w, coreset), new_state
