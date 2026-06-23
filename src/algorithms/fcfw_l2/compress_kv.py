"""
CompressKV variant that selects the coreset by fully-corrective Frank-Wolfe.

Reuses wildcat2's temperature (``find_kernel_temperature``), key rescaling, and
weight/value aggregation (``_finish_compress_kv``) verbatim, swapping only the
pivot selection from randomly-pivoted Nystrom to FCFW. This guarantees that any
difference vs WildCat2 is attributable purely to the selection rule.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Optional, Tuple

import torch

from ..wildcat2.compress_kv import _finish_compress_kv, find_kernel_temperature
from .fcfw_select import fcfw_select


def compress_kv_fcfw(
    keys: torch.Tensor,
    values: torch.Tensor,
    r: int,
    scale: Optional[float] = None,
    q_scale: Optional[torch.Tensor] = None,
    phi: Optional[float] = None,
    state: Optional[Dict] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Optional[Dict]]:
    """Reduce key-value pairs to a weighted coreset of size ``r`` via FCFW.

    Returns ``((cmpd_keys, cmpd_values, w, coreset_local), new_state)`` where
    ``new_state`` warm-starts a larger budget on the same (rescaled) keys.
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
    keys = keys * key_multiplier.unsqueeze(-1)
    sqd_knorm = sqd_knorm * (key_multiplier ** 2)

    coreset, kernel_inv, kernel_core, new_state = fcfw_select(
        keys, sqd_knorm, r, state=state,
    )

    out = _finish_compress_kv(
        keys, values, sqd_knorm, key_multiplier,
        coreset, kernel_inv, kernel_core, r,
    )
    return out, new_state
