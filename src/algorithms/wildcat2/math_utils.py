"""Lambert W helpers (ported from microsoft/wildcat)."""

import math

import torch


def lambert_w_circ_exp(log_y: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    beta_n = torch.where(
        log_y > 1.0,
        log_y - torch.log(log_y),
        (log_y - 1.0).exp(),
    )
    for _ in range(iterations):
        beta_n = (
            beta_n / (1.0 + beta_n)
            * (1.0 + log_y - torch.log(beta_n))
        )
    return beta_n
