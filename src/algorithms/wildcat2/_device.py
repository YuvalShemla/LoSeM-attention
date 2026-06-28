"""Device selection for WildCat2 torch kernels."""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        p = prefer.lower()
        if p in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "wildcat2 device='cuda' requested but CUDA is unavailable",
                )
            return torch.device("cuda")
        if p == "cpu":
            return torch.device("cpu")
        raise ValueError(
            f"wildcat2 device must be 'cuda', 'cpu', or null; got {prefer!r}",
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_generator_from_numpy(
    rng,
    device: torch.device,
) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(0, 2**63 - 1)))
    return gen
