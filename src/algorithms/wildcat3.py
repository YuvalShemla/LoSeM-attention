"""
WildCat3: direct adapter to the vendored microsoft/wildcat kvcache path.

Unlike WildCat2, this does not port or wrap the kernels in loco-attention.
It imports and calls ``wildcat/examples/kvcache/compress_kv_cache.py`` plus
the original ``wildcat.weighted_attention`` implementation directly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax


def _wildcat_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "wildcat"


def _ensure_wildcat_imports() -> None:
    root = _wildcat_repo_root()
    kvcache = root / "examples" / "kvcache"
    if not (root / "wildcat" / "compress_kv.py").exists():
        raise ImportError(
            f"Vendored WildCat package not found at {root}. "
            "Expected wildcat/wildcat/compress_kv.py.",
        )
    for path in (root, kvcache):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_device(prefer: Optional[str]) -> torch.device:
    if prefer is not None:
        p = prefer.lower()
        if p in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "wildcat3 device='cuda' requested but CUDA is unavailable",
                )
            return torch.device("cuda")
        if p == "cpu":
            return torch.device("cpu")
        raise ValueError(
            f"wildcat3 device must be 'cuda', 'cpu', or null; got {prefer!r}",
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(
        f"wildcat3 dtype must be float32, bfloat16, or float16; got {name!r}",
    )


def _sink_and_window_from_indices(
    n: int,
    special_idx: np.ndarray,
    candidate_idx: np.ndarray,
) -> tuple[int, int]:
    if len(candidate_idx) > 0:
        sink_size = int(candidate_idx[0])
        window_size = n - int(candidate_idx[-1]) - 1
        return sink_size, window_size

    special = set(int(i) for i in special_idx.tolist())
    sink_size = 0
    while sink_size in special:
        sink_size += 1
    return sink_size, max(0, n - sink_size)


class WildCat3(AttentionAlgorithm):
    """
    Direct original-WildCat kvcache adapter.

    The evaluator still supplies query/key/value tensors and budget sweeps, but
    compression and WtdAttn are executed by the vendored WildCat package. The
    requested budget is interpreted as the target compressed middle size.
    """

    def __init__(
        self,
        num_bins: Optional[int] = 1,
        bin_r: Optional[int] = None,
        device: Optional[str] = None,
        dtype: str = "float32",
    ):
        _ensure_wildcat_imports()
        self.num_bins = int(num_bins) if num_bins is not None else None
        self.bin_r = int(bin_r) if bin_r is not None else None
        self._device = _resolve_device(device)
        self._dtype = _resolve_dtype(dtype)

    @property
    def name(self) -> str:
        if self.num_bins is not None:
            return f"WildCat3-C{self.num_bins}"
        if self.bin_r is not None:
            return f"WildCat3-bin{self.bin_r}"
        return "WildCat3"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        del rng

        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n = len(problem.keys)
        n_sp = len(special_idx)

        if len(candidate_idx) == 0:
            out = softmax(problem.logits[special_idx]) @ problem.values[special_idx]
            return AttentionOutput(
                output=out.astype(np.float32),
                actual_budget=n_sp,
                selected_indices=special_idx,
            )

        from compress_kv_cache import CompressKVConfig, compress_kv_cache
        from wildcat.weighted_attention import weighted_attention

        sink_size, window_size = _sink_and_window_from_indices(
            n, special_idx, candidate_idx,
        )
        middle_size = n - sink_size - window_size
        if middle_size <= 0:
            out = softmax(problem.logits[special_idx]) @ problem.values[special_idx]
            return AttentionOutput(
                output=out.astype(np.float32),
                actual_budget=n_sp,
                selected_indices=special_idx,
            )

        target_r = min(max(int(budget), 1), middle_size)
        compression_ratio = 1.0 - (target_r / middle_size)

        num_bins = self.num_bins
        if num_bins is not None:
            num_bins = max(1, min(num_bins, middle_size))
        cfg = CompressKVConfig(
            compression_ratio=compression_ratio,
            num_bins=num_bins,
            bin_r=self.bin_r,
            sink_size=sink_size,
            window_size=window_size,
        )

        scale = 1.0 / math.sqrt(problem.head_dim)
        keys = torch.as_tensor(
            problem.keys, dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0).to(
            device=self._device, dtype=self._dtype,
        )
        values = torch.as_tensor(
            problem.values, dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0).to(
            device=self._device, dtype=self._dtype,
        )

        with torch.no_grad():
            cmp_keys, cmp_values = compress_kv_cache(keys, values, cfg, scale)
            core_keys = cmp_keys.squeeze(1)
            core_values = cmp_values[..., :-1].squeeze(1)
            core_one = cmp_values[..., -1].squeeze(1)

            q = torch.as_tensor(
                problem.query, dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0).to(
                device=self._device, dtype=self._dtype,
            )
            vmin = values.amin(dim=-2, keepdim=True).squeeze(1)
            vmax = values.amax(dim=-2, keepdim=True).squeeze(1)
            out = weighted_attention(
                queries=q,
                core_keys=core_keys,
                core_values=core_values,
                core_one=core_one,
                scale=scale,
                min_val=vmin,
                max_val=vmax,
            )

        output = out.squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
        actual_budget = int(core_keys.shape[-2])
        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
            selected_indices=None,
            debug_payload={
                "target_middle_budget": int(target_r),
                "middle_size": int(middle_size),
                "compression_ratio": float(compression_ratio),
                "num_bins": num_bins,
                "bin_r": self.bin_r,
                "dtype": str(self._dtype).replace("torch.", ""),
            },
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        return [
            WildCat3(
                num_bins=cfg.get("num_bins", 1),
                bin_r=cfg.get("bin_r"),
                device=cfg.get("device"),
                dtype=cfg.get("dtype", "float32"),
            ),
        ]
