"""WildCat2: parity with reference wildcat, CUDA smoke, interface."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.wildcat2 import WildCat2, compress_kv
from src.algorithms.wildcat2._device import resolve_device
from src.algorithms.base import AttentionInput
from src.core import compute_special_indices


def _ref_wildcat_available() -> bool:
    ref = Path.home() / "projects" / "wildcat"
    return (ref / "wildcat" / "compress_kv.py").exists()


@pytest.mark.skipif(
    not _ref_wildcat_available(),
    reason="reference ~/projects/wildcat not found",
)
def test_compress_kv_matches_reference_cpu():
    sys.path.insert(0, str(Path.home() / "projects" / "wildcat"))
    from wildcat.compress_kv import compress_kv as ref_compress_kv

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)

    n, d = 500, 64
    keys = torch.randn(1, n, d)
    values = torch.randn(1, n, d)
    r = 64
    scale = 1.0 / (d ** 0.5)
    q_scale = torch.tensor([[1.5]])

    ck, cv, w, loc = compress_kv(
        keys, values, r, scale=scale, q_scale=q_scale, generator=gen,
    )
    torch.manual_seed(42)
    rck, rcv, rw = ref_compress_kv(
        keys, values, r, scale=scale, q_scale=q_scale,
    )

    np.testing.assert_allclose(
        ck.numpy(), rck.numpy(), rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        cv.numpy(), rcv.numpy(), rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        w.numpy(), rw.numpy(), rtol=1e-4, atol=1e-5,
    )
    assert loc.shape == (1, r)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_wildcat2_auto_uses_cuda():
    assert resolve_device().type == "cuda"
    assert WildCat2()._device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_wildcat2_cuda_deterministic_small_budget():
    rng = np.random.default_rng(0)
    head_dim = 128
    n = 80
    keys = rng.standard_normal((n, head_dim)).astype(np.float32)
    values = rng.standard_normal((n, head_dim)).astype(np.float32)
    q = rng.standard_normal(head_dim).astype(np.float32)
    logits = (q @ keys.T) / np.sqrt(head_dim)
    sp_idx, cand_idx = compute_special_indices(n, 1, 10)
    problem = AttentionInput(
        query=q, keys=keys, values=values, head_dim=head_dim,
        logits=logits, special_idx=sp_idx, candidate_idx=cand_idx,
    )
    wc = WildCat2(device="cpu")
    wc.prepare(keys, values, head_dim, seed=0)
    out_cpu = wc.run(problem, 500, np.random.default_rng(0))
    wc_gpu = WildCat2(device="cuda")
    wc_gpu.prepare(keys, values, head_dim, seed=0)
    out_gpu = wc_gpu.run(problem, 500, np.random.default_rng(0))
    np.testing.assert_allclose(
        out_gpu.output, out_cpu.output, rtol=1e-4, atol=1e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_wildcat2_cuda_smoke():
    rng = np.random.default_rng(1)
    head_dim = 128
    n = 2000
    keys = rng.standard_normal((n, head_dim)).astype(np.float32)
    values = rng.standard_normal((n, head_dim)).astype(np.float32)
    q = rng.standard_normal(head_dim).astype(np.float32)
    logits = (q @ keys.T) / np.sqrt(head_dim)
    sp_idx, cand_idx = compute_special_indices(n, 1, 32)
    problem = AttentionInput(
        query=q, keys=keys, values=values, head_dim=head_dim,
        logits=logits, special_idx=sp_idx, candidate_idx=cand_idx,
    )
    out = WildCat2(device="cuda").run(
        problem, 256, np.random.default_rng(0),
    )
    assert np.all(np.isfinite(out.output))
    assert out.actual_budget > 0
