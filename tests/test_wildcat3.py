"""WildCat3 direct original-package adapter smoke tests."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.base import AttentionInput
from src.algorithms.wildcat3 import WildCat3
from src.core import compute_special_indices


def test_wildcat3_cpu_smoke():
    rng = np.random.default_rng(0)
    head_dim = 64
    n = 256
    keys = rng.standard_normal((n, head_dim)).astype(np.float32)
    values = rng.standard_normal((n, head_dim)).astype(np.float32)
    q = rng.standard_normal(head_dim).astype(np.float32)
    logits = (q @ keys.T) / np.sqrt(head_dim)
    sp_idx, cand_idx = compute_special_indices(n, 1, 32)
    problem = AttentionInput(
        query=q,
        keys=keys,
        values=values,
        head_dim=head_dim,
        logits=logits,
        special_idx=sp_idx,
        candidate_idx=cand_idx,
    )

    method = WildCat3(device="cpu", dtype="float32", num_bins=4)
    out = method.run(problem, 64, np.random.default_rng(1))

    assert out.output.shape == (head_dim,)
    assert out.actual_budget > len(sp_idx)
    assert np.all(np.isfinite(out.output))
