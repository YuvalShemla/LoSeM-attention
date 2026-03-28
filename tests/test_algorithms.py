"""
Test all registered algorithms for interface compliance.

Every method must:
  - Return AttentionOutput with correct shapes
  - Respect the budget (actual_budget > 0)
  - Implement the ABC fully
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent / "src")
)

from src.algorithms import METHOD_REGISTRY
from src.algorithms.base import (
    AttentionInput, AttentionOutput, AttentionAlgorithm,
)
from src.math_utils import (
    softmax, compute_special_indices,
)

HEAD_DIM = 128
SEQ_LEN = 500
N_QUERIES = 10
SEED = 42


@pytest.fixture
def synthetic_data():
    """Synthetic attention problem for testing."""
    rng = np.random.default_rng(SEED)
    Q = rng.standard_normal(
        (SEQ_LEN, HEAD_DIM)
    ).astype(np.float32)
    K = rng.standard_normal(
        (SEQ_LEN, HEAD_DIM)
    ).astype(np.float32)
    V = rng.standard_normal(
        (SEQ_LEN, HEAD_DIM)
    ).astype(np.float32)
    return Q, K, V


def _make_problem(Q, K, V, qpos):
    """Create AttentionInput for a query position."""
    n_causal = qpos + 1
    q = Q[qpos]
    keys = K[:n_causal]
    values = V[:n_causal]
    logits = (q @ keys.T) / np.sqrt(HEAD_DIM)
    sp_idx, cand_idx = (
        compute_special_indices(n_causal, 1, 10)
    )
    return AttentionInput(
        query=q, keys=keys, values=values,
        head_dim=HEAD_DIM, logits=logits,
        special_idx=sp_idx, candidate_idx=cand_idx,
    )


def test_all_methods_registered():
    """At least baselines + some algorithms exist."""
    baselines = [
        k for k, v in METHOD_REGISTRY.items()
        if v.kind == "baseline"
    ]
    algorithms = [
        k for k, v in METHOD_REGISTRY.items()
        if v.kind == "algorithm"
    ]
    assert len(baselines) >= 2
    assert len(algorithms) >= 1


def test_all_methods_are_subclass():
    """Every registered method is an AttentionAlgorithm."""
    for name, spec in METHOD_REGISTRY.items():
        assert issubclass(spec.cls, AttentionAlgorithm), (
            f"{name} is not AttentionAlgorithm"
        )


@pytest.mark.parametrize(
    "method_name",
    list(METHOD_REGISTRY.keys()),
)
def test_method_interface(method_name, synthetic_data):
    """Each method returns correct output shape."""
    Q, K, V = synthetic_data
    spec = METHOD_REGISTRY[method_name]
    rng = np.random.default_rng(SEED)

    # Get one instance
    if spec.kind == "baseline":
        instances = spec.cls.expand_from_config({})
    else:
        # Minimal config for algorithms
        cfg = {
            "n_groups": 8,
            "n_clusters": 16,
            "n_query_clusters": [4],
            "modes": ["hybrid"],
            "top_k_sweep": [2],
        }
        instances = spec.cls.expand_from_config(cfg)

    assert len(instances) > 0

    method = instances[0]
    qpos_list = list(range(
        SEQ_LEN - N_QUERIES, SEQ_LEN,
    ))

    # Prepare
    method.prepare(
        K, V, HEAD_DIM,
        queries=Q,
        query_positions=qpos_list,
        seed=SEED,
    )

    # Run on one query
    problem = _make_problem(
        Q, K, V, SEQ_LEN - 1,
    )
    budget = 50
    out = method.run(problem, budget, rng)

    assert isinstance(out, AttentionOutput)
    assert out.output.shape == (HEAD_DIM,)
    assert out.actual_budget > 0
    assert np.all(np.isfinite(out.output))


@pytest.mark.parametrize(
    "method_name",
    list(METHOD_REGISTRY.keys()),
)
def test_method_has_name(method_name):
    """Each method has a non-empty name."""
    spec = METHOD_REGISTRY[method_name]
    if spec.kind == "baseline":
        instances = spec.cls.expand_from_config({})
    else:
        cfg = {
            "n_groups": 8, "n_clusters": 16,
            "n_query_clusters": [4],
            "modes": ["hybrid"], "top_k_sweep": [2],
        }
        instances = spec.cls.expand_from_config(cfg)
    for inst in instances:
        assert len(inst.name) > 0
        assert isinstance(inst.kind, str)


def test_oracle_topk_respects_budget(synthetic_data):
    """OracleTopK should use at most budget + special."""
    Q, K, V = synthetic_data
    rng = np.random.default_rng(SEED)
    from src.algorithms.baselines import OracleTopK
    method = OracleTopK()
    problem = _make_problem(Q, K, V, SEQ_LEN - 1)
    budget = 20
    out = method.run(problem, budget, rng)
    # actual_budget <= budget + n_special
    n_special = len(problem.special_idx)
    assert out.actual_budget <= budget + n_special + 1
