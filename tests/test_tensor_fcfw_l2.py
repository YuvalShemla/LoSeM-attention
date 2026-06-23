"""Smoke + determinism tests for Tensor FCFW l2."""

import numpy as np

from src.algorithms.base import AttentionInput
from src.algorithms.tensor_fcfw_l2 import TensorFCFWL2
from src.algorithms.fcfw_l2 import FCFrankWolfeL2


def _make_problem(n=2000, d=64, sink=4, window=64, seed=0):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n, d)).astype(np.float32)
    values = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal((d,)).astype(np.float32)
    scale = 1.0 / np.sqrt(d)
    logits = scale * (keys @ query)
    special_idx = np.concatenate(
        [np.arange(sink), np.arange(n - window, n)],
    ).astype(np.int64)
    candidate_idx = np.arange(sink, n - window).astype(np.int64)
    problem = AttentionInput(
        query=query, keys=keys, values=values, head_dim=d,
        logits=logits, special_idx=special_idx, candidate_idx=candidate_idx,
    )
    return problem, scale


def _exact(problem, scale):
    logits = scale * (problem.keys @ problem.query)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return w @ problem.values


def _rel_l2(approx, exact):
    return float(np.linalg.norm(approx - exact) / (np.linalg.norm(exact) + 1e-12))


def test_runs_and_improves():
    problem, scale = _make_problem()
    exact = _exact(problem, scale)
    algo = TensorFCFWL2(device="cpu")
    algo.prepare(problem.keys, problem.values, problem.head_dim)

    budgets = [16, 64, 256, 1024]
    errs = []
    rng = np.random.default_rng(42)
    for b in budgets:
        out = algo.run(problem, b, rng)
        assert out.output.shape == (problem.head_dim,)
        assert out.actual_budget == len(problem.special_idx) + b
        errs.append(_rel_l2(out.output, exact))

    print("budgets:", budgets)
    print("tensor errors :", errs)
    assert all(np.isfinite(e) for e in errs)
    for a, b in zip(errs, errs[1:]):
        assert b <= a * 1.05 + 1e-6, f"large non-monotone jump: {errs}"
    assert errs[-1] < errs[0]


def test_determinism():
    problem, _ = _make_problem(seed=1)
    a1 = TensorFCFWL2(device="cpu")
    a1.prepare(problem.keys, problem.values, problem.head_dim)
    a2 = TensorFCFWL2(device="cpu")
    a2.prepare(problem.keys, problem.values, problem.head_dim)
    rng = np.random.default_rng(0)
    o1 = a1.run(problem, 128, rng)
    o2 = a2.run(problem, 128, rng)
    assert np.allclose(o1.output, o2.output, atol=1e-6)


def test_nested_cache_matches_fresh():
    """A budget reached via warm-start must match a fresh single-budget run."""
    problem, scale = _make_problem(seed=2)
    exact = _exact(problem, scale)
    rng = np.random.default_rng(0)

    warm = TensorFCFWL2(device="cpu")
    warm.prepare(problem.keys, problem.values, problem.head_dim)
    warm.run(problem, 64, rng)        # seeds the cache
    out_warm = warm.run(problem, 256, rng)

    fresh = TensorFCFWL2(device="cpu")
    fresh.prepare(problem.keys, problem.values, problem.head_dim)
    out_fresh = fresh.run(problem, 256, rng)

    assert np.allclose(out_warm.output, out_fresh.output, atol=1e-5)


def test_differs_from_plain_fcfw():
    """Value-aware selection should generally pick a different coreset."""
    problem, _ = _make_problem(seed=3)
    rng = np.random.default_rng(0)

    tensor = TensorFCFWL2(device="cpu")
    tensor.prepare(problem.keys, problem.values, problem.head_dim)
    plain = FCFrankWolfeL2(device="cpu")
    plain.prepare(problem.keys, problem.values, problem.head_dim)

    ot = tensor.run(problem, 128, rng)
    op = plain.run(problem, 128, rng)
    # Coresets need not be identical; the selected candidate sets should differ.
    st = set(ot.selected_indices.tolist())
    sp = set(op.selected_indices.tolist())
    assert st != sp


if __name__ == "__main__":
    test_runs_and_improves()
    test_determinism()
    test_nested_cache_matches_fresh()
    test_differs_from_plain_fcfw()
    print("OK")
