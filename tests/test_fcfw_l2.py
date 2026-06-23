"""Smoke + monotonicity tests for FC Frank-Wolfe l2."""

import numpy as np

from src.algorithms.base import AttentionInput
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


def test_runs_and_monotone():
    problem, scale = _make_problem()
    exact = _exact(problem, scale)
    algo = FCFrankWolfeL2(device="cpu")
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
    print("errors :", errs)
    assert all(np.isfinite(e) for e in errs)
    # Kernel-space residual is monotone by construction; the downstream
    # attention error is typically (not strictly) monotone. Require overall
    # improvement and only mild non-monotonic bumps between steps.
    for a, b in zip(errs, errs[1:]):
        assert b <= a * 1.05 + 1e-6, f"large non-monotone jump: {errs}"
    assert errs[-1] < errs[0]


def test_determinism():
    problem, _ = _make_problem(seed=1)
    a1 = FCFrankWolfeL2(device="cpu")
    a1.prepare(problem.keys, problem.values, problem.head_dim)
    a2 = FCFrankWolfeL2(device="cpu")
    a2.prepare(problem.keys, problem.values, problem.head_dim)
    rng = np.random.default_rng(0)
    o1 = a1.run(problem, 128, rng)
    o2 = a2.run(problem, 128, rng)
    assert np.allclose(o1.output, o2.output, atol=1e-6)
    assert np.array_equal(np.sort(o1.selected_indices), np.sort(o2.selected_indices))


if __name__ == "__main__":
    test_runs_and_monotone()
    test_determinism()
    print("OK")
