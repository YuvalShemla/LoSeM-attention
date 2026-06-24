"""Smoke + determinism + nesting tests for Tensor FCFW lq."""

import numpy as np
import torch

from src.core import compute_special_indices
from src.algorithms.base import AttentionInput
from src.algorithms.tensor_fcfw_lq import TensorFCFWLq
from src.algorithms.tensor_fcfw_lq.select_lq import (
    build_attention_profiles,
    select_lq_coreset,
)


def _make_problem(n=2000, d=64, sink=4, window=64, seed=0):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n, d)).astype(np.float32)
    values = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((n, d)).astype(np.float32)
    query = queries[n - 1]
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
    return problem, queries, scale


def _exact(problem, scale):
    logits = scale * (problem.keys @ problem.query)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return w @ problem.values


def _rel_l2(approx, exact):
    return float(np.linalg.norm(approx - exact) / (np.linalg.norm(exact) + 1e-12))


def _prepare(algo, problem, queries, query_positions=None):
    if query_positions is None:
        query_positions = [len(queries) - 1]
    algo.prepare(
        problem.keys, problem.values, problem.head_dim,
        queries=queries, query_positions=query_positions,
    )


def test_runs_and_is_finite_and_bounded():
    """Output is finite, right shape, and stays within the value range."""
    problem, queries, _ = _make_problem()
    algo = TensorFCFWLq(device="cpu", exact_denominator=True)
    _prepare(algo, problem, queries)

    vmin = problem.values.min(axis=0)
    vmax = problem.values.max(axis=0)
    rng = np.random.default_rng(42)
    for b in [16, 64, 256]:
        out = algo.run(problem, b, rng)
        assert out.output.shape == (problem.head_dim,)
        assert out.actual_budget == len(problem.special_idx) + b
        assert np.all(np.isfinite(out.output))
        assert np.all(out.output >= vmin - 1e-4)
        assert np.all(out.output <= vmax + 1e-4)


def test_lq_objective_decreases_with_budget():
    """Nested greedy + corrective solve => lq error over Q is non-increasing."""
    rng = np.random.default_rng(0)
    n, d, m = 1500, 64, 400
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    a = build_attention_profiles(probes, keys, scale)
    mat = torch.cat([torch.ones(n, 1), values], dim=1)
    target = a @ mat

    def lq_obj(sel, cv, w):
        u = torch.cat([w.unsqueeze(1), cv], dim=1)
        return (target - a[:, sel] @ u).norm(dim=1).mean().item()

    prev = None
    for b in [8, 16, 32, 64, 128, 256]:
        sel, cv, w, _ = select_lq_coreset(probes, keys, values, b, scale)
        j = lq_obj(sel, cv, w)
        if prev is not None:
            assert j <= prev + 1e-3, f"lq objective increased at b={b}"
        prev = j


def test_irls_reduces_lq_objective_vs_l2():
    """IRLS correction yields a lower lq objective than the plain L2 solve."""
    rng = np.random.default_rng(1)
    n, d, m = 1200, 48, 300
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    a = build_attention_profiles(probes, keys, scale)
    mat = torch.cat([torch.ones(n, 1), values], dim=1)
    target = a @ mat

    def lq_obj(sel, cv, w):
        u = torch.cat([w.unsqueeze(1), cv], dim=1)
        return (target - a[:, sel] @ u).norm(dim=1).mean().item()

    # Well-determined regime (r < m) with negligible damping isolates the
    # IRLS-vs-L2 property on the (undamped) L_{2,1} objective.
    sel_l2, cv_l2, w_l2, _ = select_lq_coreset(
        probes, keys, values, 128, scale, irls_iters=1, rcond=1e-10,
    )
    sel_lq, cv_lq, w_lq, _ = select_lq_coreset(
        probes, keys, values, 128, scale, irls_iters=10, rcond=1e-10,
    )
    assert lq_obj(sel_lq, cv_lq, w_lq) <= lq_obj(sel_l2, cv_l2, w_l2) + 1e-5


def test_bounded_when_budget_exceeds_probes():
    """Regression: budget >= #probes must not explode the synthetic values.

    The value solve is under-determined once r >= m; the truncated SVD solve
    must keep the output bounded within the value range (previously it blew up).
    """
    n, d = 3000, 64
    problem, queries, _ = _make_problem(n=n, d=d, seed=7)
    algo = TensorFCFWLq(device="cpu", exact_denominator=True)
    m = algo.n_train_queries
    vmin = problem.values.min(axis=0)
    vmax = problem.values.max(axis=0)
    _prepare(algo, problem, queries)
    rng = np.random.default_rng(0)
    for b in [m // 2, m, 2 * m]:        # below, at, and above the #probes
        out = algo.run(problem, b, rng)
        assert np.all(np.isfinite(out.output))
        assert np.all(out.output >= vmin - 1e-4)
        assert np.all(out.output <= vmax + 1e-4)


def test_coreset_reused_across_query_positions():
    """Coreset is built once at ref_pos; only special tokens change per query."""
    rng = np.random.default_rng(5)
    n, d, window, n_q = 3000, 64, 128, 10
    keys = rng.standard_normal((n, d)).astype(np.float32)
    values = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((n, d)).astype(np.float32)
    scale = 1.0 / np.sqrt(d)

    ref_pos = n - 1
    qpos_list = list(range(n - n_q, n))
    algo = TensorFCFWLq(
        device="cpu", n_sink=1, local_window=window, exact_denominator=True,
    )
    algo.prepare(
        keys, values, d, queries=queries, query_positions=qpos_list,
    )

    budget = 64
    rng_run = np.random.default_rng(0)
    outs = []
    core_parts = []
    for qpos in qpos_list:
        n_causal = qpos + 1
        sp, cand = compute_special_indices(n_causal, 1, window)
        query = queries[qpos]
        logits = scale * (keys[:n_causal] @ query)
        problem = AttentionInput(
            query=query,
            keys=keys[:n_causal],
            values=values[:n_causal],
            head_dim=d,
            logits=logits,
            special_idx=sp,
            candidate_idx=cand,
        )
        out = algo.run(problem, budget, rng_run)
        outs.append(out)
        core_parts.append(out.selected_indices[len(sp):])

    assert all(np.array_equal(core_parts[0], cp) for cp in core_parts[1:])
    assert len({len(cp) for cp in core_parts}) == 1


def test_determinism():
    problem, queries, _ = _make_problem(seed=1)
    a1 = TensorFCFWLq(device="cpu")
    _prepare(a1, problem, queries)
    a2 = TensorFCFWLq(device="cpu")
    _prepare(a2, problem, queries)
    rng = np.random.default_rng(0)
    o1 = a1.run(problem, 128, rng)
    o2 = a2.run(problem, 128, rng)
    assert np.allclose(o1.output, o2.output, atol=1e-6)


def test_nested_cache_matches_fresh():
    """A budget reached via warm-start must match a fresh single-budget run."""
    problem, queries, _ = _make_problem(seed=2)
    rng = np.random.default_rng(0)

    warm = TensorFCFWLq(device="cpu")
    _prepare(warm, problem, queries)
    warm.run(problem, 64, rng)        # seeds the cache
    out_warm = warm.run(problem, 256, rng)

    fresh = TensorFCFWLq(device="cpu")
    _prepare(fresh, problem, queries)
    out_fresh = fresh.run(problem, 256, rng)

    assert np.allclose(out_warm.output, out_fresh.output, atol=1e-5)
    assert np.array_equal(
        np.sort(out_warm.selected_indices), np.sort(out_fresh.selected_indices),
    )


def test_exact_denominator_changes_output():
    problem, queries, _ = _make_problem(seed=3)
    rng = np.random.default_rng(0)
    a0 = TensorFCFWLq(device="cpu", exact_denominator=False)
    a1 = TensorFCFWLq(device="cpu", exact_denominator=True)
    _prepare(a0, problem, queries)
    _prepare(a1, problem, queries)
    o0 = a0.run(problem, 64, rng)
    o1 = a1.run(problem, 64, rng)
    assert np.linalg.norm(o0.output - o1.output) > 0


def test_irls_differs_from_l2_solve():
    """IRLS (lq) correction should differ from the plain L2 query-space solve."""
    problem, queries, _ = _make_problem(seed=4)
    rng = np.random.default_rng(0)
    l2 = TensorFCFWLq(device="cpu", irls_iters=1)
    lq = TensorFCFWLq(device="cpu", irls_iters=8)
    _prepare(l2, problem, queries)
    _prepare(lq, problem, queries)
    o_l2 = l2.run(problem, 128, rng)
    o_lq = lq.run(problem, 128, rng)
    assert np.linalg.norm(o_l2.output - o_lq.output) > 0


if __name__ == "__main__":
    test_runs_and_is_finite_and_bounded()
    test_lq_objective_decreases_with_budget()
    test_irls_reduces_lq_objective_vs_l2()
    test_bounded_when_budget_exceeds_probes()
    test_coreset_reused_across_query_positions()
    test_determinism()
    test_nested_cache_matches_fresh()
    test_exact_denominator_changes_output()
    test_irls_differs_from_l2_solve()
    print("OK")
