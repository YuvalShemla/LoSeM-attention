"""Smoke + determinism + nesting tests for Tensor FCFW lq."""

import numpy as np
import torch

from src.core import compute_special_indices
from src.algorithms.base import AttentionInput
from src.algorithms.tensor_fcfw_lq import TensorFCFWLq
from src.algorithms.tensor_fcfw_lq.select_lq import (
    build_attention_profiles,
    correction_interval,
    lq_objective,
    select_lq_coreset,
    _normalized_correction_system,
    _residual_lq_best_column_scalar,
    _score_residual_lq_candidates,
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
    """Nested greedy + corrective solve => lq numerator error over Q is non-increasing."""
    rng = np.random.default_rng(0)
    n, d, m = 1500, 64, 400
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    a = build_attention_profiles(probes, keys, scale)
    target = a @ values

    def num_obj(sel, cv):
        return (target - a[:, sel] @ cv).norm(dim=1).mean().item()

    prev = None
    for b in [8, 16, 32, 64, 128, 256]:
        sel, cv, w, _ = select_lq_coreset(
            probes, keys, values, b, scale, numerator_only=True,
        )
        assert torch.allclose(w, torch.ones_like(w))
        j = num_obj(sel, cv)
        if prev is not None:
            assert j <= prev + 1e-3, f"lq numerator objective increased at b={b}"
        prev = j


def test_irls_reduces_lq_objective_vs_l2():
    """IRLS correction yields a lower lq numerator objective than the plain L2 solve."""
    rng = np.random.default_rng(1)
    n, d, m = 1200, 48, 300
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    a = build_attention_profiles(probes, keys, scale)
    target = a @ values

    def num_obj(sel, cv):
        return (target - a[:, sel] @ cv).norm(dim=1).mean().item()

    sel_l2, cv_l2, w_l2, _ = select_lq_coreset(
        probes, keys, values, 128, scale, irls_iters=1, rcond=1e-10,
        numerator_only=True,
    )
    sel_lq, cv_lq, w_lq, _ = select_lq_coreset(
        probes, keys, values, 128, scale, irls_iters=10, rcond=1e-10,
        numerator_only=True,
    )
    assert num_obj(sel_lq, cv_lq) <= num_obj(sel_l2, cv_l2) + 1e-5


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


def test_normalized_correction_beats_numerator_only():
    """On the same support, normalized IRLS beats numerator-only IRLS at exact-d eval."""
    from src.algorithms.tensor_fcfw_lq.select_lq import (
        build_normalized_candidate_design,
        _irls_solve,
    )

    rng = np.random.default_rng(2)
    n, d, m = 2000, 64, 400
    sink, window = 4, 64
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    sp_idx = np.concatenate([np.arange(sink), np.arange(n - window, n)])
    cand_idx = np.arange(sink, n - window)
    ref_keys = keys
    ref_values = values
    sp_t = torch.tensor(sp_idx, dtype=torch.long)

    cand_keys = keys[cand_idx]
    cand_values = values[cand_idx]

    coef, rhs = build_normalized_candidate_design(
        probes, cand_keys, ref_keys, ref_values, sp_t, scale,
    )
    a = build_attention_profiles(probes, cand_keys, scale)
    num_target = a @ cand_values

    def norm_err(sel, cv):
        resid = rhs - coef[:, sel] @ cv
        return resid.norm(dim=1).mean().item()

    b = 128
    sel, cv_norm, _, _ = select_lq_coreset(
        probes, cand_keys, cand_values, b, scale, numerator_only=True,
        ref_keys=ref_keys, ref_values=ref_values, sp_idx=sp_t,
    )
    cv_num = _irls_solve(a[:, sel], num_target, irls_iters=5, rcond=1e-3)
    assert norm_err(sel, cv_norm) <= norm_err(sel, cv_num) + 1e-4


def test_fc_lq_objective_non_increasing():
    """Exact FC-lq: mean per-query residual is non-increasing in budget."""
    rng = np.random.default_rng(11)
    n, d, m = 80, 16, 40
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)
    design = build_attention_profiles(probes, keys, scale)
    target = design @ values

    prev = None
    state = None
    for b in [2, 4, 8, 16]:
        sel, cv, _, state = select_lq_coreset(
            probes, keys, values, b, scale,
            oracle="fc_lq", numerator_only=True, state=state,
        )
        obj = lq_objective(design[:, sel], cv, target).item()
        if prev is not None:
            assert obj <= prev + 1e-4, f"fc_lq objective increased at b={b}"
        prev = obj


def test_fc_lq_values_optimal_for_its_support():
    """FC-lq values minimize the lq objective on the selected support."""
    rng = np.random.default_rng(12)
    n, d, m = 60, 16, 30
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)
    design = build_attention_profiles(probes, keys, scale)
    target = design @ values
    b = 12

    sel, cv, _, _ = select_lq_coreset(
        probes, keys, values, b, scale, oracle="fc_lq", numerator_only=True,
    )
    obj = lq_objective(design[:, sel], cv, target).item()
    cv_pert = cv + 0.01 * torch.randn_like(cv)
    obj_pert = lq_objective(design[:, sel], cv_pert, target).item()
    assert obj <= obj_pert + 1e-4


def test_fc_lq_nested_warm_start():
    """Warm-started fc_lq must match a fresh run at the same budget."""
    rng = np.random.default_rng(13)
    n, d, m = 50, 16, 25
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    state = None
    _, _, _, state = select_lq_coreset(
        probes, keys, values, 6, scale, oracle="fc_lq", state=state,
    )
    sel_w, cv_w, _, _ = select_lq_coreset(
        probes, keys, values, 12, scale, oracle="fc_lq", state=state,
    )
    sel_f, cv_f, _, _ = select_lq_coreset(
        probes, keys, values, 12, scale, oracle="fc_lq",
    )
    assert torch.equal(sel_w, sel_f)
    assert torch.allclose(cv_w, cv_f, atol=1e-5)


def test_fc_lq_algorithm_runs():
    problem, queries, _ = _make_problem(n=500, d=32, sink=2, window=32, seed=8)
    algo = TensorFCFWLq(device="cpu", oracle="fc_lq", exact_denominator=True)
    _prepare(algo, problem, queries)
    rng = np.random.default_rng(0)
    out = algo.run(problem, 16, rng)
    assert np.all(np.isfinite(out.output))


def test_residual_lq_vectorized_scoring_matches_scalar():
    """Batched candidate scoring agrees with the per-column reference."""
    rng = np.random.default_rng(23)
    n, d, m = 80, 16, 32
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)
    design = build_attention_profiles(probes, keys, scale)
    residual = (design @ values) - design[:, :3] @ _irls_ref_values(
        design[:, :3], design @ values, 5, 1e-3,
    )

    eligible = torch.ones(n, dtype=torch.bool)
    eligible[:3] = False
    best_i, gains = _score_residual_lq_candidates(
        design, residual, irls_iters=5, rcond=1e-3, eligible=eligible,
    )
    for i in range(n):
        if not eligible[i]:
            continue
        ref = _residual_lq_best_column_scalar(
            design, residual, i, irls_iters=5, rcond=1e-3,
        )
        assert torch.allclose(gains[i], ref, rtol=1e-4, atol=1e-5), f"mismatch at {i}"

    ref_best = max(
        (i for i in range(n) if eligible[i]),
        key=lambda i: _residual_lq_best_column_scalar(
            design, residual, i, irls_iters=5, rcond=1e-3,
        ).item(),
    )
    assert best_i == ref_best


def _irls_ref_values(a_sel, target, irls_iters, rcond):
    from src.algorithms.tensor_fcfw_lq.select_lq import _irls_solve
    return _irls_solve(a_sel, target, irls_iters, rcond)


def test_correction_interval_schedule():
  assert correction_interval(1, 400) == 1
  assert correction_interval(400, 400) == 1
  assert correction_interval(401, 400) == 2
  assert correction_interval(2000, 400) == 5
  assert correction_interval(4000, 400) == 10
  assert correction_interval(2000, 0) == 1


def test_residual_lq_deflated_runs_and_differs_on_correlated_keys():
    """Deflated scoring should diverge from plain residual_lq when keys correlate."""
    rng = np.random.default_rng(24)
    n, d, m = 120, 16, 40
    base = torch.tensor(rng.standard_normal((1, d)), dtype=torch.float32)
    # Many near-duplicate keys plus a few distinct ones.
    keys = base + 0.02 * torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    sel_plain, _, _, _ = select_lq_coreset(
        probes, keys, values, 8, scale,
        oracle="residual_lq", correction_period=0,
    )
    sel_defl, _, _, _ = select_lq_coreset(
        probes, keys, values, 8, scale,
        oracle="residual_lq_deflated", correction_period=0,
    )
    assert sel_plain.numel() == 8
    assert sel_defl.numel() == 8
    assert not torch.equal(sel_plain, sel_defl)


def test_residual_lq_deflated_algorithm_runs():
    problem, queries, _ = _make_problem(n=500, d=32, sink=2, window=32, seed=10)
    algo = TensorFCFWLq(
        device="cpu", oracle="residual_lq_deflated", exact_denominator=True,
    )
    _prepare(algo, problem, queries)
    rng = np.random.default_rng(0)
    out = algo.run(problem, 16, rng)
    assert np.all(np.isfinite(out.output))


def test_residual_lq_objective_non_increasing():
    """Residual-lq: training lq objective is non-increasing in budget."""
    rng = np.random.default_rng(21)
    n, d, m = 100, 16, 40
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)
    design = build_attention_profiles(probes, keys, scale)
    target = design @ values

    prev = None
    state = None
    for b in [2, 4, 8, 16, 32]:
        sel, cv, _, state = select_lq_coreset(
            probes, keys, values, b, scale,
            oracle="residual_lq", numerator_only=True, state=state,
        )
        obj = lq_objective(design[:, sel], cv, target).item()
        if prev is not None:
            assert obj <= prev + 1e-4, f"residual_lq objective increased at b={b}"
        prev = obj


def test_residual_lq_nested_warm_start():
    rng = np.random.default_rng(22)
    n, d, m = 60, 16, 30
    keys = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    values = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    probes = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float32)
    scale = 1.0 / np.sqrt(d)

    state = None
    _, _, _, state = select_lq_coreset(
        probes, keys, values, 6, scale, oracle="residual_lq", state=state,
    )
    sel_w, cv_w, _, _ = select_lq_coreset(
        probes, keys, values, 12, scale, oracle="residual_lq", state=state,
    )
    sel_f, cv_f, _, _ = select_lq_coreset(
        probes, keys, values, 12, scale, oracle="residual_lq",
    )
    assert torch.equal(sel_w, sel_f)
    assert torch.allclose(cv_w, cv_f, atol=1e-5)


def test_residual_lq_algorithm_runs():
    problem, queries, _ = _make_problem(n=500, d=32, sink=2, window=32, seed=9)
    algo = TensorFCFWLq(device="cpu", oracle="residual_lq", exact_denominator=True)
    _prepare(algo, problem, queries)
    rng = np.random.default_rng(0)
    out = algo.run(problem, 16, rng)
    assert np.all(np.isfinite(out.output))


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


def test_lbfgs_post_refinement_runs_and_reduces_unnorm_loss():
    from src.algorithms.tensor_fcfw_lq.lbfgs_refine import (
        core_unnorm_numerator_pred,
        core_unnorm_numerator_target,
        relative_unnorm_numerator_loss,
        _per_query_logit_shift,
    )

    problem, queries, _ = _make_problem(n=500, d=32, sink=2, window=32, seed=11)
    kw = dict(
        device="cpu", oracle="omp", exact_denominator=True,
        n_sink=2, local_window=32, lbfgs_steps=0,
    )
    base = TensorFCFWLq(**kw)
    refined = TensorFCFWLq(**{**kw, "lbfgs_steps": 8})
    _prepare(base, problem, queries)
    _prepare(refined, problem, queries)
    rng = np.random.default_rng(0)
    refined.run(problem, 16, rng)
    base.run(problem, 16, rng)

    k_b, v_b, w_b, _ = base._coreset_cache[16]
    k_r, v_r, w_r, _ = refined._coreset_cache[16]
    assert k_b.shape[0] > 0
    assert np.all(np.isfinite(k_r))
    assert np.all(np.isfinite(v_r))
    assert np.allclose(w_r, 1.0)

    probe_t = torch.tensor(base._probe_queries, dtype=torch.float32)
    ref_k = torch.tensor(problem.keys[: base._ref_pos + 1], dtype=torch.float32)
    ref_v = torch.tensor(problem.values[: base._ref_pos + 1], dtype=torch.float32)
    sp_idx, _ = compute_special_indices(base._ref_pos + 1, 2, 32)
    sp_t = torch.tensor(sp_idx, dtype=torch.long)
    scale = 1.0 / np.sqrt(32)
    shift = _per_query_logit_shift(probe_t, ref_k, scale)
    target = core_unnorm_numerator_target(
        probe_t, ref_k, ref_v, sp_t, scale, shift,
    )
    v_b_eff = torch.tensor(v_b) * torch.tensor(w_b).unsqueeze(-1)
    with torch.no_grad():
        pred_before = core_unnorm_numerator_pred(
            probe_t, torch.tensor(k_b), v_b_eff, scale, shift,
        )
        pred_after = core_unnorm_numerator_pred(
            probe_t, torch.tensor(k_r), torch.tensor(v_r), scale, shift,
        )
        loss_before = relative_unnorm_numerator_loss(pred_before, target)
        loss_after = relative_unnorm_numerator_loss(pred_after, target)
    assert loss_after <= loss_before + 1e-6


def test_lbfgs_steps_zero_is_default():
    algo = TensorFCFWLq(device="cpu", oracle="fw")
    assert algo.lbfgs_steps == 0
    assert algo.name == "TFCFW-lq-fw"


if __name__ == "__main__":
    test_runs_and_is_finite_and_bounded()
    test_lq_objective_decreases_with_budget()
    test_irls_reduces_lq_objective_vs_l2()
    test_bounded_when_budget_exceeds_probes()
    test_coreset_reused_across_query_positions()
    test_determinism()
    test_nested_cache_matches_fresh()
    test_exact_denominator_changes_output()
    test_normalized_correction_beats_numerator_only()
    test_fc_lq_objective_non_increasing()
    test_fc_lq_values_optimal_for_its_support()
    test_fc_lq_nested_warm_start()
    test_fc_lq_algorithm_runs()
    test_residual_lq_vectorized_scoring_matches_scalar()
    test_correction_interval_schedule()
    test_residual_lq_deflated_runs_and_differs_on_correlated_keys()
    test_residual_lq_deflated_algorithm_runs()
    test_residual_lq_objective_non_increasing()
    test_residual_lq_nested_warm_start()
    test_residual_lq_algorithm_runs()
    test_irls_differs_from_l2_solve()
    test_lbfgs_post_refinement_runs_and_reduces_unnorm_loss()
    test_lbfgs_steps_zero_is_default()
    print("OK")
