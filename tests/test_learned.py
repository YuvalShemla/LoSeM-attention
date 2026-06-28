"""Tests for learned coreset (pure init, nested monotone budget, exact-eval objective)."""

import numpy as np
import torch

from src.algorithms.base import AttentionInput
from src.algorithms.learned import LearnedCoreset
from src.algorithms.learned.learn_coreset import (
    _forward_pred,
    _precompute_targets,
    build_probe_queries,
    learn_kv_coreset,
    reference_position,
)
from src.algorithms.wildcat2.weighted_attention import weighted_attention
from src.core import compute_special_indices, full_attention


def _make_example(n=1500, d=48, sink=4, window=64, seed=0):
    rng = np.random.default_rng(seed)
    # Low-rank-ish keys/values so attention has structure a coreset can capture.
    basis_k = rng.standard_normal((8, d)).astype(np.float32)
    basis_v = rng.standard_normal((8, d)).astype(np.float32)
    coeff = rng.standard_normal((n, 8)).astype(np.float32)
    keys = (coeff @ basis_k + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    values = (coeff @ basis_v + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    queries = (
        rng.standard_normal((n, 8)).astype(np.float32) @ basis_k
    ).astype(np.float32)
    qpos = n - 1
    query = queries[qpos]
    scale = 1.0 / np.sqrt(d)
    logits = scale * (keys[: qpos + 1] @ query)
    sp_idx, cand_idx = compute_special_indices(qpos + 1, sink, window)
    problem = AttentionInput(
        query=query,
        keys=keys[: qpos + 1],
        values=values[: qpos + 1],
        head_dim=d,
        logits=logits,
        special_idx=sp_idx,
        candidate_idx=cand_idx,
    )
    return problem, keys, values, queries, qpos, sink, window


def _rel_l2(approx, exact):
    return float(np.linalg.norm(approx - exact) / (np.linalg.norm(exact) + 1e-12))


def test_train_excludes_test_queries():
    q = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    probes = build_probe_queries(
        queries=q, query_positions=[98, 99], ref_pos=99, n_train_queries=10,
    )
    for excluded in (q[98], q[99]):
        assert not np.any(np.all(probes == excluded, axis=1))
    # Trailing window positions 90..99 minus tests -> 90..97
    expected = [p for p in range(90, 100) if p not in (98, 99)]
    assert probes.shape[0] == len(expected)
    for i, p in enumerate(expected):
        assert np.all(probes[i] == q[p])


def test_probe_queries_use_latest_window_not_prefix_start():
    q = np.arange(200 * 4, dtype=np.float32).reshape(200, 4)
    probes = build_probe_queries(
        queries=q, query_positions=[199], ref_pos=199, n_train_queries=10,
    )
    # Window 190..199, test at 199 -> rows 190..198
    assert probes.shape[0] == 9
    assert np.all(probes[0] == q[190])
    assert np.all(probes[-1] == q[198])
    assert not np.any(np.all(probes == q[0], axis=1))


def test_reference_position():
    assert reference_position(100, [99]) == 99
    assert reference_position(100, None) == 99
    assert reference_position(100, [40, 41]) == 41


def test_no_random_queries():
    q = np.arange(50 * 4, dtype=np.float32).reshape(50, 4)
    probes = build_probe_queries(q, [49], 49, n_train_queries=999)
    # Only real context rows (positions 0..48), no synthetic augmentation.
    assert probes.shape[0] == 49
    for row in probes:
        assert np.any(np.all(q == row, axis=1))


def test_runs_and_eval_includes_special():
    problem, keys, values, queries, qpos, sink, window = _make_example(seed=2)
    full_out, _, _ = full_attention(
        problem.query, problem.keys, problem.values, problem.head_dim,
    )
    algo = LearnedCoreset(
        n_train_queries=64,
        init="kmeans",
        n_steps=80,
        exact_denominator=True,
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    algo.prepare(
        keys, values, problem.head_dim,
        queries=queries, query_positions=[qpos], seed=42,
    )
    out = algo.run(problem, 64, np.random.default_rng(0))
    assert out.output.shape == (problem.head_dim,)
    assert out.actual_budget == len(problem.special_idx) + 64
    assert np.isfinite(_rel_l2(out.output, full_out))


def test_nested_budget_monotone_validation():
    """Larger budget (nested) cannot do worse than the smaller on the probe set."""
    problem, keys, values, queries, qpos, sink, window = _make_example(seed=1)
    algo = LearnedCoreset(
        n_train_queries=80,
        init="kmeans",
        n_steps=120,
        nested_budget=True,
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    algo.prepare(
        keys, values, problem.head_dim,
        queries=queries, query_positions=[qpos], seed=7,
    )

    probes = algo._probe_queries
    ref = algo._ref_pos
    sp_idx, _ = compute_special_indices(ref + 1, sink, window)
    K = keys[: ref + 1]
    V = values[: ref + 1]
    scale = 1.0 / np.sqrt(keys.shape[1])

    def _probe_err(triple):
        kp, vp, wp = triple
        errs = []
        for q in probes:
            logits = scale * (K @ q)
            ml = logits.max()
            e = np.exp(logits - ml)
            z = e.sum()
            target = (e @ V) / z
            n_sp = e[sp_idx] @ V[sp_idx]
            e_pr = np.exp(np.clip(scale * (kp @ q) - ml, None, 40.0))
            pred = (n_sp + (e_pr * wp) @ vp) / z
            errs.append(_rel_l2(pred, target))
        return float(np.mean(errs))

    rng = np.random.default_rng(0)
    # Request budgets in ascending order (as the evaluator does).
    algo.run(problem, 16, rng)
    algo.run(problem, 32, rng)
    algo.run(problem, 64, rng)

    e16 = _probe_err(algo._learned_cache[16])
    e32 = _probe_err(algo._learned_cache[32])
    e64 = _probe_err(algo._learned_cache[64])
    assert algo._learned_cache[32][0].shape[0] == 32
    assert algo._learned_cache[64][0].shape[0] == 64
    # Nested: probe error non-increasing in budget (small slack for optimizer noise).
    assert e32 <= e16 * 1.05 + 1e-6
    assert e64 <= e32 * 1.05 + 1e-6


def test_training_forward_matches_weighted_attention():
    """Training forward must mirror weighted_attention for both denominator modes."""
    problem, keys, values, queries, qpos, sink, window = _make_example(seed=3)
    ref = qpos
    probes = build_probe_queries(queries, [qpos], ref, n_train_queries=32)
    budget = 16
    scale = 1.0 / np.sqrt(keys.shape[1])
    sp_idx, _ = compute_special_indices(ref + 1, sink, window)
    device = torch.device("cpu")

    keys_ref = torch.as_tensor(keys[: ref + 1], dtype=torch.float32, device=device)
    values_ref = torch.as_tensor(values[: ref + 1], dtype=torch.float32, device=device)
    probe_t = torch.as_tensor(probes, dtype=torch.float32, device=device)

    rng = np.random.default_rng(11)
    d = keys.shape[1]
    k_new = torch.as_tensor(
        rng.standard_normal((budget, d)).astype(np.float32), device=device,
    )
    v_new = torch.as_tensor(
        rng.standard_normal((budget, d)).astype(np.float32), device=device,
    )
    log_w = torch.log(torch.full((budget,), 0.5, device=device))

    consts = _precompute_targets(
        probe_t, keys_ref, values_ref, sp_idx, None, scale,
    )
    idx = torch.arange(probes.shape[0], device=device)
    vmin = values_ref.amin(dim=0, keepdim=True)
    vmax = values_ref.amax(dim=0, keepdim=True)

    sp_keys = keys_ref[sp_idx].unsqueeze(0)
    sp_vals = values_ref[sp_idx].unsqueeze(0)
    sp_one = torch.ones((1, len(sp_idx)), dtype=torch.float32, device=device)
    core_keys = torch.cat([sp_keys, k_new.unsqueeze(0)], dim=1)
    core_values = torch.cat([sp_vals, v_new.unsqueeze(0)], dim=1)
    core_one = torch.cat([sp_one, log_w.exp().unsqueeze(0)], dim=-1)

    for exact in (False, True):
        pred_train, _ = _forward_pred(
            consts, idx, k_new, v_new, log_w, scale, exact,
        )
        preds_eval = []
        for i in range(probes.shape[0]):
            q = probe_t[i].unsqueeze(0).unsqueeze(0)
            al = None
            if exact:
                al = torch.as_tensor(
                    scale * (keys_ref @ probe_t[i]),
                    dtype=torch.float32,
                    device=device,
                )
            out = weighted_attention(
                q,
                core_keys,
                core_values,
                core_one,
                scale,
                vmin.unsqueeze(0),
                vmax.unsqueeze(0),
                all_logits=al,
            )
            preds_eval.append(out.squeeze(0).squeeze(0))
        pred_eval = torch.stack(preds_eval, dim=0)
        assert torch.allclose(pred_train, pred_eval, atol=1e-5, rtol=1e-5), exact


def test_training_improves_over_init_estimated_denominator():
    """Estimated-denominator training should still beat the init on probes."""
    _, keys, values, queries, qpos, sink, window = _make_example(seed=1)
    ref = qpos
    probes = build_probe_queries(queries, [qpos], ref, n_train_queries=64)
    budget = 32
    common = dict(
        keys=keys, values=values, head_dim=keys.shape[1],
        probe_queries=probes, ref_pos=ref, budget=budget,
        n_sink=sink, local_window=window,
        init="kmeans", exact_denominator=False, seed=5,
    )
    k0, v0, w0 = learn_kv_coreset(**common, n_steps=0)
    k1, v1, w1 = learn_kv_coreset(**common, n_steps=300)

    sp_idx, _ = compute_special_indices(ref + 1, sink, window)
    scale = 1.0 / np.sqrt(keys.shape[1])
    device = torch.device("cpu")
    keys_ref = torch.as_tensor(keys[: ref + 1], dtype=torch.float32, device=device)
    values_ref = torch.as_tensor(values[: ref + 1], dtype=torch.float32, device=device)
    probe_t = torch.as_tensor(probes, dtype=torch.float32, device=device)
    consts = _precompute_targets(
        probe_t, keys_ref, values_ref, sp_idx, None, scale,
    )
    idx = torch.arange(probes.shape[0], device=device)

    def _probe_err(kp, vp, wp):
        log_w = torch.log(torch.as_tensor(wp, device=device))
        k = torch.as_tensor(kp, device=device)
        v = torch.as_tensor(vp, device=device)
        pred, target = _forward_pred(
            consts, idx, k, v, log_w, scale, exact_denominator=False,
        )
        sq = (pred - target).pow(2).sum(dim=-1)
        tnorm2 = target.pow(2).sum(dim=-1).clamp_min(1e-12)
        return float((sq / tnorm2).mean())

    assert _probe_err(k1, v1, w1) <= _probe_err(k0, v0, w0) + 1e-6


def test_training_improves_over_init():
    _, keys, values, queries, qpos, sink, window = _make_example(seed=1)
    ref = qpos
    probes = build_probe_queries(queries, [qpos], ref, n_train_queries=64)
    budget = 32
    common = dict(
        keys=keys, values=values, head_dim=keys.shape[1],
        probe_queries=probes, ref_pos=ref, budget=budget,
        n_sink=sink, local_window=window,
        init="kmeans", exact_denominator=True, seed=5,
    )
    k0, v0, w0 = learn_kv_coreset(**common, n_steps=0)
    k1, v1, w1 = learn_kv_coreset(**common, n_steps=300)

    sp_idx, _ = compute_special_indices(ref + 1, sink, window)
    K = keys[: ref + 1]
    V = values[: ref + 1]
    scale = 1.0 / np.sqrt(keys.shape[1])

    def _probe_err(kp, vp, wp):
        errs = []
        for q in probes:
            logits = scale * (K @ q)
            ml = logits.max()
            e = np.exp(logits - ml)
            z = e.sum()
            target = (e @ V) / z
            n_sp = e[sp_idx] @ V[sp_idx]
            e_pr = np.exp(np.clip(scale * (kp @ q) - ml, None, 40.0))
            pred = (n_sp + (e_pr * wp) @ vp) / z
            errs.append(_rel_l2(pred, target))
        return float(np.mean(errs))

    assert _probe_err(k1, v1, w1) <= _probe_err(k0, v0, w0) + 1e-6


def test_determinism():
    problem, keys, values, queries, qpos, sink, window = _make_example(seed=4)

    def _make_algo():
        algo = LearnedCoreset(
            n_train_queries=40, init="random", n_steps=60,
            n_sink=sink, local_window=window, device="cpu",
        )
        algo.prepare(
            keys, values, problem.head_dim,
            queries=queries, query_positions=[qpos], seed=99,
        )
        return algo

    o1 = _make_algo().run(problem, 32, np.random.default_rng(0))
    o2 = _make_algo().run(problem, 32, np.random.default_rng(0))
    assert np.allclose(o1.output, o2.output, atol=1e-5)


if __name__ == "__main__":
    test_train_excludes_test_queries()
    test_probe_queries_use_latest_window_not_prefix_start()
    test_reference_position()
    test_no_random_queries()
    test_runs_and_eval_includes_special()
    test_training_forward_matches_weighted_attention()
    test_nested_budget_monotone_validation()
    test_training_improves_over_init()
    test_training_improves_over_init_estimated_denominator()
    test_determinism()
    print("all passed")
