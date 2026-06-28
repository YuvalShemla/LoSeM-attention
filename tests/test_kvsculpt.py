"""Tests for KVSculpt distillation (arXiv:2603.27819)."""

import numpy as np
import torch

from src.algorithms.base import AttentionInput
from src.algorithms.kvsculpt import KVSculpt
from src.algorithms.probe_queries import (
    apply_rope,
    build_kvsculpt_train_queries as build_training_queries,
    inverse_rope,
)
from src.algorithms.kvsculpt.kvsculpt_distill import distill_kv_cache
from src.core import compute_special_indices, full_attention


def _make_example(n=800, d=48, sink=2, window=64, seed=0):
    rng = np.random.default_rng(seed)
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


def test_rope_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 32)).astype(np.float32)
    pos = np.array([0, 10, 50, 100, 200], dtype=np.float64)
    y = apply_rope(x, pos, head_dim=32, rope_theta=10_000.0)
    x_back = inverse_rope(y, pos, head_dim=32, rope_theta=10_000.0)
    assert np.allclose(x, x_back, atol=1e-5)


def test_training_queries_include_retain_and_synthetic():
    _, _, _, queries, qpos, sink, window = _make_example()
    sp_idx, _ = compute_special_indices(qpos + 1, sink, window)
    train_q = build_training_queries(
        queries, qpos, sp_idx, n_synthetic=8,
        head_dim=queries.shape[1], seed=1,
    )
    assert train_q.shape[0] == len(sp_idx) + 8


def test_distill_reduces_probe_error_vs_topk_init():
    problem, keys, values, queries, qpos, sink, window = _make_example(seed=2)
    sp_idx, _ = compute_special_indices(qpos + 1, sink, window)
    train_q = build_training_queries(
        queries, qpos, sp_idx, n_synthetic=16,
        head_dim=problem.head_dim, seed=3,
    )
    budget = 24
    k_c, v_c = distill_kv_cache(
        keys, values, problem.head_dim, train_q, qpos, budget,
        sink, window, n_k_steps=25, device=torch.device("cpu"), seed=4,
    )
    assert k_c.shape == (budget, problem.head_dim)
    assert v_c.shape == (budget, problem.head_dim)

    scale = 1.0 / np.sqrt(problem.head_dim)
    errs = []
    for q in train_q[:16]:
        full_out, _, _ = full_attention(q, problem.keys, problem.values, problem.head_dim)
        k_cat = np.concatenate([problem.keys[sp_idx], k_c], axis=0)
        v_cat = np.concatenate([problem.values[sp_idx], v_c], axis=0)
        logits = scale * (q @ k_cat.T)
        approx = (np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()) @ v_cat
        errs.append(_rel_l2(approx, full_out))
    assert np.mean(errs) < 0.5


def test_kvsculpt_run_interface():
    problem, keys, values, queries, qpos, sink, window = _make_example(seed=5)
    algo = KVSculpt(
        n_synthetic=8,
        n_k_steps=15,
        n_train_queries=32,
        exact_denominator=True,
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    algo.prepare(
        keys, values, problem.head_dim,
        queries=queries, query_positions=[qpos], seed=6,
    )
    out = algo.run(problem, budget=16, rng=np.random.default_rng(0))
    assert out.output.shape == (problem.head_dim,)
    assert out.actual_budget == len(problem.special_idx) + 16

    full_out, _, _ = full_attention(
        problem.query, problem.keys, problem.values, problem.head_dim,
    )
    assert _rel_l2(out.output, full_out) < 1.0


def test_expand_from_config_defaults():
    instances = KVSculpt.expand_from_config({})
    assert len(instances) == 1
    assert instances[0].name == "KVSculpt"
