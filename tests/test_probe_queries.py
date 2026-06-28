"""Tests for shared training-query construction."""

import numpy as np

from src.algorithms.kvsculpt import KVSculpt
from src.algorithms.learned import LearnedCoreset
from src.algorithms.probe_queries import (
    build_kvsculpt_train_queries,
    build_probe_queries,
    build_train_queries,
    prepare_probe_queries,
)
from src.core import compute_special_indices


def test_build_train_queries_trailing_matches_probe():
    q = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    trailing = build_train_queries(
        "trailing", q, [99], 99, n_train_queries=10,
    )
    expected = build_probe_queries(q, [99], 99, 10)
    assert np.array_equal(trailing, expected)


def test_build_train_queries_kvsculpt_includes_retain_and_synthetic():
    n, d, sink, window = 200, 16, 1, 32
    rng = np.random.default_rng(0)
    queries = rng.standard_normal((n, d)).astype(np.float32)
    ref = n - 1
    sp_idx, _ = compute_special_indices(ref + 1, sink, window)
    train_q = build_train_queries(
        "kvsculpt",
        queries,
        [ref],
        ref,
        special_idx=sp_idx,
        n_synthetic=8,
        head_dim=d,
        seed=1,
    )
    assert train_q.shape[0] == len(sp_idx) + 8


def test_prepare_probe_queries_shared_by_learned_and_kvsculpt():
    n, d, sink, window = 300, 24, 1, 48
    rng = np.random.default_rng(2)
    keys = rng.standard_normal((n, d)).astype(np.float32)
    values = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((n, d)).astype(np.float32)
    ref = n - 1

    learned = LearnedCoreset(
        n_train_queries=20,
        train_q_strategy="kvsculpt",
        n_synthetic=6,
        n_steps=0,
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    kv = KVSculpt(
        n_synthetic=6,
        n_k_steps=0,
        train_q_strategy="kvsculpt",
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    learned.prepare(
        keys, values, d, queries=queries, query_positions=[ref], seed=3,
    )
    kv.prepare(
        keys, values, d, queries=queries, query_positions=[ref], seed=3,
    )
    assert np.array_equal(learned._probe_queries, kv._probe_queries)


def test_kvsculpt_trailing_strategy_uses_trailing_window():
    n, d, sink, window = 400, 16, 1, 32
    rng = np.random.default_rng(4)
    keys = rng.standard_normal((n, d)).astype(np.float32)
    values = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((n, d)).astype(np.float32)
    ref = n - 1

    ref_pos, trailing_q = prepare_probe_queries(
        queries, [ref], d, sink, window,
        "trailing", n_train_queries=15, n_synthetic=8, rope_theta=500_000.0, seed=0,
    )
    assert ref_pos == ref
    assert trailing_q.shape[0] == 14  # 15-window minus held-out test position

    _, kvsculpt_q = prepare_probe_queries(
        queries, [ref], d, sink, window,
        "kvsculpt", n_train_queries=15, n_synthetic=8, rope_theta=500_000.0, seed=0,
    )
    sp_idx, _ = compute_special_indices(ref + 1, sink, window)
    expected = build_kvsculpt_train_queries(
        queries, ref, sp_idx, 8, head_dim=d, seed=0,
    )
    assert np.array_equal(kvsculpt_q, expected)
    assert kvsculpt_q.shape[0] != trailing_q.shape[0]
