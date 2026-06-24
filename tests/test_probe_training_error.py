"""Tests for probe-set training error evaluation."""

import numpy as np

from src.algorithms.learned import LearnedCoreset
from src.algorithms.tensor_fcfw_lq import TensorFCFWLq
from src.evaluation.evaluator import (
    evaluate_probe_set_errors,
    is_probe_q_method,
)


def _make_example(n=400, d=32, sink=2, window=32, seed=0):
    rng = np.random.default_rng(seed)
    basis_k = rng.standard_normal((6, d)).astype(np.float32)
    basis_v = rng.standard_normal((6, d)).astype(np.float32)
    coeff = rng.standard_normal((n, 6)).astype(np.float32)
    keys = (coeff @ basis_k + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
    values = (coeff @ basis_v + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
    queries = (rng.standard_normal((n, 6)).astype(np.float32) @ basis_k).astype(np.float32)
    return keys, values, queries, n - 1, sink, window


def test_is_probe_q_method():
    learned = LearnedCoreset(n_train_queries=16, n_steps=0, device="cpu")
    lq = TensorFCFWLq(n_train_queries=16, device="cpu")
    assert is_probe_q_method(learned)
    assert is_probe_q_method(lq)


def test_evaluate_probe_set_errors_learned():
    keys, values, queries, qpos, sink, window = _make_example(seed=1)
    algo = LearnedCoreset(
        n_train_queries=24,
        init="random",
        n_steps=40,
        exact_denominator=False,
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    algo.prepare(
        keys, values, keys.shape[1],
        queries=queries, query_positions=[qpos], seed=3,
    )
    out = evaluate_probe_set_errors(
        algo, keys, values, keys.shape[1],
        budgets=[8, 16],
        n_sink=sink,
        local_window=window,
        rng=np.random.default_rng(0),
    )
    assert "Learned-random-8" in out
    assert "Learned-random-16" in out
    assert out["Learned-random-8"]["n_probes"] == 24
    assert np.isfinite(out["Learned-random-8"]["error"])


def test_evaluate_probe_set_errors_lq():
    keys, values, queries, qpos, sink, window = _make_example(seed=2)
    algo = TensorFCFWLq(
        n_train_queries=20,
        exact_denominator=False,
        n_sink=sink,
        local_window=window,
        device="cpu",
    )
    algo.prepare(
        keys, values, keys.shape[1],
        queries=queries, query_positions=[qpos], seed=4,
    )
    out = evaluate_probe_set_errors(
        algo, keys, values, keys.shape[1],
        budgets=[8],
        n_sink=sink,
        local_window=window,
        rng=np.random.default_rng(0),
    )
    assert "TFCFW-lq-8" in out
    assert out["TFCFW-lq-8"]["n_probes"] == 20
