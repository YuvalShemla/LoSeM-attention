"""Tests for evaluation timing aggregation."""

import time

import numpy as np
import pytest

from src.algorithms.base import AttentionAlgorithm, AttentionInput, AttentionOutput
from src.core import compute_special_indices, full_attention
from src.evaluation.evaluator import (
    aggregate_fit_timings,
    aggregate_inference_timings,
    aggregate_prepare_timings,
    aggregate_probe_eval_timings,
    aggregate_results,
    evaluate_query,
    format_timing_summary,
)


class _SlowMethod(AttentionAlgorithm):
    def __init__(self, name: str, delay: float):
        self._name = name
        self._delay = delay

    @property
    def name(self) -> str:
        return self._name

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim, **kwargs) -> None:
        self.reset_method_timing()
        time.sleep(self._delay * 0.1)

    def run(self, problem, budget, rng) -> AttentionOutput:
        if budget not in getattr(self, "_fit_done", set()):
            self._fit_done = getattr(self, "_fit_done", set()) | {budget}
            time.sleep(self._delay)
            self.record_coreset_fit(budget, self._delay)
        t0 = time.perf_counter()
        out, _, _ = full_attention(
            problem.query, problem.keys, problem.values, problem.head_dim,
        )
        self.record_inference_seconds(time.perf_counter() - t0)
        return AttentionOutput(output=out, actual_budget=budget)


def test_evaluate_query_records_run_seconds():
    rng = np.random.default_rng(0)
    d, n = 8, 32
    q = rng.standard_normal(d)
    keys = rng.standard_normal((n, d))
    values = rng.standard_normal((n, d))
    methods = [_SlowMethod("fast", 0.001), _SlowMethod("slow", 0.010)]
    budgets = [4, 8]

    qr = evaluate_query(
        q, keys, values, methods, budgets, d, 1, 4, rng,
        measure_timing=True,
    )

    assert "fast-4" in qr and "slow-8" in qr
    assert qr["fast-4"]["run_seconds"] >= 0.001
    assert qr["slow-8"]["run_seconds"] >= 0.010
    assert "run_seconds" not in qr.get("_query_stats", {})


def test_aggregate_results_includes_run_timing():
    results = [
        {"m-4": {"error": 0.1, "budget": 4, "run_seconds": 0.01}},
        {"m-4": {"error": 0.2, "budget": 4, "run_seconds": 0.03}},
    ]
    agg = aggregate_results(results)
    assert agg["m-4"]["run_seconds_mean"] == pytest.approx(0.02)
    assert agg["m-4"]["run_seconds_total"] == pytest.approx(0.04)


def test_aggregate_prepare_timings():
    records = [
        {"method": "A", "prepare_seconds": 1.0},
        {"method": "A", "prepare_seconds": 3.0},
        {"method": "B", "prepare_seconds": 2.0},
    ]
    agg = aggregate_prepare_timings(records)
    assert agg["A"]["prepare_seconds_mean"] == pytest.approx(2.0)
    assert agg["A"]["n_examples"] == 2
    assert agg["B"]["prepare_seconds_mean"] == pytest.approx(2.0)


def test_format_timing_summary_ms_for_fast_runs():
    text = format_timing_summary(
        {"KVSculpt": {
            "prepare_seconds_mean": 0.006,
            "prepare_seconds_std": 0.001,
            "prepare_seconds_total": 0.03,
            "n_examples": 5,
        }},
        {"KVSculpt-1024": {
            "run_seconds_mean": 0.002,
            "run_seconds_std": 0.0005,
            "n_queries": 10,
        }},
        fit_agg={"KVSculpt-1024": {
            "fit_seconds_mean": 45.0,
            "fit_seconds_std": 2.0,
            "fit_seconds_total": 225.0,
            "n_examples": 5,
        }},
        probe_eval_agg={"KVSculpt": {
            "probe_eval_seconds_mean": 600.0,
            "probe_eval_seconds_std": 10.0,
            "probe_eval_seconds_total": 3000.0,
            "n_examples": 5,
        }},
        algorithm_only=False,
    )
    assert "Coreset fit" in text
    assert "Probe-set eval" in text
    assert "KVSculpt" in text


def test_aggregate_fit_and_probe_timings():
    fit = aggregate_fit_timings([
        {"method": "KVSculpt", "budget": 1024, "fit_seconds": 10.0},
        {"method": "KVSculpt", "budget": 1024, "fit_seconds": 20.0},
    ])
    assert fit["KVSculpt-1024"]["fit_seconds_mean"] == pytest.approx(15.0)

    probe = aggregate_probe_eval_timings([
        {"method": "KVSculpt", "probe_eval_seconds": 100.0},
        {"method": "KVSculpt", "probe_eval_seconds": 200.0},
    ])
    assert probe["KVSculpt"]["probe_eval_seconds_total"] == pytest.approx(300.0)

    inf = aggregate_inference_timings([
        {"method": "A", "inference_seconds": 1.0, "inference_calls": 10},
        {"method": "A", "inference_seconds": 3.0, "inference_calls": 30},
    ])
    assert inf["A"]["inference_seconds_total"] == pytest.approx(4.0)
    assert inf["A"]["inference_calls"] == 40
