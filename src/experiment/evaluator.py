"""
Per-query evaluation and result aggregation.

Separated from run_experiment.py to keep each file focused.
"""

import numpy as np
from typing import Dict, List

from ..algorithms.base import AttentionInput
from ..core import (
    full_attention, compute_special_indices,
    relative_l2_error, stats_from_weights,
)


def evaluate_query(
    q: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    methods: list,
    budgets: List[int],
    head_dim: int,
    n_sink: int,
    local_window: int,
    rng: np.random.Generator,
    compute_statistics: bool = False,
) -> Dict:
    """Evaluate one query across all methods."""
    n_causal = len(keys)

    full_out, logits, weights = full_attention(
        q, keys, values, head_dim,
    )

    sp_idx, cand_idx = compute_special_indices(
        n_causal, n_sink, local_window,
    )

    problem = AttentionInput(
        query=q, keys=keys, values=values,
        head_dim=head_dim, logits=logits,
        special_idx=sp_idx, candidate_idx=cand_idx,
    )

    results = {}
    for m in methods:
        if m.sweeps_budget:
            for b in budgets:
                out = m.run(problem, b, rng)
                err = relative_l2_error(
                    out.output, full_out,
                )
                k = f"{m.name}-{b}"
                results[k] = {
                    "error": err,
                    "budget": out.actual_budget,
                }
        else:
            out = m.run(problem, 0, rng)
            err = relative_l2_error(
                out.output, full_out,
            )
            results[m.name] = {
                "error": err,
                "budget": out.actual_budget,
            }

    if compute_statistics:
        results["_query_stats"] = stats_from_weights(
            weights, n_sink, local_window,
        )

    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Compute mean/std per method key."""
    all_keys = set()
    for qr in all_results:
        all_keys.update(qr.keys())

    agg = {}
    for key in sorted(all_keys):
        if key == "_query_stats":
            continue
        entries = [
            qr[key] for qr in all_results
            if key in qr
        ]
        if not entries:
            continue
        errors = [e["error"] for e in entries]
        budgets = [e["budget"] for e in entries]
        agg[key] = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "budget_mean": float(np.mean(budgets)),
            "budget_std": float(np.std(budgets)),
            "n_queries": len(entries),
        }
    return agg


def aggregate_query_stats(
    all_results: List[Dict],
) -> Dict[str, float]:
    """
    Aggregate per-query attention statistics
    into mean/std summaries.

    Returns empty dict if no stats were computed.
    """
    stat_entries = [
        qr["_query_stats"] for qr in all_results
        if "_query_stats" in qr
    ]
    if not stat_entries:
        return {}

    accum = {}
    for entry in stat_entries:
        for k, v in entry.items():
            accum.setdefault(k, []).append(v)

    agg = {}
    for k, vals in accum.items():
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg
