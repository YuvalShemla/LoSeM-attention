"""
Per-query evaluation and result aggregation.

Separated from run_evaluation.py to keep each file focused.
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


PERCENTILE_WEIGHTS = {0: 1, 25: 2, 50: 3, 75: 2, 100: 1}


def weighted_aggregate_heads(
    per_head_aggs: Dict[int, Dict],
    head_meta: list,
) -> Dict:
    """
    Weighted aggregate across heads by percentile.

    Triangular weighting: p50 gets 3x the weight of
    p0/p100. Falls back to equal weights if percentile
    metadata is missing.
    """
    weights = []
    aggs = []
    for idx, info in per_head_aggs.items():
        meta = head_meta[idx] if head_meta else {}
        pct = meta.get("percentile")
        w = PERCENTILE_WEIGHTS.get(pct, 1)
        weights.append(w)
        aggs.append(info["agg"])

    all_keys = set()
    for a in aggs:
        all_keys.update(a.keys())

    total_w = sum(weights)
    result = {}
    for key in sorted(all_keys):
        present = [
            (a[key], w)
            for a, w in zip(aggs, weights)
            if key in a
        ]
        if not present:
            continue
        w_sum = sum(w for _, w in present)
        err_mean = sum(
            e["error_mean"] * w for e, w in present
        ) / w_sum
        bud_mean = sum(
            e["budget_mean"] * w for e, w in present
        ) / w_sum
        err_std = sum(
            e["error_std"] * w for e, w in present
        ) / w_sum
        bud_std = sum(
            e["budget_std"] * w for e, w in present
        ) / w_sum
        n_total = sum(
            e["n_queries"] for e, _ in present
        )
        result[key] = {
            "error_mean": float(err_mean),
            "error_std": float(err_std),
            "budget_mean": float(bud_mean),
            "budget_std": float(bud_std),
            "n_queries": n_total,
            "weighting": "percentile_triangular",
        }
    return result


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
