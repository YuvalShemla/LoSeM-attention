"""
Approximation methods analysis.

Compares TopK truncation, uniform sampling, distribution
sampling, ideal equal splits and ideal equal weight
splits at various absolute budgets. Measures value-space
error (L2 on aggregated output).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from ..core import softmax
from ..experiment.plotting import (
    setup_style, save_figure,
)

_DEFAULT_BUDGETS = [
    16, 32, 64, 128, 512,
    1024, 2048, 4096, 8192, 16384,
]


def _equal_weight_groups(
    sorted_idx: np.ndarray,
    sorted_weights: np.ndarray,
    num_groups: int,
) -> list:
    """
    Split so each group captures ~equal total weight mass.
    """
    n = len(sorted_idx)
    num_groups = min(num_groups, n)
    if num_groups >= n:
        return [sorted_idx[i:i + 1] for i in range(n)]

    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    if total < 1e-12:
        group_size = max(1, n // num_groups)
        groups = []
        for i in range(num_groups):
            start = i * group_size
            end = (
                (i + 1) * group_size
                if i < num_groups - 1
                else n
            )
            if start >= n:
                break
            groups.append(sorted_idx[start:end])
        return groups

    targets = np.linspace(
        0, total, num_groups + 1,
    )[1:-1]
    split_indices = np.searchsorted(cumsum, targets)
    split_indices = np.unique(
        np.clip(split_indices, 1, n - 1)
    )

    groups = []
    prev = 0
    for sp in split_indices:
        groups.append(sorted_idx[prev:sp])
        prev = sp
    groups.append(sorted_idx[prev:])

    return groups


def _ideal_grouping_error(
    q, keys, vals, head_dim, true_out, t_norm,
    groups,
):
    """Compute error for a set of groups."""
    sqrt_d = np.sqrt(head_dim)
    n_groups = len(groups)
    group_scores = np.empty(n_groups)
    group_vals = np.empty((n_groups, head_dim))
    for gi, g_idx in enumerate(groups):
        count = len(g_idx)
        avg_k = np.mean(keys[g_idx], axis=0)
        avg_v = np.mean(vals[g_idx], axis=0)
        group_scores[gi] = (
            q @ avg_k / sqrt_d + np.log(count)
        )
        group_vals[gi] = avg_v

    w = softmax(group_scores)
    out = w @ group_vals
    return np.linalg.norm(out - true_out) / t_norm


def compute_bias_data(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    budgets: List[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Compare TopK, Uniform, IdealSampling,
    IdealEqualSplits, IdealEqualWeightSplits.

    budgets: absolute number of keys to use at each point.
    Budgets exceeding the context length for a query are
    skipped for that query.

    Returns per-budget mean/std of value errors across
    queries.
    """
    if budgets is None:
        budgets = list(_DEFAULT_BUDGETS)
    rng = np.random.default_rng(seed)

    results = {
        "budgets": budgets,
        "topk_value_error": [],
        "uniform_value_error": [],
        "ideal_sampling_value_error": [],
        "ideal_equal_splits_value_error": [],
        "ideal_equal_weight_splits_value_error": [],
        "topk_value_std": [],
        "uniform_value_std": [],
        "ideal_sampling_value_std": [],
        "ideal_equal_splits_value_std": [],
        "ideal_equal_weight_splits_value_std": [],
    }

    for budget in budgets:
        tk_errs, un_errs, or_errs = [], [], []
        es_errs, ew_errs = [], []
        for qpos in query_positions:
            q = Q[qpos]
            keys = K[:qpos + 1]
            vals = V[:qpos + 1]
            n = len(keys)
            if budget >= n:
                continue
            logits = (q @ keys.T) / np.sqrt(head_dim)
            weights = softmax(logits)
            true_out = weights @ vals
            t_norm = np.linalg.norm(true_out)
            if t_norm < 1e-10:
                continue

            b = budget

            # TopK
            top_idx = np.argpartition(
                logits, -b
            )[-b:]
            w_tk = softmax(logits[top_idx])
            out_tk = w_tk @ vals[top_idx]
            tk_errs.append(
                np.linalg.norm(out_tk - true_out)
                / t_norm
            )

            # Uniform
            u_idx = rng.choice(
                n, size=b, replace=False,
            )
            w_un = softmax(logits[u_idx])
            out_un = w_un @ vals[u_idx]
            un_errs.append(
                np.linalg.norm(out_un - true_out)
                / t_norm
            )

            # IdealSampling (distribution sampling)
            o_idx = rng.choice(
                n, size=b, p=weights, replace=True,
            )
            o_idx = np.unique(o_idx)
            w_or = softmax(logits[o_idx])
            out_or = w_or @ vals[o_idx]
            or_errs.append(
                np.linalg.norm(out_or - true_out)
                / t_norm
            )

            # IdealEqualSplits
            sort_order = np.argsort(logits)[::-1]
            sorted_idx = np.arange(n)[sort_order]
            num_groups = min(b, n)
            group_size = max(1, n // num_groups)
            groups = []
            for i in range(num_groups):
                start = i * group_size
                end = (
                    (i + 1) * group_size
                    if i < num_groups - 1
                    else n
                )
                if start >= n:
                    break
                groups.append(sorted_idx[start:end])
            es_errs.append(
                _ideal_grouping_error(
                    q, keys, vals, head_dim,
                    true_out, t_norm, groups,
                )
            )

            # IdealEqualWeightSplits
            w_sort = np.argsort(weights)[::-1]
            sorted_idx_w = np.arange(n)[w_sort]
            sorted_weights = weights[w_sort]
            groups_w = _equal_weight_groups(
                sorted_idx_w, sorted_weights, b,
            )
            ew_errs.append(
                _ideal_grouping_error(
                    q, keys, vals, head_dim,
                    true_out, t_norm, groups_w,
                )
            )

        for key, errs in [
            ("topk", tk_errs),
            ("uniform", un_errs),
            ("ideal_sampling", or_errs),
            ("ideal_equal_splits", es_errs),
            ("ideal_equal_weight_splits", ew_errs),
        ]:
            results[f"{key}_value_error"].append(
                float(np.mean(errs)) if errs else 0.0
            )
            results[f"{key}_value_std"].append(
                float(np.std(errs)) if errs else 0.0
            )

    return results


def plot_bias_comparison(
    data: Dict,
    out_path: Path,
    title: str = "",
):
    """
    Plot approximation methods value errors.

    Number of keys on x-axis, relative L2 error
    on y-axis, with error bands.
    """
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    budgets = data["budgets"]

    for method, color, marker in [
        ("topk", "#d62728", "o"),
        ("uniform", "#ff7f0e", "^"),
        ("ideal_sampling", "#2ca02c", "s"),
        ("ideal_equal_splits", "#1f77b4", "D"),
        ("ideal_equal_weight_splits", "#9467bd", "X"),
    ]:
        y = data.get(f"{method}_value_error", [])
        s = data.get(f"{method}_value_std", [])
        if not y:
            continue
        label = method.replace("_", " ").title()
        ax.plot(
            budgets, y, color=color, marker=marker,
            lw=2, ms=7, label=label,
        )
        y_arr = np.array(y)
        s_arr = np.array(s)
        ax.fill_between(
            budgets,
            np.maximum(y_arr - s_arr, 1e-10),
            y_arr + s_arr,
            color=color, alpha=0.15,
        )

    ax.set_xlabel(
        "Number of Keys", fontsize=12,
    )
    ax.set_ylabel(
        "Relative L2 Error", fontsize=12,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")

    if title:
        ax.set_title(
            title, fontsize=13, fontweight="bold",
        )
    plt.tight_layout()
    save_figure(fig, out_path)
