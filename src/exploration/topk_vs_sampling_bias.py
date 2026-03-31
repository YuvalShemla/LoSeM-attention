"""
Approximation methods analysis.

Compares TopK truncation, uniform sampling, distribution
sampling, and oracle grouping at various absolute budgets.
Measures value-space error (L2 on aggregated output).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from ..core import softmax, make_doubling_boundaries
from ..experiment.plotting import (
    setup_style, save_figure,
)

_DEFAULT_BUDGETS = [
    16, 32, 64, 128, 512,
    1024, 2048, 4096, 8192, 16384,
]


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
    Compare TopK, Uniform, Oracle Sampling, Oracle Grouping.

    budgets: absolute number of keys to use at each point.
    Budgets exceeding the context length for a query are
    skipped for that query.

    Returns per-budget mean/std of value errors across
    queries, plus oracle grouping (fixed budget).
    """
    if budgets is None:
        budgets = list(_DEFAULT_BUDGETS)
    rng = np.random.default_rng(seed)

    results = {
        "budgets": budgets,
        "topk_value_error": [],
        "uniform_value_error": [],
        "oracle_sampling_value_error": [],
        "topk_value_std": [],
        "uniform_value_std": [],
        "oracle_sampling_value_std": [],
    }

    # Oracle grouping: fixed budget (~log2(N) groups)
    og_errs = []

    for budget in budgets:
        tk_errs, un_errs, or_errs = [], [], []
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

            # Oracle (distribution sampling)
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

        for key, errs in [
            ("topk", tk_errs),
            ("uniform", un_errs),
            ("oracle_sampling", or_errs),
        ]:
            results[f"{key}_value_error"].append(
                float(np.mean(errs)) if errs else 0.0
            )
            results[f"{key}_value_std"].append(
                float(np.std(errs)) if errs else 0.0
            )

    # Oracle grouping (fixed budget, computed once)
    for qpos in query_positions:
        q = Q[qpos]
        keys = K[:qpos + 1]
        vals = V[:qpos + 1]
        n = len(keys)
        logits = (q @ keys.T) / np.sqrt(head_dim)
        weights = softmax(logits)
        true_out = weights @ vals
        t_norm = np.linalg.norm(true_out)
        if t_norm < 1e-10:
            continue

        sorted_idx = np.argsort(logits)[::-1]
        boundaries = make_doubling_boundaries(n)
        n_groups = len(boundaries)
        sqrt_d = np.sqrt(head_dim)

        group_scores = np.empty(n_groups)
        group_vals = np.empty((n_groups, head_dim))
        for gi, (start, end) in enumerate(boundaries):
            g_idx = sorted_idx[start:end]
            count = len(g_idx)
            avg_k = np.mean(keys[g_idx], axis=0)
            avg_v = np.mean(vals[g_idx], axis=0)
            group_scores[gi] = (
                q @ avg_k / sqrt_d + np.log(count)
            )
            group_vals[gi] = avg_v

        w_og = softmax(group_scores)
        out_og = w_og @ group_vals
        og_errs.append(
            np.linalg.norm(out_og - true_out) / t_norm
        )

    results["oracle_grouping_value_error"] = (
        float(np.mean(og_errs)) if og_errs else 0.0
    )
    results["oracle_grouping_value_std"] = (
        float(np.std(og_errs)) if og_errs else 0.0
    )
    # Store absolute budget for oracle grouping
    if query_positions:
        n_last = query_positions[-1] + 1
        n_groups = len(make_doubling_boundaries(n_last))
        results["oracle_grouping_budget"] = n_groups
    else:
        results["oracle_grouping_budget"] = 0

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
        ("oracle_sampling", "#2ca02c", "s"),
    ]:
        y = data[f"{method}_value_error"]
        s = data[f"{method}_value_std"]
        ax.plot(
            budgets, y, color=color, marker=marker,
            lw=2, ms=7,
            label=method.replace("_", " ").title(),
        )
        y_arr = np.array(y)
        s_arr = np.array(s)
        ax.fill_between(
            budgets,
            np.maximum(y_arr - s_arr, 1e-10),
            y_arr + s_arr,
            color=color, alpha=0.15,
        )

    # Oracle grouping
    og_err = data.get("oracle_grouping_value_error")
    og_budget = data.get("oracle_grouping_budget")
    if og_err is not None and og_budget:
        ax.plot(
            og_budget, og_err, color="#1f77b4",
            marker="D", ms=9, zorder=5,
            label="Oracle Grouping",
        )
        ax.axhline(
            og_err, color="#1f77b4", ls="--",
            lw=1, alpha=0.5,
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
