"""
TopK vs sampling bias analysis.

Compares TopK truncation, uniform sampling, and oracle
sampling at various budgets. Measures both weight-space
error (L1 on softmax vector) and value-space error
(L2 on aggregated output).
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


def compute_bias_data(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    budget_fractions: List[float] = None,
    seed: int = 42,
) -> Dict:
    """
    Compare TopK, Uniform, Oracle at each budget.

    Returns per-budget mean/std of weight and value
    errors across queries.
    """
    if budget_fractions is None:
        budget_fractions = [
            0.03, 0.05, 0.1, 0.2, 0.5,
        ]
    rng = np.random.default_rng(seed)

    results = {
        "budget_fractions": budget_fractions,
        "topk_value_error": [],
        "uniform_value_error": [],
        "oracle_sampling_value_error": [],
        "topk_value_std": [],
        "uniform_value_std": [],
        "oracle_sampling_value_std": [],
    }

    for frac in budget_fractions:
        tk_errs, un_errs, or_errs = [], [], []
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

            b = max(1, int(n * frac))

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

            # Oracle
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

    return results


def plot_bias_comparison(
    data: Dict,
    out_path: Path,
    title: str = "",
):
    """
    Plot TopK vs Uniform vs Oracle value errors.

    Budget (% of keys) on x-axis, relative L2 error
    on y-axis, with error bands.
    """
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    fracs = [f * 100 for f in data["budget_fractions"]]

    for method, color, marker in [
        ("topk", "#d62728", "o"),
        ("uniform", "#2ca02c", "^"),
        ("oracle_sampling", "#1f77b4", "s"),
    ]:
        y = data[f"{method}_value_error"]
        s = data[f"{method}_value_std"]
        ax.plot(
            fracs, y, color=color, marker=marker,
            lw=2, ms=7,
            label=method.replace("_", " ").title(),
        )
        y_arr = np.array(y)
        s_arr = np.array(s)
        ax.fill_between(
            fracs,
            np.maximum(y_arr - s_arr, 1e-10),
            y_arr + s_arr,
            color=color, alpha=0.15,
        )

    ax.set_xlabel(
        "Budget (% of keys)", fontsize=12,
    )
    ax.set_ylabel(
        "Relative L2 Error", fontsize=12,
    )
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")

    if title:
        ax.set_title(
            title, fontsize=13, fontweight="bold",
        )
    plt.tight_layout()
    save_figure(fig, out_path)
