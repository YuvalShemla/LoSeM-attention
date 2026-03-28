"""
Attention concentration analysis.

For each query position, measures how much attention
mass is captured by the top X% of keys. Produces
concentration curves and top-K mass vs position plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from ..core import (
    softmax, top_k_mass, no_sink_local_mask,
    concentration_curve,
)
from ..experiment.plotting import (
    setup_style, save_figure,
)


def compute_concentration_data(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    top_k_values: List[int] = None,
) -> Dict:
    """
    Compute concentration stats for multiple queries.

    Returns dict with per-query top-K mass and
    concentration curves.
    """
    if top_k_values is None:
        top_k_values = [10, 50, 100, 200, 500]

    results = {
        "query_positions": query_positions,
        "top_k_values": top_k_values,
        "top_k_mass": {
            k: [] for k in top_k_values
        },
        "concentration_curves": [],
    }

    for qpos in query_positions:
        q = Q[qpos]
        keys = K[:qpos + 1]
        logits = (q @ keys.T) / np.sqrt(head_dim)
        weights = softmax(logits)

        for k in top_k_values:
            results["top_k_mass"][k].append(
                top_k_mass(weights, k)
            )
        results["concentration_curves"].append(
            concentration_curve(weights)
        )

    return results


def plot_concentration(
    data: Dict,
    out_path: Path,
    title: str = "",
):
    """
    Plot concentration curves and top-K mass.

    Generates a 2-panel figure: top-K mass vs
    query position (left) and aggregate
    concentration curve (right).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6),
    )

    qpos = data["query_positions"]
    for k, masses in data["top_k_mass"].items():
        ax1.plot(
            qpos, masses, label=f"Top-{k}",
            lw=1.5,
        )
    ax1.set_xlabel("Query Position")
    ax1.set_ylabel("Attention Mass")
    ax1.set_title("Top-K Mass vs Query Position")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Aggregate concentration curve
    curves = data["concentration_curves"]
    if curves:
        pcts = sorted(
            k for k in curves[0].keys()
        )
        x_vals = [
            float(p.split("_")[1].replace("pct", ""))
            for p in pcts
        ]
        y_mean = [
            np.mean([c[p] for c in curves])
            for p in pcts
        ]
        y_std = [
            np.std([c[p] for c in curves])
            for p in pcts
        ]

        ax2.plot(
            x_vals, y_mean, "b-", lw=2,
            label="Mean",
        )
        ax2.fill_between(
            x_vals,
            np.array(y_mean) - np.array(y_std),
            np.minimum(
                np.array(y_mean) + np.array(y_std),
                1.0,
            ),
            alpha=0.2,
        )
        ax2.plot(
            [0, 100], [0, 1], "k--", alpha=0.3,
            label="Uniform",
        )
    ax2.set_xlabel("% of Keys (sorted by weight)")
    ax2.set_ylabel("Cumulative Attention Mass")
    ax2.set_title("Concentration Curve")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(
            title, fontsize=13, fontweight="bold",
        )
    plt.tight_layout()
    save_figure(fig, out_path)
