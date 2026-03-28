"""
Entropy distribution analysis.

Computes attention entropy across query positions
and heads. Produces entropy-vs-position line plots
and entropy histograms, both with and without
sink/local window exclusion.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from ..core import (
    softmax, entropy_nats, nonlocal_mask,
)
from ..experiment.plotting import (
    setup_style, save_figure,
)


def compute_entropy_data(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    n_sink: int = 1,
    local_window: int = 1024,
) -> Dict:
    """
    Compute entropy at each query position.

    Returns dict with per-position entropy
    (full and no-sink-local variants).
    """
    results = {
        "positions": query_positions,
        "full_entropy": [],
        "nonlocal_entropy": [],
    }

    for qpos in query_positions:
        q = Q[qpos]
        keys = K[:qpos + 1]
        n = len(keys)
        logits = (q @ keys.T) / np.sqrt(head_dim)
        weights = softmax(logits)

        results["full_entropy"].append(
            entropy_nats(weights)
        )

        mask = nonlocal_mask(
            n, n_sink, local_window,
        )
        w_masked = weights[mask]
        total = np.sum(w_masked)
        if total > 1e-10:
            w_normed = w_masked / total
            results["nonlocal_entropy"].append(
                entropy_nats(w_normed)
            )
        else:
            results["nonlocal_entropy"].append(
                0.0
            )

    return results


def plot_entropy(
    data: Dict,
    out_path: Path,
    title: str = "",
):
    """
    Plot entropy vs position and histogram.

    Two panels: entropy vs query position (with
    reference lines for uniform over N keys),
    and histogram of entropy values.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6),
    )

    pos = data["positions"]
    e_full = data["full_entropy"]
    e_no_sl = data["nonlocal_entropy"]

    ax1.plot(
        pos, e_full, label="Full", lw=1.5,
        alpha=0.8,
    )
    ax1.plot(
        pos, e_no_sl,
        label="Excl. sink + local",
        lw=1.5, alpha=0.8,
    )
    # Uniform reference: log(N) for N = position
    ref_pos = np.array(pos)
    ref_ent = np.log(np.maximum(ref_pos, 1))
    ax1.plot(
        pos, ref_ent, "k--", alpha=0.3,
        label="log(N) uniform",
    )
    ax1.set_xlabel("Query Position")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title("Entropy vs Position")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.hist(
        e_full, bins=40, alpha=0.6,
        label="Full", density=True,
    )
    ax2.hist(
        e_no_sl, bins=40, alpha=0.6,
        label="Excl. sink + local", density=True,
    )
    ax2.set_xlabel("Entropy (nats)")
    ax2.set_ylabel("Density")
    ax2.set_title("Entropy Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(
            title, fontsize=13, fontweight="bold",
        )
    plt.tight_layout()
    save_figure(fig, out_path)
