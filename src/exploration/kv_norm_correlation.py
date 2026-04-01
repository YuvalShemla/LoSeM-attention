"""
Key-value norm correlation analysis.

Examines whether key and value L2 norms are
correlated — overall, per sequence, and specifically
for the top-scoring keys per query. This informs
whether value-weighted sampling has an advantage.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from ..core import (
    softmax, norm_statistics, kv_norm_correlation,
)
from ..evaluation.plotting import (
    setup_style, save_figure,
)


def compute_kv_norm_data(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    top_pct: float = 10.0,
) -> Dict:
    """
    Compute K-V norm correlation data.

    Returns dict with overall correlation, per-query
    top-keys correlation, and norm arrays.
    """
    k_norms = np.linalg.norm(K, axis=1)
    v_norms = np.linalg.norm(V, axis=1)

    overall_corr = kv_norm_correlation(K, V)
    k_stats = norm_statistics(K)
    v_stats = norm_statistics(V)

    top_k_norms = []
    top_v_norms = []
    for qpos in query_positions:
        q = Q[qpos]
        keys = K[:qpos + 1]
        n = len(keys)
        logits = (q @ keys.T) / np.sqrt(head_dim)

        n_top = max(1, int(n * top_pct / 100))
        top_idx = np.argpartition(
            logits, -n_top
        )[-n_top:]
        top_k_norms.extend(
            k_norms[top_idx].tolist()
        )
        top_v_norms.extend(
            v_norms[top_idx].tolist()
        )

    top_corr = 0.0
    if len(top_k_norms) > 1:
        top_corr = float(np.corrcoef(
            top_k_norms, top_v_norms
        )[0, 1])

    return {
        "overall_correlation": overall_corr,
        "top_keys_correlation": top_corr,
        "k_norms": k_norms.tolist(),
        "v_norms": v_norms.tolist(),
        "k_stats": k_stats,
        "v_stats": v_stats,
        "top_k_norms": top_k_norms,
        "top_v_norms": top_v_norms,
    }


def plot_kv_norms(
    data: Dict,
    out_path: Path,
    title: str = "",
):
    """
    Plot K-V norm relationship.

    Three panels: hexbin of all K vs V norms,
    histogram of within-position correlations,
    and hexbin for top-scoring keys only.
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 5),
    )

    k_n = np.array(data["k_norms"])
    v_n = np.array(data["v_norms"])

    # Subsample for hexbin
    n = len(k_n)
    if n > 10000:
        idx = np.random.default_rng(42).choice(
            n, 10000, replace=False,
        )
        k_sub, v_sub = k_n[idx], v_n[idx]
    else:
        k_sub, v_sub = k_n, v_n

    ax1.hexbin(
        k_sub, v_sub, gridsize=40,
        cmap="Blues", mincnt=1,
    )
    ax1.set_xlabel("||K|| (L2 norm)")
    ax1.set_ylabel("||V|| (L2 norm)")
    r = data["overall_correlation"]
    ax1.set_title(f"All Keys (r={r:.3f})")

    # K and V norm distributions
    ax2.hist(
        k_n, bins=50, alpha=0.5, label="K norms",
        density=True,
    )
    ax2.hist(
        v_n, bins=50, alpha=0.5, label="V norms",
        density=True,
    )
    ax2.set_xlabel("L2 Norm")
    ax2.set_ylabel("Density")
    ax2.set_title("Norm Distributions")
    ax2.legend(fontsize=8)

    # Top-scoring keys only
    tk = np.array(data["top_k_norms"])
    tv = np.array(data["top_v_norms"])
    if len(tk) > 5000:
        idx = np.random.default_rng(42).choice(
            len(tk), 5000, replace=False,
        )
        tk, tv = tk[idx], tv[idx]
    if len(tk) > 0:
        ax3.hexbin(
            tk, tv, gridsize=40,
            cmap="Oranges", mincnt=1,
        )
    r_top = data["top_keys_correlation"]
    ax3.set_xlabel("||K|| (top keys)")
    ax3.set_ylabel("||V|| (top keys)")
    ax3.set_title(
        f"Top-10% Keys (r={r_top:.3f})"
    )

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.2)

    if title:
        fig.suptitle(
            title, fontsize=13, fontweight="bold",
        )
    plt.tight_layout()
    save_figure(fig, out_path)
