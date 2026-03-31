"""
Pairwise comparison dashboard — 12-panel figure per head.

Layout (GridSpec 3×4):
         Cosine Hist      Cosine vs Dist    Dot Hist         Dot vs Dist
  QK:    sink/local/glob  binned mean±SEM   sink/local/glob  binned mean±SEM
  QQ:    sink/local/glob  binned mean±SEM   sink/local/glob  binned mean±SEM
  KK:    sink/local/glob  binned mean±SEM   sink/local/glob  binned mean±SEM
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List

from ..experiment.plotting import setup_style, save_figure

from .pairwise_similarity import compute_pairwise_data


# ═══════════════════════════════════════════════════════
# Compute all pairwise data
# ═══════════════════════════════════════════════════════

def compute_pairwise_dashboard_data(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    config: dict,
) -> Dict:
    """Compute pairwise data for all three pair types."""
    pw_cfg = config.get("pairwise", {})
    n_bins = pw_cfg.get("n_distance_bins", 15)
    max_targets = pw_cfg.get("n_comparison_positions", 20000)
    local_window = config.get(
        "exploration", {},
    ).get("local_window", {}).get("size", 1024)
    seed = config.get("exploration", {}).get("seed", 42)

    result = {}
    for pt in ["qk", "qq", "kk"]:
        result[pt] = compute_pairwise_data(
            Q, K, head_dim, query_positions,
            pair_type=pt,
            n_distance_bins=n_bins,
            local_window=local_window,
            max_targets=max_targets,
            seed=seed,
        )
    return result


# ═══════════════════════════════════════════════════════
# Panel drawing helpers
# ═══════════════════════════════════════════════════════

def _softmax_weights(data):
    """Get per-query softmax attention weights by category.

    Uses pre-computed per-query softmax from pairwise data.
    Returns (w_all, w_sink, w_local, w_global) or all None
    if the pair type has no attention semantics.
    """
    pair_type = data.get("pair_type", "qk")
    if pair_type != "qk":
        return None, None, None, None

    w = data.get("attn_weights")
    if w is None:
        return None, None, None, None

    is_sink = data["is_sink"]
    distances = data["distances"]
    local_w = data.get("local_window", 1024)
    local_mask = ~is_sink & (distances <= local_w)
    global_mask = ~is_sink & (distances > local_w)

    return w, w[is_sink], w[local_mask], w[global_mask]


def _panel_histogram(
    ax, data, metric, pair_label, is_variance=False,
):
    """
    Histogram with sink / local / global overlays.

    For QK pairs, uses softmax (attention-weighted) density.
    Sink shown in orange for pair types involving keys (QK, KK).

    metric: "cosine" or "dots"
    """
    values = data[f"{metric}_all"]
    pair_type = data.get("pair_type", "qk")
    has_keys = pair_type in ("qk", "kk")

    # Recompute category masks from raw arrays
    is_sink = data["is_sink"]
    distances = data["distances"]
    local_w = data.get("local_window", 1024)
    local_mask = ~is_sink & (distances <= local_w)
    global_mask = ~is_sink & (distances > local_w)

    sink_vals = values[is_sink]
    local_vals = values[local_mask]
    global_vals = values[global_mask]

    # Softmax weights for QK (attention-weighted density)
    w_all, w_sink, w_local, w_global = (
        _softmax_weights(data)
    )
    use_softmax = w_all is not None

    # Variance: faint per-head histograms
    if is_variance and f"_all_heads_{metric}" in data:
        for head_vals in data[f"_all_heads_{metric}"]:
            ax.hist(
                head_vals, bins=50, density=True,
                color="gray", alpha=0.08,
                histtype="step", lw=0.5,
            )

    # Background: all pairs (light gray fill)
    ax.hist(
        values, bins=60, density=True,
        weights=w_all,
        color="#e0e0e0", alpha=0.6,
        edgecolor="white", linewidth=0.3,
    )

    # Global pairs (green step)
    if len(global_vals) > 10:
        ax.hist(
            global_vals, bins=50, density=True,
            weights=w_global,
            histtype="step", lw=1.3,
            color="#2ca02c",
            label=f"global (d>{local_w})",
        )

    # Local pairs (blue step)
    if len(local_vals) > 10:
        ax.hist(
            local_vals, bins=50, density=True,
            weights=w_local,
            histtype="step", lw=1.3,
            color="#1f77b4",
            label=f"local (d\u2264{local_w})",
        )

    # Sink pairs (orange) — only for pair types with keys
    if has_keys and len(sink_vals) > 0:
        ax.hist(
            sink_vals, bins=30, density=True,
            weights=w_sink,
            histtype="step", lw=2.0,
            color="#ff7f0e",
            label=u"sink (k\u2080)",
        )

    dot_labels = {
        "QK": r"$q \cdot k / \sqrt{d}$",
        "QQ": r"$q \cdot q / \sqrt{d}$",
        "KK": r"$k \cdot k / \sqrt{d}$",
    }
    metric_label = (
        "Cosine Similarity" if metric == "cosine"
        else dot_labels.get(pair_label, "Scaled Dot")
    )
    ax.set_xlabel(metric_label)
    ylabel = (
        "Density (attn-weighted)" if use_softmax
        else "Density"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{pair_label} {metric_label}")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def _ema_smooth(distances, values, span):
    """EMA smoothing sorted by distance in log-space."""
    order = np.argsort(distances)
    sorted_d = distances[order]
    sorted_v = values[order]

    alpha = 2.0 / (span + 1)
    smoothed = np.empty_like(sorted_v)
    smoothed[0] = sorted_v[0]
    for i in range(1, len(sorted_v)):
        smoothed[i] = (
            alpha * sorted_v[i]
            + (1 - alpha) * smoothed[i - 1]
        )
    return sorted_d, smoothed


def _panel_vs_distance(
    ax, data, metric, pair_label,
    is_variance=False, ema_span=200,
):
    """
    Scatter of all pairs (distance vs metric) with EMA.

    Sink pairs shown as red markers, non-sink as gray dots.
    EMA smoothing curve overlaid on non-sink data.
    """
    values = data[f"{metric}_all"]
    distances = data["distances"]
    is_sink = data["is_sink"]

    if len(values) == 0:
        ax.text(
            0.5, 0.5, "No data",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        return

    # Non-sink scatter
    ns = ~is_sink
    ns_d = distances[ns].astype(float)
    ns_v = values[ns]

    # Subsample for scatter if too many
    n_ns = len(ns_d)
    if n_ns > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_ns, 5000, replace=False)
        plot_d, plot_v = ns_d[idx], ns_v[idx]
    else:
        plot_d, plot_v = ns_d, ns_v

    # Offset zero distances for log scale
    plot_d_log = np.maximum(plot_d, 0.5)
    ax.scatter(
        plot_d_log, plot_v,
        s=1, alpha=0.08, c="gray",
        rasterized=True,
    )

    # Sink scatter (orange stars)
    sink_d = distances[is_sink].astype(float)
    sink_v = values[is_sink]
    if len(sink_d) > 0:
        sink_d_log = np.maximum(sink_d, 0.5)
        ax.scatter(
            sink_d_log, sink_v,
            s=50, alpha=0.7, c="#ff7f0e",
            marker="*", label=u"sink (k\u2080)",
            rasterized=True, edgecolors="black",
            linewidths=0.3, zorder=5,
        )

    # EMA smoothing on non-sink data
    if len(ns_d) > 50:
        ema_d, ema_v = _ema_smooth(ns_d, ns_v, ema_span)
        ema_d_log = np.maximum(ema_d, 0.5)
        color = "#1f77b4" if metric == "cosine" else "#ff7f0e"
        ax.plot(
            ema_d_log, ema_v,
            color=color, lw=1.8, alpha=0.9,
            label=f"EMA (span={ema_span})",
        )

    # Local window boundary
    local_w = data.get("local_window", 1024)
    ax.axvline(
        local_w, color="#1f77b4", ls=":",
        lw=1, alpha=0.5, label=f"local={local_w}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Positional Distance")
    dot_labels = {
        "QK": r"$q \cdot k / \sqrt{d}$",
        "QQ": r"$q \cdot q / \sqrt{d}$",
        "KK": r"$k \cdot k / \sqrt{d}$",
    }
    metric_label = (
        "Cosine Sim" if metric == "cosine"
        else dot_labels.get(pair_label, "Scaled Dot")
    )
    ax.set_ylabel(metric_label)
    ax.set_title(f"{pair_label} {metric_label} vs Distance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls="--")


# ═══════════════════════════════════════════════════════
# Dashboard assembly
# ═══════════════════════════════════════════════════════

def create_pairwise_dashboard(
    data: Dict,
    title: str,
    out_path: Path,
    ema_span: int = 200,
):
    """
    Create 12-panel pairwise comparison dashboard.

    Layout (GridSpec 3×4):
      Rows: QK, QQ, KK
      Cols: Cosine Hist | Cosine vs Dist | Dot Hist | Dot vs Dist
    """
    is_variance = any(
        "_all_heads_cosine" in data.get(pt, {})
        for pt in ["qk", "qq", "kk"]
    )

    setup_style()
    fig = plt.figure(figsize=(28, 18))
    gs = GridSpec(
        3, 4, figure=fig,
        hspace=0.35, wspace=0.35,
    )

    pair_types = ["qk", "qq", "kk"]
    pair_labels = ["QK", "QQ", "KK"]

    for row, (pt, pl) in enumerate(
        zip(pair_types, pair_labels),
    ):
        pt_data = data[pt]

        # Col 0: Cosine histogram
        ax = fig.add_subplot(gs[row, 0])
        _panel_histogram(
            ax, pt_data, "cosine", pl, is_variance,
        )

        # Col 1: Cosine vs distance
        ax = fig.add_subplot(gs[row, 1])
        _panel_vs_distance(
            ax, pt_data, "cosine", pl,
            is_variance, ema_span,
        )

        # Col 2: Dot histogram
        ax = fig.add_subplot(gs[row, 2])
        _panel_histogram(
            ax, pt_data, "dots", pl, is_variance,
        )

        # Col 3: Dot vs distance
        ax = fig.add_subplot(gs[row, 3])
        _panel_vs_distance(
            ax, pt_data, "dots", pl,
            is_variance, ema_span,
        )

    fig.suptitle(
        title, fontsize=16, fontweight="bold", y=0.98,
    )
    save_figure(fig, out_path, dpi=150)
