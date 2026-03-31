"""
RoPE vs Raw comparison dashboard.

For each head, loads both post-RoPE and pre-RoPE vectors
and shows the same analyses side by side.

Layout (GridSpec 4 rows × 12 cols):
  Row 0: [RoPE] TopK Mass | Entropy | Bias   [Raw] same
  Row 1: [RoPE] QK cos | QK dot              [Raw] same
  Row 2: [RoPE] KK cos | KK dot              [Raw] same
  Row 3: [RoPE] PCA | t-SNE | UMAP           [Raw] same
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List

from ..experiment.plotting import setup_style, save_figure

from .embedding_projections import compute_embedding_projections
from .pairwise_similarity import compute_pairwise_data
from .attention_concentration import compute_concentration_data
from .entropy_distribution import compute_entropy_data
from .topk_vs_sampling_bias import compute_bias_data
from .dashboard_global import (
    _panel_topk_mass,
    _panel_entropy,
    _panel_bias,
)


# ═══════════════════════════════════════════════════════
# Compute data for one variant (rope or raw)
# ═══════════════════════════════════════════════════════

def _compute_variant(
    Q, K, head_dim, query_positions, config,
    V=None,
    all_query_positions=None,
):
    """Compute embedding + pairwise + analysis data.

    Parameters
    ----------
    query_positions : list
        Last-N query positions (for pairwise plots).
    V : np.ndarray, optional
        Value vectors (needed for bias computation).
    all_query_positions : list, optional
        Every token position (for embedding projections).
        Falls back to *query_positions* if not provided.
    """
    if all_query_positions is None:
        all_query_positions = query_positions

    ecfg = config.get("exploration", {})
    pw_cfg = config.get("pairwise", {})
    n_bins = pw_cfg.get("n_distance_bins", 15)
    max_targets = pw_cfg.get("n_comparison_positions", 20000)
    n_sink = ecfg.get(
        "attention_sink", {},
    ).get("n_sink_tokens", 1)
    local_window = ecfg.get(
        "local_window", {},
    ).get("size", 1024)
    seed = ecfg.get("seed", 42)

    conc_cfg = config.get("concentration", {})
    top_k_values = conc_cfg.get(
        "top_k_values", [10, 50, 100, 200, 500],
    )
    bias_cfg = config.get("bias_comparison", {})
    budgets = bias_cfg.get(
        "budgets",
        [16, 32, 64, 128, 512,
         1024, 2048, 4096, 8192, 16384],
    )

    # Embeddings → all queries for full picture
    embedding = None
    try:
        embedding = compute_embedding_projections(
            Q, K, all_query_positions, config,
        )
    except Exception as e:
        print(f"        Embedding failed: {e}")

    # Pairwise → last queries only
    pairwise = {}
    for pt in ["qk", "kk"]:
        pairwise[pt] = compute_pairwise_data(
            Q, K, head_dim, query_positions,
            pair_type=pt,
            n_distance_bins=n_bins,
            local_window=local_window,
            max_targets=max_targets,
            seed=seed,
        )

    # Concentration, entropy, bias
    result = {
        "embedding": embedding,
        "pairwise": pairwise,
    }

    result["concentration"] = compute_concentration_data(
        Q, K, head_dim, query_positions,
        top_k_values=top_k_values,
    )
    result["entropy"] = compute_entropy_data(
        Q, K, head_dim, query_positions,
        n_sink, local_window,
    )
    if V is not None:
        result["bias"] = compute_bias_data(
            Q, K, V, head_dim, query_positions,
            budgets=budgets,
            seed=seed,
        )

    return result


# ═══════════════════════════════════════════════════════
# Panel helpers
# ═══════════════════════════════════════════════════════

_POS_GROUP_STYLE = {
    "Q_mid":   {"c": "#bbdefb", "s": 2, "alpha": 0.4},
    "K_mid":   {"c": "#e1bee7", "s": 2, "alpha": 0.4},
    "Q_first": {"c": "#00acc1", "s": 4, "alpha": 0.45},
    "Q_last":  {"c": "#1565c0", "s": 4, "alpha": 0.45},
    "K_first": {"c": "#ec407a", "s": 4, "alpha": 0.45},
    "K_last":  {"c": "#7b1fa2", "s": 4, "alpha": 0.45},
}
_POS_GROUP_LABELS = {
    "Q_first": "Q first 1K", "Q_mid": "Q mid",
    "Q_last": "Q last 1K", "K_first": "K first 1K",
    "K_mid": "K mid", "K_last": "K last 1K",
    "sink": "sink",
}


def _scatter_qk(ax, data, proj_key, coord_key, title):
    """Q/K scatter colored by position group."""
    if data is None or coord_key not in data:
        ax.text(
            0.5, 0.5, "unavailable",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=8,
        )
        return

    coords = data[coord_key]
    labels = data[f"{proj_key}_labels"]
    is_sink = data[f"{proj_key}_is_sink"]
    pos_groups = data.get(f"{proj_key}_pos_groups")

    if pos_groups is not None:
        draw_order = [
            "Q_mid", "K_mid",
            "Q_first", "K_first",
            "Q_last", "K_last",
        ]
        for group in draw_order:
            mask = pos_groups == group
            if not np.any(mask):
                continue
            sty = _POS_GROUP_STYLE[group]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=sty["s"], alpha=sty["alpha"],
                c=sty["c"],
                label=_POS_GROUP_LABELS[group],
                rasterized=True,
            )
        sink_mask = pos_groups == "sink"
        if np.any(sink_mask):
            ax.scatter(
                coords[sink_mask, 0], coords[sink_mask, 1],
                s=80, marker="*", c="red", label="sink",
                zorder=5, edgecolors="black",
                linewidths=0.5,
            )
    else:
        q_mask = (labels == "Q") & ~is_sink
        k_mask = (labels == "K") & ~is_sink
        ax.scatter(
            coords[q_mask, 0], coords[q_mask, 1],
            s=2, alpha=0.25, c="#1f77b4", label="Q",
            rasterized=True,
        )
        ax.scatter(
            coords[k_mask, 0], coords[k_mask, 1],
            s=2, alpha=0.25, c="#9467bd", label="K",
            rasterized=True,
        )
        if np.any(is_sink):
            ax.scatter(
                coords[is_sink, 0], coords[is_sink, 1],
                s=80, marker="*", c="red", label="sink",
                zorder=5, edgecolors="black",
                linewidths=0.5,
            )

    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=5, markerscale=2, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])


def _hist_panel(ax, data, metric, pair_label, title_suffix):
    """Histogram with sink/local/global for one variant.

    QK uses softmax attention-weighted density.
    Sink shown in orange for pair types with keys.
    """
    pair_type = data.get("pair_type", "qk")
    has_keys = pair_type in ("qk", "kk")
    values = data[f"{metric}_all"]

    # Recompute masks from raw arrays
    is_sink = data["is_sink"]
    distances = data["distances"]
    local_w = data.get("local_window", 1024)
    local_mask = ~is_sink & (distances <= local_w)
    global_mask = ~is_sink & (distances > local_w)

    sink_vals = values[is_sink]
    local_vals = values[local_mask]
    global_vals = values[global_mask]

    # Per-query softmax weights for QK
    if pair_type == "qk":
        w_all = data.get("attn_weights")
        if w_all is not None:
            w_sink = w_all[is_sink]
            w_local = w_all[local_mask]
            w_global = w_all[global_mask]
        else:
            w_sink = w_local = w_global = None
    else:
        w_all = w_sink = w_local = w_global = None

    ax.hist(
        values, bins=60, density=True,
        weights=w_all,
        color="#e0e0e0", alpha=0.6,
        edgecolor="white", linewidth=0.3,
    )
    if len(global_vals) > 10:
        ax.hist(
            global_vals, bins=50, density=True,
            weights=w_global,
            histtype="step", lw=1.2,
            color="#2ca02c", label="global",
        )
    if len(local_vals) > 10:
        ax.hist(
            local_vals, bins=50, density=True,
            weights=w_local,
            histtype="step", lw=1.2,
            color="#1f77b4", label="local",
        )
    if has_keys and len(sink_vals) > 0:
        ax.hist(
            sink_vals, bins=30, density=True,
            weights=w_sink,
            histtype="step", lw=2.0,
            color="#ff7f0e",
            label=u"sink (k\u2080)",
        )

    dot_labels = {
        "QK": r"$q{\cdot}k/\sqrt{d}$",
        "KK": r"$k{\cdot}k/\sqrt{d}$",
    }
    metric_label = (
        "Cosine" if metric == "cosine"
        else dot_labels.get(pair_label, "Scaled Dot")
    )
    ax.set_title(
        f"{pair_label} {metric_label} ({title_suffix})",
        fontsize=9,
    )
    ylabel = (
        "Density (attn-weighted)" if pair_type == "qk"
        else "Density"
    )
    ax.set_ylabel(ylabel, fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _stats_panel(ax, data, title_suffix):
    """Silhouette-style metrics bar chart."""
    if data is None:
        ax.text(
            0.5, 0.5, "unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        return

    m = data.get("orig_metrics")
    if m is None:
        ax.text(
            0.5, 0.5, "No metrics",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        return

    labels = ["Intra-Q", "Intra-K", "Inter Q-K"]
    vals = [
        m["mean_intra_q"], m["mean_intra_k"],
        m["mean_inter_qk"],
    ]
    colors = ["#1976d2", "#7b1fa2", "#8e5a2d"]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v,
            f"{v:.1f}", ha="center", va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("Mean Distance", fontsize=8)
    ax.set_title(
        f"Q-K Clusters ({title_suffix}, "
        f"sep={m['separation_ratio']:.2f})",
        fontsize=9,
    )
    ax.grid(True, alpha=0.2, axis="y")


# ═══════════════════════════════════════════════════════
# Dashboard assembly
# ═══════════════════════════════════════════════════════

def create_rope_comparison_dashboard(
    rope_data: Dict,
    raw_data: Dict,
    title: str,
    out_path: Path,
):
    """
    Create RoPE vs Raw comparison dashboard.

    4 rows × 12 cols:
      Row 0: [RoPE] TopK Mass|Entropy|Bias  [Raw] same
      Row 1: [RoPE] QK cos|QK dot   [Raw] QK cos|QK dot
      Row 2: [RoPE] KK cos|KK dot   [Raw] KK cos|KK dot
      Row 3: [RoPE] PCA|t-SNE|UMAP  [Raw] PCA|t-SNE|UMAP
    """
    setup_style()
    fig = plt.figure(figsize=(30, 24))
    gs = GridSpec(
        4, 12, figure=fig,
        hspace=0.40, wspace=0.35,
    )

    # Column labels
    fig.text(
        0.28, 0.97, "Post-RoPE",
        fontsize=14, fontweight="bold",
        ha="center", color="#2d5e8e",
    )
    fig.text(
        0.75, 0.97, "Pre-RoPE (Raw)",
        fontsize=14, fontweight="bold",
        ha="center", color="#8e2d2d",
    )

    # Row 0 — TopK Mass | Entropy | Bias
    for col_off, variant, suffix in [
        (0, rope_data, "RoPE"),
        (6, raw_data, "Raw"),
    ]:
        conc = variant.get("concentration")
        if conc:
            ax = fig.add_subplot(
                gs[0, col_off:col_off + 2],
            )
            _panel_topk_mass(ax, conc)
            ax.set_title(
                f"Top-K Mass ({suffix})", fontsize=9,
            )
        else:
            ax = fig.add_subplot(
                gs[0, col_off:col_off + 2],
            )
            ax.text(
                0.5, 0.5, "unavailable",
                ha="center", va="center",
                transform=ax.transAxes,
            )

        ent = variant.get("entropy")
        if ent:
            ax = fig.add_subplot(
                gs[0, col_off + 2:col_off + 4],
            )
            _panel_entropy(ax, ent)
            ax.set_title(
                f"Entropy ({suffix})", fontsize=9,
            )
        else:
            ax = fig.add_subplot(
                gs[0, col_off + 2:col_off + 4],
            )
            ax.text(
                0.5, 0.5, "unavailable",
                ha="center", va="center",
                transform=ax.transAxes,
            )

        bias = variant.get("bias")
        if bias:
            ax = fig.add_subplot(
                gs[0, col_off + 4:col_off + 6],
            )
            _panel_bias(ax, bias)
            ax.set_title(
                f"Bias ({suffix})", fontsize=9,
            )
        else:
            ax = fig.add_subplot(
                gs[0, col_off + 4:col_off + 6],
            )
            ax.text(
                0.5, 0.5, "unavailable",
                ha="center", va="center",
                transform=ax.transAxes,
            )

    # Row 1 — QK histograms
    for col_off, variant, suffix in [
        (0, rope_data, "RoPE"),
        (6, raw_data, "Raw"),
    ]:
        qk = variant["pairwise"]["qk"]
        ax = fig.add_subplot(gs[1, col_off:col_off + 3])
        _hist_panel(ax, qk, "cosine", "QK", suffix)

        ax = fig.add_subplot(gs[1, col_off + 3:col_off + 6])
        _hist_panel(ax, qk, "dots", "QK", suffix)

    # Row 2 — KK histograms
    for col_off, variant, suffix in [
        (0, rope_data, "RoPE"),
        (6, raw_data, "Raw"),
    ]:
        kk = variant["pairwise"]["kk"]
        ax = fig.add_subplot(gs[2, col_off:col_off + 3])
        _hist_panel(ax, kk, "cosine", "KK", suffix)

        ax = fig.add_subplot(gs[2, col_off + 3:col_off + 6])
        _hist_panel(ax, kk, "dots", "KK", suffix)

    # Row 3 — 2D projections
    for col_off, variant, suffix in [
        (0, rope_data, "RoPE"),
        (6, raw_data, "Raw"),
    ]:
        emb = variant.get("embedding")
        ax = fig.add_subplot(gs[3, col_off:col_off + 2])
        if emb and "pca_coords" in emb:
            ev = emb["pca_explained_var"]
            _scatter_qk(
                ax, emb, "pca", "pca_coords",
                f"PCA ({ev[0]:.0%}+{ev[1]:.0%})",
            )
        else:
            _scatter_qk(
                ax, None, "pca", "pca_coords", "PCA",
            )

        ax = fig.add_subplot(
            gs[3, col_off + 2:col_off + 4],
        )
        _scatter_qk(
            ax, emb, "tsne", "tsne_coords",
            "t-SNE (local structure)",
        )

        ax = fig.add_subplot(
            gs[3, col_off + 4:col_off + 6],
        )
        _scatter_qk(
            ax, emb, "umap", "umap_coords",
            "UMAP (global structure)",
        )

    fig.suptitle(
        title, fontsize=16, fontweight="bold", y=0.99,
    )
    save_figure(fig, out_path, dpi=150)
