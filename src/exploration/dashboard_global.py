"""
Global overview dashboard — 8-panel figure per head.

Layout (GridSpec 3×12):
  Row 0: Top-K Mass [0:4] | Entropy [4:8] | Bias [8:12]
  Row 1: Mean-Q vs Full [0:6] | Query Deviation [6:12]
  Row 2: PCA (Q+K) | t-SNE (Q+K) | UMAP (Q+K) | Stats
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List

from ..evaluation.plotting import setup_style, save_figure

from .attention_concentration import compute_concentration_data
from .entropy_distribution import compute_entropy_data
from .topk_vs_sampling_bias import compute_bias_data
from .query_analysis import (
    compute_meanquery_data,
    compute_query_deviation_data,
)
from .embedding_projections import compute_embedding_projections


# ═══════════════════════════════════════════════════════
# Compute all global data
# ═══════════════════════════════════════════════════════

def compute_global_data(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    config: dict,
    all_query_positions: List[int] = None,
) -> Dict:
    """Compute all analysis data for the global dashboard.

    Parameters
    ----------
    query_positions : list
        Last-N query positions (for position-based plots:
        concentration, entropy, bias).
    all_query_positions : list, optional
        Every token position (for full-picture plots:
        embeddings, query deviation, mean-query).
        Falls back to *query_positions* if not provided.
    """
    if all_query_positions is None:
        all_query_positions = query_positions

    ecfg = config.get("exploration", {})
    n_sink = (
        1 if ecfg.get("exclude_sink_token", True)
        else 0
    )
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

    # Position-based plots → last queries only
    result = {
        "concentration": compute_concentration_data(
            Q, K, head_dim, query_positions,
            top_k_values=top_k_values,
        ),
        "entropy": compute_entropy_data(
            Q, K, head_dim, query_positions,
            n_sink, local_window,
        ),
        "bias": compute_bias_data(
            Q, K, V, head_dim, query_positions,
            budgets=budgets,
            seed=seed,
        ),
    }

    # Full-picture plots → all queries
    result["query_deviation"] = compute_query_deviation_data(
        Q, all_query_positions,
    )
    result["meanquery"] = compute_meanquery_data(
        Q, K, head_dim, all_query_positions,
        seed=seed,
    )

    try:
        result["embedding"] = compute_embedding_projections(
            Q, K, all_query_positions, config,
        )
    except Exception as e:
        print(f"      Embedding projections failed: {e}")
        result["embedding"] = None

    return result


# ═══════════════════════════════════════════════════════
# Panel drawing functions
# ═══════════════════════════════════════════════════════

def _panel_topk_mass(ax, data, is_variance=False):
    """Top-K attention mass vs query position."""
    qpos = data["query_positions"]
    for k, masses in data["top_k_mass"].items():
        ax.plot(qpos, masses, label=f"Top-{k}", lw=1.2)
        if is_variance and "_all_heads_topk" in data:
            for head_vals in data["_all_heads_topk"][k]:
                ax.plot(
                    qpos, head_vals,
                    color="gray", alpha=0.15, lw=0.5,
                )
            spread = data.get("_spread_topk", {}).get(k)
            if spread:
                ax.fill_between(
                    qpos, spread["p25"], spread["p75"],
                    alpha=0.15, color="gray",
                )
    ax.set_xlabel("Query Position")
    ax.set_ylabel("Attention Mass")
    ax.set_title("Top-K Mass vs Position")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


def _panel_concentration(ax, data):
    """Concentration curves at query-position percentiles."""
    curves = data["concentration_curves"]
    if not curves:
        return
    pcts = sorted(curves[0].keys())
    x_vals = [
        float(p.split("_")[1].replace("pct", ""))
        for p in pcts
    ]

    n = len(curves)
    percentile_labels = [
        ("p10", int(n * 0.1)),
        ("p25", int(n * 0.25)),
        ("p50", int(n * 0.5)),
        ("p75", int(n * 0.75)),
        ("p90", int(n * 0.9)),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#d62728",
              "#ff7f0e", "#1f77b4"]
    alphas = [0.5, 0.7, 1.0, 0.7, 0.5]

    for (label, idx), c, a in zip(
        percentile_labels, colors, alphas,
    ):
        idx = min(idx, n - 1)
        y = [curves[idx][p] for p in pcts]
        ax.plot(
            x_vals, y, color=c, alpha=a,
            lw=1.5, label=label,
        )

    y_mean = [
        np.mean([c[p] for c in curves])
        for p in pcts
    ]
    ax.plot(
        x_vals, y_mean, "k-", lw=2,
        label="mean", alpha=0.8,
    )
    ax.plot(
        [0, 100], [0, 1], "k--", alpha=0.2,
        label="uniform",
    )
    ax.set_xlabel("% of Keys (sorted)")
    ax.set_ylabel("Cumulative Mass")
    ax.set_title("Concentration Curves")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _panel_entropy(ax, data, is_variance=False):
    """Entropy vs query position."""
    pos = data["positions"]
    ax.plot(
        pos, data["full_entropy"],
        label="Full", lw=1.2, alpha=0.8,
    )
    ax.plot(
        pos, data["effective_entropy"],
        label="Excl. sink+local", lw=1.2, alpha=0.8,
    )
    ref = np.log(np.maximum(np.array(pos), 1))
    ax.plot(
        pos, ref, "k--", alpha=0.2,
        label="log(N)",
    )

    if is_variance and "_all_heads" in data:
        for key, color in [
            ("full_entropy", "#1f77b4"),
            ("effective_entropy", "#ff7f0e"),
        ]:
            for head_vals in data["_all_heads"][key]:
                ax.plot(
                    pos, head_vals,
                    color="gray", alpha=0.12, lw=0.5,
                )
            spread = data.get("_spread", {}).get(key)
            if spread:
                ax.fill_between(
                    pos, spread["p25"], spread["p75"],
                    alpha=0.12, color=color,
                )

    ax.set_xlabel("Query Position")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Entropy vs Position")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _panel_bias(ax, data, is_variance=False):
    """Approximation methods: TopK, Uniform, IdealSampling."""
    budgets = data["budgets"]
    for method, label, color, marker in [
        ("topk", "TopK", "#d62728", "o"),
        ("uniform", "Uniform Sampling", "#ff7f0e", "^"),
        ("ideal_sampling", "Distribution Sampling",
         "#2ca02c", "s"),
    ]:
        key = f"{method}_value_error"
        y = data[key]
        ax.plot(
            budgets, y, color=color, marker=marker,
            lw=1.5, ms=4, label=label,
        )

        if is_variance and "_all_heads" in data:
            for head_vals in data["_all_heads"].get(key, []):
                ax.plot(
                    budgets, head_vals,
                    color="gray", alpha=0.15, lw=0.5,
                )
            spread = data.get("_spread", {}).get(key)
            if spread:
                ax.fill_between(
                    budgets, spread["p25"], spread["p75"],
                    alpha=0.12, color=color,
                )

    # Ideal Equal Splits: single point + horizontal line
    es_err = data.get("ideal_equal_splits_value_error")
    es_budget = data.get("ideal_equal_splits_budget")
    if es_err is not None and es_budget:
        ax.plot(
            es_budget, es_err, color="#1f77b4",
            marker="D", ms=7, zorder=5,
            label="IdealEqualSplits",
        )
        ax.axhline(
            es_err, color="#1f77b4", ls="--",
            lw=1, alpha=0.5,
        )

    ax.set_xlabel("Number of Keys")
    ax.set_ylabel("Rel. L2 Error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Approximation Methods")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, ls="--")


def _panel_meanquery(ax, data, is_variance=False):
    """Mean-Q logits vs individual-Q logits scatter."""
    if is_variance and "_all_heads_correlations" in data:
        corrs = data["_all_heads_correlations"]
        ax.boxplot(corrs, vert=True)
        ax.set_ylabel("Pearson r")
        ax.set_title("Mean-Q Correlation (per head)")
        ax.grid(True, alpha=0.3)
        return

    ms = data["mean_scores"]
    fs = data["full_scores"]
    corr = data["correlation"]

    n = len(ms)
    if n > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 5000, replace=False)
        ms, fs = ms[idx], fs[idx]

    ax.scatter(
        ms, fs, s=1, alpha=0.15, c="#1f77b4",
        rasterized=True,
    )
    lims = [
        min(ms.min(), fs.min()),
        max(ms.max(), fs.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1, alpha=0.5)
    ax.set_xlabel("mean(Q) logit")
    ax.set_ylabel("Individual Q logit")
    ax.set_title(f"Mean-Q vs Full (r={corr:.3f})")
    ax.grid(True, alpha=0.3)


def _panel_query_deviation(ax, data, is_variance=False):
    r"""||q - mean(Q)|| distribution."""
    if is_variance and "_all_heads_deviations" in data:
        for head_devs in data["_all_heads_deviations"]:
            ax.hist(
                head_devs, bins=40, density=True,
                color="gray", alpha=0.12,
                histtype="step", lw=0.5,
            )

    devs = data["deviations"]
    ax.hist(
        devs, bins=40, density=True,
        color="#9467bd", alpha=0.7,
        edgecolor="white", linewidth=0.3,
    )
    med = float(np.median(devs))
    ax.axvline(
        med, color="red", ls="--", lw=1,
        label=f"median={med:.2f}",
    )
    mean_norm = data["mean_q_norm"]
    ax.set_xlabel(r"$\| q - \overline{Q} \|$")
    ax.set_ylabel("Density")
    ax.set_title(
        r"Query Deviation "
        f"($\\|\\overline{{Q}}\\|$={mean_norm:.1f})"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# Position-group styling:
# middle = faded background, first/last = foreground pops
_POS_GROUP_STYLE = {
    "Q_mid":   {"c": "#bbdefb", "s": 2, "alpha": 0.4},
    "K_mid":   {"c": "#e1bee7", "s": 2, "alpha": 0.4},
    "Q_first": {"c": "#00acc1", "s": 5, "alpha": 0.5},
    "Q_last":  {"c": "#1565c0", "s": 5, "alpha": 0.5},
    "K_first": {"c": "#ec407a", "s": 5, "alpha": 0.5},
    "K_last":  {"c": "#7b1fa2", "s": 5, "alpha": 0.5},
}
_POS_GROUP_LABELS = {
    "Q_first": "Q first 1K",
    "Q_mid":   "Q middle",
    "Q_last":  "Q last 1K",
    "K_first": "K first 1K",
    "K_mid":   "K middle",
    "K_last":  "K last 1K",
    "sink":    "sink",
}


def _scatter_qk(ax, data, proj_key, coord_key, title):
    """Shared scatter logic for projection panels.

    Colors by position group if available, otherwise
    falls back to simple Q/K coloring.
    """
    coords = data[coord_key]
    labels = data[f"{proj_key}_labels"]
    is_sink = data[f"{proj_key}_is_sink"]
    pos_groups = data.get(f"{proj_key}_pos_groups")

    if pos_groups is not None:
        # Draw back to front: mid (background), first, last
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
        # Sink on top
        sink_mask = pos_groups == "sink"
        if np.any(sink_mask):
            ax.scatter(
                coords[sink_mask, 0], coords[sink_mask, 1],
                s=80, marker="*", c="red", label="sink",
                zorder=5, edgecolors="black",
                linewidths=0.5,
            )
    else:
        # Fallback: simple Q/K coloring
        q_mask = (labels == "Q") & ~is_sink
        k_mask = (labels == "K") & ~is_sink
        ax.scatter(
            coords[q_mask, 0], coords[q_mask, 1],
            s=3, alpha=0.3, c="#1f77b4", label="Q",
            rasterized=True,
        )
        ax.scatter(
            coords[k_mask, 0], coords[k_mask, 1],
            s=3, alpha=0.3, c="#9467bd", label="K",
            rasterized=True,
        )
        sink_mask = is_sink
        if np.any(sink_mask):
            ax.scatter(
                coords[sink_mask, 0], coords[sink_mask, 1],
                s=80, marker="*", c="red", label="sink",
                zorder=5, edgecolors="black",
                linewidths=0.5,
            )

    ax.set_title(title, fontsize=9)
    ax.legend(
        fontsize=6, markerscale=2, ncol=2,
        loc="upper right",
    )
    ax.grid(True, alpha=0.3)


def _panel_projection(ax, data, proj_key, coord_key, title):
    """Single projection panel (PCA/t-SNE/UMAP)."""
    if data is None or coord_key not in data:
        ax.text(
            0.5, 0.5, f"{proj_key.upper()} unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        return
    _scatter_qk(ax, data, proj_key, coord_key, title)


def _panel_embedding_stats(ax, data):
    """Cluster separation metrics (original space)."""
    if data is None:
        ax.text(
            0.5, 0.5, "Stats unavailable",
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

    labels = [
        "Intra-Q",
        "Intra-K",
        "Inter Q-K",
    ]
    vals = [
        m["mean_intra_q"],
        m["mean_intra_k"],
        m["mean_inter_qk"],
    ]
    colors = ["#1976d2", "#7b1fa2", "#8e5a2d"]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v,
            f"{v:.1f}", ha="center", va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_title(
        f"Q-K Clusters (sep="
        f"{m['separation_ratio']:.2f})"
    )
    ax.grid(True, alpha=0.2, axis="y")


def _panel_embedding_stats_agg(ax, embeddings):
    """Aggregated: per-head silhouette + intra distances."""
    n = len(embeddings)
    intra_q = [e["orig_metrics"]["mean_intra_q"] for e in embeddings]
    intra_k = [e["orig_metrics"]["mean_intra_k"] for e in embeddings]
    inter = [e["orig_metrics"]["mean_inter_qk"] for e in embeddings]

    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, intra_q, w, color="#1976d2", alpha=0.85, label="Intra-Q")
    ax.bar(x, intra_k, w, color="#7b1fa2", alpha=0.85, label="Intra-K")
    ax.bar(x + w, inter, w, color="#8e5a2d", alpha=0.85, label="Inter Q-K")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"H{i}" for i in range(n)], fontsize=7,
    )
    ax.set_ylabel("Mean Distance")
    ax.set_title("Q-K Clusters (per head)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")


def _panel_multiples(gs_slot, fig, embeddings, proj_key,
                     coord_key, title_fn):
    """Small multiples of projections across heads."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    valid = [
        e for e in embeddings if coord_key in e
    ]
    if not valid:
        ax = fig.add_subplot(gs_slot)
        ax.text(
            0.5, 0.5, f"{proj_key} unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        return

    n = len(valid)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    inner = GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=gs_slot,
        hspace=0.3, wspace=0.3,
    )
    for i, emb in enumerate(valid):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(inner[r, c])
        _scatter_qk(
            ax, emb, proj_key, coord_key,
            title_fn(i, emb),
        )
        ax.set_xticks([])
        ax.set_yticks([])


# ═══════════════════════════════════════════════════════
# Dashboard assembly
# ═══════════════════════════════════════════════════════

def create_global_dashboard(
    data: Dict,
    title: str,
    out_path: Path,
):
    """
    Create global exploration dashboard.

    Layout (3 rows via GridSpec(3, 12)):
      Row 0: TopK mass | Entropy | Bias
      Row 1: Mean-Q vs Full | Query Deviation
      Row 2: PCA | t-SNE | UMAP | Embedding Stats
    """
    is_variance = "_all_heads_topk" in data.get(
        "concentration", {},
    )

    setup_style()
    fig = plt.figure(figsize=(28, 20))
    gs = GridSpec(
        3, 12, figure=fig,
        hspace=0.35, wspace=0.40,
    )

    # Row 0 — 3 panels
    ax = fig.add_subplot(gs[0, 0:4])
    _panel_topk_mass(ax, data["concentration"], is_variance)

    ax = fig.add_subplot(gs[0, 4:8])
    _panel_entropy(ax, data["entropy"], is_variance)

    ax = fig.add_subplot(gs[0, 8:12])
    _panel_bias(ax, data["bias"], is_variance)

    # Row 1 — 2 wide panels
    ax = fig.add_subplot(gs[1, 0:6])
    _panel_meanquery(ax, data["meanquery"], is_variance)

    ax = fig.add_subplot(gs[1, 6:12])
    _panel_query_deviation(
        ax, data["query_deviation"], is_variance,
    )

    # Row 2 — PCA | t-SNE | UMAP | Stats
    emb = data.get("embedding")

    if emb is not None and "_small_multiples" in emb:
        multiples = emb["_small_multiples"]

        _panel_multiples(
            gs[2, 0:3], fig, multiples,
            "pca", "pca_coords",
            lambda i, e: (
                f"H{i} "
                f"({e['pca_explained_var'][0]:.0%})"
            ),
        )
        _panel_multiples(
            gs[2, 3:6], fig, multiples,
            "tsne", "tsne_coords",
            lambda i, e: f"H{i}",
        )
        _panel_multiples(
            gs[2, 6:9], fig, multiples,
            "umap", "umap_coords",
            lambda i, e: f"H{i}",
        )

        ax = fig.add_subplot(gs[2, 9:12])
        _panel_embedding_stats_agg(ax, multiples)

    else:
        ax = fig.add_subplot(gs[2, 0:3])
        if emb is not None:
            ev = emb["pca_explained_var"]
            _panel_projection(
                ax, emb, "pca", "pca_coords",
                f"PCA ({ev[0]:.1%}+{ev[1]:.1%})",
            )
        else:
            _panel_projection(
                ax, None, "pca", "pca_coords", "PCA",
            )

        ax = fig.add_subplot(gs[2, 3:6])
        _panel_projection(
            ax, emb, "tsne", "tsne_coords",
            "t-SNE (local structure)",
        )

        ax = fig.add_subplot(gs[2, 6:9])
        _panel_projection(
            ax, emb, "umap", "umap_coords",
            "UMAP (global structure)",
        )

        ax = fig.add_subplot(gs[2, 9:12])
        _panel_embedding_stats(ax, emb)

    fig.suptitle(
        title, fontsize=16, fontweight="bold", y=0.98,
    )
    save_figure(fig, out_path, dpi=150)
