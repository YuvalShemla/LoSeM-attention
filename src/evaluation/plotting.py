"""
Plotting for attention evaluations.

Matches the style from evaluation 10 (math_calc_bootstrap):
  - Idealized: dashed lines (IdealTopK red, Sampling green,
    Equal Splits blue, Equal Weight Splits purple)
  - Algorithms: TopK (dashed) + Hybrid (solid) families
  - Color families: blue, orange, pink, gold
  - Log and linear scale versions
  - Shaded error bands for std across examples
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Dict, List, Optional


def setup_style():
    """Publication-quality matplotlib config."""
    sns.set_style("white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial", "DejaVu Sans", "Helvetica",
    ]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def save_figure(fig, path, dpi=200):
    """Save figure with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _format_log_axes(ax, budgets=None):
    """Format log-scale axes with readable labels."""
    if budgets:
        from matplotlib.ticker import FixedLocator
        ax.xaxis.set_major_locator(
            FixedLocator(budgets)
        )
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: (
            f"{int(x)}" if x >= 1 else f"{x:.1f}"
        )
    ))
    ax.xaxis.set_minor_formatter(
        FuncFormatter(lambda x, _: "")
    )
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: (
            f"{y:.4f}" if y < 0.01
            else f"{y:.3f}" if y < 0.1
            else f"{y:.2f}"
        )
    ))


def _plot_with_error_band(
    ax, x, y_mean, y_std, color, **plot_kwargs,
):
    """Plot line with shaded std error band."""
    line = ax.plot(x, y_mean, color=color,
                   **plot_kwargs)
    if y_std is not None and len(y_std) > 0:
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)
        ax.fill_between(
            x,
            np.maximum(y_mean - y_std, 1e-10),
            y_mean + y_std,
            color=color, alpha=0.15,
        )
    return line


def plot_idealized_methods(
    ax: plt.Axes,
    agg: Dict,
    budgets: List[int],
    plot_cfg: Dict,
):
    """Plot idealized method curves."""
    colors = plot_cfg.get("idealized_colors", {})
    show_bands = plot_cfg.get("error_bands", True)

    idealized_specs = [
        ("IdealTopK", "IdealTopK (individual keys)"),
        ("IdealSampling", "IdealSampling"),
        ("IdealEqualSplits", "IdealEqualSplits"),
        ("IdealEqualWeightSplits", "IdealEqualWeightSplits"),
    ]
    default_colors = {
        "IdealTopK": "#d62728",
        "IdealSampling": "#2ca02c",
        "IdealEqualSplits": "#1f77b4",
        "IdealEqualWeightSplits": "#9467bd",
    }

    for method_name, label in idealized_specs:
        x, y, s = [], [], []
        for b in budgets:
            k = f"{method_name}-{b}"
            if k in agg:
                x.append(agg[k]["budget_mean"])
                y.append(agg[k]["error_mean"])
                s.append(agg[k].get("error_std", 0))
        if x:
            _plot_with_error_band(
                ax, x, y, s if show_bands else None,
                color=colors.get(
                    method_name,
                    default_colors[method_name],
                ),
                ls="--", marker="o", lw=2.5, ms=7,
                zorder=4,
                label=label,
            )


def plot_algorithm_family(
    ax: plt.Axes,
    agg: Dict,
    prefix: str,
    label: str,
    color_topk: str,
    color_hybrid: str,
    marker: str,
    top_k_sweep: List[int],
    show_bands: bool = True,
    annotate_ks: Optional[set] = None,
):
    """Plot TopK (dashed) + Hybrid (solid) curves."""
    if annotate_ks is None:
        annotate_ks = {1, 5, 10}

    # TopK curve (dashed, skip k=0)
    x_tk, y_tk, s_tk, vals = [], [], [], []
    for tk in top_k_sweep:
        if tk == 0:
            continue
        k = f"{prefix}-topk-k{tk}"
        if k in agg:
            x_tk.append(agg[k]["budget_mean"])
            y_tk.append(agg[k]["error_mean"])
            s_tk.append(agg[k].get("error_std", 0))
            vals.append(tk)
    if x_tk:
        _plot_with_error_band(
            ax, x_tk, y_tk,
            s_tk if show_bands else None,
            color=color_topk,
            marker=marker, ls="--",
            lw=2.2, ms=7, alpha=0.85, zorder=5,
            label=f"{label} TopK",
        )
        for tk, xv, yv in zip(vals, x_tk, y_tk):
            if tk in annotate_ks:
                ax.annotate(
                    f"k={tk}", xy=(xv, yv),
                    fontsize=7, color=color_topk,
                    xytext=(3, 4),
                    textcoords="offset points",
                )

    # Hybrid curve (solid)
    x_hy, y_hy, s_hy, vals = [], [], [], []
    for tk in top_k_sweep:
        k = f"{prefix}-hybrid-k{tk}"
        if k in agg:
            x_hy.append(agg[k]["budget_mean"])
            y_hy.append(agg[k]["error_mean"])
            s_hy.append(agg[k].get("error_std", 0))
            vals.append(tk)
    if x_hy:
        _plot_with_error_band(
            ax, x_hy, y_hy,
            s_hy if show_bands else None,
            color=color_hybrid,
            marker=marker, ls="-",
            lw=2.8, ms=7, alpha=0.9, zorder=5,
            label=f"{label} Hybrid",
        )
        for tk, xv, yv in zip(vals, x_hy, y_hy):
            if tk in annotate_ks:
                ax.annotate(
                    f"k={tk}", xy=(xv, yv),
                    fontsize=7, color=color_hybrid,
                    xytext=(3, -9),
                    textcoords="offset points",
                )


def plot_evaluation(
    agg: Dict,
    out_dir: Path,
    plot_cfg: Dict,
    budgets: List[int],
    algorithm_families: List[Dict],
    title: str = "",
    filename: str = "results",
    n_queries: int = 0,
):
    """
    Generate log + linear scale plots.

    algorithm_families: list of dicts with keys
      prefix, label, color_topk, color_hybrid,
      marker, top_k_sweep.
    """
    setup_style()
    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)
    show_bands = plot_cfg.get("error_bands", True)

    scales = []
    if plot_cfg.get("log_scale", True):
        scales.append(True)
    if plot_cfg.get("linear_scale", True):
        scales.append(False)

    for log_scale in scales:
        scale = "log" if log_scale else "linear"
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_idealized_methods(ax, agg, budgets, plot_cfg)

        for fam in algorithm_families:
            plot_algorithm_family(
                ax, agg,
                prefix=fam["prefix"],
                label=fam["label"],
                color_topk=fam["color_topk"],
                color_hybrid=fam["color_hybrid"],
                marker=fam["marker"],
                top_k_sweep=fam["top_k_sweep"],
                show_bands=show_bands,
            )

        ax.set_xlabel(
            "Effective Budget "
            "(# items in final softmax)",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel(
            "Mean Relative L2 Error",
            fontsize=12, fontweight="bold",
        )

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _format_log_axes(ax, budgets)

        subtitle = (
            f"{n_queries} queries" if n_queries
            else ""
        )
        if title:
            full_title = f"{title}\n{subtitle}"
        else:
            full_title = subtitle
        if full_title.strip():
            ax.set_title(
                full_title, fontsize=13,
                fontweight="bold",
            )

        ax.legend(
            fontsize=8, loc="upper right", ncol=2,
        )

        plt.tight_layout()
        fname = f"{filename}_{scale}.png"
        save_figure(fig, out_dir / fname, dpi=dpi)


def plot_overview(
    per_task_agg: Dict[str, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    budgets: List[int],
    algorithm_families: List[Dict],
    task_seq_info: Dict[str, str] = None,
):
    """
    Cross-task summary plots.

    One subplot per task, shared y-axis, showing
    idealized methods + algorithms for quick comparison.
    """
    setup_style()
    tasks = list(per_task_agg.keys())
    n = len(tasks)
    if n == 0:
        return

    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)
    show_bands = plot_cfg.get("error_bands", True)

    scales = []
    if plot_cfg.get("log_scale", True):
        scales.append(True)
    if plot_cfg.get("linear_scale", True):
        scales.append(False)

    cols = min(n, 3)
    rows_n = (n + cols - 1) // cols

    for log_scale in scales:
        scale = "log" if log_scale else "linear"
        fig, axes = plt.subplots(
            rows_n, cols,
            figsize=(figsize[0], figsize[1] * rows_n / 2),
            squeeze=False,
        )

        for i, task in enumerate(tasks):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            agg = per_task_agg[task]
            plot_idealized_methods(ax, agg, budgets, plot_cfg)
            for fam in algorithm_families:
                plot_algorithm_family(
                    ax, agg,
                    prefix=fam["prefix"],
                    label=fam["label"],
                    color_topk=fam["color_topk"],
                    color_hybrid=fam["color_hybrid"],
                    marker=fam["marker"],
                    top_k_sweep=fam["top_k_sweep"],
                    show_bands=show_bands,
                )
            ttl = task
            if task_seq_info and task in task_seq_info:
                ttl += f"\n{task_seq_info[task]}"
            ax.set_title(ttl, fontsize=11)
            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")
                _format_log_axes(ax, budgets)
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")

        for i in range(n, rows_n * cols):
            r, c = divmod(i, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            f"Cross-Task Summary ({scale})",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        save_figure(
            fig,
            out_dir / f"cross_task_summary_{scale}.png",
            dpi=dpi,
        )


def _build_info_panel(ax, per_head_aggs, sorted_idxs,
                      task_name):
    """Fill a spare subplot with legend + head table."""
    ax.axis("off")

    # Collect legend handles from sibling axes
    fig = ax.get_figure()
    handles, labels = [], []
    seen = set()
    for other in fig.axes:
        if other is ax:
            continue
        for h, l in zip(*other.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)

    if handles:
        ax.legend(
            handles, labels,
            loc="upper left",
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=False,
            borderpad=1.0,
            labelspacing=0.8,
            title="Methods",
            title_fontsize=10,
        )

    # Head summary table below the legend
    lines = []
    for idx in sorted_idxs:
        info = per_head_aggs[idx]
        tag = f"L{info['layer']}H{info['q_head']}"
        lbl = info.get("selection_label", "")
        ent = info.get("effective_entropy")
        nq = info.get("n_queries", 0)
        parts = [tag]
        if lbl:
            parts.append(lbl)
        if ent is not None:
            parts.append(f"ent={ent:.2f}")
        parts.append(f"n={nq}")
        lines.append("  ".join(parts))

    if lines:
        table_text = "Heads:\n" + "\n".join(lines)
        ax.text(
            0.03, 0.02, table_text,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="bottom",
        )


def plot_per_head_comparison(
    per_head_aggs: Dict[int, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    budgets: List[int],
    algorithm_families: List[Dict],
    task_name: str = "",
    seq_desc: str = "",
):
    """
    Per-head subplot comparison.

    per_head_aggs: {head_idx: {agg, layer, q_head,
        selection_label, effective_entropy, ...}}
    Uses spare subplot cells for a legend + info panel.
    """
    setup_style()
    n = len(per_head_aggs)
    if n == 0:
        return

    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)
    show_bands = plot_cfg.get("error_bands", True)

    cols = min(n, 3)
    rows_n = (n + cols - 1) // cols
    # Ensure at least one spare cell for info panel
    if n == rows_n * cols:
        rows_n += 1

    scales = []
    if plot_cfg.get("log_scale", True):
        scales.append(True)
    if plot_cfg.get("linear_scale", True):
        scales.append(False)

    sorted_idxs = sorted(per_head_aggs.keys())

    for log_scale in scales:
        scale = "log" if log_scale else "linear"
        fig, axes = plt.subplots(
            rows_n, cols,
            figsize=(
                figsize[0],
                figsize[1] * rows_n / 2,
            ),
            squeeze=False,
        )

        for i, idx in enumerate(sorted_idxs):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            info = per_head_aggs[idx]
            agg = info["agg"]

            plot_idealized_methods(
                ax, agg, budgets, plot_cfg,
            )
            for fam in algorithm_families:
                plot_algorithm_family(
                    ax, agg,
                    prefix=fam["prefix"],
                    label=fam["label"],
                    color_topk=fam["color_topk"],
                    color_hybrid=fam["color_hybrid"],
                    marker=fam["marker"],
                    top_k_sweep=fam["top_k_sweep"],
                    show_bands=show_bands,
                )

            title = (
                f"L{info['layer']}H{info['q_head']}"
            )
            lbl = info.get("selection_label", "")
            ent = info.get("effective_entropy")
            if lbl:
                title += f" ({lbl}"
                if ent is not None:
                    title += f", ent={ent:.2f}"
                title += ")"
            elif ent is not None:
                title += f" (ent={ent:.2f})"
            ax.set_title(title, fontsize=10)

            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")
                _format_log_axes(ax, budgets)

        # Use the first spare cell for info panel,
        # hide remaining spare cells
        spare_start = n
        info_placed = False
        for i in range(spare_start, rows_n * cols):
            r, c = divmod(i, cols)
            if not info_placed:
                _build_info_panel(
                    axes[r][c], per_head_aggs,
                    sorted_idxs, task_name,
                )
                info_placed = True
            else:
                axes[r][c].set_visible(False)

        suptitle = "Per-Head Comparison"
        if task_name:
            suptitle = f"{task_name} — {suptitle}"
        if seq_desc:
            suptitle += f" — {seq_desc}"
        suptitle += f" ({scale})"
        fig.suptitle(
            suptitle, fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(
            fig,
            out_dir
            / f"per_head_comparison_{scale}.png",
            dpi=dpi,
        )
