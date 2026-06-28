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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from typing import Dict, List, Optional


def _order_group_cosine_table_columns(
    columns: List[Dict],
) -> List[Dict]:
    """
    Same column order as aggregate_group_cosines_by_head_group: low
    effective_entropy to high, then layer/head, then n_groups.
    Used so replayed JSON and plots stay consistent with per-head
    comparison ordering.
    """

    def _col_key(col: Dict) -> tuple:
        ent = col.get("effective_entropy")
        if ent is None:
            e = float("inf")
        else:
            try:
                e = float(ent)
                if not np.isfinite(e):
                    e = float("inf")
            except (TypeError, ValueError):
                e = float("inf")
        h = col.get("head", "")
        try:
            p = h.split("(")[0]
            layer = int(p.split("H")[0][1:])
            qh = int(p.split("H")[1])
            ht = (layer, qh, h)
        except Exception:
            ht = (10**9, 10**9, h)
        return (e, ht, int(col.get("n_groups", 0)))

    return sorted(columns, key=_col_key)


def _format_exp_l1_over_z(x: float) -> str:
    """
    Format L1(exp residual)/Z for annotations: enough digits that small
    positives are visible; scientific notation when |x| is very small.
    """
    xf = float(x)
    if not np.isfinite(xf):
        return str(xf)
    ax = abs(xf)
    if ax == 0.0:
        return "0"
    if ax < 0.01:
        return f"{xf:.4e}"
    return f"{xf:.8f}"


def _group_l1_cgm_over_z_quantiles_line(meta: Optional[Dict]) -> str:
    """
    One-line summary of pooled m·c_g quantiles (all groups, all queries).
    """
    if not meta:
        return ""
    keys = ("p50", "p75", "p90")
    if not any(k in meta for k in keys):
        return ""
    parts = []
    for k in keys:
        if k in meta:
            parts.append(
                f"{k.upper()}={_format_exp_l1_over_z(float(meta[k]))}",
            )
    return "m·c_g: " + " ".join(parts)


def _group_l1_znorm_quantiles_line(meta: Optional[Dict]) -> str:
    """
    One-line summary of pooled m·c_g (Z-norm) quantiles.
    c_g here uses Z instead of Z' for the group probability.
    """
    if not meta:
        return ""
    keys = ("p50", "p75", "p90")
    if not any(k in meta for k in keys):
        return ""
    parts = []
    for k in keys:
        if k in meta:
            parts.append(
                f"{k.upper()}={_format_exp_l1_over_z(float(meta[k]))}",
            )
    return "m·c_g(Z): " + " ".join(parts)


def _overlay_p75_z_norm_histogram(ax, entry: Dict) -> None:
    """
    Overlay density histogram of e^{l_i} * m / Z for tokens i in the
    group with maximal e^{l_g} * m / Z, where l_g is the group mean logit.
    Drawn on a twin top x-axis (cosines use bottom).
    """
    mh_tok = entry.get("max_group_exp_logits_histogram")
    if not mh_tok:
        mh_tok = entry.get("p75_z_norm_histogram")
    if not mh_tok:
        mh_tok = entry.get("median_z_norm_histogram")
    if not mh_tok:
        return
    edges = np.array(mh_tok.get("bin_edges", []), dtype=np.float64)
    counts = np.array(mh_tok.get("counts", []), dtype=np.float64)
    if len(edges) < 2 or len(counts) != len(edges) - 1:
        return
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    total = max(float(np.sum(counts)), 1.0)
    density = counts / total
    ax2 = ax.twiny()
    ax2.bar(
        centers,
        density,
        width=widths,
        align="center",
        alpha=0.42,
        edgecolor="none",
        color="tab:orange",
        zorder=2,
    )
    mh_lg = entry.get("max_group_exp_lg_histogram")
    if mh_lg:
        lg_edges = np.array(
            mh_lg.get("bin_edges", []), dtype=np.float64
        )
        lg_counts = np.array(
            mh_lg.get("counts", []), dtype=np.float64
        )
        if (
            len(lg_edges) == len(edges)
            and len(lg_counts) == len(counts)
        ):
            lg_total = max(float(np.sum(lg_counts)), 1.0)
            lg_density = lg_counts / lg_total
            ax2.bar(
                centers,
                lg_density,
                width=0.55 * widths,
                align="center",
                alpha=0.78,
                facecolor="none",
                edgecolor="#c76a00",
                linewidth=0.9,
                zorder=2.6,
            )
    ax2.set_xlim(edges[0], edges[-1])
    ax2.set_xlabel(
        r"max-$e^{\ell_g}$ group: $e^{\ell_i}\,m/Z$",
        fontsize=7,
        labelpad=2,
        color="tab:orange",
    )
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-2, 2))
    ax2.xaxis.set_major_formatter(sf)
    ax2.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(-2, 2),
    )
    ax2.tick_params(
        axis="x",
        labelsize=6,
        pad=1,
        colors="tab:orange",
    )
    # Keep scientific multiplier (offset text) visually consistent.
    off = ax2.xaxis.get_offset_text()
    off.set_color("tab:orange")
    off.set_fontsize(6)


def _overlay_group_l1_contrib_histogram(ax, entry: Dict) -> None:
    """
    Cumulative curves over thresholds x for c_g*m:
      - error-share curve: share of total L1 contribution from groups
        with c_g*m <= x
      - group-share curve: share of groups with c_g*m <= x
    Twin x-axis on a lower horizontal spine.
    """
    gh = entry.get("group_l1_contrib_histogram")
    if not gh:
        return
    edges = np.array(gh.get("bin_edges", []), dtype=np.float64)
    g_counts = np.array(gh.get("counts", []), dtype=np.float64)
    cum = entry.get("group_l1_contrib_cumulative") or {}
    g_cum = np.array(cum.get("cum_share", []), dtype=np.float64)
    if (
        len(edges) < 2
        or len(g_cum) != len(edges) - 1
        or len(g_counts) != len(edges) - 1
    ):
        return
    g_count_total = float(np.sum(g_counts))
    if g_count_total > 0.0:
        g_count_cum = np.cumsum(g_counts) / g_count_total
    else:
        g_count_cum = np.zeros_like(g_counts)
    meta = entry.get("group_l1_hist_meta") or {}
    use_log = bool(meta.get("log_scale_x"))
    x_right = edges[1:]
    ax3 = ax.twiny()
    ax3.patch.set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")
    ax3.spines["bottom"].set_position(("axes", 0.14))
    if use_log:
        ax3.set_xscale("log")
    ax3.plot(
        x_right,
        g_cum,
        color="tab:green",
        lw=1.5,
        alpha=0.9,
        zorder=4,
    )
    ax3.fill_between(
        x_right,
        0.0,
        g_cum,
        step="pre",
        alpha=0.18,
        edgecolor="none",
        color="tab:green",
        zorder=3,
    )
    ax3.plot(
        x_right,
        g_count_cum,
        color="tab:green",
        lw=1.35,
        alpha=0.95,
        linestyle="--",
        zorder=5,
    )
    ax3.set_ylim(0.0, 1.02)
    ax3.set_xlim(edges[0], edges[-1])
    ax3.tick_params(axis="x", labelsize=5, pad=1)
    _xl = (
        r"$c_g\,m$ threshold $x$: cumulative $L_1$ contribution "
        r"(solid) and cumulative group share (dashed)"
    )
    if use_log:
        _xl += r"; $\log_{10}$-spaced bins; log $x$; $x \geq 10^{-6}$"
    elif meta.get("display_cap_quantile") is not None:
        dq = float(meta.get("display_cap_quantile", 90.0))
        cap_v = float(meta.get("display_cap_value", 0.0))
        cap_s = _format_exp_l1_over_z(cap_v)
        _xl += (
            rf"; $x$: $[\min,\,P_{{{dq:.0f}}}]$ + overflow "
            rf"($P_{{{dq:.0f}}}\!\approx\!{cap_s}$)"
        )
    ax3.set_xlabel(_xl, fontsize=6, labelpad=4)
    if meta and not use_log:
        ova = int(meta.get("overflow_n_all_g", 0))
        if ova > 0:
            ax.text(
                0.99,
                0.03,
                f"Overflow bin (all-g): {ova}",
                transform=ax.transAxes,
                fontsize=5,
                ha="right",
                va="bottom",
                color="#2e2e2e",
                zorder=22,
            )


from .caption import format_eval_config_caption, format_probe_method_variants
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
    """Save figure with tight layout, capping pixel size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cap figure so pixel dimensions stay under 2^16
    max_px = 65000
    w_in, h_in = fig.get_size_inches()
    if h_in * dpi > max_px:
        fig.set_size_inches(w_in, max_px / dpi)
    if w_in * dpi > max_px:
        fig.set_size_inches(max_px / dpi, fig.get_size_inches()[1])
    try:
        fig.savefig(
            path, dpi=dpi, bbox_inches="tight",
            facecolor="white",
        )
    except ValueError:
        try:
            fig.savefig(
                path, dpi=dpi, facecolor="white",
            )
        except ValueError:
            fig.savefig(
                path, dpi=max(50, dpi // 2),
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
    skip_prefixes: Optional[set] = None,
):
    """Plot idealized method curves."""
    colors = plot_cfg.get("idealized_colors", {})
    show_bands = plot_cfg.get("error_bands", True)

    idealized_specs = [
        ("IdealTopK", "Ideal TopK"),
        ("IdealSampling-Subset", "Ideal Sampling (Subset)"),
        ("IdealSampling-IS", "Ideal Sampling (IS)"),
        ("IdealTopK+Uniform", "Ideal TopK+Uniform"),
        ("vAttention(oracle)", "vAttention(oracle)"),
        ("IdealEqualWeightSplits",
         "Ideal Equal Weight Splits"),
        ("EWS+NoiseK", "EWS+NoiseK"),
        ("EWS+NoiseV", "EWS+NoiseV"),
        ("EWS+NoiseKV", "EWS+NoiseKV"),
        ("EWS+NoiseKV-G0", "EWS+NoiseKV (group 0)"),
        ("EWS+NoiseKV-Hi", "EWS+NoiseKV (top half)"),
        ("EWS+NoiseKV-Lo", "EWS+NoiseKV (bottom half)"),
    ]
    default_colors = {
        "IdealTopK": "#d62728",
        "IdealSampling-Subset": "#2ca02c",
        "IdealSampling-IS": "#17becf",
        "IdealTopK+Uniform": "#ff6f00",
        "vAttention(oracle)": "#ff1493",  # hot pink
        "IdealEqualSplits": "#1f77b4",
        "IdealEqualWeightSplits": "#9467bd",
        "EWS+NoiseK": "#00bfff",
        "EWS+NoiseV": "#ff8c00",
        "EWS+NoiseKV": "#e377c2",
        "EWS+NoiseKV-G0": "#8c564b",
        "EWS+NoiseKV-Hi": "#17becf",
        "EWS+NoiseKV-Lo": "#bcbd22",
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
                ls="--", marker="o", lw=1.5, ms=5,
                zorder=3,
                label=label,
            )

    # Budget-sweeping algorithms (solid lines, scatter)
    budget_sweep_colors = plot_cfg.get(
        "budget_sweep_colors", {},
    )
    seen_prefixes = set()
    for key in sorted(agg.keys()):
        if key.startswith("_"):
            continue
        # Match keys like "HierKM-32x16x8-0-256"
        # that aren't idealized or topk/hybrid format
        parts = key.rsplit("-", 1)
        if len(parts) != 2:
            continue
        prefix, maybe_budget = parts
        if not maybe_budget.isdigit():
            continue
        # Skip idealized methods already plotted
        if any(
            prefix == spec[0]
            for spec in idealized_specs
        ):
            continue
        # Skip methods with point labels (plotted
        # by plot_algorithm_family as scatter)
        if agg[key].get("point_label"):
            continue
        if prefix in seen_prefixes:
            continue
        if skip_prefixes and prefix in skip_prefixes:
            continue
        # Collect all budget points for this prefix
        x, y, s = [], [], []
        for b in budgets:
            k = f"{prefix}-{b}"
            if k in agg:
                x.append(agg[k]["budget_mean"])
                y.append(agg[k]["error_mean"])
                s.append(agg[k].get("error_std", 0))
        if x:
            seen_prefixes.add(prefix)
            color = budget_sweep_colors.get(
                prefix, colors.get(prefix, None),
            )
            if color is None:
                # Auto-assign from a palette
                palette = [
                    "#1f77b4", "#ff7f0e", "#2ca02c",
                    "#d62728", "#9467bd", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22",
                    "#17becf",
                ]
                idx = len(seen_prefixes) % len(palette)
                color = palette[idx]
            _plot_with_error_band(
                ax, x, y,
                s if show_bands else None,
                color=color,
                ls="-", marker="D", lw=2, ms=6,
                zorder=5,
                label=prefix,
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

    # Fallback for algorithms that don't use topk/hybrid naming.
    # Handles both budget-sweep keys ({prefix}-{int}) and
    # single-point keys ({prefix} exactly).
    if not x_tk and not x_hy:
        import re
        pat = re.compile(
            rf"^{re.escape(prefix)}-(\d+)$",
        )
        bx, by, bs, pt_labels = [], [], [], []
        has_pt_labels = False
        has_hlines = False
        for k in sorted(
            agg.keys(),
            key=lambda k: agg[k].get(
                "budget_mean", 0,
            ),
        ):
            if pat.match(k):
                e = agg[k]
                pl = e.get("point_label", "")
                if e.get("horizontal_line"):
                    has_hlines = True
                if pl:
                    has_pt_labels = True
                    if (not has_hlines
                            and e["budget_mean"] > 2500):
                        continue
                bx.append(e["budget_mean"])
                by.append(e["error_mean"])
                bs.append(e.get("error_std", 0))
                pt_labels.append(pl)

        # Horizontal lines mode: dashed lines spanning
        # the full plot width, labeled on the right edge.
        if has_hlines and by:
            n = len(by)
            alphas = np.linspace(0.35, 0.75, max(n, 1))
            for i, (yv, pl) in enumerate(
                zip(by, pt_labels),
            ):
                lbl = (
                    f"{label} (K={pl})" if i == 0
                    else f"  K={pl}"
                )
                ax.axhline(
                    yv, color=color_hybrid,
                    ls="--", lw=1.5,
                    alpha=float(alphas[i]),
                    zorder=4, label=lbl,
                )
                ax.annotate(
                    f"K={pl}",
                    xy=(0.98, yv),
                    xycoords=(
                        "axes fraction", "data",
                    ),
                    fontsize=8, color=color_hybrid,
                    va="bottom", ha="right",
                    fontweight="bold",
                )
        elif bx:
            leg_label = (
                f"{label} (C/L)" if has_pt_labels
                else label
            )
            if has_pt_labels:
                ax.scatter(
                    bx, by,
                    color=color_hybrid,
                    marker=marker, s=50,
                    alpha=0.9, zorder=8,
                    label=leg_label,
                )
            else:
                _plot_with_error_band(
                    ax, bx, by,
                    bs if show_bands else None,
                    color=color_hybrid,
                    marker=marker, ls="-",
                    lw=1.5, ms=5, alpha=0.9,
                    zorder=5, label=leg_label,
                )
            for xv, yv, pl in zip(
                bx, by, pt_labels,
            ):
                if pl:
                    ax.annotate(
                        pl, xy=(xv, yv),
                        fontsize=7,
                        fontweight="bold",
                        color=color_hybrid,
                        xytext=(6, 6),
                        textcoords="offset points",
                        zorder=10,
                    )
        elif prefix in agg:
            e = agg[prefix]
            bm = e["budget_mean"]
            em = e["error_mean"]
            es = e.get("error_std", 0)
            pl = e.get("point_label", "")
            yerr = (
                [es] if show_bands and es > 0
                else None
            )
            ax.errorbar(
                [bm], [em], yerr=yerr,
                color=color_hybrid,
                marker=marker, ls="none",
                capsize=4, lw=2.0, ms=9,
                zorder=6, label=label,
            )
            if pl:
                ax.annotate(
                    pl, xy=(bm, em),
                    fontsize=5,
                    color=color_hybrid,
                    xytext=(4, 3),
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
    config_caption: str = "",
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
    max_scatter_points = int(
        plot_cfg.get("fullattention_pq_topk_scatter_max_points", 20000)
    )
    max_scatter_points = int(
        plot_cfg.get("fullattention_pq_topk_scatter_max_points", 20000)
    )
    max_scatter_points = int(
        plot_cfg.get("fullattention_pq_topk_scatter_max_points", 20000)
    )
    max_scatter_points = int(
        plot_cfg.get("fullattention_pq_topk_scatter_max_points", 20000)
    )
    show_bands = plot_cfg.get("error_bands", True)

    scales = []
    if plot_cfg.get("log_scale", True):
        scales.append(True)
    if plot_cfg.get("linear_scale", True):
        scales.append(False)

    for log_scale in scales:
        scale = "log" if log_scale else "linear"
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        algo_prefixes = {
            fam["prefix"] for fam in algorithm_families
        }
        plot_idealized_methods(
            ax, agg, budgets, plot_cfg,
            skip_prefixes=algo_prefixes,
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
        parts = []
        if title:
            parts.append(title)
        if config_caption:
            parts.append(config_caption)
        if subtitle:
            parts.append(subtitle)
        full_title = "\n".join(parts)
        if full_title.strip():
            ax.set_title(
                full_title, fontsize=13,
                fontweight="bold",
            )

        ax.legend(
            fontsize=8, loc="upper right", ncol=1,
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
    config_caption: str = "",
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

        algo_prefixes = {
            fam["prefix"] for fam in algorithm_families
        }
        for i, task in enumerate(tasks):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            agg = per_task_agg[task]
            plot_idealized_methods(
                ax, agg, budgets, plot_cfg,
                skip_prefixes=algo_prefixes,
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

        st = f"Cross-Task Summary ({scale})"
        if config_caption:
            st = f"{st}\n{config_caption}"
        fig.suptitle(
            st, fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        save_figure(
            fig,
            out_dir / f"cross_task_summary_{scale}.png",
            dpi=dpi,
        )


def _build_info_panel(
    ax,
    per_head_aggs,
    sorted_idxs,
    task_name,
    variant_caption: str = "",
):
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

    body_parts = []
    if variant_caption.strip():
        body_parts.append(variant_caption.strip())
    if lines:
        body_parts.append("Heads:\n" + "\n".join(lines))
    if body_parts:
        ax.text(
            0.03, 0.02, "\n\n".join(body_parts),
            transform=ax.transAxes,
            fontsize=8,
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
    config_caption: str = "",
    variant_caption: str = "",
):
    """
    Per-head subplot comparison.

    per_head_aggs: {head_idx: {agg, layer, q_head,
        selection_label, effective_entropy, ...}}
    Uses spare subplot cells for a legend + info panel.
    """
    if not variant_caption and config_caption and "\n" in config_caption:
        _, _, variant_caption = config_caption.partition("\n")
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

    def _per_head_sort_key(i: int) -> tuple:
        info = per_head_aggs[i]
        ent = info.get("effective_entropy")
        if ent is None:
            e = float("inf")
        else:
            try:
                e = float(ent)
                if not np.isfinite(e):
                    e = float("inf")
            except (TypeError, ValueError):
                e = float("inf")
        return (
            e,
            info.get("layer", 10**9),
            info.get("q_head", 10**9),
        )

    sorted_idxs = sorted(
        per_head_aggs.keys(),
        key=_per_head_sort_key,
    )

    for log_scale in scales:
        scale = "log" if log_scale else "linear"
        fig, axes = plt.subplots(
            rows_n, cols,
            figsize=(
                figsize[0],
                min(figsize[1] * rows_n / 2, 50),
            ),
            squeeze=False,
        )

        algo_prefixes = {
            fam["prefix"] for fam in algorithm_families
        }
        for i, idx in enumerate(sorted_idxs):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            info = per_head_aggs[idx]
            agg = info["agg"]

            plot_idealized_methods(
                ax, agg, budgets, plot_cfg,
                skip_prefixes=algo_prefixes,
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
                    variant_caption=variant_caption,
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
        if config_caption:
            suptitle = f"{suptitle}\n{config_caption}"
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


def _probe_method_prefix(method_key: str) -> str:
    """``Learned-kmeans-1024`` -> ``Learned-kmeans``; ``TFCFW-lq-4096`` -> ``TFCFW-lq``."""
    return method_key.rsplit("-", 1)[0]


def _probe_method_color_key(method_prefix: str) -> str:
    if method_prefix.startswith("Learned"):
        return "learned"
    if method_prefix.startswith("TFCFW-lq"):
        return "tensor_fcfw_lq"
    return method_prefix.lower().replace("-", "_")


def _probe_method_budget_points(
    agg: Dict,
    method_prefix: str,
    budgets: List[int],
) -> tuple:
    """Extract (x, y, y_std) for a method family across requested budgets."""
    x_vals, y_vals, s_vals = [], [], []
    for b in budgets:
        key = f"{method_prefix}-{b}"
        if key not in agg:
            continue
        entry = agg[key]
        x_vals.append(entry["budget_mean"])
        y_vals.append(entry["error_mean"])
        s_vals.append(
            entry.get(
                "probe_error_std_mean",
                entry.get("error_std", 0.0),
            ),
        )
    return x_vals, y_vals, s_vals


def plot_probe_training_error(
    per_head_probe_aggs: Dict[int, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    budgets: List[int],
    task_name: str = "",
    seq_desc: str = "",
    config_caption: str = "",
):
    """
    Per-head probe mean rel-L2 vs budget for probe-Q methods.

    Solid lines: mean rel-L2 over the training probe set ``Q`` (same metric as
    eval). Dashed overlays: held-out test-query eval error from the main run.
    """
    setup_style()
    if not per_head_probe_aggs:
        return

    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)
    show_bands = plot_cfg.get("error_bands", True)
    algo_colors = plot_cfg.get("algorithm_colors", {})

    n = len(per_head_probe_aggs)
    cols = min(n, 3)
    rows_n = (n + cols - 1) // cols
    if n == rows_n * cols:
        rows_n += 1

    scales = []
    if plot_cfg.get("log_scale", True):
        scales.append(True)
    if plot_cfg.get("linear_scale", True):
        scales.append(False)

    def _per_head_sort_key(i: int) -> tuple:
        info = per_head_probe_aggs[i]
        ent = info.get("effective_entropy")
        if ent is None:
            e = float("inf")
        else:
            try:
                e = float(ent)
                if not np.isfinite(e):
                    e = float("inf")
            except (TypeError, ValueError):
                e = float("inf")
        return (
            e,
            info.get("layer", 10**9),
            info.get("q_head", 10**9),
        )

    sorted_idxs = sorted(
        per_head_probe_aggs.keys(),
        key=_per_head_sort_key,
    )

    for log_scale in scales:
        scale = "log" if log_scale else "linear"
        fig, axes = plt.subplots(
            rows_n, cols,
            figsize=(
                figsize[0],
                min(figsize[1] * rows_n / 2, 50),
            ),
            squeeze=False,
        )

        for i, idx in enumerate(sorted_idxs):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            info = per_head_probe_aggs[idx]
            probe_agg = info.get("probe_agg", {})
            test_agg = info.get("test_agg", {})

            prefixes = sorted({
                _probe_method_prefix(k)
                for k in probe_agg
            })
            for prefix in prefixes:
                ck = _probe_method_color_key(prefix)
                fam = algo_colors.get(ck, {})
                color = fam.get("hybrid", fam.get("topk", "#333333"))
                marker = fam.get("marker", "o")

                x_p, y_p, s_p = _probe_method_budget_points(
                    probe_agg, prefix, budgets,
                )
                if x_p:
                    _plot_with_error_band(
                        ax, x_p, y_p,
                        s_p if show_bands else None,
                        color=color,
                        ls="-", marker=marker,
                        lw=2.2, ms=7, zorder=5,
                        label=f"{prefix} (probe)",
                    )

                x_t, y_t, s_t = _probe_method_budget_points(
                    test_agg, prefix, budgets,
                )
                if x_t:
                    _plot_with_error_band(
                        ax, x_t, y_t,
                        s_t if show_bands else None,
                        color=color,
                        ls="--", marker=marker,
                        lw=2.0, ms=6, alpha=0.85, zorder=4,
                        label=f"{prefix} (test)",
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
            n_probes = info.get("n_probes")
            if n_probes:
                title += f"\n|Q|={n_probes:,}"
            ax.set_title(title, fontsize=10)

            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")
                _format_log_axes(ax, budgets)
            ax.set_xlabel("Budget")
            ax.set_ylabel("rel-L2 error")
            ax.legend(fontsize=7, loc="best")

        spare_start = n
        info_placed = False
        for j in range(spare_start, rows_n * cols):
            r, c = divmod(j, cols)
            if not info_placed:
                ax_info = axes[r][c]
                ax_info.axis("off")
                lines = [
                    "Probe training error",
                    "",
                    "Solid: mean rel-L2 over probe set Q",
                    "(same metric as eval).",
                    "",
                    "Dashed: held-out test-query",
                    "eval error (main run).",
                ]
                ax_info.text(
                    0.02, 0.98, "\n".join(lines),
                    transform=ax_info.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    family="monospace",
                )
                info_placed = True
            else:
                axes[r][c].set_visible(False)

        suptitle = "Probe Training Error vs Budget"
        if task_name:
            suptitle = f"{task_name} — {suptitle}"
        if seq_desc:
            suptitle += f" — {seq_desc}"
        suptitle += f" ({scale})"
        if config_caption:
            suptitle = f"{suptitle}\n{config_caption}"
        fig.suptitle(
            suptitle, fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(
            fig,
            out_dir
            / f"probe_training_error_{scale}.png",
            dpi=dpi,
        )


def plot_fullattention_pq_topk_profiles(
    per_head_profiles: Dict[int, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    task_name: str = "",
    seq_desc: str = "",
    config_caption: str = "",
):
    """
    For each requested budget B, plot per-head curves of true p_i vs
    PQ-estimated p_i for FullAttentionPQ_topk.

    Within each head, p_i are sorted in decreasing order by true p_i.
    Top-left annotation reports Z and Z_hat means over plotted queries.
    """
    setup_style()
    if not per_head_profiles:
        return

    all_budgets = sorted({
        int(b)
        for info in per_head_profiles.values()
        for b in info.get("profiles_by_budget", {}).keys()
    })
    if not all_budgets:
        return

    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)
    max_scatter_points = int(
        plot_cfg.get("fullattention_pq_topk_scatter_max_points", 20000)
    )

    def _per_head_sort_key(i: int) -> tuple:
        info = per_head_profiles[i]
        ent = info.get("effective_entropy")
        if ent is None:
            e = float("inf")
        else:
            try:
                e = float(ent)
                if not np.isfinite(e):
                    e = float("inf")
            except (TypeError, ValueError):
                e = float("inf")
        return (
            e,
            info.get("layer", 10**9),
            info.get("q_head", 10**9),
        )

    sorted_idxs = sorted(
        per_head_profiles.keys(), key=_per_head_sort_key,
    )
    n = len(sorted_idxs)
    cols = min(n, 3)
    rows_n = (n + cols - 1) // cols

    for b in all_budgets:
        fig, axes = plt.subplots(
            rows_n,
            cols,
            figsize=(
                figsize[0],
                min(figsize[1] * rows_n / 2, 50),
            ),
            squeeze=False,
        )
        plotted_any = False

        for i, idx in enumerate(sorted_idxs):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            info = per_head_profiles[idx]
            recs = info.get("profiles_by_budget", {}).get(b, [])
            if not recs:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"L{info['layer']}H{info['q_head']}",
                    fontsize=10,
                )
                continue

            pairs = []
            z_true_vals = []
            z_est_vals = []
            min_len = None
            for rec in recs:
                p_true = np.asarray(
                    rec.get("p_true", []), dtype=np.float64,
                )
                p_est = np.asarray(
                    rec.get("p_est_true_z", []), dtype=np.float64,
                )
                l_true = np.asarray(
                    rec.get("logits_true", []), dtype=np.float64,
                )
                l_est = np.asarray(
                    rec.get("logits_est", []), dtype=np.float64,
                )
                if (
                    p_true.size == 0
                    or p_est.size == 0
                    or l_true.size == 0
                    or l_est.size == 0
                    or p_true.size != p_est.size
                    or p_true.size != l_true.size
                    or p_true.size != l_est.size
                ):
                    continue
                ord_idx = np.argsort(-np.abs(l_true - l_est))
                t = p_true[ord_idx]
                e = p_est[ord_idx]
                d = np.abs(l_true[ord_idx] - l_est[ord_idx])
                if min_len is None:
                    min_len = t.size
                else:
                    min_len = min(min_len, t.size)
                pairs.append((t, e, d))
                z_true_vals.append(float(rec.get("z_true", np.nan)))
                z_est_vals.append(float(rec.get("z_est", np.nan)))

            if not pairs or min_len is None or min_len < 1:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"L{info['layer']}H{info['q_head']}",
                    fontsize=10,
                )
                continue

            true_stack = np.stack(
                [t[:min_len] for t, _, _ in pairs], axis=0,
            )
            est_stack = np.stack(
                [e[:min_len] for _, e, _ in pairs], axis=0,
            )
            diff_stack = np.stack(
                [d[:min_len] for _, _, d in pairs], axis=0,
            )
            true_mean = np.mean(true_stack, axis=0)
            est_mean = np.mean(est_stack, axis=0)
            diff_mean = np.mean(diff_stack, axis=0)
            x = np.arange(1, min_len + 1, dtype=np.int32)

            ax.plot(
                x, true_mean, color="#1f77b4",
                lw=1.8, label=r"$p_i$ (true)",
            )
            ax.plot(
                x, est_mean, color="#ff7f0e",
                lw=1.5, linestyle="--",
                label=r"$\hat{p}_i$ (PQ est, true-$Z$ norm)",
            )
            ax_diff = ax.twinx()
            ax_diff.plot(
                x, diff_mean, color="#d62728",
                lw=1.4, label=r"$|\ell_i - s_i|$",
                zorder=10,
            )
            ax.set_yscale("log")
            ax.set_xlabel(
                r"rank $i$ (sorted by $|\ell_i - s_i|$ desc)"
            )
            if c == 0:
                ax.set_ylabel("probability (true-Z normalized)")
                ax_diff.set_ylabel(r"$|\ell_i - s_i|$", color="#d62728")
            else:
                ax.set_yticklabels([])
                ax_diff.set_yticklabels([])
            ax_diff.tick_params(axis="y", colors="#d62728")
            lbl = info.get("selection_label", "")
            ent = info.get("effective_entropy")
            title = f"L{info['layer']}H{info['q_head']}"
            if lbl:
                title += f" ({lbl}"
                if ent is not None:
                    title += f", ent={ent:.2f}"
                title += ")"
            elif ent is not None:
                title += f" (ent={ent:.2f})"
            ax.set_title(title, fontsize=10)
            zt = float(np.nanmean(np.asarray(z_true_vals, dtype=np.float64)))
            ze = float(np.nanmean(np.asarray(z_est_vals, dtype=np.float64)))
            ax.text(
                0.02, 0.98,
                f"Z={zt:.3e}\nZ_hat={ze:.3e}",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white", alpha=0.75,
                    edgecolor="none",
                ),
            )
            if r == 0 and c == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax_diff.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
            plotted_any = True

        for i in range(n, rows_n * cols):
            r, c = divmod(i, cols)
            axes[r][c].set_visible(False)

        if not plotted_any:
            plt.close(fig)
            continue

        suptitle = (
            "FullAttentionPQ_topk diagnostic (probabilities): "
            r"blue $p_i=\exp(\ell_i)/Z$, "
            r"orange $\hat{p}_i=\exp(s_i)/Z$ "
            r"(mixed score: $s_i=\ell_i$ for PQ-top-B, "
            r"$s_i=\hat{\ell}_i$ otherwise), "
            r"red $|\ell_i-s_i|$ (linear) on secondary y-axis, "
            r"blue/orange on log-scale primary y-axis, "
            f"sorted by decreasing $|\ell_i-s_i|$ (B={b})"
        )
        if task_name:
            suptitle = f"{task_name} — {suptitle}"
        if seq_desc:
            suptitle += f" — {seq_desc}"
        if config_caption:
            suptitle = f"{suptitle}\n{config_caption}"
        fig.suptitle(
            suptitle, fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        save_figure(
            fig,
            out_dir / f"fullattention_pq_topk_pi_vs_est_b{b}.png",
            dpi=dpi,
        )

        # Per-head scatter of estimated vs true logits (s_i vs l_i).
        fig_sc, axes_sc = plt.subplots(
            rows_n,
            cols,
            figsize=(
                figsize[0],
                min(figsize[1] * rows_n / 2, 50),
            ),
            squeeze=False,
        )
        plotted_scatter = False
        for i, idx in enumerate(sorted_idxs):
            r, c = divmod(i, cols)
            ax_sc = axes_sc[r][c]
            info = per_head_profiles[idx]
            recs = info.get("profiles_by_budget", {}).get(b, [])
            if not recs:
                ax_sc.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax_sc.transAxes,
                )
                ax_sc.set_title(
                    f"L{info['layer']}H{info['q_head']}",
                    fontsize=10,
                )
                continue

            l_true_list = []
            l_est_list = []
            for rec in recs:
                l_true = np.asarray(
                    rec.get("logits_true", []), dtype=np.float64,
                )
                l_est = np.asarray(
                    rec.get("logits_est", []), dtype=np.float64,
                )
                if (
                    l_true.size == 0
                    or l_est.size == 0
                    or l_true.size != l_est.size
                ):
                    continue
                l_true_list.append(l_true)
                l_est_list.append(l_est)

            if not l_true_list:
                ax_sc.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    transform=ax_sc.transAxes,
                )
                ax_sc.set_title(
                    f"L{info['layer']}H{info['q_head']}",
                    fontsize=10,
                )
                continue

            x_all = np.concatenate(l_true_list, axis=0)
            y_all = np.concatenate(l_est_list, axis=0)
            if max_scatter_points > 0 and x_all.size > max_scatter_points:
                stride = int(np.ceil(x_all.size / max_scatter_points))
                x_all = x_all[::stride]
                y_all = y_all[::stride]

            ax_sc.scatter(
                x_all, y_all,
                s=4, alpha=0.2, c="#6a1b9a",
                edgecolors="none",
            )
            mn = float(min(np.min(x_all), np.min(y_all)))
            mx = float(max(np.max(x_all), np.max(y_all)))
            ax_sc.plot(
                [mn, mx], [mn, mx],
                color="#111111", lw=1.0, linestyle="--",
                label="y=x",
            )
            ax_sc.set_xlim(mn, mx)
            ax_sc.set_ylim(mn, mx)
            ax_sc.set_aspect("equal", adjustable="box")
            ax_sc.set_xlabel(r"$\ell_i$ (true logit)")
            if c == 0:
                ax_sc.set_ylabel(r"$s_i$ (estimated/mixed logit)")
            else:
                ax_sc.set_yticklabels([])
            lbl = info.get("selection_label", "")
            ent = info.get("effective_entropy")
            title = f"L{info['layer']}H{info['q_head']}"
            if lbl:
                title += f" ({lbl}"
                if ent is not None:
                    title += f", ent={ent:.2f}"
                title += ")"
            elif ent is not None:
                title += f" (ent={ent:.2f})"
            ax_sc.set_title(title, fontsize=10)
            if r == 0 and c == 0:
                ax_sc.legend(loc="upper left", fontsize=8)
            plotted_scatter = True

        for i in range(n, rows_n * cols):
            r, c = divmod(i, cols)
            axes_sc[r][c].set_visible(False)

        if not plotted_scatter:
            plt.close(fig_sc)
            continue

        suptitle_sc = (
            "FullAttentionPQ_topk logit scatter: "
            r"$x=\ell_i$ (true), $y=s_i$ (estimated/mixed), "
            r"dashed line is $y=x$, "
            f"B={b}"
        )
        if task_name:
            suptitle_sc = f"{task_name} — {suptitle_sc}"
        if seq_desc:
            suptitle_sc += f" — {seq_desc}"
        if config_caption:
            suptitle_sc = f"{suptitle_sc}\n{config_caption}"
        fig_sc.suptitle(
            suptitle_sc, fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        save_figure(
            fig_sc,
            out_dir / f"fullattention_pq_topk_scatter_logits_b{b}.png",
            dpi=dpi,
        )


def plot_group_cosine_distributions(
    group_cosine_stats: Dict[str, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    title: str = "",
    config_caption: str = "",
    filename: str = "group_cosine_distribution",
):
    """
    Plot cosine(key, group_mean_key) histograms per method.
    """
    if not group_cosine_stats:
        return
    setup_style()
    methods = sorted(group_cosine_stats.keys())
    n = len(methods)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)

    fig, axes = plt.subplots(
        rows_n, cols,
        figsize=(figsize[0], figsize[1] * rows_n / 2),
        squeeze=False,
    )
    for i, method in enumerate(methods):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        entry = group_cosine_stats[method]
        hist = entry.get("histogram", {})
        edges = np.array(hist.get("bin_edges", []), dtype=np.float64)
        counts = np.array(hist.get("counts", []), dtype=np.float64)
        if len(edges) >= 2 and len(counts) == len(edges) - 1:
            widths = np.diff(edges)
            centers = edges[:-1] + widths / 2
            total = max(float(np.sum(counts)), 1.0)
            density = counts / total
            ax.bar(
                centers,
                density,
                width=widths,
                align="center",
                alpha=0.75,
                edgecolor="none",
                color="C0",
                zorder=1,
            )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("cos(key, group_mean_key)")
        ax.set_ylabel("fraction")
        mu = entry.get("cos_mean", 0.0)
        sd = entry.get("cos_std", 0.0)
        nv = entry.get("n_values", 0)
        sse_m = entry.get("exp_residual_l1_over_z_mean")
        sse_s = entry.get("exp_residual_l1_over_z_std")
        if sse_m is None:
            sse_m = entry.get("exp_residual_sse_over_z2_mean")
        if sse_s is None:
            sse_s = entry.get("exp_residual_sse_over_z2_std")
        if sse_m is None:
            sse_m = entry.get("exp_residual_sse_sum_mean")
        if sse_s is None:
            sse_s = entry.get("exp_residual_sse_sum_std")
        if sse_m is None:
            sse_m = entry.get("logit_residual_sse_sum_mean")
        if sse_s is None:
            sse_s = entry.get("logit_residual_sse_sum_std")
        if sse_m is None:
            sse_m = entry.get("logit_within_group_var_sum_mean")
        if sse_s is None:
            sse_s = entry.get("logit_within_group_var_sum_std")
        tlines = [
            str(method),
            f"mu={mu:.3f}, sd={sd:.3f}, n={nv}",
        ]
        ze_m = entry.get("sum_exp_logits_mean")
        ze_s = entry.get("sum_exp_logits_std")
        if ze_m is not None:
            tlines.append(
                f"Z={ze_m:.3g}" + (f"\u00b1{ze_s:.3g}" if ze_s else ""),
            )
        zb_m = entry.get("sum_exp_bar_logits_mean")
        zb_s = entry.get("sum_exp_bar_logits_std")
        if zb_m is not None:
            tlines.append(
                f"Z'={zb_m:.3g}" + (f"\u00b1{zb_s:.3g}" if zb_s else ""),
            )
        if sse_m is not None:
            tlines.append(
                f"L1 {_format_exp_l1_over_z(sse_m)}"
                + (f"\u00b1{_format_exp_l1_over_z(sse_s)}" if sse_s is not None else ""),
            )
        l1z_m = entry.get("exp_residual_l1_znorm_mean")
        l1z_s = entry.get("exp_residual_l1_znorm_std")
        if l1z_m is not None:
            tlines.append(
                f"L1(Z) {_format_exp_l1_over_z(l1z_m)}"
                + (f"\u00b1{_format_exp_l1_over_z(l1z_s)}" if l1z_s is not None else ""),
            )
        vl_m = entry.get("value_softmax_mismatch_ratio_mean")
        vl_s = entry.get("value_softmax_mismatch_ratio_std")
        if vl_m is None:
            vl_m = entry.get("value_logit_group_l2_sq_mean")
        if vl_s is None:
            vl_s = entry.get("value_logit_group_l2_sq_std")
        if vl_m is not None:
            tlines.append(
                f"||\u0394o||/||o*|| {vl_m:.4g}"
                + (f"\u00b1{vl_s:.4g}" if vl_s is not None else ""),
            )
        vlz_m = entry.get("value_mismatch_ratio_znorm_mean")
        vlz_s = entry.get("value_mismatch_ratio_znorm_std")
        if vlz_m is not None:
            tlines.append(
                f"||\u0394o(Z)||/||o*|| {vlz_m:.4g}"
                + (f"\u00b1{vlz_s:.4g}" if vlz_s is not None else ""),
            )
        g1m = entry.get("group_l1_hist_meta") or {}
        _qcg = _group_l1_cgm_over_z_quantiles_line(g1m)
        if _qcg:
            tlines.append(_qcg)
        g1z = entry.get("group_l1_znorm_meta") or {}
        _qcgz = _group_l1_znorm_quantiles_line(g1z)
        if _qcgz:
            tlines.append(_qcgz)
        _overlay_p75_z_norm_histogram(ax, entry)
        _overlay_group_l1_contrib_histogram(ax, entry)
        ax.set_title("\n".join(tlines), fontsize=9)
        leg_handles = [
            Patch(
                facecolor="C0",
                alpha=0.75,
                edgecolor="none",
                label="cos(key, group mean)",
            ),
        ]
        if entry.get("max_group_exp_logits_histogram") or entry.get(
            "p75_z_norm_histogram",
        ) or entry.get(
            "median_z_norm_histogram",
        ):
            leg_handles.append(
                Patch(
                    facecolor="tab:orange",
                    alpha=0.42,
                    edgecolor="none",
                    label=(
                        r"max-$e^{\ell_g}$ group: "
                        r"$e^{\ell_i}\,m/Z$"
                    ),
                ),
            )
        if entry.get("max_group_exp_lg_histogram"):
            leg_handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="#c76a00",
                    linewidth=0.9,
                    label=(
                        r"max-$e^{\ell_g}$ group: "
                        r"$e^{\ell_g}\,m/Z$"
                    ),
                ),
            )
        if entry.get("group_l1_contrib_cumulative"):
            leg_handles.append(
                Line2D(
                    [0], [0],
                    label=(
                        r"cum. $L_1$ share vs threshold on $c_g\,m$"
                    ),
                    color="tab:green",
                    lw=1.6,
                ),
            )
            leg_handles.append(
                Line2D(
                    [0], [0],
                    label=(
                        r"cum. group share vs threshold on $c_g\,m$"
                    ),
                    color="tab:green",
                    lw=1.35,
                    linestyle="--",
                ),
            )
        if len(leg_handles) > 1:
            ax.legend(
                handles=leg_handles,
                loc="upper left",
                fontsize=6,
                framealpha=0.92,
            )

    for i in range(n, rows_n * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    suptitle = (
        "Key-to-Group-Mean Cosine Distributions "
        "(orange: max-$e^{\\ell_g}$ group $e^{\\ell_i}m/Z$; "
        "orange-outline: $e^{\\ell_g}m/Z$; "
        "green: cumulative $L_1$ share vs threshold on $c_g\\,m$)"
    )
    if title:
        suptitle = f"{title} — {suptitle}"
    if config_caption:
        suptitle = f"{suptitle}\n{config_caption}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, out_dir / f"{filename}.png", dpi=dpi)


def plot_group_cosine_table(
    table_stats: Dict[str, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    title: str = "",
    config_caption: str = "",
    filename: str = "group_cosine_distribution_table",
):
    """
    Table layout: rows are algorithms, columns are (head, #groups).
    """
    rows = table_stats.get("row_algorithms", [])
    columns = _order_group_cosine_table_columns(
        list(table_stats.get("columns", [])),
    )
    cells = table_stats.get("cells", {})
    if not rows or not columns:
        return

    setup_style()
    n_rows = len(rows)
    n_cols = len(columns)
    dpi = plot_cfg.get("dpi", 200)

    # Slightly wider cells for readable column headers.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(4 * n_cols, 10), max(2.6 * n_rows, 6)),
        squeeze=False,
    )

    for r, algo in enumerate(rows):
        for c, col in enumerate(columns):
            ax = axes[r][c]
            col_key = col["key"]
            entry = cells.get(algo, {}).get(col_key)
            if entry:
                hist = entry.get("histogram", {})
                edges = np.array(
                    hist.get("bin_edges", []), dtype=np.float64
                )
                counts = np.array(
                    hist.get("counts", []), dtype=np.float64
                )
                if len(edges) >= 2 and len(counts) == len(edges) - 1:
                    widths = np.diff(edges)
                    centers = edges[:-1] + widths / 2
                    total = max(float(np.sum(counts)), 1.0)
                    density = counts / total
                    ax.bar(
                        centers,
                        density,
                        width=widths,
                        align="center",
                        alpha=0.75,
                        edgecolor="none",
                        color="C0",
                        zorder=1,
                    )
                _overlay_p75_z_norm_histogram(ax, entry)
                _overlay_group_l1_contrib_histogram(ax, entry)
                mu = entry.get("cos_mean", 0.0)
                sd = entry.get("cos_std", 0.0)
                nv = entry.get("n_values", 0)
                sse_m = entry.get("exp_residual_l1_over_z_mean")
                sse_s = entry.get("exp_residual_l1_over_z_std")
                if sse_m is None:
                    sse_m = entry.get("exp_residual_sse_over_z2_mean")
                if sse_s is None:
                    sse_s = entry.get("exp_residual_sse_over_z2_std")
                if sse_m is None:
                    sse_m = entry.get("exp_residual_sse_sum_mean")
                if sse_s is None:
                    sse_s = entry.get("exp_residual_sse_sum_std")
                if sse_m is None:
                    sse_m = entry.get(
                        "logit_residual_sse_sum_mean",
                    )
                if sse_s is None:
                    sse_s = entry.get(
                        "logit_residual_sse_sum_std",
                    )
                if sse_m is None:
                    sse_m = entry.get(
                        "logit_within_group_var_sum_mean",
                    )
                if sse_s is None:
                    sse_s = entry.get(
                        "logit_within_group_var_sum_std",
                    )
                txt_lines = [
                    f"mu={mu:.3f}  sd={sd:.3f}  n={nv}",
                ]
                ze_m = entry.get("sum_exp_logits_mean")
                ze_s = entry.get("sum_exp_logits_std")
                if ze_m is not None:
                    txt_lines.append(
                        f"Z {ze_m:.3g}" + (f"±{ze_s:.3g}" if ze_s else ""),
                    )
                zb_m = entry.get("sum_exp_bar_logits_mean")
                zb_s = entry.get("sum_exp_bar_logits_std")
                if zb_m is not None:
                    txt_lines.append(
                        f"Z' {zb_m:.3g}" + (f"±{zb_s:.3g}" if zb_s else ""),
                    )
                if sse_m is not None:
                    txt_lines.append(
                        f"L1 {_format_exp_l1_over_z(sse_m)}"
                        + (f"±{_format_exp_l1_over_z(sse_s)}" if sse_s is not None else ""),
                    )
                l1z_m = entry.get("exp_residual_l1_znorm_mean")
                l1z_s = entry.get("exp_residual_l1_znorm_std")
                if l1z_m is not None:
                    txt_lines.append(
                        f"L1(Z) {_format_exp_l1_over_z(l1z_m)}"
                        + (f"±{_format_exp_l1_over_z(l1z_s)}" if l1z_s is not None else ""),
                    )
                vl_m = entry.get("value_softmax_mismatch_ratio_mean")
                vl_s = entry.get("value_softmax_mismatch_ratio_std")
                if vl_m is None:
                    vl_m = entry.get("value_logit_group_l2_sq_mean")
                if vl_s is None:
                    vl_s = entry.get("value_logit_group_l2_sq_std")
                if vl_m is not None:
                    txt_lines.append(
                        f"||Δo||/||o*|| {vl_m:.3g}"
                        + (f"±{vl_s:.3g}" if vl_s is not None else ""),
                    )
                vlz_m = entry.get("value_mismatch_ratio_znorm_mean")
                vlz_s = entry.get("value_mismatch_ratio_znorm_std")
                if vlz_m is not None:
                    txt_lines.append(
                        f"||Δo(Z)||/||o*|| {vlz_m:.3g}"
                        + (f"±{vlz_s:.3g}" if vlz_s is not None else ""),
                    )
                g1m_c = entry.get("group_l1_hist_meta") or {}
                _qcg_c = _group_l1_cgm_over_z_quantiles_line(g1m_c)
                if _qcg_c:
                    txt_lines.append(_qcg_c)
                g1z_c = entry.get("group_l1_znorm_meta") or {}
                _qcgz_c = _group_l1_znorm_quantiles_line(g1z_c)
                if _qcgz_c:
                    txt_lines.append(_qcgz_c)
                ax.text(
                    0.02, 0.96,
                    "\n".join(txt_lines),
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    zorder=25,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )
            else:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center", fontsize=10,
                    alpha=0.6, transform=ax.transAxes,
                )
            ax.set_xlim(-1.0, 1.0)
            if r == n_rows - 1:
                ax.set_xlabel("cos(key, group_mean_key)")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{algo}\nfraction")
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(
                    f"{col['head']}\nG={col['n_groups']}",
                    fontsize=10,
                )

    suptitle = (
        "Group Cosine Table (cols=head+groups; "
        "orange: max-$e^{\\ell_g}$ group $e^{\\ell_i}m/Z$; "
        "orange-outline: $e^{\\ell_g}m/Z$; "
        "green: cumulative $L_1$ share vs threshold on $c_g\\,m$)"
    )
    if title:
        suptitle = f"{title} — {suptitle}"
    if config_caption:
        suptitle = f"{suptitle}\n{config_caption}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, out_dir / f"{filename}.png", dpi=dpi)


def _add_cg_eg_scatter_y_exp_quantiles(
    ax,
    y: np.ndarray,
    *,
    show_legend: bool,
) -> Dict[str, float]:
    r"""
    Horizontal guides for pooled $e^{\ell_g} m/Z$: P25, P50, P90, P100
    (P100 = max y).
    """
    if y.size == 0:
        return {}
    q25 = float(np.quantile(y, 0.25))
    q50 = float(np.quantile(y, 0.50))
    q90 = float(np.quantile(y, 0.90))
    q100 = float(np.max(y))
    specs = [
        (q25, "P25", ":", "#6b6b6b", 0.95, 0.75),
        (q50, "P50", "-", "#1f77b4", 1.05, 0.82),
        (q90, "P90", "--", "#ff7f0e", 1.0, 0.82),
        (q100, "P100", "-", "#d62728", 1.2, 0.88),
    ]
    for qv, name, ls, color, lw, al in specs:
        ax.axhline(
            qv,
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=al,
            zorder=4.5,
        )
    if show_legend:
        handles = [
            Line2D(
                [0], [0],
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=name,
            )
            for _, name, ls, color, lw, _ in specs
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=5,
            framealpha=0.92,
            title=r"$e^{\ell_g}m/Z$",
            title_fontsize=5,
        )
    return {
        "p25": q25,
        "p50": q50,
        "p90": q90,
        "p100": q100,
    }


def _setup_cg_eg_scatter_axes(
    ax,
    entry: Dict,
    *,
    draw_axis_labels: bool = True,
    show_y_quantile_legend: bool = True,
    show_y_quantile_text: bool = True,
) -> bool:
    """
    Scatter of (c_g m/n_g, exp(l_g) m/Z) per group; c_g is the group L1
    difference in normalized probability space; l_g is the mean logit in
    the group.
    Log-log when all plotted points are strictly positive.
    """
    sc = entry.get("cg_eg_scatter")
    if not sc:
        return False
    x = np.asarray(sc.get("x", []), dtype=np.float64)
    y = np.asarray(sc.get("y", []), dtype=np.float64)
    if x.size == 0 or x.shape != y.shape:
        return False
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return False
    use_log = bool(np.all((x > 0) & (y > 0)))
    ax.scatter(
        x,
        y,
        s=5,
        alpha=0.28,
        c="C0",
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    qvals = _add_cg_eg_scatter_y_exp_quantiles(
        ax, y, show_legend=show_y_quantile_legend,
    )
    if show_y_quantile_text and qvals:
        txt = (
            f"P25={_format_exp_l1_over_z(qvals['p25'])}\n"
            f"P50={_format_exp_l1_over_z(qvals['p50'])}\n"
            f"P90={_format_exp_l1_over_z(qvals['p90'])}\n"
            f"P100={_format_exp_l1_over_z(qvals['p100'])}"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            zorder=6,
            bbox=dict(
                facecolor="white",
                alpha=0.72,
                edgecolor="none",
            ),
        )
    ax.tick_params(axis="both", labelsize=7)
    if draw_axis_labels:
        ax.set_xlabel(r"$c_g\,m/n_g$", fontsize=9)
        ax.set_ylabel(r"$e^{\ell_g}\,m/Z$", fontsize=9)
    return True


def plot_group_cosine_cg_eg_scatter(
    group_cosine_stats: Dict[str, Dict],
    out_dir: Path,
    plot_cfg: Dict,
    title: str = "",
    config_caption: str = "",
    filename: str = "group_cosine_cg_eg_scatter",
):
    """
    Per-method scatter: c_g m/n_g (x) vs exp(l_g) m/Z (y), same grid as cosine.
    """
    if not group_cosine_stats:
        return
    setup_style()
    methods = sorted(group_cosine_stats.keys())
    n = len(methods)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    figsize = tuple(plot_cfg.get("figsize", [16, 10]))
    dpi = plot_cfg.get("dpi", 200)
    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(figsize[0], figsize[1] * rows_n / 2),
        squeeze=False,
    )
    for i, method in enumerate(methods):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        entry = group_cosine_stats[method]
        if not _setup_cg_eg_scatter_axes(
            ax,
            entry,
            draw_axis_labels=False,
            show_y_quantile_legend=True,
        ):
            ax.text(
                0.5,
                0.5,
                "n/a",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(str(method), fontsize=9)
    for i in range(n, rows_n * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)
    suptitle = (
        r"Per-group scatter: $c_g\,m/n_g$ vs $e^{\ell_g}\,m/Z$ · "
        r"$c_g=\sum_{i\in g}|e^{\ell_i}/Z\!-\!e^{\bar\ell_g}/Z'|$, "
        r"$Z'=\sum_g n_g e^{\bar\ell_g}$, "
        r"$\ell_g=\mathrm{mean}_{i\in g}\ell_i$"
    )
    if title:
        suptitle = f"{title} — {suptitle}"
    if config_caption:
        suptitle = f"{suptitle}\n{config_caption}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0.08, 0.06, 1.0, 0.94))
    fig.supxlabel(r"$c_g\,m/n_g$", fontsize=11, y=0.02)
    fig.supylabel(r"$e^{\ell_g}\,m/Z$", fontsize=11, x=0.02)
    save_figure(fig, out_dir / f"{filename}.png", dpi=dpi)


def plot_group_cosine_cg_eg_scatter_table(
    table_stats: Dict,
    out_dir: Path,
    plot_cfg: Dict,
    title: str = "",
    config_caption: str = "",
    filename: str = "group_cosine_cg_eg_scatter_table",
):
    """Table layout: same rows/cols as group cosine table, scatter per cell."""
    rows = table_stats.get("row_algorithms", [])
    columns = _order_group_cosine_table_columns(
        list(table_stats.get("columns", [])),
    )
    cells = table_stats.get("cells", {})
    if not rows or not columns:
        return
    setup_style()
    n_rows = len(rows)
    n_cols = len(columns)
    dpi = plot_cfg.get("dpi", 200)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            max(4 * n_cols, 10),
            max(2.6 * n_rows, 6),
        ),
        squeeze=False,
    )
    for r, algo in enumerate(rows):
        for c, col in enumerate(columns):
            ax = axes[r][c]
            col_key = col["key"]
            entry = cells.get(algo, {}).get(col_key)
            if entry:
                if not _setup_cg_eg_scatter_axes(
                    ax,
                    entry,
                    draw_axis_labels=False,
                    show_y_quantile_legend=False,
                ):
                    ax.text(
                        0.5,
                        0.5,
                        "n/a",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            if r == n_rows - 1:
                ax.set_xlabel(r"$c_g\,m/n_g$", fontsize=8)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(
                    f"{algo}\n$e^{{\\ell_g}}\\,m/Z$",
                    fontsize=8,
                )
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(
                    f"{col['head']}\nG={col['n_groups']}",
                    fontsize=10,
                )
    suptitle = (
        r"Scatter $c_g\,m/n_g$ vs $e^{\ell_g}\,m/Z$ "
        r"($c_g=\sum_{i\in g}|e^{\ell_i}/Z\!-\!e^{\bar\ell_g}/Z'|$, "
        r"$Z'=\sum_g n_g e^{\bar\ell_g}$, "
        r"$\ell_g=\mathrm{mean}_{i\in g}\ell_i$)"
    )
    if title:
        suptitle = f"{title} — {suptitle}"
    if config_caption:
        suptitle = f"{suptitle}\n{config_caption}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0.06, 0.04, 1.0, 0.95))
    save_figure(fig, out_dir / f"{filename}.png", dpi=dpi)


def _plot_group_token_probability_profile(
    ax,
    entry: Dict,
    *,
    order_mode: str = "group_then_key",
) -> bool:
    """
    Plot per-key probabilities after sorting groups by decreasing e^{ell_g}
    and sorting keys within each group by decreasing e^{ell_i}/Z.
    """
    prof = entry.get("group_token_probability_profile") or {}
    key_probs = np.asarray(
        prof.get("key_probs", []), dtype=np.float64,
    )
    group_probs = np.asarray(
        prof.get("group_probs", []), dtype=np.float64,
    )
    group_probs_over_z = np.asarray(
        prof.get("group_probs_over_z", []), dtype=np.float64,
    )
    boundaries = np.asarray(
        prof.get("group_boundaries", []), dtype=np.int32,
    )
    if key_probs.size == 0 or key_probs.shape != group_probs.shape:
        return False
    key_probs_orig = key_probs.copy()
    group_probs_orig = group_probs.copy()
    boundaries_orig = boundaries.copy()
    if group_probs_over_z.shape != key_probs.shape:
        group_probs_over_z = np.array([], dtype=np.float64)
    # Main group probability for this plot: normalize by Z so it is
    # directly comparable to p_i = exp(l_i)/Z.
    if group_probs_over_z.size > 0:
        group_probs_main = group_probs_over_z.copy()
    else:
        group_probs_main = group_probs.copy()
    if order_mode == "key_logit":
        ord_idx = np.argsort(-key_probs)
        key_probs = key_probs[ord_idx]
        group_probs = group_probs[ord_idx]
        group_probs_main = group_probs_main[ord_idx]
        if group_probs_over_z.size > 0:
            group_probs_over_z = group_probs_over_z[ord_idx]
        boundaries = np.array([], dtype=np.int32)
    cum_cg_x = np.array([], dtype=np.float64)
    cum_cg_y = np.array([], dtype=np.float64)
    if boundaries.size > 0 and int(boundaries[-1]) == int(key_probs.size):
        starts = np.concatenate(
            [np.array([0], dtype=np.int32), boundaries[:-1]],
        )
        cg_vals: List[float] = []
        x_vals: List[float] = []
        for s, e in zip(starts, boundaries):
            si = int(s)
            ei = int(e)
            if ei <= si:
                continue
            cg_vals.append(
                float(np.sum(np.abs(key_probs[si:ei] - group_probs_main[si:ei]))),
            )
            x_vals.append(float(ei))
        if cg_vals:
            cg_arr = np.asarray(cg_vals, dtype=np.float64)
            cg_total = float(np.sum(cg_arr))
            if cg_total > 0.0:
                cum_cg_y = np.cumsum(cg_arr) / cg_total
            else:
                cum_cg_y = np.zeros_like(cg_arr)
            cum_cg_x = np.asarray(x_vals, dtype=np.float64)
    elif key_probs.size > 0:
        tok_diff = np.abs(key_probs - group_probs_main)
        tok_total = float(np.sum(tok_diff))
        if tok_total > 0.0:
            cum_cg_y = np.cumsum(tok_diff) / tok_total
        else:
            cum_cg_y = np.zeros_like(tok_diff)
        cum_cg_x = np.arange(
            1, key_probs.size + 1, dtype=np.float64,
        )
    x = np.arange(1, key_probs.size + 1, dtype=np.int32)
    # Continuous curve (the one that reads as a single trace) should
    # draw on top: key-sorted → blue; group-sorted → orange.
    if order_mode == "key_logit":
        z_key_line = 5.2
        z_group_line = 3.8
        z_zprime_line = 3.2
    else:
        z_key_line = 3.8
        z_group_line = 5.2
        z_zprime_line = 4.6
    above = key_probs >= group_probs_main
    below = ~above
    ax.fill_between(
        x, key_probs, group_probs_main,
        where=above,
        color="#d62728",
        alpha=0.22,
        interpolate=True,
        zorder=1,
    )
    ax.fill_between(
        x, key_probs, group_probs_main,
        where=below,
        color="#2ca02c",
        alpha=0.22,
        interpolate=True,
        zorder=1,
    )
    if order_mode == "key_logit":
        ax.plot(
            x,
            key_probs,
            color="C0",
            lw=0.5,
            alpha=0.9,
            zorder=z_key_line,
            label=r"$e^{\ell_i}/Z$",
        )
    else:
        seg_starts = [0]
        if boundaries.size > 0:
            seg_starts.extend([int(b) for b in boundaries[:-1]])
        seg_ends = [int(b) for b in boundaries] if boundaries.size > 0 else [key_probs.size]
        for si, (s, e) in enumerate(zip(seg_starts, seg_ends)):
            if e <= s:
                continue
            xs = x[s:e]
            ys = key_probs[s:e]
            ax.plot(
                xs,
                ys,
                color="C0",
                lw=0.5,
                alpha=0.9,
                zorder=z_key_line,
                label=r"$e^{\ell_i}/Z$" if si == 0 else None,
            )
    ax.plot(
        x,
        group_probs_main,
        color="#ff7f0e",
        lw=1.25,
        alpha=0.95,
        zorder=z_group_line,
        label=r"$e^{\ell_g}/Z$",
    )
    if group_probs_over_z.size > 0:
        ax.plot(
            x,
            group_probs,
            color="#ff7f0e",
            lw=1.05,
            alpha=0.9,
            linestyle=":",
            zorder=z_zprime_line,
        )
    if cum_cg_x.size > 0 and cum_cg_y.size > 0:
        ax_cg = ax.twinx()
        ax_cg.plot(
            cum_cg_x,
            cum_cg_y,
            color="black",
            lw=1.1,
            linestyle="-.",
            alpha=0.9,
            zorder=max(z_key_line, z_group_line) + 0.5,
            label=r"cum $|p_i-\tilde p_i|$ (norm.)",
        )
        ax_cg.set_ylim(0.0, 1.02)
        ax_cg.set_yticks([0.0, 0.5, 1.0])
        ax_cg.tick_params(axis="y", labelsize=6, colors="black")
        ax_cg.set_ylabel(r"cum $|p_i-\tilde p_i|$", fontsize=7, color="black")
    p50_key = float(np.quantile(key_probs_orig, 0.50))
    if boundaries_orig.size > 0:
        g_starts = np.concatenate(
            [np.array([0], dtype=np.int32), boundaries_orig[:-1]],
        )
        if group_probs_over_z.size > 0:
            g_vals = group_probs_over_z[g_starts]
        else:
            g_vals = group_probs_orig[g_starts]
    else:
        if group_probs_over_z.size > 0:
            g_vals = group_probs_over_z
        else:
            g_vals = group_probs_orig
    p50_group = float(np.quantile(g_vals, 0.50))
    ax.text(
        0.02,
        0.98,
        "P50 $e^{\\ell_i}$="
        f"{_format_exp_l1_over_z(p50_key)}\n"
        "P50 $e^{\\ell_g}/Z$="
        f"{_format_exp_l1_over_z(p50_group)}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        ha="left",
        zorder=6,
        bbox=dict(
            facecolor="white",
            alpha=0.72,
            edgecolor="none",
        ),
    )
    ax.tick_params(axis="both", labelsize=7)
    return True


def plot_group_token_probability_table(
    table_stats: Dict,
    out_dir: Path,
    plot_cfg: Dict,
    title: str = "",
    config_caption: str = "",
    filename: str = "group_token_probability_table",
):
    """
    Table layout: rows are algorithms, columns are (head, #groups).

    Sorting is controlled by ``plot_cfg["group_token_probability_order"]``:
    ``key_logit`` (default) sorts all keys by decreasing p_i; ``group_then_key``
    sorts by group then key within each group. The continuous curve is drawn on
    top in each mode (blue in key order, orange in group order).
    """
    rows = table_stats.get("row_algorithms", [])
    columns = _order_group_cosine_table_columns(
        list(table_stats.get("columns", [])),
    )
    cells = table_stats.get("cells", {})
    if not rows or not columns:
        return
    setup_style()
    order_mode = str(
        plot_cfg.get("group_token_probability_order", "key_logit"),
    ).strip().lower()
    if order_mode not in {"group_then_key", "key_logit"}:
        order_mode = "key_logit"
    n_rows = len(rows)
    n_cols = len(columns)
    base_dpi = int(plot_cfg.get("dpi", 200))
    dpi = max(base_dpi, 800)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            max(4 * n_cols, 10),
            max(2.8 * n_rows, 6),
        ),
        squeeze=False,
    )
    legend_handles = [
        Line2D([0], [0], color="C0", lw=1.0, label=r"$e^{\ell_i}/Z$"),
        Line2D([0], [0], color="#ff7f0e", lw=1.25, label=r"$e^{\ell_g}/Z$"),
        Line2D([0], [0], color="black", lw=1.1, linestyle="-.", label=r"cum $|p_i-\tilde p_i|$ (norm.)"),
        Patch(facecolor="#d62728", alpha=0.22, label=r"$e^{\ell_i}/Z \geq e^{\ell_g}/Z$"),
        Patch(facecolor="#2ca02c", alpha=0.22, label=r"$e^{\ell_i}/Z < e^{\ell_g}/Z$"),
    ]
    col_ylims: Dict[int, tuple] = {}
    for c, col in enumerate(columns):
        col_key = col["key"]
        all_key_vals: List[float] = []
        for algo in rows:
            entry = cells.get(algo, {}).get(col_key)
            if not entry:
                continue
            prof = entry.get("group_token_probability_profile") or {}
            kp = prof.get("key_probs", [])
            if kp:
                arr = np.asarray(kp, dtype=np.float64)
                pos = arr[arr > 0]
                if pos.size > 0:
                    all_key_vals.extend(pos.tolist())
        if all_key_vals:
            lo = float(min(all_key_vals))
            hi = float(max(all_key_vals))
            span = hi - lo if hi > lo else hi
            margin = 0.05 * span
            col_ylims[c] = (max(0.0, lo - margin), hi + margin)
        else:
            col_ylims[c] = (0.0, 1.0)
    for r, algo in enumerate(rows):
        for c, col in enumerate(columns):
            ax = axes[r][c]
            col_key = col["key"]
            entry = cells.get(algo, {}).get(col_key)
            if entry and _plot_group_token_probability_profile(
                ax, entry, order_mode=order_mode,
            ):
                pass
            else:
                ax.text(
                    0.5, 0.5, "n/a",
                    ha="center", va="center",
                    fontsize=10, alpha=0.6,
                    transform=ax.transAxes,
                )
            ax.set_ylim(col_ylims[c])
            if r == n_rows - 1:
                ax.set_xlabel("keys in sorted group order", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(f"{algo}\nprobability", fontsize=8)
            if r == 0:
                ax.set_title(
                    f"{col['head']}\nG={col['n_groups']}",
                    fontsize=10,
                )
    if order_mode == "key_logit":
        suptitle = (
            "Grouped Key Probability Table "
            "(all keys sorted by decreasing $e^{\\ell_i}/Z$)"
        )
    else:
        suptitle = (
            "Grouped Key Probability Table "
            "(groups sorted by decreasing $e^{\\ell_g}$; within-group keys sorted by "
            "decreasing $e^{\\ell_i}/Z$)"
        )
    if title:
        suptitle = f"{title} — {suptitle}"
    if config_caption:
        suptitle = f"{suptitle}\n{config_caption}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        fontsize=8,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.985),
    )
    plt.tight_layout(rect=(0.03, 0.03, 1.0, 0.93))
    save_figure(fig, out_dir / f"{filename}.png", dpi=dpi)
