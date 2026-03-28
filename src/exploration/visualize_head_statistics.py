#!/usr/bin/env python3
"""
Visualize pre-computed head statistics from extraction.

Reads head_statistics JSON files and produces:
  1. Per-task summary box plots (distribution across all heads)
  2. Per-layer bar plots (mean across heads) for each task
  3. Per-example box plots (when per_example/ data exists)
  4. Cross-example stability plots (median + IQR bands)

Usage:
  python -m src.exploration.visualize_head_statistics \
      --stats-dir data/head_statistics/llama3.1_8b \
      --out-dir results/head_statistics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Style ─────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

C_FULL = "#4C72B0"
C_NSL = "#DD8452"

# All 6 metrics used in per-example and stability plots
METRICS = [
    "full_entropy", "nonlocal_entropy",
    "full_top1pct_mass", "nonlocal_top1pct_mass",
    "full_top5pct_mass", "nonlocal_top5pct_mass",
]

# Paired metrics for box plots: (full, nonlocal, label, ylim)
METRIC_PAIRS = [
    ("full_entropy", "nonlocal_entropy",
     "entropy", "Entropy (nats)", "Entropy", None),
    ("full_top1pct_mass", "nonlocal_top1pct_mass",
     "top1pct_mass", "Top 1% Mass", "Top 1% Attention Mass",
     (-0.05, 1.08)),
    ("full_top5pct_mass", "nonlocal_top5pct_mass",
     "top5pct_mass", "Top 5% Mass", "Top 5% Attention Mass",
     (-0.05, 1.08)),
]


# ── Backward-compat metric name mapping ──────────

_METRIC_ALIASES = {
    "entropy_full": "full_entropy",
    "entropy_nonlocal": "nonlocal_entropy",
    "top1pct_mass_full": "full_top1pct_mass",
    "top1pct_mass_nonlocal": "nonlocal_top1pct_mass",
    "top5pct_mass_full": "full_top5pct_mass",
    "top5pct_mass_nonlocal": "nonlocal_top5pct_mass",
}


def _normalize_metric_names(stats: dict) -> dict:
    """Rename old metric names to current convention."""
    out = {}
    for layer_key, heads in stats.items():
        new_heads = {}
        for head_key, head_stats in heads.items():
            new_hs = {}
            for k, v in head_stats.items():
                new_hs[_METRIC_ALIASES.get(k, k)] = v
            new_heads[head_key] = new_hs
        out[layer_key] = new_heads
    return out


# ── Data loading ──────────────────────────────────

def load_all_stats(stats_dir: Path) -> tuple:
    """Load stats JSONs. Returns (stats, example_meta).

    example_meta: {task: (example_id, seq_len)} built
    from embedded metadata when available.
    """
    stats = {}
    example_meta = {}
    for p in sorted(stats_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        meta = data.pop("metadata", None)
        if meta:
            example_meta[p.stem] = (
                meta.get("example_id", ""),
                meta.get("sequence_length", 0),
            )
        stats[p.stem] = _normalize_metric_names(data)
    return stats, example_meta


def load_per_example_stats(
    stats_dir: Path, task: str,
) -> List[Tuple[dict, dict]]:
    """Load per_example/{task}/ex_*.json files.

    Returns list of (metadata, stats_dict) tuples sorted
    by example index.
    """
    pe_dir = stats_dir / "per_example" / task
    if not pe_dir.exists():
        return []
    results = []
    for p in sorted(pe_dir.glob("ex_*.json")):
        with open(p) as f:
            data = json.load(f)
        meta = data.pop("metadata", {})
        results.append((meta, _normalize_metric_names(data)))
    return results


def collect_all_values(
    task_stats: dict, metric: str,
) -> list:
    """Collect metric values from every head in every layer."""
    vals = []
    for heads in task_stats.values():
        for hs in heads.values():
            if metric in hs:
                vals.append(hs[metric])
    return vals


def collect_all_values_multi(
    per_example_stats: List[Tuple[dict, dict]],
    metric: str,
) -> list:
    """Flat list of head values across all examples."""
    vals = []
    for _meta, stats in per_example_stats:
        vals.extend(collect_all_values(stats, metric))
    return vals


def extract_layer_averages(task_stats: dict) -> dict:
    result = {}
    for layer_key, heads in task_stats.items():
        layer_idx = int(layer_key.split("_")[1])
        accum = {}
        for head_stats in heads.values():
            for metric, val in head_stats.items():
                accum.setdefault(metric, []).append(val)
        result[layer_idx] = {
            k: float(np.mean(v)) for k, v in accum.items()
        }
    return result


# ── Task-level box plots ─────────────────────────

def _task_boxplot(
    ax, tasks, data_full, data_nsl,
    ylabel, title, ylim=None, example_meta=None,
    n_heads_label=None,
):
    """Paired box plots comparing full vs nonlocal per task."""
    positions_full = []
    positions_nsl = []
    spacing = 1.0
    box_width = 0.3

    for i in range(len(tasks)):
        center = i * spacing
        positions_full.append(center - 0.2)
        positions_nsl.append(center + 0.2)

    bp_kw = dict(
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        showmeans=False,
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=0.8),
    )

    bp_full = ax.boxplot(
        data_full, positions=positions_full,
        flierprops=dict(
            marker="o", markersize=3.5, alpha=0.5,
            markerfacecolor=C_FULL, markeredgecolor="none",
        ),
        **bp_kw,
    )
    for patch in bp_full["boxes"]:
        patch.set_facecolor(C_FULL)
        patch.set_alpha(0.8)
    for w in bp_full["whiskers"] + bp_full["caps"]:
        w.set_color(C_FULL)

    bp_nsl = ax.boxplot(
        data_nsl, positions=positions_nsl,
        flierprops=dict(
            marker="o", markersize=3.5, alpha=0.5,
            markerfacecolor=C_NSL, markeredgecolor="none",
        ),
        **bp_kw,
    )
    for patch in bp_nsl["boxes"]:
        patch.set_facecolor(C_NSL)
        patch.set_alpha(0.8)
    for w in bp_nsl["whiskers"] + bp_nsl["caps"]:
        w.set_color(C_NSL)

    # Mean diamonds
    for i in range(len(tasks)):
        ax.scatter(
            positions_full[i], np.mean(data_full[i]),
            marker="D", s=28, color="white",
            edgecolors=C_FULL, linewidths=1, zorder=5,
        )
        ax.scatter(
            positions_nsl[i], np.mean(data_nsl[i]),
            marker="D", s=28, color="white",
            edgecolors=C_NSL, linewidths=1, zorder=5,
        )

    centers = [i * spacing for i in range(len(tasks))]
    ax.set_xticks(centers)
    em = example_meta or {}
    xlabels = []
    for t in tasks:
        if t in em:
            ex_id, n_tok = em[t]
            xlabels.append(
                f"{t}\n{ex_id}  ({n_tok:,} tok)"
            )
        else:
            xlabels.append(t)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel(ylabel)

    if n_heads_label is not None:
        head_str = n_heads_label
    else:
        head_str = "1024 heads"
    ax.set_title(
        f"{title} ({head_str})",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, 1.02,
        "Box: Q25\u2013Q75  |  Line: median  |  "
        "\u25c7: mean  |  Dots: outliers",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.5, color="#888888",
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        lo, hi = ax.get_ylim()
        margin = (hi - lo) * 0.08
        ax.set_ylim(lo - margin, hi + margin)

    legend_handles = [
        mpatches.Patch(
            facecolor=C_FULL, alpha=0.8, label="Full",
        ),
        mpatches.Patch(
            facecolor=C_NSL, alpha=0.8, label="Nonlocal",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=9)


def plot_task_summary_entropy(
    all_stats: dict, out_path: Path,
    example_meta: dict = None,
    n_heads_label: str = None,
):
    tasks = list(all_stats.keys())
    data_full = [
        collect_all_values(all_stats[t], "full_entropy")
        for t in tasks
    ]
    data_nsl = [
        collect_all_values(all_stats[t], "nonlocal_entropy")
        for t in tasks
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _task_boxplot(
        ax, tasks, data_full, data_nsl,
        "Entropy (nats)",
        "Entropy Distribution per Task",
        example_meta=example_meta,
        n_heads_label=n_heads_label,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_task_summary_mass(
    all_stats: dict, pct: int, out_path: Path,
    example_meta: dict = None,
    n_heads_label: str = None,
):
    tasks = list(all_stats.keys())
    data_full = [
        collect_all_values(
            all_stats[t], f"full_top{pct}pct_mass",
        ) for t in tasks
    ]
    data_nsl = [
        collect_all_values(
            all_stats[t], f"nonlocal_top{pct}pct_mass",
        ) for t in tasks
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _task_boxplot(
        ax, tasks, data_full, data_nsl,
        f"Top {pct}% Mass",
        f"Top {pct}% Attention Mass Distribution per Task",
        ylim=(-0.05, 1.08),
        example_meta=example_meta,
        n_heads_label=n_heads_label,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Per-layer bar plots ──────────────────────────

def plot_layer_bars(
    task: str, layer_avgs: dict,
    metric_full: str, metric_nsl: str,
    ylabel: str, title_suffix: str,
    out_path: Path, ylim=None,
):
    layers = sorted(layer_avgs.keys())
    full = [layer_avgs[l][metric_full] for l in layers]
    nsl = [layer_avgs[l][metric_nsl] for l in layers]

    x = np.arange(len(layers))
    w = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, full, w, label="Full", color=C_FULL)
    ax.bar(x + w / 2, nsl, w, label="Nonlocal", color=C_NSL)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(l) for l in layers], fontsize=8,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task} — {title_suffix}")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Per-example box plots ────────────────────────

def plot_example_boxplots(
    task: str,
    per_example_stats: List[Tuple[dict, dict]],
    metric_full: str, metric_nsl: str,
    ylabel: str, title_suffix: str,
    out_path: Path, ylim=None,
):
    """Box plot: one pair of boxes per example."""
    n = len(per_example_stats)
    if n == 0:
        return

    data_full = []
    data_nsl = []
    xlabels = []
    for meta, stats in per_example_stats:
        data_full.append(
            collect_all_values(stats, metric_full)
        )
        data_nsl.append(
            collect_all_values(stats, metric_nsl)
        )
        eid = meta.get("example_id", "?")
        slen = meta.get("sequence_length", 0)
        xlabels.append(f"{eid}\n({slen:,} tok)")

    spacing = 1.0
    box_width = 0.3
    pos_full = [i * spacing - 0.2 for i in range(n)]
    pos_nsl = [i * spacing + 0.2 for i in range(n)]

    bp_kw = dict(
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        showmeans=False,
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=0.8),
    )

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5.5))

    bp_f = ax.boxplot(
        data_full, positions=pos_full,
        flierprops=dict(
            marker="o", markersize=3, alpha=0.4,
            markerfacecolor=C_FULL, markeredgecolor="none",
        ),
        **bp_kw,
    )
    for p in bp_f["boxes"]:
        p.set_facecolor(C_FULL)
        p.set_alpha(0.8)
    for w in bp_f["whiskers"] + bp_f["caps"]:
        w.set_color(C_FULL)

    bp_n = ax.boxplot(
        data_nsl, positions=pos_nsl,
        flierprops=dict(
            marker="o", markersize=3, alpha=0.4,
            markerfacecolor=C_NSL, markeredgecolor="none",
        ),
        **bp_kw,
    )
    for p in bp_n["boxes"]:
        p.set_facecolor(C_NSL)
        p.set_alpha(0.8)
    for w in bp_n["whiskers"] + bp_n["caps"]:
        w.set_color(C_NSL)

    # Mean diamonds
    for i in range(n):
        ax.scatter(
            pos_full[i], np.mean(data_full[i]),
            marker="D", s=28, color="white",
            edgecolors=C_FULL, linewidths=1, zorder=5,
        )
        ax.scatter(
            pos_nsl[i], np.mean(data_nsl[i]),
            marker="D", s=28, color="white",
            edgecolors=C_NSL, linewidths=1, zorder=5,
        )

    centers = [i * spacing for i in range(n)]
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{task} — {title_suffix} (1024 heads/example)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, 1.02,
        "Box: Q25\u2013Q75  |  Line: median  |  "
        "\u25c7: mean  |  Dots: outliers",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.5, color="#888888",
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        lo, hi = ax.get_ylim()
        margin = (hi - lo) * 0.08
        ax.set_ylim(lo - margin, hi + margin)

    legend_handles = [
        mpatches.Patch(
            facecolor=C_FULL, alpha=0.8, label="Full",
        ),
        mpatches.Patch(
            facecolor=C_NSL, alpha=0.8, label="Nonlocal",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Cross-example stability plots ────────────────

def plot_stability(
    task: str,
    per_example_stats: List[Tuple[dict, dict]],
    metric: str, ylabel: str,
    out_path: Path, ylim=None,
):
    """Median + Q25-Q75 band + Q5-Q95 whiskers across examples.

    X-axis: example index (sorted by seq_len, which is
    the order in per_example_stats).
    """
    n = len(per_example_stats)
    if n == 0:
        return

    # Gather per-example distributions
    medians, q25s, q75s, q05s, q95s = [], [], [], [], []
    xlabels = []
    for meta, stats in per_example_stats:
        vals = np.array(collect_all_values(stats, metric))
        if len(vals) == 0:
            continue
        medians.append(np.median(vals))
        q25s.append(np.percentile(vals, 25))
        q75s.append(np.percentile(vals, 75))
        q05s.append(np.percentile(vals, 5))
        q95s.append(np.percentile(vals, 95))
        slen = meta.get("sequence_length", 0)
        xlabels.append(f"{slen:,}\ntok")

    x = np.arange(len(medians))
    is_full = metric.startswith("full_")
    color = C_FULL if is_full else C_NSL
    label = "Full" if is_full else "Nonlocal"

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))

    # Q5-Q95 thin whiskers
    ax.fill_between(
        x, q05s, q95s, alpha=0.12, color=color,
        label="Q5\u2013Q95",
    )
    # Q25-Q75 band
    ax.fill_between(
        x, q25s, q75s, alpha=0.35, color=color,
        label="Q25\u2013Q75",
    )
    # Median line
    ax.plot(
        x, medians, color=color, linewidth=2,
        marker="o", markersize=6, label="Median",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Example (sorted by sequence length)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{task} — {label} {_metric_display(metric)} "
        f"Stability",
        fontsize=13, fontweight="bold",
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _metric_display(metric: str) -> str:
    """Human-readable metric name for titles."""
    m = metric.replace("full_", "").replace("nonlocal_", "")
    return {
        "entropy": "Entropy",
        "top1pct_mass": "Top 1% Mass",
        "top5pct_mass": "Top 5% Mass",
    }.get(m, m)


def _metric_ylabel(metric: str) -> str:
    m = metric.replace("full_", "").replace("nonlocal_", "")
    return {
        "entropy": "Entropy (nats)",
        "top1pct_mass": "Top 1% Mass",
        "top5pct_mass": "Top 5% Mass",
    }.get(m, m)


# ── Pooled cross-task data builder ───────────────

def _build_pooled_stats(
    all_stats: dict,
    per_example_data: Dict[str, List[Tuple[dict, dict]]],
) -> dict:
    """Build pooled stats dict for cross-task plots.

    When per-example data exists for a task, merge all
    examples into a single dict so box plots show ~N*1024
    heads. Otherwise fall back to scout-only stats.
    """
    pooled = {}
    for task, scout_stats in all_stats.items():
        pe = per_example_data.get(task, [])
        if pe:
            merged = {}
            for _meta, stats in pe:
                for lk, heads in stats.items():
                    if lk not in merged:
                        merged[lk] = {}
                    for hk, hs in heads.items():
                        merged[lk].setdefault(hk, [])
                        merged[lk][hk].append(hs)
            # Flatten: each head gets all values as separate
            # entries (we abuse the dict structure by creating
            # synthetic head keys per example)
            flat = {}
            for lk, heads in merged.items():
                flat[lk] = {}
                for hk, hs_list in heads.items():
                    for ei, hs in enumerate(hs_list):
                        flat[lk][f"{hk}_ex{ei}"] = hs
            pooled[task] = flat
        else:
            pooled[task] = scout_stats
    return pooled


def _heads_per_task_label(
    all_stats: dict,
    per_example_data: Dict[str, List[Tuple[dict, dict]]],
) -> str:
    """Build label like '~5120 heads/task' or '1024 heads'."""
    counts = []
    for task in all_stats:
        pe = per_example_data.get(task, [])
        if pe:
            n = sum(
                len(collect_all_values(s, "full_entropy"))
                for _, s in pe
            )
        else:
            n = len(
                collect_all_values(
                    all_stats[task], "full_entropy",
                )
            )
        counts.append(n)
    if not counts:
        return "1024 heads"
    avg = int(np.mean(counts))
    if all(c == counts[0] for c in counts):
        return f"{counts[0]} heads/task"
    return f"~{avg} heads/task"


# ── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize head statistics"
    )
    parser.add_argument(
        "--stats-dir", type=Path, required=True,
        help="Directory with per-task JSON stat files",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/head_statistics"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if not args.stats_dir.exists():
        print(f"ERROR: {args.stats_dir} not found")
        sys.exit(1)

    all_stats, example_meta = load_all_stats(
        args.stats_dir,
    )
    if not all_stats:
        print(f"No JSON files in {args.stats_dir}")
        sys.exit(1)

    print(f"Tasks: {list(all_stats.keys())}")
    if example_meta:
        print(f"Metadata found for: "
              f"{list(example_meta.keys())}")

    # Load per-example data for all tasks
    per_example_data: Dict[
        str, List[Tuple[dict, dict]]
    ] = {}
    for task in all_stats:
        pe = load_per_example_stats(args.stats_dir, task)
        if pe:
            per_example_data[task] = pe
    if per_example_data:
        print(f"Per-example data for: "
              f"{list(per_example_data.keys())}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    per_task_dir = out / "per_task"
    per_task_dir.mkdir(exist_ok=True)

    # ── Cross-task box plots (pooled if available) ──
    if per_example_data:
        pooled = _build_pooled_stats(
            all_stats, per_example_data,
        )
        label = _heads_per_task_label(
            all_stats, per_example_data,
        )
    else:
        pooled = all_stats
        label = None

    print("Saving task summary box plots...")
    plot_task_summary_entropy(
        pooled, out / "task_entropy.png",
        example_meta=example_meta,
        n_heads_label=label,
    )
    plot_task_summary_mass(
        pooled, 1, out / "task_top1pct_mass.png",
        example_meta=example_meta,
        n_heads_label=label,
    )
    plot_task_summary_mass(
        pooled, 5, out / "task_top5pct_mass.png",
        example_meta=example_meta,
        n_heads_label=label,
    )

    # ── Per-task plots ──
    for task, stats in all_stats.items():
        print(f"  {task}...")
        td = per_task_dir / task
        td.mkdir(exist_ok=True)

        # Layer bar plots (scout only)
        layer_avgs = extract_layer_averages(stats)
        plot_layer_bars(
            task, layer_avgs,
            "full_entropy", "nonlocal_entropy",
            "Entropy (nats)",
            "Average Entropy by Layer",
            td / "layer_entropy.png",
        )
        plot_layer_bars(
            task, layer_avgs,
            "full_top1pct_mass",
            "nonlocal_top1pct_mass",
            "Top 1% Mass",
            "Average Top 1% Mass by Layer",
            td / "layer_top1pct_mass.png",
            ylim=(0, 1.05),
        )
        plot_layer_bars(
            task, layer_avgs,
            "full_top5pct_mass",
            "nonlocal_top5pct_mass",
            "Top 5% Mass",
            "Average Top 5% Mass by Layer",
            td / "layer_top5pct_mass.png",
            ylim=(0, 1.05),
        )

        # Per-example plots (if data exists)
        pe = per_example_data.get(task, [])
        if not pe:
            continue

        print(f"    Per-example plots ({len(pe)} examples)")

        # Example box plots
        for (m_full, m_nsl, suffix, ylabel,
             title, ylim) in METRIC_PAIRS:
            plot_example_boxplots(
                task, pe, m_full, m_nsl,
                ylabel,
                f"{title} per Example",
                td / f"example_{suffix}.png",
                ylim=ylim,
            )

        # Stability plots
        for metric in METRICS:
            ylabel = _metric_ylabel(metric)
            ylim = None
            if "mass" in metric:
                ylim = (-0.05, 1.08)
            plot_stability(
                task, pe, metric, ylabel,
                td / f"stability_{metric}.png",
                ylim=ylim,
            )

    print(f"\nDone. Output in {out}/")


if __name__ == "__main__":
    main()
