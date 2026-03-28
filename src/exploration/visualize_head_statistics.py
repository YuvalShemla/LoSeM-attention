#!/usr/bin/env python3
"""
Visualize pre-computed head statistics from extraction.

Reads head_statistics JSON files and produces:
  1. Per-task summary box plots (distribution across all heads)
  2. Per-layer bar plots (mean across heads) for each task

Usage:
  python -m src.exploration.visualize_head_statistics \
      --stats-dir llama3.1_8b \
      --out-dir results/head_statistics
"""

import argparse
import json
import sys
from pathlib import Path

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
        stats[p.stem] = data
    return stats, example_meta


def collect_all_values(task_stats: dict, metric: str) -> list:
    """Collect metric values from every head in every layer."""
    vals = []
    for heads in task_stats.values():
        for hs in heads.values():
            if metric in hs:
                vals.append(hs[metric])
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

def _task_boxplot(ax, tasks, data_full, data_nsl, ylabel, title, ylim=None, example_meta=None):
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
            xlabels.append(f"{t}\n{ex_id}  ({n_tok:,} tok)")
        else:
            xlabels.append(t)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (1024 heads)", fontsize=13, fontweight="bold", pad=20)
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
        # Add padding so outlier dots aren't clipped
        lo, hi = ax.get_ylim()
        margin = (hi - lo) * 0.08
        ax.set_ylim(lo - margin, hi + margin)

    legend_handles = [
        mpatches.Patch(facecolor=C_FULL, alpha=0.8, label="Full"),
        mpatches.Patch(facecolor=C_NSL, alpha=0.8, label="Nonlocal"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)


def plot_task_summary_entropy(all_stats: dict, out_path: Path, example_meta: dict = None):
    tasks = list(all_stats.keys())
    data_full = [collect_all_values(all_stats[t], "full_entropy") for t in tasks]
    data_nsl = [collect_all_values(all_stats[t], "nonlocal_entropy") for t in tasks]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _task_boxplot(ax, tasks, data_full, data_nsl,
                  "Entropy (nats)", "Entropy Distribution per Task",
                  example_meta=example_meta)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_task_summary_mass(all_stats: dict, pct: int, out_path: Path, example_meta: dict = None):
    tasks = list(all_stats.keys())
    data_full = [collect_all_values(all_stats[t], f"full_top{pct}pct_mass") for t in tasks]
    data_nsl = [collect_all_values(all_stats[t], f"nonlocal_top{pct}pct_mass") for t in tasks]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _task_boxplot(ax, tasks, data_full, data_nsl,
                  f"Top {pct}% Mass",
                  f"Top {pct}% Attention Mass Distribution per Task",
                  ylim=(-0.05, 1.08), example_meta=example_meta)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Per-layer bar plots ──────────────────────────

def plot_layer_bars(task: str, layer_avgs: dict,
                    metric_full: str, metric_nsl: str,
                    ylabel: str, title_suffix: str,
                    out_path: Path, ylim=None):
    layers = sorted(layer_avgs.keys())
    full = [layer_avgs[l][metric_full] for l in layers]
    nsl = [layer_avgs[l][metric_nsl] for l in layers]

    x = np.arange(len(layers))
    w = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, full, w, label="Full", color=C_FULL)
    ax.bar(x + w / 2, nsl, w, label="Nonlocal", color=C_NSL)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task} — {title_suffix}")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    per_task_dir = out / "per_task"
    per_task_dir.mkdir(exist_ok=True)

    # Task-level box plots
    print("Saving task summary box plots...")
    plot_task_summary_entropy(
        all_stats, out / "task_entropy.png",
        example_meta=example_meta,
    )
    plot_task_summary_mass(
        all_stats, 1, out / "task_top1pct_mass.png",
        example_meta=example_meta,
    )
    plot_task_summary_mass(
        all_stats, 5, out / "task_top5pct_mass.png",
        example_meta=example_meta,
    )

    # Per-task layer bar plots
    for task, stats in all_stats.items():
        print(f"  {task}...")
        td = per_task_dir / task
        td.mkdir(exist_ok=True)
        layer_avgs = extract_layer_averages(stats)

        plot_layer_bars(
            task, layer_avgs,
            "full_entropy", "nonlocal_entropy",
            "Entropy (nats)", "Average Entropy by Layer",
            td / "layer_entropy.png",
        )
        plot_layer_bars(
            task, layer_avgs,
            "full_top1pct_mass", "nonlocal_top1pct_mass",
            "Top 1% Mass", "Average Top 1% Mass by Layer",
            td / "layer_top1pct_mass.png",
            ylim=(0, 1.05),
        )
        plot_layer_bars(
            task, layer_avgs,
            "full_top5pct_mass", "nonlocal_top5pct_mass",
            "Top 5% Mass", "Average Top 5% Mass by Layer",
            td / "layer_top5pct_mass.png",
            ylim=(0, 1.05),
        )

    print(f"\nDone. Output in {out}/")


if __name__ == "__main__":
    main()
