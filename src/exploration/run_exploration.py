#!/usr/bin/env python3
"""
CLI for generating exploration dashboards.

Loads .pt data, runs configured analysis plots,
saves to results/exploration_{date}/.

Supports two head modes:
  selected_heads — iterate all selected heads per task
                   (from metadata.json)
  custom         — single hardcoded head from config

Usage:
  python -m src.exploration.run_exploration \
    --tasks math_calc code_run \
    --plots attention_concentration entropy

  python -m src.exploration.run_exploration --all
"""

import argparse
import yaml
from datetime import datetime
from pathlib import Path

from ..experiment.data_loader import (
    load_examples, load_task_metadata,
)
from .attention_concentration import (
    compute_concentration_data, plot_concentration,
)
from .entropy_distribution import (
    compute_entropy_data, plot_entropy,
)
from .kv_norm_correlation import (
    compute_kv_norm_data, plot_kv_norms,
)
from .topk_vs_sampling_bias import (
    compute_bias_data, plot_bias_comparison,
)


PLOT_REGISTRY = {
    "attention_concentration": (
        compute_concentration_data,
        plot_concentration,
    ),
    "entropy_distribution": (
        compute_entropy_data, plot_entropy,
    ),
    "kv_norm_correlation": (
        compute_kv_norm_data, plot_kv_norms,
    ),
    "topk_vs_sampling_bias": (
        compute_bias_data, plot_bias_comparison,
    ),
}


def _last_query_positions(
    seq_len: int, n: int,
) -> list:
    """Take the last n token positions as queries."""
    start = max(0, seq_len - n)
    return list(range(start, seq_len))


def _resolve_heads(config, vectors_dir, task):
    """Resolve which heads to analyze for a task.

    Returns list of (layer, q_head, kv_head) tuples.
    """
    ecfg = config.get("exploration", {})
    mode = ecfg.get("head_mode", "custom")

    if mode == "selected_heads":
        meta = load_task_metadata(
            Path(vectors_dir), task,
        )
        heads = meta.get("selected_heads", [])
        if not heads:
            print(f"    No selected_heads in metadata "
                  f"for {task}, falling back to custom")
            return [(
                ecfg.get("layer", 17),
                ecfg.get("q_head", 0),
                ecfg.get("kv_head", 0),
            )]
        return [
            (h["layer"], h["q_head"], h["kv_head"])
            for h in heads
        ]

    # custom mode
    return [(
        ecfg.get("layer", 17),
        ecfg.get("q_head", 0),
        ecfg.get("kv_head", 0),
    )]


def _run_plots_for_head(
    Q, K, V, head_dim, qpos,
    label, out_dir, plots, config,
):
    """Run all requested plots for a single head."""
    ecfg = config.get("exploration", {})
    n_sink = ecfg.get(
        "attention_sink", {}
    ).get("n_sink_tokens", 1)
    local_window = ecfg.get(
        "local_window", {}
    ).get("size", 1024)
    seed = ecfg.get("seed", 42)

    conc_cfg = config.get("concentration", {})
    top_k_values = conc_cfg.get(
        "top_k_values", [10, 50, 100, 200, 500]
    )
    kv_cfg = config.get("kv_norms", {})
    top_pct = kv_cfg.get("top_pct", 10.0)
    bias_cfg = config.get("bias_comparison", {})
    budget_fractions = bias_cfg.get(
        "budget_fractions",
        [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
    )

    for pname in plots:
        if pname not in PLOT_REGISTRY:
            print(f"      Unknown plot: {pname}")
            continue
        print(f"      {pname}...")
        out_path = out_dir / f"{pname}.png"

        if pname == "attention_concentration":
            data = compute_concentration_data(
                Q, K, head_dim, qpos,
                top_k_values=top_k_values,
            )
            plot_concentration(
                data, out_path, title=label,
            )
        elif pname == "entropy_distribution":
            data = compute_entropy_data(
                Q, K, head_dim, qpos,
                n_sink, local_window,
            )
            plot_entropy(
                data, out_path, title=label,
            )
        elif pname == "kv_norm_correlation":
            data = compute_kv_norm_data(
                Q, K, V, head_dim, qpos,
                top_pct=top_pct,
            )
            plot_kv_norms(
                data, out_path, title=label,
            )
        elif pname == "topk_vs_sampling_bias":
            data = compute_bias_data(
                Q, K, V, head_dim, qpos,
                budget_fractions=budget_fractions,
                seed=seed,
            )
            plot_bias_comparison(
                data, out_path, title=label,
            )


def run_exploration(config_path: str,
                    tasks: list = None,
                    plots: list = None,
                    vectors_dir: str = None):
    """Run exploration analysis on .pt data."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ecfg = config.get("exploration", {})
    n_queries = ecfg.get("n_queries", 200)
    n_examples = ecfg.get("n_examples", 1)
    head_dim = config["model"]["head_dim"]

    data_cfg = config.get("data", {})
    vdir = vectors_dir or data_cfg.get(
        "vectors_dir", "data/vectors",
    )
    results_dir = Path(
        data_cfg.get("results_dir", "results")
    )

    if tasks is None:
        tasks = config.get("tasks", [])
    if plots is None:
        plots = config.get(
            "plots", list(PLOT_REGISTRY.keys())
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_base = results_dir / f"exploration_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"\n  Task: {task}")
        task_dir = out_base / task

        heads = _resolve_heads(config, vdir, task)
        print(f"    Heads: {len(heads)} — "
              + ", ".join(
                  f"L{l}H{q}" for l, q, _ in heads
              ))

        for layer, q_head, kv_head in heads:
            head_label = f"L{layer}H{q_head}"
            print(f"    {head_label}:")

            # Per-head subdirectory when multiple heads
            if len(heads) > 1:
                head_dir = task_dir / head_label
            else:
                head_dir = task_dir
            head_dir.mkdir(parents=True, exist_ok=True)

            examples = list(load_examples(
                Path(vdir), task, layer,
                head=q_head, kv_head=kv_head,
                max_examples=n_examples,
            ))
            if not examples:
                print(f"      No data for {head_label}"
                      f", skipping")
                continue

            ex = examples[0]
            Q, K, V = ex["Q"], ex["K"], ex["V"]
            seq_len = Q.shape[0]
            qpos = _last_query_positions(
                seq_len, n_queries,
            )
            print(f"      {seq_len} tokens, "
                  f"{len(qpos)} queries")

            label = f"{task} — {head_label}"
            _run_plots_for_head(
                Q, K, V, head_dim, qpos,
                label, head_dir, plots, config,
            )

    print(f"\n  Exploration saved: {out_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate exploration plots.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to analyze (default: from config).",
    )
    parser.add_argument(
        "--plots", nargs="+", default=None,
        choices=list(PLOT_REGISTRY.keys()),
        help="Which plots to generate.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all plots on all tasks.",
    )
    parser.add_argument(
        "--vectors-dir", default=None,
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parent
            / "exploration_config.yaml"
        ),
    )
    args = parser.parse_args()

    tasks = args.tasks
    plots = args.plots
    if args.all:
        tasks = None
        plots = None

    run_exploration(
        args.config,
        tasks=tasks,
        plots=plots,
        vectors_dir=args.vectors_dir,
    )


if __name__ == "__main__":
    main()
