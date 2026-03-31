#!/usr/bin/env python3
"""
CLI for generating exploration dashboards.

Produces two dashboards per head (global + pairwise),
then aggregated views across heads and tasks.

Usage:
  python -m src.exploration.run_exploration \
    --tasks math_calc code_run

  python -m src.exploration.run_exploration --all
"""

import argparse
import yaml
from datetime import datetime
from pathlib import Path

from ..experiment.data_loader import (
    load_examples, load_task_metadata,
)
from .dashboard_global import (
    compute_global_data, create_global_dashboard,
)
from .dashboard_pairwise import (
    compute_pairwise_dashboard_data,
    create_pairwise_dashboard,
)
from .dashboard_rope_comparison import (
    _compute_variant as compute_rope_variant,
    create_rope_comparison_dashboard,
)
from .aggregation import (
    aggregate_global_data,
    aggregate_pairwise_data,
)


def _last_query_positions(
    seq_len: int, n: int,
) -> list:
    """Take the last n token positions as queries."""
    start = max(0, seq_len - n)
    return list(range(start, seq_len))


def _resolve_heads(config, vectors_dir, task):
    """Resolve which heads to analyze for a task.

    Returns list of dicts with keys:
        layer, q_head, kv_head, selection_label,
        nonlocal_entropy.
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
            return [{
                "layer": ecfg.get("layer", 17),
                "q_head": ecfg.get("q_head", 0),
                "kv_head": ecfg.get("kv_head", 0),
                "selection_label": None,
                "nonlocal_entropy": None,
            }]
        return [
            {
                "layer": h["layer"],
                "q_head": h["q_head"],
                "kv_head": h["kv_head"],
                "selection_label": h.get(
                    "selection_label",
                ),
                "nonlocal_entropy": h.get(
                    "nonlocal_entropy",
                ),
            }
            for h in heads
        ]

    # custom mode
    return [{
        "layer": ecfg.get("layer", 17),
        "q_head": ecfg.get("q_head", 0),
        "kv_head": ecfg.get("kv_head", 0),
        "selection_label": None,
        "nonlocal_entropy": None,
    }]


def _run_aggregation(
    all_global: list,
    all_pairwise: list,
    task_dir: Path,
    title_prefix: str,
    ema_span: int = 200,
):
    """Run aggregation for a set of head data."""
    if len(all_global) < 2:
        return

    methods = ["mean", "median", "variance"]
    for method in methods:
        agg_dir = task_dir / f"aggregated_{method}"
        agg_dir.mkdir(parents=True, exist_ok=True)

        label = f"{title_prefix} — {method}"

        agg_g = aggregate_global_data(
            all_global, method,
        )
        create_global_dashboard(
            agg_g, label,
            agg_dir / "global_dashboard.png",
        )

        agg_p = aggregate_pairwise_data(
            all_pairwise, method,
        )
        create_pairwise_dashboard(
            agg_p, label,
            agg_dir / "pairwise_dashboard.png",
            ema_span=ema_span,
        )
        print(f"      {method}: {agg_dir}")

    # Percentiles (p25, p75, p90)
    pct_dir = task_dir / "aggregated_percentiles"
    pct_dir.mkdir(parents=True, exist_ok=True)
    for pct in ["p25", "p75", "p90"]:
        agg_g = aggregate_global_data(all_global, pct)
        create_global_dashboard(
            agg_g,
            f"{title_prefix} — {pct}",
            pct_dir / f"global_dashboard_{pct}.png",
        )
        agg_p = aggregate_pairwise_data(all_pairwise, pct)
        create_pairwise_dashboard(
            agg_p,
            f"{title_prefix} — {pct}",
            pct_dir / f"pairwise_dashboard_{pct}.png",
            ema_span=ema_span,
        )
    print(f"      percentiles: {pct_dir}")


def run_exploration(config_path: str,
                    tasks: list = None,
                    vectors_dir: str = None):
    """Run exploration dashboards on .pt data."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ecfg = config.get("exploration", {})
    n_queries = ecfg.get("n_queries", 50)
    n_examples = ecfg.get("n_examples", 1)
    head_dim = config["model"]["head_dim"]
    use_rope = ecfg.get("use_rope", True)
    ema_span = config.get("pairwise", {}).get(
        "ema_span", 200,
    )

    data_cfg = config.get("data", {})
    vdir = vectors_dir or data_cfg.get(
        "vectors_dir", "data/vectors",
    )
    results_dir = Path(
        data_cfg.get("results_dir", "results")
    )

    if tasks is None:
        tasks = config.get("tasks", [])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_base = results_dir / f"exploration_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)

    # Collect across ALL tasks for global aggregation
    all_tasks_global = []
    all_tasks_pairwise = []

    for task in tasks:
        print(f"\n  Task: {task}")
        task_dir = out_base / task

        heads = _resolve_heads(config, vdir, task)
        print(f"    Heads: {len(heads)} — "
              + ", ".join(
                  f"L{h['layer']}H{h['q_head']}"
                  for h in heads
              ))

        task_global = []
        task_pairwise = []

        for head_info in heads:
            layer = head_info["layer"]
            q_head = head_info["q_head"]
            kv_head = head_info["kv_head"]
            sel_label = head_info.get("selection_label")
            nl_entropy = head_info.get("nonlocal_entropy")
            head_label = f"L{layer}H{q_head}"
            print(f"    {head_label}:")

            head_dir = task_dir / head_label
            head_dir.mkdir(parents=True, exist_ok=True)

            examples = list(load_examples(
                Path(vdir), task, layer,
                head=q_head, kv_head=kv_head,
                max_examples=n_examples,
                use_rope=use_rope,
            ))
            if not examples:
                print(f"      No data for {head_label}"
                      f", skipping")
                continue

            ex = examples[0]
            Q, K, V = ex["Q"], ex["K"], ex["V"]
            seq_len = Q.shape[0]
            last_qpos = _last_query_positions(
                seq_len, n_queries,
            )
            all_qpos = list(range(seq_len))
            print(f"      {seq_len:,} tokens, "
                  f"{len(last_qpos)} last queries, "
                  f"{len(all_qpos)} all queries")

            # Build title with head metadata
            meta_parts = []
            if sel_label:
                meta_parts.append(sel_label)
            if nl_entropy is not None:
                meta_parts.append(
                    f"ent={nl_entropy:.2f}"
                )
            meta_str = (
                f" ({', '.join(meta_parts)})"
                if meta_parts else ""
            )
            info = (
                f"{task} — {head_label}{meta_str}"
                f" ({seq_len:,} tok)"
            )

            # Global dashboard
            print(f"      Computing global analyses...")
            global_data = compute_global_data(
                Q, K, V, head_dim, last_qpos, config,
                all_query_positions=all_qpos,
            )
            print(f"      Creating global dashboard...")
            create_global_dashboard(
                global_data, info,
                head_dir / "global_dashboard.png",
            )

            # Pairwise dashboard
            print(f"      Computing pairwise analyses...")
            pairwise_data = compute_pairwise_dashboard_data(
                Q, K, head_dim, last_qpos, config,
            )
            print(f"      Creating pairwise dashboard...")
            create_pairwise_dashboard(
                pairwise_data, info,
                head_dir / "pairwise_dashboard.png",
                ema_span=ema_span,
            )

            task_global.append(global_data)
            task_pairwise.append(pairwise_data)

            # RoPE comparison dashboard
            # Reuse already-computed RoPE embedding + pairwise
            print(f"      Computing RoPE comparison "
                  f"(raw only)...")
            raw_examples = list(load_examples(
                Path(vdir), task, layer,
                head=q_head, kv_head=kv_head,
                max_examples=n_examples,
                use_rope=False,
            ))
            if raw_examples:
                raw_ex = raw_examples[0]
                Q_raw = raw_ex["Q"]
                K_raw = raw_ex["K"]
                V_raw = raw_ex["V"]

                rope_variant = {
                    "embedding": global_data.get("embedding"),
                    "pairwise": {
                        "qk": pairwise_data["qk"],
                        "kk": pairwise_data["kk"],
                    },
                    "concentration": global_data.get(
                        "concentration",
                    ),
                    "entropy": global_data.get("entropy"),
                    "bias": global_data.get("bias"),
                }
                raw_variant = compute_rope_variant(
                    Q_raw, K_raw, head_dim, last_qpos,
                    config,
                    V=V_raw,
                    all_query_positions=all_qpos,
                )
                create_rope_comparison_dashboard(
                    rope_variant, raw_variant, info,
                    head_dir / "rope_comparison.png",
                )
            else:
                print(f"      No raw vectors, skipping "
                      f"RoPE comparison")

            print(f"      Saved: {head_dir}")

        # Per-task aggregation
        if len(task_global) >= 2:
            print(f"\n    Aggregating {task}...")
            _run_aggregation(
                task_global, task_pairwise,
                task_dir, task, ema_span,
            )

        all_tasks_global.extend(task_global)
        all_tasks_pairwise.extend(task_pairwise)

    # Global (all-tasks) aggregation
    if len(all_tasks_global) >= 2 and len(tasks) > 1:
        print(f"\n  All-tasks aggregation...")
        _run_aggregation(
            all_tasks_global, all_tasks_pairwise,
            out_base / "all_tasks", "all_tasks",
            ema_span,
        )

    print(f"\n  Exploration saved: {out_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate exploration dashboards.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to analyze (default: from config).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tasks from config.",
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
    if args.all:
        tasks = None

    run_exploration(
        args.config,
        tasks=tasks,
        vectors_dir=args.vectors_dir,
    )


if __name__ == "__main__":
    main()
