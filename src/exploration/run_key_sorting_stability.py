#!/usr/bin/env python3
"""
CLI for key sorting stability + PCA subspace analysis.

Usage:
  python -m src.exploration.run_key_sorting_stability --all
  python -m src.exploration.run_key_sorting_stability --tasks math_calc code_run
  python -m src.exploration.run_key_sorting_stability --all --pca-only
"""

import argparse
import yaml
from datetime import datetime
from pathlib import Path

from ..evaluation.data_loader import (
    load_examples, load_task_metadata,
)
from .key_sorting_stability import (
    compute_all_analyses,
    create_stability_dashboard,
    create_persistence_dashboard,
    create_agreement_dashboard,
    compute_pca_projection_stability,
    create_pca_dashboard,
)


def _resolve_heads(config, vectors_dir, task):
    ecfg = config.get("exploration", {})
    mode = ecfg.get("head_mode", "custom")

    if mode == "selected_heads":
        meta = load_task_metadata(
            Path(vectors_dir), task,
        )
        heads = meta.get("selected_heads", [])
        if not heads:
            return [{
                "layer": ecfg.get("layer", 17),
                "q_head": ecfg.get("q_head", 0),
                "kv_head": ecfg.get("kv_head", 0),
                "selection_label": None,
            }]
        return [
            {
                "layer": h["layer"],
                "q_head": h["q_head"],
                "kv_head": h["kv_head"],
                "selection_label": h.get(
                    "selection_label",
                ),
            }
            for h in heads
        ]

    return [{
        "layer": ecfg.get("layer", 17),
        "q_head": ecfg.get("q_head", 0),
        "kv_head": ecfg.get("kv_head", 0),
        "selection_label": None,
    }]


def run(config_path: str,
        tasks: list = None,
        vectors_dir: str = None,
        pca_only: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ecfg = config.get("exploration", {})
    head_dim = config["model"]["head_dim"]
    use_rope = ecfg.get("use_rope", True)
    n_examples = ecfg.get("n_examples", 1)

    data_cfg = config.get("data", {})
    vdir = vectors_dir or data_cfg.get(
        "vectors_dir", "data/vectors",
    )
    results_dir = Path(
        data_cfg.get("results_dir", "results"),
    )

    if tasks is None:
        tasks = config.get("tasks", [])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_base = results_dir / f"key_sorting_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"\n  Task: {task}")
        task_dir = out_base / task

        heads = _resolve_heads(config, vdir, task)
        print(f"    Heads: {len(heads)} — "
              + ", ".join(
                  f"L{h['layer']}H{h['q_head']}"
                  for h in heads
              ))

        for head_info in heads:
            layer = head_info["layer"]
            q_head = head_info["q_head"]
            kv_head = head_info["kv_head"]
            sel_label = head_info.get("selection_label")
            head_label = f"L{layer}H{q_head}"
            meta = f" ({sel_label})" if sel_label else ""
            print(f"\n    {head_label}{meta}:")

            head_dir = task_dir / head_label
            head_dir.mkdir(parents=True, exist_ok=True)

            examples = list(load_examples(
                Path(vdir), task, layer,
                head=q_head, kv_head=kv_head,
                max_examples=n_examples,
                use_rope=use_rope,
            ))
            if not examples:
                print(f"      No data, skipping")
                continue

            ex = examples[0]
            Q, K = ex["Q"], ex["K"]
            seq_len = Q.shape[0]
            info = (
                f"{task} — {head_label}{meta}"
                f" ({seq_len:,} tok)"
            )
            print(f"      {seq_len:,} tokens")

            if not pca_only:
                # Full analysis: stability, persistence,
                # agreement
                print(f"      Computing all analyses...")
                (global_data, clustered_data,
                 persistence_data,
                 agreement_data) = compute_all_analyses(
                    Q, K, head_dim,
                )

                print(f"      Creating dashboards...")
                create_stability_dashboard(
                    global_data, clustered_data,
                    info,
                    head_dir / "stability.png",
                )
                create_persistence_dashboard(
                    persistence_data, info,
                    head_dir / "persistence.png",
                )
                create_agreement_dashboard(
                    agreement_data, info,
                    head_dir / "agreement.png",
                )

            # PCA subspace projection
            print(f"      Computing PCA projection...")
            pca_data = compute_pca_projection_stability(
                Q, K, head_dim,
            )

            print(f"      Creating PCA dashboard...")
            create_pca_dashboard(
                pca_data, info,
                head_dir / "pca_subspace.png",
            )

            print(f"      Saved: {head_dir}")

    print(f"\n  Done: {out_base}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Key sorting stability + PCA analysis."
        ),
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tasks from config.",
    )
    parser.add_argument(
        "--pca-only", action="store_true",
        help="Only run PCA subspace analysis.",
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

    run(
        args.config,
        tasks=tasks,
        vectors_dir=args.vectors_dir,
        pca_only=args.pca_only,
    )


if __name__ == "__main__":
    main()
