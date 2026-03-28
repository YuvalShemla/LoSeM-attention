"""
Experiment runner: load .pt data, evaluate methods, plot.

Handles the full lifecycle across multiple tasks.
Baselines are auto-included. Results organized into
per_task/ subfolders and overview/ summaries.

Usage:
  python -m src.experiment.run_experiment \\
    --algorithms meanq kmeans \\
    --tasks math_calc code_run \\
    --name grouping_comparison_v1
"""

import argparse
import gc
import json
import logging
import sys
import time
import csv
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..algorithms import METHOD_REGISTRY
from .data_loader import (
    load_examples, count_examples, discover_examples,
)
from .evaluator import (
    evaluate_query, aggregate_results,
    aggregate_query_stats,
)
from .plotting import (
    plot_experiment, plot_overview, setup_style,
)

log = logging.getLogger("experiment")


def _resolve_methods(algo_names, algo_configs):
    """Expand algorithm configs into instances."""
    methods = []
    for name in algo_names:
        spec = METHOD_REGISTRY[name]
        cfg = algo_configs.get(name, {})
        methods.extend(
            spec.cls.expand_from_config(cfg)
        )
    return methods


def _last_query_positions(
    seq_len: int,
    n_queries: int,
) -> List[int]:
    """
    Take the last N token positions as queries.

    Deterministic — always evaluates the positions
    where the model would actually be generating.
    """
    start = max(0, seq_len - n_queries)
    return list(range(start, seq_len))


def _setup_logging():
    """Configure logging for experiment output."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(name)s] %(message)s"
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)


class Experiment:
    """Concrete experiment class for .pt data."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        name: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        vectors_dir: Optional[str] = None,
    ):
        _setup_logging()

        if config_path is None:
            config_path = (
                Path(__file__).parent
                / "experiment_config.yaml"
            )
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        exp = self.config["experiment"]
        data_cfg = self.config.get("data", {})

        if tasks:
            self.tasks = tasks
        elif "tasks" in self.config:
            self.tasks = self.config["tasks"]
        else:
            self.tasks = []

        vdir = vectors_dir or data_cfg.get(
            "vectors_dir", "data/vectors/llama3.1_8b"
        )
        self.vectors_dir = Path(vdir)
        results_dir = Path(
            data_cfg.get("results_dir", "results")
        )

        self.seed = exp["seed"]
        self.n_queries = exp["n_queries"]
        self.n_examples = exp.get("n_examples", 10)
        self.budgets = exp["budget_sweep"]["absolute"]
        self.head_dim = self.config["model"]["head_dim"]
        self.n_sink = exp["attention_sink"][
            "n_sink_tokens"
        ]
        self.local_window = exp["local_window"]["size"]

        self.compute_statistics = exp.get(
            "compute_statistics", False
        )

        self.head_mode = exp.get(
            "head_mode", "selected_heads"
        )
        raw_layers = exp.get("layers", [17])
        if raw_layers == "all":
            self.layers = list(range(
                self.config["model"].get(
                    "num_layers", 32
                )
            ))
        else:
            self.layers = list(raw_layers)
        self.custom_heads = exp.get("custom_heads", [])

        mcfg = self.config["model"]
        self.gqa_group = (
            mcfg["num_q_heads"]
            // mcfg["num_kv_heads"]
        )
        self.n_q_heads = mcfg["num_q_heads"]

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.name = name or "exp"
        self.out_dir = (
            results_dir / f"{self.name}_{ts}"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, algo_names: List[str]):
        """Run experiment across all tasks."""
        t0 = time.time()
        rng = np.random.default_rng(self.seed)

        baselines = []
        for spec in METHOD_REGISTRY.values():
            if spec.kind == "baseline":
                baselines.extend(
                    spec.cls.expand_from_config({})
                )
        algo_cfgs = self.config.get(
            "algorithm_configs", {}
        )
        algorithms = _resolve_methods(
            algo_names, algo_cfgs,
        )
        methods = baselines + algorithms

        # --- Log experiment plan ---
        phase, _ = self._resolve_heads(
            self.tasks[0]
        )
        n_heads = self._count_heads()
        log.info(
            "Experiment: %d tasks, %s mode, "
            "%d examples/task, %d queries/example",
            len(self.tasks), self.head_mode,
            self.n_examples, self.n_queries,
        )
        log.info(
            "Heads: %d per task (%s data)",
            n_heads, phase,
        )
        log.info(
            "Methods: %s",
            ", ".join(m.name for m in methods),
        )
        log.info(
            "Budgets: %s",
            self.budgets,
        )
        log.info("Output: %s", self.out_dir)

        # --- Validate data before running ---
        self._validate_data()

        self._save_spec(methods, algo_names)

        all_rows = []
        per_task_agg = {}
        tasks_completed = []
        tasks_failed = []

        for ti, task in enumerate(self.tasks, 1):
            task_header = (
                f"Task {ti}/{len(self.tasks)}: {task}"
            )
            log.info("")
            log.info("=" * 50)
            log.info(task_header)
            log.info("=" * 50)
            try:
                task_rows, task_agg = self._run_task(
                    task, methods, algorithms, rng,
                )
                all_rows.extend(task_rows)
                per_task_agg[task] = task_agg
                tasks_completed.append(task)
            except Exception as e:
                log.error("FAILED %s: %s", task, e)
                tasks_failed.append(task)
                raise

        if per_task_agg:
            families = self._build_families(algorithms)
            ov_dir = self.out_dir / "overview"
            ov_dir.mkdir(exist_ok=True)
            plot_overview(
                per_task_agg, ov_dir,
                self.config.get("plotting", {}),
                self.budgets, families,
            )
            self._save_json(
                "overview/cross_task_stats.json",
                per_task_agg,
            )

        self._save_csv(all_rows)
        elapsed = time.time() - t0
        self._save_json("run.json", {
            "start_time": datetime.fromtimestamp(
                t0
            ).isoformat(),
            "end_time": datetime.now().isoformat(),
            "wall_clock_seconds": elapsed,
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
        })

        log.info("")
        log.info(
            "Done in %.0fs — %d tasks, %d total rows",
            elapsed, len(tasks_completed),
            len(all_rows),
        )
        log.info("Results: %s", self.out_dir)

    # ── Validation ──────────────────────────────────

    def _validate_data(self):
        """
        Check that all tasks have enough data before
        starting. Fail fast with a clear message.
        """
        for task in self.tasks:
            phase, heads = self._resolve_heads(task)
            n_available = count_examples(
                self.vectors_dir, task, phase,
            )
            if n_available == 0:
                if phase is None:
                    loc = f"{self.vectors_dir}/{task}/"
                else:
                    loc = (f"{self.vectors_dir}/"
                           f"{phase}/{task}/")
                raise FileNotFoundError(
                    f"No data for task '{task}' in "
                    f"{loc}. "
                    f"Run the extraction pipeline first."
                )
            if n_available < self.n_examples:
                raise FileNotFoundError(
                    f"Task '{task}' has {n_available} "
                    f"examples but config requires "
                    f"n_examples={self.n_examples}. "
                    f"Either extract more data or "
                    f"reduce n_examples in config."
                )
        log.info(
            "Data validated: all %d tasks have >= %d "
            "examples",
            len(self.tasks), self.n_examples,
        )

    # ── Per-task execution ──────────────────────────

    def _run_task(self, task, methods, algorithms,
                  rng):
        """Run all methods on one task."""
        task_t0 = time.time()
        task_dir = self.out_dir / "per_task" / task
        task_dir.mkdir(parents=True, exist_ok=True)

        phase, heads = self._resolve_heads(task)
        all_results = []
        rows = []

        for hi, (layer_idx, q_head, kv_head) in (
            enumerate(heads, 1)
        ):
            log.info(
                "  Head %d/%d: L%d H%d (kv=%d)",
                hi, len(heads),
                layer_idx, q_head, kv_head,
            )

            examples = list(load_examples(
                self.vectors_dir, task,
                layer_idx, q_head, kv_head,
                phase=phase,
                max_examples=self.n_examples,
            ))
            if not examples:
                raise FileNotFoundError(
                    f"No Q/K/V data for {task} "
                    f"L{layer_idx} H{q_head}. "
                    f"Check that layer_{layer_idx:02d}.pt "
                    f"contains Q_rope_head{q_head}."
                )

            for ei, ex in enumerate(examples, 1):
                Q, K, V = (
                    ex["Q"], ex["K"], ex["V"],
                )
                seq_len = Q.shape[0]
                qpos_list = _last_query_positions(
                    seq_len, self.n_queries,
                )

                log.info(
                    "    Example %d/%d: %s "
                    "(%d tok, %d queries)",
                    ei, len(examples),
                    ex["example_id"][:20],
                    seq_len, len(qpos_list),
                )

                for m in methods:
                    m.prepare(
                        K, V, self.head_dim,
                        queries=Q,
                        query_positions=qpos_list,
                        seed=self.seed,
                    )

                for qpos in qpos_list:
                    qr = evaluate_query(
                        Q[qpos], K[:qpos + 1],
                        V[:qpos + 1], methods,
                        self.budgets, self.head_dim,
                        self.n_sink,
                        self.local_window,
                        rng,
                        compute_statistics=(
                            self.compute_statistics
                        ),
                    )
                    all_results.append(qr)
                    for key, val in qr.items():
                        if key == "_query_stats":
                            continue
                        mname = key.rsplit("-", 1)[0]
                        mk = "baseline"
                        for m in methods:
                            if m.name == mname:
                                mk = m.kind
                                break
                        rows.append({
                            "task": task,
                            "layer": layer_idx,
                            "head": q_head,
                            "example_id": (
                                ex["example_id"][:12]
                            ),
                            "query_pos": qpos,
                            "method": key,
                            "method_kind": mk,
                            "budget": val["budget"],
                            "actual_budget": (
                                val["budget"]
                            ),
                            "rel_l2_error": (
                                val["error"]
                            ),
                            "seed": self.seed,
                        })

                del Q, K, V
                gc.collect()

        agg = aggregate_results(all_results)
        n_total = len(all_results)
        task_elapsed = time.time() - task_t0

        families = self._build_families(algorithms)
        plot_experiment(
            agg, task_dir,
            self.config.get("plotting", {}),
            self.budgets, families,
            title=f"{task} ({phase or 'flat'})",
            n_queries=n_total,
        )

        self._save_json(
            f"per_task/{task}/aggregated_stats.json",
            agg,
        )
        data_stats = {
            "task": task,
            "n_queries": n_total,
            "n_examples": len(
                set(r.get("example_id", "")
                    for r in rows
                    if r.get("task") == task)
            ),
            "heads": [
                {"layer": l, "q_head": h,
                 "kv_head": k}
                for l, h, k in heads
            ],
        }
        if self.compute_statistics:
            qstats = aggregate_query_stats(
                all_results,
            )
            data_stats["attention_statistics"] = qstats
            log.info(
                "  Attention stats: "
                "entropy=%.2f±%.2f, "
                "top1pct_mass=%.3f±%.3f",
                qstats.get(
                    "nonlocal_entropy_mean", 0
                ),
                qstats.get(
                    "nonlocal_entropy_std", 0
                ),
                qstats.get(
                    "nonlocal_top1pct_mass_mean",
                    0,
                ),
                qstats.get(
                    "nonlocal_top1pct_mass_std",
                    0,
                ),
            )
        self._save_json(
            f"per_task/{task}/data_statistics.json",
            data_stats,
        )

        log.info(
            "  Task complete: %d queries, %.1fs",
            n_total, task_elapsed,
        )
        return rows, agg

    # ── Head resolution ─────────────────────────────

    def _resolve_heads(self, task):
        """
        Determine (phase, heads) from head_mode config.

        Returns (phase_str, list of
                 (layer, q_head, kv_head)).
        phase is None for flat layout, or a string
        for legacy phase-based layout.
        """
        mode = self.head_mode

        if mode == "custom":
            ch = self.custom_heads
            if not ch:
                raise ValueError(
                    "head_mode='custom' but "
                    "custom_heads is empty"
                )
            triples = [
                (h["layer"], h["q_head"],
                 h["kv_head"])
                for h in ch
            ]
            phase = self._detect_phase(task)
            return phase, triples

        if mode == "selected_heads":
            # Try flat layout first
            flat_mp = (
                self.vectors_dir / task
                / "metadata.json"
            )
            # Fall back to old phase-based layout
            old_mp = (
                self.vectors_dir
                / "selected_heads" / task
                / "metadata.json"
            )
            if flat_mp.exists():
                mp = flat_mp
                phase = None
            elif old_mp.exists():
                mp = old_mp
                phase = "selected_heads"
            else:
                raise FileNotFoundError(
                    f"head_mode='selected_heads' but "
                    f"no metadata at {flat_mp} or "
                    f"{old_mp}. Run the extraction "
                    f"pipeline first, or switch to "
                    f"'all_heads' or 'custom'."
                )
            with open(mp) as f:
                meta = json.load(f)
            sel = meta.get("selected_heads", [])
            if not sel:
                raise ValueError(
                    f"metadata.json for {task} has no "
                    f"selected_heads list."
                )
            return phase, [
                (s["layer"], s["q_head"],
                 s["kv_head"])
                for s in sel
            ]

        if mode == "all_heads":
            triples = []
            for layer in self.layers:
                for h in range(self.n_q_heads):
                    triples.append(
                        (layer, h,
                         h // self.gqa_group)
                    )
            return "all_heads", triples

        raise ValueError(
            f"Unknown head_mode: '{mode}'. "
            f"Use 'all_heads', 'selected_heads', "
            f"or 'custom'."
        )

    def _count_heads(self):
        """Count heads for the current mode."""
        if self.head_mode == "all_heads":
            return len(self.layers) * self.n_q_heads
        if self.head_mode == "custom":
            return len(self.custom_heads)
        return 3  # typical selected_heads count

    def _detect_phase(self, task):
        """Check which directory layout exists.

        Returns None for flat layout, or phase string.
        """
        # Flat layout first
        flat = self.vectors_dir / task
        if flat.exists():
            return None
        for phase in ["selected_heads", "all_heads"]:
            d = self.vectors_dir / phase / task
            if d.exists():
                return phase
        return None

    # ── Plot families ───────────────────────────────

    def _build_families(self, algorithms):
        """Build plot family specs from algorithms."""
        colors = self.config.get("plotting", {}).get(
            "algorithm_color_families", []
        )
        markers = ["D", "X", "s", "v", "P", "^"]
        seen = {}
        families = []
        for m in algorithms:
            algo_name = None
            for aname, spec in METHOD_REGISTRY.items():
                if isinstance(m, spec.cls):
                    algo_name = aname
                    break
            pfx = algo_name or m.name
            if pfx in seen:
                continue
            ci = len(seen) % max(len(colors), 1)
            mi = len(seen) % len(markers)
            c = (
                colors[ci] if ci < len(colors)
                else {"topk": "#888", "hybrid": "#444"}
            )
            seen[pfx] = True
            tk_sweep = self.config.get(
                "algorithm_configs", {}
            ).get(algo_name, {}).get(
                "top_k_sweep", [0, 1, 3, 5, 10]
            )
            families.append({
                "prefix": pfx,
                "label": pfx.replace("_", " ").title(),
                "color_topk": c.get("topk", "#888"),
                "color_hybrid": c.get(
                    "hybrid", "#444"
                ),
                "marker": markers[mi],
                "top_k_sweep": tk_sweep,
            })
        return families

    # ── Save helpers ────────────────────────────────

    def _save_spec(self, methods, algo_names):
        self._save_json("spec.json", {
            "date": datetime.now().isoformat(),
            "algorithms": algo_names,
            "tasks": self.tasks,
            "head_mode": self.head_mode,
            "n_examples": self.n_examples,
            "n_queries": self.n_queries,
            "budgets": self.budgets,
            "seed": self.seed,
            "methods": [m.name for m in methods],
            "resolved_config": self.config,
        })

    def _save_json(self, filename, data):
        path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2,
                      default=str)

    def _save_csv(self, rows):
        if not rows:
            return
        path = self.out_dir / "results.csv"
        fields = [
            "task", "layer", "head", "example_id",
            "query_pos", "method", "method_kind",
            "budget", "actual_budget",
            "rel_l2_error", "seed",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def main():
    algo_choices = [
        k for k, v in METHOD_REGISTRY.items()
        if v.kind == "algorithm"
    ]

    parser = argparse.ArgumentParser(
        description="Run attention approximation "
        "experiments.",
    )
    parser.add_argument(
        "--algorithms", nargs="+", required=True,
        choices=algo_choices,
        help="Algorithms to evaluate.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to run on (default: all "
        "configured tasks).",
    )
    parser.add_argument(
        "--name", default=None,
        help="Experiment name (auto-generated "
        "if omitted).",
    )
    parser.add_argument(
        "--vectors-dir", default=None,
        help="Path to vectors/ directory.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to experiment_config.yaml.",
    )

    args = parser.parse_args()

    exp = Experiment(
        tasks=args.tasks,
        name=args.name,
        vectors_dir=args.vectors_dir,
        config_path=args.config,
    )
    exp.run(algo_names=args.algorithms)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
