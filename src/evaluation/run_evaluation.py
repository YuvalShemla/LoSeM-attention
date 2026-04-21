"""
Evaluation runner: load .pt data, evaluate methods, plot.

Handles the full lifecycle across multiple tasks.
Idealized methods are auto-included. Results organized
into per_task/ subfolders and overview/ summaries.

Usage:
  python -m src.evaluation.run_evaluation \\
    --algorithms multiq kmeans \\
    --tasks math_calc code_run \\
    --name grouping_comparison_v1

  # Regenerate plots only (needs prior run with spec.json):
  python -m src.evaluation.run_evaluation \\
    --plots-only --from-dir results/my_run_2024-01-01_12-00
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
    aggregate_group_cosines,
    aggregate_group_cosines_by_head_group,
    _algorithm_family,
    weighted_aggregate_heads,
)
from .plotting import (
    format_eval_config_caption,
    _group_l1_cgm_over_z_quantiles_line,
    plot_evaluation, plot_overview,
    plot_per_head_comparison,
    plot_group_cosine_distributions,
    plot_group_cosine_table,
    plot_group_cosine_cg_eg_scatter,
    plot_group_cosine_cg_eg_scatter_table,
    plot_group_token_probability_table,
    setup_style,
)

log = logging.getLogger("evaluation")

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_MAGENTA = "\033[35m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"


def _c(text: str, color: str, *, bold: bool = False) -> str:
    pfx = (ANSI_BOLD if bold else "") + color
    return f"{pfx}{text}{ANSI_RESET}"


def build_algorithm_plot_families(algorithms, config: dict):
    """
    Plot family specs (prefix, colors, top_k sweep) for algorithm
    instances. Used by Evaluation and by replay_plots.
    """
    import re
    color_map = config.get("plotting", {}).get(
        "algorithm_colors", {}
    )
    seen = {}
    families = []
    for m in algorithms:
        algo_name = None
        for aname, spec in METHOD_REGISTRY.items():
            if isinstance(m, spec.cls):
                algo_name = aname
                break
        pfx = re.sub(
            r"-(topk|hybrid)-k\d+$", "", m.name,
        )
        if pfx in seen:
            continue
        seen[pfx] = True
        c = color_map.get(algo_name, {})
        tk_sweep = config.get(
            "algorithm_configs", {}
        ).get(algo_name, {}).get(
            "top_k_sweep", [0, 1, 3, 5, 10]
        )
        families.append({
            "prefix": pfx,
            "label": pfx.replace("-", " "),
            "color_topk": c.get("topk", "#888"),
            "color_hybrid": c.get(
                "hybrid", "#444"
            ),
            "marker": c.get("marker", "o"),
            "top_k_sweep": tk_sweep,
        })
    return families


def replay_plots(results_dir: Path) -> None:
    """
    Regenerate PNGs from a previous run using spec.json and saved
    aggregated_stats (no vector data or method re-execution).
    """
    results_dir = Path(results_dir).resolve()
    spec_path = results_dir / "spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Missing {spec_path} — need a completed "
            f"evaluation directory."
        )
    with open(spec_path) as f:
        spec = json.load(f)
    config = spec.get("resolved_config")
    if not config:
        raise ValueError(
            "spec.json has no resolved_config; cannot "
            "replay plots."
        )
    algo_names = spec.get("algorithms", [])
    algo_cfgs = config.get("algorithm_configs", {})
    algorithms = _resolve_methods(algo_names, algo_cfgs)
    families = build_algorithm_plot_families(
        algorithms, config,
    )
    plotting = config.get("plotting", {})
    budgets = config["evaluation"]["budget_sweep"][
        "absolute"
    ]
    config_caption = format_eval_config_caption(config)
    tasks = spec.get("tasks", [])
    task_details = spec.get("task_details", {})

    per_task_agg = {}
    task_seq_info = {}

    for task in tasks:
        task_dir = results_dir / "per_task" / task
        agg_path = task_dir / "aggregated_stats.json"
        if not agg_path.is_file():
            log.warning(
                "Skip task %s: no %s",
                task, agg_path,
            )
            continue
        with open(agg_path) as f:
            agg = json.load(f)

        per_task_agg[task] = agg

        ds_path = task_dir / "data_statistics.json"
        n_queries = 0
        if ds_path.is_file():
            with open(ds_path) as f:
                ds = json.load(f)
                n_queries = int(ds.get("n_queries", 0))

        td = task_details.get(task, {})
        seq_lens = td.get("seq_lens", [])
        if len(seq_lens) == 1:
            seq_desc = f"{seq_lens[0]:,} tok"
        elif seq_lens:
            avg = int(np.mean(seq_lens))
            seq_desc = f"avg {avg:,} tok"
        else:
            seq_desc = ""
        if seq_lens:
            u = sorted(set(seq_lens))
            if len(u) == 1:
                task_seq_info[task] = f"{u[0]:,} tok"
            else:
                task_seq_info[task] = (
                    f"avg {int(np.mean(seq_lens)):,} tok"
                )

        plot_title = (
            f"{task} — {seq_desc}" if seq_desc else task
        )
        plot_evaluation(
            agg, task_dir, plotting, budgets,
            families,
            title=plot_title,
            n_queries=n_queries,
            config_caption=config_caption,
        )
        gcs_path = task_dir / "group_cosine_stats.json"
        if gcs_path.is_file():
            with open(gcs_path) as f:
                gcs = json.load(f)
            plot_group_cosine_distributions(
                gcs,
                task_dir,
                plotting,
                title=plot_title,
                config_caption=config_caption,
            )
            plot_group_cosine_cg_eg_scatter(
                gcs,
                task_dir,
                plotting,
                title=plot_title,
                config_caption=config_caption,
            )
        gcs_hg_path = task_dir / "group_cosine_by_head_group_stats.json"
        if gcs_hg_path.is_file():
            with open(gcs_hg_path) as f:
                gcs_hg = json.load(f)
            plot_group_cosine_table(
                gcs_hg,
                task_dir,
                plotting,
                title=plot_title,
                config_caption=config_caption,
            )
            plot_group_cosine_cg_eg_scatter_table(
                gcs_hg,
                task_dir,
                plotting,
                title=plot_title,
                config_caption=config_caption,
            )
            plot_group_token_probability_table(
                gcs_hg,
                task_dir,
                plotting,
                title=plot_title,
                config_caption=config_caption,
            )

        per_head_dir = task_dir / "per_head"
        if per_head_dir.is_dir():
            head_files = sorted(
                per_head_dir.glob("*.json"),
            )
            per_head_aggs = {}
            for idx, hp in enumerate(head_files):
                with open(hp) as f:
                    d = json.load(f)
                per_head_aggs[idx] = {
                    "agg": d["aggregated_stats"],
                    "layer": d["layer"],
                    "q_head": d["q_head"],
                    "kv_head": d["kv_head"],
                    "selection_label": d.get(
                        "selection_label", "",
                    ),
                    "effective_entropy": d.get(
                        "effective_entropy",
                    ),
                    "n_queries": d.get("n_queries", 0),
                }
            if len(per_head_aggs) > 1:
                plot_per_head_comparison(
                    per_head_aggs, task_dir,
                    plotting, budgets, families,
                    task_name=task,
                    seq_desc=seq_desc,
                    config_caption=config_caption,
                )

    if per_task_agg:
        ov_dir = results_dir / "overview"
        ov_dir.mkdir(parents=True, exist_ok=True)
        plot_overview(
            per_task_agg, ov_dir, plotting,
            budgets, families,
            task_seq_info=task_seq_info,
            config_caption=config_caption,
        )
        with open(ov_dir / "cross_task_stats.json", "w") as f:
            json.dump(per_task_agg, f, indent=2)

    log.info(
        "Replayed plots into %s",
        results_dir,
    )


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
    """Configure logging for evaluation output."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(name)s] %(message)s"
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)


class Evaluation:
    """Concrete evaluation class for .pt data."""

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
                / "evaluation_config.yaml"
            )
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        exp = self.config["evaluation"]
        data_cfg = self.config.get("data", {})

        if tasks:
            self.tasks = tasks
        elif "tasks" in self.config:
            self.tasks = self.config["tasks"]
        else:
            self.tasks = []

        vdir = vectors_dir or data_cfg.get(
            "vectors_dir", "data/vectors"
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
        self.n_sink = (
            1 if exp.get("exclude_sink_token", True)
            else 0
        )
        self.local_window = exp["local_window"]["size"]

        self.compute_statistics = exp.get(
            "compute_statistics", False
        )
        self.compute_group_cosine_distribution = exp.get(
            "compute_group_cosine_distribution", False
        )
        self.group_cosine_bins = int(
            exp.get("group_cosine_distribution_bins", 50)
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
        self.name = name or "eval"
        self.out_dir = (
            results_dir / f"{self.name}_{ts}"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, algo_names: List[str]):
        """Run evaluation across all tasks."""
        t0 = time.time()
        rng = np.random.default_rng(self.seed)

        idealized = []
        for spec in METHOD_REGISTRY.values():
            if spec.kind == "idealized":
                idealized.extend(
                    spec.cls.expand_from_config({})
                )
        algo_cfgs = self.config.get(
            "algorithm_configs", {}
        )
        algorithms = _resolve_methods(
            algo_names, algo_cfgs,
        )
        methods = idealized + algorithms

        # --- Log evaluation plan ---
        phase, _, _ = self._resolve_heads(
            self.tasks[0]
        )
        n_heads = self._count_heads()
        log.info(
            "Evaluation: %d tasks, %s mode, "
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

        all_rows = []
        per_task_agg = {}
        per_task_detail = {}
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
                task_rows, task_agg, task_detail = (
                    self._run_task(
                        task, methods, algorithms, rng,
                    )
                )
                all_rows.extend(task_rows)
                per_task_agg[task] = task_agg
                per_task_detail[task] = task_detail
                tasks_completed.append(task)
            except Exception as e:
                log.error("FAILED %s: %s", task, e)
                tasks_failed.append(task)
                raise

        self._save_spec(
            methods, algo_names, per_task_detail,
        )

        if per_task_agg:
            families = build_algorithm_plot_families(
                algorithms, self.config,
            )
            config_caption = format_eval_config_caption(
                self.config,
            )
            ov_dir = self.out_dir / "overview"
            ov_dir.mkdir(exist_ok=True)
            task_seq_info = {}
            for t, det in per_task_detail.items():
                slens = det.get("seq_lens", [])
                if len(slens) == 1:
                    task_seq_info[t] = (
                        f"{slens[0]:,} tok"
                    )
                elif slens:
                    avg = int(np.mean(slens))
                    task_seq_info[t] = (
                        f"avg {avg:,} tok"
                    )
            plot_overview(
                per_task_agg, ov_dir,
                self.config.get("plotting", {}),
                self.budgets, families,
                task_seq_info=task_seq_info,
                config_caption=config_caption,
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
        math_task_dir = self.out_dir / "per_task" / "math_calc"
        direct_plot_paths = [
            math_task_dir / "group_cosine_distribution_table.png",
            math_task_dir / "group_cosine_cg_eg_scatter_table.png",
            math_task_dir / "group_token_probability_table.png",
        ]
        def _trim_to_results(path: Path) -> str:
            parts = path.resolve().parts
            if "results" in parts:
                i = parts.index("results")
                return str(Path(*parts[i:]))
            return str(path.resolve())
        for p in direct_plot_paths:
            if p.is_file():
                print(f"  {_trim_to_results(p)}")
            else:
                print(f"  (missing) {_trim_to_results(p)}")

    # ── Validation ──────────────────────────────────

    def _validate_data(self):
        """
        Check that all tasks have enough data before
        starting. Fail fast with a clear message.
        """
        for task in self.tasks:
            phase, heads, _ = self._resolve_heads(task)
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

        phase, heads, head_meta = (
            self._resolve_heads(task)
        )
        all_results = []
        group_cosine_records = []
        per_head_results = {}
        rows = []
        example_ids = set()
        seq_lens = []
        # One id per evaluate_query call; used to dedupe hg table cells
        # when budget-sweep methods share the same (head, n_groups).
        eval_index = 0

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
                seq_lens.append(seq_len)
                qpos_list = _last_query_positions(
                    seq_len, self.n_queries,
                )

                example_ids.add(ex["example_id"])
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
                    this_eval = eval_index
                    eval_index += 1
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
                        compute_group_cosine_distribution=(
                            self.compute_group_cosine_distribution
                        ),
                    )
                    all_results.append(qr)
                    if self.compute_group_cosine_distribution:
                        gc_payload = qr.get(
                            "_group_cosines", {},
                        )
                        q_metrics = qr.get("_query_metrics", {})
                        head_tag = (
                            f"L{layer_idx}H{q_head}(kv={kv_head})"
                        )
                        for method_key, entry in gc_payload.items():
                            p75z = None
                            gl1 = None
                            gl1_znorm = None
                            sse_znorm = None
                            p75_gl1 = None
                            eg_l2 = None
                            max_exp_lg = None
                            v_rat_znorm = None
                            group_sizes = None
                            tok_prof_key = None
                            tok_prof_group = None
                            tok_prof_group_over_z = None
                            tok_prof_bounds = None
                            if isinstance(entry, dict):
                                cos = entry.get("cosines")
                                n_groups = int(entry.get("n_groups", 0))
                            else:
                                cos = entry
                                n_groups = 0
                            if cos is None or len(cos) == 0:
                                continue
                            sse = None
                            s_bar = None
                            v_rat = None
                            if isinstance(entry, dict):
                                sse = entry.get(
                                    "exp_residual_l1_over_z",
                                )
                                if sse is None:
                                    sse = entry.get(
                                        "exp_residual_sse_over_z2",
                                    )
                                if sse is None:
                                    sse = entry.get(
                                        "exp_residual_sse_sum",
                                    )
                                if sse is None:
                                    sse = entry.get(
                                        "logit_residual_sse_sum",
                                    )
                                if sse is None:
                                    sse = entry.get(
                                        "logit_within_group_var_sum",
                                    )
                                sse_znorm = entry.get(
                                    "exp_residual_l1_znorm",
                                )
                                s_bar = entry.get("sum_exp_bar_logits")
                                v_rat = entry.get(
                                    "value_softmax_mismatch_ratio",
                                )
                                if v_rat is None:
                                    v_rat = entry.get(
                                        "value_logit_group_l2_sq",
                                    )
                                v_rat_znorm = entry.get(
                                    "value_mismatch_ratio_znorm",
                                )
                                p75z = entry.get(
                                    "max_group_exp_logits_m_over_z",
                                )
                                if p75z is None:
                                    p75z = entry.get(
                                        "p75_group_z_norm_residuals",
                                    )
                                if p75z is None:
                                    p75z = entry.get(
                                        "median_group_z_norm_residuals",
                                    )
                                gl1 = entry.get("group_l1_cg_m_over_z")
                                gl1_znorm = entry.get("group_l1_cg_m_znorm")
                                group_sizes = entry.get("group_sizes")
                                p75_gl1 = entry.get(
                                    "p75_group_l1_cg_m_over_z",
                                )
                                if p75_gl1 is None:
                                    p75_gl1 = entry.get(
                                        "median_group_l1_cg_m_over_z",
                                    )
                                eg_l2 = entry.get(
                                    "group_out_l2_err_sq_m_over_z",
                                )
                                exp_lg = entry.get(
                                    "group_exp_lg_m_over_z",
                                )
                                max_exp_lg = entry.get(
                                    "max_group_exp_lg_m_over_z",
                                )
                                tok_prof_key = entry.get(
                                    "token_profile_key_probs",
                                )
                                tok_prof_group = entry.get(
                                    "token_profile_group_probs",
                                )
                                tok_prof_group_over_z = entry.get(
                                    "token_profile_group_probs_over_z",
                                )
                                tok_prof_bounds = entry.get(
                                    "token_profile_group_boundaries",
                                )
                                top_logits = entry.get(
                                    "max_group_top_logits",
                                )
                                if top_logits is None:
                                    top_logits = []
                            else:
                                p75z = None
                                gl1 = None
                                gl1_znorm = None
                                sse_znorm = None
                                p75_gl1 = None
                                eg_l2 = None
                                exp_lg = None
                                max_exp_lg = None
                                v_rat_znorm = None
                                group_sizes = None
                                top_logits = []
                                tok_prof_key = None
                                tok_prof_group = None
                                tok_prof_group_over_z = None
                                tok_prof_bounds = None
                            print(
                                _c("[max_e_lg_top100_logits]", ANSI_MAGENTA, bold=True),
                                _c(f"task={task}", ANSI_CYAN),
                                _c(f"eval_index={this_eval}", ANSI_YELLOW),
                                _c(f"head={head_tag}", ANSI_GREEN),
                                _c(f"method={method_key}", ANSI_CYAN, bold=True),
                                top_logits,
                            )
                            head_ent = None
                            if head_meta and (hi - 1) < len(
                                head_meta,
                            ):
                                head_ent = head_meta[hi - 1].get(
                                    "effective_entropy",
                                )
                            group_cosine_records.append({
                                "algorithm": _algorithm_family(method_key),
                                "method": method_key,
                                "eval_index": this_eval,
                                "head": head_tag,
                                "effective_entropy": head_ent,
                                "n_groups": n_groups,
                                "cosines": cos,
                                "exp_residual_l1_over_z": sse,
                                "exp_residual_sse_over_z2": sse,
                                "exp_residual_sse_sum": sse,
                                "logit_residual_sse_sum": sse,
                                "sum_exp_logits": q_metrics.get(
                                    "sum_exp_logits",
                                ),
                                "sum_exp_bar_logits": s_bar,
                                "exp_residual_l1_znorm": sse_znorm,
                                "value_softmax_mismatch_ratio": v_rat,
                                "value_logit_group_l2_sq": v_rat,
                                "value_mismatch_ratio_znorm": v_rat_znorm,
                                "p75_group_z_norm_residuals": p75z,
                                "max_group_exp_lg_m_over_z": max_exp_lg,
                                "group_l1_cg_m_over_z": gl1,
                                "group_l1_cg_m_znorm": gl1_znorm,
                                "group_sizes": group_sizes,
                                "p75_group_l1_cg_m_over_z": p75_gl1,
                                "group_out_l2_err_sq_m_over_z": eg_l2,
                                "group_exp_lg_m_over_z": exp_lg,
                                "token_profile_key_probs": tok_prof_key,
                                "token_profile_group_probs": tok_prof_group,
                                "token_profile_group_probs_over_z": tok_prof_group_over_z,
                                "token_profile_group_boundaries": tok_prof_bounds,
                            })
                    per_head_results.setdefault(
                        hi - 1, []
                    ).append(qr)
                    for key, val in qr.items():
                        if key.startswith("_"):
                            continue
                        mname = key.rsplit("-", 1)[0]
                        mk = "idealized"
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

        n_total = len(all_results)
        task_elapsed = time.time() - task_t0

        families = build_algorithm_plot_families(
            algorithms, self.config,
        )
        config_caption = format_eval_config_caption(
            self.config,
        )

        # Per-head aggregation and plots
        per_head_dir = task_dir / "per_head"
        per_head_dir.mkdir(exist_ok=True)
        per_head_aggs = {}
        for idx, results in per_head_results.items():
            l, h, k = heads[idx]
            hm = head_meta[idx] if head_meta else {}
            label = hm.get("selection_label", "")
            ent = hm.get("effective_entropy")
            tag = f"L{l}H{h}"
            if label:
                tag += f"_{label}"
            head_agg = aggregate_results(results)
            per_head_aggs[idx] = {
                "agg": head_agg,
                "layer": l, "q_head": h,
                "kv_head": k,
                "n_queries": len(results),
                "selection_label": label,
                "effective_entropy": ent,
            }
            self._save_json(
                f"per_task/{task}/per_head/"
                f"{tag}.json",
                {
                    "layer": l, "q_head": h,
                    "kv_head": k,
                    "selection_label": label,
                    "effective_entropy": ent,
                    "n_queries": len(results),
                    "aggregated_stats": head_agg,
                },
            )

        if len(per_head_aggs) > 1:
            unique_lens = sorted(set(seq_lens))
            if len(unique_lens) == 1:
                seq_desc = f"{unique_lens[0]:,} tok"
            else:
                avg = int(np.mean(seq_lens))
                seq_desc = f"avg {avg:,} tok"
            plot_per_head_comparison(
                per_head_aggs, task_dir,
                self.config.get("plotting", {}),
                self.budgets, families,
                task_name=task,
                seq_desc=seq_desc,
                config_caption=config_caption,
            )

        # Weighted aggregate across heads
        if per_head_aggs and head_meta:
            agg = weighted_aggregate_heads(
                per_head_aggs, head_meta,
            )
        else:
            agg = aggregate_results(all_results)

        unique_lens = sorted(set(seq_lens))
        if len(unique_lens) == 1:
            seq_desc = f"{unique_lens[0]:,} tok"
        else:
            avg = int(np.mean(seq_lens))
            seq_desc = f"avg {avg:,} tok"

        plot_evaluation(
            agg, task_dir,
            self.config.get("plotting", {}),
            self.budgets, families,
            title=f"{task} — {seq_desc}",
            n_queries=n_total,
            config_caption=config_caption,
        )
        if self.compute_group_cosine_distribution:
            group_cosine_stats = aggregate_group_cosines(
                all_results,
                n_bins=self.group_cosine_bins,
            )
            if group_cosine_stats:
                self._save_json(
                    f"per_task/{task}/group_cosine_stats.json",
                    group_cosine_stats,
                )
                for _mk, _row in sorted(
                    group_cosine_stats.items(),
                ):
                    _qline = _group_l1_cgm_over_z_quantiles_line(
                        (_row.get("group_l1_hist_meta") or {}),
                    )
                    if _qline:
                        log.info(
                            "  [%s] %s",
                            _mk,
                            _qline,
                        )
                plot_group_cosine_distributions(
                    group_cosine_stats,
                    task_dir,
                    self.config.get("plotting", {}),
                    title=f"{task} — {seq_desc}",
                    config_caption=config_caption,
                )
                plot_group_cosine_cg_eg_scatter(
                    group_cosine_stats,
                    task_dir,
                    self.config.get("plotting", {}),
                    title=f"{task} — {seq_desc}",
                    config_caption=config_caption,
                )
            hg_stats = aggregate_group_cosines_by_head_group(
                group_cosine_records,
                n_bins=self.group_cosine_bins,
            )
            if hg_stats.get("columns") and hg_stats.get("row_algorithms"):
                self._save_json(
                    f"per_task/{task}/group_cosine_by_head_group_stats.json",
                    hg_stats,
                )
                plot_group_cosine_table(
                    hg_stats,
                    task_dir,
                    self.config.get("plotting", {}),
                    title=f"{task} — {seq_desc}",
                    config_caption=config_caption,
                )
                plot_group_cosine_cg_eg_scatter_table(
                    hg_stats,
                    task_dir,
                    self.config.get("plotting", {}),
                    title=f"{task} — {seq_desc}",
                    config_caption=config_caption,
                )
                plot_group_token_probability_table(
                    hg_stats,
                    task_dir,
                    self.config.get("plotting", {}),
                    title=f"{task} — {seq_desc}",
                    config_caption=config_caption,
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
                    "effective_entropy_mean", 0
                ),
                qstats.get(
                    "effective_entropy_std", 0
                ),
                qstats.get(
                    "effective_top1pct_mass_mean",
                    0,
                ),
                qstats.get(
                    "effective_top1pct_mass_std",
                    0,
                ),
            )
        if self.compute_group_cosine_distribution:
            data_stats["group_cosine_distribution"] = {
                "enabled": True,
                "bins": self.group_cosine_bins,
            }
        self._save_json(
            f"per_task/{task}/data_statistics.json",
            data_stats,
        )

        log.info(
            "  Task complete: %d queries, %.1fs",
            n_total, task_elapsed,
        )
        task_detail = {
            "heads": [
                {"layer": l, "q_head": h,
                 "kv_head": k}
                for l, h, k in heads
            ],
            "examples": sorted(example_ids),
            "n_queries_per_example": self.n_queries,
            "total_queries": n_total,
            "seq_lens": sorted(set(seq_lens)),
        }
        return rows, agg, task_detail

    # ── Head resolution ─────────────────────────────

    def _resolve_heads(self, task):
        """
        Determine (phase, heads, head_meta) from config.

        Returns (phase_str,
                 list of (layer, q_head, kv_head),
                 list of meta dicts or None).
        phase is None for flat layout, or a string
        for legacy phase-based layout.
        head_meta entries have percentile,
        effective_entropy, selection_label when
        available.
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
            return phase, triples, None

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
            triples = [
                (s["layer"], s["q_head"],
                 s["kv_head"])
                for s in sel
            ]
            head_meta = [
                {
                    "percentile": s.get("percentile"),
                    "effective_entropy": s.get(
                        "effective_entropy"
                    ),
                    "selection_label": s.get(
                        "selection_label"
                    ),
                }
                for s in sel
            ]
            return phase, triples, head_meta

        if mode == "all_heads":
            triples = []
            for layer in self.layers:
                for h in range(self.n_q_heads):
                    triples.append(
                        (layer, h,
                         h // self.gqa_group)
                    )
            return "all_heads", triples, None

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
        return 5  # typical selected_heads count

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

    # ── Save helpers ────────────────────────────────

    def _save_spec(self, methods, algo_names,
                   per_task_detail=None):
        self._save_json("spec.json", {
            "date": datetime.now().isoformat(),
            "algorithms": algo_names,
            "tasks": self.tasks,
            "head_mode": self.head_mode,
            "n_examples_per_task": self.n_examples,
            "n_queries_per_example": self.n_queries,
            "budgets": self.budgets,
            "seed": self.seed,
            "methods": [m.name for m in methods],
            "task_details": per_task_detail or {},
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
        "evaluations.",
    )
    parser.add_argument(
        "--algorithms", nargs="*", default=[],
        choices=algo_choices,
        help="Algorithms to evaluate (omit for "
        "idealized methods only).",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to run on (default: all "
        "configured tasks).",
    )
    parser.add_argument(
        "--name", default=None,
        help="Evaluation name (auto-generated "
        "if omitted).",
    )
    parser.add_argument(
        "--vectors-dir", default=None,
        help="Path to vectors/ directory.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to evaluation_config.yaml.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate PNGs from an existing run "
        "(requires --from-dir). No experiments.",
    )
    parser.add_argument(
        "--from-dir",
        default=None,
        help="Results directory containing spec.json "
        "(used with --plots-only).",
    )

    args = parser.parse_args()

    if args.plots_only:
        if not args.from_dir:
            parser.error(
                "--plots-only requires --from-dir "
                "POINTING_TO_results/run_folder",
            )
        _setup_logging()
        replay_plots(Path(args.from_dir))
        return

    exp = Evaluation(
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
