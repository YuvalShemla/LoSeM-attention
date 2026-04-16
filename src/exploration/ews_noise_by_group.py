"""
EWS noise-by-group-rank analysis — 3 noise types.

Fix budget=512. For each group rank, inject 1 random
candidate with three modes:
  - NoiseK:  corrupt key mean only (affects routing score)
  - NoiseV:  corrupt value mean only (affects output)
  - NoiseKV: corrupt both

Plot: error (y) vs group rank (x) with three curves,
plus horizontal baselines. Both log and linear scale.
"""

import numpy as np
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.data_loader import (
    load_examples, load_task_metadata,
)
from src.core import (
    softmax, full_attention, compute_special_indices,
    relative_l2_error,
)
from src.algorithms.idealized_methods import (
    IdealEqualWeightSplits, IdealTopK, IdealSampling,
    IdealEqualSplits,
)
from src.algorithms.base import AttentionInput

BUDGET = 512
N_SEEDS = 10
TASKS = [
    "math_calc", "code_run", "longbook_sum_eng",
    "kv_retrieval", "multi_doc_qa", "single_doc_qa",
]

GROUP_POSITIONS = (
    list(range(10))
    + [15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
       150, 200, 300, 400, 511]
)

NOISE_MODES = ["NoiseK", "NoiseV", "NoiseKV"]


class PrecomputedEWS:
    """Precompute clean EWS state, then patch one group."""

    def __init__(
        self, query, keys, values, logits, head_dim,
        special_idx, candidate_idx, groups,
    ):
        self.query = query
        self.keys = keys
        self.values = values
        sqrt_d = np.sqrt(head_dim)
        d = keys.shape[1]
        n_groups = len(groups)
        n_sp = len(special_idx)

        sizes = np.array(
            [len(g) for g in groups], dtype=np.int64,
        )
        self.sizes_f = sizes.astype(np.float64)
        self.n_groups = n_groups
        self.n_sp = n_sp
        self.candidate_idx = candidate_idx

        flat_idx = np.concatenate(groups)
        labels = np.repeat(np.arange(n_groups), sizes)
        k_flat = keys[flat_idx].astype(np.float64)
        v_flat = values[flat_idx].astype(np.float64)

        sum_k = np.empty((n_groups, d), dtype=np.float64)
        sum_v = np.empty((n_groups, d), dtype=np.float64)
        for j in range(d):
            sum_k[:, j] = np.bincount(
                labels, weights=k_flat[:, j],
                minlength=n_groups,
            )
            sum_v[:, j] = np.bincount(
                labels, weights=v_flat[:, j],
                minlength=n_groups,
            )
        self.sum_k = sum_k
        self.sum_v = sum_v

        sf = self.sizes_f[:, None]
        self.avg_k = (sum_k / sf).astype(np.float32)
        self.avg_v = sum_v / sf

        n_total = n_sp + n_groups
        self.scores = np.empty(n_total, dtype=np.float64)
        self.out_vals = np.empty(
            (n_total, d), dtype=np.float64,
        )
        self.scores[:n_sp] = logits[special_idx]
        self.out_vals[:n_sp] = values[special_idx]
        self.scores[n_sp:] = (
            self.avg_k @ query / sqrt_d
            + np.log(self.sizes_f)
        )
        self.out_vals[n_sp:] = self.avg_v
        self.sqrt_d = sqrt_d

    def inject_noise(self, target_group, rng, mode):
        """
        mode: "NoiseK", "NoiseV", or "NoiseKV"
        """
        ni = rng.choice(self.candidate_idx)
        nk = self.keys[ni].astype(np.float64)
        nv = self.values[ni].astype(np.float64)
        tg = target_group
        sf = self.sizes_f[tg]
        idx = self.n_sp + tg

        old_score = self.scores[idx]
        old_val = self.out_vals[idx].copy()

        if mode in ("NoiseK", "NoiseKV"):
            noisy_mk = (
                (self.sum_k[tg] + nk) / (sf + 1)
            ).astype(np.float32)
            self.scores[idx] = (
                float(noisy_mk @ self.query / self.sqrt_d)
                + np.log(sf)
            )

        if mode in ("NoiseV", "NoiseKV"):
            self.out_vals[idx] = (
                (self.sum_v[tg] + nv) / (sf + 1)
            )

        w = softmax(self.scores)
        out = (w @ self.out_vals).astype(np.float32)

        self.scores[idx] = old_score
        self.out_vals[idx] = old_val
        return out


def get_baseline_errors(problem, gt_out):
    baselines = {}
    for cls, name in [
        (IdealTopK, "IdealTopK"),
        (IdealSampling, "IdealSampling"),
        (IdealEqualSplits, "IdealEqualSplits"),
        (IdealEqualWeightSplits, "EWS (clean)"),
    ]:
        m = cls()
        if name == "IdealSampling":
            errs = []
            for s in range(N_SEEDS):
                r = np.random.default_rng(42 + s)
                out = m.run(problem, BUDGET, r)
                errs.append(
                    relative_l2_error(out.output, gt_out)
                )
            baselines[name] = float(np.mean(errs))
        else:
            rng = np.random.default_rng(42)
            out = m.run(problem, BUDGET, rng)
            baselines[name] = relative_l2_error(
                out.output, gt_out,
            )
    return baselines


def run_task(task, vectors_dir, return_per_head=False):
    """Run noise analysis for one task, return results."""
    meta = load_task_metadata(vectors_dir, task)
    heads = meta.get("selected_heads", [])
    if not heads:
        print(f"  No heads for {task}, skipping")
        return None

    head_errors = {m: [] for m in NOISE_MODES}
    head_sizes = []
    head_baselines = []
    head_labels = []
    per_head_results = []

    for h in heads:
        examples = list(load_examples(
            vectors_dir, task,
            layer=h["layer"], head=h["q_head"],
            kv_head=h["kv_head"], max_examples=1,
        ))
        if not examples:
            continue

        ex = examples[0]
        Q, K, V = ex["Q"], ex["K"], ex["V"]
        seq_len = Q.shape[0]
        head_dim = Q.shape[1]
        qpos = seq_len - 1
        query = Q[qpos]
        keys = K[:qpos + 1]
        values = V[:qpos + 1]

        special_idx, candidate_idx = (
            compute_special_indices(
                n_causal=len(keys),
                n_sink=1, local_window=0,
            )
        )
        logits = (query @ keys.T) / np.sqrt(head_dim)
        gt_out, _, _ = full_attention(
            query, keys, values, head_dim,
        )

        cand_logits = logits[candidate_idx]
        cand_weights = softmax(cand_logits)
        sort_order = np.argsort(cand_weights)[::-1]
        sorted_idx = candidate_idx[sort_order]
        sorted_weights = cand_weights[sort_order]
        groups = (
            IdealEqualWeightSplits._equal_weight_groups(
                sorted_idx, sorted_weights, BUDGET,
            )
        )
        n_groups = len(groups)
        group_sizes = [len(g) for g in groups]

        print(f"    L{h['layer']}H{h['q_head']} "
              f"({len(keys)} tok, {n_groups} grp)",
              flush=True)

        problem = AttentionInput(
            query=query, keys=keys, values=values,
            head_dim=head_dim, logits=logits,
            special_idx=special_idx,
            candidate_idx=candidate_idx,
        )
        baselines = get_baseline_errors(problem, gt_out)
        head_baselines.append(baselines)

        ews = PrecomputedEWS(
            query, keys, values, logits, head_dim,
            special_idx, candidate_idx, groups,
        )

        positions = [
            p for p in GROUP_POSITIONS if p < n_groups
        ]

        for mode in NOISE_MODES:
            errors = {}
            for gpos in positions:
                errs = []
                for seed in range(N_SEEDS):
                    rng = np.random.default_rng(
                        42 + seed,
                    )
                    out = ews.inject_noise(
                        gpos, rng, mode,
                    )
                    errs.append(
                        relative_l2_error(out, gt_out)
                    )
                errors[gpos] = np.mean(errs)
            head_errors[mode].append(errors)

        head_sizes.append(
            {p: group_sizes[p] for p in positions}
        )
        ent = h.get("effective_entropy")
        tag = f"L{h['layer']}H{h['q_head']}"
        if ent is not None:
            tag += f" (ent={ent:.2f})"
        head_labels.append(tag)

        if return_per_head:
            ph = {"positions": positions}
            ph["avg"] = {}
            for mode in NOISE_MODES:
                ph["avg"][mode] = head_errors[mode][-1]
            ph["avg_sizes"] = head_sizes[-1]
            ph["avg_baselines"] = head_baselines[-1]
            ph["label"] = head_labels[-1]
            per_head_results.append(ph)

    if not head_sizes:
        return None

    # Average across heads for this task
    positions = sorted(set(
        p for s in head_sizes for p in s
    ))
    avg = {}
    for mode in NOISE_MODES:
        avg[mode] = {}
        for p in positions:
            vals = [
                e[p] for e in head_errors[mode]
                if p in e
            ]
            avg[mode][p] = np.mean(vals)

    avg_sizes = {}
    for p in positions:
        sz = [s[p] for s in head_sizes if p in s]
        avg_sizes[p] = np.mean(sz)

    avg_baselines = {}
    for key in head_baselines[0]:
        avg_baselines[key] = np.mean(
            [b[key] for b in head_baselines]
        )

    result = {
        "positions": positions,
        "avg": avg,
        "avg_sizes": avg_sizes,
        "avg_baselines": avg_baselines,
    }
    if return_per_head:
        result["per_head"] = per_head_results
    return result


MODE_STYLES = {
    "NoiseK": (
        "#00bfff", "o", "-", 0.9, 2.0,
        "EWS+NoiseK (key only)",
    ),
    "NoiseV": (
        "#ff8c00", "s", "-", 0.9, 2.0,
        "EWS+NoiseV (value only)",
    ),
    "NoiseKV": (
        "#e377c2", "D", "--", 0.7, 2.5,
        "EWS+NoiseKV (both)",
    ),
}
BASELINE_STYLES = {
    "IdealTopK": ("#d62728", "--"),
    "IdealSampling": ("#2ca02c", "--"),
    "IdealEqualSplits": ("#1f77b4", "--"),
    "EWS (clean)": ("#9467bd", "-"),
}


def make_plot(
    positions, avg, avg_baselines, title, out_path,
    scale="log",
):
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.array(positions)

    for mode in NOISE_MODES:
        y = np.array([avg[mode][p] for p in positions])
        color, marker, ls, alpha, lw, label = (
            MODE_STYLES[mode]
        )
        ax.plot(
            x, y, marker=marker, markersize=5,
            linewidth=lw, color=color, linestyle=ls,
            alpha=alpha, label=label,
        )

    for name, err in avg_baselines.items():
        color, ls = BASELINE_STYLES[name]
        ax.axhline(
            y=err, color=color, linestyle=ls,
            linewidth=1.5, alpha=0.8,
            label=f"{name} ({err:.4f})",
        )

    ax.set_xlabel("Group Rank (0 = highest attention)")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    if scale == "log":
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close()


def run_per_head(vectors_dir, out_dir):
    """Per-head plots + CSV for code_run."""
    task = "code_run"
    print(f"Per-head analysis: {task}", flush=True)
    res = run_task(task, vectors_dir, return_per_head=True)
    if not res or "per_head" not in res:
        print("No data")
        return

    ph_dir = out_dir / "per_head"
    ph_dir.mkdir(parents=True, exist_ok=True)

    all_csv_rows = []

    for ph in res["per_head"]:
        label = ph["label"]
        print(f"  Plotting {label}", flush=True)

        for scale in ("log", "linear"):
            make_plot(
                ph["positions"], ph["avg"],
                ph["avg_baselines"],
                f"EWS Noise by Group — {task} {label} "
                f"(budget={BUDGET})",
                ph_dir / f"{label}_{scale}.png",
                scale=scale,
            )

        clean_err = ph["avg_baselines"]["EWS (clean)"]
        for p in ph["positions"]:
            all_csv_rows.append({
                "head": label,
                "group_rank": p,
                "group_size": ph["avg_sizes"].get(p, 0),
                "NoiseK": ph["avg"]["NoiseK"].get(p, 0),
                "NoiseV": ph["avg"]["NoiseV"].get(p, 0),
                "NoiseKV": ph["avg"]["NoiseKV"].get(p, 0),
                "clean_ews": clean_err,
            })

    # CSV
    csv_path = ph_dir / "per_head.csv"
    if all_csv_rows:
        fields = list(all_csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_csv_rows)
        print(f"\nCSV: {csv_path}")

    # Combined subplot figure — all heads
    n_heads = len(res["per_head"])
    for scale in ("log", "linear"):
        fig, axes = plt.subplots(
            1, n_heads, figsize=(5 * n_heads, 5),
            sharey=True,
        )
        if n_heads == 1:
            axes = [axes]

        for i, (ax, ph) in enumerate(
            zip(axes, res["per_head"])
        ):
            x = np.array(ph["positions"])
            for mode in NOISE_MODES:
                y = np.array([
                    ph["avg"][mode].get(p, np.nan)
                    for p in ph["positions"]
                ])
                color, marker, ls, alpha, lw, label = (
                    MODE_STYLES[mode]
                )
                ax.plot(
                    x, y, marker=marker, markersize=3,
                    linewidth=lw, color=color,
                    linestyle=ls, alpha=alpha,
                    label=label if i == 0 else None,
                )

            for name, err in ph["avg_baselines"].items():
                color, ls = BASELINE_STYLES[name]
                ax.axhline(
                    y=err, color=color, linestyle=ls,
                    linewidth=1, alpha=0.6,
                    label=(
                        f"{name}" if i == 0 else None
                    ),
                )

            ax.set_title(ph["label"], fontsize=11)
            ax.set_xlabel("Group Rank")
            if scale == "log":
                ax.set_yscale("log")
            if i == 0:
                ax.set_ylabel("Relative L2 Error")

        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper center",
            ncol=4, fontsize=8,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(
            f"EWS Noise by Group Rank — {task} "
            f"per head (budget={BUDGET})",
            y=1.06, fontsize=13,
        )
        fig.tight_layout()
        fig_path = ph_dir / f"all_heads_{scale}.png"
        fig.savefig(
            fig_path, dpi=200, bbox_inches="tight",
        )
        plt.close()
        print(f"Combined plot: {fig_path}")

    # Print per-head baselines
    print(f"\nPer-head baselines (budget={BUDGET}):")
    print(f"  {'Head':>8} {'EWS clean':>10} "
          f"{'TopK':>10} {'Sampling':>10}")
    print(f"  {'-'*42}")
    for ph in res["per_head"]:
        b = ph["avg_baselines"]
        print(
            f"  {ph['label']:>8} "
            f"{b['EWS (clean)']:>10.6f} "
            f"{b['IdealTopK']:>10.6f} "
            f"{b['IdealSampling']:>10.6f}"
        )

    # Print per-head noise at key positions
    for ph in res["per_head"]:
        label = ph["label"]
        clean = ph["avg_baselines"]["EWS (clean)"]
        print(f"\n  {label}:")
        print(f"  {'Rank':>5} {'Size':>6} "
              f"{'NoiseK':>10} {'NoiseV':>10} "
              f"{'NoiseKV':>10}")
        print(f"  {'-'*42}")
        for p in ph["positions"]:
            print(
                f"  {p:>3} "
                f"{ph['avg_sizes'].get(p, 0):>6.0f} "
                f"{ph['avg']['NoiseK'].get(p, 0):>10.6f} "
                f"{ph['avg']['NoiseV'].get(p, 0):>10.6f} "
                f"{ph['avg']['NoiseKV'].get(p, 0):>10.6f}"
            )


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-head", action="store_true",
        help="Per-head plots for code_run",
    )
    args = parser.parse_args()

    config_path = Path("src/evaluation/evaluation_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vectors_dir = Path(config["data"]["vectors_dir"])
    out_dir = Path("results/ews_noise_by_group")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_head:
        run_per_head(vectors_dir, out_dir)
        return

    task_results = {}

    for task in TASKS:
        print(f"\n{'='*50}")
        print(f"  {task}")
        print(f"{'='*50}", flush=True)
        res = run_task(task, vectors_dir)
        if res:
            task_results[task] = res

            # Per-task plots
            for scale in ("log", "linear"):
                make_plot(
                    res["positions"], res["avg"],
                    res["avg_baselines"],
                    f"EWS Noise by Group Rank "
                    f"(budget={BUDGET}, {task})",
                    out_dir / f"{task}_{scale}.png",
                    scale=scale,
                )
            print(f"  Plots saved for {task}")

    if not task_results:
        print("No data")
        return

    # Cross-task average
    all_positions = sorted(set(
        p for r in task_results.values()
        for p in r["positions"]
    ))

    cross_avg = {}
    for mode in NOISE_MODES:
        cross_avg[mode] = {}
        for p in all_positions:
            vals = [
                r["avg"][mode][p]
                for r in task_results.values()
                if p in r["avg"][mode]
            ]
            if vals:
                cross_avg[mode][p] = np.mean(vals)

    cross_sizes = {}
    for p in all_positions:
        sz = [
            r["avg_sizes"][p]
            for r in task_results.values()
            if p in r["avg_sizes"]
        ]
        if sz:
            cross_sizes[p] = np.mean(sz)

    cross_baselines = {}
    for key in list(task_results.values())[0]["avg_baselines"]:
        cross_baselines[key] = np.mean([
            r["avg_baselines"][key]
            for r in task_results.values()
        ])

    # Filter positions that have data
    valid_positions = [
        p for p in all_positions
        if p in cross_avg["NoiseK"]
    ]

    for scale in ("log", "linear"):
        make_plot(
            valid_positions, cross_avg,
            cross_baselines,
            f"EWS Noise by Group Rank — Cross-Task Average "
            f"(budget={BUDGET})",
            out_dir / f"cross_task_{scale}.png",
            scale=scale,
        )

    # Save CSV
    csv_path = out_dir / "noise_by_group.csv"
    clean_err = cross_baselines["EWS (clean)"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "group_rank", "avg_group_size",
            "NoiseK_error", "NoiseV_error",
            "NoiseKV_error", "clean_ews_error",
        ])
        for p in valid_positions:
            w.writerow([
                p,
                f"{cross_sizes[p]:.1f}",
                f"{cross_avg['NoiseK'][p]:.8f}",
                f"{cross_avg['NoiseV'][p]:.8f}",
                f"{cross_avg['NoiseKV'][p]:.8f}",
                f"{clean_err:.8f}",
            ])
    print(f"\nCSV: {csv_path}")

    # Print table
    print(f"\nCross-task baselines at budget={BUDGET}:")
    for name, err in sorted(
        cross_baselines.items(), key=lambda x: x[1],
    ):
        print(f"  {name:30s}: {err:.6f}")

    print(f"\n{'Rank':>5} {'Size':>6} "
          f"{'NoiseK':>10} {'NoiseV':>10} "
          f"{'NoiseKV':>10}")
    print(f"  {'-'*46}")
    for p in valid_positions:
        print(
            f"  {p:>3} {cross_sizes[p]:>6.0f} "
            f"{cross_avg['NoiseK'][p]:>10.6f} "
            f"{cross_avg['NoiseV'][p]:>10.6f} "
            f"{cross_avg['NoiseKV'][p]:>10.6f}"
        )


if __name__ == "__main__":
    main()
