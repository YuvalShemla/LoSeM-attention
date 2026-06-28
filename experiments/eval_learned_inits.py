#!/usr/bin/env python3
"""
Learned coreset initialization comparison.

Compares 5 initializations (mqbeta, kmeans, random-keys, first-tokens,
random-gauss) with 3000 training steps, plotting training curves and
final error vs baselines (IdealTopK, vAttention) per budget per head.

Usage:
  python experiments/eval_learned_inits.py --dataset us_presidential_debates
"""

import argparse
import inspect
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.algorithms.learned.algorithm import LearnedCoreset
from src.algorithms.idealized_methods import IdealTopK, VAttentionOracle
from src.evaluation.evaluator import aggregate_results, weighted_aggregate_heads
from src.core import full_attention, relative_l2_error
from src.algorithms.base import AttentionInput

HEADS = [
    {"layer": 31, "q_head": 14, "kv_head": 3, "label": "p0",
     "selection_label": "p0_lowest", "percentile": 0, "entropy": 0.14},
    {"layer": 12, "q_head": 25, "kv_head": 6, "label": "p25",
     "selection_label": "p25", "percentile": 25, "entropy": 3.98},
    {"layer": 15, "q_head": 22, "kv_head": 5, "label": "p50",
     "selection_label": "p50_median", "percentile": 50, "entropy": 4.71},
    {"layer": 27, "q_head":  7, "kv_head": 1, "label": "p75",
     "selection_label": "p75", "percentile": 75, "entropy": 5.64},
    {"layer":  0, "q_head": 22, "kv_head": 5, "label": "p100",
     "selection_label": "p100_highest", "percentile": 100, "entropy": 11.03},
]

HEAD_DIM = 128
N_SINK = 1
LOCAL_WINDOW = 128
SEED = 42
N_TRAIN_QUERIES = 10000
N_STEPS = 3000

DATASET_BUDGETS = {
    "multi_doc_qa_sanofi":      [64, 256, 1024, 2048],
    "us_presidential_debates":  [128, 512, 2048, 4096],
    "apple_samsung_financials": [512, 2048, 4096, 8192],
}
DEFAULT_BUDGETS = [128, 512, 2048, 4096]

INIT_METHODS = ["mqbeta", "kmeans", "random", "first", "random_gauss"]
INIT_COLORS = {
    "mqbeta": "#1f77b4",
    "kmeans": "#2ca02c",
    "random": "#ff7f0e",
    "first": "#9467bd",
    "random_gauss": "#d62728",
    "TFCFW-lq": "#8c564b",
}
INIT_LABELS = {
    "mqbeta": "MQBeta init",
    "kmeans": "KMeans init",
    "random": "Random keys",
    "first": "First tokens",
    "random_gauss": "Random gauss",
    "TFCFW-lq": "TFCFW-lq",
}


def load_context_kv(cartridge_dir, layer, q_head, kv_head):
    pt_path = cartridge_dir / "context" / f"layer_{layer:02d}.pt"
    tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    return (tensors[f"K_rope_kvhead{kv_head}"].float().numpy(),
            tensors[f"V_kvhead{kv_head}"].float().numpy(),
            tensors[f"Q_rope_head{q_head}"].float().numpy())


def load_all_conv_queries(cartridge_dir, layer, q_head):
    conv_parent = cartridge_dir / "conversations"
    q_key = f"Q_rope_head{q_head}"
    pt_name = f"layer_{layer:02d}.pt"
    all_q = []
    for cd in sorted(d for d in conv_parent.iterdir()
                     if d.is_dir() and d.name.startswith("conv_")):
        pt_path = cd / pt_name
        if pt_path.exists():
            t = torch.load(pt_path, map_location="cpu", weights_only=True)
            all_q.append(t[q_key].float().numpy() if q_key in t
                         else np.zeros((0, HEAD_DIM), np.float32))
        else:
            all_q.append(np.zeros((0, HEAD_DIM), np.float32))
    return all_q


def load_conv_kv(cartridge_dir, conv_idx, layer, kv_head):
    pt_path = (cartridge_dir / "conversations" / f"conv_{conv_idx:04d}"
               / f"layer_{layer:02d}.pt")
    tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    return (tensors[f"K_rope_kvhead{kv_head}"].float().numpy(),
            tensors[f"V_kvhead{kv_head}"].float().numpy())


def select_test_queries(cartridge_dir, layer, test_conv_idx, max_total=100):
    import json as _json
    result = []
    for ci in test_conv_idx:
        meta_path = (cartridge_dir / "conversations" / f"conv_{ci:04d}"
                     / "example.json")
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = _json.load(f)
        q_tokens = meta.get("question_tokens", 0)
        if q_tokens > 0:
            result.append((ci, q_tokens - 1))
    if len(result) > max_total:
        rng_sub = np.random.default_rng(42)
        sel = rng_sub.choice(len(result), max_total, replace=False)
        result = [result[i] for i in sorted(sel)]
    return result


def plot_training_curves(all_histories, budgets, out_dir, head_label, entropy):
    """Plot train/val loss curves: one subplot per budget, lines per init."""
    n_budgets = len(budgets)
    fig, axes = plt.subplots(1, n_budgets, figsize=(5 * n_budgets, 4), squeeze=False)
    fig.suptitle(f"Training Curves — {head_label} (ent={entropy:.2f})",
                 fontsize=14, fontweight="bold")

    for bi, b in enumerate(budgets):
        ax = axes[0][bi]
        for init_name in INIT_METHODS:
            key = f"Learned-{init_name}"
            if key not in all_histories or b not in all_histories[key]:
                continue
            h = all_histories[key][b]
            color = INIT_COLORS[init_name]
            label = INIT_LABELS[init_name]
            if h["val_loss"]:
                ax.plot(h["val_loss"], color=color, label=label, lw=1.5)
            if h["train_loss"]:
                ax.plot(h["train_loss"], color=color, ls="--", alpha=0.4, lw=0.8)

        ax.set_title(f"Budget = {b}", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_yscale("log")
        if bi == 0:
            ax.set_ylabel("Loss (rel L2)")
        if bi == n_budgets - 1:
            ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    path = out_dir / f"training_curves_{head_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_init_comparison(per_head_results, budgets, out_dir, dataset):
    """Per-head subplots: Learned variants vs IdealTopK/vAttention."""
    n_heads = len(per_head_results)
    cols = min(n_heads, 3)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"Learned Init Comparison — {dataset}\n"
                 f"LW={LOCAL_WINDOW} sink={N_SINK} | {N_STEPS} steps | "
                 f"{N_TRAIN_QUERIES} train queries",
                 fontsize=13, fontweight="bold")

    sorted_heads = sorted(per_head_results.keys(),
                          key=lambda k: per_head_results[k]["entropy"])

    for i, hkey in enumerate(sorted_heads):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        info = per_head_results[hkey]
        agg = info["agg"]
        ent = info["entropy"]
        label = info["label"]

        # IdealTopK
        x_tk = [agg[f"IdealTopK-{b}"]["budget_mean"] for b in budgets
                if f"IdealTopK-{b}" in agg]
        y_tk = [agg[f"IdealTopK-{b}"]["error_mean"] for b in budgets
                if f"IdealTopK-{b}" in agg]
        if x_tk:
            ax.plot(x_tk, y_tk, "r--", marker="x", lw=2, ms=8, label="IdealTopK",
                    zorder=10)

        # vAttention
        x_va = [agg[f"vAttention(oracle)-{b}"]["budget_mean"] for b in budgets
                if f"vAttention(oracle)-{b}" in agg]
        y_va = [agg[f"vAttention(oracle)-{b}"]["error_mean"] for b in budgets
                if f"vAttention(oracle)-{b}" in agg]
        if x_va:
            ax.plot(x_va, y_va, "r-", marker="+", lw=2, ms=8, label="vAttention",
                    zorder=10)

        # Learned variants
        for init_name in INIT_METHODS:
            mname = f"Learned-{init_name}"
            x = [agg[f"{mname}-{b}"]["budget_mean"] for b in budgets
                 if f"{mname}-{b}" in agg]
            y = [agg[f"{mname}-{b}"]["error_mean"] for b in budgets
                 if f"{mname}-{b}" in agg]
            if x:
                ax.plot(x, y, color=INIT_COLORS[init_name],
                        marker="o", lw=2, ms=6, label=INIT_LABELS[init_name])

        ax.set_title(f"L{info['layer']}H{info['q_head']} ({label}, ent={ent:.2f})",
                     fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Budget")
        if c == 0:
            ax.set_ylabel("Rel L2 Error")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for i in range(len(sorted_heads), rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = out_dir / "init_comparison_log.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="us_presidential_debates")
    parser.add_argument("--heads", type=str, nargs="+", default=None)
    args = parser.parse_args()

    dataset = args.dataset
    cartridge_dir = Path(f"cartridge/datasets/{dataset}/vectors")
    if not cartridge_dir.exists():
        print(f"ERROR: {cartridge_dir} not found"); sys.exit(1)

    budgets = DATASET_BUDGETS.get(dataset, DEFAULT_BUDGETS)
    rng = np.random.default_rng(SEED)
    heads = HEADS
    if args.heads:
        heads = [h for h in HEADS if h["label"] in args.heads]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_root = Path("results") / f"learned_inits_{dataset}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Build methods: 5 Learned variants + 2 baselines
    learned_common = dict(
        n_train_queries=N_TRAIN_QUERIES,
        lr=0.01, n_steps=N_STEPS,
        nested_budget=False,
        exact_denominator=False,
        n_sink=N_SINK, local_window=LOCAL_WINDOW,
        batch_size=256,
        early_stop_patience=500,
        lr_decay_step=500, lr_decay_gamma=0.5,
    )
    learned_methods = []
    for init_name in INIT_METHODS:
        learned_methods.append(LearnedCoreset(init=init_name, **learned_common))

    baselines = [IdealTopK(), VAttentionOracle()]
    all_methods = learned_methods + baselines

    print(f"Methods: {[m.name for m in all_methods]}")
    print(f"Budgets: {budgets}")
    print(f"Steps: {N_STEPS}")
    print(f"Heads: {[h['label'] for h in heads]}")
    print(f"Output: {out_root}\n")

    per_head_results = {}
    head_meta = []

    for hi, head in enumerate(heads):
        layer = head["layer"]
        q_head = head["q_head"]
        kv_head = head["kv_head"]
        label = head["label"]

        print(f"{'='*60}")
        print(f"HEAD {hi+1}/{len(heads)}: L{layer} H{q_head} KV{kv_head} ({label})")
        print(f"{'='*60}")

        K_ctx, V_ctx, Q_ctx = load_context_kv(cartridge_dir, layer, q_head, kv_head)
        ctx_len = K_ctx.shape[0]

        conv_qs = load_all_conv_queries(cartridge_dir, layer, q_head)
        n_conv = len(conv_qs)
        n_train_conv = int(n_conv * 0.8)
        perm = rng.permutation(n_conv)
        train_conv_idx = sorted(perm[:n_train_conv])
        test_conv_idx = sorted(perm[n_train_conv:])

        Q_train_all = np.concatenate(
            [conv_qs[i] for i in train_conv_idx if len(conv_qs[i]) > 0], axis=0)
        if N_TRAIN_QUERIES > 0 and Q_train_all.shape[0] > N_TRAIN_QUERIES:
            idx = rng.choice(Q_train_all.shape[0], N_TRAIN_QUERIES, replace=False)
            Q_train = Q_train_all[idx]
        else:
            Q_train = Q_train_all
        print(f"  Context: {ctx_len:,} tokens, Train Q: {Q_train.shape[0]:,}")

        test_q_list = select_test_queries(cartridge_dir, layer, test_conv_idx)
        print(f"  Test queries: {len(test_q_list)}")

        conv_to_qs = {}
        for ci, qi in test_q_list:
            conv_to_qs.setdefault(ci, []).append(qi)

        # Precompute test problems (shared across methods)
        print(f"  Precomputing test problems...")
        test_problems = []
        for ci, q_indices in conv_to_qs.items():
            K_qa, V_qa = load_conv_kv(cartridge_dir, ci, layer, kv_head)
            K_full = np.concatenate([K_ctx, K_qa], axis=0)
            V_full = np.concatenate([V_ctx, V_qa], axis=0)
            for qi in q_indices:
                qpos = ctx_len + qi
                q = conv_qs[ci][qi]
                n_causal = qpos + 1
                keys_c = K_full[:n_causal]
                vals_c = V_full[:n_causal]
                full_out, logits, _ = full_attention(q, keys_c, vals_c, HEAD_DIM)
                lw_start = max(N_SINK, n_causal - LOCAL_WINDOW)
                special_start = min(lw_start, ctx_len)
                special_idx = np.concatenate([
                    np.arange(N_SINK, dtype=np.intp),
                    np.arange(special_start, n_causal, dtype=np.intp),
                ])
                candidate_idx = np.arange(N_SINK, special_start, dtype=np.intp)
                problem = AttentionInput(
                    query=q, keys=keys_c, values=vals_c,
                    head_dim=HEAD_DIM, logits=logits,
                    special_idx=special_idx, candidate_idx=candidate_idx)
                test_problems.append((problem, full_out))
        print(f"    {len(test_problems)} problems cached")

        # Evaluate one method at a time to keep memory low
        import gc
        all_per_query = [{} for _ in test_problems]
        all_histories = {}

        for m in all_methods:
            t0 = time.time()
            sig = inspect.signature(m.prepare)
            if "train_queries" in sig.parameters:
                m.prepare(K_ctx, V_ctx, HEAD_DIM, queries=Q_ctx,
                          query_positions=[], seed=SEED, train_queries=Q_train)
            else:
                m.prepare(K_ctx, V_ctx, HEAD_DIM, queries=Q_ctx,
                          query_positions=[], seed=SEED)
            prep_t = time.time() - t0

            t0 = time.time()
            for qi, (problem, full_out) in enumerate(test_problems):
                for b in budgets:
                    try:
                        out = m.run(problem, b, rng)
                        err = relative_l2_error(out.output, full_out)
                    except Exception:
                        err = float("nan")
                        out = type("O", (), {"actual_budget": b})()
                    all_per_query[qi][f"{m.name}-{b}"] = {
                        "error": err,
                        "budget": out.actual_budget,
                        "requested_budget": int(b),
                    }
            eval_t = time.time() - t0

            # Collect training history before clearing
            if hasattr(m, "_training_history"):
                all_histories[m.name] = dict(m._training_history)

            print(f"    {m.name}: prep={prep_t:.1f}s eval={eval_t:.1f}s")

            # Clear learned caches to free memory
            if hasattr(m, "_learned_cache"):
                m._learned_cache = {}
            gc.collect()

        all_results = all_per_query
        print(f"  Done: {len(test_problems)} queries x {len(all_methods)} methods")

        agg = aggregate_results(all_results)

        # Print summary
        print(f"\n  {'Method':<30s} {'Budget':>7s} {'Error':>10s}")
        print(f"  {'─'*50}")
        for key in sorted(agg.keys()):
            e = agg[key]
            print(f"  {key:<30s} {e['budget_mean']:>7.0f} {e['error_mean']:>10.6f}")

        # Plot training curves for this head
        plot_training_curves(all_histories, budgets, out_root, label, head["entropy"])

        per_head_results[hi] = {
            "agg": agg,
            "layer": layer,
            "q_head": q_head,
            "label": label,
            "entropy": head["entropy"],
        }
        head_meta.append({"percentile": head["percentile"]})

        # Save per-head JSON
        head_path = out_root / f"head_{label}.json"
        with open(head_path, "w") as f:
            json.dump({
                "layer": layer, "q_head": q_head, "kv_head": kv_head,
                "label": label, "entropy": head["entropy"],
                "n_queries": len(test_problems),
                "aggregated_stats": agg,
            }, f, indent=2)
        print()

    # Plot init comparison across all heads
    plot_init_comparison(per_head_results, budgets, out_root, dataset)

    # Save spec
    with open(out_root / "spec.json", "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "dataset": dataset,
            "init_methods": INIT_METHODS,
            "budgets": budgets,
            "n_steps": N_STEPS,
            "n_train_queries": N_TRAIN_QUERIES,
            "n_sink": N_SINK, "local_window": LOCAL_WINDOW,
        }, f, indent=2)

    print(f"\nResults saved to: {out_root}")


if __name__ == "__main__":
    main()
