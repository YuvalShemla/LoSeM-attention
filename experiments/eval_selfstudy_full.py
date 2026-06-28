#!/usr/bin/env python3
"""
Full self-study evaluation: budget sweep across algorithms.

Loads cartridge vectors, splits into train/test, runs all algorithms
with self-study training queries, and generates standard evaluation
plots (budget vs error, per-head comparison).

Usage:
  python experiments/eval_selfstudy_full.py --dataset multi_doc_qa_sanofi
  python experiments/eval_selfstudy_full.py --dataset us_presidential_debates
  python experiments/eval_selfstudy_full.py --heads p50 --skip-learned
"""

import argparse
import inspect
import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.algorithms.mq_beta_cluster import MQBetaCluster
from src.algorithms.idealized_methods import IdealTopK, VAttentionOracle
from src.evaluation.evaluator import (
    aggregate_results, weighted_aggregate_heads,
)
from src.core import full_attention, compute_special_indices, relative_l2_error
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
QUERIES_PER_CONV = 3

# Budget sweep scaled to context size (~0.5-10% of context tokens)
DATASET_BUDGETS = {
    "multi_doc_qa_sanofi":      [64, 256, 1024, 2048],       # 23K ctx
    "us_presidential_debates":  [128, 512, 2048, 4096],      # 45K ctx
    "apple_samsung_financials": [256, 1024, 4096, 8192],     # 119K ctx
}
DEFAULT_BUDGETS = [128, 512, 2048, 4096]


def load_context_kv(cartridge_dir, layer, q_head, kv_head):
    pt_path = cartridge_dir / "context" / f"layer_{layer:02d}.pt"
    tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    K = tensors[f"K_rope_kvhead{kv_head}"].float().numpy()
    V = tensors[f"V_kvhead{kv_head}"].float().numpy()
    Q = tensors[f"Q_rope_head{q_head}"].float().numpy()
    return K, V, Q


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
            if q_key in t:
                all_q.append(t[q_key].float().numpy())
            else:
                all_q.append(np.zeros((0, HEAD_DIM), dtype=np.float32))
        else:
            all_q.append(np.zeros((0, HEAD_DIM), dtype=np.float32))
    return all_q


def load_conv_kv(cartridge_dir, conv_idx, layer, kv_head):
    pt_path = (cartridge_dir / "conversations" / f"conv_{conv_idx:04d}"
               / f"layer_{layer:02d}.pt")
    tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    K = tensors[f"K_rope_kvhead{kv_head}"].float().numpy()
    V = tensors[f"V_kvhead{kv_head}"].float().numpy()
    return K, V


def select_test_queries(cartridge_dir, layer, test_conv_idx, max_total=100):
    """Select test queries from question tokens only (not model-generated answer)."""
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
        if q_tokens == 0:
            continue
        # Use the last question token (most context-aware)
        result.append((ci, q_tokens - 1))
    # Subsample if over limit
    if len(result) > max_total:
        rng_sub = np.random.default_rng(42)
        sel = rng_sub.choice(len(result), max_total, replace=False)
        result = [result[i] for i in sorted(sel)]
    return result


class _MQBetaCtx(MQBetaCluster):
    """MQBeta wrapper that uses context queries instead of self-study."""

    @property
    def name(self):
        return f"MQBeta-ctx-C{self.n_clusters}"

    @property
    def _uses_context_queries(self):
        return True


def build_methods(budgets, skip_learned=False):
    methods = []

    # MQBeta (self-study trained): one instance per budget
    for c in budgets:
        methods.append(MQBetaCluster(topk_frac=0, m_pq=8, n_clusters=c))

    # MQBeta (context-query trained): same budgets, for comparison
    for c in budgets:
        methods.append(_MQBetaCtx(topk_frac=0, m_pq=8, n_clusters=c))

    # Learned variants — 3 initializations, 1000 training steps
    if not skip_learned:
        from src.algorithms.learned.algorithm import LearnedCoreset
        _learned_common = dict(
            n_train_queries=N_TRAIN_QUERIES,
            lr=0.01, n_steps=1000,
            nested_budget=False,
            exact_denominator=False,
            n_sink=N_SINK, local_window=LOCAL_WINDOW,
            batch_size=256,
            early_stop_patience=150,
            lr_decay_step=300, lr_decay_gamma=0.5,
        )
        methods.append(LearnedCoreset(init="kmeans", **_learned_common))
        methods.append(LearnedCoreset(init="mqbeta", **_learned_common))
        methods.append(LearnedCoreset(init="first", **_learned_common))

    # TFCFW-lq — exact_denominator=False
    if not skip_learned:
        from src.algorithms.tensor_fcfw_lq.algorithm import TensorFCFWLq
        methods.append(TensorFCFWLq(
            oracle="fw", irls_iters=5,
            exact_denominator=False,
            n_sink=N_SINK, local_window=LOCAL_WINDOW,
        ))

    # Idealized baselines
    methods.append(IdealTopK())
    methods.append(VAttentionOracle())

    return methods


def build_algorithm_families(methods):
    families = []
    seen = set()
    colors = [
        ("#1f77b4", "#aec7e8"),  # blue
        ("#2ca02c", "#98df8a"),  # green
        ("#d62728", "#ff9896"),  # red
        ("#9467bd", "#c5b0d5"),  # purple
        ("#ff7f0e", "#ffbb78"),  # orange
        ("#8c564b", "#c49c94"),  # brown
        ("#e377c2", "#f7b6d2"),  # pink
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    ci = 0
    for m in methods:
        if m.kind == "idealized":
            continue
        # Group MQBeta-C* and MQBeta-ctx-C* into families
        prefix = m.name
        if prefix.startswith("MQBeta-ctx-C"):
            prefix = "MQBeta-ctx"
        elif prefix.startswith("MQBeta-C"):
            prefix = "MQBeta"
        if prefix in seen:
            continue
        seen.add(prefix)
        families.append({
            "prefix": prefix,
            "label": prefix,
            "color_topk": colors[ci % len(colors)][1],
            "color_hybrid": colors[ci % len(colors)][0],
            "marker": markers[ci % len(markers)],
            "top_k_sweep": [],
        })
        ci += 1
    return families


def prepare_method(m, K_ctx, V_ctx, Q_ctx, Q_train):
    """Prepare method, handling train_queries support and context-query variants."""
    sig = inspect.signature(m.prepare)
    uses_ctx = getattr(m, "_uses_context_queries", False)
    if "train_queries" in sig.parameters:
        tq = None if uses_ctx else Q_train  # None → falls back to context queries
        m.prepare(K_ctx, V_ctx, HEAD_DIM, queries=Q_ctx,
                  query_positions=[], seed=SEED, train_queries=tq)
    else:
        m.prepare(K_ctx, V_ctx, HEAD_DIM, queries=Q_ctx,
                  query_positions=[], seed=SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="multi_doc_qa_sanofi",
                        help="Dataset name under cartridge/datasets/")
    parser.add_argument("--heads", type=str, nargs="+", default=None)
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    cartridge_dir = Path(f"cartridge/datasets/{dataset}/vectors")
    if not cartridge_dir.exists():
        print(f"ERROR: {cartridge_dir} not found")
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    heads = HEADS
    if args.heads:
        heads = [h for h in HEADS if h["label"] in args.heads]

    budgets = DATASET_BUDGETS.get(dataset, DEFAULT_BUDGETS)

    run_name = args.name or f"selfstudy_{dataset}"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_root = Path("results") / f"{run_name}_{ts}"
    task_name = f"{dataset}_selfstudy"
    task_dir = out_root / "per_task" / task_name
    head_dir = task_dir / "per_head"
    head_dir.mkdir(parents=True, exist_ok=True)

    methods = build_methods(budgets, skip_learned=args.skip_learned)

    # Deduplicate method names for display (MQBeta-C128..C2048 → MQBeta sweep)
    unique_names = []
    for m in methods:
        if m.name not in unique_names:
            unique_names.append(m.name)
    print(f"Methods: {unique_names}")
    print(f"Budgets: {budgets}")
    print(f"Heads: {[h['label'] for h in heads]}")
    print(f"Train queries: {N_TRAIN_QUERIES}")
    print(f"Local window: {LOCAL_WINDOW}, Sink: {N_SINK}")
    print(f"Output: {out_root}\n")

    per_head_aggs = {}
    head_meta = []
    ctx_len = 0
    n_test_total = 0

    for hi, head in enumerate(heads):
        layer = head["layer"]
        q_head = head["q_head"]
        kv_head = head["kv_head"]
        label = head["label"]

        print(f"{'='*60}")
        print(f"HEAD {hi+1}/{len(heads)}: L{layer} H{q_head} KV{kv_head} ({label})")
        print(f"{'='*60}")

        K_ctx, V_ctx, Q_ctx = load_context_kv(
            cartridge_dir, layer, q_head, kv_head)
        ctx_len = K_ctx.shape[0]

        conv_qs = load_all_conv_queries(cartridge_dir, layer, q_head)
        n_conv = len(conv_qs)

        n_train_conv = int(n_conv * 0.8)
        perm = rng.permutation(n_conv)
        train_conv_idx = sorted(perm[:n_train_conv])
        test_conv_idx = sorted(perm[n_train_conv:])

        Q_train_all = np.concatenate(
            [conv_qs[i] for i in train_conv_idx if len(conv_qs[i]) > 0], axis=0)
        if Q_train_all.shape[0] > N_TRAIN_QUERIES:
            idx = rng.choice(Q_train_all.shape[0], N_TRAIN_QUERIES, replace=False)
            Q_train = Q_train_all[idx]
        else:
            Q_train = Q_train_all
        print(f"  Context: {ctx_len:,} tokens")
        print(f"  Train Q: {Q_train.shape[0]:,} (from {n_train_conv} convs)")

        test_q_list = select_test_queries(cartridge_dir, layer, test_conv_idx, max_total=100)
        n_test_total = len(test_q_list)
        print(f"  Test queries: {n_test_total}")

        conv_to_qs = {}
        for ci, qi in test_q_list:
            conv_to_qs.setdefault(ci, []).append(qi)

        # Prepare methods
        print(f"  Preparing methods...")
        for m in methods:
            t0 = time.time()
            try:
                prepare_method(m, K_ctx, V_ctx, Q_ctx, Q_train)
                print(f"    {m.name}: {time.time()-t0:.1f}s")
            except Exception as e:
                import traceback
                print(f"    {m.name}: FAILED — {e}")
                traceback.print_exc()

        # Evaluate
        print(f"  Evaluating...")
        t0 = time.time()
        all_results = []
        n_done = 0
        n_total = len(test_q_list)
        n_convs_done = 0
        n_convs_total = len(conv_to_qs)

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

                full_out, logits, _ = full_attention(
                    q, keys_c, vals_c, HEAD_DIM)

                # Special = sink + local_window + QA tokens (all kept exact)
                lw_start = max(N_SINK, n_causal - LOCAL_WINDOW)
                qa_start = ctx_len
                special_start = min(lw_start, qa_start)
                special_idx = np.concatenate([
                    np.arange(N_SINK, dtype=np.intp),
                    np.arange(special_start, n_causal, dtype=np.intp),
                ])
                candidate_idx = np.arange(N_SINK, special_start, dtype=np.intp)

                problem = AttentionInput(
                    query=q, keys=keys_c, values=vals_c,
                    head_dim=HEAD_DIM, logits=logits,
                    special_idx=special_idx, candidate_idx=candidate_idx)

                result = {}
                for m in methods:
                    for b in budgets:
                        try:
                            out = m.run(problem, b, rng)
                            err = relative_l2_error(out.output, full_out)
                        except Exception:
                            err = float("nan")
                            out = type("O", (), {"actual_budget": b})()

                        # MQBeta-C{n} / MQBeta-ctx-C{n}: only record when C matches b
                        if m.name.startswith("MQBeta-ctx-C"):
                            c_val = int(m.name.split("-C")[1])
                            if c_val == b:
                                result[f"MQBeta-ctx-{b}"] = {
                                    "error": err,
                                    "budget": out.actual_budget,
                                    "requested_budget": int(b),
                                }
                        elif m.name.startswith("MQBeta-C"):
                            c_val = int(m.name.split("-C")[1])
                            if c_val == b:
                                result[f"MQBeta-{b}"] = {
                                    "error": err,
                                    "budget": out.actual_budget,
                                    "requested_budget": int(b),
                                }
                        else:
                            result[f"{m.name}-{b}"] = {
                                "error": err,
                                "budget": out.actual_budget,
                                "requested_budget": int(b),
                            }
                all_results.append(result)
                n_done += 1

            n_convs_done += 1
            if n_convs_done % 20 == 0 or n_convs_done == n_convs_total:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"    {n_done}/{n_total} queries "
                      f"({n_convs_done}/{n_convs_total} convs) "
                      f"{elapsed:.0f}s, ~{eta:.0f}s left", flush=True)

        eval_time = time.time() - t0
        print(f"  Done: {n_done} queries in {eval_time:.1f}s")

        # Aggregate
        agg = aggregate_results(all_results)

        # Print summary
        print(f"\n  {'Method':<30s} {'Budget':>7s} {'Error':>10s}")
        print(f"  {'─'*50}")
        for key in sorted(agg.keys()):
            e = agg[key]
            print(f"  {key:<30s} {e['budget_mean']:>7.0f} {e['error_mean']:>10.6f}")

        per_head_aggs[hi] = {
            "agg": agg,
            "layer": layer,
            "q_head": q_head,
            "kv_head": kv_head,
            "selection_label": head["selection_label"],
            "effective_entropy": head["entropy"],
            "n_queries": n_done,
        }
        head_meta.append({"percentile": head["percentile"]})

        # Save per-head JSON
        tag = f"L{layer}H{q_head}_{head['selection_label']}"
        per_head_path = head_dir / f"{tag}.json"
        with open(per_head_path, "w") as f:
            json.dump({
                "layer": layer,
                "q_head": q_head,
                "kv_head": kv_head,
                "selection_label": head["selection_label"],
                "effective_entropy": head["entropy"],
                "n_queries": n_done,
                "aggregated_stats": agg,
            }, f, indent=2)
        print(f"  Saved: {per_head_path}\n")

    # Weighted aggregate across heads
    overall_agg = weighted_aggregate_heads(per_head_aggs, head_meta)
    agg_path = task_dir / "aggregated_stats.json"
    with open(agg_path, "w") as f:
        json.dump(overall_agg, f, indent=2)
    print(f"Aggregated stats: {agg_path}")

    # Save spec
    spec = {
        "date": datetime.now().isoformat(),
        "dataset": dataset,
        "task": task_name,
        "heads": [{
            "layer": h["layer"], "q_head": h["q_head"],
            "kv_head": h["kv_head"], "label": h["label"],
            "entropy": h["entropy"],
        } for h in heads],
        "methods": unique_names,
        "budgets": budgets,
        "n_sink": N_SINK,
        "local_window": LOCAL_WINDOW,
        "n_train_queries": N_TRAIN_QUERIES,
        "n_test_queries": n_test_total,
        "test_query_source": "question_tokens_only",
        "train_split": 0.8,
        "seed": SEED,
    }
    spec_path = out_root / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    # Generate plots
    print(f"\nGenerating plots...")
    from src.evaluation.plotting import (
        plot_evaluation, plot_per_head_comparison,
    )

    plot_cfg = {
        "figsize": [16, 10],
        "dpi": 200,
        "log_scale": True,
        "linear_scale": True,
        "error_bands": False,
    }
    families = build_algorithm_families(methods)

    title = (f"Self-Study Evaluation — {dataset}\n"
             f"LW={LOCAL_WINDOW}, sink={N_SINK}, "
             f"train={Q_train.shape[0]:,} queries, "
             f"test={n_test_total} queries/head")

    # Task-level aggregated plot
    plot_evaluation(
        overall_agg, task_dir, plot_cfg, budgets, families,
        title=title,
        n_queries=sum(h.get("n_queries", 0) for h in per_head_aggs.values()),
    )

    # Per-head comparison subplot
    plot_per_head_comparison(
        per_head_aggs, task_dir, plot_cfg, budgets, families,
        task_name=task_name,
        seq_desc=f"{dataset} ({ctx_len:,} ctx tokens) | "
                 f"LW={LOCAL_WINDOW} sink={N_SINK} | "
                 f"train={Q_train.shape[0]:,} test={n_test_total}",
    )

    print(f"\nResults saved to: {out_root}")
    print(f"Plots: {task_dir}/results_log.png, per_head_comparison_log.png")


if __name__ == "__main__":
    main()
