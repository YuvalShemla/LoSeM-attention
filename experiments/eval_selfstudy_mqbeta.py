#!/usr/bin/env python3
"""
Full self-study evaluation for MQBeta.

Compares MQBeta trained on self-study queries vs context queries,
tested on held-out self-study test queries (unseen questions).

Experiment design:
  - Split 1000 self-study conversations into train/test (800/200)
  - Train queries: Q vectors from train conversations (or context prefill)
  - Test queries: sampled from test conversations (unseen questions)
  - Sweep n_train_queries: [100, 500, 1000, 5000, 20000, ALL]
  - Run on all 5 heads across entropy spectrum
  - Baselines: IdealTopK, IdealSampling-IS (oracle, no training)

Test query selection: sample --queries-per-conv queries per test conversation,
spread across the QA portion. Adjacent tokens are highly correlated so
using all ~286 per conversation adds runtime without statistical benefit.

Usage:
  python experiments/eval_selfstudy_mqbeta.py
  python experiments/eval_selfstudy_mqbeta.py --heads p50 --queries-per-conv 3
  python experiments/eval_selfstudy_mqbeta.py --output results.json
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.mq_beta_cluster import MQBetaCluster
from src.algorithms.idealized_methods import IdealTopK, IdealSamplingIS
from src.evaluation.evaluator import evaluate_query
from src.core import compute_special_indices

# 5 heads by entropy percentile
HEADS = [
    {"layer": 31, "q_head": 14, "kv_head": 3, "label": "p0"},
    {"layer": 12, "q_head": 25, "kv_head": 6, "label": "p25"},
    {"layer": 15, "q_head": 22, "kv_head": 5, "label": "p50"},
    {"layer": 27, "q_head":  7, "kv_head": 1, "label": "p75"},
    {"layer":  0, "q_head": 22, "kv_head": 5, "label": "p100"},
]

HEAD_DIM = 128
N_SINK = 1
LOCAL_WINDOW = 1024
N_CLUSTERS = 4096
SEED = 42


def load_all_conversation_queries(cartridge_dir, layer, q_head, kv_head):
    """Load Q vectors per conversation (list of arrays)."""
    import torch
    conv_parent = cartridge_dir / "conversations"
    q_key = f"Q_rope_head{q_head}"
    pt_name = f"layer_{layer:02d}.pt"
    all_q = []
    conv_dirs = sorted(
        d for d in conv_parent.iterdir()
        if d.is_dir() and d.name.startswith("conv_")
    )
    for cd in conv_dirs:
        pt_path = cd / pt_name
        if not pt_path.exists():
            continue
        tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
        if q_key in tensors:
            all_q.append(tensors[q_key].float().numpy())
    return all_q


def load_context_kv(cartridge_dir, layer, q_head, kv_head):
    """Load context K, V, Q_context."""
    import torch
    pt_path = cartridge_dir / "context" / f"layer_{layer:02d}.pt"
    tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    K = tensors[f"K_rope_kvhead{kv_head}"].float().numpy()
    V = tensors[f"V_kvhead{kv_head}"].float().numpy()
    Q = tensors[f"Q_rope_head{q_head}"].float().numpy()
    return K, V, Q


def load_conversation_kv(cartridge_dir, conv_idx, layer, q_head, kv_head):
    """Load K, V for one conversation's QA portion."""
    import torch
    pt_path = (cartridge_dir / "conversations" / f"conv_{conv_idx:04d}"
               / f"layer_{layer:02d}.pt")
    tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    K = tensors[f"K_rope_kvhead{kv_head}"].float().numpy()
    V = tensors[f"V_kvhead{kv_head}"].float().numpy()
    return K, V


def select_test_queries(conv_qs, test_conv_idx, queries_per_conv, rng):
    """Sample queries spread across each test conversation's QA portion."""
    test_q_list = []
    for ci in test_conv_idx:
        n_tok = len(conv_qs[ci])
        if n_tok == 0:
            continue
        if n_tok <= queries_per_conv:
            indices = list(range(n_tok))
        else:
            # Spread evenly: first, last, and evenly spaced in between
            indices = np.linspace(0, n_tok - 1, queries_per_conv,
                                  dtype=int).tolist()
        for qi in indices:
            test_q_list.append((ci, qi))
    return test_q_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--n-train-sweep", type=int, nargs="+",
                        default=[100, 500, 1000, 5000, 20000])
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction of conversations for training")
    parser.add_argument("--queries-per-conv", type=int, default=5,
                        help="Test queries sampled per test conversation")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--heads", type=str, nargs="+", default=None,
                        help="Head labels to run (e.g. p0 p50), default all")
    args = parser.parse_args()

    cartridge_dir = Path("cartridge/datasets/multi_doc_qa_sanofi/vectors")
    if not cartridge_dir.exists():
        print(f"ERROR: {cartridge_dir} not found")
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    heads = HEADS
    if args.heads:
        heads = [h for h in HEADS if h["label"] in args.heads]

    all_results = []

    for head in heads:
        layer = head["layer"]
        q_head = head["q_head"]
        kv_head = head["kv_head"]
        label = head["label"]

        print(f"\n{'='*70}")
        print(f"HEAD: L{layer} H{q_head} KV{kv_head} ({label})")
        print(f"{'='*70}")

        # Load context K/V (shared prefix — this is what we compress)
        K_ctx, V_ctx, Q_ctx = load_context_kv(
            cartridge_dir, layer, q_head, kv_head)
        ctx_len = K_ctx.shape[0]
        print(f"Context: {ctx_len:,} tokens")

        # Load all conversation Q vectors (per conversation)
        conv_qs = load_all_conversation_queries(
            cartridge_dir, layer, q_head, kv_head)
        n_conv = len(conv_qs)
        print(f"Conversations: {n_conv}")

        # Train/test split on conversations
        n_train_conv = int(n_conv * args.train_split)
        n_test_conv = n_conv - n_train_conv
        perm = rng.permutation(n_conv)
        train_conv_idx = sorted(perm[:n_train_conv])
        test_conv_idx = sorted(perm[n_train_conv:])
        print(f"Split: {n_train_conv} train / {n_test_conv} test conversations")

        # Aggregate ALL train Q vectors (for M_Q covariance — 128x128, always cheap)
        Q_train_all = np.concatenate(
            [conv_qs[i] for i in train_conv_idx], axis=0)
        print(f"Train Q pool: {Q_train_all.shape[0]:,} vectors")

        # Select test queries: spread across each test conversation
        test_q_list = select_test_queries(
            conv_qs, test_conv_idx, args.queries_per_conv, rng)
        print(f"Test queries: {len(test_q_list)} "
              f"({args.queries_per_conv}/conv x {n_test_conv} convs)")

        # Group test queries by conversation (load K/V once per conv)
        conv_to_test_qs = {}
        for ci, qi in test_q_list:
            conv_to_test_qs.setdefault(ci, []).append(qi)

        # n_train_queries sweep values (cap at available)
        n_train_values = [n for n in args.n_train_sweep
                          if n <= Q_train_all.shape[0]]
        n_train_values.append(Q_train_all.shape[0])  # add "ALL"
        n_train_values = sorted(set(n_train_values))

        # ── Conditions ──
        conditions = []
        for n_tq in n_train_values:
            conditions.append({
                "name": f"selfstudy-{n_tq}",
                "type": "selfstudy",
                "n_train": n_tq,
            })
        conditions.append({
            "name": "context",
            "type": "context",
            "n_train": None,
        })

        for ci_cond, cond in enumerate(conditions):
            cond_name = cond["name"]
            print(f"\n  ── [{ci_cond+1}/{len(conditions)}] Condition: {cond_name} ──")

            # Select training queries
            if cond["type"] == "selfstudy":
                n_tq = cond["n_train"]
                if n_tq < Q_train_all.shape[0]:
                    idx = rng.choice(Q_train_all.shape[0], n_tq, replace=False)
                    Q_train = Q_train_all[idx]
                else:
                    Q_train = Q_train_all
                    n_tq = Q_train_all.shape[0]
                print(f"    Train queries: {Q_train.shape[0]:,} (self-study)")
            else:
                # Context prefill queries — subsample to 20K for tractable runtime
                max_ctx = min(Q_ctx.shape[0], 20000)
                idx = rng.choice(Q_ctx.shape[0], max_ctx, replace=False)
                Q_train = Q_ctx[idx]
                n_tq = Q_train.shape[0]
                print(f"    Train queries: {Q_train.shape[0]:,} (context prefill)")

            # Prepare MQBeta
            mqbeta = MQBetaCluster(topk_frac=0, m_pq=8, n_clusters=N_CLUSTERS)
            t0 = time.time()
            mqbeta.prepare(
                K_ctx, V_ctx, HEAD_DIM,
                queries=Q_ctx,
                query_positions=[],
                seed=SEED,
                train_queries=Q_train,
            )
            prep_time = time.time() - t0
            print(f"    MQBeta prepare: {prep_time:.1f}s")

            # Baselines
            ideal_topk = IdealTopK()
            ideal_sampling = IdealSamplingIS()
            ideal_topk.prepare(K_ctx, V_ctx, HEAD_DIM)
            ideal_sampling.prepare(K_ctx, V_ctx, HEAD_DIM)
            methods = [mqbeta, ideal_topk, ideal_sampling]

            # Evaluate on test queries
            t0 = time.time()
            all_errors = {m.name: {b: [] for b in args.budgets}
                          for m in methods}

            n_done = 0
            n_total = len(test_q_list)
            n_convs_done = 0
            n_convs_total = len(conv_to_test_qs)

            for ci, q_indices in conv_to_test_qs.items():
                K_qa, V_qa = load_conversation_kv(
                    cartridge_dir, ci, layer, q_head, kv_head)
                K_full = np.concatenate([K_ctx, K_qa], axis=0)
                V_full = np.concatenate([V_ctx, V_qa], axis=0)

                for qi in q_indices:
                    qpos = ctx_len + qi
                    q = conv_qs[ci][qi]

                    results = evaluate_query(
                        q, K_full[:qpos + 1], V_full[:qpos + 1],
                        methods, args.budgets, HEAD_DIM,
                        N_SINK, LOCAL_WINDOW, rng,
                    )

                    for key, val in results.items():
                        if key.startswith("_"):
                            continue
                        err = val.get("error", float("nan"))
                        budget = val.get("requested_budget", 0)
                        method_name = key.rsplit("-", 1)[0] if "-" in key else key
                        if method_name in all_errors and budget in all_errors[method_name]:
                            all_errors[method_name][budget].append(err)

                    n_done += 1

                n_convs_done += 1
                if n_convs_done % 20 == 0 or n_convs_done == n_convs_total:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_done) / rate if rate > 0 else 0
                    print(f"    eval: {n_done}/{n_total} queries "
                          f"({n_convs_done}/{n_convs_total} convs) "
                          f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                          flush=True)

            eval_time = time.time() - t0
            print(f"    Done: {n_done} queries in {eval_time:.1f}s")

            # Report
            print(f"\n    {'Method':<25s} {'Budget':>7s} {'Mean Err':>10s} "
                  f"{'Std':>8s} {'N':>5s}")
            print(f"    {'─'*58}")

            for m in methods:
                for b in args.budgets:
                    errs = all_errors[m.name].get(b, [])
                    if not errs:
                        continue
                    mean_err = np.mean(errs)
                    std_err = np.std(errs)
                    print(f"    {m.name:<25s} {b:>7d} {mean_err:>10.6f} "
                          f"{std_err:>8.6f} {len(errs):>5d}")

                    all_results.append({
                        "head": label,
                        "layer": layer,
                        "q_head": q_head,
                        "kv_head": kv_head,
                        "condition": cond_name,
                        "condition_type": cond["type"],
                        "n_train_queries": n_tq,
                        "method": m.name,
                        "budget": b,
                        "mean_error": float(mean_err),
                        "std_error": float(std_err),
                        "n_test": len(errs),
                        "prep_time": prep_time,
                    })

    # Save results
    output_path = args.output or "experiments/selfstudy_mqbeta_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Total entries: {len(all_results)}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Mean error across all heads (budget=1024)")
    print(f"{'='*70}")
    summary = {}
    for r in all_results:
        if r["budget"] != 1024:
            continue
        key = (r["condition"], r["method"])
        summary.setdefault(key, []).append(r["mean_error"])

    print(f"{'Condition':<30s} {'Method':<25s} {'Mean Err':>10s}")
    print(f"{'─'*68}")
    for (cond, method), errs in sorted(summary.items()):
        print(f"{cond:<30s} {method:<25s} {np.mean(errs):>10.6f}")


if __name__ == "__main__":
    main()
