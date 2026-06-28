#!/usr/bin/env python3
"""
Smoke test: verify all algorithms work on multi_doc_qa with
optional self-study training queries.

Tests one head (L15 H22 KV5, p50 median), one query, three budgets.
Validates output is finite, error in [0,1], and error decreases with budget.

Usage:
  python experiments/smoke_test_selfstudy.py
  python experiments/smoke_test_selfstudy.py --with-self-study
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import METHOD_REGISTRY
from src.evaluation.data_loader import load_pt_example
from src.evaluation.evaluator import evaluate_query
from src.core import compute_special_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-self-study", action="store_true",
                        help="Use self-study training queries")
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 512, 4096])
    args = parser.parse_args()

    # ── Config ──
    HEAD_DIM = 128
    N_SINK = 1
    LOCAL_WINDOW = 1024
    LAYER, Q_HEAD, KV_HEAD = 15, 22, 5

    vectors_dir = Path("data/vectors")
    cartridge_dir = Path("cartridge/datasets/multi_doc_qa_sanofi/vectors")

    # ── Load test data ──
    print("Loading multi_doc_qa vectors...")
    ex_dir = vectors_dir / "multi_doc_qa" / "ex_000"
    data = load_pt_example(ex_dir, LAYER, Q_HEAD, KV_HEAD, use_rope=True)
    Q, K, V = data["Q"], data["K"], data["V"]
    seq_len = Q.shape[0]
    print(f"  Sequence: {seq_len:,} tokens")

    # Last query position as test
    qpos = seq_len - 1
    print(f"  Test query position: {qpos}")

    # ── Load self-study training queries ──
    Q_train = None
    if args.with_self_study:
        print("\nLoading self-study training queries...")
        from src.evaluation.self_study_loader import load_self_study_train_queries
        Q_train_full = load_self_study_train_queries(
            cartridge_dir, LAYER, Q_HEAD, KV_HEAD,
        )
        # Subsample to 5000 for practical runtime (matches default n_train_queries)
        MAX_TRAIN = 5000
        if Q_train_full.shape[0] > MAX_TRAIN:
            rng_sub = np.random.default_rng(42)
            idx = rng_sub.choice(Q_train_full.shape[0], MAX_TRAIN, replace=False)
            Q_train = Q_train_full[idx]
            print(f"  Self-study Q_train: {Q_train_full.shape[0]} -> subsampled to {Q_train.shape}")
        else:
            Q_train = Q_train_full
            print(f"  Self-study Q_train: {Q_train.shape}")
        del Q_train_full

    # ── Algorithms to test ──
    ALGO_NAMES = [
        "wildcat2", "wildcat3",
        "mq_beta_cluster",
        "learned",
        "tensor_fcfw_lq",
    ]

    methods = []
    for name in ALGO_NAMES:
        if name not in METHOD_REGISTRY:
            print(f"  SKIP {name} (not in registry)")
            continue
        spec = METHOD_REGISTRY[name]
        cfg = {}
        if name == "mq_beta_cluster":
            cfg = {"variants": [{"topk_frac": 0, "m_pq": 8,
                                 "n_clusters": 4096, "oracle_topk": False}]}
        elif name == "learned":
            cfg = {"init": "kmeans", "lr": 0.05, "n_steps": 200,
                   "nested_budget": True, "n_sink": 1, "local_window": 1024}
        elif name == "tensor_fcfw_lq":
            cfg = {"oracle": "fw", "irls_iters": 5,
                   "n_sink": 1, "local_window": 1024}
        elif name in ("wildcat2", "wildcat3"):
            cfg = {"num_bins": 1}

        try:
            instances = spec.cls.expand_from_config(cfg)
            methods.extend(instances)
            print(f"  {name}: {len(instances)} instance(s)")
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    # Add idealized baselines
    from src.algorithms.idealized_methods import IdealTopK, IdealSamplingIS
    methods.append(IdealTopK())
    methods.append(IdealSamplingIS())
    print(f"\nTotal methods: {len(methods)}")

    # ── Prepare ──
    print(f"\nPreparing algorithms (train_queries={'self-study' if Q_train is not None else 'context'})...")
    rng = np.random.default_rng(42)

    for m in methods:
        t0 = time.time()
        try:
            m.prepare(K, V, HEAD_DIM, queries=Q, query_positions=[qpos],
                      seed=42, train_queries=Q_train)
            dt = time.time() - t0
            print(f"  {m.name}: {dt:.1f}s")
        except Exception as e:
            print(f"  {m.name}: FAILED prepare — {e}")

    # ── Evaluate ──
    print(f"\nEvaluating on query position {qpos}...")
    results = evaluate_query(
        Q[qpos], K[:qpos + 1], V[:qpos + 1],
        methods, args.budgets, HEAD_DIM, N_SINK, LOCAL_WINDOW, rng,
    )

    # ── Report ──
    print(f"\n{'Method':<30s} {'Budget':>8s} {'Error':>10s} {'Actual':>8s}")
    print("─" * 60)

    method_errors = {}
    for key, val in sorted(results.items()):
        if key.startswith("_"):
            continue
        err = val.get("error", val.get("rel_l2_error", float("nan")))
        budget = val.get("budget", val.get("actual_budget", 0))
        print(f"  {key:<28s} {budget:>8d} {err:>10.6f}")

        # Track per-method for monotonicity check
        base = key.rsplit("-", 1)[0] if "-" in key else key
        method_errors.setdefault(base, []).append((budget, err))

    # ── Validation ──
    print(f"\n{'─' * 60}")
    print("Validation:")
    all_ok = True

    for key, val in results.items():
        if key.startswith("_"):
            continue
        err = val.get("error", val.get("rel_l2_error", float("nan")))
        if not np.isfinite(err):
            print(f"  FAIL: {key} has non-finite error: {err}")
            all_ok = False
        elif err < 0 or err > 2.0:
            print(f"  WARN: {key} error out of range: {err}")

    # Check monotonicity for budget-sweep methods
    for base, pairs in method_errors.items():
        if len(pairs) < 2:
            continue
        pairs.sort()
        for i in range(1, len(pairs)):
            if pairs[i][1] > pairs[i-1][1] + 0.01:
                print(f"  WARN: {base} not monotone: "
                      f"err@{pairs[i-1][0]}={pairs[i-1][1]:.4f} -> "
                      f"err@{pairs[i][0]}={pairs[i][1]:.4f}")

    if all_ok:
        print("  All checks passed!")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
