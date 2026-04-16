"""
Compare cluster compactness: LSH Cross-Polytope (full dim) vs KMeans.

For each task, first selected head:
  1. LSH with k_dim = head_dim = 128 (65,537 possible buckets)
  2. KMeans with C = N_nonempty from LSH
  3. Mean and median cosine similarity for both

Usage:
    python -m src.exploration.cluster_quality_comparison
"""

import json
import sys
import numpy as np
from pathlib import Path

from src.evaluation.data_loader import (
    load_examples, load_task_metadata,
)
from src.algorithms.lsh_crosspoly_multiprobe import (
    _random_orthogonal, crosspolytope_bucket_labels,
)
from src.core import cached_flat_kmeans


VECTORS_DIR = Path("data/vectors")
SEED = 42
TASKS = [
    "math_calc", "code_run", "longbook_sum_eng",
    "kv_retrieval", "multi_doc_qa", "single_doc_qa",
]


def cluster_stats(keys, labels, n_labels):
    """
    Per-key cosine similarity AND inner product to cluster mean.
    Returns mean/median for both metrics across ALL keys.
    """
    all_cosines = []
    all_dots = []
    n_nonempty = 0
    for b in range(n_labels):
        mask = labels == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        n_nonempty += 1
        k_b = keys[mask]
        mu = np.mean(k_b, axis=0)
        mu_norm = np.linalg.norm(mu)
        dots = k_b @ mu
        all_dots.extend(dots.tolist())
        if mu_norm < 1e-10:
            all_cosines.extend([0.0] * count)
            continue
        k_norms = np.linalg.norm(k_b, axis=1)
        cos = np.where(
            k_norms > 1e-10,
            dots / (k_norms * mu_norm),
            0.0,
        )
        all_cosines.extend(cos.tolist())

    cos_arr = np.array(all_cosines)
    dot_arr = np.array(all_dots)
    return {
        "cos_mean": float(np.mean(cos_arr)),
        "cos_median": float(np.median(cos_arr)),
        "dot_mean": float(np.mean(dot_arr)),
        "dot_median": float(np.median(dot_arr)),
        "n_nonempty": n_nonempty,
    }


def run_lsh_full_dim(keys, seed):
    """LSH with k_dim = head_dim (full dimensionality)."""
    d = keys.shape[1]
    n_cp = 2 * d
    n_buckets = n_cp * n_cp + 1
    sink_b = n_cp * n_cp
    n = len(keys)
    rng = np.random.default_rng(seed)

    key_mean = np.mean(
        keys, axis=0, dtype=np.float64,
    ).astype(np.float32)
    x_c = keys.astype(np.float32) - key_mean
    x64 = x_c.astype(np.float64)

    R1 = _random_orthogonal(d, rng).astype(np.float64)
    R2 = _random_orthogonal(d, rng).astype(np.float64)
    z1 = (x64 @ R1.T).astype(np.float32)
    z2 = (x64 @ R2.T).astype(np.float32)

    labels = np.empty(n, dtype=np.int64)
    labels[0] = sink_b
    if n > 1:
        b1 = crosspolytope_bucket_labels(z1[1:])
        b2 = crosspolytope_bucket_labels(z2[1:])
        labels[1:] = b1 * n_cp + b2
    return labels, n_buckets


def main():
    rows = []

    for task in TASKS:
        meta = load_task_metadata(VECTORS_DIR, task)
        heads = meta.get("selected_heads", [])
        if not heads:
            continue
        h = heads[0]
        examples = list(load_examples(
            VECTORS_DIR, task,
            layer=h["layer"], head=h["q_head"],
            kv_head=h["kv_head"], max_examples=1,
        ))
        if not examples:
            continue
        keys = examples[0]["K"]
        n = len(keys)
        print(f"{task} ({n:,} tok)...", end=" ", flush=True)

        # LSH full dim
        lsh_labels, lsh_n = run_lsh_full_dim(keys, SEED)
        lsh = cluster_stats(keys, lsh_labels, lsh_n)
        print(f"LSH done...", end=" ", flush=True)

        # KMeans matched
        centroids, km_labels = cached_flat_kmeans(
            keys, lsh["n_nonempty"], seed=SEED,
        )
        km = cluster_stats(keys, km_labels, lsh["n_nonempty"])
        print(f"KMeans done", flush=True)

        rows.append({
            "task": task, "tokens": n,
            "n_clusters": lsh["n_nonempty"],
            **{f"lsh_{k}": v for k, v in lsh.items()
               if k != "n_nonempty"},
            **{f"km_{k}": v for k, v in km.items()
               if k != "n_nonempty"},
        })

    # ── Cosine table ──
    print("\n\n=== COSINE SIMILARITY (key to cluster mean) ===\n")
    print(f"{'Task':<20} {'Tokens':>7} {'Clust':>6}  "
          f"{'LSH mean':>9} {'LSH med':>8}  "
          f"{'KM mean':>8} {'KM med':>7}  "
          f"{'D mean':>7} {'D med':>6}")
    print("-" * 95)
    for r in rows:
        dm = r["km_cos_mean"] - r["lsh_cos_mean"]
        dd = r["km_cos_median"] - r["lsh_cos_median"]
        print(f"{r['task']:<20} {r['tokens']:>7,} {r['n_clusters']:>6,}  "
              f"{r['lsh_cos_mean']:>9.4f} {r['lsh_cos_median']:>8.4f}  "
              f"{r['km_cos_mean']:>8.4f} {r['km_cos_median']:>7.4f}  "
              f"{dm:>+7.4f} {dd:>+6.4f}")
    avg = lambda k: float(np.mean([r[k] for r in rows]))
    print("-" * 95)
    print(f"{'AVERAGE':<20} {'':>7} {'':>6}  "
          f"{avg('lsh_cos_mean'):>9.4f} {avg('lsh_cos_median'):>8.4f}  "
          f"{avg('km_cos_mean'):>8.4f} {avg('km_cos_median'):>7.4f}  "
          f"{avg('km_cos_mean')-avg('lsh_cos_mean'):>+7.4f} "
          f"{avg('km_cos_median')-avg('lsh_cos_median'):>+6.4f}")

    # ── Inner product table ──
    print("\n\n=== INNER PRODUCT (key . cluster_mean) ===\n")
    print(f"{'Task':<20} {'Tokens':>7} {'Clust':>6}  "
          f"{'LSH mean':>9} {'LSH med':>8}  "
          f"{'KM mean':>8} {'KM med':>7}  "
          f"{'D mean':>7} {'D med':>6}")
    print("-" * 95)
    for r in rows:
        dm = r["km_dot_mean"] - r["lsh_dot_mean"]
        dd = r["km_dot_median"] - r["lsh_dot_median"]
        print(f"{r['task']:<20} {r['tokens']:>7,} {r['n_clusters']:>6,}  "
              f"{r['lsh_dot_mean']:>9.4f} {r['lsh_dot_median']:>8.4f}  "
              f"{r['km_dot_mean']:>8.4f} {r['km_dot_median']:>7.4f}  "
              f"{dm:>+7.4f} {dd:>+6.4f}")
    print("-" * 95)
    print(f"{'AVERAGE':<20} {'':>7} {'':>6}  "
          f"{avg('lsh_dot_mean'):>9.4f} {avg('lsh_dot_median'):>8.4f}  "
          f"{avg('km_dot_mean'):>8.4f} {avg('km_dot_median'):>7.4f}  "
          f"{avg('km_dot_mean')-avg('lsh_dot_mean'):>+7.4f} "
          f"{avg('km_dot_median')-avg('lsh_dot_median'):>+6.4f}")

    # Save JSON
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cluster_quality_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"k_dim": 128, "seed": SEED},
                   "rows": rows}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
