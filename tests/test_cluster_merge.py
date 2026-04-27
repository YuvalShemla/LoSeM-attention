"""
Test: 4096 KMeans clusters → keep top-K clusters as-is,
merge remaining into super-clusters by proximity.

Approach:
1. KMeans C=4096 on candidate keys
2. Oracle top-K on individual keys (half budget)
3. Remove top-K from clusters
4. Score all 4096 clusters by q·mean_key/√d + log(count)
5. Keep top-Kc clusters as-is (fine-grained)
6. Merge remaining into M super-clusters by centroid proximity
7. Joint softmax over special + topK + fine_clusters + super_clusters
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import time
from pathlib import Path
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, cached_flat_kmeans, flat_kmeans,
)
from src.evaluation.data_loader import load_examples


def run_merged_clusters(q, keys, values, logits, sp, cand, d,
                        budget, n_base_clusters=4096,
                        n_fine=64, seed=42):
    """
    Oracle topK + fine clusters (kept) + merged super-clusters.

    budget: total budget
    n_base_clusters: number of base KMeans clusters
    n_fine: how many top-scored clusters to keep as-is
    """
    sqrt_d = np.sqrt(d)
    n_cand = len(cand)
    cand_keys = keys[cand]
    cand_vals = values[cand]

    # Half budget for oracle top-K
    b_topk = budget // 2
    b_cluster = budget - b_topk
    b_topk = min(b_topk, n_cand)

    # Oracle top-K
    cand_logits = logits[cand]
    if b_topk < n_cand:
        topk_local = np.argpartition(cand_logits, -b_topk)[-b_topk:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = cand[topk_local]

    # KMeans on candidate keys
    n_clust = min(n_base_clusters, n_cand)
    centroids, labels = cached_flat_kmeans(cand_keys, n_clust, seed=seed)

    # Compute per-cluster stats
    cand_keys_f = cand_keys.astype(np.float64)
    cand_vals_f = cand_vals.astype(np.float64)
    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    v_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(labels, weights=cand_keys_f[:, j], minlength=n_clust)
        v_sums[:, j] = np.bincount(labels, weights=cand_vals_f[:, j], minlength=n_clust)
    cnts = np.bincount(labels, minlength=n_clust).astype(np.float64)

    # Remove top-K from clusters
    sel_labels = labels[topk_local]
    sel_k = cand_keys_f[topk_local]
    sel_v = cand_vals_f[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_clust)
    sel_cnts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
    cnts_r = cnts - sel_cnts

    # Active clusters (non-empty after removal)
    active = np.where(cnts_r > 0)[0]
    n_active = len(active)

    if n_active <= b_cluster:
        # Enough budget for all clusters individually
        n_fine_actual = n_active
        n_super = 0
        fine_idx = active
        super_clusters = []
    else:
        # Score all active clusters
        q64 = q.astype(np.float64)
        scores_c = np.empty(n_active, dtype=np.float64)
        for i, c in enumerate(active):
            mk = k_sums[c] / cnts_r[c]
            scores_c[i] = float(q64 @ mk) / sqrt_d + np.log(cnts_r[c])

        # Keep top n_fine clusters as-is
        n_fine_actual = min(n_fine, b_cluster - 1)  # leave at least 1 slot for super-cluster
        n_fine_actual = min(n_fine_actual, n_active)
        n_super = b_cluster - n_fine_actual

        top_fine = np.argpartition(scores_c, -n_fine_actual)[-n_fine_actual:]
        fine_idx = active[top_fine]

        # Remaining clusters → merge into n_super super-clusters
        remaining_mask = np.ones(n_active, dtype=bool)
        remaining_mask[top_fine] = False
        remaining_idx = active[remaining_mask]
        n_remaining = len(remaining_idx)

        if n_remaining > 0 and n_super > 0:
            # Merge by centroid proximity: KMeans on remaining centroids
            remaining_centroids = np.empty((n_remaining, d), dtype=np.float32)
            for i, c in enumerate(remaining_idx):
                remaining_centroids[i] = (k_sums[c] / cnts_r[c]).astype(np.float32)

            n_super = min(n_super, n_remaining)
            _, merge_labels = flat_kmeans(
                remaining_centroids, n_super, seed=seed + 1, n_iter=20,
            )

            # Build super-cluster stats
            super_clusters = []
            for s in range(n_super):
                members = remaining_idx[merge_labels == s]
                if len(members) == 0:
                    continue
                sk = k_sums[members].sum(axis=0)
                sv = v_sums[members].sum(axis=0)
                sc = cnts_r[members].sum()
                super_clusters.append((sk, sv, sc))
        else:
            super_clusters = []

    # Build joint softmax
    n_sp = len(sp)
    n_topk = len(topk_local)
    n_f = len(fine_idx) if isinstance(fine_idx, np.ndarray) else n_fine_actual
    n_s = len(super_clusters)
    n_total = n_sp + n_topk + n_f + n_s

    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float32)
    q64 = q.astype(np.float64)

    # Special
    scores[:n_sp] = logits[sp].astype(np.float64)
    out_vals[:n_sp] = values[sp]

    # TopK individuals
    off = n_sp
    scores[off:off+n_topk] = logits[topk_global].astype(np.float64)
    out_vals[off:off+n_topk] = values[topk_global]

    # Fine clusters (kept as-is)
    off = n_sp + n_topk
    for i, c in enumerate(fine_idx):
        nc = cnts_r[c]
        mk = k_sums[c] / nc
        mv = (v_sums[c] / nc).astype(np.float32)
        ml = float(q64 @ mk) / sqrt_d
        scores[off + i] = ml + np.log(nc)
        out_vals[off + i] = mv

    # Super-clusters (merged)
    off = n_sp + n_topk + n_f
    for i, (sk, sv, sc) in enumerate(super_clusters):
        mk = sk / sc
        mv = (sv / sc).astype(np.float32)
        ml = float(q64 @ mk) / sqrt_d
        scores[off + i] = ml + np.log(sc)
        out_vals[off + i] = mv

    w = softmax(scores).astype(np.float32)
    output = w @ out_vals

    return output, n_total


def main():
    budgets = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    heads = [
        ('math_calc', 25, 1, 0, 'p25 ent=2.88'),
        ('math_calc', 2, 13, 3, 'p50 ent=3.58'),
        ('math_calc', 30, 13, 3, 'p75 ent=4.71'),
        ('math_calc', 0, 22, 5, 'p100 ent=10.19'),
    ]

    for task, layer, qh, kvh, label in heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K, V = ex['Q'], ex['K'], ex['V']
        d = 128; q = Q[-1]
        full_out, logits, weights = full_attention(q, K, V, d)
        sp, cand = compute_special_indices(len(K), 1, 0)

        print(f"\n{'='*80}", flush=True)
        print(f"  {label} — {len(cand)} candidates", flush=True)
        print(f"{'='*80}", flush=True)

        header = "                " + "  ".join(f"B={b:7d}" for b in budgets)
        print(header, flush=True)

        # Baseline: KClusterTopK C=4096 (all clusters, no merging)
        from src.algorithms.kcluster_topk import KClusterTopK
        from src.algorithms.idealized_methods import IdealTopK, IdealEqualWeightSplits
        from src.algorithms.base import AttentionInput
        problem = AttentionInput(query=q, keys=K, values=V, head_dim=d,
            logits=logits, special_idx=sp, candidate_idx=cand)
        rng = np.random.default_rng(42)

        for name, algo in [("IdealTopK", IdealTopK()), ("EWS", IdealEqualWeightSplits())]:
            algo.prepare(K, V, d, seed=42)
            row = f"{name:15s} "
            for b in budgets:
                out = algo.run(problem, b, rng)
                row += f"  {relative_l2_error(out.output, full_out):.6f}"
            print(row, flush=True)

        kc = KClusterTopK(n_clusters=4096, topk_method='oracle', order=0)
        kc.prepare(K, V, d, seed=42)
        row = "KC-4096         "
        for b in budgets:
            out = kc.run(problem, b, rng)
            row += f"  {relative_l2_error(out.output, full_out):.6f}"
        print(row, flush=True)

        # Merged clusters: vary n_fine
        for n_fine in [16, 32, 64, 128, 256]:
            row = f"Merge-F{n_fine:<4d}C4096 "
            for b in budgets:
                out, actual = run_merged_clusters(
                    q, K, V, logits, sp, cand, d, b,
                    n_base_clusters=4096, n_fine=n_fine, seed=42,
                )
                err = relative_l2_error(out, full_out)
                row += f"  {err:.6f}"
            print(row, flush=True)

    from src.core import clear_kmeans_cache
    clear_kmeans_cache()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
