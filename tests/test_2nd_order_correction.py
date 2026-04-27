"""
Test PQ top-k + key clusters with 2nd-order correction.
Compares 0th-order (mean-key) vs 2nd-order (mean-key + var/2)
scoring on residual key clusters after removing PQ top-k keys.
"""
import numpy as np
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.core import (
    full_attention, compute_special_indices, softmax,
    cached_flat_kmeans, clear_kmeans_cache, relative_l2_error,
)
from src.algorithms.pq_topk import PQIndex
from src.evaluation.data_loader import load_examples

d = 128; sqrt_d = np.sqrt(d)

heads = [
    ('math_calc', 31, 14, 3, 'p0 ent=0.20'),
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
    q = Q[-1]; keys = K; values = V
    full_out, logits, weights = full_attention(q, keys, values, d)
    sp, cand = compute_special_indices(len(keys), 1, 0)
    cand_keys = keys[cand]; cand_vals = values[cand]
    n_cand = len(cand); n = len(keys)
    q64 = q.astype(np.float64)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    clear_kmeans_cache()
    n_clust = 1024

    centroids, labels = cached_flat_kmeans(
        cand_keys, min(n_clust, n_cand), seed=42,
    )
    nc_actual = min(n_clust, n_cand)

    # Precompute per-cluster: key sums, value sums, M = Σ k k^T, counts
    k_sums = np.zeros((nc_actual, d), dtype=np.float64)
    v_sums = np.zeros((nc_actual, d), dtype=np.float64)
    M = np.zeros((nc_actual, d, d), dtype=np.float64)
    cnts = np.zeros(nc_actual, dtype=np.float64)
    for c in range(nc_actual):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ck = cand_keys[mask].astype(np.float64)
        cnts[c] = len(ck)
        k_sums[c] = ck.sum(axis=0)
        v_sums[c] = cand_vals[mask].astype(np.float64).sum(axis=0)
        M[c] = ck.T @ ck

    # PQ index for approximate top-k
    pq = PQIndex(m=8, n_codes=256, seed=42)
    pq.fit(keys)
    cand_mask = np.zeros(n, dtype=bool)
    cand_mask[cand] = True
    global_to_local = np.full(n, -1, dtype=np.int64)
    global_to_local[cand] = np.arange(n_cand)

    print(f"  {'budget':>6s}  {'0th-order':>10s}  {'2nd-order':>10s}  {'ratio':>8s}")

    for budget in [64, 128, 256, 512, 1024, 2048]:
        topk_global = pq.approximate_topk(
            q, budget, candidate_mask=cand_mask,
        )
        topk_local = global_to_local[topk_global]
        valid = topk_local >= 0
        topk_local = topk_local[valid]
        topk_global = topk_global[valid]
        buse = len(topk_local)

        # Remove selected from cluster stats
        k_s = k_sums.copy()
        v_s = v_sums.copy()
        M_r = M.copy()
        cnts_r = cnts.copy()
        sel_labels = labels[topk_local]
        sel_k = cand_keys[topk_local].astype(np.float64)
        sel_v = cand_vals[topk_local].astype(np.float64)
        for j in range(d):
            k_s[:, j] -= np.bincount(
                sel_labels, weights=sel_k[:, j],
                minlength=nc_actual,
            )
            v_s[:, j] -= np.bincount(
                sel_labels, weights=sel_v[:, j],
                minlength=nc_actual,
            )
        sel_cnts = np.bincount(
            sel_labels, minlength=nc_actual,
        ).astype(np.float64)
        cnts_r -= sel_cnts
        for idx in range(buse):
            c = sel_labels[idx]
            k_vec = sel_k[idx]
            M_r[c] -= np.outer(k_vec, k_vec)

        active = cnts_r > 0
        active_idx = np.where(active)[0]
        n_active = len(active_idx)
        n_sp = len(sp)
        n_total = n_sp + buse + n_active

        scores0 = np.empty(n_total, dtype=np.float64)
        scores2 = np.empty(n_total, dtype=np.float64)
        out_v = np.empty((n_total, d), dtype=np.float32)

        scores0[:n_sp] = logits[sp].astype(np.float64)
        scores2[:n_sp] = logits[sp].astype(np.float64)
        out_v[:n_sp] = values[sp]

        off = n_sp
        scores0[off:off+buse] = logits[topk_global].astype(np.float64)
        scores2[off:off+buse] = logits[topk_global].astype(np.float64)
        out_v[off:off+buse] = values[topk_global]

        off = n_sp + buse
        for i, c in enumerate(active_idx):
            nc = cnts_r[c]
            mean_k = (k_s[c] / nc).astype(np.float32)
            mean_v = (v_s[c] / nc).astype(np.float32)
            ml = float(q64 @ mean_k.astype(np.float64)) / sqrt_d

            scores0[off+i] = ml + np.log(nc)

            qMq = float(q64 @ M_r[c] @ q64)
            var_l = qMq / (nc * d) - ml**2
            var_l = max(var_l, 0.0)
            scores2[off+i] = ml + var_l/2 + np.log(nc)

            out_v[off+i] = mean_v

        out0 = softmax(scores0).astype(np.float32) @ out_v
        out2 = softmax(scores2).astype(np.float32) @ out_v
        err0 = relative_l2_error(out0, full_out)
        err2 = relative_l2_error(out2, full_out)

        tag = "BETTER" if err2 < err0 else "worse"
        ratio = err0 / max(err2, 1e-10)
        print(
            f"  {budget:6d}  {err0:10.6f}  {err2:10.6f}  "
            f"{ratio:6.2f}x [{tag}]"
        )

    clear_kmeans_cache()

print("\nDone.")
