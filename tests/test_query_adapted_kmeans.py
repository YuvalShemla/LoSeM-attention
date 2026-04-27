"""
Query-adapted KMeans: use task queries to build better clusters.

Insight: for key k_i, its logit profile across queries Q is
  l_i = Q @ k_i / sqrt(d)  — a vector telling how this key scores
  across all queries. Clustering keys by logit-profile similarity
  directly minimizes within-cluster logit variance.

Methods:
  1. Standard KMeans on keys (baseline)
  2. Logit-profile KMeans (last N queries): cluster keys by their
     logit signature across recent queries
  3. PCA-Mahalanobis (last N queries, top-r): project keys onto
     top-r query covariance directions, then KMeans

Full setup: 1024 oracle topK (removed from clusters) + all 1024
cluster reps. Measure actual attention output error.

Test on all heads of math_calc and code_run.
"""
import sys; sys.path.insert(0, ".")
import json
import numpy as np
from pathlib import Path
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, flat_kmeans,
)
from src.evaluation.data_loader import load_examples


def cluster_stats_and_remove(cand_keys, cand_vals, labels, n_clust, d,
                              topk_local):
    """Compute cluster stats, remove topK, return residual stats."""
    cand_keys_f = cand_keys.astype(np.float64)
    cand_vals_f = cand_vals.astype(np.float64)
    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    v_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(labels, weights=cand_keys_f[:, j], minlength=n_clust)
        v_sums[:, j] = np.bincount(labels, weights=cand_vals_f[:, j], minlength=n_clust)
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)

    # Remove topK
    sel_labels = labels[topk_local]
    sel_k = cand_keys_f[topk_local]
    sel_v = cand_vals_f[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_clust)
    sel_counts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
    cnts_r = counts - sel_counts
    return k_sums, v_sums, cnts_r


def run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk):
    """All clusters + oracle topK. Returns error vs full attention."""
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)
    n_cand = len(cand)
    cand_keys = K[cand]
    cand_vals = V[cand]
    full_out = (softmax(logits) @ V.astype(np.float64)).astype(np.float32)

    # Oracle topK
    cand_logits = logits[cand]
    bt = min(b_topk, n_cand)
    if bt < n_cand:
        topk_local = np.argpartition(cand_logits, -bt)[-bt:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = cand[topk_local]

    k_sums, v_sums, cnts_r = cluster_stats_and_remove(
        cand_keys, cand_vals, labels, n_clust, d, topk_local)

    active = np.where(cnts_r > 0)[0]
    n_sp = len(sp)
    n_topk = len(topk_local)
    n_active = len(active)
    n_total = n_sp + n_topk + n_active

    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float32)

    scores[:n_sp] = logits[sp].astype(np.float64)
    out_vals[:n_sp] = V[sp]
    off = n_sp
    scores[off:off + n_topk] = logits[topk_global].astype(np.float64)
    out_vals[off:off + n_topk] = V[topk_global]
    off = n_sp + n_topk
    for i, c in enumerate(active):
        nc = cnts_r[c]
        mk = k_sums[c] / nc
        mv = (v_sums[c] / nc).astype(np.float32)
        scores[off + i] = float(q64 @ mk) / sqrt_d + np.log(nc)
        out_vals[off + i] = mv

    w = softmax(scores).astype(np.float32)
    output = w @ out_vals
    return relative_l2_error(output, full_out)


def main():
    d = 128
    n_clusters = 1024
    b_topk = 1024
    seed = 42

    all_heads = []
    for task in ['math_calc', 'code_run']:
        with open(f'data/vectors/{task}/metadata.json') as f:
            meta = json.load(f)
        for h in meta['selected_heads']:
            all_heads.append((task, h['layer'], h['q_head'], h['kv_head'],
                              h['selection_label'], h['effective_entropy']))

    print(f"1024 clusters + 1024 oracle topK (removed from clusters)")
    print(f"Query-adapted clustering: use task queries to improve clusters")
    print(f"Metric: attention output relative L2 error\n", flush=True)

    methods = ['Std-KMeans', 'Logit-100Q', 'Logit-1000Q', 'PCA-r32-100Q', 'PCA-r64-100Q']
    header = f"{'Task':10s} {'Head':5s} {'Ent':>5s}"
    for m in methods:
        header += f"  {m:>12s}"
    print(header, flush=True)
    print("-" * (28 + 14 * len(methods)), flush=True)

    all_results = {m: [] for m in methods}

    for task, layer, qh, kvh, label, ent_meta in all_heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K, V = ex['Q'], ex['K'], ex['V']
        q = Q[-1]  # test query
        full_out, logits, _ = full_attention(q, K, V, d)
        sp, cand = compute_special_indices(len(K), 1, 0)
        cand_keys = K[cand]
        n_cand = len(cand)
        n_clust = min(n_clusters, n_cand)
        sqrt_d = np.sqrt(d)

        short = label.replace('_lowest', '').replace('_highest', '').replace('_median', '')
        row = f"{task:10s} {short:5s} {ent_meta:5.1f}"

        # 1. Standard KMeans on (RoPE'd) keys
        _, labels_std = flat_kmeans(cand_keys, n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_std, n_clust, b_topk)
        row += f"  {err:12.6f}"
        all_results['Std-KMeans'].append(err)

        # Training queries (all except last = test query)
        # Last 100 and last 1000
        Q_100 = Q[-101:-1].astype(np.float64)
        n_q_1000 = min(1000, len(Q) - 1)
        Q_1000 = Q[-(n_q_1000 + 1):-1].astype(np.float64)

        # 2. Logit-profile KMeans (last 100 queries)
        # Feature for each key: Q_100 @ k_i / sqrt(d) — shape [100]
        logit_features_100 = (Q_100 @ cand_keys.astype(np.float64).T / sqrt_d).T  # [n_cand, 100]
        _, labels_lp100 = flat_kmeans(logit_features_100.astype(np.float32), n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_lp100, n_clust, b_topk)
        row += f"  {err:12.6f}"
        all_results['Logit-100Q'].append(err)

        # 3. Logit-profile KMeans (last 1000 queries)
        logit_features_1000 = (Q_1000 @ cand_keys.astype(np.float64).T / sqrt_d).T  # [n_cand, 1000]
        _, labels_lp1000 = flat_kmeans(logit_features_1000.astype(np.float32), n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_lp1000, n_clust, b_topk)
        row += f"  {err:12.6f}"
        all_results['Logit-1000Q'].append(err)

        # 4. PCA-Mahalanobis (last 100 queries, r=32)
        mu_q = Q_100.mean(axis=0)
        Q_c = Q_100 - mu_q
        M_cov = (Q_c.T @ Q_c) / len(Q_100)
        eigvals, eigvecs = np.linalg.eigh(M_cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        P32 = eigvecs[:, :32]
        K_proj32 = (cand_keys.astype(np.float64) @ P32).astype(np.float32)
        _, labels_pca32 = flat_kmeans(K_proj32, n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_pca32, n_clust, b_topk)
        row += f"  {err:12.6f}"
        all_results['PCA-r32-100Q'].append(err)

        # 5. PCA-Mahalanobis (last 100 queries, r=64)
        P64 = eigvecs[:, :64]
        K_proj64 = (cand_keys.astype(np.float64) @ P64).astype(np.float32)
        _, labels_pca64 = flat_kmeans(K_proj64, n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_pca64, n_clust, b_topk)
        row += f"  {err:12.6f}"
        all_results['PCA-r64-100Q'].append(err)

        print(row, flush=True)

    # Global and per-task averages
    print("-" * (28 + 14 * len(methods)), flush=True)
    row = f"{'GLOBAL':10s} {'avg':5s} {'':>5s}"
    for m in methods:
        row += f"  {np.mean(all_results[m]):12.6f}"
    print(row, flush=True)

    for task in ['math_calc', 'code_run']:
        task_idx = [i for i, (t, *_) in enumerate(all_heads) if t == task]
        row = f"{task:10s} {'avg':5s} {'':>5s}"
        for m in methods:
            vals = [all_results[m][i] for i in task_idx]
            row += f"  {np.mean(vals):12.6f}"
        print(row, flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
