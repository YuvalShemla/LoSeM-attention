"""
Test PCA-r32 KMeans vs standard KMeans on full attention output error.

All 1024 clusters included + oracle topK at various budgets.
Compare on all 5 code_run heads.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, flat_kmeans,
)
from src.evaluation.data_loader import load_examples


def cluster_stats(data, labels, n_clust, d):
    data_f = data.astype(np.float64)
    sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        sums[:, j] = np.bincount(labels, weights=data_f[:, j], minlength=n_clust)
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)
    return sums, counts


def run_allclusters_topk(q, K, V, logits, sp, cand, d, labels, n_clust, budget):
    """All clusters + oracle topK. Returns output and actual_budget."""
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)
    n_cand = len(cand)
    cand_keys = K[cand]
    cand_vals = V[cand]

    # Cluster stats
    k_sums, counts = cluster_stats(cand_keys, labels, n_clust, d)
    v_sums, _ = cluster_stats(cand_vals, labels, n_clust, d)

    # Oracle topK
    b_topk = min(budget, n_cand)
    cand_logits = logits[cand]
    if b_topk < n_cand:
        topk_local = np.argpartition(cand_logits, -b_topk)[-b_topk:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = cand[topk_local]

    # Remove topK from clusters
    sel_labels = labels[topk_local]
    sel_k = cand_keys[topk_local].astype(np.float64)
    sel_v = cand_vals[topk_local].astype(np.float64)
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_clust)
    sel_counts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
    cnts_r = counts - sel_counts

    # Build joint softmax
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
    return output, n_total


def main():
    d = 128
    n_clusters = 1024
    budgets = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    seed = 42

    heads = [
        ('code_run', 31, 14, 3, 'p0 ent=0.03'),
        ('code_run', 12, 26, 6, 'p25 ent=3.42'),
        ('code_run', 8, 24, 6, 'p50 ent=4.09'),
        ('code_run', 15, 21, 5, 'p75 ent=4.92'),
        ('code_run', 0, 22, 5, 'p100 ent=10.56'),
    ]

    for task, layer, qh, kvh, label in heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K, V = ex['Q'], ex['K'], ex['V']
        q = Q[-1]
        full_out, logits, weights = full_attention(q, K, V, d)
        sp, cand = compute_special_indices(len(K), 1, 0)
        cand_keys = K[cand]
        n_cand = len(cand)
        n_clust = min(n_clusters, n_cand)

        print(f"\n{'='*75}", flush=True)
        print(f"  {label} — {n_cand} candidates, C={n_clust}", flush=True)
        print(f"{'='*75}", flush=True)

        # Standard KMeans on keys
        _, labels_std = flat_kmeans(cand_keys, n_clust, seed=seed, n_iter=50)

        # PCA KMeans variants: use last 100 queries (excluding test query) for covariance
        Q_local = Q[-101:-1].astype(np.float64)  # last 100, held-out test = Q[-1]
        mu_q = Q_local.mean(axis=0)
        Q_c = Q_local - mu_q[None, :]
        M_cov_local = (Q_c.T @ Q_c) / len(Q_local)
        eigvals, eigvecs = np.linalg.eigh(M_cov_local)
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        cumvar = np.cumsum(eigvals_sorted) / eigvals_sorted.sum()
        d90 = np.searchsorted(cumvar, 0.90) + 1
        d99 = np.searchsorted(cumvar, 0.99) + 1
        print(f"  Local Q Cov (last 100): 90% in {d90}d, 99% in {d99}d", flush=True)

        # Also compute with all queries for comparison
        Q_all_f = Q.astype(np.float64)
        mu_all = Q_all_f.mean(axis=0)
        Q_all_c = Q_all_f - mu_all[None, :]
        M_cov_all = (Q_all_c.T @ Q_all_c) / len(Q_all_f)
        eigvals_all, eigvecs_all = np.linalg.eigh(M_cov_all)
        idx_all = np.argsort(eigvals_all)[::-1]
        eigvecs_all = eigvecs_all[:, idx_all]

        clustering_variants = []

        # PCA with local queries at various ranks
        for r in [16, 32, 64]:
            P = eigvecs[:, :r]
            K_proj = cand_keys.astype(np.float64) @ P
            _, labels_p = flat_kmeans(K_proj.astype(np.float32), n_clust, seed=seed, n_iter=50)
            clustering_variants.append((f"Local-r{r}", labels_p))

        # PCA with all queries at r=32 for comparison
        P_all = eigvecs_all[:, :32]
        K_proj_all = cand_keys.astype(np.float64) @ P_all
        _, labels_all32 = flat_kmeans(K_proj_all.astype(np.float32), n_clust, seed=seed, n_iter=50)
        clustering_variants.append(("AllQ-r32", labels_all32))

        # Compare
        header = f"  {'':12s}" + "  ".join(f"B={b:5d}" for b in budgets)
        print(header, flush=True)

        all_variants = [("Std-KMeans", labels_std)] + clustering_variants
        for name, labels in all_variants:
            row = f"  {name:12s}"
            for b in budgets:
                out, _ = run_allclusters_topk(q, K, V, logits, sp, cand, d, labels, n_clust, b)
                err = relative_l2_error(out, full_out)
                row += f"  {err:.6f}"
            print(row, flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
