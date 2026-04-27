"""
Co-clustering approaches on the interaction matrix.
Test on code_run p50 only. Q_train = last 999 queries.

Methods:
1. Std-KMeans (baseline)
2. Logit-100Q (best so far)
3. Logit-999Q (raw logit profiles)
4. SVD-logit-r32: SVD on raw logit matrix A, top-32 right singular vectors
5. SVD-softmax-r32: SVD on softmax(A) — attention weight space
6. SVD-softmax-r64: same, rank 64
7. SVD-logsoftmax-r32: SVD on log(softmax(A)+eps) — log-probability space
8. Weighted-logit-999Q: exponential decay weighting on queries (recent = more weight)
9. Bregman-style: iterative KL-minimizing assignment on softmax(A)
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, flat_kmeans,
)
from src.evaluation.data_loader import load_examples


def run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk):
    """All clusters + oracle topK. Returns error."""
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)
    n_cand = len(cand)
    cand_keys = K[cand]
    cand_vals = V[cand]
    full_out = (softmax(logits) @ V.astype(np.float64)).astype(np.float32)

    cand_logits = logits[cand]
    bt = min(b_topk, n_cand)
    if bt < n_cand:
        topk_local = np.argpartition(cand_logits, -bt)[-bt:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = cand[topk_local]

    cand_keys_f = cand_keys.astype(np.float64)
    cand_vals_f = cand_vals.astype(np.float64)
    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    v_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(labels, weights=cand_keys_f[:, j], minlength=n_clust)
        v_sums[:, j] = np.bincount(labels, weights=cand_vals_f[:, j], minlength=n_clust)
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)

    sel_labels = labels[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=cand_keys_f[topk_local, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=cand_vals_f[topk_local, j], minlength=n_clust)
    cnts_r = counts - np.bincount(sel_labels, minlength=n_clust).astype(np.float64)

    active = np.where(cnts_r > 0)[0]
    n_sp = len(sp)
    n_topk = len(topk_local)
    n_total = n_sp + n_topk + len(active)
    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float32)

    scores[:n_sp] = logits[sp].astype(np.float64)
    out_vals[:n_sp] = V[sp]
    off = n_sp
    scores[off:off+n_topk] = logits[topk_global].astype(np.float64)
    out_vals[off:off+n_topk] = V[topk_global]
    off = n_sp + n_topk
    for i, c in enumerate(active):
        nc = cnts_r[c]
        mk = k_sums[c] / nc
        scores[off+i] = float(q64 @ mk) / sqrt_d + np.log(nc)
        out_vals[off+i] = (v_sums[c] / nc).astype(np.float32)

    w = softmax(scores).astype(np.float32)
    return relative_l2_error(w @ out_vals, full_out)


def bregman_kl_clustering(W, n_clust, n_iter=20, seed=42):
    """
    Bregman co-clustering with KL divergence on attention weight matrix.

    W: [n_q, n_cand] — each row is attention weights for one query.
    Cluster columns (keys) so that keys in the same cluster have
    similar weight profiles across queries.

    Minimizes: sum_q sum_i KL(w_q,i || pi_{c(i),q})
    where pi_c is the average weight profile of cluster c.
    """
    rng = np.random.default_rng(seed)
    n_q, n_cand = W.shape

    # Initialize with KMeans on weight profiles
    _, labels = flat_kmeans(W.T.astype(np.float32), n_clust, seed=seed, n_iter=10)

    for iteration in range(n_iter):
        # Compute cluster prototypes: average weight profile
        prototypes = np.zeros((n_clust, n_q), dtype=np.float64)
        counts = np.zeros(n_clust, dtype=np.float64)
        for c in range(n_clust):
            mask = labels == c
            if mask.sum() > 0:
                prototypes[c] = W[:, mask].mean(axis=1)
                counts[c] = mask.sum()

        # Reassign: for each key, find cluster with smallest KL divergence
        # KL(w[:,i] || proto[c]) = sum_q w[q,i] * log(w[q,i] / proto[c,q])
        # = sum_q w[q,i] * log(w[q,i]) - sum_q w[q,i] * log(proto[c,q])
        # First term is constant per key, so minimize: -sum_q w[q,i] * log(proto[c,q])

        # Clip prototypes for log stability
        proto_log = np.log(np.maximum(prototypes, 1e-30))  # [n_clust, n_q]

        new_labels = np.zeros(n_cand, dtype=np.int32)
        # Batch: for each key, compute -W[:,i].T @ log(proto[c,:]) for all c
        # = -(proto_log @ W)  — shape [n_clust, n_cand]
        neg_cross_ent = proto_log @ W  # [n_clust, n_cand]
        new_labels = np.argmax(neg_cross_ent, axis=0).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels


def main():
    d = 128
    n_clusters = 1024
    b_topk = 1024
    seed = 42
    sqrt_d = np.sqrt(d)

    # code_run p50 only
    ex = list(load_examples(
        Path('data/vectors'), 'code_run',
        layer=8, head=24, kv_head=6,
        phase=None, max_examples=1, use_rope=True,
    ))[0]
    Q, K, V = ex['Q'], ex['K'], ex['V']
    q = Q[-1]
    full_out, logits, _ = full_attention(q, K, V, d)
    sp, cand = compute_special_indices(len(K), 1, 0)
    cand_keys = K[cand]
    n_cand = len(cand)
    n_clust = min(n_clusters, n_cand)

    print(f"code_run p50 (ent=4.09): {n_cand} candidates, C={n_clust}")
    print(f"1024 topK removed from clusters")
    print(f"Q_train = last 999 queries\n", flush=True)

    Q_999 = Q[-1000:-1].astype(np.float64)
    Q_100 = Q[-101:-1].astype(np.float64)
    cand_keys_f = cand_keys.astype(np.float64)

    # Build interaction matrices
    A_logit = (Q_999 @ cand_keys_f.T) / sqrt_d          # [999, n_cand]
    A_soft = np.zeros_like(A_logit)                       # softmax per row
    for i in range(len(Q_999)):
        A_soft[i] = softmax(A_logit[i])

    A_logsm = np.log(np.maximum(A_soft, 1e-30))          # log-softmax

    print(f"Interaction matrix A: {A_logit.shape}", flush=True)
    print(f"A_soft row sums check: {A_soft.sum(axis=1)[:3]}", flush=True)

    results = {}

    # 1. Standard KMeans
    print("Running Std-KMeans...", flush=True)
    _, labels = flat_kmeans(cand_keys, n_clust, seed=seed, n_iter=50)
    results['Std-KMeans'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 2. Logit-100Q
    print("Running Logit-100Q...", flush=True)
    feat = (Q_100 @ cand_keys_f.T / sqrt_d).T.astype(np.float32)
    _, labels = flat_kmeans(feat, n_clust, seed=seed, n_iter=50)
    results['Logit-100Q'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 3. Logit-999Q
    print("Running Logit-999Q...", flush=True)
    _, labels = flat_kmeans(A_logit.T.astype(np.float32), n_clust, seed=seed, n_iter=50)
    results['Logit-999Q'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 4. SVD on raw logits
    print("Running SVD-logit...", flush=True)
    U, S, Vt = np.linalg.svd(A_logit, full_matrices=False)
    for r in [16, 32, 64]:
        feat = (Vt[:r].T * S[:r]).astype(np.float32)
        _, labels = flat_kmeans(feat, n_clust, seed=seed, n_iter=50)
        results[f'SVD-logit-r{r}'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 5. SVD on softmax (attention weight space)
    print("Running SVD-softmax...", flush=True)
    U_s, S_s, Vt_s = np.linalg.svd(A_soft, full_matrices=False)
    for r in [16, 32, 64]:
        feat = (Vt_s[:r].T * S_s[:r]).astype(np.float32)
        _, labels = flat_kmeans(feat, n_clust, seed=seed, n_iter=50)
        results[f'SVD-softmax-r{r}'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 6. SVD on log-softmax
    print("Running SVD-logsoftmax...", flush=True)
    U_l, S_l, Vt_l = np.linalg.svd(A_logsm, full_matrices=False)
    for r in [16, 32, 64]:
        feat = (Vt_l[:r].T * S_l[:r]).astype(np.float32)
        _, labels = flat_kmeans(feat, n_clust, seed=seed, n_iter=50)
        results[f'SVD-logsm-r{r}'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 7. Weighted logit profiles (exponential decay)
    print("Running Weighted-logit...", flush=True)
    alpha = 0.005  # decay rate
    weights_q = np.exp(-alpha * np.arange(999)[::-1])  # most recent = highest weight
    weights_q /= weights_q.sum()
    A_weighted = A_logit * np.sqrt(weights_q)[:, None]  # weight rows
    _, labels = flat_kmeans(A_weighted.T.astype(np.float32), n_clust, seed=seed, n_iter=50)
    results['Wt-logit-999Q'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # 8. Bregman KL co-clustering on softmax weights
    print("Running Bregman-KL...", flush=True)
    # Use subset of queries for speed (999 queries × 80K keys is big)
    labels = bregman_kl_clustering(A_soft, n_clust, n_iter=20, seed=seed)
    results['Bregman-KL'] = run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk)

    # Print results
    print(f"\n{'Method':20s} {'Error':>10s}", flush=True)
    print("-" * 32, flush=True)
    for name, err in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- best" if err == min(results.values()) else ""
        print(f"{name:20s} {err:10.6f}{marker}", flush=True)

    # SVD spectrum analysis
    print(f"\nSVD spectrum of A_logit:", flush=True)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    for r in [8, 16, 32, 64, 128]:
        print(f"  r={r:3d}: {cumvar[r-1]*100:.1f}% variance", flush=True)

    print(f"\nSVD spectrum of A_soft:", flush=True)
    cumvar_s = np.cumsum(S_s**2) / (S_s**2).sum()
    for r in [8, 16, 32, 64, 128]:
        print(f"  r={r:3d}: {cumvar_s[r-1]*100:.1f}% variance", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
