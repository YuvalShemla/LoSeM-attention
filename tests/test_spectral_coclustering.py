"""
Spectral co-clustering: SVD of the logit matrix.

1. Build logit matrix A = Q_train @ K_cand.T / sqrt(d)  [n_q × n_cand]
2. SVD: A ≈ U_r Σ_r V_r^T
3. Cluster keys on V_r (their top-r right singular vectors)
   Each key gets an r-dimensional feature capturing how it
   interacts with the query distribution.
4. Compare to Logit-100Q, Logit-999Q, and standard KMeans.

Full setup: 1024 clusters + 1024 oracle topK (removed).
Test on all code_run heads.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, flat_kmeans,
)
from src.evaluation.data_loader import load_examples
import json


def run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk):
    """All clusters + oracle topK. Returns error vs full attention."""
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
    sel_k = cand_keys_f[topk_local]
    sel_v = cand_vals_f[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=sel_k[:, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=sel_v[:, j], minlength=n_clust)
    sel_counts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
    cnts_r = counts - sel_counts

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
    sqrt_d = np.sqrt(d)

    with open('data/vectors/code_run/metadata.json') as f:
        meta = json.load(f)
    heads = [(h['layer'], h['q_head'], h['kv_head'],
              h['selection_label'], h['effective_entropy'])
             for h in meta['selected_heads']]

    print(f"Spectral co-clustering on code_run")
    print(f"1024 clusters + 1024 oracle topK (removed)")
    print(f"Q_train = last 999 queries (excluding test)\n", flush=True)

    methods = ['Std-KMeans', 'Logit-100Q', 'Logit-999Q',
               'SVD-r8', 'SVD-r16', 'SVD-r32', 'SVD-r64', 'SVD-r128']
    header = f"{'Head':5s} {'Ent':>5s}"
    for m in methods:
        header += f"  {m:>11s}"
    print(header, flush=True)
    print("-" * (12 + 13 * len(methods)), flush=True)

    all_results = {m: [] for m in methods}

    for layer, qh, kvh, label, ent_meta in heads:
        ex = list(load_examples(
            Path('data/vectors'), 'code_run',
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K, V = ex['Q'], ex['K'], ex['V']
        q = Q[-1]
        full_out, logits, _ = full_attention(q, K, V, d)
        sp, cand = compute_special_indices(len(K), 1, 0)
        cand_keys = K[cand]
        n_cand = len(cand)
        n_clust = min(n_clusters, n_cand)

        short = label.replace('_lowest', '').replace('_highest', '').replace('_median', '')
        row = f"{short:5s} {ent_meta:5.1f}"

        # Training queries
        Q_999 = Q[-1000:-1].astype(np.float64)  # last 999
        Q_100 = Q[-101:-1].astype(np.float64)    # last 100

        # Build logit matrix: A = Q_train @ K_cand.T / sqrt(d)
        # Shape: [999, n_cand]
        cand_keys_f = cand_keys.astype(np.float64)
        A = (Q_999 @ cand_keys_f.T) / sqrt_d  # [999, n_cand]

        # SVD of A (truncated)
        # A = U @ diag(S) @ V^T
        # V columns = right singular vectors = key features
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # Vt shape: [min(999, n_cand), n_cand]
        # Key features for rank r: Vt[:r, :].T @ diag(S[:r])
        # Or just Vt[:r, :].T (direction only)

        # 1. Standard KMeans
        _, labels_std = flat_kmeans(cand_keys, n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_std, n_clust, b_topk)
        row += f"  {err:11.6f}"
        all_results['Std-KMeans'].append(err)

        # 2. Logit-100Q
        feat_100 = (Q_100 @ cand_keys_f.T / sqrt_d).T.astype(np.float32)
        _, labels_l100 = flat_kmeans(feat_100, n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_l100, n_clust, b_topk)
        row += f"  {err:11.6f}"
        all_results['Logit-100Q'].append(err)

        # 3. Logit-999Q
        feat_999 = A.T.astype(np.float32)  # [n_cand, 999]
        _, labels_l999 = flat_kmeans(feat_999, n_clust, seed=seed, n_iter=50)
        err = run_method(q, K, V, logits, sp, cand, d, labels_l999, n_clust, b_topk)
        row += f"  {err:11.6f}"
        all_results['Logit-999Q'].append(err)

        # 4-8. SVD features at various ranks
        for r in [8, 16, 32, 64, 128]:
            # Key features: V_r scaled by singular values
            # This weights dimensions by importance
            key_features = (Vt[:r, :].T * S[:r][None, :]).astype(np.float32)
            _, labels_svd = flat_kmeans(key_features, n_clust, seed=seed, n_iter=50)
            err = run_method(q, K, V, logits, sp, cand, d, labels_svd, n_clust, b_topk)
            row += f"  {err:11.6f}"
            all_results[f'SVD-r{r}'].append(err)

        print(row, flush=True)

    # Averages
    print("-" * (12 + 13 * len(methods)), flush=True)
    row = f"{'avg':5s} {'':>5s}"
    for m in methods:
        row += f"  {np.mean(all_results[m]):11.6f}"
    print(row, flush=True)

    # Show SVD spectrum
    print(f"\nSVD spectrum (last head, code_run p100):", flush=True)
    cumvar = np.cumsum(S**2) / (S**2).sum()
    for r in [8, 16, 32, 64, 128, 256]:
        if r <= len(cumvar):
            print(f"  Top-{r:3d} singular values capture {cumvar[r-1]*100:.1f}% of variance", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
