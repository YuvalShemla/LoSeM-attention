"""
Test: partial RoPE on Q and K, with realistic method:
  1024 KMeans clusters + 256 oracle topK (removed from clusters).

For each RoPE variant (none, proportional 25%, full):
  1. Apply same RoPE to both raw Q and raw K
  2. Compute attention weights with that RoPE
  3. KMeans on those keys → 1024 clusters
  4. Oracle topK 256 keys by logit
  5. Remove topK from cluster stats
  6. Joint softmax: special + topK + all residual clusters
  7. Measure weight vector error: ||w_real - w_approx|| / ||w_real||

"Proportional RoPE" = apply RoPE to first 25% of dimensions (32 out of 128).
The other 75% are pure content with no positional encoding.

All 5 code_run heads tested.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import torch
from pathlib import Path
from src.core import compute_special_indices, softmax, flat_kmeans


def apply_rope(x, positions, theta=500000.0, rope_dims=None, head_dim=128):
    """
    Apply RoPE to first rope_dims dimensions.
    Llama layout: dim i pairs with dim i + d/2 (half-split, not interleaved).
    """
    n, d = x.shape
    half = d // 2
    if rope_dims is None:
        rope_dims = d
    # rope_dims = how many dims get RoPE; must be even
    # Number of rotation pairs = rope_dims // 2
    n_pairs = min(rope_dims // 2, half)

    x_out = x.copy().astype(np.float64)
    positions = positions.astype(np.float64)

    for i in range(n_pairs):
        freq = 1.0 / (theta ** (2.0 * i / head_dim))
        angles = positions * freq
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        d0 = i           # first half
        d1 = i + half     # second half
        x0 = x_out[:, d0].copy()
        x1 = x_out[:, d1].copy()
        x_out[:, d0] = x0 * cos_a - x1 * sin_a
        x_out[:, d1] = x0 * sin_a + x1 * cos_a

    return x_out.astype(np.float32)


def run_topk_plus_all_clusters(q, K, sp, cand, d, labels, n_clust, b_topk):
    """
    Realistic method: KMeans clusters + oracle topK removed from clusters.

    Returns (w_approx over ALL tokens, w_true over ALL tokens).
    w_approx has: true weights for special + topK, cluster-mean-based
    weights for residual candidates.
    """
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)
    n = len(K)
    n_cand = len(cand)
    cand_keys = K[cand]

    # True attention weights
    logits = (q64 @ K.astype(np.float64).T / sqrt_d)
    w_true = softmax(logits)

    # Oracle topK
    cand_logits = logits[cand]
    b = min(b_topk, n_cand)
    if b < n_cand:
        topk_local = np.argpartition(cand_logits, -b)[-b:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = cand[topk_local]

    # Cluster stats
    cand_keys_f = cand_keys.astype(np.float64)
    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(
            labels, weights=cand_keys_f[:, j], minlength=n_clust,
        )
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)

    # Remove topK from clusters
    sel_labels = labels[topk_local]
    sel_k = cand_keys_f[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(
            sel_labels, weights=sel_k[:, j], minlength=n_clust,
        )
    sel_counts = np.bincount(sel_labels, minlength=n_clust).astype(np.float64)
    cnts_r = counts - sel_counts

    # Build joint softmax: special + topK + all residual clusters
    active = np.where(cnts_r > 0)[0]
    n_sp = len(sp)
    n_topk = len(topk_local)
    n_active = len(active)
    n_total = n_sp + n_topk + n_active

    scores = np.empty(n_total, dtype=np.float64)

    # Special
    scores[:n_sp] = logits[sp]
    # TopK individuals (true logits)
    off = n_sp
    scores[off:off + n_topk] = logits[topk_global]
    # Residual cluster reps
    off = n_sp + n_topk
    for i, c in enumerate(active):
        nc = cnts_r[c]
        mk = k_sums[c] / nc
        scores[off + i] = float(q64 @ mk) / sqrt_d + np.log(nc)

    w_hat = softmax(scores)

    # Map back to full weight vector for comparison
    w_approx_full = np.zeros(n, dtype=np.float64)
    # Special: exact
    w_approx_full[sp] = w_hat[:n_sp]
    # TopK: exact
    w_approx_full[topk_global] = w_hat[n_sp:n_sp + n_topk]
    # Cluster reps: distribute cluster weight equally among members
    for i, c in enumerate(active):
        nc = cnts_r[c]
        cluster_w = w_hat[n_sp + n_topk + i]
        # Find remaining members of this cluster (excluding topK)
        topk_set = set(topk_local.tolist())
        members = [idx for idx in range(n_cand) if labels[idx] == c and idx not in topk_set]
        if len(members) > 0:
            per_member = cluster_w / len(members)
            for m in members:
                w_approx_full[cand[m]] = per_member

    return w_approx_full, w_true


def weight_vector_error(w_real, w_approx):
    return np.linalg.norm(w_real - w_approx) / (np.linalg.norm(w_real) + 1e-30)


def main():
    d = 128
    n_clusters = 1024
    b_topk = 256
    theta = 500000.0
    seed = 42

    heads = [
        (31, 14, 3, 'p0'),
        (12, 26, 6, 'p25'),
        (8, 24, 6, 'p50'),
        (15, 21, 5, 'p75'),
        (0, 22, 5, 'p100'),
    ]

    rope_configs = [
        (0, "NoRoPE"),
        (32, "RoPE-25%"),
        (128, "FullRoPE"),
    ]

    print(f"C={n_clusters}, topK={b_topk} (removed from clusters)", flush=True)
    print(f"Metric: weight vector relative L2 error\n", flush=True)

    # Header
    print(f"{'Head':6s}", end="", flush=True)
    for _, rname in rope_configs:
        print(f"  | {'Entropy':>7s} {'Top1':>7s} {'Wt err':>9s}", end="", flush=True)
    print(flush=True)
    print("-" * 80, flush=True)

    for layer, qh, kvh, label in heads:
        pt = torch.load(
            f'data/vectors/code_run/ex_000/layer_{layer:02d}.pt',
            map_location='cpu', weights_only=True,
        )
        Q_raw = pt[f'Q_raw_head{qh}'].detach().float().numpy()
        K_raw = pt[f'K_raw_kvhead{kvh}'].detach().float().numpy()
        n = len(Q_raw)
        positions = np.arange(n)
        sp, cand = compute_special_indices(n, 1, 0)
        n_cand = len(cand)
        n_clust = min(n_clusters, n_cand)
        sqrt_d = np.sqrt(d)

        print(f"{label:6s}", end="", flush=True)

        for rope_dims, rname in rope_configs:
            # Apply same RoPE to Q and K
            Q_pr = apply_rope(Q_raw, positions, theta=theta,
                              rope_dims=rope_dims, head_dim=d)
            K_pr = apply_rope(K_raw, positions, theta=theta,
                              rope_dims=rope_dims, head_dim=d)

            q = Q_pr[-1]
            K_test = K_pr

            # Entropy of this variant
            logits = (q.astype(np.float64) @ K_test.astype(np.float64).T / sqrt_d)
            w = softmax(logits)
            ent = -np.sum(w[w > 0] * np.log(w[w > 0]))
            top1 = w.max()

            # KMeans on these keys
            _, labels = flat_kmeans(K_test[cand], n_clust, seed=seed, n_iter=50)

            # Run topK + all clusters method
            w_approx, w_true = run_topk_plus_all_clusters(
                q, K_test, sp, cand, d, labels, n_clust, b_topk,
            )

            wt_err = weight_vector_error(w_true, w_approx)
            print(f"  | {ent:7.2f} {top1:7.4f} {wt_err:9.6f}", end="", flush=True)

        print(flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
