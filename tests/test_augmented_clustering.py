"""
Test: augment keys with query-derived features before clustering.

Idea: for each key k_i, append extra dimensions that capture
how this key responds to representative queries. This adds
logit-relevant information to the geometric clustering.

Progression:
1. Standard KMeans (128d) — baseline
2. +1 dim: q_mean · k_i (mean query logit, rope)
3. +1 dim: q_mean_raw · k_i (mean query logit, no rope)
4. +P dims: P query prototypes (KMeans on queries), each key gets
   P logit scores against these prototypes
5. Logit-100Q (pure logit profile, for reference)

For each: measure within-cluster logit variance AND actual output error.
Test on code_run p50.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import torch
from pathlib import Path
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, flat_kmeans,
)
from src.evaluation.data_loader import load_examples


def within_cluster_logit_var(labels, cand_logits, n_clust):
    """Mean within-cluster logit variance."""
    vars_ = []
    for c in range(n_clust):
        mask = labels == c
        if mask.sum() < 2:
            continue
        vars_.append(cand_logits[mask].var())
    return np.mean(vars_)


def run_method(q, K, V, logits, sp, cand, d, labels, n_clust, b_topk):
    """All clusters + oracle topK. Returns error."""
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)
    n_cand = len(cand)
    cand_keys = K[cand].astype(np.float64)
    cand_vals = V[cand].astype(np.float64)
    full_out = (softmax(logits) @ V.astype(np.float64)).astype(np.float32)

    cand_logits = logits[cand]
    bt = min(b_topk, n_cand)
    topk_local = np.argpartition(cand_logits, -bt)[-bt:] if bt < n_cand else np.arange(n_cand)
    topk_global = cand[topk_local]

    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    v_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(labels, weights=cand_keys[:, j], minlength=n_clust)
        v_sums[:, j] = np.bincount(labels, weights=cand_vals[:, j], minlength=n_clust)
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)

    sel_labels = labels[topk_local]
    for j in range(d):
        k_sums[:, j] -= np.bincount(sel_labels, weights=cand_keys[topk_local, j], minlength=n_clust)
        v_sums[:, j] -= np.bincount(sel_labels, weights=cand_vals[topk_local, j], minlength=n_clust)
    cnts_r = counts - np.bincount(sel_labels, minlength=n_clust).astype(np.float64)

    active = np.where(cnts_r > 0)[0]
    n_sp = len(sp); n_topk = len(topk_local); n_total = n_sp + n_topk + len(active)
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


def main():
    d = 128; sqrt_d = np.sqrt(d)
    n_clusters = 1024; b_topk = 1024; seed = 42

    # Load code_run p50
    ex = list(load_examples(Path('data/vectors'), 'code_run',
        layer=8, head=24, kv_head=6, phase=None, max_examples=1, use_rope=True))[0]
    Q_rope, K_rope, V = ex['Q'], ex['K'], ex['V']
    q = Q_rope[-1]
    full_out, logits, _ = full_attention(q, K_rope, V, d)
    sp, cand = compute_special_indices(len(K_rope), 1, 0)
    cand_keys = K_rope[cand]
    cand_logits = logits[cand].astype(np.float64)
    n_cand = len(cand)
    n_clust = min(n_clusters, n_cand)

    # Load raw Q, K
    pt = torch.load('data/vectors/code_run/ex_000/layer_08.pt',
                     map_location='cpu', weights_only=True)
    Q_raw = pt['Q_raw_head24'].detach().float().numpy()
    K_raw = pt['K_raw_kvhead6'].detach().float().numpy()
    cand_keys_raw = K_raw[cand]

    # Training queries (last 999, excluding test)
    Q_train_rope = Q_rope[-1000:-1].astype(np.float64)
    Q_train_raw = Q_raw[-1000:-1].astype(np.float64)

    # Mean queries
    q_mean_rope = Q_train_rope.mean(axis=0)
    q_mean_raw = Q_train_raw.mean(axis=0)

    print(f"code_run p50: {n_cand} candidates, C={n_clust}, topK={b_topk}")
    print(f"||q_mean_rope|| = {np.linalg.norm(q_mean_rope):.2f}")
    print(f"||q_mean_raw||  = {np.linalg.norm(q_mean_raw):.2f}")

    # Mean logit for each key
    mean_logit_rope = (cand_keys.astype(np.float64) @ q_mean_rope) / sqrt_d  # [n_cand]
    mean_logit_raw = (cand_keys_raw.astype(np.float64) @ q_mean_raw) / sqrt_d

    # How correlated is mean_logit with actual test logit?
    corr_rope = np.corrcoef(mean_logit_rope, cand_logits)[0, 1]
    corr_raw = np.corrcoef(mean_logit_raw, cand_logits)[0, 1]
    print(f"Correlation(mean_logit_rope, test_logit) = {corr_rope:.4f}")
    print(f"Correlation(mean_logit_raw, test_logit)  = {corr_raw:.4f}")
    print(flush=True)

    results = {}

    # 1. Standard KMeans (128d)
    _, labels = flat_kmeans(cand_keys, n_clust, seed=seed, n_iter=50)
    lv = within_cluster_logit_var(labels, cand_logits, n_clust)
    err = run_method(q, K_rope, V, logits, sp, cand, d, labels, n_clust, b_topk)
    results['Std-KMeans(128d)'] = (lv, err)

    # 2-3. +1 dim: mean query logit (rope / raw)
    for name, extra_feat in [
        ('+1d mean_rope', mean_logit_rope),
        ('+1d mean_raw', mean_logit_raw),
    ]:
        # Scale extra dim to match key norm scale
        key_std = cand_keys.std()
        feat_std = extra_feat.std()
        for scale in [0.5, 1.0, 2.0, 5.0]:
            alpha = scale * key_std / (feat_std + 1e-10)
            augmented = np.column_stack([
                cand_keys.astype(np.float32),
                (extra_feat * alpha).astype(np.float32).reshape(-1, 1),
            ])
            _, labels = flat_kmeans(augmented, n_clust, seed=seed, n_iter=50)
            lv = within_cluster_logit_var(labels, cand_logits, n_clust)
            err = run_method(q, K_rope, V, logits, sp, cand, d, labels, n_clust, b_topk)
            results[f'{name} s={scale}'] = (lv, err)

    # 4. +P dims: query prototypes (KMeans on training queries)
    for n_proto in [4, 8, 16, 32, 64]:
        # Cluster training queries to get prototypes
        proto_centers, _ = flat_kmeans(
            Q_train_rope.astype(np.float32), n_proto, seed=seed, n_iter=30)
        # Each key's logit against each prototype
        proto_logits = (cand_keys.astype(np.float64) @ proto_centers.astype(np.float64).T) / sqrt_d
        # Scale
        key_std = cand_keys.std()
        feat_std = proto_logits.std()
        alpha = key_std / (feat_std + 1e-10)
        augmented = np.column_stack([
            cand_keys.astype(np.float32),
            (proto_logits * alpha).astype(np.float32),
        ])
        _, labels = flat_kmeans(augmented, n_clust, seed=seed, n_iter=50)
        lv = within_cluster_logit_var(labels, cand_logits, n_clust)
        err = run_method(q, K_rope, V, logits, sp, cand, d, labels, n_clust, b_topk)
        results[f'+{n_proto}d proto_rope'] = (lv, err)

    # 5. Same with raw query prototypes
    for n_proto in [16, 32]:
        proto_centers, _ = flat_kmeans(
            Q_train_raw.astype(np.float32), n_proto, seed=seed, n_iter=30)
        proto_logits = (cand_keys_raw.astype(np.float64) @ proto_centers.astype(np.float64).T) / sqrt_d
        key_std = cand_keys.std()
        feat_std = proto_logits.std()
        alpha = key_std / (feat_std + 1e-10)
        augmented = np.column_stack([
            cand_keys.astype(np.float32),
            (proto_logits * alpha).astype(np.float32),
        ])
        _, labels = flat_kmeans(augmented, n_clust, seed=seed, n_iter=50)
        lv = within_cluster_logit_var(labels, cand_logits, n_clust)
        err = run_method(q, K_rope, V, logits, sp, cand, d, labels, n_clust, b_topk)
        results[f'+{n_proto}d proto_raw'] = (lv, err)

    # 6. Logit-100Q reference
    Q_100 = Q_rope[-101:-1].astype(np.float64)
    feat_100 = (Q_100 @ cand_keys.astype(np.float64).T / sqrt_d).T.astype(np.float32)
    _, labels = flat_kmeans(feat_100, n_clust, seed=seed, n_iter=50)
    lv = within_cluster_logit_var(labels, cand_logits, n_clust)
    err = run_method(q, K_rope, V, logits, sp, cand, d, labels, n_clust, b_topk)
    results['Logit-100Q'] = (lv, err)

    # Print sorted by error
    print(f"\n{'Method':25s} {'Logit var':>10s} {'Output err':>11s}", flush=True)
    print("-" * 48, flush=True)
    for name, (lv, err) in sorted(results.items(), key=lambda x: x[1][1]):
        marker = " <--" if name == 'Std-KMeans(128d)' else ""
        print(f"{name:25s} {lv:10.4f} {err:11.6f}{marker}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
