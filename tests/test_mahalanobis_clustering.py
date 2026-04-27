"""
Test Mahalanobis-adapted KMeans vs standard KMeans.

Direct metric: replace each key with its cluster's mean_key,
compute softmax, measure relative L2 error of the weight vector
vs true weights. No values involved — pure clustering quality.

Compare:
  1. Standard KMeans on keys
  2. Mahalanobis KMeans (M = E[qq^T], uncentered second moment)
  3. Mahalanobis KMeans (M = Cov(q), centered covariance)
  4. PCA-projected KMeans (top-r query PCA directions)
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import compute_special_indices, softmax, flat_kmeans
from src.evaluation.data_loader import load_examples


def weight_error(q, K_real, K_approx, d):
    """Relative L2 error of attention weights when keys are approximated."""
    sqrt_d = np.sqrt(d)
    logits_real = (q.astype(np.float64) @ K_real.astype(np.float64).T / sqrt_d)
    logits_approx = (q.astype(np.float64) @ K_approx.astype(np.float64).T / sqrt_d)
    w_real = softmax(logits_real)
    w_approx = softmax(logits_approx)
    err = np.linalg.norm(w_real - w_approx) / (np.linalg.norm(w_real) + 1e-30)
    return err


def cluster_and_replace(cand_keys, labels, n_clust, d):
    """Replace each key with its cluster's mean key."""
    k_sums = np.zeros((n_clust, d), dtype=np.float64)
    for j in range(d):
        k_sums[:, j] = np.bincount(
            labels, weights=cand_keys[:, j].astype(np.float64),
            minlength=n_clust,
        )
    counts = np.bincount(labels, minlength=n_clust).astype(np.float64)
    counts = np.maximum(counts, 1)
    means = (k_sums / counts[:, None]).astype(np.float32)
    return means[labels]  # [n_cand, d] — each key replaced by its cluster mean


def run_comparison(q, Q_all, K_cand, d, n_clusters, seed=42):
    """
    Run 4 clustering variants, return weight vector errors.
    """
    n_cand = len(K_cand)
    n_clust = min(n_clusters, n_cand)
    sqrt_d = np.sqrt(d)
    results = {}

    # 1. Standard KMeans on keys
    _, labels_std = flat_kmeans(K_cand, n_clust, seed=seed, n_iter=50)
    K_approx_std = cluster_and_replace(K_cand, labels_std, n_clust, d)
    results['Standard'] = K_approx_std

    # Compute query statistics
    Q_f = Q_all.astype(np.float64)
    n_q = len(Q_f)
    mu_q = Q_f.mean(axis=0)
    mu_q_norm = np.linalg.norm(mu_q)

    # M_uncentered = E[qq^T] = (1/n) Q^T Q
    M_unc = (Q_f.T @ Q_f) / n_q
    # M_centered = Cov(q) = E[(q-mu)(q-mu)^T]
    Q_c = Q_f - mu_q[None, :]
    M_cov = (Q_c.T @ Q_c) / n_q

    # Eigendecompose both
    for name, M in [('Mahal(E[qqT])', M_unc), ('Mahal(Cov)', M_cov)]:
        eigvals, eigvecs = np.linalg.eigh(M)
        # Clip small/negative eigenvalues for stability
        eigvals = np.maximum(eigvals, 1e-8)
        # Transform: K' = K @ eigvecs @ diag(sqrt(eigvals))
        transform = eigvecs * np.sqrt(eigvals)[None, :]  # [d, d]
        K_transformed = K_cand.astype(np.float64) @ transform
        K_transformed = K_transformed.astype(np.float32)

        _, labels_m = flat_kmeans(K_transformed, n_clust, seed=seed, n_iter=50)
        K_approx_m = cluster_and_replace(K_cand, labels_m, n_clust, d)
        results[name] = K_approx_m

    # 4. PCA-projected: top-r directions of query covariance
    eigvals_cov, eigvecs_cov = np.linalg.eigh(M_cov)
    idx = np.argsort(eigvals_cov)[::-1]
    eigvals_cov = eigvals_cov[idx]
    eigvecs_cov = eigvecs_cov[:, idx]

    for r in [16, 32, 64]:
        P = eigvecs_cov[:, :r]  # [d, r] projection matrix
        K_proj = K_cand.astype(np.float64) @ P  # [n, r]
        K_proj = K_proj.astype(np.float32)
        _, labels_p = flat_kmeans(K_proj, n_clust, seed=seed, n_iter=50)
        K_approx_p = cluster_and_replace(K_cand, labels_p, n_clust, d)
        results[f'PCA-r{r}'] = K_approx_p

    return results, mu_q_norm, eigvals_cov


def main():
    d = 128
    n_clusters = 1024

    heads = [
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
        sp, cand = compute_special_indices(len(K), 1, 0)
        K_cand = K[cand]

        print(f"\n{'='*70}", flush=True)
        print(f"  {task} {label} — {len(cand)} candidates, C={n_clusters}", flush=True)
        print(f"{'='*70}", flush=True)

        # Attention stats for this query
        sqrt_d = np.sqrt(d)
        logits = (q.astype(np.float64) @ K.astype(np.float64).T / sqrt_d)
        w = softmax(logits)
        ent = -np.sum(w[w > 0] * np.log(w[w > 0]))
        print(f"  Query entropy: {ent:.2f} nats", flush=True)

        results, mu_q_norm, eigvals = run_comparison(
            q, Q, K_cand, d, n_clusters, seed=42,
        )

        # Query distribution stats
        print(f"  ||mu_q||: {mu_q_norm:.2f}", flush=True)
        cumvar = np.cumsum(eigvals[::-1]) / eigvals.sum()
        d90 = np.searchsorted(cumvar, 0.90) + 1
        d99 = np.searchsorted(cumvar, 0.99) + 1
        print(f"  Query Cov: 90% in {d90}d, 99% in {d99}d", flush=True)

        # Build full key array with approximated candidates
        print(f"\n  Weight vector relative L2 error (lower = better):", flush=True)
        for name, K_approx_cand in results.items():
            # Build full K with original special + approximated candidates
            K_test = K.copy()
            K_test[cand] = K_approx_cand
            err = weight_error(q, K, K_test, d)
            print(f"    {name:20s}: {err:.6f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
