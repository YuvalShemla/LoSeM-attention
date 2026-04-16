"""
Key Cluster PCA Analysis.

Clusters keys with K-means, then for each cluster measures
how much within-cluster variance is explained by a few PCA
dimensions of the deviations from the cluster centroid.

Compared against a baseline of random (non-clustered)
groups of the same sizes — if clustering concentrates
variance into fewer dims, it means nearby keys share
a low-dimensional subspace of variation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from ..core import flat_kmeans
from ..evaluation.plotting import setup_style, save_figure


# ═══════════════════════════════════════════════════════
# Compute
# ═══════════════════════════════════════════════════════

def _cluster_pca_stats(
    vectors: np.ndarray,
    pca_dims: Tuple[int, ...],
) -> Dict:
    """PCA on deviations from mean for a single group.

    Returns cumulative variance ratios at each dim count,
    the eigenvalues, and the mean residual norms.
    """
    n, d = vectors.shape
    if n < 2:
        return None

    mean = vectors.mean(axis=0)
    delta = vectors - mean
    norms_sq = np.sum(delta ** 2, axis=1)
    total_var = norms_sq.mean()

    if total_var < 1e-12:
        return {
            "cum_var": {dim: 1.0 for dim in pca_dims},
            "eigenvalues": np.zeros(min(n - 1, d)),
            "total_var": float(total_var),
            "mean_norm": float(np.sqrt(norms_sq).mean()),
            "n": n,
        }

    # Covariance and PCA
    cov = (delta.T @ delta) / n
    evals, _ = np.linalg.eigh(cov)
    evals = evals[::-1].copy()
    evals = np.maximum(evals, 0)

    cum = np.cumsum(evals) / max(evals.sum(), 1e-12)
    cum_at_dims = {}
    for dim in pca_dims:
        idx = min(dim, len(cum)) - 1
        cum_at_dims[dim] = float(cum[idx])

    return {
        "cum_var": cum_at_dims,
        "eigenvalues": evals,
        "total_var": float(total_var),
        "mean_norm": float(np.sqrt(norms_sq).mean()),
        "n": n,
    }


def compute_key_cluster_pca(
    K: np.ndarray,
    head_dim: int,
    n_clusters: int = 1024,
    pca_dims: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    n_random_trials: int = 5,
    min_cluster_size: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Cluster keys, then measure within-cluster PCA
    concentration vs random baseline.

    K is the full key matrix (seq_len, head_dim).
    Excludes the sink token at position 0.
    """
    K_no_sink = K[1:]
    N = len(K_no_sink)

    print(f"        {N:,} keys (excl. sink), "
          f"{n_clusters} clusters")

    # ── K-means ──
    print(f"        K-means clustering...")
    n_c = min(n_clusters, N)
    centroids, labels = flat_kmeans(
        K_no_sink, n_c, seed=seed,
    )

    # Build cluster index
    cluster_indices = {}
    for i, c in enumerate(labels):
        c = int(c)
        if c not in cluster_indices:
            cluster_indices[c] = []
        cluster_indices[c].append(i)

    sizes = [
        len(v) for v in cluster_indices.values()
    ]
    active = [
        c for c, v in cluster_indices.items()
        if len(v) >= min_cluster_size
    ]

    print(f"        {len(active)} clusters with "
          f">={min_cluster_size} keys  "
          f"(median size={np.median(sizes):.0f})")

    # ── Per-cluster PCA ──
    print(f"        Computing per-cluster PCA...")
    cluster_stats = {}
    for c in active:
        idx = cluster_indices[c]
        vecs = K_no_sink[idx]
        stats = _cluster_pca_stats(vecs, pca_dims)
        if stats is not None:
            cluster_stats[c] = stats

    # Aggregate: cumulative variance at each PCA dim
    # across all clusters (weighted by cluster size)
    cum_var_per_dim = {dim: [] for dim in pca_dims}
    weights = []
    total_vars = []
    mean_norms = []
    cluster_sizes = []
    all_eigenvalues = []  # per-cluster eigenvalue arrays

    for c, st in cluster_stats.items():
        w = st["n"]
        weights.append(w)
        total_vars.append(st["total_var"])
        mean_norms.append(st["mean_norm"])
        cluster_sizes.append(st["n"])
        all_eigenvalues.append(st["eigenvalues"])
        for dim in pca_dims:
            cum_var_per_dim[dim].append(
                st["cum_var"][dim],
            )

    weights = np.array(weights, dtype=float)
    for dim in pca_dims:
        cum_var_per_dim[dim] = np.array(
            cum_var_per_dim[dim],
        )

    # ── Random baseline ──
    print(f"        Computing random baseline "
          f"({n_random_trials} trials)...")
    rng = np.random.default_rng(seed)

    # Match cluster size distribution: for each trial,
    # randomly assign keys to groups with same sizes
    random_cum_var = {
        dim: [] for dim in pca_dims
    }

    for trial in range(n_random_trials):
        perm = rng.permutation(N)
        offset = 0
        trial_results = {dim: [] for dim in pca_dims}

        for c in active:
            sz = len(cluster_indices[c])
            if offset + sz > N:
                break
            idx = perm[offset:offset + sz]
            offset += sz
            vecs = K_no_sink[idx]
            stats = _cluster_pca_stats(vecs, pca_dims)
            if stats is not None:
                for dim in pca_dims:
                    trial_results[dim].append(
                        stats["cum_var"][dim],
                    )

        for dim in pca_dims:
            if trial_results[dim]:
                random_cum_var[dim].append(
                    np.array(trial_results[dim]),
                )

        if (trial + 1) % 2 == 0:
            print(f"          trial {trial + 1}"
                  f"/{n_random_trials}")

    # Aggregate random baseline
    random_stats = {}
    for dim in pca_dims:
        all_vals = np.concatenate(random_cum_var[dim])
        random_stats[dim] = {
            "mean": float(np.mean(all_vals)),
            "median": float(np.median(all_vals)),
            "std": float(np.std(all_vals)),
            "values": all_vals,
        }

    # ── Global PCA (all keys, no clustering) ──
    print(f"        Global PCA (all keys)...")
    global_stats = _cluster_pca_stats(
        K_no_sink, pca_dims,
    )

    # Build padded eigenvalue matrix for percentile
    # computation (pad shorter arrays with zeros)
    max_eig_len = max(len(e) for e in all_eigenvalues)
    eig_matrix = np.zeros(
        (len(all_eigenvalues), max_eig_len),
    )
    for i, ev in enumerate(all_eigenvalues):
        eig_matrix[i, :len(ev)] = ev

    return {
        "cluster_cum_var": cum_var_per_dim,
        "cluster_weights": weights,
        "cluster_total_vars": np.array(total_vars),
        "cluster_mean_norms": np.array(mean_norms),
        "cluster_sizes": np.array(cluster_sizes),
        "cluster_eigenvalues": eig_matrix,
        "random_stats": random_stats,
        "global_stats": global_stats,
        "pca_dims": list(pca_dims),
        "n_clusters": n_c,
        "n_active": len(active),
        "n_keys": N,
        "head_dim": head_dim,
        "min_cluster_size": min_cluster_size,
        "n_random_trials": n_random_trials,
    }


# ═══════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════

def create_key_cluster_pca_dashboard(
    data: Dict,
    title: str,
    out_path,
):
    """
    Dashboard: within-cluster PCA vs random baseline.

    Layout 2×3:
      [0,0] Cumulative variance vs PCA dims (cluster
            vs random vs global) — box + line
      [0,1] Per-cluster cumvar distribution (histograms
            at selected dims)
      [0,2] Cluster size vs total variance scatter
      [1,0] Eigenvalue spectra (median cluster vs global)
      [1,1] Gain: cluster cumvar − random cumvar
      [1,2] Summary statistics
    """
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"Key Cluster PCA — {title}",
        fontsize=14, fontweight="bold",
    )

    dims = data["pca_dims"]
    c_cv = data["cluster_cum_var"]
    r_st = data["random_stats"]
    g_st = data["global_stats"]
    weights = data["cluster_weights"]

    # ── [0,0] Cumvar vs PCA dims: cluster vs random ──
    ax = axes[0, 0]

    # Cluster: weighted mean + unweighted median
    c_means = []
    c_medians = []
    c_p25 = []
    c_p75 = []
    r_means = []
    r_medians = []
    for dim in dims:
        vals = c_cv[dim]
        w = weights / weights.sum()
        c_means.append(float(np.average(vals, weights=w)))
        c_medians.append(float(np.median(vals)))
        c_p25.append(float(np.percentile(vals, 25)))
        c_p75.append(float(np.percentile(vals, 75)))
        r_means.append(r_st[dim]["mean"])
        r_medians.append(r_st[dim]["median"])

    xs = np.arange(len(dims))
    ax.plot(
        xs, c_means, "o-", color="#1f77b4", lw=2,
        label="Cluster (weighted mean)",
    )
    ax.plot(
        xs, c_medians, "s--", color="#1f77b4", lw=1.5,
        alpha=0.7, label="Cluster (median)",
    )
    ax.fill_between(
        xs, c_p25, c_p75,
        color="#1f77b4", alpha=0.12,
        label="Cluster P25–P75",
    )
    ax.plot(
        xs, r_means, "o-", color="#d62728", lw=2,
        label="Random (mean)",
    )
    ax.plot(
        xs, r_medians, "s--", color="#d62728", lw=1.5,
        alpha=0.7, label="Random (median)",
    )

    # Global
    g_vals = [g_st["cum_var"][d] for d in dims]
    ax.plot(
        xs, g_vals, "^-", color="#2ca02c", lw=2,
        label="Global (all keys)",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Variance Concentration")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ── [0,1] Histograms at selected dims ──
    ax = axes[0, 1]
    show_dims = [d for d in [2, 8, 32] if d in dims]
    if not show_dims:
        show_dims = dims[:3]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, dim in enumerate(show_dims):
        c_vals = c_cv[dim]
        r_vals = r_st[dim]["values"]
        color = colors[i % len(colors)]
        ax.hist(
            c_vals, bins=40, alpha=0.5, color=color,
            label=f"Cluster d={dim}",
            density=True,
        )
        ax.hist(
            r_vals, bins=40, alpha=0.25, color=color,
            linestyle="--", histtype="step", lw=2,
            label=f"Random d={dim}",
            density=True,
        )
    ax.set_xlabel("Cumulative variance explained")
    ax.set_ylabel("Density")
    ax.set_title("Distribution at Selected Dims")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── [0,2] Cluster size vs total variance ──
    ax = axes[0, 2]
    sizes = data["cluster_sizes"]
    tvars = data["cluster_total_vars"]
    ax.scatter(
        sizes, tvars, s=8, alpha=0.4,
        color="#1f77b4", edgecolors="none",
    )
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Mean within-cluster ||δk||²")
    ax.set_title("Cluster Size vs Variance")
    ax.grid(alpha=0.3)

    # ── [1,0] Eigenvalue spectra: median cluster vs global ──
    ax = axes[1, 0]
    eig_matrix = data["cluster_eigenvalues"]
    n_show = min(64, eig_matrix.shape[1])

    # Normalize each cluster's spectrum by its top eigenvalue
    top_evals = eig_matrix[:, 0:1].copy()
    top_evals[top_evals < 1e-12] = 1e-12
    normed = eig_matrix[:, :n_show] / top_evals

    # Percentiles across clusters
    med_spec = np.median(normed, axis=0)
    p25_spec = np.percentile(normed, 25, axis=0)
    p75_spec = np.percentile(normed, 75, axis=0)
    xs_eig = np.arange(1, n_show + 1)

    # Clip to positive for log scale
    med_spec = np.maximum(med_spec, 1e-15)
    p25_spec = np.maximum(p25_spec, 1e-15)
    p75_spec = np.maximum(p75_spec, 1e-15)

    ax.semilogy(
        xs_eig, med_spec, "o-", color="#1f77b4",
        lw=2, markersize=3,
        label="Cluster (median)",
    )
    ax.fill_between(
        xs_eig, p25_spec, p75_spec,
        color="#1f77b4", alpha=0.15,
        label="Cluster P25–P75",
    )

    # Global spectrum (normalized)
    g_evals = g_st["eigenvalues"]
    g_norm = g_evals[:n_show] / max(g_evals[0], 1e-12)
    g_norm = np.maximum(g_norm, 1e-15)
    ax.semilogy(
        xs_eig, g_norm, "^-", color="#2ca02c",
        lw=2, markersize=3,
        label="Global",
    )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Normalized eigenvalue (log)")
    ax.set_title("Eigenvalue Spectrum: Cluster vs Global")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── [1,1] Gain: cluster − random ──
    ax = axes[1, 1]
    gains_mean = [
        c_means[i] - r_means[i]
        for i in range(len(dims))
    ]
    gains_med = [
        c_medians[i] - r_medians[i]
        for i in range(len(dims))
    ]
    ax.bar(
        xs - 0.15, gains_mean, width=0.3,
        color="#1f77b4", alpha=0.7,
        label="Mean gain",
    )
    ax.bar(
        xs + 0.15, gains_med, width=0.3,
        color="#ff7f0e", alpha=0.7,
        label="Median gain",
    )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("Cluster − Random (cumvar)")
    ax.set_title("Clustering Advantage")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── [1,2] Summary ──
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "Summary Statistics\n",
        f"n_keys: {data['n_keys']:,}",
        f"head_dim: {data['head_dim']}",
        f"n_clusters: {data['n_clusters']}",
        f"active clusters: {data['n_active']}",
        f"min_cluster_size: {data['min_cluster_size']}",
        f"random trials: {data['n_random_trials']}",
        "",
        f"cluster sizes: "
        f"med={np.median(sizes):.0f}  "
        f"mean={np.mean(sizes):.0f}  "
        f"min={np.min(sizes)}  "
        f"max={np.max(sizes)}",
        "",
        "─── Cumulative Variance ───",
        f"{'dim':>4s}  {'clust':>7s}  {'rand':>7s}"
        f"  {'global':>7s}  {'gain':>7s}",
    ]
    for i, dim in enumerate(dims):
        lines.append(
            f"  {dim:3d}  {c_means[i]:7.3f}  "
            f"{r_means[i]:7.3f}  "
            f"{g_vals[i]:7.3f}  "
            f"{gains_mean[i]:+7.3f}"
        )

    lines.append("")
    lines.append("─── Within-Cluster Variance ───")
    lines.append(
        f"  mean ||δk||²: "
        f"{np.mean(tvars):.4f}"
    )
    lines.append(
        f"  mean ||δk||:  "
        f"{np.mean(data['cluster_mean_norms']):.4f}"
    )
    lines.append(
        f"  global ||δk||²: "
        f"{g_st['total_var']:.4f}"
    )
    ratio = (
        np.mean(tvars) / max(g_st["total_var"], 1e-12)
    )
    lines.append(
        f"  cluster/global var: {ratio:.3f}"
    )

    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        va="top", ha="left",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, out_path)
