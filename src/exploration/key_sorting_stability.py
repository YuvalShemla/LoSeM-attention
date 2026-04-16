"""
Key Sorting Stability Analysis.

Shared-key approach: all test queries see the same key set
(keys before the first test query, excluding sink).

K-means with 512 clusters on ALL queries (full sequence),
then test the last 25K queries. Overlap is computed only
within clusters, then median across clusters.

Four analyses:
1. Sorting Stability: Global mean + per-cluster centroid
   vs per-query group overlap at 2/4/8 splits.
2. Group Persistence: Are certain keys always close or far
   across all queries?
3. Query Agreement: Within-cluster pairwise top-K overlap.
4. PCA Subspace: Decompose score = mean_q·k + δq·k,
   project δq onto top-k PCA dims, measure how well
   the approximate scores recover the full-dim ranking.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

from ..core import flat_kmeans
from ..evaluation.plotting import setup_style, save_figure


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

def _assign_groups_from_ranks(
    ranks: np.ndarray, n_keys: int, n_groups: int,
) -> np.ndarray:
    return np.minimum(
        (ranks * n_groups // n_keys).astype(np.int32),
        n_groups - 1,
    )


def _ranks_from_order(order: np.ndarray, n: int) -> np.ndarray:
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(n, dtype=np.int32)
    return ranks


def _overlap(g1: np.ndarray, g2: np.ndarray) -> float:
    return float(np.mean(g1 == g2))


def _ema(values: np.ndarray, span: int = 50) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ═══════════════════════════════════════════════════════
# Compute — single pass
# ═══════════════════════════════════════════════════════

def compute_all_analyses(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    n_test_queries: int = 25000,
    n_query_clusters: int = 512,
    n_groups_list: Tuple[int, ...] = (2, 4, 8),
    persistence_n_groups: int = 8,
    top_k_values: Tuple[int, ...] = (10, 100),
    top_pct_values: Tuple[float, ...] = (1, 5, 10),
    max_pairs_per_cluster: int = 50,
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Single pass computing all analyses.

    Returns (global_data, clustered_data,
             persistence_data, agreement_data).
    """
    sqrt_d = np.sqrt(head_dim)
    seq_len = Q.shape[0]

    # Test queries: last n_test_queries
    n_test = min(n_test_queries, seq_len - 2)
    test_start = seq_len - n_test
    test_positions = list(range(test_start, seq_len))

    # Shared keys: K[1:test_start] (exclude sink, before
    # first test query)
    K_shared = K[1:test_start]
    N_keys = len(K_shared)
    if N_keys < max(n_groups_list) * 2:
        raise ValueError(
            f"Only {N_keys} shared keys, need more"
        )

    print(f"        {N_keys:,} shared keys, "
          f"{n_test} test queries")

    # ── K-means on ALL queries ──
    print(f"        K-means ({n_query_clusters} "
          f"clusters on {seq_len:,} queries)...")
    n_clusters = min(n_query_clusters, seq_len)
    centroids, all_labels = flat_kmeans(
        Q, n_clusters, seed=seed,
    )
    test_labels = all_labels[test_positions]

    # Group test queries by cluster
    cluster_test_indices = defaultdict(list)
    for i, qpos in enumerate(test_positions):
        c = int(test_labels[i])
        cluster_test_indices[c].append(i)

    # ── Precompute: global mean groups ──
    mean_q = Q[test_positions].mean(axis=0)
    mean_scores = (mean_q @ K_shared.T) / sqrt_d
    mean_order = np.argsort(mean_scores)[::-1]
    mean_ranks = _ranks_from_order(mean_order, N_keys)
    mean_groups = {
        ng: _assign_groups_from_ranks(
            mean_ranks, N_keys, ng,
        )
        for ng in n_groups_list
    }
    mean_ranks_f = mean_ranks.astype(float)

    # ── Precompute: per-cluster centroid groups ──
    centroid_groups = {}
    for c in cluster_test_indices:
        cs = (centroids[c] @ K_shared.T) / sqrt_d
        c_order = np.argsort(cs)[::-1]
        c_ranks = _ranks_from_order(c_order, N_keys)
        centroid_groups[c] = {
            ng: _assign_groups_from_ranks(
                c_ranks, N_keys, ng,
            )
            for ng in n_groups_list
        }

    # ── Precompute: persistence (first/last group) ──
    mean_groups_p = _assign_groups_from_ranks(
        mean_ranks, N_keys, persistence_n_groups,
    )
    first_keys = np.where(mean_groups_p == 0)[0]
    last_keys = np.where(
        mean_groups_p == persistence_n_groups - 1,
    )[0]
    first_pers = np.zeros(len(first_keys))
    last_pers = np.zeros(len(last_keys))

    # ── Precompute: agreement thresholds ──
    thresholds = {}
    for k in top_k_values:
        thresholds[f"top {k}"] = min(k, N_keys)
    for pct in top_pct_values:
        k = max(1, int(N_keys * pct / 100))
        thresholds[f"top {pct}%"] = min(k, N_keys)

    inclusion_freq = {
        name: np.zeros(N_keys) for name in thresholds
    }
    top_k_by_query = {
        name: [] for name in thresholds
    }

    # ── Accumulators ──
    global_overlaps = {ng: [] for ng in n_groups_list}
    cluster_overlaps = {
        ng: defaultdict(list) for ng in n_groups_list
    }
    global_positions = []
    spearman_rhos = []

    # ═══ SINGLE PASS ═══
    print(f"        Processing {n_test} queries...")
    for i, qpos in enumerate(test_positions):
        scores = (Q[qpos] @ K_shared.T) / sqrt_d
        order = np.argsort(scores)[::-1]
        ranks = _ranks_from_order(order, N_keys)

        c = int(test_labels[i])

        # ── Stability: global ──
        for ng in n_groups_list:
            qg = _assign_groups_from_ranks(
                ranks, N_keys, ng,
            )
            global_overlaps[ng].append(
                _overlap(mean_groups[ng], qg),
            )
            # ── Stability: clustered ──
            cluster_overlaps[ng][c].append(
                _overlap(centroid_groups[c][ng], qg),
            )

        global_positions.append(qpos)

        # Spearman
        q_ranks_f = ranks.astype(float)
        rho = np.corrcoef(mean_ranks_f, q_ranks_f)[0, 1]
        spearman_rhos.append(rho)

        # ── Persistence ──
        qg_p = _assign_groups_from_ranks(
            ranks, N_keys, persistence_n_groups,
        )
        first_pers += (qg_p[first_keys] == 0)
        last_pers += (
            qg_p[last_keys] == persistence_n_groups - 1
        )

        # ── Agreement: top-K from sorted order ──
        for name, k in thresholds.items():
            top_idx = order[:k].copy()
            top_k_by_query[name].append(top_idx)
            inclusion_freq[name][top_idx] += 1

        if (i + 1) % 5000 == 0:
            print(f"          {i+1}/{n_test}")

    # ═══ Post-process ═══

    # ── Global stability ──
    global_data = {
        "overlaps": {
            ng: np.array(v)
            for ng, v in global_overlaps.items()
        },
        "positions": np.array(global_positions),
        "n_groups_list": list(n_groups_list),
        "spearman_rho": np.array(spearman_rhos),
    }

    # ── Clustered stability ──
    per_cluster_means = {ng: [] for ng in n_groups_list}
    for ng in n_groups_list:
        for c, ovs in cluster_overlaps[ng].items():
            if len(ovs) > 0:
                per_cluster_means[ng].append(np.mean(ovs))

    all_clustered_overlaps = {ng: [] for ng in n_groups_list}
    for ng in n_groups_list:
        for ovs in cluster_overlaps[ng].values():
            all_clustered_overlaps[ng].extend(ovs)

    n_active = len(cluster_test_indices)
    clustered_data = {
        "per_cluster_means": {
            ng: np.array(v)
            for ng, v in per_cluster_means.items()
        },
        "median_overlaps": {
            ng: float(np.median(per_cluster_means[ng]))
            if per_cluster_means[ng] else 0.0
            for ng in n_groups_list
        },
        "all_overlaps": {
            ng: np.array(v)
            for ng, v in all_clustered_overlaps.items()
        },
        "n_clusters": n_clusters,
        "n_active_clusters": n_active,
        "n_groups_list": list(n_groups_list),
        "cluster_sizes": {
            c: len(idxs)
            for c, idxs in cluster_test_indices.items()
        },
    }

    # ── Persistence ──
    first_pers /= n_test
    last_pers /= n_test
    persistence_data = {
        "first_group_persistence": first_pers,
        "last_group_persistence": last_pers,
        "n_first_group": len(first_keys),
        "n_last_group": len(last_keys),
        "n_groups": persistence_n_groups,
        "n_queries": n_test,
    }

    # ── Agreement: within-cluster Jaccard ──
    for name in thresholds:
        inclusion_freq[name] /= n_test

    rng = np.random.default_rng(seed)
    within_pairs = []
    for c, indices in cluster_test_indices.items():
        n_c = len(indices)
        if n_c < 2:
            continue
        n_pairs = min(
            n_c * (n_c - 1) // 2,
            max_pairs_per_cluster,
        )
        for _ in range(n_pairs):
            ii, jj = rng.choice(n_c, 2, replace=False)
            within_pairs.append(
                (indices[ii], indices[jj]),
            )

    print(f"        Agreement: {len(within_pairs)} "
          f"within-cluster pairs...")
    jaccards = {}
    for name, k in thresholds.items():
        j_vals = np.empty(len(within_pairs))
        for idx, (qi, qj) in enumerate(within_pairs):
            a = set(top_k_by_query[name][qi].tolist())
            b = set(top_k_by_query[name][qj].tolist())
            inter = len(a & b)
            union = len(a | b)
            j_vals[idx] = inter / union if union > 0 else 0
        jaccards[name] = j_vals

    agreement_data = {
        "jaccards": jaccards,
        "inclusion_freq": inclusion_freq,
        "thresholds": thresholds,
        "n_queries": n_test,
        "n_keys": N_keys,
        "n_pairs": len(within_pairs),
    }

    return (
        global_data, clustered_data,
        persistence_data, agreement_data,
    )


# ═══════════════════════════════════════════════════════
# Plotting — Stability Dashboard
# ═══════════════════════════════════════════════════════

GROUP_COLORS = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
GROUP_LABELS = {2: "2 groups", 4: "4 groups", 8: "8 groups"}


def _panel_boxplot(ax, overlaps, n_groups_list, title):
    data, labels, colors = [], [], []
    for ng in n_groups_list:
        vals = overlaps.get(ng, np.array([]))
        if len(vals) > 0:
            data.append(vals)
            labels.append(GROUP_LABELS[ng])
            colors.append(GROUP_COLORS[ng])
    if not data:
        ax.set_title(title)
        ax.text(
            0.5, 0.5, "No data", ha="center",
            va="center", transform=ax.transAxes,
        )
        return
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True,
        widths=0.6, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for ng in n_groups_list:
        ax.axhline(
            1.0 / ng, color=GROUP_COLORS[ng],
            ls="--", alpha=0.5, lw=1,
        )
    ax.set_ylabel("Overlap fraction")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)


def _panel_position(
    ax, positions, overlaps, n_groups_list, title,
    ema_span=200,
):
    for ng in n_groups_list:
        vals = overlaps.get(ng, np.array([]))
        if len(vals) == 0 or len(positions) == 0:
            continue
        n = min(len(positions), len(vals))
        pos, v = positions[:n], vals[:n]
        order = np.argsort(pos)
        ps, vs = pos[order], v[order]
        span = min(ema_span, max(2, len(vs) // 5))
        smoothed = _ema(vs, span=span)
        ax.plot(
            ps, smoothed, color=GROUP_COLORS[ng],
            lw=2, label=GROUP_LABELS[ng],
        )
    ax.set_xlabel("Query position")
    ax.set_ylabel("Overlap fraction (EMA)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _panel_cluster_hist(ax, clustered_data, n_groups_list):
    """Histogram of per-cluster mean overlaps."""
    for ng in n_groups_list:
        means = clustered_data["per_cluster_means"].get(
            ng, np.array([]),
        )
        if len(means) > 0:
            ax.hist(
                means, bins=30, alpha=0.5,
                color=GROUP_COLORS[ng],
                label=(
                    f"{GROUP_LABELS[ng]} "
                    f"(med={np.median(means):.3f})"
                ),
            )
    ax.set_xlabel("Per-cluster mean overlap")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Distribution of Cluster Overlaps")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _panel_summary(ax, global_data, clustered_data):
    ax.axis("off")
    lines = ["Summary Statistics\n"]
    lines.append("─── Global Mean Query ───")
    ng_list = global_data["n_groups_list"]
    for ng in ng_list:
        vals = global_data["overlaps"].get(ng, np.array([]))
        if len(vals) > 0:
            lines.append(
                f"  {ng}g: {np.mean(vals):.3f} "
                f"± {np.std(vals):.3f}"
                f"  (rand: {1.0/ng:.3f})"
            )
    rho = global_data.get("spearman_rho", np.array([]))
    if len(rho) > 0:
        lines.append(
            f"  Spearman ρ: {np.mean(rho):.3f} "
            f"± {np.std(rho):.3f}"
        )

    nc = clustered_data["n_clusters"]
    na = clustered_data["n_active_clusters"]
    lines.append(
        f"\n─── Clustered ({nc}c, "
        f"{na} active) ───"
    )
    lines.append("  [median of per-cluster means]")
    for ng in ng_list:
        med = clustered_data["median_overlaps"].get(ng, 0)
        means = clustered_data["per_cluster_means"].get(
            ng, np.array([]),
        )
        if len(means) > 0:
            lines.append(
                f"  {ng}g: median={med:.3f}  "
                f"mean={np.mean(means):.3f}"
            )

    lines.append("\n─── Δ (clustered med − global) ───")
    for ng in ng_list:
        gv = global_data["overlaps"].get(ng, np.array([]))
        med = clustered_data["median_overlaps"].get(ng, 0)
        if len(gv) > 0:
            diff = med - np.mean(gv)
            sign = "+" if diff >= 0 else ""
            lines.append(f"  {ng}g: {sign}{diff:.3f}")

    n_q = len(global_data["positions"])
    lines.append(f"\nn_test_queries: {n_q:,}")
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9, fontfamily="monospace",
        va="top", ha="left",
    )


def create_stability_dashboard(
    global_data: Dict,
    clustered_data: Dict,
    title: str,
    out_path,
):
    """2×3 key sorting stability dashboard."""
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Key Sorting Stability — {title}",
        fontsize=14, fontweight="bold",
    )
    ng_list = global_data["n_groups_list"]

    # Row 0: Global Mean
    _panel_boxplot(
        axes[0, 0], global_data["overlaps"], ng_list,
        "Global Mean — Overlap Distribution",
    )
    _panel_position(
        axes[0, 1], global_data["positions"],
        global_data["overlaps"], ng_list,
        "Global Mean — Overlap vs Position",
    )
    _panel_summary(axes[0, 2], global_data, clustered_data)

    # Row 1: Clustered
    _panel_boxplot(
        axes[1, 0], clustered_data["all_overlaps"],
        ng_list,
        f"Clustered ({clustered_data['n_clusters']}c) "
        f"— Within-Cluster Overlap",
    )
    _panel_cluster_hist(axes[1, 1], clustered_data, ng_list)
    # Per-cluster size distribution
    ax = axes[1, 2]
    sizes = list(clustered_data["cluster_sizes"].values())
    ax.hist(
        sizes, bins=30, color="#9467bd",
        alpha=0.7, edgecolor="white",
    )
    ax.set_xlabel("Queries per cluster")
    ax.set_ylabel("Number of clusters")
    ax.set_title(
        f"Cluster Size Distribution "
        f"(median={np.median(sizes):.0f})"
    )
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════
# Plotting — Group Persistence
# ═══════════════════════════════════════════════════════

def create_persistence_dashboard(
    persistence_data: Dict,
    title: str,
    out_path,
):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Group Persistence — {title}",
        fontsize=14, fontweight="bold",
    )

    ng = persistence_data["n_groups"]
    n_q = persistence_data["n_queries"]

    fp = persistence_data["first_group_persistence"]
    ax = axes[0]
    ax.hist(
        fp, bins=40, color="#1f77b4",
        alpha=0.7, edgecolor="white",
    )
    ax.axvline(
        np.mean(fp), color="red", ls="--", lw=2,
        label=f"mean={np.mean(fp):.3f}",
    )
    ax.set_xlabel("Persistence rate")
    ax.set_ylabel("Number of keys")
    ax.set_title(
        f"Close Group (group 0 of {ng}) — "
        f"{len(fp):,} keys",
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    lp = persistence_data["last_group_persistence"]
    ax = axes[1]
    ax.hist(
        lp, bins=40, color="#ff7f0e",
        alpha=0.7, edgecolor="white",
    )
    ax.axvline(
        np.mean(lp), color="red", ls="--", lw=2,
        label=f"mean={np.mean(lp):.3f}",
    )
    ax.set_xlabel("Persistence rate")
    ax.set_ylabel("Number of keys")
    ax.set_title(
        f"Far Group (group {ng-1} of {ng}) — "
        f"{len(lp):,} keys",
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axis("off")
    lines = [
        f"Group Persistence Summary\n",
        f"n_groups: {ng}",
        f"n_queries: {n_q:,}",
        f"",
        f"── Close Group (top 1/{ng}) ──",
        f"  n_keys: {len(fp):,}",
        f"  mean persistence: {np.mean(fp):.3f}",
        f"  median: {np.median(fp):.3f}",
        f"  >90% persistent: "
        f"{np.mean(fp > 0.9):.1%}",
        f"  >50% persistent: "
        f"{np.mean(fp > 0.5):.1%}",
        f"",
        f"── Far Group (bottom 1/{ng}) ──",
        f"  n_keys: {len(lp):,}",
        f"  mean persistence: {np.mean(lp):.3f}",
        f"  median: {np.median(lp):.3f}",
        f"  >90% persistent: "
        f"{np.mean(lp > 0.9):.1%}",
        f"  >50% persistent: "
        f"{np.mean(lp > 0.5):.1%}",
    ]
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10, fontfamily="monospace",
        va="top", ha="left",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_path, dpi=150)


# ═══════════════════════════════════════════════════════
# Plotting — Query Agreement
# ═══════════════════════════════════════════════════════

def create_agreement_dashboard(
    agreement_data: Dict,
    title: str,
    out_path,
):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Query Agreement (within-cluster) — {title}",
        fontsize=14, fontweight="bold",
    )

    jaccards = agreement_data["jaccards"]
    inc_freq = agreement_data["inclusion_freq"]
    thresholds = agreement_data["thresholds"]
    n_q = agreement_data["n_queries"]
    n_k = agreement_data["n_keys"]
    n_pairs = agreement_data["n_pairs"]

    # Jaccard box plots
    ax = axes[0]
    names = list(jaccards.keys())
    data = [jaccards[n] for n in names]
    bp = ax.boxplot(
        data, labels=names, patch_artist=True,
        widths=0.6, showfliers=False,
    )
    cmap = plt.cm.viridis(
        np.linspace(0.2, 0.8, len(names)),
    )
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(
        f"Within-Cluster Agreement "
        f"({n_pairs:,} pairs)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # Inclusion frequency — top 1%
    pct_name = "top 1%"
    if pct_name in inc_freq:
        ax = axes[1]
        freq = inc_freq[pct_name]
        ax.hist(
            freq, bins=50, color="#2ca02c",
            alpha=0.7, edgecolor="white",
        )
        ax.axvline(
            np.mean(freq), color="red", ls="--", lw=2,
            label=f"mean={np.mean(freq):.3f}",
        )
        ax.set_xlabel(
            "Fraction of queries including this key",
        )
        ax.set_ylabel("Number of keys")
        ax.set_title(f"Per-Key Inclusion ({pct_name})")
        ax.set_xlim(
            0, max(0.1, np.percentile(freq, 99.5)),
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    else:
        axes[1].axis("off")

    # Summary
    ax = axes[2]
    ax.axis("off")
    lines = [
        f"Query Agreement Summary\n",
        f"n_queries: {n_q:,}",
        f"n_shared_keys: {n_k:,}",
        f"within-cluster pairs: {n_pairs:,}",
        f"",
    ]
    for name in names:
        j = jaccards[name]
        k_val = thresholds[name]
        lines.append(f"── {name} (k={k_val:,}) ──")
        lines.append(
            f"  Jaccard: {np.mean(j):.3f} "
            f"± {np.std(j):.3f}"
        )
        lines.append(f"  median: {np.median(j):.3f}")
        if name in inc_freq:
            f = inc_freq[name]
            n_always = np.sum(f > 0.9)
            lines.append(
                f"  keys >90% included: {n_always:,}"
            )
        lines.append("")

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9, fontfamily="monospace",
        va="top", ha="left",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_path)


# ═══════════════════════════════════════════════════════
# Compute — PCA Subspace Projection
# ═══════════════════════════════════════════════════════

def _pca_descending(X: np.ndarray):
    """PCA on rows of X. Returns (eigenvalues, eigenvectors,
    cumulative_variance) all in descending-eigenvalue order."""
    mean = X.mean(axis=0)
    delta = X - mean
    cov = (delta.T @ delta) / len(X)
    evals, evecs = np.linalg.eigh(cov)
    # eigh returns ascending → flip
    evals = evals[::-1].copy()
    evecs = evecs[:, ::-1].copy()
    total = max(evals.sum(), 1e-10)
    cum_var = np.cumsum(evals) / total
    return evals, evecs, cum_var


def compute_pca_projection_stability(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    n_test_queries: int = 25000,
    pca_dims: Tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    top_pct_values: Tuple[float, ...] = (0.1, 1, 5, 10),
    seed: int = 42,
) -> Dict:
    """
    PCA subspace projection — two methods compared.

    Decomposes score(q, k) = q·mean_k/√d + q·δk/√d
    where δk = k − mean_k.  The first term is full-dim
    and shared across all keys (doesn't affect ranking).
    The second term is approximated by projecting δk
    onto a low-rank subspace V:

      q·δk ≈ q·(VV^T δk) = (V^T q)·(V^T δk)

    Query stays full-dimensional; only keys are reduced.

    Two choices of V:
      query_pca — V from PCA of ALL queries
      key_pca   — V from PCA of ALL keys (excl. sink)

    PCA computed on full sequences.  Testing uses last
    n_test_queries and shared keys K[1:test_start].
    """
    sqrt_d = np.sqrt(head_dim)
    seq_len = Q.shape[0]

    n_test = min(n_test_queries, seq_len - 2)
    test_start = seq_len - n_test
    K_shared = K[1:test_start]
    N_keys = len(K_shared)

    print(f"        PCA: {N_keys:,} shared keys, "
          f"{n_test:,} test queries")

    # ── PCA on ALL queries ──
    print("        PCA on all queries...")
    evals_q, evecs_q, cum_var_q = _pca_descending(Q)

    # ── PCA on ALL keys (exclude sink at pos 0) ──
    print("        PCA on all keys...")
    K_no_sink = K[1:]
    evals_k, evecs_k, cum_var_k = _pca_descending(
        K_no_sink,
    )

    # ── Mean and deviation of shared keys ──
    mean_k = K_shared.mean(axis=0)
    delta_K = K_shared - mean_k  # (N_keys, head_dim)

    # Precompute δK projections for each PCA dim
    dK_proj_q = {}  # δK projected onto query PCA
    dK_proj_k = {}  # δK projected onto key PCA
    for d in pca_dims:
        d_eff = min(d, head_dim)
        dK_proj_q[d] = delta_K @ evecs_q[:, :d_eff]
        dK_proj_k[d] = delta_K @ evecs_k[:, :d_eff]

    # Thresholds: number of keys for each percentage
    thresholds = {}
    for pct in top_pct_values:
        thresholds[pct] = max(1, int(N_keys * pct / 100))

    print(
        "        Thresholds: "
        + ", ".join(
            f"{p}%={thresholds[p]:,}"
            for p in top_pct_values
        )
    )

    methods = ("query_pca", "key_pca")

    # Accumulators  [method][dim][pct] = array(n_test)
    nearest_recall = {
        m: {
            d: {p: np.empty(n_test) for p in top_pct_values}
            for d in pca_dims
        }
        for m in methods
    }
    farthest_recall = {
        m: {
            d: {p: np.empty(n_test) for p in top_pct_values}
            for d in pca_dims
        }
        for m in methods
    }
    spearman_rho = {
        m: {d: np.empty(n_test) for d in pca_dims}
        for m in methods
    }

    print(f"        Processing {n_test:,} queries "
          f"× {len(pca_dims)} dims × 2 methods...")

    for qi in range(n_test):
        qpos = test_start + qi
        q = Q[qpos]

        # Full-dim scores and ranking (once per query)
        full_scores = (q @ K_shared.T) / sqrt_d
        full_order = np.argsort(full_scores)[::-1]
        full_ranks = _ranks_from_order(
            full_order, N_keys,
        )
        full_ranks_f = full_ranks.astype(float)

        # True top/bottom sets (sorted for intersect1d)
        true_tops = {}
        true_bottoms = {}
        for pct, k in thresholds.items():
            true_tops[pct] = np.sort(full_order[:k])
            true_bottoms[pct] = np.sort(full_order[-k:])

        # Base score: q·mean_k (constant across keys)
        base = (q @ mean_k) / sqrt_d

        for d in pca_dims:
            d_eff = min(d, head_dim)

            for mname, evecs, dK_proj in [
                ("query_pca", evecs_q, dK_proj_q),
                ("key_pca", evecs_k, dK_proj_k),
            ]:
                # (V^T q) · (V^T δk) for all shared keys
                q_proj = q @ evecs[:, :d_eff]  # (d_eff,)
                approx_scores = (
                    base + (q_proj @ dK_proj[d].T) / sqrt_d
                )

                a_order = np.argsort(
                    approx_scores,
                )[::-1]
                a_ranks = _ranks_from_order(
                    a_order, N_keys,
                )

                for pct, k in thresholds.items():
                    a_top = np.sort(a_order[:k])
                    a_bot = np.sort(a_order[-k:])
                    nearest_recall[mname][d][pct][qi] = (
                        len(np.intersect1d(
                            true_tops[pct], a_top,
                            assume_unique=True,
                        )) / k
                    )
                    farthest_recall[mname][d][pct][qi] = (
                        len(np.intersect1d(
                            true_bottoms[pct], a_bot,
                            assume_unique=True,
                        )) / k
                    )

                rho = np.corrcoef(
                    full_ranks_f,
                    a_ranks.astype(float),
                )[0, 1]
                spearman_rho[mname][d][qi] = rho

        if (qi + 1) % 5000 == 0:
            print(f"          {qi+1}/{n_test}")

    return {
        "methods": list(methods),
        "nearest_recall": nearest_recall,
        "farthest_recall": farthest_recall,
        "spearman": spearman_rho,
        "pca_dims": list(pca_dims),
        "top_pct_values": list(top_pct_values),
        "thresholds": thresholds,
        "eigenvalues_q": evals_q,
        "eigenvalues_k": evals_k,
        "cum_var_q": cum_var_q,
        "cum_var_k": cum_var_k,
        "n_test": n_test,
        "n_keys": N_keys,
        "head_dim": head_dim,
    }


# ═══════════════════════════════════════════════════════
# Plotting — PCA Subspace
# ═══════════════════════════════════════════════════════

_PCT_COLORS = {
    0.1: "#e41a1c",
    1: "#377eb8",
    5: "#4daf4a",
    10: "#984ea3",
}

_METHOD_COLORS = {
    "query_pca": "#1f77b4",
    "key_pca": "#ff7f0e",
}

_METHOD_LABELS = {
    "query_pca": "Query-PCA",
    "key_pca": "Key-PCA",
}


def _grouped_recall_boxplot(
    ax, data_dict, dims, pct_values, title,
):
    """
    Grouped box plot: one group per PCA dim, one box
    per percentile threshold.

    Box: Q25–Q75, whiskers: P10–P90, diamond: mean,
    line: median, no outliers.
    """
    n_pct = len(pct_values)
    group_width = 0.75
    box_width = group_width / n_pct * 0.88

    positions = []
    all_data = []
    all_colors = []

    for gi, d in enumerate(dims):
        center = gi + 1
        offsets = np.linspace(
            -group_width / 2 + box_width / 2,
            group_width / 2 - box_width / 2,
            n_pct,
        )
        for bi, pct in enumerate(pct_values):
            positions.append(center + offsets[bi])
            all_data.append(data_dict[d][pct])
            all_colors.append(
                _PCT_COLORS.get(pct, "#333333"),
            )

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="black",
            markeredgecolor="black", markersize=3,
        ),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        whis=[10, 90],
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks(range(1, len(dims) + 1))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.3)

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            fc=_PCT_COLORS.get(p, "#333"), alpha=0.65,
        )
        for p in pct_values
    ]
    ax.legend(
        handles, [f"top {p}%" for p in pct_values],
        loc="lower right", fontsize=8,
        title="Threshold", title_fontsize=8,
    )


def _stats_text(arr):
    """One-line stats: mean, median, P10, P90."""
    return (
        f"μ={np.mean(arr):.3f}  "
        f"med={np.median(arr):.3f}  "
        f"P10={np.percentile(arr, 10):.3f}  "
        f"P90={np.percentile(arr, 90):.3f}"
    )


def _var_dims_needed(cum_var, target):
    """Dims needed to reach target cumulative variance."""
    idx = np.searchsorted(cum_var, target)
    return idx + 1 if idx < len(cum_var) else f">{len(cum_var)}"


def create_pca_dashboard(
    pca_data: Dict,
    title: str,
    out_path,
):
    """
    PCA projection dashboard — two methods side by side.

    Layout 3×2:
      [0,0] Nearest — Query-PCA   [0,1] Nearest — Key-PCA
      [1,0] Farthest — Query-PCA  [1,1] Farthest — Key-PCA
      [2,0] Spearman (both) + var [2,1] Summary statistics
    """
    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(22, 20))
    fig.suptitle(
        f"PCA Key-Subspace Projection — {title}",
        fontsize=14, fontweight="bold",
    )

    dims = pca_data["pca_dims"]
    pct_vals = pca_data["top_pct_values"]
    methods = pca_data["methods"]
    nearest = pca_data["nearest_recall"]
    farthest = pca_data["farthest_recall"]
    spearman = pca_data["spearman"]
    cum_var_q = pca_data["cum_var_q"]
    cum_var_k = pca_data["cum_var_k"]
    head_dim = pca_data["head_dim"]
    thresholds = pca_data["thresholds"]

    # ── Row 0: Nearest recall ──
    _grouped_recall_boxplot(
        axes[0, 0], nearest["query_pca"], dims, pct_vals,
        "Nearest (Top-K) — Keys → Query-PCA",
    )
    _grouped_recall_boxplot(
        axes[0, 1], nearest["key_pca"], dims, pct_vals,
        "Nearest (Top-K) — Keys → Key-PCA",
    )

    # ── Row 1: Farthest recall ──
    _grouped_recall_boxplot(
        axes[1, 0], farthest["query_pca"], dims, pct_vals,
        "Farthest (Bottom-K) — Keys → Query-PCA",
    )
    _grouped_recall_boxplot(
        axes[1, 1], farthest["key_pca"], dims, pct_vals,
        "Farthest (Bottom-K) — Keys → Key-PCA",
    )

    # ── [2,0] Spearman (both methods) + variance ──
    ax = axes[2, 0]
    ax2 = ax.twinx()

    n_m = len(methods)
    grp_w = 0.6
    bw = grp_w / n_m * 0.85
    positions_all = []
    data_all = []
    colors_all = []
    for gi, d in enumerate(dims):
        center = gi + 1
        offsets = np.linspace(
            -grp_w / 2 + bw / 2,
            grp_w / 2 - bw / 2,
            n_m,
        )
        for mi, m in enumerate(methods):
            positions_all.append(center + offsets[mi])
            data_all.append(spearman[m][d])
            colors_all.append(_METHOD_COLORS[m])

    bp = ax.boxplot(
        data_all,
        positions=positions_all,
        widths=bw,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="black",
            markeredgecolor="black", markersize=3,
        ),
        medianprops=dict(color="black", linewidth=1.5),
        whis=[10, 90],
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Cumulative variance curves on twin axis
    xs = np.arange(1, len(dims) + 1)
    var_q = [
        cum_var_q[min(d, len(cum_var_q)) - 1]
        for d in dims
    ]
    var_k = [
        cum_var_k[min(d, len(cum_var_k)) - 1]
        for d in dims
    ]
    ax2.plot(
        xs, var_q, "s--",
        color=_METHOD_COLORS["query_pca"],
        lw=1.5, markersize=5, alpha=0.7,
    )
    ax2.plot(
        xs, var_k, "^--",
        color=_METHOD_COLORS["key_pca"],
        lw=1.5, markersize=5, alpha=0.7,
    )
    for i, (vq, vk) in enumerate(zip(var_q, var_k)):
        ax2.annotate(
            f"{vq:.2f}", (xs[i] - 0.15, vq),
            fontsize=6, color=_METHOD_COLORS["query_pca"],
            ha="right",
        )
        ax2.annotate(
            f"{vk:.2f}", (xs[i] + 0.15, vk),
            fontsize=6, color=_METHOD_COLORS["key_pca"],
            ha="left",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("Spearman ρ")
    ax2.set_ylabel("Cumulative variance")
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax.set_title("Rank Correlation & Variance Explained")
    ax.grid(axis="y", alpha=0.3)

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            fc=_METHOD_COLORS[m], alpha=0.6,
        )
        for m in methods
    ]
    ax.legend(
        handles,
        [f"{_METHOD_LABELS[m]} (ρ + var)"
         for m in methods],
        loc="lower right", fontsize=8,
    )

    # ── [2,1] Summary statistics ──
    ax = axes[2, 1]
    ax.axis("off")
    lines = [
        "Summary Statistics\n",
        f"head_dim: {head_dim}",
        f"n_shared_keys: {pca_data['n_keys']:,}",
        f"n_test_queries: {pca_data['n_test']:,}",
        "",
    ]

    for m in methods:
        label = _METHOD_LABELS[m]
        lines.append(f"═══ {label} ═══")

        lines.append("  Nearest (mean recall):")
        for d in dims:
            parts = [
                f"{p}%={np.mean(nearest[m][d][p]):.3f}"
                for p in pct_vals
            ]
            lines.append(
                f"    d={d:3d}: {' '.join(parts)}"
            )

        lines.append("  Farthest (mean recall):")
        for d in dims:
            parts = [
                f"{p}%={np.mean(farthest[m][d][p]):.3f}"
                for p in pct_vals
            ]
            lines.append(
                f"    d={d:3d}: {' '.join(parts)}"
            )

        lines.append("  Spearman ρ:")
        for d in dims:
            lines.append(
                f"    d={d:3d}: "
                f"{_stats_text(spearman[m][d])}"
            )

        cv = (
            cum_var_q if m == "query_pca" else cum_var_k
        )
        lines.append("  Variance dims:")
        for t, lb in [
            (0.90, "90%"), (0.95, "95%"), (0.99, "99%"),
        ]:
            lines.append(
                f"    {lb}: {_var_dims_needed(cv, t)}"
            )
        lines.append("")

    lines.append("─── Thresholds (# keys) ───")
    for pct in pct_vals:
        lines.append(
            f"  {pct}%: {thresholds[pct]:,} keys"
        )

    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7, fontfamily="monospace",
        va="top", ha="left",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_path)
