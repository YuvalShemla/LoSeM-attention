#!/usr/bin/env python3
"""
QKV Clustering Analysis — compares clustering structure
of Queries, Keys, and Values in 2D projections.

Extends the existing Q+K embedding projections to include V,
and adds comprehensive clustering statistics:
  - Intra-cluster compactness (mean pairwise distance within Q/K/V)
  - Inter-cluster separation (mean pairwise distance between Q-K, Q-V, K-V)
  - Separation ratios
  - Cluster size statistics (after K-Means)
  - Silhouette scores
  - Centroid distances

Produces a multi-panel dashboard comparing Q, K, V geometry.

Usage:
  python -m src.exploration.qkv_clustering \
    --tasks math_calc code_run

  python -m src.exploration.qkv_clustering --all
"""

import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..evaluation.data_loader import (
    load_examples, load_task_metadata,
)
from ..evaluation.plotting import setup_style, save_figure


# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

_POS_WINDOW = 1024

_TYPE_COLORS = {
    "Q": "#1976d2",
    "K": "#7b1fa2",
    "V": "#2e7d32",
}
_TYPE_COLORS_LIGHT = {
    "Q": "#bbdefb",
    "K": "#e1bee7",
    "V": "#c8e6c9",
}

_POS_GROUP_STYLE = {
    "Q_mid":   {"c": "#bbdefb", "s": 2, "alpha": 0.35},
    "K_mid":   {"c": "#e1bee7", "s": 2, "alpha": 0.35},
    "V_mid":   {"c": "#c8e6c9", "s": 2, "alpha": 0.35},
    "Q_first": {"c": "#00acc1", "s": 5, "alpha": 0.5},
    "Q_last":  {"c": "#1565c0", "s": 5, "alpha": 0.5},
    "K_first": {"c": "#ec407a", "s": 5, "alpha": 0.5},
    "K_last":  {"c": "#7b1fa2", "s": 5, "alpha": 0.5},
    "V_first": {"c": "#66bb6a", "s": 5, "alpha": 0.5},
    "V_last":  {"c": "#1b5e20", "s": 5, "alpha": 0.5},
}
_POS_GROUP_LABELS = {
    "Q_first": "Q first 1K",
    "Q_mid":   "Q middle",
    "Q_last":  "Q last 1K",
    "K_first": "K first 1K",
    "K_mid":   "K middle",
    "K_last":  "K last 1K",
    "V_first": "V first 1K",
    "V_mid":   "V middle",
    "V_last":  "V last 1K",
    "sink":    "sink",
}


# ═══════════════════════════════════════════════════════
# Clustering statistics
# ═══════════════════════════════════════════════════════

def _pairwise_distances_sampled(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    n_pairs: int = 5000,
    rng=None,
) -> np.ndarray:
    """Sample pairwise L2 distances between two sets."""
    if rng is None:
        rng = np.random.default_rng(42)
    na, nb = len(vecs_a), len(vecs_b)
    idx_a = rng.integers(0, na, size=n_pairs)
    idx_b = rng.integers(0, nb, size=n_pairs)
    if vecs_a is vecs_b:
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
    return np.linalg.norm(
        vecs_a[idx_a] - vecs_b[idx_b], axis=1,
    )


def _cosine_distances_sampled(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    n_pairs: int = 5000,
    rng=None,
) -> np.ndarray:
    """Sample pairwise cosine distances between two sets."""
    if rng is None:
        rng = np.random.default_rng(42)
    na, nb = len(vecs_a), len(vecs_b)
    idx_a = rng.integers(0, na, size=n_pairs)
    idx_b = rng.integers(0, nb, size=n_pairs)
    if vecs_a is vecs_b:
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
    a = vecs_a[idx_a]
    b = vecs_b[idx_b]
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos_sim = dots / np.maximum(norms, 1e-10)
    return 1.0 - cos_sim


def compute_qkv_cluster_stats(
    q_vecs: np.ndarray,
    k_vecs: np.ndarray,
    v_vecs: np.ndarray,
    n_pairs: int = 5000,
) -> Dict:
    """
    Comprehensive clustering statistics for Q, K, V.

    Returns metrics in both L2 and cosine distance.
    """
    rng = np.random.default_rng(42)

    results = {}
    for metric_name, dist_fn in [
        ("l2", _pairwise_distances_sampled),
        ("cosine", _cosine_distances_sampled),
    ]:
        # Intra-cluster distances
        intra_q = dist_fn(q_vecs, q_vecs, n_pairs, rng)
        intra_k = dist_fn(k_vecs, k_vecs, n_pairs, rng)
        intra_v = dist_fn(v_vecs, v_vecs, n_pairs, rng)

        # Inter-cluster distances
        inter_qk = dist_fn(q_vecs, k_vecs, n_pairs, rng)
        inter_qv = dist_fn(q_vecs, v_vecs, n_pairs, rng)
        inter_kv = dist_fn(k_vecs, v_vecs, n_pairs, rng)

        # Centroids
        q_cent = q_vecs.mean(axis=0)
        k_cent = k_vecs.mean(axis=0)
        v_cent = v_vecs.mean(axis=0)

        if metric_name == "l2":
            cent_qk = float(np.linalg.norm(q_cent - k_cent))
            cent_qv = float(np.linalg.norm(q_cent - v_cent))
            cent_kv = float(np.linalg.norm(k_cent - v_cent))
        else:
            def _cos_dist(a, b):
                d = np.dot(a, b)
                n = np.linalg.norm(a) * np.linalg.norm(b)
                return float(1.0 - d / max(n, 1e-10))
            cent_qk = _cos_dist(q_cent, k_cent)
            cent_qv = _cos_dist(q_cent, v_cent)
            cent_kv = _cos_dist(k_cent, v_cent)

        results[metric_name] = {
            "intra_q_mean": float(np.mean(intra_q)),
            "intra_q_std": float(np.std(intra_q)),
            "intra_k_mean": float(np.mean(intra_k)),
            "intra_k_std": float(np.std(intra_k)),
            "intra_v_mean": float(np.mean(intra_v)),
            "intra_v_std": float(np.std(intra_v)),
            "inter_qk_mean": float(np.mean(inter_qk)),
            "inter_qv_mean": float(np.mean(inter_qv)),
            "inter_kv_mean": float(np.mean(inter_kv)),
            "centroid_qk": cent_qk,
            "centroid_qv": cent_qv,
            "centroid_kv": cent_kv,
            "separation_qk": float(
                np.mean(inter_qk) / max(
                    (np.mean(intra_q) + np.mean(intra_k)) / 2,
                    1e-10,
                )
            ),
            "separation_qv": float(
                np.mean(inter_qv) / max(
                    (np.mean(intra_q) + np.mean(intra_v)) / 2,
                    1e-10,
                )
            ),
            "separation_kv": float(
                np.mean(inter_kv) / max(
                    (np.mean(intra_k) + np.mean(intra_v)) / 2,
                    1e-10,
                )
            ),
            # Raw distance arrays for histograms
            "intra_q_dists": intra_q,
            "intra_k_dists": intra_k,
            "intra_v_dists": intra_v,
            "inter_qk_dists": inter_qk,
            "inter_qv_dists": inter_qv,
            "inter_kv_dists": inter_kv,
        }

    # Norm statistics
    q_norms = np.linalg.norm(q_vecs, axis=1)
    k_norms = np.linalg.norm(k_vecs, axis=1)
    v_norms = np.linalg.norm(v_vecs, axis=1)
    results["norms"] = {
        "q_mean": float(np.mean(q_norms)),
        "q_std": float(np.std(q_norms)),
        "k_mean": float(np.mean(k_norms)),
        "k_std": float(np.std(k_norms)),
        "v_mean": float(np.mean(v_norms)),
        "v_std": float(np.std(v_norms)),
        "q_norms": q_norms,
        "k_norms": k_norms,
        "v_norms": v_norms,
    }

    return results


def compute_kmeans_analysis(
    vecs: np.ndarray,
    label: str,
    n_clusters: int = 128,
    max_samples: int = 50000,
) -> Dict:
    """
    Run K-Means on a vector set and return cluster statistics.

    Returns:
      - cluster_sizes: sorted array of cluster sizes
      - largest_cluster, smallest_cluster
      - size_ratio: largest / smallest
      - intra_cluster_dists: mean pairwise distance within each cluster
      - silhouette: silhouette score (sampled)
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(42)
    if len(vecs) > max_samples:
        idx = rng.choice(len(vecs), max_samples, replace=False)
        vecs_fit = vecs[idx]
    else:
        vecs_fit = vecs

    n_clusters = min(n_clusters, len(vecs_fit) // 2)
    if n_clusters < 2:
        return {"label": label, "error": "too few vectors"}

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(1024, len(vecs_fit)),
        n_init=3,
    )
    cluster_labels = km.fit_predict(vecs_fit)
    centers = km.cluster_centers_

    # Cluster sizes
    sizes = np.bincount(cluster_labels, minlength=n_clusters)
    sorted_sizes = np.sort(sizes)[::-1]

    # Intra-cluster mean distance to centroid
    intra_dists = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 1:
            diffs = vecs_fit[mask] - centers[c]
            intra_dists[c] = float(np.mean(
                np.linalg.norm(diffs, axis=1)
            ))

    # Inter-cluster centroid distances
    n_c = len(centers)
    inter_centroid = np.zeros((n_c, n_c))
    for i in range(n_c):
        for j in range(i + 1, n_c):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            inter_centroid[i, j] = d
            inter_centroid[j, i] = d

    # Silhouette (subsample for speed)
    sil_n = min(5000, len(vecs_fit))
    sil_idx = rng.choice(len(vecs_fit), sil_n, replace=False)
    sil = float(silhouette_score(
        vecs_fit[sil_idx], cluster_labels[sil_idx],
    ))

    return {
        "label": label,
        "n_clusters": n_clusters,
        "cluster_sizes": sorted_sizes,
        "largest_cluster": int(sorted_sizes[0]),
        "smallest_cluster": int(sorted_sizes[-1]),
        "size_ratio": float(
            sorted_sizes[0] / max(sorted_sizes[-1], 1)
        ),
        "mean_intra_dist": float(np.mean(intra_dists)),
        "max_intra_dist": float(np.max(intra_dists)),
        "min_intra_dist": float(np.min(intra_dists)),
        "intra_dists_per_cluster": intra_dists,
        "mean_inter_centroid": float(
            inter_centroid[inter_centroid > 0].mean()
        ),
        "min_inter_centroid": float(
            inter_centroid[inter_centroid > 0].min()
        ),
        "compactness_ratio": float(
            np.mean(intra_dists) / max(
                inter_centroid[inter_centroid > 0].mean(),
                1e-10,
            )
        ),
        "silhouette": sil,
        "centers": centers,
    }


# ═══════════════════════════════════════════════════════
# 2D projections for Q, K, V
# ═══════════════════════════════════════════════════════

def _assign_qkv_position_groups(
    positions: np.ndarray,
    labels: np.ndarray,
    is_sink: np.ndarray,
    seq_len: int,
):
    """Assign position groups for Q, K, V coloring."""
    groups = np.empty(len(positions), dtype="U10")
    first_end = _POS_WINDOW
    last_start = seq_len - _POS_WINDOW

    for qkv in ["Q", "K", "V"]:
        type_mask = labels == qkv
        first = type_mask & (positions < first_end)
        last = type_mask & (positions >= last_start)
        middle = type_mask & ~first & ~last

        groups[first] = f"{qkv}_first"
        groups[middle] = f"{qkv}_mid"
        groups[last] = f"{qkv}_last"

    groups[is_sink] = "sink"
    return groups


def compute_qkv_projections(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    query_positions: List[int],
    config: dict,
) -> Dict:
    """
    Compute PCA, t-SNE, UMAP projections of Q+K+V jointly.

    Similar to compute_embedding_projections but includes V.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    emb_cfg = config.get("embedding", {})
    n_fit = emb_cfg.get("n_fit", 50000)
    n_plot = emb_cfg.get("n_plot", 15000)
    perplexity = emb_cfg.get("tsne_perplexity", 100.0)
    tsne_max_iter = emb_cfg.get("tsne_max_iter", 1500)
    umap_n_neighbors = emb_cfg.get("umap_n_neighbors", 50)
    umap_min_dist = emb_cfg.get("umap_min_dist", 0.05)
    umap_metric = emb_cfg.get("umap_metric", "cosine")

    qpos_arr = np.array(query_positions)
    max_qpos = int(qpos_arr.max())
    seq_len = max_qpos + 1

    q_vecs = Q[qpos_arr]
    k_vecs = K[:seq_len]
    v_vecs = V[:seq_len]

    # Subsample for fitting — split budget 3 ways
    rng = np.random.default_rng(42)
    third_fit = n_fit // 3

    q_n = min(third_fit, len(q_vecs))
    k_n = min(third_fit, len(k_vecs))
    v_n = min(third_fit, len(v_vecs))

    q_idx = rng.choice(len(q_vecs), q_n, replace=False)
    k_idx = rng.choice(len(k_vecs), k_n, replace=False)
    v_idx = rng.choice(len(v_vecs), v_n, replace=False)

    # Ensure sink (K[0]) is included
    if 0 not in k_idx and len(k_vecs) > 0:
        k_idx[0] = 0

    fit_vecs = np.vstack([
        q_vecs[q_idx], k_vecs[k_idx], v_vecs[v_idx],
    ])
    fit_labels = np.concatenate([
        np.full(len(q_idx), "Q"),
        np.full(len(k_idx), "K"),
        np.full(len(v_idx), "V"),
    ])
    fit_is_sink = np.zeros(len(fit_vecs), dtype=bool)
    # Mark K[0] as sink
    sink_pos = np.where(
        (fit_labels == "K")
        & (np.concatenate([
            qpos_arr[q_idx],
            np.arange(seq_len)[k_idx],
            np.arange(seq_len)[v_idx],
        ]) == 0)
    )[0]
    fit_is_sink[sink_pos] = True

    fit_positions = np.concatenate([
        qpos_arr[q_idx],
        np.arange(seq_len)[k_idx],
        np.arange(seq_len)[v_idx],
    ])

    fit_pos_groups = _assign_qkv_position_groups(
        fit_positions, fit_labels,
        fit_is_sink, seq_len,
    )

    # Subsample for plotting
    def _subsample_plot(coords):
        if len(coords) <= n_plot:
            return (
                coords, fit_labels, fit_is_sink,
                fit_positions, fit_pos_groups,
            )
        sink_idx = np.where(fit_is_sink)[0]
        non_sink_idx = np.where(~fit_is_sink)[0]
        n_non = n_plot - len(sink_idx)
        chosen = rng.choice(
            non_sink_idx,
            min(n_non, len(non_sink_idx)),
            replace=False,
        )
        idx = np.concatenate([sink_idx, chosen])
        return (
            coords[idx], fit_labels[idx],
            fit_is_sink[idx], fit_positions[idx],
            fit_pos_groups[idx],
        )

    # PCA
    pca = PCA(n_components=2)
    pca_all = pca.fit_transform(fit_vecs)
    pca_explained = pca.explained_variance_ratio_
    (pca_coords, pca_labels, pca_is_sink,
     pca_positions, pca_pos_groups) = _subsample_plot(pca_all)

    # t-SNE
    effective_perp = min(perplexity, len(fit_vecs) / 4)
    tsne = TSNE(
        n_components=2,
        perplexity=max(5.0, effective_perp),
        max_iter=tsne_max_iter,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    tsne_all = tsne.fit_transform(fit_vecs)
    (tsne_coords, tsne_labels, tsne_is_sink,
     tsne_positions, tsne_pos_groups) = _subsample_plot(tsne_all)

    # UMAP
    umap_result = {}
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=42,
            n_jobs=1,
        )
        umap_all = reducer.fit_transform(fit_vecs)
        (umap_coords_out, umap_labels_out, umap_is_sink_out,
         umap_positions_out, umap_pos_groups_out) = (
            _subsample_plot(umap_all)
        )
        umap_result = {
            "umap_coords": umap_coords_out,
            "umap_labels": umap_labels_out,
            "umap_is_sink": umap_is_sink_out,
            "umap_pos_groups": umap_pos_groups_out,
        }
    except ImportError:
        pass

    return {
        "pca_coords": pca_coords,
        "pca_labels": pca_labels,
        "pca_is_sink": pca_is_sink,
        "pca_pos_groups": pca_pos_groups,
        "pca_explained_var": pca_explained,
        "tsne_coords": tsne_coords,
        "tsne_labels": tsne_labels,
        "tsne_is_sink": tsne_is_sink,
        "tsne_pos_groups": tsne_pos_groups,
        **umap_result,
    }


# ═══════════════════════════════════════════════════════
# Full computation
# ═══════════════════════════════════════════════════════

def compute_qkv_data(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    query_positions: List[int],
    config: dict,
    n_kmeans_clusters: int = 128,
) -> Dict:
    """Compute all QKV clustering data for the dashboard."""
    qpos_arr = np.array(query_positions)
    seq_len = int(qpos_arr.max()) + 1

    q_vecs = Q[qpos_arr]
    k_vecs = K[:seq_len]
    v_vecs = V[:seq_len]

    n_pairs = config.get("embedding", {}).get(
        "n_metric_pairs", 5000,
    )

    # QKV cluster statistics
    print("        Computing QKV cluster statistics...")
    cluster_stats = compute_qkv_cluster_stats(
        q_vecs, k_vecs, v_vecs, n_pairs,
    )

    # K-Means analysis per type
    print("        Running K-Means analysis...")
    kmeans = {
        "Q": compute_kmeans_analysis(
            q_vecs, "Q", n_kmeans_clusters,
        ),
        "K": compute_kmeans_analysis(
            k_vecs, "K", n_kmeans_clusters,
        ),
        "V": compute_kmeans_analysis(
            v_vecs, "V", n_kmeans_clusters,
        ),
    }

    # Joint QKV 2D projections (skip for speed)
    projections = None

    return {
        "cluster_stats": cluster_stats,
        "kmeans": kmeans,
        "projections": projections,
        "seq_len": seq_len,
        "n_queries": len(query_positions),
    }


# ═══════════════════════════════════════════════════════
# Dashboard panels
# ═══════════════════════════════════════════════════════

def _scatter_qkv(ax, proj_data, proj_key, title):
    """QKV scatter with position-group coloring."""
    coord_key = f"{proj_key}_coords"
    if coord_key not in proj_data:
        ax.text(
            0.5, 0.5, f"{proj_key.upper()} unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        return

    coords = proj_data[coord_key]
    labels = proj_data[f"{proj_key}_labels"]
    is_sink = proj_data[f"{proj_key}_is_sink"]
    pos_groups = proj_data.get(f"{proj_key}_pos_groups")

    if pos_groups is not None:
        draw_order = [
            "Q_mid", "K_mid", "V_mid",
            "Q_first", "K_first", "V_first",
            "Q_last", "K_last", "V_last",
        ]
        for group in draw_order:
            mask = pos_groups == group
            if not np.any(mask):
                continue
            sty = _POS_GROUP_STYLE[group]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=sty["s"], alpha=sty["alpha"],
                c=sty["c"],
                label=_POS_GROUP_LABELS[group],
                rasterized=True,
            )
        sink_mask = pos_groups == "sink"
        if np.any(sink_mask):
            ax.scatter(
                coords[sink_mask, 0], coords[sink_mask, 1],
                s=80, marker="*", c="red", label="sink",
                zorder=5, edgecolors="black",
                linewidths=0.5,
            )
    else:
        for qkv, color in _TYPE_COLORS.items():
            mask = (labels == qkv) & ~is_sink
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=3, alpha=0.3, c=color, label=qkv,
                    rasterized=True,
                )
        sink_mask = is_sink
        if np.any(sink_mask):
            ax.scatter(
                coords[sink_mask, 0], coords[sink_mask, 1],
                s=80, marker="*", c="red", label="sink",
                zorder=5, edgecolors="black",
                linewidths=0.5,
            )

    ax.set_title(title, fontsize=9)
    ax.legend(
        fontsize=5, markerscale=2, ncol=3,
        loc="upper right",
    )
    ax.grid(True, alpha=0.3)


def _panel_intra_inter_bars(ax, stats, metric="l2"):
    """Bar chart: intra vs inter distances for Q, K, V."""
    m = stats[metric]

    labels = [
        "Intra-Q", "Intra-K", "Intra-V",
        "Q↔K", "Q↔V", "K↔V",
    ]
    vals = [
        m["intra_q_mean"], m["intra_k_mean"],
        m["intra_v_mean"],
        m["inter_qk_mean"], m["inter_qv_mean"],
        m["inter_kv_mean"],
    ]
    colors = [
        _TYPE_COLORS["Q"], _TYPE_COLORS["K"],
        _TYPE_COLORS["V"],
        "#d4a056", "#7cb342", "#ab47bc",
    ]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v,
            f"{v:.2f}", ha="center", va="bottom",
            fontsize=7,
        )
    ax.set_ylabel(f"Mean {metric.upper()} Distance")
    ax.set_title(
        f"Intra/Inter Distances ({metric.upper()})",
        fontsize=9,
    )
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.grid(True, alpha=0.2, axis="y")


def _panel_separation_ratios(ax, stats):
    """Bar chart: separation ratios for each pair."""
    pairs = ["Q↔K", "Q↔V", "K↔V"]
    l2_vals = [
        stats["l2"]["separation_qk"],
        stats["l2"]["separation_qv"],
        stats["l2"]["separation_kv"],
    ]
    cos_vals = [
        stats["cosine"]["separation_qk"],
        stats["cosine"]["separation_qv"],
        stats["cosine"]["separation_kv"],
    ]

    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(
        x - w / 2, l2_vals, w, color="#5c6bc0",
        alpha=0.85, label="L2",
    )
    ax.bar(
        x + w / 2, cos_vals, w, color="#ef6c00",
        alpha=0.85, label="Cosine",
    )

    for i, (l2v, cosv) in enumerate(zip(l2_vals, cos_vals)):
        ax.text(
            i - w / 2, l2v, f"{l2v:.2f}",
            ha="center", va="bottom", fontsize=7,
        )
        ax.text(
            i + w / 2, cosv, f"{cosv:.2f}",
            ha="center", va="bottom", fontsize=7,
        )

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("Separation Ratio")
    ax.set_title("Separation: inter / mean(intra)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")


def _panel_distance_histograms(ax, stats, metric="l2"):
    """Overlapping histograms of intra/inter distances."""
    m = stats[metric]

    for arr_key, label, color in [
        ("intra_q_dists", "Intra-Q", _TYPE_COLORS["Q"]),
        ("intra_k_dists", "Intra-K", _TYPE_COLORS["K"]),
        ("intra_v_dists", "Intra-V", _TYPE_COLORS["V"]),
        ("inter_qk_dists", "Q↔K", "#d4a056"),
        ("inter_kv_dists", "K↔V", "#ab47bc"),
    ]:
        ax.hist(
            m[arr_key], bins=60, alpha=0.4,
            density=True, color=color, label=label,
        )

    ax.set_xlabel(f"{metric.upper()} Distance")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Distance Distributions ({metric.upper()})",
        fontsize=9,
    )
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)


def _panel_kmeans_sizes(ax, kmeans_data):
    """Cluster size distribution as histograms."""
    for qkv, km in kmeans_data.items():
        if "error" in km:
            continue
        sizes = km["cluster_sizes"]
        ax.hist(
            sizes, bins=50, alpha=0.45,
            density=True, color=_TYPE_COLORS[qkv],
            label=(
                f"{qkv} (max={sizes[0]}, "
                f"min={sizes[-1]})"
            ),
        )

    ax.set_xlabel("Cluster Size (# vectors)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"K-Means Cluster Size Distribution "
        f"(K={kmeans_data['Q'].get('n_clusters', '?')})",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def _panel_kmeans_compactness(ax, kmeans_data):
    """Compactness distribution as histograms."""
    for qkv, km in kmeans_data.items():
        if "error" in km:
            continue
        dists = km["intra_dists_per_cluster"]
        ax.hist(
            dists, bins=50, alpha=0.45,
            density=True, color=_TYPE_COLORS[qkv],
            label=(
                f"{qkv} "
                f"(μ={np.mean(dists):.2f})"
            ),
        )

    ax.set_xlabel("Mean Distance to Centroid")
    ax.set_ylabel("Density")
    ax.set_title("Compactness Distribution", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def _panel_kmeans_summary_table(ax, kmeans_data):
    """Summary table of K-Means statistics."""
    ax.axis("off")

    rows = []
    for qkv in ["Q", "K", "V"]:
        km = kmeans_data[qkv]
        if "error" in km:
            rows.append([qkv, "—"] * 6)
            continue
        rows.append([
            qkv,
            f"{km['silhouette']:.3f}",
            f"{km['compactness_ratio']:.3f}",
            f"{km['largest_cluster']}",
            f"{km['smallest_cluster']}",
            f"{km['size_ratio']:.1f}",
            f"{km['mean_inter_centroid']:.2f}",
        ])

    col_labels = [
        "Type", "Silhouette", "Compact.\nRatio",
        "Largest\nCluster", "Smallest\nCluster",
        "Size\nRatio", "Mean Inter\nCentroid",
    ]
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#e0e0e0")
    for i, qkv in enumerate(["Q", "K", "V"]):
        table[i + 1, 0].set_facecolor(
            _TYPE_COLORS_LIGHT[qkv]
        )

    ax.set_title(
        "K-Means Clustering Summary", fontsize=9,
        pad=10,
    )


def _panel_norm_histograms(ax, stats):
    """Norm distributions for Q, K, V."""
    norms = stats["norms"]
    for qkv, color in _TYPE_COLORS.items():
        arr = norms[f"{qkv.lower()}_norms"]
        ax.hist(
            arr, bins=60, alpha=0.45,
            density=True, color=color,
            label=(
                f"{qkv} "
                f"(μ={norms[f'{qkv.lower()}_mean']:.1f})"
            ),
        )
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Vector Norm Distributions", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════
# Dashboard assembly
# ═══════════════════════════════════════════════════════

def create_qkv_dashboard(
    data: Dict,
    title: str,
    out_path: Path,
):
    """
    Create QKV clustering dashboard.

    Layout (4 rows × 12 columns):
      Row 0: PCA (Q+K+V) | t-SNE (Q+K+V) | UMAP (Q+K+V)
      Row 1: Intra/Inter L2 | Separation Ratios | Norms
      Row 2: Distance Histograms (L2) | Distance Histograms (Cosine) | Intra/Inter Cosine
      Row 3: K-Means Sizes | K-Means Compactness | Summary Table
    """
    setup_style()
    fig = plt.figure(figsize=(30, 28))
    gs = GridSpec(
        4, 12, figure=fig,
        hspace=0.40, wspace=0.45,
    )

    proj = data["projections"]
    stats = data["cluster_stats"]
    km = data["kmeans"]

    # Row 0 — 2D projections (blank if skipped)
    if proj is not None:
        ax = fig.add_subplot(gs[0, 0:4])
        ev = proj["pca_explained_var"]
        _scatter_qkv(
            ax, proj, "pca",
            f"PCA ({ev[0]:.1%}+{ev[1]:.1%})",
        )
        ax = fig.add_subplot(gs[0, 4:8])
        _scatter_qkv(
            ax, proj, "tsne",
            "t-SNE (local structure)",
        )
        ax = fig.add_subplot(gs[0, 8:12])
        _scatter_qkv(
            ax, proj, "umap",
            "UMAP (global structure)",
        )
    else:
        for col_slice, label in [
            (slice(0, 4), "PCA"),
            (slice(4, 8), "t-SNE"),
            (slice(8, 12), "UMAP"),
        ]:
            ax = fig.add_subplot(gs[0, col_slice])
            ax.text(
                0.5, 0.5,
                f"{label}\n(skipped for speed)",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])

    # Row 1 — cluster stats
    ax = fig.add_subplot(gs[1, 0:4])
    _panel_intra_inter_bars(ax, stats, metric="l2")

    ax = fig.add_subplot(gs[1, 4:8])
    _panel_separation_ratios(ax, stats)

    ax = fig.add_subplot(gs[1, 8:12])
    _panel_norm_histograms(ax, stats)

    # Row 2 — distance distributions
    ax = fig.add_subplot(gs[2, 0:4])
    _panel_distance_histograms(ax, stats, metric="l2")

    ax = fig.add_subplot(gs[2, 4:8])
    _panel_distance_histograms(ax, stats, metric="cosine")

    ax = fig.add_subplot(gs[2, 8:12])
    _panel_intra_inter_bars(ax, stats, metric="cosine")

    # Row 3 — K-Means analysis
    ax = fig.add_subplot(gs[3, 0:4])
    _panel_kmeans_sizes(ax, km)

    ax = fig.add_subplot(gs[3, 4:8])
    _panel_kmeans_compactness(ax, km)

    ax = fig.add_subplot(gs[3, 8:12])
    _panel_kmeans_summary_table(ax, km)

    fig.suptitle(
        f"QKV Clustering Analysis — {title}",
        fontsize=16, fontweight="bold", y=0.99,
    )
    save_figure(fig, out_path, dpi=150)


# ═══════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════

def _last_query_positions(seq_len, n):
    start = max(0, seq_len - n)
    return list(range(start, seq_len))


def _resolve_heads(config, vectors_dir, task):
    ecfg = config.get("exploration", {})
    mode = ecfg.get("head_mode", "custom")

    if mode == "selected_heads":
        meta = load_task_metadata(
            Path(vectors_dir), task,
        )
        heads = meta.get("selected_heads", [])
        if not heads:
            return [{
                "layer": ecfg.get("layer", 17),
                "q_head": ecfg.get("q_head", 0),
                "kv_head": ecfg.get("kv_head", 0),
            }]
        return [
            {
                "layer": h["layer"],
                "q_head": h["q_head"],
                "kv_head": h["kv_head"],
            }
            for h in heads
        ]

    return [{
        "layer": ecfg.get("layer", 17),
        "q_head": ecfg.get("q_head", 0),
        "kv_head": ecfg.get("kv_head", 0),
    }]


def run_qkv_clustering(
    config_path: str,
    tasks: list = None,
    vectors_dir: str = None,
    n_clusters: int = 8,
):
    """Run QKV clustering analysis on .pt data."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ecfg = config.get("exploration", {})
    n_examples = ecfg.get("n_examples", 1)
    use_rope = ecfg.get("use_rope", True)

    data_cfg = config.get("data", {})
    vdir = vectors_dir or data_cfg.get(
        "vectors_dir", "data/vectors",
    )
    results_dir = Path(
        data_cfg.get("results_dir", "results")
    )

    if tasks is None:
        tasks = config.get("tasks", [])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_base = results_dir / f"qkv_clustering_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"\n  Task: {task}")
        task_dir = out_base / task

        heads = _resolve_heads(config, vdir, task)
        print(f"    Heads: {len(heads)}")

        for head_info in heads:
            layer = head_info["layer"]
            q_head = head_info["q_head"]
            kv_head = head_info["kv_head"]
            head_label = f"L{layer}H{q_head}"
            print(f"    {head_label}:")

            head_dir = task_dir / head_label
            head_dir.mkdir(parents=True, exist_ok=True)

            examples = list(load_examples(
                Path(vdir), task, layer,
                head=q_head, kv_head=kv_head,
                max_examples=n_examples,
                use_rope=use_rope,
            ))
            if not examples:
                print(f"      No data, skipping")
                continue

            ex = examples[0]
            Q, K, V = ex["Q"], ex["K"], ex["V"]
            seq_len = Q.shape[0]
            all_qpos = list(range(seq_len))
            print(f"      {seq_len:,} tokens")

            print(f"      Computing QKV clustering...")
            qkv_data = compute_qkv_data(
                Q, K, V, all_qpos, config,
                n_kmeans_clusters=n_clusters,
            )

            info = (
                f"{task} — {head_label}"
                f" ({seq_len:,} tok)"
            )
            print(f"      Creating dashboard...")
            create_qkv_dashboard(
                qkv_data, info,
                head_dir / "qkv_clustering.png",
            )
            print(f"      Saved: {head_dir}")

    print(f"\n  QKV clustering saved: {out_base}")


def main():
    parser = argparse.ArgumentParser(
        description="QKV Clustering Analysis.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
    )
    parser.add_argument(
        "--all", action="store_true",
    )
    parser.add_argument(
        "--vectors-dir", default=None,
    )
    parser.add_argument(
        "--n-clusters", type=int, default=128,
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parent
            / "exploration_config.yaml"
        ),
    )
    args = parser.parse_args()

    tasks = args.tasks
    if args.all:
        tasks = None

    run_qkv_clustering(
        args.config,
        tasks=tasks,
        vectors_dir=args.vectors_dir,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
