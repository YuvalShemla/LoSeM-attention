"""
Embedding projections — PCA, t-SNE, and UMAP of Q+K spaces.

Projects query and key vectors into 2D for visualization.
Computes cluster separation metrics in both original and
reduced spaces.

All three projections are fit on the same subsample
(n_fit vectors), then subsampled again for plotting
(n_plot vectors). The sink key (position 0) is always
preserved in both stages.

Scatter points are colored by position group:
  Q/K × first 1024 / middle / last 1024
with shade encoding position (light=early, dark=late)
and hue encoding type (blue=Q, purple=K).
"""

import numpy as np
from typing import Dict, List

# Position window for first/last grouping
_POS_WINDOW = 1024


def _cluster_metrics(q_vecs, k_vecs, n_pairs=2000):
    """
    Compute Q-K cluster separation metrics.

    Returns dict with:
      - centroid_distance: ||mean(Q) - mean(K)||
      - mean_intra_q: avg ||qi - qj|| (sampled)
      - mean_intra_k: avg ||ki - kj|| (sampled)
      - mean_inter_qk: avg ||q - k|| (sampled)
      - separation_ratio: inter / mean(intra_q, intra_k)
    """
    rng = np.random.default_rng(42)

    q_centroid = q_vecs.mean(axis=0)
    k_centroid = k_vecs.mean(axis=0)
    centroid_dist = float(np.linalg.norm(
        q_centroid - k_centroid,
    ))

    # Intra-Q distances
    nq = len(q_vecs)
    qi = rng.integers(0, nq, size=n_pairs)
    qj = rng.integers(0, nq, size=n_pairs)
    mask = qi != qj
    intra_q = float(np.mean(np.linalg.norm(
        q_vecs[qi[mask]] - q_vecs[qj[mask]], axis=1,
    )))

    # Intra-K distances
    nk = len(k_vecs)
    ki = rng.integers(0, nk, size=n_pairs)
    kj = rng.integers(0, nk, size=n_pairs)
    mask = ki != kj
    intra_k = float(np.mean(np.linalg.norm(
        k_vecs[ki[mask]] - k_vecs[kj[mask]], axis=1,
    )))

    # Inter Q-K distances
    q_idx = rng.integers(0, nq, size=n_pairs)
    k_idx = rng.integers(0, nk, size=n_pairs)
    inter_qk = float(np.mean(np.linalg.norm(
        q_vecs[q_idx] - k_vecs[k_idx], axis=1,
    )))

    mean_intra = (intra_q + intra_k) / 2
    separation = (
        inter_qk / max(mean_intra, 1e-10)
    )

    return {
        "centroid_distance": centroid_dist,
        "mean_intra_q": intra_q,
        "mean_intra_k": intra_k,
        "mean_inter_qk": inter_qk,
        "separation_ratio": separation,
    }


def _assign_position_groups(
    positions: np.ndarray,
    labels: np.ndarray,
    is_sink: np.ndarray,
    seq_len: int,
):
    """Assign position groups for coloring.

    Groups: Q_first, Q_middle, Q_last,
            K_first, K_middle, K_last, sink.
    """
    groups = np.empty(len(positions), dtype="U10")
    first_end = _POS_WINDOW
    last_start = seq_len - _POS_WINDOW

    for qk in ["Q", "K"]:
        type_mask = labels == qk
        pos = positions[type_mask]

        first = type_mask & (positions < first_end)
        last = type_mask & (positions >= last_start)
        middle = type_mask & ~first & ~last

        groups[first] = f"{qk}_first"
        groups[middle] = f"{qk}_mid"
        groups[last] = f"{qk}_last"

    # Sink overrides
    groups[is_sink] = "sink"
    return groups


def _subsample_for_plot(
    rng, coords, labels, is_sink, positions,
    pos_groups, n,
):
    """Subsample projected points, always keeping sinks."""
    if len(coords) <= n:
        return (
            coords, labels, is_sink,
            positions, pos_groups,
        )
    sink_idx = np.where(is_sink)[0]
    non_sink_idx = np.where(~is_sink)[0]
    n_non_sink = n - len(sink_idx)
    chosen = rng.choice(
        non_sink_idx,
        min(n_non_sink, len(non_sink_idx)),
        replace=False,
    )
    plot_idx = np.concatenate([sink_idx, chosen])
    return (
        coords[plot_idx], labels[plot_idx],
        is_sink[plot_idx], positions[plot_idx],
        pos_groups[plot_idx],
    )


def compute_embedding_projections(
    Q: np.ndarray,
    K: np.ndarray,
    query_positions: List[int],
    config: dict,
) -> Dict:
    """
    Compute PCA, t-SNE, UMAP projections + cluster metrics.

    Fit on n_fit vectors (25K Q + 25K K), plot n_plot vectors.
    Only K at position 0 is marked as the attention sink.
    Returns position groups for coloring by first/mid/last.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    emb_cfg = config.get("embedding", {})
    n_fit = emb_cfg.get("n_fit", 50000)
    n_plot = emb_cfg.get("n_plot", 15000)
    n_metric_pairs = emb_cfg.get("n_metric_pairs", 5000)
    perplexity = emb_cfg.get("tsne_perplexity", 100.0)
    tsne_max_iter = emb_cfg.get("tsne_max_iter", 1500)
    tsne_init = emb_cfg.get("tsne_init", "pca")
    tsne_lr = emb_cfg.get("tsne_learning_rate", "auto")
    umap_n_neighbors = emb_cfg.get(
        "umap_n_neighbors", 50,
    )
    umap_min_dist = emb_cfg.get("umap_min_dist", 0.05)
    umap_metric = emb_cfg.get("umap_metric", "cosine")

    qpos_arr = np.array(query_positions)
    max_qpos = int(qpos_arr.max())
    seq_len = max_qpos + 1

    q_vecs = Q[qpos_arr]
    k_vecs = K[:seq_len]

    # Only K[0] is the attention sink
    q_is_sink = np.zeros(len(q_vecs), dtype=bool)
    k_is_sink = np.zeros(len(k_vecs), dtype=bool)
    k_is_sink[0] = True

    q_labels = np.array(["Q"] * len(q_vecs))
    k_labels = np.array(["K"] * len(k_vecs))

    # Track original positions
    q_positions = qpos_arr.copy()
    k_positions = np.arange(seq_len)

    # Original-space metrics (on all vectors)
    orig_metrics = _cluster_metrics(
        q_vecs, k_vecs, n_metric_pairs,
    )

    # ── Subsample for fitting (shared by all methods) ──
    rng = np.random.default_rng(42)
    half_fit = n_fit // 2
    q_n = min(half_fit, len(q_vecs))
    k_n = min(half_fit, len(k_vecs))

    q_idx = rng.choice(len(q_vecs), q_n, replace=False)
    k_idx = rng.choice(len(k_vecs), k_n, replace=False)

    # Ensure sink key (position 0) is in fit set
    if 0 not in k_idx and len(k_vecs) > 0:
        k_idx[0] = 0

    fit_vecs = np.vstack([q_vecs[q_idx], k_vecs[k_idx]])
    fit_labels = np.concatenate([
        q_labels[q_idx], k_labels[k_idx],
    ])
    fit_is_sink = np.concatenate([
        q_is_sink[q_idx], k_is_sink[k_idx],
    ])
    fit_positions = np.concatenate([
        q_positions[q_idx], k_positions[k_idx],
    ])

    # Assign position groups for coloring
    fit_pos_groups = _assign_position_groups(
        fit_positions, fit_labels,
        fit_is_sink, seq_len,
    )

    # ── PCA ──
    pca = PCA(n_components=2)
    pca_all = pca.fit_transform(fit_vecs)
    pca_explained = pca.explained_variance_ratio_

    pca_q = pca_all[fit_labels == "Q"]
    pca_k = pca_all[fit_labels == "K"]
    pca_metrics = _cluster_metrics(
        pca_q, pca_k, n_metric_pairs,
    )

    (pca_coords, pca_labels, pca_is_sink,
     pca_positions, pca_pos_groups) = (
        _subsample_for_plot(
            rng, pca_all, fit_labels,
            fit_is_sink, fit_positions,
            fit_pos_groups, n_plot,
        )
    )

    # ── t-SNE ──
    effective_perp = min(perplexity, len(fit_vecs) / 4)
    tsne = TSNE(
        n_components=2,
        perplexity=max(5.0, effective_perp),
        max_iter=tsne_max_iter,
        init=tsne_init,
        learning_rate=tsne_lr,
        random_state=42,
    )
    tsne_all = tsne.fit_transform(fit_vecs)

    tsne_q = tsne_all[fit_labels == "Q"]
    tsne_k = tsne_all[fit_labels == "K"]
    tsne_metrics = _cluster_metrics(
        tsne_q, tsne_k, n_metric_pairs,
    )

    (tsne_coords, tsne_labels, tsne_is_sink,
     tsne_positions, tsne_pos_groups) = (
        _subsample_for_plot(
            rng, tsne_all, fit_labels,
            fit_is_sink, fit_positions,
            fit_pos_groups, n_plot,
        )
    )

    # ── UMAP ──
    umap_coords = None
    umap_labels = None
    umap_is_sink = None
    umap_pos_groups = None
    umap_metrics = None

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

        umap_q = umap_all[fit_labels == "Q"]
        umap_k = umap_all[fit_labels == "K"]
        umap_metrics = _cluster_metrics(
            umap_q, umap_k, n_metric_pairs,
        )

        (umap_coords, umap_labels, umap_is_sink,
         umap_positions, umap_pos_groups) = (
            _subsample_for_plot(
                rng, umap_all, fit_labels,
                fit_is_sink, fit_positions,
                fit_pos_groups, n_plot,
            )
        )

    except ImportError:
        pass

    result = {
        "pca_coords": pca_coords,
        "pca_labels": pca_labels,
        "pca_is_sink": pca_is_sink,
        "pca_pos_groups": pca_pos_groups,
        "pca_explained_var": pca_explained,
        "tsne_coords": tsne_coords,
        "tsne_labels": tsne_labels,
        "tsne_is_sink": tsne_is_sink,
        "tsne_pos_groups": tsne_pos_groups,
        "orig_metrics": orig_metrics,
        "pca_metrics": pca_metrics,
        "tsne_metrics": tsne_metrics,
    }

    if umap_coords is not None:
        result["umap_coords"] = umap_coords
        result["umap_labels"] = umap_labels
        result["umap_is_sink"] = umap_is_sink
        result["umap_pos_groups"] = umap_pos_groups
        result["umap_metrics"] = umap_metrics

    return result
