"""
Pairwise similarity analysis for QK, QQ, and KK pairs.

Computes cosine similarities and scaled dot products with
positional distance binning. Each anchor (last N queries/keys)
is compared against a subsampled set of comparison positions
from the full context.

  QK — each of the last N queries vs sampled causal keys
  QQ — each of the last N queries vs sampled positions
  KK — each of the last N keys vs sampled positions

Sink = key at position 0. Only applies to pair types that
involve keys (QK, KK). QQ has no sink concept.
"""

import numpy as np
from typing import Dict, List, Literal


def _subsample_targets(
    all_positions: np.ndarray,
    max_targets: int,
    include_zero: bool,
    rng: np.random.Generator,
):
    """Subsample comparison targets from all positions.

    Always includes position 0 (sink) when include_zero=True.
    Returns sorted array of target positions.
    """
    if len(all_positions) <= max_targets:
        return all_positions

    if include_zero:
        # Reserve slot for sink, sample rest
        non_zero = all_positions[all_positions != 0]
        n_sample = min(max_targets - 1, len(non_zero))
        sampled = rng.choice(
            non_zero, n_sample, replace=False,
        )
        targets = np.concatenate([[0], sampled])
    else:
        targets = rng.choice(
            all_positions, max_targets, replace=False,
        )
    return np.sort(targets)


def _generate_pairs(
    pair_type: Literal["qk", "qq", "kk"],
    query_positions: List[int],
    max_targets: int = 20000,
    seed: int = 42,
):
    """
    Generate position pairs with subsampled targets.

    Each anchor (last N positions) is compared against up to
    max_targets positions from the full context. Sink (pos 0)
    is always included for pair types with keys.

    Returns (pos_a, pos_b, distances, is_sink) arrays.
    """
    rng = np.random.default_rng(seed)
    qpos_arr = np.array(query_positions)
    max_qpos = int(qpos_arr.max())
    all_positions = np.arange(max_qpos + 1)

    if pair_type == "qk":
        # Each query vs subsampled causal keys
        chunks_a, chunks_b = [], []
        for qp in qpos_arr:
            keys = np.arange(qp + 1)
            targets = _subsample_targets(
                keys, max_targets,
                include_zero=True, rng=rng,
            )
            chunks_a.append(np.full(len(targets), qp))
            chunks_b.append(targets)
        pos_a = np.concatenate(chunks_a)
        pos_b = np.concatenate(chunks_b)
        distances = pos_a - pos_b
        is_sink = (pos_b == 0)
        return pos_a, pos_b, distances, is_sink

    elif pair_type == "qq":
        # Each of last N queries vs subsampled positions
        chunks_a, chunks_b = [], []
        for qp in qpos_arr:
            others = all_positions[all_positions != qp]
            targets = _subsample_targets(
                others, max_targets,
                include_zero=False, rng=rng,
            )
            chunks_a.append(np.full(len(targets), qp))
            chunks_b.append(targets)
        pos_a = np.concatenate(chunks_a)
        pos_b = np.concatenate(chunks_b)
        distances = np.abs(pos_a - pos_b)
        is_sink = np.zeros(len(pos_a), dtype=bool)
        return pos_a, pos_b, distances, is_sink

    elif pair_type == "kk":
        kpos_selected = qpos_arr
        if 0 not in kpos_selected:
            kpos_selected = np.concatenate(
                [[0], kpos_selected],
            )
        # Each selected key vs subsampled positions
        chunks_a, chunks_b = [], []
        for kp in kpos_selected:
            others = all_positions[all_positions != kp]
            targets = _subsample_targets(
                others, max_targets,
                include_zero=True, rng=rng,
            )
            chunks_a.append(np.full(len(targets), kp))
            chunks_b.append(targets)
        pos_a = np.concatenate(chunks_a)
        pos_b = np.concatenate(chunks_b)
        distances = np.abs(pos_a - pos_b)
        is_sink = (pos_a == 0) | (pos_b == 0)
        return pos_a, pos_b, distances, is_sink

    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")


def _compute_similarities(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    head_dim: int,
):
    """Compute cosine similarities and scaled dot products."""
    dots_raw = np.sum(vecs_a * vecs_b, axis=1)
    dots_scaled = dots_raw / np.sqrt(head_dim)

    norms_a = np.linalg.norm(vecs_a, axis=1)
    norms_b = np.linalg.norm(vecs_b, axis=1)
    cosines = dots_raw / np.maximum(
        norms_a * norms_b, 1e-10,
    )
    return cosines, dots_scaled


def _bin_by_distance(
    values: np.ndarray,
    distances: np.ndarray,
    n_bins: int,
):
    """
    Bin values by distance using percentile-based edges.

    Returns bin_centers, binned_mean, binned_sem.
    Every bin has roughly equal sample count.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(distances, percentiles)
    edges = np.unique(edges)

    centers = []
    means = []
    sems = []
    for i in range(len(edges) - 1):
        if i < len(edges) - 2:
            mask = (
                (distances >= edges[i])
                & (distances < edges[i + 1])
            )
        else:
            mask = (
                (distances >= edges[i])
                & (distances <= edges[i + 1])
            )
        vals = values[mask]
        if len(vals) == 0:
            continue
        center = np.sqrt(
            max(edges[i], 1) * max(edges[i + 1], 1)
        )
        centers.append(center)
        means.append(float(np.mean(vals)))
        sems.append(
            float(np.std(vals) / np.sqrt(len(vals)))
        )

    return (
        np.array(centers),
        np.array(means),
        np.array(sems),
    )


def _categorize_pairs(
    values: np.ndarray,
    distances: np.ndarray,
    is_sink: np.ndarray,
    local_window: int,
):
    """
    Split values into sink / local / global categories.

    - sink: key at position 0
    - local: distance <= local_window (and not sink)
    - global: distance > local_window (and not sink)
    """
    local_mask = ~is_sink & (distances <= local_window)
    global_mask = ~is_sink & (distances > local_window)

    return {
        "sink": values[is_sink],
        "local": values[local_mask],
        "global": values[global_mask],
    }


def _per_query_softmax(
    query_ids: np.ndarray,
    dots: np.ndarray,
) -> np.ndarray:
    """Compute per-query softmax weights.

    Each query independently softmaxes over its own keys,
    matching real attention behavior. All queries contribute
    equally (weights within each query sum to 1).
    """
    weights = np.empty_like(dots)
    for qid in np.unique(query_ids):
        mask = query_ids == qid
        q_dots = dots[mask]
        # Numerically stable softmax
        q_dots_shifted = q_dots - q_dots.max()
        exp_dots = np.exp(q_dots_shifted)
        weights[mask] = exp_dots / exp_dots.sum()
    return weights


def compute_pairwise_data(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    pair_type: Literal["qk", "qq", "kk"] = "qk",
    n_distance_bins: int = 15,
    local_window: int = 1024,
    max_targets: int = 20000,
    seed: int = 42,
    **kwargs,
) -> Dict:
    """
    Compute pairwise similarities with distance analysis.

    Each anchor is compared against up to max_targets
    subsampled positions from the full context.

    Parameters
    ----------
    pair_type : "qk", "qq", or "kk"
    n_distance_bins : bins for "vs distance" plots
    local_window : positions within this distance are "local"
    max_targets : max comparison positions per anchor
    seed : random seed for subsampling

    Returns
    -------
    Dict with cosine_all, dots_all, distances, is_sink,
    binned data, and sink/local/global category data.
    """
    pos_a, pos_b, distances, is_sink = _generate_pairs(
        pair_type, query_positions,
        max_targets=max_targets, seed=seed,
    )

    # Look up vectors
    if pair_type == "qk":
        vecs_a = Q[pos_a]
        vecs_b = K[pos_b]
    elif pair_type == "qq":
        vecs_a = Q[pos_a]
        vecs_b = Q[pos_b]
    elif pair_type == "kk":
        vecs_a = K[pos_a]
        vecs_b = K[pos_b]

    cosines, dots = _compute_similarities(
        vecs_a, vecs_b, head_dim,
    )

    # Per-query softmax attention weights (QK only)
    attn_weights = None
    if pair_type == "qk":
        attn_weights = _per_query_softmax(pos_a, dots)

    # Distance binning for "vs distance" plots
    cos_centers, cos_mean, cos_sem = _bin_by_distance(
        cosines, distances, n_distance_bins,
    )
    dot_centers, dot_mean, dot_sem = _bin_by_distance(
        dots, distances, n_distance_bins,
    )

    # Sink / local / global categories
    cosine_cats = _categorize_pairs(
        cosines, distances, is_sink, local_window,
    )
    dots_cats = _categorize_pairs(
        dots, distances, is_sink, local_window,
    )

    return {
        "pair_type": pair_type,
        "cosine_all": cosines,
        "dots_all": dots,
        "distances": distances,
        "is_sink": is_sink,
        "local_window": local_window,
        "attn_weights": attn_weights,
        # Binned data for "vs distance" plots
        "distance_bin_centers": cos_centers,
        "cosine_binned_mean": cos_mean,
        "cosine_binned_sem": cos_sem,
        "dots_bin_centers": dot_centers,
        "dots_binned_mean": dot_mean,
        "dots_binned_sem": dot_sem,
        # Sink / local / global categories
        "cosine_sink": cosine_cats["sink"],
        "cosine_local": cosine_cats["local"],
        "cosine_global": cosine_cats["global"],
        "dots_sink": dots_cats["sink"],
        "dots_local": dots_cats["local"],
        "dots_global": dots_cats["global"],
    }
