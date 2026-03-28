"""
Shared math utilities for the entire codebase.

Core attention primitives (softmax, full/subset attention),
error metrics, grouping helpers, KMeans, and attention
statistics (entropy, concentration, norms).

Imported by algorithms/, experiment/, and exploration/.
"""

import numpy as np
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════
# Core primitives
# ═══════════════════════════════════════════════════════

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def full_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full causal attention (ground truth).

    Keys/values should already be sliced to causal window.

    Returns (output, logits, weights).
    """
    logits = (query @ keys.T) / np.sqrt(head_dim)
    weights = softmax(logits)
    output = weights @ values
    return output, logits, weights


def subset_attention(
    logits: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Softmax attention over a subset of positions.

    Renormalizes softmax over only the selected indices,
    then returns the weighted sum of their values.
    """
    w = softmax(logits[indices])
    return w @ values[indices]


def relative_l2_error(
    approx: np.ndarray,
    truth: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """||approx - truth||_2 / ||truth||_2."""
    denom = np.linalg.norm(truth)
    if denom < eps:
        return 0.0
    return float(np.linalg.norm(approx - truth) / denom)


# ═══════════════════════════════════════════════════════
# Attention statistics
# ═══════════════════════════════════════════════════════

def entropy_nats(weights: np.ndarray) -> float:
    """Shannon entropy in nats. weights: [N]."""
    w = weights[weights > 1e-30]
    return float(-np.sum(w * np.log(w)))


def top_k_mass(
    weights: np.ndarray, k: int = 100,
) -> float:
    """Fraction of attention in top-k positions."""
    if len(weights) <= k:
        return 1.0
    top = np.partition(weights, -k)[-k:]
    return float(np.sum(top))


def no_sink_local_mask(
    n: int,
    n_sink: int = 1,
    local_window: int = 1024,
) -> np.ndarray:
    """Bool mask: True for non-sink, non-local."""
    mask = np.ones(n, dtype=bool)
    mask[:n_sink] = False
    local_start = max(n_sink, n - local_window)
    mask[local_start:] = False
    return mask


def attention_stats_for_query(
    query: np.ndarray,
    keys: np.ndarray,
    head_dim: int,
    n_sink: int = 1,
    local_window: int = 1024,
) -> Dict[str, float]:
    """
    Full attention statistics for one query.

    Returns entropy, top-100 mass, and their
    no-sink-local variants.
    """
    n = len(keys)
    logits = (query @ keys.T) / np.sqrt(head_dim)
    weights = softmax(logits)
    mask = no_sink_local_mask(
        n, n_sink, local_window,
    )

    e_full = entropy_nats(weights)
    t100_full = top_k_mass(weights)

    w_masked = weights[mask]
    total = np.sum(w_masked)
    if total > 1e-10:
        w_normed = w_masked / total
        e_no_sl = entropy_nats(w_normed)
        t100_no_sl = top_k_mass(w_normed)
    else:
        e_no_sl = 0.0
        t100_no_sl = 1.0

    return {
        "entropy_full": e_full,
        "entropy_no_sink_local": e_no_sl,
        "top100_mass_full": t100_full,
        "top100_mass_no_sink_local": t100_no_sl,
    }


def stats_from_weights(
    weights: np.ndarray,
    n_sink: int = 1,
    local_window: int = 1024,
) -> Dict[str, float]:
    """
    Compute entropy and top-100 mass from pre-computed
    attention weights (avoids recomputing logits).
    """
    n = len(weights)
    mask = no_sink_local_mask(n, n_sink, local_window)

    e_full = entropy_nats(weights)
    t100_full = top_k_mass(weights)

    w_masked = weights[mask]
    total = np.sum(w_masked)
    if total > 1e-10:
        w_normed = w_masked / total
        e_no_sl = entropy_nats(w_normed)
        t100_no_sl = top_k_mass(w_normed)
    else:
        e_no_sl = 0.0
        t100_no_sl = 1.0

    return {
        "entropy_full": e_full,
        "entropy_no_sink_local": e_no_sl,
        "top100_mass_full": t100_full,
        "top100_mass_no_sink_local": t100_no_sl,
    }


def concentration_curve(
    weights: np.ndarray,
    percentages: List[float] = None,
) -> Dict[str, float]:
    """
    Fraction of mass captured by top X% of keys.

    Returns dict like {"top_1pct": 0.35, ...}.
    """
    if percentages is None:
        percentages = [1, 5, 10, 20, 50]
    n = len(weights)
    sorted_w = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_w)
    result = {}
    for pct in percentages:
        k = max(1, int(n * pct / 100))
        k = min(k, n)
        result[f"top_{pct}pct"] = float(cumsum[k - 1])
    return result


def qk_cosine_similarities(
    queries: np.ndarray,
    keys: np.ndarray,
    n_sample: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Cosine similarities between sampled Q-K pairs.

    Returns flat array of cosine similarities.
    """
    rng = np.random.default_rng(seed)
    nq, nk = len(queries), len(keys)
    n = min(n_sample, nq * nk)

    qi = rng.integers(0, nq, size=n)
    ki = rng.integers(0, nk, size=n)

    q_norms = np.linalg.norm(
        queries[qi], axis=1, keepdims=True,
    )
    k_norms = np.linalg.norm(
        keys[ki], axis=1, keepdims=True,
    )

    dots = np.sum(queries[qi] * keys[ki], axis=1)
    norms = (q_norms * k_norms).squeeze()
    norms = np.maximum(norms, 1e-10)
    return dots / norms


def norm_statistics(
    vectors: np.ndarray,
) -> Dict[str, float]:
    """L2 norm mean, std, and CV for a vector array."""
    norms = np.linalg.norm(vectors, axis=1)
    mean = float(np.mean(norms))
    std = float(np.std(norms))
    return {
        "norm_mean": mean,
        "norm_std": std,
        "norm_cv": std / max(mean, 1e-10),
    }


def kv_norm_correlation(
    keys: np.ndarray,
    values: np.ndarray,
) -> float:
    """Pearson correlation between K and V norms."""
    k_norms = np.linalg.norm(keys, axis=1)
    v_norms = np.linalg.norm(values, axis=1)
    if len(k_norms) < 2:
        return 0.0
    return float(np.corrcoef(k_norms, v_norms)[0, 1])


# ═══════════════════════════════════════════════════════
# Grouping and clustering
# ═══════════════════════════════════════════════════════

def compute_special_indices(
    n_causal: int,
    n_sink: int = 1,
    local_window: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sink + local window indices.

    Returns (special_idx, candidate_idx).
    """
    local_start = max(n_sink, n_causal - local_window)
    sink = np.arange(n_sink)
    local = np.arange(local_start, n_causal)
    special_idx = np.unique(
        np.concatenate([sink, local])
    ).astype(np.int64)
    special_set = set(special_idx.tolist())
    candidate_idx = np.array(
        [i for i in range(n_causal)
         if i not in special_set],
        dtype=np.int64,
    )
    return special_idx, candidate_idx


def make_doubling_boundaries(n: int) -> List[Tuple[int, int]]:
    """Doubling group boundaries: sizes 1, 1, 2, 4, 8, ..."""
    groups = []
    pos = 0
    size = 1
    while pos < n:
        end = min(pos + size, n)
        groups.append((pos, end))
        pos = end
        if len(groups) >= 2:
            size *= 2
    return groups


def make_equal_groups(
    sorted_indices: np.ndarray,
    n_groups: int,
) -> List[np.ndarray]:
    """Split sorted indices into n_groups equal-sized groups."""
    n = len(sorted_indices)
    if n == 0:
        return []
    group_size = max(1, n // n_groups)
    groups = []
    for i in range(n_groups):
        start = i * group_size
        if i < n_groups - 1:
            end = (i + 1) * group_size
        else:
            end = n
        if start >= n:
            break
        groups.append(sorted_indices[start:end])
    return groups


def flat_kmeans(
    data: np.ndarray,
    C: int,
    seed: int = 42,
    n_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means with k-means++ init (numpy only).

    Returns (centroids [C, d], labels [N]).
    """
    rng = np.random.default_rng(seed)
    N, d = data.shape
    C = min(C, N)

    centroids = np.empty((C, d), dtype=data.dtype)
    centroids[0] = data[rng.integers(N)]
    data_sq = np.sum(data ** 2, axis=1)
    min_dists = np.full(N, np.inf, dtype=np.float64)

    for c in range(1, C):
        cent_sq = float(
            np.sum(centroids[c - 1] ** 2)
        )
        new_d = (
            data_sq + cent_sq
            - 2.0 * (data @ centroids[c - 1])
        )
        np.minimum(min_dists, new_d, out=min_dists)
        np.maximum(min_dists, 0.0, out=min_dists)
        probs = min_dists / (min_dists.sum() + 1e-10)
        centroids[c] = data[rng.choice(N, p=probs)]

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        c_sq = np.sum(
            centroids ** 2, axis=1, keepdims=True
        )
        dists = (
            data_sq[:, None]
            - 2.0 * (data @ centroids.T)
            + c_sq.T
        )
        new_labels = np.argmin(
            dists, axis=1
        ).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(C):
            mask = labels == c
            if np.sum(mask) > 0:
                centroids[c] = np.mean(
                    data[mask], axis=0
                )

    return centroids, labels


def hybrid_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    logits: np.ndarray,
    group_indices_list: List[np.ndarray],
    top_k: int,
    head_dim: int,
    special_idx: np.ndarray,
    mode: str = "hybrid",
) -> Tuple[np.ndarray, int]:
    """
    Two-mode attention over pre-sorted groups.

    Modes:
      topk:   top_k groups as individual keys + special.
      hybrid: top_k individual + remaining as reps +
              special. Count-weighted softmax over all
              items. With top_k=0 this reduces to pure
              grouped mode (all groups as reps).

    Returns (output, effective_budget).
    """
    n_special = len(special_idx)
    G = len(group_indices_list)
    sqrt_d = np.sqrt(head_dim)
    top_k = min(top_k, G)

    top_individual = []
    for gi in range(top_k):
        top_individual.append(group_indices_list[gi])
    if top_individual:
        top_keys_idx = np.concatenate(top_individual)
    else:
        top_keys_idx = np.array([], dtype=np.int64)
    n_top = len(top_keys_idx)

    if mode == "topk":
        all_idx = np.concatenate(
            [special_idx, top_keys_idx]
        ).astype(np.int64)
        if len(all_idx) == 0:
            return np.zeros(head_dim), 0
        output = subset_attention(logits, values, all_idx)
        return output, len(all_idx)

    n_far = G - top_k
    n_total = n_special + n_top + n_far
    if n_total == 0:
        return np.zeros(head_dim), 0

    scores = np.empty(n_total)
    out_values = np.empty((n_total, head_dim))

    scores[:n_special] = logits[special_idx]
    out_values[:n_special] = values[special_idx]

    off = n_special
    scores[off:off + n_top] = logits[top_keys_idx]
    out_values[off:off + n_top] = values[top_keys_idx]

    off = n_special + n_top
    for fi, gi in enumerate(range(top_k, G)):
        idx = group_indices_list[gi]
        count = len(idx)
        if count == 0:
            scores[off + fi] = -1e9
            out_values[off + fi] = 0.0
            continue
        avg_key = np.mean(keys[idx], axis=0)
        avg_val = np.mean(values[idx], axis=0)
        scores[off + fi] = (
            query @ avg_key / sqrt_d + np.log(count)
        )
        out_values[off + fi] = avg_val

    w = softmax(scores)
    return w @ out_values, n_total
