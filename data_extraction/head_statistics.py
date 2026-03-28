"""
Per-head attention statistics for head selection.

Computes entropy and top-K mass over causal attention
distributions, with and without sink/local window.
Used to select 3 representative heads for Phase 2:
highest, lowest, and median entropy.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(
        exp_x, axis=-1, keepdims=True
    )


def _entropy(weights: np.ndarray) -> float:
    """Shannon entropy in nats. weights: [N]."""
    w = weights[weights > 1e-30]
    return float(-np.sum(w * np.log(w)))


def _top_k_mass(
    weights: np.ndarray, k: int = 100,
) -> float:
    """Fraction of attention in top-k positions."""
    if len(weights) <= k:
        return 1.0
    top_k = np.partition(weights, -k)[-k:]
    return float(np.sum(top_k))


def _no_sink_local_mask(
    n: int, n_sink: int = 1, local_window: int = 1024,
) -> np.ndarray:
    """Boolean mask: True for non-sink, non-local."""
    mask = np.ones(n, dtype=bool)
    mask[:n_sink] = False
    local_start = max(n_sink, n - local_window)
    mask[local_start:] = False
    return mask


def compute_query_stats(
    query: np.ndarray,
    keys: np.ndarray,
    head_dim: int,
    n_sink: int = 1,
    local_window: int = 1024,
    top_k_for_mass: int = 100,
) -> Dict[str, float]:
    """
    Compute attention stats for one query position.

    Returns dict with entropy and top-K mass, both
    full and excluding sink + local window.
    """
    n = len(keys)
    logits = (query @ keys.T) / np.sqrt(head_dim)
    weights = _softmax(logits)

    mask = _no_sink_local_mask(
        n, n_sink, local_window
    )

    e_full = _entropy(weights)
    tk_full = _top_k_mass(weights, top_k_for_mass)

    w_masked = weights[mask]
    if np.sum(w_masked) > 1e-10:
        w_normed = w_masked / np.sum(w_masked)
        e_no_sl = _entropy(w_normed)
        tk_no_sl = _top_k_mass(w_normed, top_k_for_mass)
    else:
        e_no_sl = 0.0
        tk_no_sl = 1.0

    k = top_k_for_mass
    return {
        "entropy_full": e_full,
        "entropy_no_sink_local": e_no_sl,
        f"top{k}_mass_full": tk_full,
        f"top{k}_mass_no_sink_local": tk_no_sl,
    }


def compute_head_statistics(
    Q_all: np.ndarray,
    K_all: np.ndarray,
    head_dim: int,
    n_queries: int = 10,
    n_sink: int = 1,
    local_window: int = 1024,
    top_k_for_mass: int = 100,
) -> Dict[str, float]:
    """
    Average attention stats for one head over
    the last N query positions.

    Q_all: [seq_len, head_dim] — all queries
    K_all: [seq_len, head_dim] — all keys

    Uses the last n_queries positions (deterministic,
    consistent with experiment evaluation).
    """
    seq_len = Q_all.shape[0]
    n_q = min(n_queries, seq_len)
    if n_q <= 0:
        return {
            "entropy_full": 0.0,
            "entropy_no_sink_local": 0.0,
            f"top{top_k_for_mass}_mass_full": 1.0,
            f"top{top_k_for_mass}_mass_no_sink_local": 1.0,
        }

    start = max(0, seq_len - n_q)
    positions = list(range(start, seq_len))

    accum = {}
    for qpos in positions:
        stats = compute_query_stats(
            Q_all[qpos], K_all[:qpos + 1],
            head_dim, n_sink, local_window,
            top_k_for_mass=top_k_for_mass,
        )
        for k, v in stats.items():
            accum.setdefault(k, []).append(v)

    return {
        k: float(np.mean(v)) for k, v in accum.items()
    }


def select_heads_for_phase2(
    stats: Dict,
    metric: str = "entropy_no_sink_local",
) -> List[Tuple[int, int]]:
    """
    Select 3 heads: max, min, median entropy.

    stats: {layer_N: {head_M: {metric: value}}}
    Returns list of (layer, head) tuples.
    """
    entries = []
    for layer_key, heads in stats.items():
        layer = int(layer_key.split("_")[1])
        for head_key, head_stats in heads.items():
            head = int(head_key.split("_")[1])
            val = head_stats.get(metric, 0.0)
            entries.append((layer, head, val))

    if not entries:
        return []

    entries.sort(key=lambda x: x[2])
    selected = []
    # Min entropy (most concentrated)
    selected.append(
        (entries[0][0], entries[0][1])
    )
    # Median entropy
    mid = len(entries) // 2
    selected.append(
        (entries[mid][0], entries[mid][1])
    )
    # Max entropy (most diffuse)
    selected.append(
        (entries[-1][0], entries[-1][1])
    )
    return selected


def save_head_statistics(
    stats: Dict,
    out_path: Path,
) -> None:
    """Save head statistics JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_head_statistics(path: Path) -> Dict:
    """Load head statistics JSON."""
    with open(path) as f:
        return json.load(f)
