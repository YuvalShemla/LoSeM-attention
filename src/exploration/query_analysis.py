"""
Query analysis — mean-query scores and query deviation.

Compares attention logits from mean(Q) vs individual queries
and measures how far each query deviates from the mean.
"""

import numpy as np
from typing import Dict, List


def compute_meanquery_data(
    Q: np.ndarray,
    K: np.ndarray,
    head_dim: int,
    query_positions: List[int],
    n_queries_sample: int = 10,
    n_keys_sample: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Compare logits from mean(Q) vs individual queries.

    For a sample of queries, scatter mean-Q logits
    against actual-Q logits over sampled key positions.
    Returns flattened arrays and Pearson correlation.
    """
    queries = Q[query_positions]
    mean_q = queries.mean(axis=0)

    rng = np.random.default_rng(seed)
    n_q = min(n_queries_sample, len(query_positions))
    q_idx = rng.choice(
        len(query_positions), n_q, replace=False,
    )

    mean_scores_all = []
    full_scores_all = []

    for qi in q_idx:
        qpos = query_positions[qi]
        keys = K[:qpos + 1]
        n = len(keys)

        mean_logits = (
            (mean_q @ keys.T) / np.sqrt(head_dim)
        )
        full_logits = (
            (Q[qpos] @ keys.T) / np.sqrt(head_dim)
        )

        if n > n_keys_sample:
            kidx = rng.choice(
                n, n_keys_sample, replace=False,
            )
        else:
            kidx = np.arange(n)

        mean_scores_all.append(mean_logits[kidx])
        full_scores_all.append(full_logits[kidx])

    mean_scores = np.concatenate(mean_scores_all)
    full_scores = np.concatenate(full_scores_all)

    corr = 0.0
    if len(mean_scores) > 1:
        corr = float(
            np.corrcoef(mean_scores, full_scores)[0, 1]
        )

    return {
        "mean_scores": mean_scores,
        "full_scores": full_scores,
        "correlation": corr,
    }


def compute_query_deviation_data(
    Q: np.ndarray,
    query_positions: List[int],
) -> Dict:
    """
    Compute ||q - mean(Q)|| for each query position.

    Large deviations imply the mean query is a poor
    representative, favoring multi-centroid approaches.
    """
    queries = Q[query_positions]
    mean_q = queries.mean(axis=0)
    deviations = np.linalg.norm(
        queries - mean_q, axis=1,
    )
    q_norms = np.linalg.norm(queries, axis=1)

    return {
        "deviations": deviations,
        "mean_q_norm": float(np.linalg.norm(mean_q)),
        "q_norms": q_norms,
        "relative_deviations": (
            deviations / np.maximum(q_norms, 1e-10)
        ),
    }
