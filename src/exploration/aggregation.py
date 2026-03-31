"""
Aggregation of exploration data across heads and tasks.

Supports mean, median, percentile, and variance modes.
Variance mode returns special keys for spaghetti plots
and IQR bands.
"""

import numpy as np
from typing import Dict, List


def _agg_arrays(
    arrays: List[np.ndarray],
    method: str,
) -> np.ndarray:
    """Aggregate position-indexed arrays (same length)."""
    stacked = np.array(arrays)
    if method == "mean":
        return np.mean(stacked, axis=0)
    elif method == "median":
        return np.median(stacked, axis=0)
    elif method == "p25":
        return np.percentile(stacked, 25, axis=0)
    elif method == "p75":
        return np.percentile(stacked, 75, axis=0)
    elif method == "p90":
        return np.percentile(stacked, 90, axis=0)
    raise ValueError(f"Unknown method: {method}")


def _agg_scalars(
    values: List[float],
    method: str,
) -> float:
    """Aggregate scalar metrics."""
    arr = np.array(values)
    if method == "mean":
        return float(np.mean(arr))
    elif method == "median":
        return float(np.median(arr))
    elif method == "p25":
        return float(np.percentile(arr, 25))
    elif method == "p75":
        return float(np.percentile(arr, 75))
    elif method == "p90":
        return float(np.percentile(arr, 90))
    raise ValueError(f"Unknown method: {method}")


def _pool_samples(
    sample_lists: List[np.ndarray],
) -> np.ndarray:
    """Pool all samples from all heads into one array."""
    return np.concatenate(sample_lists)


def aggregate_global_data(
    all_head_data: List[Dict],
    method: str = "mean",
) -> Dict:
    """
    Aggregate global dashboard data across heads.

    Parameters
    ----------
    all_head_data : list of per-head global data dicts,
        each containing "concentration", "entropy", "bias",
        "meanquery", "query_deviation" keys.
    method : "mean", "median", "p25", "p75", "p90",
        or "variance".

    Returns
    -------
    Dict in the same format as per-head data so the same
    panel functions can render it.

    For method="variance", adds special keys:
      - "_all_heads" for spaghetti plots
      - "_spread" for IQR bands
    """
    if not all_head_data:
        return {}

    is_variance = method == "variance"
    agg_method = "mean" if is_variance else method

    result = {}

    # --- Concentration ---
    conc_all = [d["concentration"] for d in all_head_data]
    ref = conc_all[0]
    agg_conc = {
        "query_positions": ref["query_positions"],
        "top_k_values": ref["top_k_values"],
        "top_k_mass": {},
        "concentration_curves": ref["concentration_curves"],
    }

    for k in ref["top_k_values"]:
        arrays = [
            np.array(c["top_k_mass"][k]) for c in conc_all
        ]
        agg_conc["top_k_mass"][k] = list(
            _agg_arrays(arrays, agg_method)
        )
        if is_variance:
            agg_conc.setdefault("_all_heads_topk", {})[k] = [
                list(a) for a in arrays
            ]
            stacked = np.array(arrays)
            agg_conc.setdefault("_spread_topk", {})[k] = {
                "p25": list(np.percentile(stacked, 25, axis=0)),
                "p75": list(np.percentile(stacked, 75, axis=0)),
            }

    result["concentration"] = agg_conc

    # --- Entropy ---
    ent_all = [d["entropy"] for d in all_head_data]
    ref_ent = ent_all[0]
    agg_ent = {
        "positions": ref_ent["positions"],
    }
    for key in ["full_entropy", "nonlocal_entropy"]:
        arrays = [np.array(e[key]) for e in ent_all]
        agg_ent[key] = list(_agg_arrays(arrays, agg_method))
        if is_variance:
            agg_ent.setdefault("_all_heads", {})[key] = [
                list(a) for a in arrays
            ]
            stacked = np.array(arrays)
            agg_ent.setdefault("_spread", {})[key] = {
                "p25": list(np.percentile(stacked, 25, axis=0)),
                "p75": list(np.percentile(stacked, 75, axis=0)),
            }

    result["entropy"] = agg_ent

    # --- Bias ---
    bias_all = [d["bias"] for d in all_head_data]
    ref_bias = bias_all[0]
    agg_bias = {
        "budgets": ref_bias["budgets"],
    }
    for prefix in ["topk", "uniform", "oracle_sampling"]:
        key = f"{prefix}_value_error"
        arrays = [np.array(b[key]) for b in bias_all]
        agg_bias[key] = list(
            _agg_arrays(arrays, agg_method)
        )
        if is_variance:
            agg_bias.setdefault("_all_heads", {})[key] = [
                list(a) for a in arrays
            ]
            stacked = np.array(arrays)
            agg_bias.setdefault("_spread", {})[key] = {
                "p25": list(np.percentile(stacked, 25, axis=0)),
                "p75": list(np.percentile(stacked, 75, axis=0)),
            }

    result["bias"] = agg_bias

    # --- Mean-query ---
    mq_all = [d["meanquery"] for d in all_head_data]
    corrs = [m["correlation"] for m in mq_all]
    agg_mq = {
        "correlation": _agg_scalars(corrs, agg_method),
        "mean_scores": _pool_samples(
            [m["mean_scores"] for m in mq_all]
        ),
        "full_scores": _pool_samples(
            [m["full_scores"] for m in mq_all]
        ),
    }
    if is_variance:
        agg_mq["_all_heads_correlations"] = corrs

    result["meanquery"] = agg_mq

    # --- Query deviation ---
    qd_all = [d["query_deviation"] for d in all_head_data]
    agg_qd = {
        "deviations": _pool_samples(
            [q["deviations"] for q in qd_all]
        ),
        "mean_q_norm": _agg_scalars(
            [q["mean_q_norm"] for q in qd_all], agg_method,
        ),
        "q_norms": _pool_samples(
            [q["q_norms"] for q in qd_all]
        ),
        "relative_deviations": _pool_samples(
            [q["relative_deviations"] for q in qd_all]
        ),
    }
    if is_variance:
        agg_qd["_all_heads_deviations"] = [
            q["deviations"] for q in qd_all
        ]
        agg_qd["_all_heads_mean_q_norm"] = [
            q["mean_q_norm"] for q in qd_all
        ]

    result["query_deviation"] = agg_qd

    # --- Embedding projections (small multiples) ---
    embeddings = [
        d.get("embedding") for d in all_head_data
    ]
    valid = [e for e in embeddings if e is not None]
    if valid:
        result["embedding"] = {
            "_small_multiples": valid,
        }
    else:
        result["embedding"] = None

    return result


def aggregate_pairwise_data(
    all_head_data: List[Dict],
    method: str = "mean",
) -> Dict:
    """
    Aggregate pairwise dashboard data across heads.

    Parameters
    ----------
    all_head_data : list of per-head pairwise data dicts,
        each containing "qk", "qq", "kk" keys.
    method : "mean", "median", "p25", "p75", "p90",
        or "variance".

    Returns
    -------
    Dict in the same format as per-head data.
    """
    if not all_head_data:
        return {}

    is_variance = method == "variance"
    agg_method = "mean" if is_variance else method
    result = {}

    for pair_type in ["qk", "qq", "kk"]:
        pt_all = [d[pair_type] for d in all_head_data]
        ref = pt_all[0]

        # Pool all samples
        all_cosines = _pool_samples(
            [p["cosine_all"] for p in pt_all]
        )
        all_dots = _pool_samples(
            [p["dots_all"] for p in pt_all]
        )
        all_distances = _pool_samples(
            [p["distances"] for p in pt_all]
        )
        all_is_sink = _pool_samples(
            [p["is_sink"] for p in pt_all]
        )

        local_window = ref.get("local_window", 1024)

        # Recompute sink/local/global from pooled data
        local_mask = ~all_is_sink & (
            all_distances <= local_window
        )
        global_mask = ~all_is_sink & (
            all_distances > local_window
        )

        agg = {
            "pair_type": pair_type,
            "cosine_all": all_cosines,
            "dots_all": all_dots,
            "distances": all_distances,
            "is_sink": all_is_sink,
            "local_window": local_window,
            # Sink / local / global from pooled
            "cosine_sink": all_cosines[all_is_sink],
            "cosine_local": all_cosines[local_mask],
            "cosine_global": all_cosines[global_mask],
            "dots_sink": all_dots[all_is_sink],
            "dots_local": all_dots[local_mask],
            "dots_global": all_dots[global_mask],
        }

        # Binned data — average across heads
        for metric in ["cosine", "dots"]:
            centers_key = (
                "distance_bin_centers"
                if metric == "cosine"
                else "dots_bin_centers"
            )
            mean_key = f"{metric}_binned_mean"
            sem_key = f"{metric}_binned_sem"

            agg[centers_key] = ref[centers_key]

            all_means = []
            for p in pt_all:
                m = p[mean_key]
                if len(m) == len(ref[centers_key]):
                    all_means.append(m)
            if all_means:
                stacked = np.array(all_means)
                agg[mean_key] = np.nanmean(
                    stacked, axis=0,
                )
                agg[sem_key] = np.nanstd(
                    stacked, axis=0,
                ) / np.sqrt(len(all_means))

                if is_variance:
                    agg.setdefault("_all_heads_binned", {})[
                        mean_key
                    ] = [list(a) for a in all_means]
            else:
                agg[mean_key] = ref[mean_key]
                agg[sem_key] = ref[sem_key]

        if is_variance:
            agg["_all_heads_cosine"] = [
                p["cosine_all"] for p in pt_all
            ]
            agg["_all_heads_dots"] = [
                p["dots_all"] for p in pt_all
            ]

        result[pair_type] = agg

    return result
