"""
Per-query evaluation and result aggregation.

Separated from run_evaluation.py to keep each file focused.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..algorithms.base import AttentionInput
from ..core import (
    full_attention, compute_special_indices,
    relative_l2_error, stats_from_weights,
)


def evaluate_query(
    q: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    methods: list,
    budgets: List[int],
    head_dim: int,
    n_sink: int,
    local_window: int,
    rng: np.random.Generator,
    compute_statistics: bool = False,
    compute_group_cosine_distribution: bool = False,
) -> Dict:
    """Evaluate one query across all methods."""
    n_causal = len(keys)

    full_out, logits, weights = full_attention(
        q, keys, values, head_dim,
    )

    sp_idx, cand_idx = compute_special_indices(
        n_causal, n_sink, local_window,
    )

    problem = AttentionInput(
        query=q, keys=keys, values=values,
        head_dim=head_dim, logits=logits,
        special_idx=sp_idx, candidate_idx=cand_idx,
    )

    results = {}
    if compute_group_cosine_distribution:
        results["_query_metrics"] = _query_logit_sums(logits)
    for m in methods:
        if m.sweeps_budget:
            for b in budgets:
                out = m.run(problem, b, rng)
                err = relative_l2_error(
                    out.output, full_out,
                )
                k = f"{m.name}-{b}"
                results[k] = {
                    "error": err,
                    "budget": out.actual_budget,
                    "requested_budget": int(b),
                }
                if out.debug_payload is not None:
                    results[k]["debug_payload"] = out.debug_payload
                if compute_group_cosine_distribution:
                    gc = _group_cosines(
                        problem.keys,
                        logits,
                        values,
                        out.grouped_member_indices,
                    )
                    if gc is not None:
                        gc["value_softmax_mismatch_ratio"] = (
                            _softmax_value_mismatch_ratio(
                                values, logits,
                                out.grouped_member_indices,
                            )
                        )
                        gc["value_mismatch_ratio_znorm"] = (
                            _softmax_value_mismatch_ratio_znorm(
                                values, logits,
                                out.grouped_member_indices,
                            )
                        )
                        gc_key = "_group_cosines"
                        if gc_key not in results:
                            results[gc_key] = {}
                        results[gc_key][k] = gc
        else:
            out = m.run(problem, 0, rng)
            err = relative_l2_error(
                out.output, full_out,
            )
            results[m.name] = {
                "error": err,
                "budget": out.actual_budget,
                "requested_budget": 0,
            }
            if out.debug_payload is not None:
                results[m.name]["debug_payload"] = out.debug_payload
            if compute_group_cosine_distribution:
                gc = _group_cosines(
                    problem.keys,
                    logits,
                    values,
                    out.grouped_member_indices,
                )
                if gc is not None:
                    gc["value_softmax_mismatch_ratio"] = (
                        _softmax_value_mismatch_ratio(
                            values, logits,
                            out.grouped_member_indices,
                        )
                    )
                    gc["value_mismatch_ratio_znorm"] = (
                        _softmax_value_mismatch_ratio_znorm(
                            values, logits,
                            out.grouped_member_indices,
                        )
                    )
                    gc_key = "_group_cosines"
                    if gc_key not in results:
                        results[gc_key] = {}
                    results[gc_key][m.name] = gc

    if compute_statistics:
        results["_query_stats"] = stats_from_weights(
            weights, n_sink, local_window,
        )

    return results


def _stable_sum_exp(a: np.ndarray) -> float:
    """Z = sum_i exp(a_i), stable in log-domain."""
    x = np.asarray(a, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    m = float(np.max(x))
    return float(np.exp(m) * np.sum(np.exp(x - m)))


def _bar_logit_vector(
    logits: np.ndarray,
    grouped_member_indices,
) -> np.ndarray:
    """
    Per position i: bar_ell_i = mean_{j in group(i)} ell_j, or ell_i if
    i is not in any listed group.
    """
    rep = logits.astype(np.float64).copy()
    for idx in grouped_member_indices:
        if idx is None or len(idx) == 0:
            continue
        rep[idx] = float(np.mean(logits[idx]))
    return rep


def _query_logit_sums(logits: np.ndarray) -> Dict[str, float]:
    """Softmax partition Z = sum_i exp(ell_i)."""
    lg = logits.astype(np.float64, copy=False)
    return {
        "sum_exp_logits": _stable_sum_exp(lg),
    }


def _softmax_value_mismatch_ratio(
    values: np.ndarray,
    logits: np.ndarray,
    grouped_member_indices,
) -> float:
    """
    ||o* - o_bar|| / ||o*||  where

      o* = (1/Z) sum_i exp(ell_i) v_i,
      o_bar = (1/Z_bar) sum_i exp(bar_ell_i) v_i,

    with bar_ell_i the mean logit in i's group (else ell_i).
    """
    eps = 1e-30
    if not grouped_member_indices:
        return float("nan")
    v = values.astype(np.float64)
    ell = logits.astype(np.float64)
    bar_ell = _bar_logit_vector(logits, grouped_member_indices)
    z = _stable_sum_exp(ell)
    z_bar = _stable_sum_exp(bar_ell)
    if z <= 0.0 or z_bar <= 0.0:
        return float("nan")
    w = np.exp(ell - np.max(ell))
    w /= np.sum(w)
    w_b = np.exp(bar_ell - np.max(bar_ell))
    w_b /= np.sum(w_b)
    o_star = (w[:, None] * v).sum(axis=0)
    o_bar = (w_b[:, None] * v).sum(axis=0)
    d = o_star - o_bar
    num = float(np.linalg.norm(d))
    den = float(np.linalg.norm(o_star))
    if den <= eps:
        return float("nan")
    return num / den


def _softmax_value_mismatch_ratio_znorm(
    values: np.ndarray,
    logits: np.ndarray,
    grouped_member_indices,
) -> float:
    """
    ||o* - o_bar_Z|| / ||o*||  where both use Z for normalization:

      o*      = (1/Z) sum_i exp(ell_i) v_i,
      o_bar_Z = (1/Z) sum_i exp(bar_ell_i) v_i.
    """
    eps = 1e-30
    if not grouped_member_indices:
        return float("nan")
    v = values.astype(np.float64)
    ell = logits.astype(np.float64)
    bar_ell = _bar_logit_vector(logits, grouped_member_indices)
    z = _stable_sum_exp(ell)
    if z <= 0.0:
        return float("nan")
    w = np.exp(ell - np.max(ell))
    w /= np.sum(w)
    w_bz = np.exp(bar_ell)
    w_bz /= z
    o_star = (w[:, None] * v).sum(axis=0)
    o_bar_z = (w_bz[:, None] * v).sum(axis=0)
    d = o_star - o_bar_z
    num = float(np.linalg.norm(d))
    den = float(np.linalg.norm(o_star))
    if den <= eps:
        return float("nan")
    return num / den


def _group_output_l2_err_sq(
    values: np.ndarray,
    logits: np.ndarray,
    idx: np.ndarray,
    bar_ell_g: float,
) -> float:
    """
    || sum_{i in group} v_i (e^{l_i} - e^{bar l_g}) ||_2^2 in value space.
    """
    v = values[idx].astype(np.float64, copy=False)
    lg = logits[idx].astype(np.float64, copy=False)
    e_bar = np.exp(bar_ell_g)
    delta = np.exp(lg) - e_bar
    vec = (v * delta[:, np.newaxis]).sum(axis=0)
    return float(np.dot(vec, vec))


def _group_cosines(
    keys: np.ndarray,
    logits: np.ndarray,
    values: np.ndarray,
    grouped_member_indices,
):
    """
    Flatten cos(key, group_mean_key) over all keys in all groups.

    Also computes per-group
      c_g = sum_{i in g} |e^{ell_i}/Z - e^{bar ell_g}/Z'|,
      with Z' = sum_g n_g * exp(bar ell_g),
    and total sum_g c_g (L1 difference between key-level distributions).
    Stores per-group values c_g * m and exp(ell_g) * m / Z where ell_g is the mean logit
    in group g (same index as the group mean key for cosines), and
    e_g * m / Z with e_g = || sum_{i in g} v_i (e^{l_i}-e^{bar l_g}) ||_2^2.

    Also returns per-token values e^{ell_i} * m / Z for the group that
    maximizes exp(ell_g) * m / Z among groups, plus that group's own
    scalar exp(ell_g) * m / Z.

    Returns None when the method did not expose grouped members.
    """
    if not grouped_member_indices:
        return None
    m_groups = int(len(grouped_member_indices))
    vals = []
    eps = 1e-12
    group_contribs = []
    eg_raw_list: List[float] = []
    bar_ell_per_group: List[float] = []
    group_sizes: List[int] = []
    for idx in grouped_member_indices:
        if idx is None or len(idx) == 0:
            continue
        lg = logits[idx].astype(np.float64, copy=False)
        bar_ell_g = float(np.mean(lg))
        bar_ell_per_group.append(bar_ell_g)
        group_sizes.append(int(len(idx)))
        eg_raw_list.append(
            _group_output_l2_err_sq(
                values, logits, idx, bar_ell_g,
            ),
        )
        group_contribs.append((idx, lg, bar_ell_g))
        gk = keys[idx]
        mean = np.mean(gk, axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm < eps:
            continue
        key_norm = np.linalg.norm(gk, axis=1)
        denom = np.maximum(key_norm * mean_norm, eps)
        cos = (gk @ mean) / denom
        cos = np.clip(cos, -1.0, 1.0)
        vals.append(cos.astype(np.float32, copy=False))
    if not vals:
        return None
    bar_vec = _bar_logit_vector(logits, grouped_member_indices)
    sum_exp_bar_logits = _stable_sum_exp(bar_vec)
    z = _stable_sum_exp(logits.astype(np.float64))
    z_den = max(z, 1e-300)
    z_bar = 0.0
    for (_, _, bar_ell_g), n_g in zip(group_contribs, group_sizes):
        z_bar += float(n_g) * float(np.exp(bar_ell_g))
    z_bar_den = max(float(z_bar), 1e-300)
    c_g_vals: List[float] = []
    c_g_znorm_vals: List[float] = []
    for _, lg, bar_ell_g in group_contribs:
        e_i_over_z = np.exp(lg) / z_den
        e_bar_over_zbar = float(np.exp(bar_ell_g)) / z_bar_den
        e_bar_over_z = float(np.exp(bar_ell_g)) / z_den
        c_g = float(np.sum(np.abs(e_i_over_z - e_bar_over_zbar)))
        c_g_znorm = float(np.sum(np.abs(e_i_over_z - e_bar_over_z)))
        c_g_vals.append(c_g)
        c_g_znorm_vals.append(c_g_znorm)
    exp_residual_l1 = float(np.sum(c_g_vals))
    exp_residual_l1_znorm = float(np.sum(c_g_znorm_vals))
    zm = max(m_groups, 1)
    scale_czm = float(zm) / z_den
    max_group_exp_logits_m_over_z = np.array([], dtype=np.float32)
    max_group_top_logits = []
    max_group_exp_lg_m_over_z = None
    p75_group_l1_cg_m_over_z = None
    scale_cm = float(zm)
    token_profile_key_probs = np.array([], dtype=np.float32)
    token_profile_group_probs = np.array([], dtype=np.float32)
    token_profile_group_probs_over_z = np.array([], dtype=np.float32)
    token_profile_group_boundaries: List[int] = []
    if group_contribs:
        contribs = c_g_vals
        order = np.argsort(np.array(contribs, dtype=np.float64))
        n_ord = len(order)
        p75_pos = int(np.floor(0.75 * (n_ord - 1))) if n_ord > 0 else 0
        p75_entry = group_contribs[int(order[p75_pos])]
        p75_midx = p75_entry[0]
        c_p75 = float(contribs[int(order[p75_pos])])
        p75_group_l1_cg_m_over_z = float(c_p75 * scale_cm)
        lg_m = logits[p75_midx].astype(np.float64, copy=False)
        bar_g = float(np.mean(lg_m))
        e_bar_m = np.exp(bar_g)
        e_im = np.exp(lg_m)
        p75_group_z_norm_residuals = (
            ((e_im - e_bar_m) / z_den * zm).astype(np.float32)
        )
    if group_contribs:
        group_order = sorted(
            range(len(group_contribs)),
            key=lambda gi: float(np.exp(group_contribs[gi][2])),
            reverse=True,
        )
        prof_key_parts: List[np.ndarray] = []
        prof_group_parts: List[np.ndarray] = []
        prof_group_over_z_parts: List[np.ndarray] = []
        running = 0
        for gi in group_order:
            idx, lg_g, bar_ell_g = group_contribs[gi]
            p_key = np.exp(lg_g) / z_den
            sort_idx = np.argsort(-p_key)
            p_key = p_key[sort_idx]
            p_group = np.full(
                len(p_key),
                float(np.exp(bar_ell_g)) / z_bar_den,
                dtype=np.float64,
            )
            p_group_over_z = np.full(
                len(p_key),
                float(np.exp(bar_ell_g)) / z_den,
                dtype=np.float64,
            )
            prof_key_parts.append(p_key.astype(np.float32))
            prof_group_parts.append(p_group.astype(np.float32))
            prof_group_over_z_parts.append(
                p_group_over_z.astype(np.float32),
            )
            running += len(p_key)
            token_profile_group_boundaries.append(running)
        if prof_key_parts:
            token_profile_key_probs = np.concatenate(prof_key_parts)
            token_profile_group_probs = np.concatenate(prof_group_parts)
            token_profile_group_probs_over_z = np.concatenate(
                prof_group_over_z_parts,
            )
    if grouped_member_indices:
        max_idx = None
        max_exp_lg = -np.inf
        for idx in grouped_member_indices:
            if idx is None or len(idx) == 0:
                continue
            lg_g = logits[idx].astype(np.float64, copy=False)
            ell_g = float(np.mean(lg_g))
            exp_lg = float(np.exp(ell_g))
            if exp_lg > max_exp_lg:
                max_exp_lg = exp_lg
                max_idx = idx
        if max_idx is not None:
            lg_max = logits[max_idx].astype(np.float64, copy=False)
            ell_g_max = float(np.mean(lg_max))
            max_group_exp_lg_m_over_z = float(
                np.exp(ell_g_max) / z_den * zm
            )
            max_group_exp_logits_m_over_z = (
                (np.exp(lg_max) / z_den * zm).astype(np.float32)
            )
            max_group_top_logits = sorted(
                [float(v) for v in lg_max.tolist()],
                reverse=True,
            )[:100]
    group_l1_cg_m_over_z = np.array(
        [float(c) * scale_cm for c in c_g_vals],
        dtype=np.float32,
    )
    group_l1_cg_m_znorm = np.array(
        [float(c) * scale_cm for c in c_g_znorm_vals],
        dtype=np.float32,
    )
    group_out_l2_err_sq_m_over_z = np.array(
        [float(e) * (float(zm) / z_den) for e in eg_raw_list],
        dtype=np.float32,
    )
    group_exp_lg_m_over_z = np.array(
        [
            float(np.exp(be)) * scale_czm
            for be in bar_ell_per_group
        ],
        dtype=np.float32,
    )
    return {
        "cosines": np.concatenate(vals),
        "n_groups": m_groups,
        "exp_residual_l1_over_z": exp_residual_l1,
        "exp_residual_l1_znorm": exp_residual_l1_znorm,
        "sum_exp_bar_logits": sum_exp_bar_logits,
        "p75_group_z_norm_residuals": p75_group_z_norm_residuals,
        "max_group_exp_logits_m_over_z": max_group_exp_logits_m_over_z,
        "max_group_exp_lg_m_over_z": max_group_exp_lg_m_over_z,
        "max_group_top_logits": max_group_top_logits,
        "group_l1_cg_m_over_z": group_l1_cg_m_over_z,
        "group_l1_cg_m_znorm": group_l1_cg_m_znorm,
        "group_sizes": np.asarray(group_sizes, dtype=np.int32),
        "group_out_l2_err_sq_m_over_z": group_out_l2_err_sq_m_over_z,
        "group_exp_lg_m_over_z": group_exp_lg_m_over_z,
        "p75_group_l1_cg_m_over_z": p75_group_l1_cg_m_over_z,
        "token_profile_key_probs": token_profile_key_probs,
        "token_profile_group_probs": token_profile_group_probs,
        "token_profile_group_probs_over_z": token_profile_group_probs_over_z,
        "token_profile_group_boundaries": np.asarray(
            token_profile_group_boundaries, dtype=np.int32,
        ),
    }


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Compute mean/std per method key."""
    all_keys = set()
    for qr in all_results:
        all_keys.update(qr.keys())

    agg = {}
    for key in sorted(all_keys):
        if key.startswith("_"):
            continue
        entries = [
            qr[key] for qr in all_results
            if key in qr
        ]
        if not entries:
            continue
        errors = [e["error"] for e in entries]
        budgets = [e["budget"] for e in entries]
        agg[key] = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "budget_mean": float(np.mean(budgets)),
            "budget_std": float(np.std(budgets)),
            "n_queries": len(entries),
        }
    return agg


PERCENTILE_WEIGHTS = {0: 1, 25: 2, 50: 3, 75: 2, 100: 1}


def weighted_aggregate_heads(
    per_head_aggs: Dict[int, Dict],
    head_meta: list,
) -> Dict:
    """
    Weighted aggregate across heads by percentile.

    Triangular weighting: p50 gets 3x the weight of
    p0/p100. Falls back to equal weights if percentile
    metadata is missing.
    """
    weights = []
    aggs = []
    for idx, info in per_head_aggs.items():
        meta = head_meta[idx] if head_meta else {}
        pct = meta.get("percentile")
        w = PERCENTILE_WEIGHTS.get(pct, 1)
        weights.append(w)
        aggs.append(info["agg"])

    all_keys = set()
    for a in aggs:
        all_keys.update(a.keys())

    total_w = sum(weights)
    result = {}
    for key in sorted(all_keys):
        present = [
            (a[key], w)
            for a, w in zip(aggs, weights)
            if key in a
        ]
        if not present:
            continue
        w_sum = sum(w for _, w in present)
        err_mean = sum(
            e["error_mean"] * w for e, w in present
        ) / w_sum
        bud_mean = sum(
            e["budget_mean"] * w for e, w in present
        ) / w_sum
        err_std = sum(
            e["error_std"] * w for e, w in present
        ) / w_sum
        bud_std = sum(
            e["budget_std"] * w for e, w in present
        ) / w_sum
        n_total = sum(
            e["n_queries"] for e, _ in present
        )
        result[key] = {
            "error_mean": float(err_mean),
            "error_std": float(err_std),
            "budget_mean": float(bud_mean),
            "budget_std": float(bud_std),
            "n_queries": n_total,
            "weighting": "percentile_triangular",
        }
    return result


def aggregate_query_stats(
    all_results: List[Dict],
) -> Dict[str, float]:
    """
    Aggregate per-query attention statistics
    into mean/std summaries.

    Returns empty dict if no stats were computed.
    """
    stat_entries = [
        qr["_query_stats"] for qr in all_results
        if "_query_stats" in qr
    ]
    if not stat_entries:
        return {}

    accum = {}
    for entry in stat_entries:
        for k, v in entry.items():
            accum.setdefault(k, []).append(v)

    agg = {}
    for k, vals in accum.items():
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


GROUP_L1_HIST_X_MIN = 1e-6


def _aggregate_group_l1_histograms(
    gl1_vals: np.ndarray,
    p75_gl1_arr: Optional[np.ndarray],
    n_bins: int,
) -> Optional[Dict]:
    """
    Histogram support stats for c_g*m with log-spaced bins.

    Also stores a cumulative contribution curve where y(x) is the share of
    total L1 contribution coming from groups with c_g*m <= x.
    The x-range is floored at GROUP_L1_HIST_X_MIN (values below are clipped
    for binning). x-axis is log scale in plots.
    """
    if gl1_vals is None or len(gl1_vals) == 0:
        return None
    gl1_vals = np.asarray(gl1_vals, dtype=np.float64)
    x_floor = float(GROUP_L1_HIST_X_MIN)
    pool_parts: List[np.ndarray] = [gl1_vals]
    if p75_gl1_arr is not None and len(p75_gl1_arr) > 0:
        pool_parts.append(
            np.asarray(p75_gl1_arr, dtype=np.float64),
        )
    pool = np.concatenate(pool_parts)
    lo_raw = float(np.min(pool))
    hi_raw = float(np.max(pool))
    lo_g = max(lo_raw, x_floor)
    hi_g = max(hi_raw, lo_g * (1.0 + 1e-12))
    if hi_g <= lo_g:
        hi_g = lo_g * (1.0 + 1e-12)
    edges = np.logspace(
        np.log10(lo_g), np.log10(hi_g), int(max(n_bins, 1)) + 1,
    )
    gl1_binned = np.maximum(gl1_vals, x_floor)
    g_counts, _ = np.histogram(gl1_binned, bins=edges)
    g_counts = g_counts.astype(np.float64)
    out: Dict = {
        "group_l1_contrib_histogram": {
            "bin_edges": edges.tolist(),
            "counts": g_counts.tolist(),
        },
        "n_group_l1_contrib_values": int(len(gl1_vals)),
        "group_l1_hist_meta": {
            "global_min": lo_raw,
            "global_max": hi_raw,
            "x_min_floor": x_floor,
            "log_scale_x": True,
            "n_bins": int(n_bins),
            "p50": float(np.quantile(gl1_vals, 0.50)),
            "p75": float(np.quantile(gl1_vals, 0.75)),
            "p90": float(np.quantile(gl1_vals, 0.90)),
        },
    }
    g_weighted, _ = np.histogram(
        gl1_binned,
        bins=edges,
        weights=gl1_vals,
    )
    g_weighted = g_weighted.astype(np.float64)
    g_total_w = float(np.sum(g_weighted))
    if g_total_w > 0.0:
        g_cum_share = np.cumsum(g_weighted) / g_total_w
    else:
        g_cum_share = np.zeros_like(g_weighted)
    out["group_l1_contrib_cumulative"] = {
        "bin_edges": edges.tolist(),
        "cum_share": g_cum_share.tolist(),
        "cum_weight": np.cumsum(g_weighted).tolist(),
        "total_weight": g_total_w,
    }
    if p75_gl1_arr is not None and len(p75_gl1_arr) > 0:
        p75_gl1_arr = np.asarray(p75_gl1_arr, dtype=np.float64)
        p75_binned = np.maximum(p75_gl1_arr, x_floor)
        r_counts, _ = np.histogram(p75_binned, bins=edges)
        r_counts = r_counts.astype(np.float64)
        out["group_l1_p75_overlay_histogram"] = {
            "bin_edges": edges.tolist(),
            "counts": r_counts.tolist(),
        }
    return out


def aggregate_group_cosines(
    all_results: List[Dict],
    n_bins: int = 50,
) -> Dict[str, Dict]:
    """
    Aggregate per-query group-cosine arrays into histograms per method.
    """
    by_method = {}
    by_method_n_groups: Dict[str, List[int]] = {}
    by_method_sse = {}
    by_method_p75z: Dict[str, List[np.ndarray]] = {}
    by_method_gl1: Dict[str, List[np.ndarray]] = {}
    by_method_gl1_znorm: Dict[str, List[np.ndarray]] = {}
    by_method_p75_gl1: Dict[str, List[float]] = {}
    by_method_cgx: Dict[str, List[np.ndarray]] = {}
    by_method_egy: Dict[str, List[np.ndarray]] = {}
    by_method_token_key: Dict[str, List[np.ndarray]] = {}
    by_method_token_group: Dict[str, List[np.ndarray]] = {}
    by_method_token_group_over_z: Dict[str, List[np.ndarray]] = {}
    by_method_token_boundaries: Dict[str, List[np.ndarray]] = {}
    by_method_max_exp_lg: Dict[str, List[float]] = {}
    by_method_sum_exp = {}
    by_method_zbar = {}
    by_method_val_l2 = {}
    by_method_sse_znorm: Dict[str, List[float]] = {}
    by_method_val_l2_znorm: Dict[str, List[float]] = {}
    for qr in all_results:
        payload = qr.get("_group_cosines")
        if not payload:
            continue
        qm = qr.get("_query_metrics", {})
        se = qm.get("sum_exp_logits")
        for method_key, entry in payload.items():
            if isinstance(entry, dict):
                cos = entry.get("cosines")
                n_groups_q = entry.get("n_groups")
                sse = entry.get("exp_residual_l1_over_z")
                if sse is None:
                    sse = entry.get("exp_residual_sse_over_z2")
                if sse is None:
                    sse = entry.get("exp_residual_sse_sum")
                if sse is None:
                    sse = entry.get("logit_residual_sse_sum")
                if sse is None:
                    sse = entry.get("logit_within_group_var_sum")
                zb = entry.get("sum_exp_bar_logits")
                vls = entry.get("value_softmax_mismatch_ratio")
                if vls is None:
                    vls = entry.get("value_logit_group_l2_sq")
                vls_znorm = entry.get("value_mismatch_ratio_znorm")
                sse_znorm = entry.get("exp_residual_l1_znorm")
                p75z = entry.get("max_group_exp_logits_m_over_z")
                if p75z is None:
                    p75z = entry.get("p75_group_z_norm_residuals")
                if p75z is None:
                    p75z = entry.get(
                        "median_group_z_norm_residuals",
                    )
                max_exp_lg = entry.get("max_group_exp_lg_m_over_z")
                gl1 = entry.get("group_l1_cg_m_over_z")
                gl1_znorm = entry.get("group_l1_cg_m_znorm")
                p75_gl1 = entry.get("p75_group_l1_cg_m_over_z")
                if p75_gl1 is None:
                    p75_gl1 = entry.get(
                        "median_group_l1_cg_m_over_z",
                    )
            else:
                # Backward compatibility for early format.
                cos = entry
                n_groups_q = None
                sse = None
                sse_znorm = None
                zb = None
                vls = None
                vls_znorm = None
                p75z = None
                max_exp_lg = None
                gl1 = None
                gl1_znorm = None
                p75_gl1 = None
            if cos is None or len(cos) == 0:
                continue
            by_method.setdefault(method_key, []).append(cos)
            if n_groups_q is not None:
                by_method_n_groups.setdefault(
                    method_key, [],
                ).append(int(n_groups_q))
            if sse is not None:
                by_method_sse.setdefault(
                    method_key, [],
                ).append(float(sse))
            if sse_znorm is not None:
                by_method_sse_znorm.setdefault(
                    method_key, [],
                ).append(float(sse_znorm))
            if se is not None:
                by_method_sum_exp.setdefault(
                    method_key, [],
                ).append(float(se))
            if zb is not None:
                by_method_zbar.setdefault(
                    method_key, [],
                ).append(float(zb))
            if vls is not None and not np.isnan(vls):
                by_method_val_l2.setdefault(
                    method_key, [],
                ).append(float(vls))
            if vls_znorm is not None and not np.isnan(vls_znorm):
                by_method_val_l2_znorm.setdefault(
                    method_key, [],
                ).append(float(vls_znorm))
            if p75z is not None and len(p75z) > 0:
                by_method_p75z.setdefault(
                    method_key, [],
                ).append(
                    np.asarray(p75z, dtype=np.float64),
                )
            if (
                max_exp_lg is not None
                and np.isfinite(float(max_exp_lg))
            ):
                by_method_max_exp_lg.setdefault(
                    method_key, [],
                ).append(float(max_exp_lg))
            if gl1 is not None and len(gl1) > 0:
                gl1_a = np.asarray(gl1, dtype=np.float64)
                by_method_gl1.setdefault(
                    method_key, [],
                ).append(gl1_a)
            if gl1_znorm is not None and len(gl1_znorm) > 0:
                gl1z_a = np.asarray(gl1_znorm, dtype=np.float64)
                by_method_gl1_znorm.setdefault(
                    method_key, [],
                ).append(gl1z_a)
            if isinstance(entry, dict):
                cg_sc = entry.get("group_l1_cg_m_over_z")
                ng_sc = entry.get("group_sizes")
                exp_sc = entry.get(
                    "group_exp_lg_m_over_z",
                )
                tkp = entry.get("token_profile_key_probs")
                tgp = entry.get("token_profile_group_probs")
                tgz = entry.get("token_profile_group_probs_over_z")
                tgb = entry.get("token_profile_group_boundaries")
            else:
                cg_sc = None
                ng_sc = None
                exp_sc = None
                tkp = None
                tgp = None
                tgz = None
                tgb = None
            if (
                cg_sc is not None
                and ng_sc is not None
                and exp_sc is not None
            ):
                cg_a = np.asarray(cg_sc, dtype=np.float64)
                ng_a = np.asarray(ng_sc, dtype=np.float64)
                exp_a = np.asarray(exp_sc, dtype=np.float64)
                if (
                    len(cg_a) == len(ng_a)
                    and len(cg_a) == len(exp_a)
                    and len(cg_a) > 0
                ):
                    x_a = cg_a / np.maximum(ng_a, 1.0)
                    by_method_cgx.setdefault(
                        method_key, [],
                    ).append(x_a)
                    by_method_egy.setdefault(
                        method_key, [],
                    ).append(exp_a)
            if (
                tkp is not None
                and tgp is not None
                and tgb is not None
            ):
                tkp_a = np.asarray(tkp, dtype=np.float64)
                tgp_a = np.asarray(tgp, dtype=np.float64)
                tgz_a = np.asarray(tgz, dtype=np.float64) if tgz is not None else None
                tgb_a = np.asarray(tgb, dtype=np.int32)
                if (
                    len(tkp_a) == len(tgp_a)
                    and tgz_a is not None
                    and len(tkp_a) == len(tgz_a)
                    and len(tkp_a) > 0
                    and (len(tgb_a) == 0 or int(tgb_a[-1]) == len(tkp_a))
                ):
                    by_method_token_key.setdefault(
                        method_key, [],
                    ).append(tkp_a)
                    by_method_token_group.setdefault(
                        method_key, [],
                    ).append(tgp_a)
                    by_method_token_group_over_z.setdefault(
                        method_key, [],
                    ).append(tgz_a)
                    by_method_token_boundaries.setdefault(
                        method_key, [],
                    ).append(tgb_a)
            if (
                p75_gl1 is not None
                and np.isfinite(float(p75_gl1))
            ):
                by_method_p75_gl1.setdefault(
                    method_key, [],
                ).append(float(p75_gl1))

    out = {}
    for method_key, chunks in by_method.items():
        vals = np.concatenate(chunks)
        counts, edges = np.histogram(
            vals, bins=n_bins, range=(-1.0, 1.0)
        )
        row = {
            "n_values": int(len(vals)),
            "cos_mean": float(np.mean(vals)),
            "cos_std": float(np.std(vals)),
            "quantiles": {
                "p05": float(np.quantile(vals, 0.05)),
                "p25": float(np.quantile(vals, 0.25)),
                "p50": float(np.quantile(vals, 0.50)),
                "p75": float(np.quantile(vals, 0.75)),
                "p95": float(np.quantile(vals, 0.95)),
            },
            "histogram": {
                "bin_edges": edges.tolist(),
                "counts": counts.tolist(),
            },
        }
        p75z_chunks = by_method_p75z.get(method_key, [])
        if p75z_chunks:
            p75z_vals = np.concatenate(p75z_chunks)
            lo = float(np.min(p75z_vals))
            hi = float(np.max(p75z_vals))
            if hi <= lo:
                lo, hi = lo - 1e-12, hi + 1e-12
            p75z_counts, p75z_edges = np.histogram(
                p75z_vals, bins=n_bins, range=(lo, hi),
            )
            row["p75_z_norm_histogram"] = {
                "bin_edges": p75z_edges.tolist(),
                "counts": p75z_counts.tolist(),
            }
            row["max_group_exp_logits_histogram"] = {
                "bin_edges": p75z_edges.tolist(),
                "counts": p75z_counts.tolist(),
            }
            max_exp_lg_vals = by_method_max_exp_lg.get(
                method_key, [],
            )
            if max_exp_lg_vals:
                max_exp_lg_counts, _ = np.histogram(
                    np.asarray(max_exp_lg_vals, dtype=np.float64),
                    bins=p75z_edges,
                )
                row["max_group_exp_lg_histogram"] = {
                    "bin_edges": p75z_edges.tolist(),
                    "counts": max_exp_lg_counts.tolist(),
                }
            row["n_p75_z_norm_values"] = int(len(p75z_vals))
        gl1_chunks = by_method_gl1.get(method_key, [])
        p75_gl1_list = by_method_p75_gl1.get(method_key, [])
        if gl1_chunks:
            gl1_vals = np.concatenate(gl1_chunks)
            p75_gl1_arr = (
                np.array(p75_gl1_list, dtype=np.float64)
                if p75_gl1_list
                else None
            )
            hpack = _aggregate_group_l1_histograms(
                gl1_vals, p75_gl1_arr, n_bins,
            )
            if hpack:
                row.update(hpack)
        gl1z_chunks = by_method_gl1_znorm.get(method_key, [])
        if gl1z_chunks:
            gl1z_vals = np.concatenate(gl1z_chunks)
            row["group_l1_znorm_meta"] = {
                "p50": float(np.quantile(gl1z_vals, 0.50)),
                "p75": float(np.quantile(gl1z_vals, 0.75)),
                "p90": float(np.quantile(gl1z_vals, 0.90)),
            }
        cgx_chunks = by_method_cgx.get(method_key, [])
        egy_chunks = by_method_egy.get(method_key, [])
        if cgx_chunks and egy_chunks:
            xs = np.concatenate(cgx_chunks)
            ys = np.concatenate(egy_chunks)
            if len(xs) == len(ys) and len(xs) > 0:
                row["cg_eg_scatter"] = {
                    "x": xs.tolist(),
                    "y": ys.tolist(),
                    "n_points": int(len(xs)),
                }
        token_key_chunks = by_method_token_key.get(method_key, [])
        token_group_chunks = by_method_token_group.get(method_key, [])
        token_group_over_z_chunks = by_method_token_group_over_z.get(method_key, [])
        token_boundary_chunks = by_method_token_boundaries.get(
            method_key, [],
        )
        if (
            token_key_chunks
            and len(token_key_chunks) == len(token_group_chunks)
            and len(token_key_chunks) == len(token_group_over_z_chunks)
            and len(token_key_chunks) == len(token_boundary_chunks)
        ):
            key_parts: List[np.ndarray] = []
            group_parts: List[np.ndarray] = []
            group_over_z_parts: List[np.ndarray] = []
            boundaries: List[int] = []
            running = 0
            for kp, gp, gz, bd in zip(
                token_key_chunks,
                token_group_chunks,
                token_group_over_z_chunks,
                token_boundary_chunks,
            ):
                if len(kp) != len(gp) or len(kp) != len(gz) or len(kp) == 0:
                    continue
                key_parts.append(kp)
                group_parts.append(gp)
                group_over_z_parts.append(gz)
                bd_i = np.asarray(bd, dtype=np.int32)
                if len(bd_i) > 0:
                    boundaries.extend((bd_i + running).tolist())
                running += len(kp)
            if key_parts and group_parts and group_over_z_parts:
                row["group_token_probability_profile"] = {
                    "key_probs": np.concatenate(key_parts).tolist(),
                    "group_probs": np.concatenate(group_parts).tolist(),
                    "group_probs_over_z": np.concatenate(group_over_z_parts).tolist(),
                    "group_boundaries": boundaries,
                }
        sse_list = by_method_sse.get(method_key, [])
        if sse_list:
            sm = float(np.mean(sse_list))
            ss = float(np.std(sse_list))
            row["exp_residual_l1_over_z_mean"] = sm
            row["exp_residual_l1_over_z_std"] = ss
            row["exp_residual_sse_over_z2_mean"] = sm
            row["exp_residual_sse_over_z2_std"] = ss
            row["exp_residual_sse_sum_mean"] = sm
            row["exp_residual_sse_sum_std"] = ss
            row["logit_residual_sse_sum_mean"] = sm
            row["logit_residual_sse_sum_std"] = ss
        sse_z_list = by_method_sse_znorm.get(method_key, [])
        if sse_z_list:
            row["exp_residual_l1_znorm_mean"] = float(np.mean(sse_z_list))
            row["exp_residual_l1_znorm_std"] = float(np.std(sse_z_list))
        se_list = by_method_sum_exp.get(method_key, [])
        if se_list:
            row["sum_exp_logits_mean"] = float(np.mean(se_list))
            row["sum_exp_logits_std"] = float(np.std(se_list))
        zb_list = by_method_zbar.get(method_key, [])
        if zb_list:
            row["sum_exp_bar_logits_mean"] = float(np.mean(zb_list))
            row["sum_exp_bar_logits_std"] = float(np.std(zb_list))
        vl_list = by_method_val_l2.get(method_key, [])
        if vl_list:
            vm = float(np.mean(vl_list))
            vs = float(np.std(vl_list))
            row["value_softmax_mismatch_ratio_mean"] = vm
            row["value_softmax_mismatch_ratio_std"] = vs
            row["value_logit_group_l2_sq_mean"] = vm
            row["value_logit_group_l2_sq_std"] = vs
        vlz_list = by_method_val_l2_znorm.get(method_key, [])
        if vlz_list:
            row["value_mismatch_ratio_znorm_mean"] = float(np.mean(vlz_list))
            row["value_mismatch_ratio_znorm_std"] = float(np.std(vlz_list))
        ng_list = by_method_n_groups.get(method_key, [])
        if ng_list:
            ng_mean = float(np.mean(ng_list))
            ng_min = int(np.min(ng_list))
            ng_max = int(np.max(ng_list))
            ng_str = f"{ng_mean:.1f}/{ng_min}/{ng_max}"
        else:
            ng_str = "n/a"
        print(
            "[group_cosines]",
            repr(method_key),
            "n_values=", len(vals),
            "n_queries=", len(chunks),
            "n_groups(mean/min/max)=", ng_str,
            "l1_mean=",
            float(np.mean(sse_list)) if sse_list else None,
            "value_mismatch_mean=",
            float(np.mean(vl_list)) if vl_list else None,
        )
        out[method_key] = row
    return out


def _algorithm_family(method_key: str) -> str:
    if method_key.startswith("KMeans-"):
        return "KMeans"
    if method_key.startswith("MultiQ-"):
        return "MultiQ"
    if method_key.startswith("LSH-CrossPoly"):
        return "LSH-CrossPoly"
    if method_key.startswith("IdealEqualSplits"):
        return "IdealEqualSplits"
    if method_key.startswith("IdealEqualWeightSplits"):
        return "IdealEqualWeightSplits"
    if method_key.startswith("IdealTopK"):
        return "IdealTopK"
    if method_key.startswith("IdealSampling"):
        return "IdealSampling"
    return method_key.split("-", 1)[0]


def _method_key_preference_rank(method_key: str) -> int:
    """
    When several budget-sweep variants map to the same hg column,
    prefer larger k / budget for a single representative per eval.
    """
    if not method_key:
        return 0
    m = re.search(r"(?:^|-)k(\d+)(?:$|-)", method_key)
    if m:
        return int(m.group(1))
    m = re.search(r"-(\d+)$", method_key)
    if m:
        return int(m.group(1))
    return 0


def _group_cosine_head_sort_tuple(head_tag: str):
    """Layer, q_head, tag for stable column ordering (tie-break)."""
    try:
        p = head_tag.split("(")[0]
        layer = int(p.split("H")[0][1:])
        qh = int(p.split("H")[1])
        return (layer, qh, head_tag)
    except Exception:
        return (10**9, 10**9, head_tag)


def _group_cosine_column_meta_sort_key(meta: Dict) -> tuple:
    """
    Order table columns like per-head plots: low effective_entropy
    to high, then layer/head, then n_groups.
    """
    ent = meta.get("effective_entropy")
    if ent is None:
        e = float("inf")
    else:
        try:
            e = float(ent)
            if not np.isfinite(e):
                e = float("inf")
        except (TypeError, ValueError):
            e = float("inf")
    h = meta.get("head", "")
    ng = int(meta.get("n_groups", 0))
    return (e, _group_cosine_head_sort_tuple(h), ng)


def aggregate_group_cosines_by_head_group(
    records: List[Dict],
    n_bins: int = 50,
) -> Dict[str, Dict]:
    """
    Build table-style stats:
      rows   = algorithm families
      cols   = (head, n_groups)
      cells  = cosine histogram/statistics

    Records with the same (algorithm, head, n_groups) and the same
    eval_index (one evaluate_query) but different method_key (budget
    sweep) are merged to one row per eval so n_queries matches the
    number of evaluate_query calls, not the number of sweep keys.
    """
    # (algo, col_key) -> eval_index -> [rec, ...]
    with_eval: Dict[
        Tuple[str, str], Dict[int, List[Dict]],
    ] = {}
    # (algo, col_key) -> [rec, ...]  records without eval_index
    no_eval: Dict[Tuple[str, str], List[Dict]] = {}
    col_meta = {}
    row_algos = set()

    for rec in records:
        algo = rec["algorithm"]
        head = rec["head"]
        n_groups = int(rec["n_groups"])
        cos = rec["cosines"]
        sse = rec.get("exp_residual_l1_over_z")
        if sse is None:
            sse = rec.get("exp_residual_sse_over_z2")
        if sse is None:
            sse = rec.get("exp_residual_sse_sum")
        if sse is None:
            sse = rec.get("logit_residual_sse_sum")
        if sse is None:
            sse = rec.get("logit_within_group_var_sum")
        if cos is None or len(cos) == 0:
            continue
        col_key = f"{head}|g{n_groups}"
        if col_key not in col_meta:
            col_meta[col_key] = {
                "head": head,
                "n_groups": n_groups,
                "effective_entropy": rec.get(
                    "effective_entropy",
                ),
            }
        elif col_meta[col_key].get(
            "effective_entropy",
        ) is None and rec.get("effective_entropy") is not None:
            col_meta[col_key]["effective_entropy"] = (
                rec.get("effective_entropy")
            )
        row_algos.add(algo)
        ei = rec.get("eval_index")
        if ei is not None:
            with_eval.setdefault(
                (algo, col_key), {},
            ).setdefault(int(ei), []).append(rec)
        else:
            no_eval.setdefault((algo, col_key), []).append(
                rec,
            )

    cell_vals = {}
    cell_p75z: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_max_exp_lg: Dict[Tuple[str, str], List[float]] = {}
    cell_gl1: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_gl1_znorm: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_p75_gl1: Dict[Tuple[str, str], List[float]] = {}
    cell_scatter_cg_per_ng: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_scatter_explg: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_token_key: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_token_group: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_token_group_over_z: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_token_boundaries: Dict[Tuple[str, str], List[np.ndarray]] = {}
    cell_lvs = {}
    cell_zb = {}
    cell_se = {}
    cell_vl = {}
    cell_se_znorm: Dict[Tuple[str, str], List[float]] = {}
    cell_vl_znorm: Dict[Tuple[str, str], List[float]] = {}

    def _emit_chosen(rec_chosen: Dict, ak: Tuple[str, str]) -> None:
        cos = rec_chosen["cosines"]
        cell_vals.setdefault(ak, []).append(cos)
        p75z_r = rec_chosen.get("max_group_exp_logits_m_over_z")
        if p75z_r is None:
            p75z_r = rec_chosen.get("p75_group_z_norm_residuals")
        if p75z_r is None:
            p75z_r = rec_chosen.get(
                "median_group_z_norm_residuals",
            )
        if p75z_r is not None and len(p75z_r) > 0:
            cell_p75z.setdefault(ak, []).append(
                np.asarray(p75z_r, dtype=np.float64),
            )
        mxlg_r = rec_chosen.get("max_group_exp_lg_m_over_z")
        if (
            mxlg_r is not None
            and np.isfinite(float(mxlg_r))
        ):
            cell_max_exp_lg.setdefault(ak, []).append(
                float(mxlg_r)
            )
        gl1_r = rec_chosen.get("group_l1_cg_m_over_z")
        if gl1_r is not None and len(gl1_r) > 0:
            gl1_a = np.asarray(gl1_r, dtype=np.float64)
            cell_gl1.setdefault(ak, []).append(gl1_a)
        gl1z_r = rec_chosen.get("group_l1_cg_m_znorm")
        if gl1z_r is not None and len(gl1z_r) > 0:
            gl1z_a = np.asarray(gl1z_r, dtype=np.float64)
            cell_gl1_znorm.setdefault(ak, []).append(gl1z_a)
        cg_r = rec_chosen.get("group_l1_cg_m_over_z")
        ng_r = rec_chosen.get("group_sizes")
        exp_r = rec_chosen.get("group_exp_lg_m_over_z")
        tkp_r = rec_chosen.get("token_profile_key_probs")
        tgp_r = rec_chosen.get("token_profile_group_probs")
        tgz_r = rec_chosen.get("token_profile_group_probs_over_z")
        tgb_r = rec_chosen.get("token_profile_group_boundaries")
        if (
            cg_r is not None
            and ng_r is not None
            and exp_r is not None
            and len(cg_r) == len(ng_r)
            and len(cg_r) == len(exp_r)
            and len(cg_r) > 0
        ):
            cg_a = np.asarray(cg_r, dtype=np.float64)
            ng_a = np.asarray(ng_r, dtype=np.float64)
            cell_scatter_cg_per_ng.setdefault(ak, []).append(
                cg_a / np.maximum(ng_a, 1.0),
            )
            cell_scatter_explg.setdefault(ak, []).append(
                np.asarray(exp_r, dtype=np.float64),
            )
        if (
            tkp_r is not None
            and tgp_r is not None
            and tgz_r is not None
            and tgb_r is not None
        ):
            tkp_a = np.asarray(tkp_r, dtype=np.float64)
            tgp_a = np.asarray(tgp_r, dtype=np.float64)
            tgz_a = np.asarray(tgz_r, dtype=np.float64)
            tgb_a = np.asarray(tgb_r, dtype=np.int32)
            if (
                len(tkp_a) == len(tgp_a)
                and len(tkp_a) == len(tgz_a)
                and len(tkp_a) > 0
                and (len(tgb_a) == 0 or int(tgb_a[-1]) == len(tkp_a))
            ):
                cell_token_key.setdefault(ak, []).append(tkp_a)
                cell_token_group.setdefault(ak, []).append(tgp_a)
                cell_token_group_over_z.setdefault(ak, []).append(tgz_a)
                cell_token_boundaries.setdefault(ak, []).append(tgb_a)
        p75_gl1_r = rec_chosen.get("p75_group_l1_cg_m_over_z")
        if p75_gl1_r is None:
            p75_gl1_r = rec_chosen.get(
                "median_group_l1_cg_m_over_z",
            )
        if p75_gl1_r is not None and np.isfinite(
            float(p75_gl1_r),
        ):
            cell_p75_gl1.setdefault(ak, []).append(
                float(p75_gl1_r),
            )
        sse = rec_chosen.get("exp_residual_l1_over_z")
        if sse is None:
            sse = rec_chosen.get("exp_residual_sse_over_z2")
        if sse is None:
            sse = rec_chosen.get("exp_residual_sse_sum")
        if sse is None:
            sse = rec_chosen.get("logit_residual_sse_sum")
        if sse is None:
            sse = rec_chosen.get("logit_within_group_var_sum")
        if sse is not None:
            cell_lvs.setdefault(ak, []).append(float(sse))
        se = rec_chosen.get("sum_exp_logits")
        if se is not None:
            cell_se.setdefault(ak, []).append(float(se))
        zb = rec_chosen.get("sum_exp_bar_logits")
        if zb is not None:
            cell_zb.setdefault(ak, []).append(float(zb))
        vl = rec_chosen.get("value_softmax_mismatch_ratio")
        if vl is None:
            vl = rec_chosen.get("value_logit_group_l2_sq")
        if vl is not None and not np.isnan(vl):
            cell_vl.setdefault(ak, []).append(float(vl))
        sse_z = rec_chosen.get("exp_residual_l1_znorm")
        if sse_z is not None:
            cell_se_znorm.setdefault(ak, []).append(float(sse_z))
        vl_z = rec_chosen.get("value_mismatch_ratio_znorm")
        if vl_z is not None and not np.isnan(vl_z):
            cell_vl_znorm.setdefault(ak, []).append(float(vl_z))

    all_keys = set(with_eval.keys()) | set(no_eval.keys())
    for ak in all_keys:
        if ak in with_eval:
            for ei in sorted(with_eval[ak].keys()):
                recs = with_eval[ak][ei]
                chosen = max(
                    recs,
                    key=lambda r: _method_key_preference_rank(
                        r.get("method", ""),
                    ),
                )
                _emit_chosen(chosen, ak)
        if ak in no_eval:
            for rec in no_eval[ak]:
                _emit_chosen(rec, ak)

    columns = sorted(
        col_meta.keys(),
        key=lambda k: _group_cosine_column_meta_sort_key(
            col_meta[k],
        ),
    )
    rows = sorted(row_algos)

    cells = {}
    for algo in rows:
        cells[algo] = {}
        for col_key in columns:
            chunks = cell_vals.get((algo, col_key), [])
            if not chunks:
                continue
            vals = np.concatenate(chunks)
            counts, edges = np.histogram(
                vals, bins=n_bins, range=(-1.0, 1.0)
            )
            cell_entry = {
                "n_values": int(len(vals)),
                "cos_mean": float(np.mean(vals)),
                "cos_std": float(np.std(vals)),
                "quantiles": {
                    "p05": float(np.quantile(vals, 0.05)),
                    "p25": float(np.quantile(vals, 0.25)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p75": float(np.quantile(vals, 0.75)),
                    "p95": float(np.quantile(vals, 0.95)),
                },
                "histogram": {
                    "bin_edges": edges.tolist(),
                    "counts": counts.tolist(),
                },
            }
            p75z_chunks_h = cell_p75z.get((algo, col_key), [])
            if p75z_chunks_h:
                p75z_vals = np.concatenate(p75z_chunks_h)
                lo = float(np.min(p75z_vals))
                hi = float(np.max(p75z_vals))
                if hi <= lo:
                    lo, hi = lo - 1e-12, hi + 1e-12
                p75z_counts, p75z_edges = np.histogram(
                    p75z_vals, bins=n_bins, range=(lo, hi),
                )
                cell_entry["p75_z_norm_histogram"] = {
                    "bin_edges": p75z_edges.tolist(),
                    "counts": p75z_counts.tolist(),
                }
                cell_entry["max_group_exp_logits_histogram"] = {
                    "bin_edges": p75z_edges.tolist(),
                    "counts": p75z_counts.tolist(),
                }
                mxlg_vals = cell_max_exp_lg.get(
                    (algo, col_key), [],
                )
                if mxlg_vals:
                    mxlg_counts, _ = np.histogram(
                        np.asarray(mxlg_vals, dtype=np.float64),
                        bins=p75z_edges,
                    )
                    cell_entry["max_group_exp_lg_histogram"] = {
                        "bin_edges": p75z_edges.tolist(),
                        "counts": mxlg_counts.tolist(),
                    }
                cell_entry["n_p75_z_norm_values"] = int(
                    len(p75z_vals),
                )
            g1_chunks_h = cell_gl1.get((algo, col_key), [])
            p75_gl1_h = cell_p75_gl1.get((algo, col_key), [])
            if g1_chunks_h:
                g1_vals = np.concatenate(g1_chunks_h)
                p75_gl1_arr = (
                    np.array(p75_gl1_h, dtype=np.float64)
                    if p75_gl1_h
                    else None
                )
                hpack = _aggregate_group_l1_histograms(
                    g1_vals, p75_gl1_arr, n_bins,
                )
                if hpack:
                    cell_entry.update(hpack)
            g1z_chunks_h = cell_gl1_znorm.get(
                (algo, col_key), [],
            )
            if g1z_chunks_h:
                g1z_vals = np.concatenate(g1z_chunks_h)
                cell_entry["group_l1_znorm_meta"] = {
                    "p50": float(np.quantile(g1z_vals, 0.50)),
                    "p75": float(np.quantile(g1z_vals, 0.75)),
                    "p90": float(np.quantile(g1z_vals, 0.90)),
                }
            cg_h = cell_scatter_cg_per_ng.get((algo, col_key), [])
            explg_h = cell_scatter_explg.get((algo, col_key), [])
            if cg_h and explg_h and len(cg_h) == len(explg_h):
                xs = np.concatenate(cg_h)
                ys = np.concatenate(explg_h)
                if len(xs) == len(ys) and len(xs) > 0:
                    cell_entry["cg_eg_scatter"] = {
                        "x": xs.tolist(),
                        "y": ys.tolist(),
                        "n_points": int(len(xs)),
                    }
            tk_h = cell_token_key.get((algo, col_key), [])
            tg_h = cell_token_group.get((algo, col_key), [])
            tz_h = cell_token_group_over_z.get((algo, col_key), [])
            tb_h = cell_token_boundaries.get((algo, col_key), [])
            if (
                tk_h
                and len(tk_h) == len(tg_h)
                and len(tk_h) == len(tz_h)
                and len(tk_h) == len(tb_h)
            ):
                key_parts: List[np.ndarray] = []
                group_parts: List[np.ndarray] = []
                group_over_z_parts: List[np.ndarray] = []
                boundaries: List[int] = []
                running = 0
                for kp, gp, gz, bd in zip(tk_h, tg_h, tz_h, tb_h):
                    if len(kp) != len(gp) or len(kp) != len(gz) or len(kp) == 0:
                        continue
                    key_parts.append(kp)
                    group_parts.append(gp)
                    group_over_z_parts.append(gz)
                    bd_i = np.asarray(bd, dtype=np.int32)
                    if len(bd_i) > 0:
                        boundaries.extend((bd_i + running).tolist())
                    running += len(kp)
                if key_parts and group_parts and group_over_z_parts:
                    cell_entry["group_token_probability_profile"] = {
                        "key_probs": np.concatenate(key_parts).tolist(),
                        "group_probs": np.concatenate(group_parts).tolist(),
                        "group_probs_over_z": np.concatenate(group_over_z_parts).tolist(),
                        "group_boundaries": boundaries,
                    }
            sse_list = cell_lvs.get((algo, col_key), [])
            if sse_list:
                sm = float(np.mean(sse_list))
                ss = float(np.std(sse_list))
                cell_entry["exp_residual_l1_over_z_mean"] = sm
                cell_entry["exp_residual_l1_over_z_std"] = ss
                cell_entry["exp_residual_sse_over_z2_mean"] = sm
                cell_entry["exp_residual_sse_over_z2_std"] = ss
                cell_entry["exp_residual_sse_sum_mean"] = sm
                cell_entry["exp_residual_sse_sum_std"] = ss
                cell_entry["logit_residual_sse_sum_mean"] = sm
                cell_entry["logit_residual_sse_sum_std"] = ss
            zb_list = cell_zb.get((algo, col_key), [])
            if zb_list:
                cell_entry["sum_exp_bar_logits_mean"] = float(
                    np.mean(zb_list),
                )
                cell_entry["sum_exp_bar_logits_std"] = float(
                    np.std(zb_list),
                )
            selist = cell_se.get((algo, col_key), [])
            if selist:
                cell_entry["sum_exp_logits_mean"] = float(
                    np.mean(selist),
                )
                cell_entry["sum_exp_logits_std"] = float(
                    np.std(selist),
                )
            vl_list = cell_vl.get((algo, col_key), [])
            if vl_list:
                vmm = float(np.mean(vl_list))
                vms = float(np.std(vl_list))
                cell_entry["value_softmax_mismatch_ratio_mean"] = vmm
                cell_entry["value_softmax_mismatch_ratio_std"] = vms
                cell_entry["value_logit_group_l2_sq_mean"] = vmm
                cell_entry["value_logit_group_l2_sq_std"] = vms
            se_z_list = cell_se_znorm.get((algo, col_key), [])
            if se_z_list:
                cell_entry["exp_residual_l1_znorm_mean"] = float(
                    np.mean(se_z_list),
                )
                cell_entry["exp_residual_l1_znorm_std"] = float(
                    np.std(se_z_list),
                )
            vlz_list = cell_vl_znorm.get((algo, col_key), [])
            if vlz_list:
                cell_entry["value_mismatch_ratio_znorm_mean"] = float(
                    np.mean(vlz_list),
                )
                cell_entry["value_mismatch_ratio_znorm_std"] = float(
                    np.std(vlz_list),
                )
            cells[algo][col_key] = cell_entry
            ng_cell = int(col_meta[col_key]["n_groups"])
            print(
                "[group_cosines_hg]",
                repr(algo),
                repr(col_key),
                "n_values=", len(vals),
                "n_queries=", len(chunks),
                "n_groups=", ng_cell,
                "l1_mean=",
                float(np.mean(sse_list)) if sse_list else None,
                "value_mismatch_mean=",
                float(np.mean(vl_list)) if vl_list else None,
            )

    return {
        "row_algorithms": rows,
        "columns": [
            {
                "key": col_key,
                "head": col_meta[col_key]["head"],
                "n_groups": col_meta[col_key]["n_groups"],
                "effective_entropy": col_meta[col_key].get(
                    "effective_entropy",
                ),
            }
            for col_key in columns
        ],
        "cells": cells,
    }
