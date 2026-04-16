"""
KMeans comparison: value-vector error vs attention-weight error.

Compares KMeans (hybrid k=0, topk k=5), IdealTopK, IdealSampling
on code_run across cluster counts up to 2048.
Measures two metrics:
  1. Value error: ||approx_output - true_output|| / ||true_output||
  2. Weight error: ||w_approx - w_true|| / ||w_true||

Uses proper causal windowing throughout.

Usage:
    python tests/test_kmeans_comparison.py
"""

import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import (
    full_attention, softmax, flat_kmeans,
    compute_special_indices, relative_l2_error,
    subset_attention, hybrid_attention,
)
from src.algorithms.kmeans_clustering import (
    precompute_cluster_stats, _filter_cluster_members,
)
from src.evaluation.data_loader import load_examples

# ── Config ──────────────────────────────────────────────

VECTORS_DIR = ROOT / "data" / "vectors"
TASK = "code_run"
N_QUERIES = 10
N_SINK = 1
LOCAL_WINDOW = 0
SEED = 42
USE_ROPE = True

CLUSTER_COUNTS = [32, 64, 128, 256, 512, 1024, 2048]

OUT_DIR = ROOT / "tests" / "results" / "kmeans_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────


def weight_error(w_approx, w_true):
    """Relative L2 error between weight vectors."""
    denom = np.linalg.norm(w_true)
    if denom < 1e-10:
        return 0.0
    return float(np.linalg.norm(w_approx - w_true) / denom)


def run_ideal_topk(logits, values, special_idx, candidate_idx, budget):
    """IdealTopK: pick top-budget candidates by logit."""
    n_cand = len(candidate_idx)
    buse = min(budget, n_cand)
    if buse <= 0:
        sel = special_idx
    else:
        cand_logits = logits[candidate_idx]
        if buse < n_cand:
            top_pos = np.argpartition(cand_logits, -buse)[-buse:]
        else:
            top_pos = np.arange(n_cand)
        sel = np.concatenate([
            special_idx, candidate_idx[top_pos],
        ]).astype(np.int64)

    output = subset_attention(logits, values, sel)
    n = len(logits)
    w_approx = np.zeros(n, dtype=np.float64)
    w_sel = softmax(logits[sel].astype(np.float64))
    w_approx[sel] = w_sel
    return output, w_approx, len(sel)


def run_ideal_sampling(logits, values, special_idx, candidate_idx,
                       budget, rng):
    """IdealSampling: sample candidates by attention mass."""
    n_cand = len(candidate_idx)
    buse = min(budget, n_cand)
    if buse <= 0:
        sel = special_idx
    else:
        cand_logits = logits[candidate_idx]
        cand_w = softmax(cand_logits.astype(np.float64))
        chosen = rng.choice(n_cand, size=buse, p=cand_w, replace=False)
        sel = np.concatenate([
            special_idx, candidate_idx[chosen],
        ]).astype(np.int64)

    output = subset_attention(logits, values, sel)
    n = len(logits)
    w_approx = np.zeros(n, dtype=np.float64)
    w_sel = softmax(logits[sel].astype(np.float64))
    w_approx[sel] = w_sel
    return output, w_approx, len(sel)


def run_kmeans_hybrid(q, keys, values, logits, head_dim,
                      special_idx, special_set,
                      cluster_stats, n_causal):
    """
    KMeans hybrid k=0: all clusters as reps, count-weighted softmax.
    Returns (output, w_approx, actual_budget).
    """
    sqrt_d = np.sqrt(head_dim)
    avg_keys = cluster_stats["avg_keys"]

    cm, cc, vm = _filter_cluster_members(
        cluster_stats, n_causal, special_set,
    )

    valid_clusters = np.where(vm)[0]
    if len(valid_clusters) == 0:
        n = len(logits)
        return np.zeros(head_dim, dtype=np.float32), np.zeros(n), 0

    scores_arr = np.array([
        float(q @ avg_keys[c] / sqrt_d + np.log(cc[c]))
        for c in valid_clusters
    ])
    order = valid_clusters[np.argsort(scores_arr)[::-1]]
    groups = [cm[c] for c in order]

    output, eff_budget = hybrid_attention(
        q, keys, values, logits, groups,
        0, head_dim, special_idx, "hybrid",
    )

    # Reconstruct per-key weight vector
    n = len(logits)
    n_special = len(special_idx)
    G = len(groups)
    n_total = n_special + G

    rep_scores = np.empty(n_total, dtype=np.float64)
    rep_scores[:n_special] = logits[special_idx]

    for fi, g in enumerate(groups):
        count = len(g)
        if count == 0:
            rep_scores[n_special + fi] = -1e9
        else:
            avg_key = np.mean(keys[g], axis=0)
            rep_scores[n_special + fi] = (
                q @ avg_key / sqrt_d + np.log(count)
            )

    w_rep = softmax(rep_scores)
    w_approx = np.zeros(n, dtype=np.float64)
    w_approx[special_idx] = w_rep[:n_special]
    for fi, g in enumerate(groups):
        if len(g) > 0:
            w_approx[g] = w_rep[n_special + fi] / len(g)

    return output, w_approx, eff_budget


def run_kmeans_topk(q, keys, values, logits, head_dim,
                    special_idx, special_set,
                    cluster_stats, n_causal, top_k=5):
    """
    KMeans topk: top-k clusters expanded, rest discarded.
    Returns (output, w_approx, actual_budget).
    """
    sqrt_d = np.sqrt(head_dim)
    avg_keys = cluster_stats["avg_keys"]

    cm, cc, vm = _filter_cluster_members(
        cluster_stats, n_causal, special_set,
    )

    valid_clusters = np.where(vm)[0]
    if len(valid_clusters) == 0:
        n = len(logits)
        return np.zeros(head_dim, dtype=np.float32), np.zeros(n), 0

    scores_arr = np.array([
        float(q @ avg_keys[c] / sqrt_d + np.log(cc[c]))
        for c in valid_clusters
    ])
    order = valid_clusters[np.argsort(scores_arr)[::-1]]
    groups = [cm[c] for c in order]

    output, eff_budget = hybrid_attention(
        q, keys, values, logits, groups,
        top_k, head_dim, special_idx, "topk",
    )

    top_individual = []
    for gi in range(min(top_k, len(groups))):
        top_individual.append(groups[gi])
    if top_individual:
        top_keys_idx = np.concatenate(top_individual)
    else:
        top_keys_idx = np.array([], dtype=np.int64)

    sel = np.concatenate([
        special_idx, top_keys_idx,
    ]).astype(np.int64)

    n = len(logits)
    w_approx = np.zeros(n, dtype=np.float64)
    if len(sel) > 0:
        w_sel = softmax(logits[sel].astype(np.float64))
        w_approx[sel] = w_sel

    return output, w_approx, eff_budget


METHODS = [
    "KMeans-hybrid-k0", "KMeans-topk-k5",
    "IdealTopK", "IdealSampling",
]


def _save_intermediate(value_errors, weight_errors, cluster_counts,
                       heads_done, heads_total):
    """Save results after each head completes."""
    agg_v, agg_w = {}, {}
    for m in METHODS:
        agg_v[m], agg_w[m] = {}, {}
        for C in cluster_counts:
            ve = value_errors[m][C]
            we = weight_errors[m][C]
            if ve:
                agg_v[m][str(C)] = {
                    "mean": float(np.mean(ve)),
                    "std": float(np.std(ve)),
                }
                agg_w[m][str(C)] = {
                    "mean": float(np.mean(we)),
                    "std": float(np.std(we)),
                }

    partial = {
        "heads_done": heads_done,
        "heads_total": heads_total,
        "cluster_counts": cluster_counts,
        "value_errors": agg_v,
        "weight_errors": agg_w,
    }
    path = OUT_DIR / "results_partial.json"
    with open(path, "w") as f:
        json.dump(partial, f, indent=2, default=float)
    print(f"  Saved partial results ({heads_done}/{heads_total} heads)")


# ── Main ────────────────────────────────────────────────

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    meta_path = VECTORS_DIR / TASK / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    heads = meta["selected_heads"]
    head_dim = 128

    value_errors = defaultdict(lambda: defaultdict(list))
    weight_errors = defaultdict(lambda: defaultdict(list))

    for hi, hinfo in enumerate(heads):
        layer = hinfo["layer"]
        q_head = hinfo["q_head"]
        kv_head = hinfo["kv_head"]
        label = hinfo.get("selection_label", "")

        print(f"\n{'='*60}")
        print(f"Head {hi+1}/{len(heads)}: L{layer} H{q_head} "
              f"(kv={kv_head}) [{label}]")
        print(f"{'='*60}")

        examples = list(load_examples(
            VECTORS_DIR, TASK, layer, q_head, kv_head,
            use_rope=USE_ROPE, max_examples=1,
        ))
        if not examples:
            print(f"  No data, skipping")
            continue

        ex = examples[0]
        Q, K, V = ex["Q"], ex["K"], ex["V"]
        seq_len = Q.shape[0]
        print(f"  Sequence length: {seq_len}")

        qpos_list = list(range(
            max(0, seq_len - N_QUERIES), seq_len
        ))

        # Precompute KMeans for each cluster count (on full keys)
        kmeans_stats = {}
        for C in CLUSTER_COUNTS:
            C_eff = min(C, seq_len)
            print(f"  KMeans C={C_eff}...", end=" ", flush=True)
            t_km = time.time()
            _, labels = flat_kmeans(K, C_eff, seed=SEED)
            stats = precompute_cluster_stats(K, V, labels, C_eff)
            kmeans_stats[C] = stats
            print(f"{time.time()-t_km:.1f}s")

        for qi, qpos in enumerate(qpos_list):
            q = Q[qpos]
            keys_causal = K[:qpos + 1]
            vals_causal = V[:qpos + 1]
            n_causal = qpos + 1

            true_out, logits, true_weights = full_attention(
                q, keys_causal, vals_causal, head_dim,
            )

            sp_idx, cand_idx = compute_special_indices(
                n_causal, N_SINK, LOCAL_WINDOW,
            )
            special_set = set(sp_idx.tolist())

            for C in CLUSTER_COUNTS:
                stats = kmeans_stats[C]

                # --- KMeans hybrid k=0 ---
                out_h, w_h, bud_h = run_kmeans_hybrid(
                    q, keys_causal, vals_causal, logits,
                    head_dim, sp_idx, special_set,
                    stats, n_causal,
                )
                ve_h = relative_l2_error(out_h, true_out)
                we_h = weight_error(w_h, true_weights)
                value_errors["KMeans-hybrid-k0"][C].append(ve_h)
                weight_errors["KMeans-hybrid-k0"][C].append(we_h)

                # --- KMeans topk k=5 ---
                out_t, w_t, bud_t = run_kmeans_topk(
                    q, keys_causal, vals_causal, logits,
                    head_dim, sp_idx, special_set,
                    stats, n_causal, top_k=5,
                )
                ve_t = relative_l2_error(out_t, true_out)
                we_t = weight_error(w_t, true_weights)
                value_errors["KMeans-topk-k5"][C].append(ve_t)
                weight_errors["KMeans-topk-k5"][C].append(we_t)

                # --- IdealTopK at same budget as hybrid ---
                budget_cand = max(0, bud_h - len(sp_idx))
                out_itk, w_itk, _ = run_ideal_topk(
                    logits, vals_causal, sp_idx, cand_idx,
                    budget_cand,
                )
                ve_itk = relative_l2_error(out_itk, true_out)
                we_itk = weight_error(w_itk, true_weights)
                value_errors["IdealTopK"][C].append(ve_itk)
                weight_errors["IdealTopK"][C].append(we_itk)

                # --- IdealSampling at same budget as hybrid ---
                out_is, w_is, _ = run_ideal_sampling(
                    logits, vals_causal, sp_idx, cand_idx,
                    budget_cand, rng,
                )
                ve_is = relative_l2_error(out_is, true_out)
                we_is = weight_error(w_is, true_weights)
                value_errors["IdealSampling"][C].append(ve_is)
                weight_errors["IdealSampling"][C].append(we_is)

            if qi == 0 or (qi + 1) % 5 == 0:
                print(f"  Query {qi+1}/{len(qpos_list)} "
                      f"(pos={qpos}) done")

        # Save intermediate results after each head
        _save_intermediate(
            value_errors, weight_errors, CLUSTER_COUNTS,
            hi + 1, len(heads),
        )

    # ── Aggregate & Print ───────────────────────────────

    print(f"\n{'='*60}")
    print("Aggregated Results (mean over queries & heads)")
    print(f"{'='*60}")

    methods = METHODS

    agg_value = {}
    agg_weight = {}
    for m in methods:
        agg_value[m] = {}
        agg_weight[m] = {}
        for C in CLUSTER_COUNTS:
            ve_list = value_errors[m][C]
            we_list = weight_errors[m][C]
            if ve_list:
                agg_value[m][C] = {
                    "mean": np.mean(ve_list),
                    "std": np.std(ve_list),
                }
                agg_weight[m][C] = {
                    "mean": np.mean(we_list),
                    "std": np.std(we_list),
                }

    print(f"\n{'Method':<25} {'C':>6} {'Val err':>10} "
          f"{'Wgt err':>10}")
    print("-" * 55)
    for m in methods:
        for C in CLUSTER_COUNTS:
            if C in agg_value[m]:
                v = agg_value[m][C]
                w = agg_weight[m][C]
                print(f"{m:<25} {C:>6} "
                      f"{v['mean']:>10.4f} {w['mean']:>10.4f}")
        print()

    results = {
        "config": {
            "task": TASK,
            "n_queries": N_QUERIES,
            "n_sink": N_SINK,
            "local_window": LOCAL_WINDOW,
            "cluster_counts": CLUSTER_COUNTS,
            "seed": SEED,
        },
        "value_errors": {
            m: {str(C): agg_value[m][C]
                for C in CLUSTER_COUNTS if C in agg_value[m]}
            for m in methods
        },
        "weight_errors": {
            m: {str(C): agg_weight[m][C]
                for C in CLUSTER_COUNTS if C in agg_weight[m]}
            for m in methods
        },
    }
    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved results to {json_path}")

    # ── Plot ────────────────────────────────────────────

    colors = {
        "KMeans-hybrid-k0": "#1f77b4",
        "KMeans-topk-k5": "#ff7f0e",
        "IdealTopK": "#2ca02c",
        "IdealSampling": "#d62728",
    }
    markers = {
        "KMeans-hybrid-k0": "o",
        "KMeans-topk-k5": "s",
        "IdealTopK": "^",
        "IdealSampling": "v",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for m in methods:
        xs = sorted(agg_value[m].keys())
        if not xs:
            continue
        ys_v = [agg_value[m][c]["mean"] for c in xs]
        es_v = [agg_value[m][c]["std"] for c in xs]
        ys_w = [agg_weight[m][c]["mean"] for c in xs]
        es_w = [agg_weight[m][c]["std"] for c in xs]

        ax1.errorbar(xs, ys_v, yerr=es_v,
                     label=m, color=colors[m],
                     marker=markers[m], markersize=6,
                     capsize=3, linewidth=1.5)
        ax2.errorbar(xs, ys_w, yerr=es_w,
                     label=m, color=colors[m],
                     marker=markers[m], markersize=6,
                     capsize=3, linewidth=1.5)

    for ax, title, ylabel in [
        (ax1, "Value Vector Error", "Relative L2 Error (output)"),
        (ax2, "Attention Weight Error", "Relative L2 Error (weights)"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("Number of Clusters (budget proxy)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"KMeans Comparison — {TASK} (causal, "
        f"{N_QUERIES} queries/head, 5 heads)",
        fontsize=13,
    )
    fig.tight_layout()
    plot_path = OUT_DIR / "comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
