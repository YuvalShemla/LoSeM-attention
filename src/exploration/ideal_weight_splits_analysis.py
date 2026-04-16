"""
Quick analysis of IdealEqualWeightSplits grouping behavior.

For a few (task, head, query) combos, look at:
- Group size distribution at different budgets
- Rank ranges of keys within each group
- How top-K keys are allocated across groups
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.data_loader import load_examples, load_task_metadata
from src.core import softmax, compute_special_indices
from src.algorithms.idealized_methods import IdealEqualWeightSplits


def analyze_one_query(query, keys, values, head_dim, special_idx, candidate_idx, budgets):
    """Analyze IdealEqualWeightSplits grouping for one query."""
    logits = (query @ keys.T) / np.sqrt(head_dim)
    cand_logits = logits[candidate_idx]
    cand_weights = softmax(cand_logits)

    # Rank map: candidate_idx[rank] = global index, rank 0 = highest weight
    sort_order = np.argsort(cand_weights)[::-1]
    sorted_idx = candidate_idx[sort_order]
    sorted_weights = cand_weights[sort_order]

    # Build reverse map: global_idx -> rank
    rank_of = {}
    for rank, gidx in enumerate(sorted_idx):
        rank_of[int(gidx)] = rank

    n_cand = len(candidate_idx)
    results = {}

    for budget in budgets:
        groups = IdealEqualWeightSplits._equal_weight_groups(
            sorted_idx.copy(), sorted_weights.copy(), budget,
        )
        n_groups = len(groups)

        group_sizes = []
        rank_mins = []
        rank_maxs = []
        group_masses = []

        for g in groups:
            size = len(g)
            group_sizes.append(size)
            ranks = [rank_of[int(idx)] for idx in g]
            rank_mins.append(min(ranks))
            rank_maxs.append(max(ranks))
            mass = sum(float(cand_weights[sort_order[r]]) for r in ranks)
            group_masses.append(mass)

        # Top-K allocation: which group contains each of the top-K keys?
        top_k_in_group = {}
        for K in [1, 5, 10, 50, 100]:
            if K > n_cand:
                continue
            top_k_global = set(int(sorted_idx[r]) for r in range(K))
            # For each top-k key, what group is it in and what's that group's size?
            sizes_for_topk = []
            for gi, g in enumerate(groups):
                for idx in g:
                    if int(idx) in top_k_global:
                        sizes_for_topk.append(len(g))
            individual = sum(1 for s in sizes_for_topk if s == 1)
            small = sum(1 for s in sizes_for_topk if 2 <= s <= 10)
            large = sum(1 for s in sizes_for_topk if s > 10)
            top_k_in_group[K] = {
                "individual": individual,
                "small_group": small,
                "large_group": large,
            }

        results[budget] = {
            "n_groups": n_groups,
            "group_sizes": group_sizes,
            "rank_mins": rank_mins,
            "rank_maxs": rank_maxs,
            "group_masses": group_masses,
            "top_k_allocation": top_k_in_group,
            "n_candidates": n_cand,
        }

    return results


def print_analysis(results, label):
    """Print a readable summary."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for budget, data in sorted(results.items()):
        n_groups = data["n_groups"]
        sizes = data["group_sizes"]
        rank_mins = data["rank_mins"]
        rank_maxs = data["rank_maxs"]
        masses = data["group_masses"]
        n_cand = data["n_candidates"]

        print(f"\n--- Budget={budget} -> {n_groups} groups, {n_cand} candidates ---")

        # Group size distribution
        sizes_arr = np.array(sizes)
        print(f"  Group sizes: min={sizes_arr.min()}, median={int(np.median(sizes_arr))}, "
              f"max={sizes_arr.max()}, mean={sizes_arr.mean():.1f}")

        # Show first 10 and last 5 groups
        print(f"  First 10 groups (high attention):")
        for i in range(min(10, n_groups)):
            print(f"    Group {i}: size={sizes[i]:>6}, "
                  f"ranks [{rank_mins[i]:>6} - {rank_maxs[i]:>6}], "
                  f"mass={masses[i]:.4f}")

        if n_groups > 15:
            print(f"  ...")
            print(f"  Last 5 groups (low attention):")
            for i in range(max(10, n_groups - 5), n_groups):
                print(f"    Group {i}: size={sizes[i]:>6}, "
                      f"ranks [{rank_mins[i]:>6} - {rank_maxs[i]:>6}], "
                      f"mass={masses[i]:.6f}")

        # Cumulative mass
        cum_mass = np.cumsum(masses)
        total_mass = cum_mass[-1]
        for frac in [0.5, 0.9, 0.95, 0.99]:
            idx = np.searchsorted(cum_mass, frac * total_mass)
            if idx < n_groups:
                print(f"  {frac*100:.0f}% mass captured by first {idx+1}/{n_groups} groups")

        # Top-K allocation
        alloc = data["top_k_allocation"]
        for K in sorted(alloc.keys()):
            a = alloc[K]
            print(f"  Top-{K} keys: {a['individual']} individual, "
                  f"{a['small_group']} in small groups (2-10), "
                  f"{a['large_group']} in large groups (>10)")


def main():
    config_path = Path("src/evaluation/evaluation_config.yaml")
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vectors_dir = Path(config["data"]["vectors_dir"])
    tasks = ["math_calc", "code_run", "kv_retrieval"]
    budgets = [16, 64, 256, 1024, 4096]

    for task in tasks:
        meta = load_task_metadata(vectors_dir, task)
        heads = meta.get("selected_heads", [])
        if not heads:
            print(f"No selected heads for {task}, skipping")
            continue

        # Just use first head
        h = heads[0]
        examples = list(load_examples(
            vectors_dir, task,
            layer=h["layer"], head=h["q_head"],
            kv_head=h["kv_head"], max_examples=1,
        ))
        if not examples:
            continue

        ex = examples[0]
        Q, K, V = ex["Q"], ex["K"], ex["V"]
        seq_len = Q.shape[0]
        head_dim = Q.shape[1]

        # Use last query position
        qpos = seq_len - 1
        query = Q[qpos]
        keys = K[:qpos + 1]
        values = V[:qpos + 1]

        special_idx, candidate_idx = compute_special_indices(
            n_causal=len(keys), n_sink=1, local_window=100,
        )

        label = f"{task} | L{h['layer']}H{h['q_head']} | {len(keys)} tokens"
        results = analyze_one_query(
            query, keys, values, head_dim,
            special_idx, candidate_idx, budgets,
        )
        print_analysis(results, label)


if __name__ == "__main__":
    main()
