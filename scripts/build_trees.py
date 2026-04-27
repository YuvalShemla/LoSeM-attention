"""
Build binary KMeans trees for all heads on math_calc and code_run.

Tree construction: level-by-level BFS approach with 2-means at each
node. Reports progress per depth level with ETA.

Storage format (.npz):
  - parent[n_nodes]:       int32, parent node id (-1 for root)
  - left_child[n_nodes]:   int32, left child id (-1 for leaf)
  - right_child[n_nodes]:  int32, right child id (-1 for leaf)
  - depth[n_nodes]:        int32
  - count[n_nodes]:        int64
  - key_sum[n_nodes, d]:   float64, sum of keys in subtree
  - value_sum[n_nodes, d]: float64, sum of values in subtree
  - leaf_of[n_candidates]: int32, maps each candidate → its leaf node
  - is_leaf[n_nodes]:      bool

This is enough for query-time usage:
  - Traversal: uses key_sum/count (mean_key) and count at each node
  - Removal: walk from leaf_of[i] up parent chain, subtract key[i]/val[i]
  - No need to store actual key/value vectors in the tree
"""
import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np
from pathlib import Path
from src.core import flat_kmeans, compute_special_indices
from src.evaluation.data_loader import load_examples


# ── Configuration ───────────────────────────────────────────────

MAX_DEPTH = 14       # max tree depth (2^14 = 16K potential leaves)
MIN_LEAF = 4         # stop splitting below this count
KMEANS_ITER = 20     # iterations per 2-means split
TREE_DIR = Path("data/trees")
D = 128              # head_dim


# ── Tree building (level-by-level BFS) ─────────────────────────

def build_and_save_tree(cand_keys, cand_vals, save_path, seed=42):
    """
    Build binary KMeans tree using level-by-level BFS.

    Returns tree metadata dict for logging.
    """
    n_cand = len(cand_keys)
    d = cand_keys.shape[1]

    # Pre-allocate arrays (upper bound: 2*n_cand nodes)
    max_nodes = 2 * n_cand + 1
    parent = np.full(max_nodes, -1, dtype=np.int32)
    left_child = np.full(max_nodes, -1, dtype=np.int32)
    right_child = np.full(max_nodes, -1, dtype=np.int32)
    node_depth = np.zeros(max_nodes, dtype=np.int32)
    node_count = np.zeros(max_nodes, dtype=np.int64)
    key_sum = np.zeros((max_nodes, d), dtype=np.float64)
    value_sum = np.zeros((max_nodes, d), dtype=np.float64)

    # leaf_of: which leaf does each candidate end up in
    leaf_of = np.full(n_cand, -1, dtype=np.int32)

    # Track which members belong to each node (temporary, for splitting)
    node_members = {}  # node_id -> np.ndarray of candidate indices

    # Initialize root (node 0)
    n_nodes = 1
    all_members = np.arange(n_cand, dtype=np.int32)
    node_members[0] = all_members
    node_count[0] = n_cand
    key_sum[0] = cand_keys.astype(np.float64).sum(axis=0)
    value_sum[0] = cand_vals.astype(np.float64).sum(axis=0)
    node_depth[0] = 0

    # BFS level by level
    current_level = [0]  # nodes to split at current depth
    total_splits = 0
    total_t0 = time.time()

    for depth in range(MAX_DEPTH):
        if not current_level:
            break

        level_t0 = time.time()
        n_to_split = len(current_level)
        n_split = 0
        next_level = []

        for node_id in current_level:
            members = node_members[node_id]
            if len(members) <= MIN_LEAF:
                # Mark as leaf
                leaf_of[members] = node_id
                del node_members[node_id]
                continue

            # 2-means split
            k_sub = cand_keys[members]
            _, labels = flat_kmeans(
                k_sub, 2,
                seed=seed + depth * 10000 + node_id,
                n_iter=KMEANS_ITER,
            )

            left_mask = labels == 0
            right_mask = labels == 1
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                # Can't split — mark as leaf
                leaf_of[members] = node_id
                del node_members[node_id]
                continue

            # Create left child
            left_id = n_nodes
            n_nodes += 1
            left_members = members[left_mask]
            parent[left_id] = node_id
            left_child[node_id] = left_id
            node_depth[left_id] = depth + 1
            node_count[left_id] = len(left_members)
            key_sum[left_id] = cand_keys[left_members].astype(np.float64).sum(axis=0)
            value_sum[left_id] = cand_vals[left_members].astype(np.float64).sum(axis=0)
            node_members[left_id] = left_members

            # Create right child
            right_id = n_nodes
            n_nodes += 1
            right_members = members[right_mask]
            parent[right_id] = node_id
            right_child[node_id] = right_id
            node_depth[right_id] = depth + 1
            node_count[right_id] = len(right_members)
            key_sum[right_id] = cand_keys[right_members].astype(np.float64).sum(axis=0)
            value_sum[right_id] = cand_vals[right_members].astype(np.float64).sum(axis=0)
            node_members[right_id] = right_members

            # Parent no longer holds members directly
            del node_members[node_id]

            next_level.extend([left_id, right_id])
            n_split += 1
            total_splits += 1

        level_time = time.time() - level_t0
        elapsed = time.time() - total_t0
        leaves_so_far = np.sum(leaf_of >= 0)

        print(f"    Depth {depth:2d}: {n_to_split:6d} nodes, "
              f"{n_split:6d} split, "
              f"{level_time:6.1f}s "
              f"(total {elapsed:.0f}s, "
              f"{n_nodes} nodes, {leaves_so_far} leaves assigned)",
              flush=True)

        current_level = next_level

    # Any remaining unsplit nodes become leaves
    for node_id, members in node_members.items():
        leaf_of[members] = node_id

    # Trim arrays to actual size
    parent = parent[:n_nodes]
    left_child = left_child[:n_nodes]
    right_child = right_child[:n_nodes]
    node_depth = node_depth[:n_nodes]
    node_count = node_count[:n_nodes]
    key_sum = key_sum[:n_nodes]
    value_sum = value_sum[:n_nodes]
    is_leaf = (left_child == -1) & (right_child == -1)

    # Verify: every candidate has a leaf assignment
    assert np.all(leaf_of >= 0), f"Some candidates unassigned: {(leaf_of < 0).sum()}"
    # Verify: leaf counts sum to n_cand
    leaf_total = node_count[is_leaf].sum()
    assert leaf_total == n_cand, f"Leaf total {leaf_total} != n_cand {n_cand}"

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        parent=parent,
        left_child=left_child,
        right_child=right_child,
        depth=node_depth,
        count=node_count,
        key_sum=key_sum,
        value_sum=value_sum,
        leaf_of=leaf_of,
        is_leaf=is_leaf,
    )

    total_time = time.time() - total_t0
    n_leaves = is_leaf.sum()
    file_size_mb = save_path.stat().st_size / 1e6

    meta = {
        "n_nodes": int(n_nodes),
        "n_leaves": int(n_leaves),
        "max_depth_reached": int(node_depth.max()),
        "n_candidates": int(n_cand),
        "total_splits": total_splits,
        "build_time_s": round(total_time, 1),
        "file_size_mb": round(file_size_mb, 1),
    }
    print(f"    Done: {n_nodes} nodes, {n_leaves} leaves, "
          f"max_depth={node_depth.max()}, "
          f"{total_time:.0f}s, {file_size_mb:.1f}MB",
          flush=True)
    return meta


# ── Main ────────────────────────────────────────────────────────

def main():
    TREE_DIR.mkdir(parents=True, exist_ok=True)

    # All heads to build trees for
    tasks_heads = []
    for task in ['math_calc', 'code_run']:
        meta_path = Path(f'data/vectors/{task}/metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        for h in meta['selected_heads']:
            tasks_heads.append((
                task,
                h['layer'], h['q_head'], h['kv_head'],
                h['selection_label'],
            ))

    total = len(tasks_heads)
    all_meta = {}

    print(f"Building {total} trees (max_depth={MAX_DEPTH}, "
          f"min_leaf={MIN_LEAF}, kmeans_iter={KMEANS_ITER})")
    print(f"Output: {TREE_DIR}/")
    print()

    for i, (task, layer, qh, kvh, label) in enumerate(tasks_heads):
        tree_name = f"{task}_L{layer}H{qh}_kv{kvh}"
        save_path = TREE_DIR / f"{tree_name}.npz"

        # Skip if already built
        if save_path.exists():
            print(f"[{i+1}/{total}] {tree_name} ({label}) — SKIPPED (exists)")
            continue

        print(f"[{i+1}/{total}] {tree_name} ({label})")

        # Load data
        t0 = time.time()
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        K, V = ex['K'], ex['V']
        sp, cand = compute_special_indices(len(K), 1, 0)
        cand_keys = K[cand]
        cand_vals = V[cand]
        print(f"  Loaded: {len(cand)} candidates ({time.time()-t0:.1f}s)",
              flush=True)

        # Build tree
        meta = build_and_save_tree(
            cand_keys, cand_vals, save_path,
            seed=42,
        )
        all_meta[tree_name] = meta
        print(flush=True)

    # Save summary
    summary_path = TREE_DIR / "build_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nAll done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
