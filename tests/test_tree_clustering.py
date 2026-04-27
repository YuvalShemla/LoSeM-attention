"""
Test hierarchical binary KMeans tree for attention approximation.

Tree structure: recursive 2-means on keys, each node stores
key_sum, value_sum, count, children.

Algorithm:
1. Oracle top-K: select top-K by logit, remove from tree
2. Priority-queue refinement: start with root (1 cluster),
   split highest-scored node into children, repeat until
   cluster budget exhausted.
3. Joint softmax over special + topK + tree cluster reps.

Compare to IdealEqualWeightSplits at same budget.
"""
import numpy as np
import heapq
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.core import (
    full_attention, compute_special_indices, softmax,
    relative_l2_error, flat_kmeans,
)
from src.evaluation.data_loader import load_examples


class TreeNode:
    __slots__ = [
        'members', 'key_sum', 'value_sum', 'count',
        'left', 'right', 'depth',
    ]

    def __init__(self, members, keys, values):
        self.members = members  # indices into candidate array
        self.count = len(members)
        self.key_sum = keys[members].astype(np.float64).sum(axis=0)
        self.value_sum = values[members].astype(np.float64).sum(axis=0)
        self.left = None
        self.right = None
        self.depth = 0

    @property
    def mean_key(self):
        return (self.key_sum / self.count).astype(np.float32)

    @property
    def mean_value(self):
        return (self.value_sum / self.count).astype(np.float32)


def build_tree(keys, values, max_depth=14, min_leaf=4, seed=42):
    """Build binary KMeans tree on keys."""
    all_members = np.arange(len(keys))
    root = TreeNode(all_members, keys, values)

    def _split(node, depth):
        node.depth = depth
        if depth >= max_depth or node.count <= min_leaf:
            return
        k_sub = keys[node.members]
        if len(k_sub) < 2:
            return
        _, labels = flat_kmeans(k_sub, 2, seed=seed + depth, n_iter=20)
        left_mask = labels == 0
        right_mask = labels == 1
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return  # can't split
        node.left = TreeNode(node.members[left_mask], keys, values)
        node.right = TreeNode(node.members[right_mask], keys, values)
        _split(node.left, depth + 1)
        _split(node.right, depth + 1)

    _split(root, 0)
    return root


def remove_keys_from_tree(root, removed_set, keys, values):
    """Remove keys from tree, updating sums along the path."""
    def _remove(node):
        if node is None:
            return
        # Find which members are removed
        mask = np.isin(node.members, list(removed_set))
        n_removed = mask.sum()
        if n_removed == 0:
            return
        removed_local = node.members[mask]
        # Update sums
        for idx in removed_local:
            node.key_sum -= keys[idx].astype(np.float64)
            node.value_sum -= values[idx].astype(np.float64)
        node.count -= n_removed
        node.members = node.members[~mask]
        # Recurse
        _remove(node.left)
        _remove(node.right)

    _remove(root)


def tree_refine(root, query, sqrt_d, cluster_budget):
    """
    Priority-queue refinement: start with root,
    split highest-scored node into children.
    Returns list of (mean_key, mean_value, count) for
    the final cluster set.
    """
    q = query.astype(np.float64)

    def score_node(node):
        if node.count == 0:
            return -1e30
        mk = node.mean_key.astype(np.float64)
        return float(q @ mk) / sqrt_d + np.log(node.count)

    # Max-heap (negate for min-heap)
    # Entries: (-score, id, node)
    heap = []
    node_id = 0
    active_nodes = {}  # id -> node

    if root.count > 0:
        s = score_node(root)
        heapq.heappush(heap, (-s, node_id, root))
        active_nodes[node_id] = root
        node_id += 1

    n_clusters = 1  # root counts as 1

    while n_clusters < cluster_budget and heap:
        # Pop highest-scored node
        neg_s, nid, node = heapq.heappop(heap)
        if nid not in active_nodes:
            continue  # stale entry
        del active_nodes[nid]

        # If leaf (no children) or can't split, keep it
        if node.left is None and node.right is None:
            # Put it back, it can't be split
            active_nodes[nid] = node
            heapq.heappush(heap, (neg_s, nid, node))
            # Try next node
            # But if all remaining are leaves, stop
            all_leaves = all(
                active_nodes[i].left is None and
                active_nodes[i].right is None
                for i in active_nodes
            )
            if all_leaves:
                break
            continue

        # Split: replace with two children
        # (going from 1 node to 2 = net +1 cluster)
        children = []
        for child in [node.left, node.right]:
            if child is not None and child.count > 0:
                s = score_node(child)
                active_nodes[node_id] = child
                heapq.heappush(heap, (-s, node_id, child))
                children.append(node_id)
                node_id += 1

        if len(children) == 2:
            n_clusters += 1  # went from 1 to 2
        elif len(children) == 1:
            pass  # went from 1 to 1, no net change
        else:
            n_clusters -= 1  # removed empty node

    # Collect all active nodes
    result = []
    for nid, node in active_nodes.items():
        if node.count > 0:
            result.append((
                node.mean_key, node.mean_value, node.count,
            ))
    return result


def run_tree_method(q, keys, values, logits, sp, cand,
                    d, budget, rng):
    """Run tree-based method: half topK, half tree clusters."""
    sqrt_d = np.sqrt(d)
    n_cand = len(cand)
    cand_keys = keys[cand]
    cand_vals = values[cand]

    b_topk = budget // 2
    b_cluster = budget - b_topk

    # Oracle top-K
    cand_logits = logits[cand]
    if b_topk > 0 and b_topk < n_cand:
        topk_local = np.argpartition(
            cand_logits, -b_topk,
        )[-b_topk:]
    else:
        topk_local = np.arange(min(b_topk, n_cand))
    topk_global = cand[topk_local]

    # Build tree
    tree = build_tree(cand_keys, cand_vals, max_depth=14)

    # Remove top-K from tree
    removed_set = set(topk_local.tolist())
    remove_keys_from_tree(tree, removed_set, cand_keys, cand_vals)

    # Tree refinement
    clusters = tree_refine(tree, q, sqrt_d, max(1, b_cluster))

    # Build joint softmax
    n_sp = len(sp)
    n_topk = len(topk_local)
    n_clust = len(clusters)
    n_total = n_sp + n_topk + n_clust

    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float32)
    q64 = q.astype(np.float64)

    # Special
    scores[:n_sp] = logits[sp].astype(np.float64)
    out_vals[:n_sp] = values[sp]

    # TopK
    off = n_sp
    scores[off:off + n_topk] = logits[topk_global].astype(np.float64)
    out_vals[off:off + n_topk] = values[topk_global]

    # Tree clusters
    off = n_sp + n_topk
    for i, (mk, mv, cnt) in enumerate(clusters):
        ml = float(q64 @ mk.astype(np.float64)) / sqrt_d
        scores[off + i] = ml + np.log(cnt)
        out_vals[off + i] = mv

    w = softmax(scores).astype(np.float32)
    output = w @ out_vals

    return output, n_total, n_clust


# Also run IdealEqualWeightSplits for comparison
def run_ideal_ews(q, keys, values, logits, sp, cand, d, budget):
    """IdealEqualWeightSplits at given budget."""
    sqrt_d = np.sqrt(d)
    n_cand = len(cand)
    cand_logits = logits[cand]
    cand_w = softmax(cand_logits)
    sort_order = np.argsort(cand_w)[::-1]
    sorted_idx = cand[sort_order]

    num_groups = min(budget, n_cand)
    groups = [
        np.asarray(g)
        for g in np.array_split(sorted_idx, num_groups)
        if len(g) > 0
    ]

    n_sp = len(sp)
    n_groups = len(groups)
    n_total = n_sp + n_groups
    scores_arr = np.empty(n_total)
    out_v = np.empty((n_total, d))

    scores_arr[:n_sp] = logits[sp]
    out_v[:n_sp] = values[sp]

    q64 = q.astype(np.float64)
    for i, g in enumerate(groups):
        cnt = len(g)
        mk = keys[g].astype(np.float64).mean(axis=0)
        mv = values[g].astype(np.float64).mean(axis=0)
        scores_arr[n_sp + i] = float(q64 @ mk) / sqrt_d + np.log(cnt)
        out_v[n_sp + i] = mv.astype(np.float32)

    w = softmax(scores_arr).astype(np.float32)
    return (w @ out_v.astype(np.float32)), n_total


def main():
    d = 128
    heads = [
        ('math_calc', 31, 14, 3, 'p0 ent=0.20'),
        ('math_calc', 25, 1, 0, 'p25 ent=2.88'),
        ('math_calc', 30, 13, 3, 'p75 ent=4.71'),
        ('math_calc', 0, 22, 5, 'p100 ent=10.19'),
    ]

    rng = np.random.default_rng(42)

    for task, layer, qh, kvh, label in heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K, V = ex['Q'], ex['K'], ex['V']
        q = Q[-1]; keys = K; values = V
        full_out, logits, weights = full_attention(q, keys, values, d)
        sp, cand = compute_special_indices(len(keys), 1, 0)

        print(f"\n{'='*60}")
        print(f"  {label} ({len(cand)} candidates)")
        print(f"{'='*60}")
        print(f"  {'budget':>6s}  {'Tree':>10s}  {'EWS':>10s}  "
              f"{'Tree actual':>12s}  {'n_clust':>8s}")

        for budget in [32, 64, 128, 256, 512, 1024]:
            # Tree method
            out_tree, actual_tree, n_clust = run_tree_method(
                q, keys, values, logits, sp, cand, d, budget, rng,
            )
            err_tree = relative_l2_error(out_tree, full_out)

            # IdealEqualWeightSplits at same budget
            out_ews, actual_ews = run_ideal_ews(
                q, keys, values, logits, sp, cand, d, budget,
            )
            err_ews = relative_l2_error(out_ews, full_out)

            print(
                f"  {budget:6d}  {err_tree:10.6f}  {err_ews:10.6f}  "
                f"{actual_tree:12d}  {n_clust:8d}"
            )

    # Analyze tree depth and node sizes
    print(f"\n{'='*60}")
    print("  Tree structure analysis (math_calc p75)")
    print(f"{'='*60}")

    ex = list(load_examples(
        Path('data/vectors'), 'math_calc',
        layer=30, head=13, kv_head=3,
        phase=None, max_examples=1, use_rope=True,
    ))[0]
    Q, K, V = ex['Q'], ex['K'], ex['V']
    q = Q[-1]; keys = K; values = V
    full_out, logits, _ = full_attention(q, keys, values, d)
    sp, cand = compute_special_indices(len(keys), 1, 0)
    cand_keys = keys[cand]
    cand_vals = values[cand]

    tree = build_tree(cand_keys, cand_vals, max_depth=14)

    # Count nodes at each depth
    depth_counts = {}
    def count_depths(node, d=0):
        if node is None:
            return
        depth_counts[d] = depth_counts.get(d, 0) + 1
        count_depths(node.left, d + 1)
        count_depths(node.right, d + 1)
    count_depths(tree)

    print("  Depth -> #nodes:")
    for d_level in sorted(depth_counts.keys()):
        print(f"    depth {d_level:2d}: {depth_counts[d_level]:6d} nodes")

    # Test: does the tree refinement produce balanced clusters?
    sqrt_d = np.sqrt(d)
    for n_clust in [16, 64, 256]:
        clusters = tree_refine(tree, q, sqrt_d, n_clust)
        sizes = [c[2] for c in clusters]
        print(f"\n  Refinement to {n_clust} clusters:")
        print(f"    Got {len(clusters)} clusters")
        print(f"    Sizes: min={min(sizes)}, max={max(sizes)}, "
              f"median={sorted(sizes)[len(sizes)//2]}")
        total = sum(sizes)
        print(f"    Total keys covered: {total}/{len(cand)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
