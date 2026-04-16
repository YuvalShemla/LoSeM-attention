"""
Hierarchical K-Means with best-first expansion.

Offline: build a multi-level tree via bisecting k-means.
Each level splits nodes into B_i children (e.g. 32 -> 16 -> 8).

Online: start with top-level clusters as frontier. Greedily
expand the highest-scoring frontier node (by mean_key @ q / sqrt_d)
into its children, until frontier_size >= budget. Unexpanded
nodes become group representatives (mean key + mean value +
log(count) scoring). Small leaves become individual keys.

Coverage is always 100% — every key is in some group or
taken individually. Budget controls resolution, not coverage.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import heapq

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, flat_kmeans


class _TreeNode:
    """Compact tree node for bisecting k-means."""
    __slots__ = [
        'member_idx', 'mean_key', 'count',
        'children', 'is_leaf',
    ]

    def __init__(
        self,
        member_idx: np.ndarray,
        keys: np.ndarray,
    ):
        self.member_idx = member_idx
        self.count = len(member_idx)
        if self.count > 0:
            self.mean_key = keys[member_idx].mean(
                axis=0,
            ).astype(np.float32)
        else:
            self.mean_key = np.zeros(
                keys.shape[1], dtype=np.float32,
            )
        self.children: List['_TreeNode'] = []
        self.is_leaf = True


def _build_tree(
    keys: np.ndarray,
    branching: List[int],
    seed: int = 42,
    min_leaf: int = 4,
) -> _TreeNode:
    """
    Build hierarchical k-means tree.

    Parameters:
        keys: [N, d] candidate keys
        branching: cluster counts per level, e.g. [32, 16, 8]
        seed: random seed
        min_leaf: don't split nodes smaller than this
    """
    root = _TreeNode(np.arange(len(keys)), keys)

    def split_node(node, level, parent_seed):
        if level >= len(branching):
            return
        if node.count <= min_leaf:
            return
        nc = min(branching[level], node.count)
        if nc <= 1:
            return

        sub = keys[node.member_idx]
        centroids, labels = flat_kmeans(
            sub, nc, seed=parent_seed, n_iter=10,
        )

        for c in range(nc):
            mask = labels == c
            if mask.sum() == 0:
                continue
            child = _TreeNode(
                node.member_idx[mask], keys,
            )
            node.children.append(child)

        if node.children:
            node.is_leaf = False
            for ci, child in enumerate(node.children):
                split_node(
                    child, level + 1,
                    parent_seed * 100 + ci + 1,
                )

    split_node(root, 0, seed)
    return root


def _best_first_expand(
    root: _TreeNode,
    query: np.ndarray,
    sqrt_d: float,
    budget: int,
) -> List[_TreeNode]:
    """
    Best-first expansion of the tree.

    Returns frontier nodes that partition all keys.
    Budget = number of frontier entries (groups + individuals).
    """
    frontier = []
    uid = 0

    for child in root.children:
        score = float(child.mean_key @ query / sqrt_d)
        heapq.heappush(frontier, (-score, uid, child))
        uid += 1

    frontier_size = len(frontier)

    while frontier_size < budget:
        # Find highest-scoring expandable node
        found = False
        for item in sorted(frontier):
            node = item[2]
            if not node.is_leaf and node.children:
                extra = len(node.children) - 1
                if frontier_size + extra <= budget:
                    frontier.remove(item)
                    heapq.heapify(frontier)
                    for child in node.children:
                        score = float(
                            child.mean_key @ query / sqrt_d
                        )
                        heapq.heappush(
                            frontier,
                            (-score, uid, child),
                        )
                        uid += 1
                    frontier_size += extra
                    found = True
                    break
        if not found:
            break

    return [item[2] for item in frontier]


class HierarchicalKMeans(AttentionAlgorithm):
    """
    Hierarchical K-Means with best-first expansion.

    Parameters:
        branching: cluster counts per level, e.g. [32, 16, 8]
        leaf_threshold: nodes with <= this many keys are
            treated as individuals (exact logits)
    """

    _next_id = 0

    def __init__(
        self,
        branching: List[int] = None,
        leaf_threshold: int = 8,
    ):
        if branching is None:
            branching = [32, 16, 8]
        self._branching = branching
        self._leaf_threshold = leaf_threshold
        self._tree = None
        self._id = HierarchicalKMeans._next_id
        HierarchicalKMeans._next_id += 1

    @property
    def name(self) -> str:
        tag = "x".join(str(b) for b in self._branching)
        return f"HierKM-{tag}-{self._id}"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: Optional[np.ndarray] = None,
        query_positions: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        self._tree = _build_tree(
            keys, self._branching,
            seed=seed, min_leaf=4,
        )
        self._head_dim = head_dim

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._tree is None:
            raise RuntimeError("Call prepare() first")

        query = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        sqrt_d = np.sqrt(head_dim)
        n_causal = len(keys)

        if len(candidate_idx) == 0:
            from ..core import subset_attention
            output = subset_attention(
                logits, values, special_idx,
            )
            return AttentionOutput(
                output=output,
                actual_budget=len(special_idx),
            )

        # Expand tree
        nodes = _best_first_expand(
            self._tree, query, sqrt_d, budget,
        )

        # Build score and value arrays
        scores_list = []
        values_list = []

        # Special keys (exact logits)
        for si in special_idx:
            scores_list.append(float(logits[si]))
            values_list.append(
                values[si].astype(np.float64),
            )

        n_indiv = 0
        n_groups = 0

        special_set = set(special_idx.tolist())

        for node in nodes:
            # node.member_idx are indices into the full
            # keys array from prepare(). Filter to causal
            # window and exclude special tokens.
            gidx = node.member_idx[
                node.member_idx < n_causal
            ]
            gidx = gidx[
                ~np.isin(gidx, special_idx)
            ]
            if len(gidx) == 0:
                continue

            if len(gidx) <= self._leaf_threshold:
                # Individual keys
                for gi in gidx:
                    scores_list.append(float(logits[gi]))
                    values_list.append(
                        values[gi].astype(np.float64),
                    )
                n_indiv += len(gidx)
            else:
                # Group representative
                mk = keys[gidx].astype(
                    np.float64,
                ).mean(axis=0)
                mv = values[gidx].astype(
                    np.float64,
                ).mean(axis=0)
                score = float(
                    mk @ query / sqrt_d
                ) + np.log(len(gidx))
                scores_list.append(score)
                values_list.append(mv)
                n_groups += 1

        sc = np.array(scores_list, dtype=np.float64)
        vl = np.stack(values_list)

        w = softmax(sc)
        output = (w @ vl).astype(np.float32)

        actual_budget = len(special_idx) + n_indiv + n_groups

        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        combos = cfg.get("branching_combos", [
            [32, 16, 8],
            [64, 16, 4],
            [16, 8, 4, 2],
            [64, 8, 8],
            [128, 32],
        ])
        leaf_threshold = cfg.get("leaf_threshold", 8)
        instances = []
        for branching in combos:
            instances.append(HierarchicalKMeans(
                branching=branching,
                leaf_threshold=leaf_threshold,
            ))
        return instances
