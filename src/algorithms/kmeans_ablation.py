"""
KMeans Key/Value ablation: independently control whether
the cluster representative uses centroid (mean) or medoid
(nearest-to-centroid) for keys and values.

All variants use count-corrected scoring: q @ rep_key / √d
+ log(count). Only the key and value representations differ.

Offline cost: KMeans on N keys (cached).
Per-query cost: O(N) per cluster for centroid/medoid lookup.
"""

import numpy as np
from typing import List

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, cached_flat_kmeans


class KMeansAblation(AttentionAlgorithm):
    """
    Ablation: independently vary key and value
    representations (centroid vs medoid) while keeping
    everything else identical.
    """

    def __init__(
        self,
        n_clusters: int = 256,
        key_mode: str = "centroid",
        val_mode: str = "centroid",
    ):
        self.n_clusters = n_clusters
        self.key_mode = key_mode
        self.val_mode = val_mode
        self._member_indices = None

    @property
    def name(self) -> str:
        km = "CentK" if self.key_mode == "centroid" else "MedK"
        vm = "CentV" if self.val_mode == "centroid" else "MedV"
        return f"{km}-{vm}-{self.n_clusters}"

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        _, labels = cached_flat_kmeans(
            keys, self.n_clusters, seed=seed,
        )
        C = self.n_clusters
        self._member_indices = [None] * C
        for c in range(C):
            mask = labels == c
            if np.any(mask):
                self._member_indices[c] = (
                    np.where(mask)[0]
                )
            else:
                self._member_indices[c] = np.array(
                    [], dtype=np.int64,
                )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._member_indices is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query
        keys = problem.keys
        values = problem.values
        logits = problem.logits
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)
        sqrt_d = np.sqrt(head_dim)
        C = self.n_clusters

        valid = []
        causal_members = [None] * C
        for c in range(C):
            idx = self._member_indices[c]
            if len(idx) == 0:
                causal_members[c] = np.array(
                    [], dtype=np.int64,
                )
                continue
            idx = idx[idx < n_causal]
            if special_set:
                keep = np.ones(len(idx), dtype=bool)
                for s in special_set:
                    keep &= (idx != s)
                idx = idx[keep]
            causal_members[c] = idx
            if len(idx) > 0:
                valid.append(c)

        n_special = (
            len(special_idx)
            if special_idx is not None else 0
        )
        n_total = n_special + len(valid)
        if n_total == 0:
            return AttentionOutput(
                output=np.zeros(head_dim),
                actual_budget=0,
            )

        scores = np.empty(n_total)
        out_vals = np.empty((n_total, head_dim))

        if n_special > 0:
            scores[:n_special] = logits[special_idx]
            out_vals[:n_special] = values[special_idx]

        off = n_special
        for fi, c in enumerate(valid):
            idx = causal_members[c]
            count = len(idx)
            ck = keys[idx]
            cv = values[idx]

            mean_k = np.mean(ck, axis=0)

            # Find medoid (nearest key to centroid)
            dists = np.sum(
                (ck - mean_k) ** 2, axis=1,
            )
            med_local = np.argmin(dists)

            # Key representation
            if self.key_mode == "centroid":
                rep_key = mean_k
            else:
                rep_key = ck[med_local]

            # Value representation
            if self.val_mode == "centroid":
                rep_val = np.mean(cv, axis=0)
            else:
                rep_val = cv[med_local]

            scores[off + fi] = (
                q @ rep_key / sqrt_d
                + np.log(count)
            )
            out_vals[off + fi] = rep_val

        w = softmax(scores)
        output = w @ out_vals

        return AttentionOutput(
            output=output.astype(np.float32),
            actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        clusters_list = cfg.get(
            "n_clusters_sweep",
            [cfg.get("n_clusters", 256)],
        )
        combos = cfg.get("combos", [
            {"key_mode": "centroid", "val_mode": "centroid"},
            {"key_mode": "medoid", "val_mode": "centroid"},
            {"key_mode": "centroid", "val_mode": "medoid"},
            {"key_mode": "medoid", "val_mode": "medoid"},
        ])
        instances = []
        for n_c in clusters_list:
            for combo in combos:
                instances.append(KMeansAblation(
                    n_clusters=n_c,
                    key_mode=combo["key_mode"],
                    val_mode=combo["val_mode"],
                ))
        return instances
