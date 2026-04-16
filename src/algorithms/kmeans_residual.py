"""
Two-level KMeans attention approximation with residual
correction.

Offline:
1. Cluster all keys into C coarse clusters.
2. Store per-cluster mean key, summed values, counts.
3. Recenter keys within each cluster and subcluster
   residuals into R centroids.

Query-time:
1. Score all coarse clusters using mean keys.
2. Select top-k coarse clusters for refinement.
3. Non-refined clusters: mean-only approximation.
4. Refined clusters: residual subcluster mixture.

For key k_i in cluster c:
    k_i = mu_c + r_i,  r_i ≈ rho_{c,j(i)}

Refined cluster contribution:
    exp(q^T mu_c / √d) · Σ_j exp(q^T rho_{c,j} / √d)
      · S_{c,j}

This reduces within-cluster bias while keeping query-time
cost at O(C·d + k·R·d) instead of O(N·d).
"""

import numpy as np
from typing import List, Dict, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax, flat_kmeans


# ═══════════════════════════════════════════════════════
# Offline helpers
# ═══════════════════════════════════════════════════════

def _precompute_coarse_stats(
    keys: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    C: int,
) -> Dict:
    """
    Per-cluster mean keys, sum values, counts, members.

    Shapes:
      mean_keys:      [C, d]
      sum_values:     [C, d_v]
      counts:         [C]
      member_indices: list of int64 arrays
    """
    d = keys.shape[1]
    d_v = values.shape[1]

    mean_keys = np.zeros((C, d), dtype=np.float32)
    sum_values = np.zeros((C, d_v), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int32)
    member_indices = [None] * C

    for c in range(C):
        mask = labels == c
        count = int(np.sum(mask))
        counts[c] = count
        if count > 0:
            mean_keys[c] = np.mean(keys[mask], axis=0)
            sum_values[c] = np.sum(
                values[mask], axis=0,
            )
            member_indices[c] = np.where(mask)[0]
        else:
            member_indices[c] = np.array(
                [], dtype=np.int64,
            )

    return {
        "mean_keys": mean_keys,
        "sum_values": sum_values,
        "counts": counts,
        "member_indices": member_indices,
    }


def _precompute_residual_stats(
    keys: np.ndarray,
    values: np.ndarray,
    coarse: Dict,
    n_residuals: int,
    seed: int = 42,
) -> List[Optional[Dict]]:
    """
    Per-cluster residual subclustering.

    For each coarse cluster c:
    1. residuals = keys[members] - mean_key[c]
    2. KMeans on residuals -> R_c subclusters
    3. Store centroids, counts, sum_values, local
       member indices (into coarse member array)

    Returns list of dicts (one per cluster), None
    for empty clusters.
    """
    C = len(coarse["counts"])
    mean_keys = coarse["mean_keys"]
    member_indices = coarse["member_indices"]
    stats = [None] * C

    if n_residuals <= 0:
        return stats

    for c in range(C):
        idx = member_indices[c]
        count = len(idx)
        if count == 0:
            continue

        cluster_keys = keys[idx]
        cluster_vals = values[idx]
        residuals = cluster_keys - mean_keys[c]

        R_c = min(n_residuals, count)
        if R_c <= 1:
            stats[c] = {
                "centroids": np.mean(
                    residuals, axis=0, keepdims=True,
                ).astype(np.float32),
                "counts": np.array(
                    [count], dtype=np.int32,
                ),
                "sum_values": np.sum(
                    cluster_vals, axis=0,
                    keepdims=True,
                ).astype(np.float32),
                "local_members": [
                    np.arange(count, dtype=np.int64),
                ],
            }
            continue

        _, sub_labels = flat_kmeans(
            residuals, R_c, seed=seed + c,
        )

        d = residuals.shape[1]
        d_v = cluster_vals.shape[1]
        centroids = np.zeros(
            (R_c, d), dtype=np.float32,
        )
        sub_counts = np.zeros(R_c, dtype=np.int32)
        sub_sv = np.zeros(
            (R_c, d_v), dtype=np.float32,
        )
        sub_members = [None] * R_c

        for j in range(R_c):
            mask = sub_labels == j
            cnt = int(np.sum(mask))
            sub_counts[j] = cnt
            if cnt > 0:
                centroids[j] = np.mean(
                    residuals[mask], axis=0,
                )
                sub_sv[j] = np.sum(
                    cluster_vals[mask], axis=0,
                )
                sub_members[j] = np.where(mask)[0]
            else:
                sub_members[j] = np.array(
                    [], dtype=np.int64,
                )

        valid = sub_counts > 0
        stats[c] = {
            "centroids": centroids[valid],
            "counts": sub_counts[valid],
            "sum_values": sub_sv[valid],
            "local_members": [
                sub_members[j]
                for j in range(R_c) if valid[j]
            ],
        }

    return stats


# ═══════════════════════════════════════════════════════
# Query-time causal filtering
# ═══════════════════════════════════════════════════════

def _causal_filter(
    coarse: Dict,
    residuals: List[Optional[Dict]],
    n_causal: int,
    special_set: set,
    keys: np.ndarray,
    values: np.ndarray,
):
    """
    Filter clusters and subclusters to causal window,
    removing special-token positions. Recomputes active
    coarse means and residual centroids for the prefix.

    Returns:
      causal_counts:     [C] int32
      causal_mean_keys:  [C, d] float64
      causal_sv:         [C, d_v] float64
      valid_mask:        [C] bool
      filtered_res:      list[Optional[Dict]] per cluster
    """
    members = coarse["member_indices"]
    C = len(coarse["counts"])
    d = keys.shape[1]
    d_v = values.shape[1]

    causal_counts = np.zeros(C, dtype=np.int32)
    causal_mean_keys = np.zeros(
        (C, d), dtype=np.float64,
    )
    causal_sv = np.zeros((C, d_v), dtype=np.float64)
    valid_mask = np.zeros(C, dtype=bool)
    filtered_res = [None] * C

    for c in range(C):
        idx = members[c]
        if len(idx) == 0:
            continue

        keep = idx < n_causal
        if special_set:
            for s in special_set:
                keep &= (idx != s)
        c_idx = idx[keep]

        cnt = len(c_idx)
        if cnt == 0:
            continue

        causal_counts[c] = cnt
        mu_c = np.mean(keys[c_idx], axis=0)
        causal_mean_keys[c] = mu_c
        causal_sv[c] = np.sum(
            values[c_idx], axis=0,
        )
        valid_mask[c] = True

        rs = residuals[c] if residuals else None
        if rs is None:
            continue

        new_centroids = []
        new_counts = []
        new_sv = []

        for j in range(len(rs["counts"])):
            local = rs["local_members"][j]
            if len(local) == 0:
                continue
            g_idx = idx[local]
            sub_keep = g_idx < n_causal
            if special_set:
                for s in special_set:
                    sub_keep &= (g_idx != s)
            g_idx = g_idx[sub_keep]

            sub_cnt = len(g_idx)
            if sub_cnt == 0:
                continue

            rho = np.mean(
                keys[g_idx] - mu_c[None, :],
                axis=0,
            )
            sv = np.sum(values[g_idx], axis=0)

            new_centroids.append(rho)
            new_counts.append(sub_cnt)
            new_sv.append(sv)

        if new_counts:
            filtered_res[c] = {
                "centroids": np.asarray(
                    new_centroids, dtype=np.float64,
                ),
                "counts": np.asarray(
                    new_counts, dtype=np.int32,
                ),
                "sum_values": np.asarray(
                    new_sv, dtype=np.float64,
                ),
            }

    return (
        causal_counts, causal_mean_keys,
        causal_sv, valid_mask, filtered_res,
    )


# ═══════════════════════════════════════════════════════
# Algorithm
# ═══════════════════════════════════════════════════════

class KMeansResidualClustering(AttentionAlgorithm):
    """
    Two-level KMeans attention approximation.

    Offline:
    1. Cluster all keys into C coarse clusters.
    2. Store per-cluster mean key, summed values, counts.
    3. Recenter keys and subcluster residuals into R
       centroids per cluster.

    Query-time:
    1. Score coarse clusters: q^T mu_c / √d + log(n_c).
    2. Top-k clusters get residual refinement.
    3. Others use mean-only approximation.

    For key k_i in cluster c:
        k_i = mu_c + r_i,  r_i ≈ rho_{c,j(i)}

    Refined contribution:
        exp(q^T mu_c/√d) · Σ_j exp(q^T rho_j/√d) · S_j

    The final output is a softmax over all items:
    special tokens (exact), non-refined cluster means
    (count-corrected), and refined subcluster centroids
    (count-corrected).

    Coarse-to-fine tradeoff:
    - All clusters get cheap mean-only approximation
    - Only top clusters get refined by residuals
    - Increasing top_k_refine → more compute
    - Increasing n_residuals → more fidelity
    """

    def __init__(
        self,
        n_clusters: int = 512,
        n_residuals: int = 8,
        top_k_refine: int = 16,
    ):
        self.n_clusters = n_clusters
        self.n_residuals = n_residuals
        self.top_k_refine = top_k_refine
        self._coarse = None
        self._residuals = None

    @property
    def name(self) -> str:
        return (
            f"KMeansRes-C{self.n_clusters}"
            f"-r{self.n_residuals}"
            f"-{self.top_k_refine}"
        )

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Step A: coarse KMeans on non-sink keys.
        Step B: compute residuals per cluster.
        Step C: residual KMeans within each cluster.

        Position 0 (sink token) is excluded from
        clustering so it doesn't bias centroids. It
        is handled as a special token at query time.
        """
        # Exclude sink, center keys (doesn't affect softmax)
        k_nosink = keys[1:]
        v_nosink = values[1:]
        self._key_mean = np.mean(k_nosink, axis=0)
        k_centered = k_nosink - self._key_mean

        _, labels = flat_kmeans(
            k_centered, self.n_clusters, seed=seed,
        )

        coarse = _precompute_coarse_stats(
            k_centered, v_nosink, labels,
            self.n_clusters,
        )
        # Shift member indices back to global positions
        for c in range(self.n_clusters):
            coarse["member_indices"][c] = (
                coarse["member_indices"][c] + 1
            )

        self._coarse = coarse
        # Build full centered key array for residual stats
        keys_centered = keys.copy()
        keys_centered[1:] -= self._key_mean
        keys_centered[0] -= self._key_mean
        self._residuals = _precompute_residual_stats(
            keys_centered, values, self._coarse,
            self.n_residuals, seed=seed,
        )

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        if self._coarse is None:
            raise RuntimeError(
                "Call prepare() before run()"
            )

        q = problem.query
        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        special_idx = problem.special_idx
        special_set = problem.special_set
        n_causal = len(keys)
        sqrt_d = np.sqrt(head_dim)
        d_v = values.shape[1]

        # Center keys to match prepare()
        keys_c = keys - self._key_mean

        # ── Causal filtering (recomputes active stats) ──
        cc, c_mk, c_sv, vm, fres = _causal_filter(
            self._coarse, self._residuals,
            n_causal, special_set, keys_c, values,
        )

        valid = np.where(vm)[0]
        n_special = (
            len(special_idx)
            if special_idx is not None else 0
        )

        if len(valid) == 0 and n_special == 0:
            return AttentionOutput(
                output=np.zeros(d_v, dtype=np.float32),
                actual_budget=0,
            )

        # ── Build (score, value) items for softmax ──
        #
        # Each item has:
        #   score  = logit + log(count)
        #   value  = sum_values / count  (average)
        #
        # Softmax over scores gives count-corrected
        # weights; weighted sum of avg values gives the
        # approximate attention output.

        all_scores = []
        all_values = []

        # Special tokens: exact logits (centered)
        if n_special > 0:
            sp_logits = (
                q @ keys_c[special_idx].T / sqrt_d
            )
            all_scores.append(sp_logits)
            all_values.append(
                values[special_idx].astype(np.float64)
            )

        # Score and rank coarse clusters
        if len(valid) > 0:
            mu_logits = (
                c_mk[valid] @ q / sqrt_d
            )
            rank_scores = mu_logits + np.log(
                cc[valid].astype(np.float64),
            )
            order = np.argsort(rank_scores)[::-1]
            sorted_valid = valid[order]
            sorted_mu = mu_logits[order]

            top_k = min(
                self.top_k_refine,
                len(sorted_valid),
            )

            for i in range(len(sorted_valid)):
                c = sorted_valid[i]
                mu_l = sorted_mu[i]

                # Refined cluster: use subclusters
                if (
                    i < top_k
                    and fres[c] is not None
                ):
                    rho = fres[c]["centroids"]
                    r_cnt = fres[c]["counts"]
                    r_sv = fres[c]["sum_values"]

                    active = r_cnt > 0
                    if np.any(active):
                        rho_a = rho[active]
                        cnt_a = r_cnt[active].astype(
                            np.float64,
                        )
                        sv_a = r_sv[active]

                        # score = mu + rho·q/√d + log(n)
                        r_logits = (
                            mu_l
                            + rho_a @ q / sqrt_d
                            + np.log(cnt_a)
                        )
                        avg_v = sv_a / cnt_a[:, None]

                        all_scores.append(r_logits)
                        all_values.append(avg_v)
                        continue

                # Non-refined: single mean entry
                n_c = float(cc[c])
                if n_c > 0:
                    score = mu_l + np.log(n_c)
                    avg_v = c_sv[c] / n_c
                    all_scores.append(
                        np.array([score]),
                    )
                    all_values.append(
                        avg_v[None, :],
                    )

        if not all_scores:
            return AttentionOutput(
                output=np.zeros(d_v, dtype=np.float32),
                actual_budget=0,
            )

        scores = np.concatenate(all_scores)
        vals = np.vstack(all_values)

        w = softmax(scores)
        output = (w @ vals).astype(np.float32)

        return AttentionOutput(
            output=output,
            actual_budget=len(scores),
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        C = cfg.get("n_clusters", 512)
        for r in cfg.get(
            "n_residuals_sweep", [0, 2, 8, 16],
        ):
            if r == 0:
                # Mean-only: top_k irrelevant
                instances.append(
                    KMeansResidualClustering(
                        n_clusters=C,
                        n_residuals=0,
                        top_k_refine=0,
                    )
                )
            else:
                for k in cfg.get(
                    "top_k_refine_sweep",
                    [1, 2, 4, 8, 16, 32, 64],
                ):
                    instances.append(
                        KMeansResidualClustering(
                            n_clusters=C,
                            n_residuals=r,
                            top_k_refine=k,
                        )
                    )
        return instances
