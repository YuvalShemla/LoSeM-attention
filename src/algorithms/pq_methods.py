"""
PQ-based approximate top-k methods.

1. vAttention(PQ): vAttention with PQ-approximate top-k + uniform IS.
2. IVF-PQ-Cluster: IVF coarse quantizer on keys provides both
   approximate top-k AND cluster residuals from un-probed cells.
"""

import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .pq_topk import PQIndex, IVFPQIndex
from ..core import softmax


def _full_index_candidate_mask(index_len: int, candidate_idx: np.ndarray) -> np.ndarray:
    """Mask over a full prepared index, allowing only current causal candidates."""
    mask = np.zeros(index_len, dtype=bool)
    valid = candidate_idx[candidate_idx < index_len]
    mask[valid] = True
    return mask


class VAttentionPQ(AttentionAlgorithm):
    """
    vAttention with PQ-approximate top-k.

    Same IS-corrected attention as vAttention(oracle),
    but uses Product Quantization to find approximate
    top-k keys instead of exact logits.

    Budget split: half top-k (via PQ), half uniform.
    """

    def __init__(self, m: int = 8, n_codes: int = 256):
        self.m = m
        self.n_codes = n_codes
        self._pq = None
        self._seed = 42

    @property
    def name(self) -> str:
        return f"vAttention(PQ-m{self.m})"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None,
                seed=42):
        self._seed = seed
        self._pq = PQIndex(
            m=self.m, n_codes=self.n_codes, seed=seed,
        )
        self._pq.fit(keys)

    def run(self, problem: AttentionInput, budget: int,
            rng: np.random.Generator) -> AttentionOutput:
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)
        n = len(problem.keys)
        d = values.shape[1]

        if n_cand == 0:
            out = softmax(logits[special_idx]) @ values[special_idx]
            return AttentionOutput(
                output=out, actual_budget=len(special_idx),
            )

        buse = min(budget, n_cand)
        b_topk = buse // 2
        b_sample = buse - b_topk

        # PQ approximate top-k
        cand_mask = _full_index_candidate_mask(
            len(self._pq.codes), candidate_idx,
        )

        if b_topk > 0:
            topk_global = self._pq.approximate_topk(
                problem.query, b_topk,
                candidate_mask=cand_mask,
            )
        else:
            topk_global = np.array([], dtype=np.int64)

        # Uniform sample from remaining candidates
        topk_set = set(topk_global.tolist())
        remaining = np.array([
            i for i in candidate_idx if i not in topk_set
        ], dtype=np.int64)
        n_s = len(remaining)

        if b_sample > 0 and n_s > 0:
            n_sample = min(b_sample, n_s)
            sampled_global = rng.choice(
                remaining, size=n_sample, replace=False,
            )
        else:
            sampled_global = np.array([], dtype=np.int64)
            n_sample = 0

        # IS-corrected attention (vAttention Eq. 5)
        fixed_idx = np.concatenate(
            [special_idx, topk_global],
        ).astype(np.int64)

        all_sel = np.concatenate([
            fixed_idx,
            sampled_global,
        ]).astype(np.int64) if n_sample > 0 else fixed_idx

        all_logits = logits[all_sel].astype(np.float64)
        s_max = np.max(all_logits) if len(all_logits) > 0 else 0.0

        # Fixed part
        fixed_s = logits[fixed_idx].astype(np.float64)
        fixed_exp = np.exp(fixed_s - s_max)
        N_f = (
            fixed_exp[:, None]
            * values[fixed_idx].astype(np.float64)
        ).sum(axis=0)
        D_f = fixed_exp.sum()

        # Sampled part
        if n_sample > 0 and n_s > 0:
            samp_s = logits[sampled_global].astype(np.float64)
            samp_exp = np.exp(samp_s - s_max)
            weight = float(n_s) / float(n_sample)
            N_dyn = weight * (
                samp_exp[:, None]
                * values[sampled_global].astype(np.float64)
            ).sum(axis=0)
            D_dyn = weight * samp_exp.sum()
        else:
            N_dyn = np.zeros(d, dtype=np.float64)
            D_dyn = 0.0

        D_total = D_f + D_dyn
        if D_total < 1e-30:
            output = np.zeros(d, dtype=np.float32)
        else:
            output = ((N_f + N_dyn) / D_total).astype(
                np.float32,
            )

        actual_budget = len(fixed_idx) + n_sample
        return AttentionOutput(
            output=output, actual_budget=actual_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        m_list = cfg.get("m_sweep", [8])
        n_codes = int(cfg.get("n_codes", 256))
        return [
            VAttentionPQ(m=m, n_codes=n_codes)
            for m in m_list
        ]


class IVFPQCluster(AttentionAlgorithm):
    """
    IVF-PQ with cluster residuals.

    Coarse KMeans on keys creates Voronoi cells that serve
    double duty:
      1. Probe nearest cells → PQ top-k within for individual
         exact attention.
      2. Un-probed cells → mean_key + mean_value cluster reps
         with log(count) scoring.

    Joint softmax over: special + topk + unprobed cluster reps.

    Budget param = number of individual keys from probed cells.
    nprobe controls how many cells to search (more probes =
    better top-k recall but fewer cluster reps).
    """

    def __init__(self, n_cells: int = 1024,
                 nprobe: int = 32,
                 m: int = 8, n_codes: int = 256):
        self.n_cells = n_cells
        self.nprobe = nprobe
        self.m = m
        self.n_codes = n_codes
        self._ivfpq = None
        self._seed = 42

    @property
    def name(self) -> str:
        return (
            f"IVFPQ-C{self.n_cells}"
            f"-p{self.nprobe}"
        )

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None,
                seed=42):
        self._seed = seed
        self._ivfpq = IVFPQIndex(
            n_cells=self.n_cells,
            m=self.m, n_codes=self.n_codes,
            seed=seed,
        )
        self._ivfpq.fit(keys, values)

    def run(self, problem: AttentionInput, budget: int,
            rng: np.random.Generator) -> AttentionOutput:
        q = problem.query
        keys = problem.keys
        values = problem.values
        head_dim = problem.head_dim
        logits = problem.logits
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        sqrt_d = np.sqrt(head_dim)
        n_cand = len(candidate_idx)
        n = len(keys)
        d = values.shape[1]

        if n_cand == 0:
            out = softmax(logits[special_idx]) @ values[special_idx]
            return AttentionOutput(
                output=out, actual_budget=len(special_idx),
            )

        buse = min(budget, n_cand)

        # Build candidate mask (exclude special)
        cand_mask = _full_index_candidate_mask(
            len(self._ivfpq.pq.codes), candidate_idx,
        )

        # IVF-PQ search: probe nearest cells, PQ top-k
        topk_global, probed_set, unprobed_info = (
            self._ivfpq.search(
                q, buse, self.nprobe,
                candidate_mask=cand_mask,
            )
        )
        n_topk = len(topk_global)

        # Build joint softmax:
        # special + topk individuals + unprobed cluster reps
        n_sp = len(special_idx)
        n_unprobed = len(unprobed_info)
        n_total = n_sp + n_topk + n_unprobed

        scores = np.empty(n_total, dtype=np.float64)
        out_vals = np.empty((n_total, d), dtype=np.float32)

        # Special
        scores[:n_sp] = logits[special_idx].astype(np.float64)
        out_vals[:n_sp] = values[special_idx]

        # TopK individuals (use TRUE logits)
        off = n_sp
        if n_topk > 0:
            scores[off:off + n_topk] = (
                logits[topk_global].astype(np.float64)
            )
            out_vals[off:off + n_topk] = values[topk_global]

        # Unprobed cluster reps
        off = n_sp + n_topk
        for i, (c, cnt, mean_k, mean_v) in enumerate(
            unprobed_info,
        ):
            scores[off + i] = (
                float(q @ mean_k) / sqrt_d
                + np.log(cnt)
            )
            out_vals[off + i] = mean_v

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals

        return AttentionOutput(
            output=output, actual_budget=n_total,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        nc = cfg.get("n_cells", 1024)
        nprobe_list = cfg.get("nprobe_sweep", [32])
        m = cfg.get("m", 8)
        n_codes = int(cfg.get("n_codes", 256))
        return [
            IVFPQCluster(
                n_cells=nc, nprobe=np, m=m, n_codes=n_codes,
            )
            for np in nprobe_list
        ]


class FullAttentionPQ(AttentionAlgorithm):
    """
    Budgeted full attention using symmetric PQ logits.

    Special indices (sink/local) keep exact logits.
    Candidate logits are estimated with quantized query +
    quantized keys:

      l_hat_j = sum_s <c_q[s], c_kj[s]> / sqrt(d)

    where c_q[s] is the nearest codeword for query subspace s,
    and c_kj[s] is key j's stored subspace codeword.

    For budget B:
      1) choose top-B candidates by estimated l_hat_j
      2) replace those B scores with exact logits
      3) keep PQ estimates for the remaining candidates
      4) run one joint softmax over special + all candidates
    """

    def __init__(self, m: int = 8, n_codes: int = 256):
        self.m = m
        self.n_codes = n_codes
        self._pq = None
        self._sqrt_d = None

    @property
    def name(self) -> str:
        return f"FullAttentionPQ_topk-m{self.m}"

    @property
    def sweeps_budget(self) -> bool:
        return True

    def prepare(self, keys, values, head_dim,
                queries=None, query_positions=None,
                seed=42):
        self._pq = PQIndex(
            m=self.m, n_codes=self.n_codes, seed=seed,
        )
        self._pq.fit(keys)
        self._sqrt_d = float(np.sqrt(head_dim))

    def run(self, problem: AttentionInput, budget: int,
            rng: np.random.Generator) -> AttentionOutput:
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_total = len(problem.keys)
        d = values.shape[1]

        n_cand = len(candidate_idx)
        if n_cand == 0:
            out = softmax(logits[special_idx]) @ values[special_idx]
            return AttentionOutput(
                output=out,
                actual_budget=len(special_idx),
                selected_indices=special_idx,
            )

        # Asymmetric PQ logits: exact query subvector against
        # quantized key codewords in each subspace.
        query = problem.query.astype(np.float32, copy=False)
        m = self._pq.m
        dsub = self._pq.dsub
        n_codes = self._pq.codebooks.shape[1]
        lut = np.empty((m, n_codes), dtype=np.float32)
        for i in range(m):
            q_sub = query[i * dsub:(i + 1) * dsub]
            lut[i] = self._pq.codebooks[i] @ q_sub

        # Approximate logits for all keys, then keep candidates only.
        approx_logits = np.zeros(n_total, dtype=np.float64)
        key_codes = self._pq.codes[:n_total]
        for i in range(m):
            approx_logits += lut[i][key_codes[:, i]].astype(np.float64)
        approx_logits /= self._sqrt_d

        scores = np.empty(n_total, dtype=np.float64)
        scores[special_idx] = logits[special_idx].astype(np.float64)
        scores[candidate_idx] = approx_logits[candidate_idx]

        # Top-B by PQ-estimated logits, then overwrite with exact logits.
        b_exact = min(max(int(budget), 0), n_cand)
        if b_exact > 0:
            cand_approx = approx_logits[candidate_idx]
            if b_exact < n_cand:
                top_local = np.argpartition(
                    cand_approx, -b_exact
                )[-b_exact:]
            else:
                top_local = np.arange(n_cand)
            top_global = candidate_idx[top_local]
            scores[top_global] = logits[top_global].astype(np.float64)
        else:
            top_global = np.array([], dtype=np.int64)

        p_true = softmax(logits.astype(np.float64)).astype(np.float64)
        p_est = softmax(scores).astype(np.float64)
        output = p_est.astype(np.float32) @ values.astype(np.float32)
        lg_m = float(np.max(logits))
        z_true = float(np.exp(lg_m) * np.sum(np.exp(logits - lg_m)))
        sc_m = float(np.max(scores))
        z_est = float(np.exp(sc_m) * np.sum(np.exp(scores - sc_m)))
        # Compute exp(scores) / Z_true stably in log space.
        log_z_true = lg_m + np.log(
            max(np.sum(np.exp(logits - lg_m)), 1e-30)
        )
        p_est_true_z = np.exp(scores - log_z_true)
        sel = np.concatenate([special_idx, top_global]).astype(np.int64)
        return AttentionOutput(
            output=output,
            actual_budget=len(special_idx) + b_exact,
            selected_indices=sel,
            debug_payload={
                "p_true": p_true.astype(np.float32),
                "p_est_true_z": p_est_true_z.astype(np.float32),
                "logits_true": logits.astype(np.float32),
                "logits_est": scores.astype(np.float32),
                "z_true": z_true,
                "z_est": z_est,
                "requested_budget": int(budget),
            },
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        m_list = cfg.get("m_sweep", [8])
        n_codes = int(cfg.get("n_codes", 256))
        return [
            FullAttentionPQ(m=m, n_codes=n_codes)
            for m in m_list
        ]
