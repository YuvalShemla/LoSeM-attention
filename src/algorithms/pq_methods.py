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
        cand_mask = np.zeros(n, dtype=bool)
        cand_mask[candidate_idx] = True

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
        return [VAttentionPQ(m=m) for m in m_list]


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
        cand_mask = np.zeros(n, dtype=bool)
        cand_mask[candidate_idx] = True

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
        return [
            IVFPQCluster(
                n_cells=nc, nprobe=np, m=m,
            )
            for np in nprobe_list
        ]
