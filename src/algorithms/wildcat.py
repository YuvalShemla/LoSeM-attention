"""
WILDCAT: Weighted Iterative Low-rank Decomposition for
Coreset ATtention (Schröder & Mackey, 2026).

Selects a weighted coreset of keys via Randomly Pivoted
Nyström (RPNYS) and uses Nyström-optimal weights to
approximate attention. The coreset+weights are computed
once (query-independent) and reused for all queries.

Algorithm:
  1. Centre keys: K_c = K - mean(K)
  2. Compute temperature τ (Eq. 4) to balance
     low-rank approximability vs query inflation.
  3. Run RPNYS (Alg. 1) on scaled keys K' = K_c/τ
     with kernel h(a,b) = exp(β ⟨a,b⟩).
     Produces pivot indices S and kernel inverse M
     such that Nyström weights W = M · R.
  4. At query time (WTDATTN, Alg. 3):
     - Special tokens: exact attention.
     - Candidates: Nyström-weighted coreset attention
       using compressed values VS = W_cand @ V_cand
       and weights w = W_cand @ 1.
"""

import math
import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from ..core import softmax


def _lambert_w0(z):
    """Principal branch of Lambert W for z > 0.

    Solves w * exp(w) = z via Newton iteration.
    """
    if z <= 0:
        return 0.0
    if z < math.e:
        w = math.exp(math.log(z) - 1)
    else:
        w = math.log(z) - math.log(math.log(z))
    for _ in range(30):
        ew = math.exp(w)
        dw = (w * ew - z) / (ew * (w + 1))
        w -= dw
        if abs(dw) < 1e-14:
            break
    return w


def _get_temperature(beta, RQ, RK, n):
    """Temperature τ from Equation 4.

    τ = sqrt(RK/RQ) · sqrt(b0 / (2·W0(b0/(2·ρ0))))
    where b0 = log(n)/(β·RQ·RK) + 2
    and ρ0 = sqrt(1 + exp(W0(2/e²) + 2)) ≈ 3.19
    """
    rho0 = math.sqrt(
        1 + math.exp(_lambert_w0(2.0 / math.e ** 2) + 2)
    )
    b0 = math.log(n) / (beta * RQ * RK) + 2
    w0_arg = b0 / (2 * rho0)
    w0_val = _lambert_w0(w0_arg)
    if w0_val < 1e-30:
        w0_val = 1e-30
    tau = math.sqrt(RK / RQ) * math.sqrt(b0 / (2 * w0_val))
    return max(tau, 1e-10)


def _rpnys(K_scaled, beta, max_r, rng):
    """Randomly Pivoted Nyström (Algorithm 1).

    Uses the GAUSSIAN kernel h(a,b) = exp(a^T b - ||a||²/2 - ||b||²/2)
    which has diagonal = 1, ensuring numerical stability.

    Returns:
      pivots: array of pivot indices [r]
      M: kernel inverse accumulator [r, r] (in Gaussian kernel space)
      R: kernel evaluations [r, n] (in Gaussian kernel space)
    """
    n, d = K_scaled.shape

    norms_sq = np.sum(K_scaled * K_scaled, axis=1)
    half_sq = norms_sq / 2.0

    # Gaussian kernel diagonal is always 1
    p = np.ones(n, dtype=np.float64)

    max_r = min(max_r, n)
    M = np.zeros((max_r, max_r), dtype=np.float64)
    R = np.zeros((max_r, n), dtype=np.float64)
    g = np.zeros(max_r, dtype=np.float64)
    pivots = []

    for i in range(max_r):
        p_safe = np.maximum(p, 0.0)
        p_sum = p_safe.sum()
        if p_sum < 1e-30:
            break
        probs = p_safe / p_sum
        s = rng.choice(n, p=probs)
        pivots.append(s)

        if i > 0:
            g[:i] = M[:i, :i] @ R[:i, s]
        g[i] = -1.0
        ps = p_safe[s]
        if ps < 1e-30:
            break
        g[:i + 1] /= math.sqrt(ps)
        gi = g[:i + 1]
        M[:i + 1, :i + 1] += np.outer(gi, gi)

        # Gaussian kernel: h(k_s, k_l) = exp(k_s^T k_l - ||k_s||²/2 - ||k_l||²/2)
        dots = K_scaled[s] @ K_scaled.T
        R[i, :] = np.exp(dots - half_sq[s] - half_sq)
        R[i, :] = np.minimum(R[i, :], 1.0)  # clamp (Gaussian diagonal = 1)

        # Update residual diagonal
        delta = g[:i + 1] @ R[:i + 1, :]
        p -= delta * delta
        p[s] = 0.0
        p = np.maximum(p, 0.0)

        g[:i + 1] = 0.0

    actual_r = len(pivots)
    return (
        np.array(pivots, dtype=np.int64),
        M[:actual_r, :actual_r].copy(),
        R[:actual_r, :].copy(),
    )


class WildcatKVCompression(AttentionAlgorithm):
    """WILDCAT KV cache compression.

    Budget = number of coreset points for candidates.
    Special tokens (sink + local) get exact attention.
    """

    def __init__(self, max_rank=4096):
        self.max_rank = max_rank
        self._pivots = None
        self._M = None
        self._R = None
        self._beta = None
        self._n = None
        self._run_cache = None
        self._run_cache_key = None

    @property
    def name(self) -> str:
        return "WILDCAT"

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
        rng = np.random.default_rng(seed)
        n, d = keys.shape
        beta = 1.0 / math.sqrt(head_dim)

        # Recentre keys (Section 2.4)
        k_bar = keys.astype(np.float64).mean(axis=0)
        K_c = keys.astype(np.float64) - k_bar

        # Compute radii
        RK = float(np.max(np.linalg.norm(K_c, axis=1)))
        if queries is not None:
            RQ = float(np.max(
                np.linalg.norm(
                    queries.astype(np.float64), axis=1,
                )
            ))
        else:
            RQ = RK
        RK = max(RK, 1e-10)
        RQ = max(RQ, 1e-10)

        # Temperature (Equation 4)
        tau = _get_temperature(beta, RQ, RK, n)

        # Scale keys for RPNYS kernel
        K_scaled = K_c / tau

        # Run RPNYS (Algorithm 1) — uses Gaussian kernel internally
        max_r = min(self.max_rank, n)
        pivots, M, R = _rpnys(K_scaled, beta, max_r, rng)

        # Store squared norms of scaled keys for Gaussian→exponential conversion
        sq_norms_scaled = np.sum(K_scaled * K_scaled, axis=1)

        self._pivots = pivots
        self._M = M
        self._R = R
        self._beta = beta
        self._n = n
        self._sq_norms_scaled = sq_norms_scaled
        self._run_cache = None
        self._run_cache_key = None

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        logits = problem.logits
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        n_cand = len(candidate_idx)
        d = values.shape[1]

        if n_cand == 0:
            w = softmax(logits[special_idx])
            out = w @ values[special_idx]
            return AttentionOutput(
                output=out,
                actual_budget=len(special_idx),
            )

        r = min(budget, len(self._pivots), n_cand)
        if r == 0:
            w = softmax(logits[special_idx])
            out = w @ values[special_idx]
            return AttentionOutput(
                output=out,
                actual_budget=len(special_idx),
            )

        # Cache: convert Gaussian kernel Nyström weights to
        # exponential kernel, then compute compressed values.
        cache_key = id(problem)
        if self._run_cache_key != cache_key:
            max_r = len(self._pivots)
            n_vals = len(values)
            vals_cand = values[candidate_idx].astype(np.float64)

            # R is in Gaussian kernel space. Convert to exponential:
            # W_gauss = M @ R
            # W_exp[j,l] = W_gauss[j,l] * exp(||k_pivot_j||²/2 + ||k_l||²/2 - ||k_pivot_j||²/2 - ||k_l||²/2)
            # Actually: exp_kernel(a,b) = gauss_kernel(a,b) * exp(||a||²/2 + ||b||²/2)
            # So: W_exp = W_gauss * exp((||k_l||²_scaled - ||k_pivot||²_scaled) / 2) per the paper
            # The paper does: scaling = -cmpd_sqd_knorms + sqd_knorms; W *= exp(scaling/2)

            sq_norms = self._sq_norms_scaled
            cand_sq = sq_norms[candidate_idx]

            # Pre-compute R_cand with exponential conversion baked in
            R_cand_gauss = self._R[:, candidate_idx]  # [max_r, n_cand] Gaussian

            # For each pivot j and candidate l:
            # R_exp[j,l] = R_gauss[j,l] * exp((cand_sq[l] - pivot_sq[j]) / 2)
            # But we apply this after M @ R, so we convert W directly.
            # Store Gaussian R for now, convert after M @ R.
            self._R_cand_gauss = R_cand_gauss
            self._vals_cand = vals_cand
            self._cand_sq = cand_sq
            self._run_cache_key = cache_key

        # Nyström weights (in Gaussian kernel space first)
        M_r = self._M[:r, :r]
        W_gauss = M_r @ self._R_cand_gauss[:r]  # [r, n_cand]

        # Convert Gaussian → exponential kernel weights
        pivot_sq = self._sq_norms_scaled[self._pivots[:r]]
        scaling = -pivot_sq[:, None] + self._cand_sq[None, :]
        W_exp = W_gauss * np.exp(scaling / 2.0)

        VS_cand = W_exp @ self._vals_cand  # [r, d]
        w_cand = W_exp.sum(axis=1)          # [r]

        # Coreset attention scores (true logits)
        S_r = self._pivots[:r]
        s_coreset = logits[S_r].astype(np.float64)
        s_special = logits[special_idx].astype(np.float64)

        s_max = max(
            float(s_coreset.max()) if len(s_coreset) > 0
            else -1e30,
            float(s_special.max()) if len(s_special) > 0
            else -1e30,
        )

        # Special contribution (exact)
        sp_exp = np.exp(s_special - s_max)
        N_sp = (
            sp_exp[:, None]
            * values[special_idx].astype(np.float64)
        ).sum(axis=0)
        D_sp = sp_exp.sum()

        # Candidate contribution (Nyström-weighted)
        cs_exp = np.exp(s_coreset - s_max)
        N_cand = cs_exp @ VS_cand
        D_cand = cs_exp @ w_cand

        # Combine
        D_total = D_sp + D_cand
        if D_total < 1e-30:
            output = np.zeros(d, dtype=np.float32)
        else:
            output = (
                (N_sp + N_cand) / D_total
            ).astype(np.float32)

        # Clip to value range (WTDATTN, Alg. 3)
        vmin = values.min(axis=0)
        vmax = values.max(axis=0)
        output = np.clip(output, vmin, vmax)

        actual_budget = len(special_idx) + r
        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        max_rank = cfg.get("max_rank", 4096)
        return [WildcatKVCompression(max_rank=max_rank)]
