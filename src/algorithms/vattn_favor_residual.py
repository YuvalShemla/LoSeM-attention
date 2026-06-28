"""
vAttention + FAVOR+ residual correction.

Three modes:

1. "replace" — exact S + exact C + FAVOR+(U)
   Replaces the IS estimator entirely. Sampled keys
   contribute exactly; unsampled remainder via FAVOR+.

2. "correct" — exact S + IS(R) + FAVOR+ control variate
   Keeps the vAttention IS estimator and adds a FAVOR+
   correction term that estimates the IS error:

     N = N_S + (|R|/|C|)·N_C + [N̂_U − (|U|/|C|)·N̂_C]
     D = D_S + (|R|/|C|)·D_C + [D̂_U − (|U|/|C|)·D̂_C]

   This simplifies to:
     N = N_S + N_C + N̂_U + (|U|/|C|)·(N_C − N̂_C)

   If FAVOR+ is perfect, it corrects IS to exact.
   If FAVOR+ is bad, the correction is noise around
   the IS baseline — it can't make things much worse.

3. "correct_num" — same as "correct" but FAVOR+
   correction only on the numerator; denominator uses
   pure IS (no FAVOR+ in D):

     N = N_S + N_C + N̂_U + (|U|/|C|)·(N_C − N̂_C)
     D = D_S + (|R|/|C|)·D_C

   This avoids instability from FAVOR+ errors in the
   normalizer while still improving the value estimate.
"""

import math
import numpy as np
from typing import List, Optional

from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)
from .pq_topk import PQIndex
from ..core import softmax


def _random_orthogonal_blocks(
    d: int, m: int, rng: np.random.Generator,
) -> np.ndarray:
    """
    Build [m, d] projection from stacked Haar-orthogonal blocks.

    Each block is d x d via QR of Gaussian. Stack ceil(m/d)
    blocks, then slice to m rows. Rows within each block are
    orthonormal; across blocks they are independent.
    """
    n_blocks = math.ceil(m / d)
    blocks = []
    for _ in range(n_blocks):
        a = rng.standard_normal((d, d)).astype(np.float64)
        q, _ = np.linalg.qr(a, mode="reduced")
        blocks.append(q)
    W = np.vstack(blocks)[:m]
    return W


def _favor_features(
    x: np.ndarray, W: np.ndarray,
) -> np.ndarray:
    """
    FAVOR+ positive random feature map:
      phi(x) = (1/sqrt(m)) * exp(W @ x - ||x||^2 / 2)

    Args:
        x: [n, d] or [d] — input vectors.
        W: [m, d] — random projection matrix.

    Returns:
        [n, m] or [m] — feature vectors (float64).
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    m = W.shape[0]
    squeeze = x.ndim == 1
    if squeeze:
        x = x[np.newaxis]

    # [n, m] = [n, d] @ [d, m]
    Wx = x @ W.T
    # [n] -> [n, 1]
    half_norm_sq = 0.5 * np.sum(x * x, axis=1, keepdims=True)
    exponent = Wx - half_norm_sq
    np.clip(exponent, -500.0, 500.0, out=exponent)
    phi = np.exp(exponent) / math.sqrt(m)

    if squeeze:
        return phi[0]
    return phi


class VAttentionFavorResidual(AttentionAlgorithm):
    """
    vAttention + FAVOR+ residual correction.

    Budget split: half top-k (oracle or PQ), half uniform sample.

    mode="replace":
      S exact + C exact (no IS) + FAVOR+(U)

    mode="correct":
      S exact + IS(R) + FAVOR+ control variate correction
      = S + C + FAVOR+(U) + (|U|/|C|)·(C_exact − C_favor)

    mode="correct_num":
      Same numerator as "correct", but denominator uses
      pure IS (no FAVOR+ in D).
    """

    def __init__(
        self,
        feature_dim: int = 64,
        feature_type: str = "orthogonal",
        oracle_topk: bool = True,
        mode: str = "correct",
        m_pq: int = 8,
    ):
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        self.oracle_topk = oracle_topk
        self.mode = mode
        self.m_pq = m_pq
        self._W = None
        self._phi_keys = None
        self._d_quarter = None
        self._pq = None

    @property
    def name(self) -> str:
        method = "Oracle" if self.oracle_topk else "PQ"
        ft = "orth" if self.feature_type == "orthogonal" else "iid"
        mode_tags = {
            "replace": "rep",
            "correct": "cv",
            "correct_num": "cvN",
        }
        mode_tag = mode_tags.get(self.mode, self.mode)
        return (
            f"vAttn+FAV{mode_tag}-{method}-{ft}"
            f"-d{self.feature_dim}"
        )

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
        self._d_quarter = head_dim ** 0.25

        if self.feature_type == "orthogonal":
            self._W = _random_orthogonal_blocks(
                head_dim, self.feature_dim, rng,
            )
        else:
            self._W = rng.standard_normal(
                (self.feature_dim, head_dim),
            ).astype(np.float64)

        k_tilde = keys.astype(np.float64) / self._d_quarter
        self._phi_keys = _favor_features(k_tilde, self._W)

        if not self.oracle_topk:
            self._pq = PQIndex(
                m=self.m_pq, n_codes=256, seed=seed,
            )
            self._pq.fit(keys)

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

        buse = min(budget, n_cand)
        b_topk = buse // 2
        b_sample = buse - b_topk

        # --- Top-k selection ---
        if self.oracle_topk:
            cand_logits = logits[candidate_idx]
            if b_topk > 0 and b_topk < n_cand:
                top_pos = np.argpartition(
                    cand_logits, -b_topk,
                )[-b_topk:]
            elif b_topk >= n_cand:
                top_pos = np.arange(n_cand)
            else:
                top_pos = np.array([], dtype=np.int64)
            topk_global = candidate_idx[top_pos]
        else:
            n_pq = len(self._pq.codes)
            cand_mask = np.zeros(n_pq, dtype=bool)
            cand_mask[candidate_idx] = True
            if b_topk > 0:
                topk_global = self._pq.approximate_topk(
                    problem.query, b_topk,
                    candidate_mask=cand_mask,
                )
            else:
                topk_global = np.array([], dtype=np.int64)

        # --- Uniform sample from remaining ---
        topk_set = set(topk_global.tolist())
        remaining = np.array([
            i for i in candidate_idx if i not in topk_set
        ], dtype=np.int64)
        n_s = len(remaining)  # |R|

        if b_sample > 0 and n_s > 0:
            n_sample = min(b_sample, n_s)
            sampled_global = rng.choice(
                remaining, size=n_sample, replace=False,
            )
        else:
            sampled_global = np.array([], dtype=np.int64)
            n_sample = 0

        n_u = n_s - n_sample  # |U|

        # --- S contribution (special + topk): exact ---
        fixed_idx = np.concatenate(
            [special_idx, topk_global],
        ).astype(np.int64)

        all_exact = np.concatenate([
            fixed_idx, sampled_global,
        ]).astype(np.int64) if n_sample > 0 else fixed_idx

        all_logits = logits[all_exact].astype(np.float64)
        s_max = (
            np.max(all_logits) if len(all_logits) > 0
            else 0.0
        )

        fixed_s = logits[fixed_idx].astype(np.float64)
        fixed_exp = np.exp(fixed_s - s_max)
        N_S = (
            fixed_exp[:, None]
            * values[fixed_idx].astype(np.float64)
        ).sum(axis=0)
        D_S = fixed_exp.sum()

        # --- C contribution (sampled): exact ---
        if n_sample > 0:
            samp_s = logits[sampled_global].astype(np.float64)
            samp_exp = np.exp(samp_s - s_max)
            N_C = (
                samp_exp[:, None]
                * values[sampled_global].astype(np.float64)
            ).sum(axis=0)
            D_C = samp_exp.sum()
        else:
            N_C = np.zeros(d, dtype=np.float64)
            D_C = 0.0

        # --- FAVOR+ estimates ---
        q_tilde = (
            problem.query.astype(np.float64) / self._d_quarter
        )
        phi_q = _favor_features(q_tilde, self._W)
        scale = np.exp(-s_max)

        if n_u > 0:
            # FAVOR+ estimate of U (unsampled)
            phi_cand = self._phi_keys[candidate_idx]
            vals_cand = values[candidate_idx].astype(
                np.float64,
            )
            Psi_cand_v = phi_cand.T @ vals_cand
            psi_cand = phi_cand.sum(axis=0)

            ts_idx = np.concatenate([
                topk_global, sampled_global,
            ]).astype(np.int64)
            phi_ts = self._phi_keys[ts_idx]
            vals_ts = values[ts_idx].astype(np.float64)
            Psi_U_v = Psi_cand_v - phi_ts.T @ vals_ts
            psi_U = psi_cand - phi_ts.sum(axis=0)

            N_hat_U = (phi_q @ Psi_U_v) * scale
            D_hat_U = (phi_q @ psi_U) * scale

            if self.mode == "correct" and n_sample > 0:
                # FAVOR+ estimate of C (sampled) for
                # control variate correction
                phi_C = self._phi_keys[sampled_global]
                vals_C = values[sampled_global].astype(
                    np.float64,
                )
                N_hat_C = (phi_q @ (phi_C.T @ vals_C)) * scale
                D_hat_C = (phi_q @ phi_C.sum(axis=0)) * scale
            else:
                N_hat_C = np.zeros(d, dtype=np.float64)
                D_hat_C = 0.0
        else:
            N_hat_U = np.zeros(d, dtype=np.float64)
            D_hat_U = 0.0
            N_hat_C = np.zeros(d, dtype=np.float64)
            D_hat_C = 0.0

        # --- Combine ---
        if self.mode == "replace":
            # S(exact) + C(exact) + U(FAVOR+)
            N_total = N_S + N_C + N_hat_U
            D_total = D_S + D_C + D_hat_U

        elif self.mode in ("correct", "correct_num"):
            # FAVOR+ control variate correction on numerator:
            # N = S + C + Û + (|U|/|C|)·(C − Ĉ)
            if n_sample > 0 and n_u > 0:
                ratio = float(n_u) / float(n_sample)
                cv_N = ratio * (N_C - N_hat_C)
                cv_D = ratio * (D_C - D_hat_C)
            else:
                cv_N = np.zeros(d, dtype=np.float64)
                cv_D = 0.0

            N_total = N_S + N_C + N_hat_U + cv_N

            if self.mode == "correct":
                # FAVOR+ correction on denominator too
                D_total = D_S + D_C + D_hat_U + cv_D
            else:
                # correct_num: IS denominator only
                weight = (
                    float(n_s) / float(n_sample)
                    if n_sample > 0 else 1.0
                )
                D_total = D_S + weight * D_C

        if D_total < 1e-30:
            output = np.zeros(d, dtype=np.float32)
        else:
            output = (N_total / D_total).astype(np.float32)

        actual_budget = len(fixed_idx) + n_sample
        return AttentionOutput(
            output=output,
            actual_budget=actual_budget,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        feature_dims = cfg.get(
            "feature_dim_sweep", [64],
        )
        feature_types = cfg.get(
            "feature_types", ["orthogonal"],
        )
        modes = cfg.get("modes", ["replace", "correct"])
        m_pq = cfg.get("m_pq", 8)
        instances = []
        for mode in modes:
            for ft in feature_types:
                for fd in feature_dims:
                    instances.append(
                        VAttentionFavorResidual(
                            feature_dim=fd,
                            feature_type=ft,
                            oracle_topk=True,
                            mode=mode,
                            m_pq=m_pq,
                        )
                    )
                    instances.append(
                        VAttentionFavorResidual(
                            feature_dim=fd,
                            feature_type=ft,
                            oracle_topk=False,
                            mode=mode,
                            m_pq=m_pq,
                        )
                    )
        return instances
