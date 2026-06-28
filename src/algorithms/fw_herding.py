"""
Frank-Wolfe herding methods for attention coreset selection.

Three variants, all using Nyström value aggregation at eval time:

1. FW-Nystrom: Simple FW (γ_t = 1/(t+1), equal weights) + Nyström values
2. FCFW-Nystrom: Fully Corrective FW (re-optimize weights via NNLS) + Nyström values
3. TensorFCFW-Nystrom: FCFW on tensor Gram G_key * G_val + Nyström values

All follow the AttentionAlgorithm interface:
  prepare() — run herding on candidate keys (once per example)
  run()     — select budget pivots, Nyström-reweight values, two-group attention
"""

import math
import numpy as np
from typing import List, Optional

from .base import AttentionAlgorithm, AttentionInput, AttentionOutput
from ..core import softmax


# ═══════════════════════════════════════════════════════════
# Kernel primitives
# ═══════════════════════════════════════════════════════════

def _gram(K, tau):
    """Exponential kernel Gram: G_ij = exp(tau * k_i^T k_j)."""
    S = tau * (K @ K.T)
    np.clip(S, -500, 500, out=S)
    return np.exp(S)


def _gram_tensor(K, V, tau):
    """Tensor Gram: G_ij = exp(tau * k_i^T k_j) * (v_i^T v_j).
    PSD by Schur product theorem."""
    G_key = _gram(K, tau)
    G_val = V @ V.T
    return G_key * G_val


# ═══════════════════════════════════════════════════════════
# LMO: scan all candidates, return scores
# ═══════════════════════════════════════════════════════════

def _lmo_scores(G, atom_indices, atom_weights, n):
    """LMO scores for denominator herding.
    score(l) = (1/n) Σ_i G[i,l] - Σ_j w_j G[atom_j, l]
    """
    target = G.sum(axis=0) / n
    if len(atom_indices) == 0:
        return target
    w = np.array(atom_weights)
    return target - w @ G[atom_indices]


def _lmo_scores_tensor(G_op, atom_indices, atom_weights, n):
    """LMO scores for operator herding (tensor Gram)."""
    return _lmo_scores(G_op, atom_indices, atom_weights, n)


# ═══════════════════════════════════════════════════════════
# FCFW weight optimization
# ═══════════════════════════════════════════════════════════

def _fcfw_reweight(G, atom_indices, n):
    """Solve: w* = argmin ||σ/n - Σ w_j ψ(k_j)||² s.t. w ≥ 0.

    The RKHS objective expands to:
      w^T G_atoms w - 2 w^T b + const
    where b_j = (1/n) Σ_i G[i, atom_j].

    Uses regularized solve + clipping (fast) for large T,
    scipy NNLS (exact) for small T.
    """
    idx = np.array(atom_indices)
    T = len(idx)
    b = G[:, idx].sum(axis=0) / n
    G_atoms = G[np.ix_(idx, idx)]
    G_atoms_reg = G_atoms + 1e-8 * np.eye(T)

    if T <= 512:
        # Exact NNLS for small problems
        from scipy.optimize import nnls
        try:
            w, _ = nnls(G_atoms_reg, b, maxiter=10 * T)
        except RuntimeError:
            # NNLS failed to converge — fall back to solve+clip
            try:
                w = np.linalg.solve(G_atoms_reg, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(G_atoms_reg, b, rcond=None)[0]
            w = np.maximum(w, 0.0)
    else:
        # Fast: solve then clip negatives
        try:
            w = np.linalg.solve(G_atoms_reg, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(G_atoms_reg, b, rcond=None)[0]
        w = np.maximum(w, 0.0)

    w_sum = w.sum()
    if w_sum > 1e-30:
        w /= w_sum
    else:
        w = np.ones(T) / T
    return w


def _residual_sq(G, atom_indices, atom_weights, n):
    """||σ/n - Σ w_j ψ(k_j)||² via kernel trick."""
    idx = np.array(atom_indices)
    w = np.array(atom_weights)
    term1 = G.sum() / (n * n)
    term2 = -2.0 / n * float(w @ G[:, idx].sum(axis=0))
    term3 = float(w @ G[np.ix_(idx, idx)] @ w)
    return max(0.0, term1 + term2 + term3)


# ═══════════════════════════════════════════════════════════
# Herding loops
# ═══════════════════════════════════════════════════════════

def _herding_simple(G, max_atoms, n):
    """Simple FW: γ_t = 1/(t+1), equal weights."""
    indices = []
    weights = []
    for t in range(max_atoms):
        scores = _lmo_scores(G, indices, weights, n)
        # Exclude already-selected to encourage diversity
        for idx in indices:
            scores[idx] = -np.inf
        best = int(np.argmax(scores))
        indices.append(best)
        gamma = 1.0 / (t + 1)
        weights = [w * (1 - gamma) for w in weights]
        weights.append(gamma)
    return indices


def _herding_fcfw(G, max_atoms, n):
    """Fully Corrective FW: re-optimize weights periodically.

    Re-optimizes weights at increasing intervals:
    - Every 10 steps for T < 256
    - Every 50 steps for T < 1024
    - Every 100 steps for T >= 1024
    Plus always at the end.
    """
    indices = []
    weights = []
    for t in range(max_atoms):
        # Adaptive reweight schedule
        if t < 256:
            reweight_every = 10
        elif t < 1024:
            reweight_every = 50
        else:
            reweight_every = 100

        if len(indices) > 0 and t % reweight_every == 0:
            weights = list(_fcfw_reweight(G, indices, n))
        scores = _lmo_scores(G, indices, weights, n)
        for idx in indices:
            scores[idx] = -np.inf
        best = int(np.argmax(scores))
        indices.append(best)
        # Simple FW weight for the new atom between re-optimizations
        gamma = 1.0 / (t + 1)
        weights = [w * (1 - gamma) for w in weights]
        weights.append(gamma)
    # Final weight optimization
    final_weights = _fcfw_reweight(G, indices, n)
    return indices, final_weights


def _herding_tensor_fcfw(G_op, max_atoms, n):
    """FCFW on tensor Gram G_op = G_key * G_val."""
    return _herding_fcfw(G_op, max_atoms, n)


# ═══════════════════════════════════════════════════════════
# Nyström value aggregation
# ═══════════════════════════════════════════════════════════

def _nystrom_compress(K_all, V_all, pivot_indices, tau):
    """Compute Nyström-weighted values and denominator weights.

    Uses Gaussian kernel internally for numerical stability,
    then converts to exponential kernel weights (same as WildCat paper).

    Returns (pivot_keys, V_compressed, w_compressed).
    """
    K64 = K_all.astype(np.float64)
    V64 = V_all.astype(np.float64)
    idx = np.array(pivot_indices)
    r = len(idx)

    K_piv = K64[idx]

    # Scale keys for Gaussian kernel (same approach as WildCat)
    # Center first
    kbar = K64.mean(axis=0)
    K_c = K64 - kbar
    K_c_piv = K_c[idx]

    # Compute temperature
    RK = float(np.max(np.linalg.norm(K_c, axis=1)))
    RK = max(RK, 1e-10)
    rho0 = math.sqrt(1 + math.exp(
        _lambert_w0(2.0 / math.e**2) + 2))
    b0 = math.log(len(K64)) / (tau * RK * RK) + 2
    w0_val = max(_lambert_w0(b0 / (2 * rho0)), 1e-30)
    tau_wc = max(math.sqrt(1.0) * math.sqrt(b0 / (2 * w0_val)), 1e-10)
    key_mult = math.sqrt(tau) / tau_wc

    K_scaled = K_c * key_mult
    K_scaled_piv = K_scaled[idx]

    sq_norms = np.sum(K_scaled * K_scaled, axis=1)
    sq_norms_piv = sq_norms[idx]
    half_sq = sq_norms / 2.0
    half_sq_piv = sq_norms_piv / 2.0

    # Gaussian kernel: pivot-to-pivot and pivot-to-all
    G_pp = np.exp(K_scaled_piv @ K_scaled_piv.T - half_sq_piv[:, None] - half_sq_piv[None, :])
    G_pp = np.minimum(G_pp, 1.0)
    G_pa = np.exp(K_scaled_piv @ K_scaled.T - half_sq_piv[:, None] - half_sq[None, :])
    G_pa = np.minimum(G_pa, 1.0)

    # Solve for Nyström weights in Gaussian kernel space
    G_pp_reg = G_pp + 1e-8 * np.eye(r)
    try:
        W = np.linalg.solve(G_pp_reg, G_pa)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(G_pp_reg, G_pa, rcond=None)[0]

    # Convert Gaussian → exponential kernel weights
    scaling = -sq_norms_piv[:, None] + sq_norms[None, :]
    W = W * np.exp(scaling / 2.0)

    V_compressed = W @ V64
    w_compressed = W.sum(axis=1)

    return K_piv, V_compressed, w_compressed


def _lambert_w0(z):
    """Principal branch of Lambert W for z > 0."""
    if z <= 0:
        return 0.0
    w = math.log(z) - math.log(math.log(z)) if z >= math.e else math.exp(math.log(z) - 1)
    for _ in range(30):
        ew = math.exp(w)
        dw = (w * ew - z) / (ew * (w + 1))
        w -= dw
        if abs(dw) < 1e-14:
            break
    return w


# ═══════════════════════════════════════════════════════════
# AttentionAlgorithm implementations
# ═══════════════════════════════════════════════════════════

class FWHerdingNystrom(AttentionAlgorithm):
    """Frank-Wolfe herding + Nyström value aggregation.

    mode:
      "simple"       — Standard FW, γ_t = 1/(t+1)
      "fcfw"         — Fully Corrective FW (NNLS weight optimization)
      "tensor_fcfw"  — FCFW on tensor Gram G_key * G_val
    """

    def __init__(self, max_atoms=2048, gram_subsample=3000, mode="simple"):
        self.max_atoms = max_atoms
        self.gram_subsample = gram_subsample
        self.mode = mode
        self._pivot_indices = None
        self._n_candidates = None
        self._tau = None

    @property
    def name(self) -> str:
        names = {
            "simple": "FW-Nystrom",
            "fcfw": "FCFW",
            "tensor_fcfw": "FCFW KeyXValue",
        }
        return names.get(self.mode, f"FW-{self.mode}")

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
        n = keys.shape[0]
        tau = 1.0 / math.sqrt(head_dim)
        self._tau = tau
        self._n_candidates = n

        # Subsample for Gram computation if needed
        if n > self.gram_subsample:
            sub_idx = np.sort(rng.choice(n, self.gram_subsample, replace=False))
            K_sub = keys[sub_idx].astype(np.float64)
            V_sub = values[sub_idx].astype(np.float64)
            n_sub = len(sub_idx)
        else:
            sub_idx = np.arange(n)
            K_sub = keys.astype(np.float64)
            V_sub = values.astype(np.float64)
            n_sub = n

        max_atoms = min(self.max_atoms, n_sub)

        import time as _time

        # Build Gram and run herding
        t0 = _time.time()
        if self.mode == "tensor_fcfw":
            G = _gram_tensor(K_sub, V_sub, tau)
            print(f"  [{self.name}] Gram ({n_sub}x{n_sub} tensor): {_time.time()-t0:.1f}s", flush=True)
            t0 = _time.time()
            indices_local, _ = _herding_tensor_fcfw(G, max_atoms, n_sub)
        elif self.mode == "fcfw":
            G = _gram(K_sub, tau)
            print(f"  [{self.name}] Gram ({n_sub}x{n_sub}): {_time.time()-t0:.1f}s", flush=True)
            t0 = _time.time()
            indices_local, _ = _herding_fcfw(G, max_atoms, n_sub)
        else:  # simple
            G = _gram(K_sub, tau)
            print(f"  [{self.name}] Gram ({n_sub}x{n_sub}): {_time.time()-t0:.1f}s", flush=True)
            t0 = _time.time()
            indices_local = _herding_simple(G, max_atoms, n_sub)
        print(f"  [{self.name}] Herding ({max_atoms} atoms): {_time.time()-t0:.1f}s", flush=True)

        # Map subsample-local indices to original array indices
        self._pivot_indices = sub_idx[indices_local]

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        keys = problem.keys
        values = problem.values
        special_idx = problem.special_idx
        candidate_idx = problem.candidate_idx
        logits = problem.logits
        d = values.shape[1]

        n_cand = len(candidate_idx)
        if n_cand == 0 or budget == 0:
            w = softmax(logits[special_idx])
            out = w @ values[special_idx].astype(np.float64)
            return AttentionOutput(
                output=out.astype(np.float32),
                actual_budget=len(special_idx),
            )

        # Select pivots from herding order that are in the candidate set
        cand_set = set(candidate_idx.tolist())
        pivots = []
        for idx in self._pivot_indices:
            if idx in cand_set:
                pivots.append(idx)
            if len(pivots) >= budget:
                break

        r = len(pivots)
        if r == 0:
            w = softmax(logits[special_idx])
            out = w @ values[special_idx].astype(np.float64)
            return AttentionOutput(
                output=out.astype(np.float32),
                actual_budget=len(special_idx),
            )

        # Nyström compression on candidates
        K_cand = keys[candidate_idx].astype(np.float64)
        V_cand = values[candidate_idx].astype(np.float64)

        # Map pivot indices to candidate-local indices
        cand_idx_map = {idx: i for i, idx in enumerate(candidate_idx)}
        pivot_local = [cand_idx_map[p] for p in pivots]

        K_piv, V_compressed, w_compressed = _nystrom_compress(
            K_cand, V_cand, pivot_local, self._tau)

        # Two-group attention
        s_special = logits[special_idx].astype(np.float64)
        s_coreset = logits[np.array(pivots)].astype(np.float64)

        s_max = max(
            float(s_coreset.max()) if len(s_coreset) > 0 else -1e30,
            float(s_special.max()) if len(s_special) > 0 else -1e30,
        )

        # Special contribution (exact)
        sp_exp = np.exp(s_special - s_max)
        N_sp = (sp_exp[:, None] * values[special_idx].astype(np.float64)).sum(axis=0)
        D_sp = sp_exp.sum()

        # Coreset contribution (Nyström-weighted)
        cs_exp = np.exp(s_coreset - s_max)
        N_cand = cs_exp @ V_compressed
        D_cand = cs_exp @ w_compressed

        # Combine
        D_total = D_sp + D_cand
        if D_total < 1e-30:
            output = np.zeros(d, dtype=np.float32)
        else:
            output = ((N_sp + N_cand) / D_total).astype(np.float32)

        # Clip to value range
        vmin = values.min(axis=0)
        vmax = values.max(axis=0)
        output = np.clip(output, vmin, vmax)

        return AttentionOutput(
            output=output,
            actual_budget=len(special_idx) + r,
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        mode = cfg.get("mode", "simple")
        max_atoms = cfg.get("max_atoms", 2048)
        gram_subsample = cfg.get("gram_subsample", 3000)
        return [FWHerdingNystrom(
            max_atoms=max_atoms,
            gram_subsample=gram_subsample,
            mode=mode,
        )]
