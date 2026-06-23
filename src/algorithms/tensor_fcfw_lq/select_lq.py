"""
Tensor FCFW under the query-weighted ``lq`` norm.

Where ``tensor_fcfw_l2`` minimizes the Frobenius (RKHS-l2) error of the tensor
target ``sigma = sum_i phi(k_i) (x) [1, v_i]``, this module minimizes a
*query-weighted* error defined by a set ``Q`` of probe queries (the earlier
context queries, exactly as the ``learned`` method uses them):

    ||sigma - sigma'||_*  =  E_{q in Q} | (sigma - sigma') . psi(q) |
                          =  (1/|Q|) sum_{q in Q} || t_q - hat_t_q ||_2 ,

where ``psi(q)`` is the attention feature of query ``q`` (``<psi(q), phi(k)> =
exp(scale q.k)``), so the projection ``sigma . psi(q) = sum_i exp(scale q.k_i)
[1, v_i] = [D(q), N(q)]`` is the *unnormalized* (denominator, numerator) readout.
The bars are the Euclidean norm on the augmented ``(1+d)`` readout, and the outer
aggregation is an L1 average over ``Q`` (hence the ``L_{2,1}`` / ``lq`` norm,
not the squared L2 of ``tensor_fcfw_l2``).

Concretely, with the attention-profile matrix ``A in R^{m x n}``,
``A[q, i] = exp(scale q.k_i - c)`` (``m = |Q|``, ``n`` candidates, ``c`` a global
shift for stability that cancels everywhere), and augmented targets
``M = [1, V] in R^{n x (1+d)}``:

    T = A @ M  in R^{m x (1+d)}   (per-probe true readout)
    hat_T = A[:, S] @ U           (coreset readout; synthetic U in R^{r x (1+d)})

* **Selection** is fully-corrective Frank-Wolfe in the *empirical attention
  Gram* ``K~ = A^T A`` (inner products of candidate attention profiles over Q),
  with the value-aware residual ``G = E_S M``. This is the same
  pivoted-Cholesky-with-labels recursion as ``tensor_fcfw_select`` but with the
  data-driven kernel ``K~`` (non-unit diagonal) instead of the analytic Gaussian
  kernel -- so selection is driven by how candidates serve the *actual* query
  distribution. Cost ``O(n r^2 + n r (1+d))`` after a one-time ``A^T(A M)`` pass.
* **Correction** refines ``U`` to minimize the true ``lq`` objective
  ``sum_q || T_q - A[:,S] U ||_2`` by iteratively reweighted least squares
  (``W = diag(1 / ||R_q||)``); ``irls_iters <= 1`` falls back to plain
  query-space least squares (the L2 surrogate).

Selection is deterministic and nested in the budget, and a ``state`` dict
warm-starts a larger budget on the same probes/candidates.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def build_attention_profiles(
    probe_queries: torch.Tensor,
    cand_keys: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Stabilized attention-profile matrix ``A[q, i] = exp(scale q.k_i - c)``.

    The global shift ``c = max_{q,i} scale q.k_i`` keeps every entry ``<= 1`` and
    rescales the whole objective by a constant ``e^{-c}`` -- which is irrelevant
    for both the greedy selection and the (I)RLS solution for ``U``.
    """
    logits = scale * (probe_queries @ cand_keys.t())   # [m, n]
    logits = logits - logits.max()
    return logits.exp()


def _truncated_solve(
    aw: torch.Tensor,
    tw: torch.Tensor,
    rcond: float,
) -> torch.Tensor:
    """Minimum-norm, ridge-damped least squares solution of ``aw U = tw``.

    ``aw``: ``[m, r]``, ``tw``: ``[m, p]`` -> ``U``: ``[r, p]``. Solved through an
    SVD with a *relative* Tikhonov damping ``lambda = (rcond * s_max)^2`` on the
    singular values (``s -> s / (s^2 + lambda)``). This is stable in **both**
    regimes: over-determined (``r <= m``) and -- crucially for this method --
    under-determined / rank-deficient (``r >= m``, i.e. budget >= #probes), where
    the primal normal equations are singular and would otherwise amplify tiny
    singular directions into exploding synthetic values.
    """
    u, s, vh = torch.linalg.svd(aw, full_matrices=False)   # u:[m,k] s:[k] vh:[k,r]
    if s.numel() == 0:
        return torch.zeros(
            (aw.shape[-1], tw.shape[-1]), dtype=aw.dtype, device=aw.device,
        )
    lam = (rcond * s[0]) ** 2
    sinv = s / (s * s + lam)                               # [k]
    coef = vh.transpose(-2, -1) @ (sinv.unsqueeze(-1) * (u.transpose(-2, -1) @ tw))
    return coef


def _irls_solve(
    a_sel: torch.Tensor,
    target: torch.Tensor,
    irls_iters: int,
    rcond: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Solve ``min_U sum_q w_q ||target_q - a_sel_q U||_2`` via IRLS for L_{2,1}.

    ``a_sel``: ``[m, r]`` selected attention columns; ``target``: ``[m, 1+d]``.
    Returns ``U`` of shape ``[r, 1+d]``. Each reweighted least-squares step uses
    the truncated SVD solve, so it is robust when ``r >= m``. ``irls_iters <= 1``
    => plain (truncated) least squares.
    """
    u = _truncated_solve(a_sel, target, rcond)
    for _ in range(max(int(irls_iters) - 1, 0)):
        resid = target - a_sel @ u                          # [m, 1+d]
        rnorm = resid.norm(dim=-1).clamp_min(eps)           # [m]
        sw = (1.0 / rnorm).sqrt().unsqueeze(-1)
        u = _truncated_solve(a_sel * sw, target * sw, rcond)
    return u


def select_lq_coreset(
    probe_queries: torch.Tensor,
    cand_keys: torch.Tensor,
    cand_values: torch.Tensor,
    r: int,
    scale: float,
    oracle: str = "fw",
    irls_iters: int = 5,
    rcond: float = 1e-3,
    jitter: float = 1e-6,
    state: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Select ``r`` candidate keys and synthetic augmented values under ``lq``.

    Args:
        probe_queries: ``[m, d]`` earlier-context queries ``Q``.
        cand_keys: ``[n, d]`` candidate keys.
        cand_values: ``[n, d]`` candidate values.
        r: coreset size.
        scale: attention scale (``1/sqrt(d)``).
        oracle: ``"fw"`` (argmax ``||G[i,:]||^2``) or ``"omp"``
            (``||G[i,:]||^2 / E[i,i]``).
        irls_iters: IRLS reweighting steps for the final ``lq`` solve
            (``<= 1`` => plain query-space least squares).
        rcond: relative singular-value cutoff for the truncated SVD value solve
            (damps directions below ``rcond * s_max``); keeps the synthetic
            values bounded when ``r >= m`` (budget >= #probes).
        jitter: floor for the residual-diagonal of the Cholesky recursion.
        state: optional warm-start dict from a previous (smaller-``r``) call.

    Returns:
        coreset: ``[r]`` selected local candidate indices (selection order).
        cmpd_values: ``[r, d]`` synthetic values (numerator channel ``U[:, 1:]``).
        w: ``[r]`` synthetic masses (denominator channel ``U[:, 0]``).
        new_state: dict for warm-starting a larger budget.
    """
    work_dtype = (
        torch.float32
        if cand_keys.dtype in (torch.bfloat16, torch.float16)
        else cand_keys.dtype
    )
    keys = cand_keys.to(work_dtype)
    values = cand_values.to(work_dtype)
    probes = probe_queries.to(work_dtype)
    device = keys.device

    n = keys.shape[0]
    r = min(int(r), n)

    # Augmented targets M = [1, V] in R^{n x (1+d)}.
    ones_col = torch.ones((n, 1), dtype=work_dtype, device=device)
    mat = torch.cat([ones_col, values], dim=-1)        # [n, 1+d]

    a = build_attention_profiles(probes, keys, scale)  # [m, n]

    reuse = (
        state is not None
        and int(state.get("n", -1)) == n
        and int(state.get("m", -1)) == a.shape[0]
    )
    if reuse:
        g_res = state["G"]
        res_diag = state["res_diag"]
        factor = state["L"]
        selected = state["selected"]
        t_done = int(state["t"])
    else:
        am = a @ mat                                   # [m, 1+d]
        g_res = a.t() @ am                             # [n, 1+d] = K~ @ M
        res_diag = (a * a).sum(dim=0)                  # [n] = diag(K~)
        factor = torch.zeros((n, r), dtype=work_dtype, device=device)
        selected = torch.empty((0,), dtype=torch.long, device=device)
        t_done = 0

    if r > t_done:
        if factor.shape[1] < r:
            grown = torch.zeros((n, r), dtype=work_dtype, device=device)
            grown[:, : factor.shape[1]] = factor
            factor = grown
        new_selected = torch.empty((r,), dtype=torch.long, device=device)
        new_selected[:t_done] = selected
        selected = new_selected

        neg_inf = torch.finfo(work_dtype).min
        for t in range(t_done, r):
            score = (g_res * g_res).sum(dim=-1)        # [n]
            if oracle == "omp":
                score = score / res_diag.clamp(min=jitter)
            if t > 0:
                score = score.scatter(0, selected[:t], neg_inf)
            idx = int(torch.argmax(score).item())
            selected[t] = idx

            kcol = a.t() @ a[:, idx]                   # [n] = K~[:, idx]
            diag = res_diag[idx].clamp(min=jitter)

            if t > 0:
                lp = factor[idx, :t]                   # [t]
                proj = factor[:, :t] @ lp              # [n]
                ecol = kcol - proj
            else:
                ecol = kcol

            ell = ecol / diag.sqrt()
            factor[:, t] = ell

            ell_mat = ell @ mat                        # [1+d]
            g_res = g_res - ell.unsqueeze(-1) * ell_mat.unsqueeze(0)
            res_diag = res_diag - ell * ell
            res_diag[idx] = 0.0
            res_diag = res_diag.clamp(min=0.0)
        t_done = r

    new_state = {
        "G": g_res,
        "res_diag": res_diag,
        "L": factor,
        "selected": selected,
        "t": t_done,
        "n": n,
        "m": int(a.shape[0]),
    }

    coreset = selected[:r]
    a_sel = a[:, coreset]                              # [m, r]
    target = a @ mat                                   # [m, 1+d]
    u = _irls_solve(a_sel, target, irls_iters, rcond)  # [r, 1+d]

    w = u[:, 0].to(cand_keys.dtype)
    cmpd_values = u[:, 1:].to(cand_keys.dtype)
    return coreset, cmpd_values, w, new_state
