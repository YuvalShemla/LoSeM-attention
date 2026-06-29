"""
Tensor FCFW under the query-weighted ``lq`` norm.

Where ``tensor_fcfw_l2`` minimizes the Frobenius (RKHS-l2) error of the tensor
target ``sigma = sum_i phi(k_i) (x) [1, v_i]``, this module minimizes a
*query-weighted* error defined by a set ``Q`` of probe queries.

``oracle='fc_lq'`` runs exact fully-corrective Frank–Wolfe under the true lq
objective: at each step, for every candidate key ``k'``, it IRLS-solves for
the best synthetic values on ``S ∪ {k'}`` (re-optimizing all prior values) and
picks the key with the largest objective decrease.

``oracle='residual_lq'`` scores each candidate by the best one-column lq fit to
the current residual, picks the key with the largest residual drop, then refreshes
the support (``correction_irls_iters``) each step and runs a final
``irls_iters`` solve on the selected coreset.

``oracle='residual_lq_deflated'``: same as ``residual_lq``, but candidate
scoring uses probe-space columns with the selected support projected out
(orthogonal OMP geometry) before the one-column lq fit.

``oracle='fw'`` / ``'omp'`` use the fast pivoted-Cholesky surrogate on the
empirical attention Gram (Frobenius / l2 selection), then a final IRLS solve.
``omp`` only differs from ``fw`` by normalizing the Cholesky scores (see below).
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _resolve_show_progress(show_progress: Optional[bool]) -> bool:
    if show_progress is None:
        return sys.stderr.isatty()
    return bool(show_progress)


def _key_selection_progress(
    t_done: int,
    r: int,
    *,
    oracle: str,
    show_progress: bool,
) -> Iterable[int]:
    """Iterate selection steps ``t_done .. r-1``, optionally with a tqdm bar."""
    steps = range(t_done, r)
    if not show_progress or r <= t_done:
        return steps
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return steps
    return tqdm(
        steps,
        total=r,
        initial=t_done,
        desc=f"TFCFW-lq ({oracle})",
        unit="key",
        leave=False,
    )


def build_attention_profiles(
    probe_queries: torch.Tensor,
    cand_keys: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Stabilized attention-profile matrix ``A[q, i] = exp(scale q.k_i - c)``."""
    logits = scale * (probe_queries @ cand_keys.t())   # [m, n]
    logits = logits - logits.max()
    return logits.exp()


def _truncated_solve(
    aw: torch.Tensor,
    tw: torch.Tensor,
    rcond: float,
) -> torch.Tensor:
    """Minimum-norm, ridge-damped least squares solution of ``aw U = tw``."""
    u, s, vh = torch.linalg.svd(aw, full_matrices=False)
    if s.numel() == 0:
        return torch.zeros(
            (aw.shape[-1], tw.shape[-1]), dtype=aw.dtype, device=aw.device,
        )
    lam = (rcond * s[0]) ** 2
    sinv = s / (s * s + lam)
    coef = vh.transpose(-2, -1) @ (sinv.unsqueeze(-1) * (u.transpose(-2, -1) @ tw))
    return coef


def _irls_solve(
    a_sel: torch.Tensor,
    target: torch.Tensor,
    irls_iters: int,
    rcond: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """IRLS for L_{2,1} over query rows; ``target`` is ``[m, p]``."""
    u = _truncated_solve(a_sel, target, rcond)
    for _ in range(max(int(irls_iters) - 1, 0)):
        resid = target - a_sel @ u
        rnorm = resid.norm(dim=-1).clamp_min(eps)
        sw = (1.0 / rnorm).sqrt().unsqueeze(-1)
        u = _truncated_solve(a_sel * sw, target * sw, rcond)
    return u


def _resolve_residual_lq_irls_iters(
    irls_iters: int,
    scoring_irls_iters: Optional[int] = None,
    correction_irls_iters: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Return ``(scoring_iters, correction_iters, final_iters)`` for residual_lq."""
    final = int(irls_iters)
    scoring = int(scoring_irls_iters if scoring_irls_iters is not None else final)
    correction = int(
        correction_irls_iters if correction_irls_iters is not None else final,
    )
    return scoring, correction, final


def correction_interval(support_size: int, period: int) -> int:
    """Keys between full support IRLS refreshes.

    ``period <= 0`` => refresh every key. Otherwise refresh every
    ``ceil(support_size / period)`` keys once the support has that size
    (more frequent when ``support_size`` is small, rarer as it grows).
    """
    if period <= 0:
        return 1
    return max(1, (int(support_size) + period - 1) // period)


def lq_objective(
    design: torch.Tensor,
    values: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-query L2 residual: ``(1/m) sum_q ||target_q - design_q @ values||_2``."""
    return (target - design @ values).norm(dim=-1).mean()


def build_normalized_candidate_design(
    probe_queries: torch.Tensor,
    cand_keys: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    sp_idx: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-candidate design for exact-denom training: ``coef[:,j] @ v_j`` sums to core num.

    Returns ``(coef [m, n], rhs [m, d])`` so ``coef @ V_sel ≈ rhs`` matches
    ``(N_sp + N_core) / Z_exact`` on the reference context.
    """
    logits = scale * (probe_queries @ ref_keys.t())
    shift = logits.amax(dim=-1, keepdim=True)
    e = (logits - shift).exp()
    z = e.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    full_target = (e @ ref_values) / z

    if sp_idx.numel() > 0:
        sp_e = (logits[:, sp_idx] - shift).exp()
        sp_num = sp_e @ ref_values[sp_idx]
    else:
        sp_num = torch.zeros_like(full_target)

    core_logits = scale * (probe_queries @ cand_keys.t())
    coef = (core_logits - shift).exp() / z
    rhs = full_target - sp_num / z
    return coef, rhs


def _normalized_correction_system(
    probe_queries: torch.Tensor,
    core_keys: torch.Tensor,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    sp_idx: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear system for a fixed core key set: ``coef @ V ≈ rhs``."""
    coef_all, rhs = build_normalized_candidate_design(
        probe_queries, core_keys, ref_keys, ref_values, sp_idx, scale,
    )
    return coef_all, rhs


def build_lq_design_and_target(
    probes: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float,
    numerator_only: bool,
    ref_keys: Optional[torch.Tensor] = None,
    ref_values: Optional[torch.Tensor] = None,
    sp_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared ``(design [m, n], target [m, p])`` for selection and correction."""
    if numerator_only:
        use_norm = (
            ref_keys is not None
            and ref_values is not None
            and sp_idx is not None
        )
        if use_norm:
            return build_normalized_candidate_design(
                probes,
                keys,
                ref_keys,
                ref_values,
                sp_idx.to(device=keys.device, dtype=torch.long),
                scale,
            )
        design = build_attention_profiles(probes, keys, scale)
        return design, design @ values

    ones_col = torch.ones((keys.shape[0], 1), dtype=keys.dtype, device=keys.device)
    mat = torch.cat([ones_col, values], dim=-1)
    design = build_attention_profiles(probes, keys, scale)
    return design, design @ mat


def _select_fc_lq(
    design: torch.Tensor,
    target: torch.Tensor,
    r: int,
    irls_iters: int,
    rcond: float,
    state: Optional[Dict],
    show_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Exact fully-corrective lq Frank–Wolfe over candidate columns."""
    n = design.shape[1]
    m = int(design.shape[0])
    device = design.device
    dtype = design.dtype
    r = min(int(r), n)

    reuse = (
        state is not None
        and state.get("oracle") == "fc_lq"
        and int(state.get("n", -1)) == n
        and int(state.get("m", -1)) == m
    )
    if reuse:
        selected: List[int] = list(state["selected_list"])
        cmpd_values = state["U"]
        t_done = len(selected)
    else:
        selected = []
        cmpd_values = torch.empty((0, target.shape[-1]), dtype=dtype, device=device)
        t_done = 0

    if r > t_done:
        selected_set = set(selected)
        for _ in _key_selection_progress(
            t_done, r, oracle="fc_lq", show_progress=show_progress,
        ):
            best_obj: Optional[torch.Tensor] = None
            best_i = -1
            best_u: Optional[torch.Tensor] = None

            for i in range(n):
                if i in selected_set:
                    continue
                trial = selected + [i]
                idx_t = torch.tensor(trial, dtype=torch.long, device=device)
                a_try = design[:, idx_t]
                u_try = _irls_solve(a_try, target, irls_iters, rcond)
                obj = lq_objective(a_try, u_try, target)
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_i = i
                    best_u = u_try

            if best_i < 0:
                break
            selected.append(best_i)
            selected_set.add(best_i)
            cmpd_values = best_u

        t_done = len(selected)

    selected_t = torch.tensor(selected[:r], dtype=torch.long, device=device)
    new_state = {
        "oracle": "fc_lq",
        "selected_list": selected[:r],
        "U": cmpd_values,
        "t": t_done,
        "n": n,
        "m": m,
    }
    return selected_t, cmpd_values, new_state


def _residual_lq_norm(target: torch.Tensor) -> torch.Tensor:
    """Mean per-query L2 norm of rows (empty-support residual)."""
    return target.norm(dim=-1).mean()


def _after_row_norms(
    residual: torch.Tensor,
    design_chunk: torch.Tensor,
    u: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-query L2 norms of one-column residual fits; returns ``[m, nc]``."""
    r_norm_sq = (residual * residual).sum(dim=-1)
    u_norm_sq = (u * u).sum(dim=-1)
    rud = residual @ u.t()
    d = design_chunk
    after_norm_sq = (
        r_norm_sq.unsqueeze(-1)
        + (d * d) * u_norm_sq.unsqueeze(0)
        - 2.0 * d * rud
    ).clamp_min(eps)
    return after_norm_sq.sqrt()


def _batched_single_column_irls(
    design_chunk: torch.Tensor,
    target: torch.Tensor,
    irls_iters: int,
    rcond: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Independent one-column IRLS for every column in ``design_chunk``."""
    col_sq = (design_chunk * design_chunk).sum(dim=0).clamp_min(eps)
    lam = (rcond * col_sq.sqrt()) ** 2

    u = (design_chunk.t() @ target) / (col_sq.unsqueeze(-1) + lam.unsqueeze(-1))

    for _ in range(max(int(irls_iters) - 1, 0)):
        after_norms = _after_row_norms(target, design_chunk, u, eps)
        sw_sq = 1.0 / after_norms.clamp_min(eps)
        weighted = design_chunk * sw_sq
        col_sq_w = (weighted * design_chunk).sum(dim=0).clamp_min(eps)
        lam_w = (rcond * col_sq_w.sqrt()) ** 2
        u = (weighted.t() @ target) / (col_sq_w.unsqueeze(-1) + lam_w.unsqueeze(-1))

    return u


def _score_residual_lq_candidates(
    design: torch.Tensor,
    residual: torch.Tensor,
    irls_iters: int,
    rcond: float,
    eligible: torch.Tensor,
    chunk_size: int = 2048,
    eps: float = 1e-12,
) -> Tuple[int, torch.Tensor]:
    """Vectorized residual-lq gains for all candidates; returns best index."""
    n = design.shape[1]
    device = design.device
    dtype = design.dtype
    gains = torch.full((n,), float("-inf"), device=device, dtype=dtype)
    base_norm = _residual_lq_norm(residual)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d_chunk = design[:, start:end]
        elig = eligible[start:end]
        if not bool(elig.any()):
            continue

        u = _batched_single_column_irls(
            d_chunk, residual, irls_iters, rcond, eps,
        )
        chunk_gains = base_norm - _after_row_norms(
            residual, d_chunk, u, eps,
        ).mean(dim=0)
        chunk_gains = chunk_gains.masked_fill(~elig, float("-inf"))
        gains[start:end] = chunk_gains

    best_i = int(torch.argmax(gains).item())
    if not torch.isfinite(gains[best_i]):
        return -1, gains
    return best_i, gains


def _deflate_design_columns(
    design: torch.Tensor,
    design_chunk: torch.Tensor,
    selected_idx: torch.Tensor,
) -> torch.Tensor:
    """Remove the component of ``design_chunk`` in span of selected columns."""
    if selected_idx.numel() == 0:
        return design_chunk
    basis = design[:, selected_idx]
    q, _ = torch.linalg.qr(basis, mode='reduced')
    return design_chunk - q @ (q.t() @ design_chunk)


def _score_residual_lq_candidates_deflated(
    design: torch.Tensor,
    residual: torch.Tensor,
    selected_idx: torch.Tensor,
    irls_iters: int,
    rcond: float,
    eligible: torch.Tensor,
    chunk_size: int = 2048,
    eps: float = 1e-12,
) -> Tuple[int, torch.Tensor]:
    """Vectorized lq gains on Cholesky-orthogonalized (QR-deflated) columns."""
    n = design.shape[1]
    device = design.device
    dtype = design.dtype
    gains = torch.full((n,), float("-inf"), device=device, dtype=dtype)
    base_norm = _residual_lq_norm(residual)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d_chunk = _deflate_design_columns(
            design, design[:, start:end], selected_idx,
        )
        elig = eligible[start:end]
        if not bool(elig.any()):
            continue

        u = _batched_single_column_irls(
            d_chunk, residual, irls_iters, rcond, eps,
        )
        chunk_gains = base_norm - _after_row_norms(
            residual, d_chunk, u, eps,
        ).mean(dim=0)
        chunk_gains = chunk_gains.masked_fill(~elig, float("-inf"))
        gains[start:end] = chunk_gains

    best_i = int(torch.argmax(gains).item())
    if not torch.isfinite(gains[best_i]):
        return -1, gains
    return best_i, gains


def _residual_lq_best_column_scalar(
    design: torch.Tensor,
    residual: torch.Tensor,
    i: int,
    irls_iters: int,
    rcond: float,
) -> torch.Tensor:
    """Scalar reference: lq residual drop from adding column ``i``."""
    base_norm = _residual_lq_norm(residual)
    col = design[:, i:i + 1]
    v_i = _irls_solve(col, residual, irls_iters, rcond)
    after = residual - col @ v_i
    return base_norm - _residual_lq_norm(after)


def _select_residual_lq(
    design: torch.Tensor,
    target: torch.Tensor,
    r: int,
    irls_iters: int,
    rcond: float,
    state: Optional[Dict],
    show_progress: bool = False,
    scoring_irls_iters: Optional[int] = None,
    correction_irls_iters: Optional[int] = None,
    correction_period: int = 400,
    oracle_name: str = "residual_lq",
    deflate_scoring: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Greedy residual lq with scheduled full support IRLS refreshes."""
    scoring_iters, correction_iters, final_iters = _resolve_residual_lq_irls_iters(
        irls_iters, scoring_irls_iters, correction_irls_iters,
    )
    period = int(correction_period)
    n = design.shape[1]
    m = int(design.shape[0])
    device = design.device
    dtype = design.dtype
    r = min(int(r), n)

    reuse = (
        state is not None
        and state.get("oracle") == oracle_name
        and int(state.get("n", -1)) == n
        and int(state.get("m", -1)) == m
        and int(state.get("correction_period", period)) == period
        and bool(state.get("deflate_scoring", False)) == bool(deflate_scoring)
    )
    if reuse:
        selected: List[int] = list(state["selected_list"])
        cmpd_values = state["U"]
        t_done = len(selected)
        keys_since_correction = int(state.get("keys_since_correction", 0))
    else:
        selected = []
        cmpd_values = torch.empty((0, target.shape[-1]), dtype=dtype, device=device)
        t_done = 0
        keys_since_correction = 0

    if r > t_done:
        eligible = torch.ones(n, dtype=torch.bool, device=device)
        if selected:
            eligible[torch.tensor(selected, dtype=torch.long, device=device)] = False
        selected_idx = torch.tensor(selected, dtype=torch.long, device=device)
        if t_done > 0:
            residual = target - design[:, selected_idx] @ cmpd_values
        else:
            residual = target

        for _ in _key_selection_progress(
            t_done, r, oracle=oracle_name, show_progress=show_progress,
        ):
            if deflate_scoring:
                best_i, _ = _score_residual_lq_candidates_deflated(
                    design,
                    residual,
                    selected_idx,
                    scoring_iters,
                    rcond,
                    eligible,
                )
            else:
                best_i, _ = _score_residual_lq_candidates(
                    design,
                    residual,
                    scoring_iters,
                    rcond,
                    eligible,
                )

            if best_i < 0:
                break

            u_new = _batched_single_column_irls(
                design[:, best_i:best_i + 1], residual, scoring_iters, rcond,
            )

            selected.append(best_i)
            eligible[best_i] = False
            selected_idx = torch.tensor(selected, dtype=torch.long, device=device)
            keys_since_correction += 1
            t_new = len(selected)
            interval = correction_interval(t_new, period)

            if keys_since_correction >= interval:
                cmpd_values = _irls_solve(
                    design[:, selected_idx], target, correction_iters, rcond,
                )
                residual = target - design[:, selected_idx] @ cmpd_values
                keys_since_correction = 0
            else:
                cmpd_values = (
                    u_new if cmpd_values.numel() == 0
                    else torch.cat([cmpd_values, u_new], dim=0)
                )
                residual = residual - design[:, best_i:best_i + 1] @ u_new

        t_done = len(selected)

    selected_t = torch.tensor(selected[:r], dtype=torch.long, device=device)
    if selected_t.numel() > 0 and correction_iters < final_iters:
        cmpd_values = _irls_solve(
            design[:, selected_t], target, final_iters, rcond,
        )
    new_state = {
        "oracle": oracle_name,
        "selected_list": selected[:r],
        "U": cmpd_values,
        "t": t_done,
        "n": n,
        "m": m,
        "keys_since_correction": keys_since_correction,
        "correction_period": period,
        "deflate_scoring": bool(deflate_scoring),
    }
    return selected_t, cmpd_values, new_state


def _select_fw_cholesky(
    design: torch.Tensor,
    target: torch.Tensor,
    values: torch.Tensor,
    mat: torch.Tensor,
    r: int,
    oracle: str,
    jitter: float,
    numerator_only: bool,
    state: Optional[Dict],
    show_progress: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Fast pivoted-Cholesky key selection (Frobenius surrogate)."""
    work_dtype = design.dtype
    device = design.device
    n = design.shape[1]
    r = min(int(r), n)

    reuse = (
        state is not None
        and state.get("oracle") in (None, "fw", "omp")
        and int(state.get("n", -1)) == n
        and int(state.get("m", -1)) == design.shape[0]
        and bool(state.get("numerator_only", True)) == bool(numerator_only)
    )
    if reuse:
        g_res = state["G"]
        res_diag = state["res_diag"]
        factor = state["L"]
        selected = state["selected"]
        t_done = int(state["t"])
    else:
        g_res = design.t() @ target
        res_diag = (design * design).sum(dim=0)
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
        for t in _key_selection_progress(
            t_done, r, oracle=oracle, show_progress=show_progress,
        ):
            score = (g_res * g_res).sum(dim=-1)
            if oracle == "omp":
                score = score / res_diag.clamp(min=jitter)
            if t > 0:
                score = score.scatter(0, selected[:t], neg_inf)
            idx = int(torch.argmax(score).item())
            selected[t] = idx

            kcol = design.t() @ design[:, idx]
            diag = res_diag[idx].clamp(min=jitter)

            if t > 0:
                lp = factor[idx, :t]
                proj = factor[:, :t] @ lp
                ecol = kcol - proj
            else:
                ecol = kcol

            ell = ecol / diag.sqrt()
            factor[:, t] = ell

            if numerator_only:
                ell_contrib = ell @ values
                g_res = g_res - ell.unsqueeze(-1) * ell_contrib.unsqueeze(0)
            else:
                ell_mat = ell @ mat
                g_res = g_res - ell.unsqueeze(-1) * ell_mat.unsqueeze(0)
            res_diag = res_diag - ell * ell
            res_diag[idx] = 0.0
            res_diag = res_diag.clamp(min=0.0)
        t_done = r

    new_state = {
        "oracle": oracle,
        "G": g_res,
        "res_diag": res_diag,
        "L": factor,
        "selected": selected,
        "t": t_done,
        "n": n,
        "m": int(design.shape[0]),
        "numerator_only": bool(numerator_only),
    }
    return selected[:r], new_state


def _finalize_lq_coreset_weights(
    coreset: torch.Tensor,
    cmpd_values: torch.Tensor,
    *,
    exact_denominator: bool,
    probes: torch.Tensor,
    cand_keys: torch.Tensor,
    ref_keys: Optional[torch.Tensor],
    ref_values: Optional[torch.Tensor],
    sp_idx: Optional[torch.Tensor],
    scale: float,
    irls_iters: int,
    rcond: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit weights at selection; local mass calibration runs in stage 2."""
    del (
        probes, cand_keys, ref_keys, ref_values, sp_idx, scale,
        irls_iters, rcond, seed,
    )
    device = cmpd_values.device
    dtype = cmpd_values.dtype
    if coreset.numel() == 0:
        return cmpd_values, torch.zeros(0, dtype=dtype, device=device)
    return cmpd_values, torch.ones(coreset.shape[0], dtype=dtype, device=device)


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
    exact_denominator: bool = True,
    ref_keys: Optional[torch.Tensor] = None,
    ref_values: Optional[torch.Tensor] = None,
    sp_idx: Optional[torch.Tensor] = None,
    show_progress: Optional[bool] = None,
    scoring_irls_iters: Optional[int] = None,
    correction_irls_iters: Optional[int] = None,
    correction_period: int = 400,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Select ``r`` candidate keys and synthetic values under the ``lq`` norm.

    ``oracle='fc_lq'``: exact fully-corrective lq Frank–Wolfe (joint ``(k', v')``
    per step via exhaustive search + IRLS on the enlarged support).

    ``oracle='residual_lq'``: OMP-style residual scoring (one-column lq fit per
    candidate key) + scheduled support refresh. ``scoring_irls_iters`` scores
    candidates; ``correction_irls_iters`` runs on refresh steps every
    ``ceil(|S| / correction_period)`` keys; ``irls_iters`` is the final solve.

    ``oracle='residual_lq_deflated'``: same loop, but scores each candidate on
    probe columns orthogonal to the current support before the one-column lq fit.

    ``oracle='fw'`` / ``'omp'``: fast Cholesky surrogate selection + final IRLS.
    """
    if oracle not in ("fw", "omp", "fc_lq", "residual_lq", "residual_lq_deflated"):
        raise ValueError(
            f"oracle must be 'fw', 'omp', 'fc_lq', 'residual_lq', or "
            f"'residual_lq_deflated'; got {oracle!r}",
        )

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
    progress = _resolve_show_progress(show_progress)

    ones_col = torch.ones((n, 1), dtype=work_dtype, device=device)
    mat = torch.cat([ones_col, values], dim=-1)

    ref_k = ref_v = None
    sp_t = None
    if ref_keys is not None and ref_values is not None and sp_idx is not None:
        ref_k = ref_keys.to(work_dtype)
        ref_v = ref_values.to(work_dtype)
        sp_t = sp_idx.to(device=device, dtype=torch.long)

    # Key selection + value IRLS always use the numerator (values-only) design.
    # When ``exact_denominator=True`` and reference context is available, use the
    # normalized exact-denom training system; otherwise plain attention profiles.
    if exact_denominator and ref_k is not None and ref_v is not None and sp_t is not None:
        design, target = build_lq_design_and_target(
            probes, keys, values, scale, True, ref_k, ref_v, sp_t,
        )
    else:
        design, target = build_lq_design_and_target(
            probes, keys, values, scale, True, None, None, None,
        )

    if oracle == "fc_lq":
        coreset, cmpd_values, new_state = _select_fc_lq(
            design, target, r, irls_iters, rcond, state, progress,
        )
        new_state["exact_denominator"] = bool(exact_denominator)
        cmpd_values, w = _finalize_lq_coreset_weights(
            coreset,
            cmpd_values,
            exact_denominator=exact_denominator,
            probes=probes,
            cand_keys=keys,
            ref_keys=ref_k,
            ref_values=ref_v,
            sp_idx=sp_t,
            scale=scale,
            irls_iters=irls_iters,
            rcond=rcond,
            seed=seed,
        )
        return coreset, cmpd_values, w, new_state

    if oracle in ("residual_lq", "residual_lq_deflated"):
        coreset, cmpd_values, new_state = _select_residual_lq(
            design,
            target,
            r,
            irls_iters,
            rcond,
            state,
            progress,
            scoring_irls_iters=scoring_irls_iters,
            correction_irls_iters=correction_irls_iters,
            correction_period=correction_period,
            oracle_name=oracle,
            deflate_scoring=(oracle == "residual_lq_deflated"),
        )
        new_state["exact_denominator"] = bool(exact_denominator)
        cmpd_values, w = _finalize_lq_coreset_weights(
            coreset,
            cmpd_values,
            exact_denominator=exact_denominator,
            probes=probes,
            cand_keys=keys,
            ref_keys=ref_k,
            ref_values=ref_v,
            sp_idx=sp_t,
            scale=scale,
            irls_iters=irls_iters,
            rcond=rcond,
            seed=seed,
        )
        return coreset, cmpd_values, w, new_state

    coreset, new_state = _select_fw_cholesky(
        design, target, values, mat, r, oracle, jitter,
        True, state, progress,
    )
    a_sel = design[:, coreset]
    cmpd_values = _irls_solve(a_sel, target, irls_iters, rcond)
    cmpd_values, w = _finalize_lq_coreset_weights(
        coreset,
        cmpd_values,
        exact_denominator=exact_denominator,
        probes=probes,
        cand_keys=keys,
        ref_keys=ref_k,
        ref_values=ref_v,
        sp_idx=sp_t,
        scale=scale,
        irls_iters=irls_iters,
        rcond=rcond,
        seed=seed,
    )
    new_state["exact_denominator"] = bool(exact_denominator)

    return coreset, cmpd_values, w, new_state
