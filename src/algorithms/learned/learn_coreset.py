"""
Learn synthetic (K', V', w') triples by gradient descent.

The training objective matches the *exact evaluation pipeline*: for each probe
query, the approximate attention (exact sink + local-window tokens concatenated
with the learned pairs, normalized exactly as ``weighted_attention``) is matched
against the true full attention over the fixed reference (test) context. Only
the residual (candidate-region) pairs ``(k'_j, v'_j, w'_j)`` are learned; the
sink and local window stay exact and are added at evaluation.

Design:

* Each pair carries a learnable mass ``w'_j`` (folded into the numerator as
  ``w'_j v'_j`` and the denominator as ``w'_j``), so the coreset matches the
  *unnormalized* numerator/denominator -- not just the softmax average.
* Initialization is "pure" (k-means or random over candidate keys/values), not
  FCFW, so the method is a self-contained learned baseline.
* **Monotone budget sweep (nested):** a larger budget freezes the smaller
  budget's trained coreset (folded into a fixed per-probe contribution, exactly
  like the special tokens) and trains only the newly added pairs, initialized at
  near-zero mass. The starting point reproduces the smaller coreset, so error is
  non-increasing in budget (in the training/validation objective).
* The forward pass matches ``weighted_attention`` numerically: with
  ``exact_denominator`` the shift uses the global max logit over all causal keys
  and divides by ``Z_exact``; otherwise the shift is the max over the coreset
  (special + frozen + new) and the denominator is the coreset KDE mass.
"""

from __future__ import annotations

from math import log
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...core import compute_special_indices, flat_kmeans

_EXP_CLAMP = 40.0
_ZERO_MASS_LOG = log(1e-3)


def _kmeans_init(
    cand_keys: np.ndarray,
    cand_values: np.ndarray,
    n_new: int,
    with_mass: bool,
    subsample: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k-means centroids over candidate keys; values = per-cluster mean."""
    n = cand_keys.shape[0]
    d = cand_keys.shape[1]
    if n > subsample:
        sub = rng.choice(n, size=subsample, replace=False)
        ck = cand_keys[sub]
        cv = cand_values[sub]
    else:
        ck = cand_keys
        cv = cand_values

    n_clusters = min(n_new, ck.shape[0])
    centroids, labels = flat_kmeans(ck, n_clusters, seed=int(rng.integers(1 << 30)))

    v_new = np.zeros((n_clusters, d), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        mask = labels == c
        cnt = int(mask.sum())
        counts[c] = cnt
        if cnt > 0:
            v_new[c] = cv[mask].mean(axis=0)
        else:
            v_new[c] = cv[rng.integers(cv.shape[0])]

    k_new = centroids.astype(np.float32)
    if n_clusters < n_new:
        pad = n_new - n_clusters
        k_new = np.concatenate(
            [k_new, rng.standard_normal((pad, d)).astype(np.float32) * 0.02], axis=0,
        )
        v_new = np.concatenate(
            [v_new, rng.standard_normal((pad, d)).astype(np.float32) * 0.02], axis=0,
        )
        counts = np.concatenate([counts, np.ones(pad)], axis=0)

    if with_mass:
        # Scale cluster counts up to the full candidate population.
        mass = counts / max(counts.sum(), 1.0) * float(n)
        logw = np.log(np.clip(mass, 1e-8, None)).astype(np.float32)
    else:
        logw = np.full(n_new, _ZERO_MASS_LOG, dtype=np.float32)

    return (
        torch.as_tensor(k_new, device=device),
        torch.as_tensor(v_new, device=device),
        torch.as_tensor(logw, device=device),
    )


def _mqbeta_init(
    cand_keys: np.ndarray,
    cand_values: np.ndarray,
    n_new: int,
    with_mass: bool,
    probe_queries: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MQBeta-style init: M_Q-weighted k-means with importance weights rho."""
    from ..mq_beta_cluster import _weighted_kmeans

    n = cand_keys.shape[0]
    d = cand_keys.shape[1]
    n_clusters = min(n_new, n)
    sqrt_d = np.sqrt(d)

    Q_train = probe_queries.astype(np.float64)
    K_f = cand_keys.astype(np.float64)

    # M_Q transform
    M_Q = Q_train.T @ Q_train + 1e-6 * np.eye(d)
    eigvals, eigvecs = np.linalg.eigh(M_Q)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_eig = np.sqrt(eigvals)

    # Importance weights rho
    BATCH = 500
    rho_sum = np.zeros(n, np.float64)
    for b0 in range(0, len(Q_train), BATCH):
        b1 = min(b0 + BATCH, len(Q_train))
        logits_b = (Q_train[b0:b1] @ K_f.T) / sqrt_d
        s_max = logits_b.max(axis=1, keepdims=True)
        rho_sum += np.sum(np.exp(logits_b - s_max), axis=0)

    # Weighted k-means in M_Q space
    K_z = (K_f @ eigvecs * sqrt_eig[None, :]).astype(np.float32)
    _, labels = _weighted_kmeans(K_z, rho_sum, n_clusters,
                                 seed=int(rng.integers(1 << 30)), n_iter=50)

    # Cluster centroids and value means
    k_new = np.zeros((n_clusters, d), np.float32)
    v_new = np.zeros((n_clusters, d), np.float32)
    counts = np.zeros(n_clusters, np.float64)
    for c in range(n_clusters):
        mask = labels == c
        cnt = int(mask.sum())
        counts[c] = cnt
        if cnt > 0:
            k_new[c] = cand_keys[mask].mean(axis=0)
            v_new[c] = cand_values[mask].mean(axis=0)

    if n_clusters < n_new:
        pad = n_new - n_clusters
        k_new = np.concatenate(
            [k_new, rng.standard_normal((pad, d)).astype(np.float32) * 0.02], axis=0)
        v_new = np.concatenate(
            [v_new, rng.standard_normal((pad, d)).astype(np.float32) * 0.02], axis=0)
        counts = np.concatenate([counts, np.ones(pad)], axis=0)

    if with_mass:
        mass = counts / max(counts.sum(), 1.0) * float(n)
        logw = np.log(np.clip(mass, 1e-8, None)).astype(np.float32)
    else:
        logw = np.full(n_new, _ZERO_MASS_LOG, dtype=np.float32)

    return (
        torch.as_tensor(k_new, device=device),
        torch.as_tensor(v_new, device=device),
        torch.as_tensor(logw, device=device),
    )


def _random_gauss_init(
    n_new: int,
    d: int,
    n_cand: int,
    with_mass: bool,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random gaussian vectors — no data-dependent initialization."""
    k_new = rng.standard_normal((n_new, d)).astype(np.float32) * 0.1
    v_new = rng.standard_normal((n_new, d)).astype(np.float32) * 0.1
    if with_mass:
        logw = np.full(n_new, log(max(n_cand / max(n_new, 1), 1e-8)), dtype=np.float32)
    else:
        logw = np.full(n_new, _ZERO_MASS_LOG, dtype=np.float32)
    return (
        torch.as_tensor(k_new, device=device),
        torch.as_tensor(v_new, device=device),
        torch.as_tensor(logw, device=device),
    )


def _first_init(
    cand_keys: np.ndarray,
    cand_values: np.ndarray,
    n_new: int,
    with_mass: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First B candidate (k, v) pairs — naive positional init."""
    n = cand_keys.shape[0]
    n_pick = min(n_new, n)
    k_new = cand_keys[:n_pick].astype(np.float32).copy()
    v_new = cand_values[:n_pick].astype(np.float32).copy()
    d = cand_keys.shape[1]
    if n_pick < n_new:
        pad = n_new - n_pick
        k_new = np.concatenate(
            [k_new, np.zeros((pad, d), dtype=np.float32)], axis=0)
        v_new = np.concatenate(
            [v_new, np.zeros((pad, d), dtype=np.float32)], axis=0)
    if with_mass:
        logw = np.full(n_new, log(max(n / max(n_new, 1), 1e-8)), dtype=np.float32)
    else:
        logw = np.full(n_new, _ZERO_MASS_LOG, dtype=np.float32)
    return (
        torch.as_tensor(k_new, device=device),
        torch.as_tensor(v_new, device=device),
        torch.as_tensor(logw, device=device),
    )


def _random_init(
    cand_keys: np.ndarray,
    cand_values: np.ndarray,
    n_new: int,
    with_mass: bool,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random subset of candidate (k, v) pairs."""
    n = cand_keys.shape[0]
    d = cand_keys.shape[1]
    n_pick = min(n_new, n)
    pick = rng.choice(n, size=n_pick, replace=False)
    k_new = cand_keys[pick].astype(np.float32)
    v_new = cand_values[pick].astype(np.float32)
    if n_pick < n_new:
        pad = n_new - n_pick
        k_new = np.concatenate(
            [k_new, rng.standard_normal((pad, d)).astype(np.float32) * 0.02], axis=0,
        )
        v_new = np.concatenate(
            [v_new, rng.standard_normal((pad, d)).astype(np.float32) * 0.02], axis=0,
        )
    if with_mass:
        logw = np.full(n_new, log(max(n / max(n_new, 1), 1e-8)), dtype=np.float32)
    else:
        logw = np.full(n_new, _ZERO_MASS_LOG, dtype=np.float32)
    return (
        torch.as_tensor(k_new, device=device),
        torch.as_tensor(v_new, device=device),
        torch.as_tensor(logw, device=device),
    )


def _precompute_targets(
    probe_queries: torch.Tensor,
    keys_ref: torch.Tensor,
    values_ref: torch.Tensor,
    sp_idx: np.ndarray,
    frozen: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    scale: float,
) -> dict:
    """
    Fixed per-probe constants over the reference context.

  Stores logits/values for each coreset component (special, frozen, new) so
  ``_forward_pred`` can apply the same max-logit shift as ``weighted_attention``:
  global max over all causal keys when ``exact_denominator`` is true, and
  coreset-local max (special + frozen + new) when it is false.
    """
    logits = scale * (probe_queries @ keys_ref.T)        # [m, Nref]
    max_global = logits.amax(dim=-1)                       # [m]
    e = (logits - max_global.unsqueeze(-1)).exp()          # [m, Nref]
    z_exact = e.sum(dim=-1)                                # [m]
    target = (e @ values_ref) / z_exact.unsqueeze(-1)      # [m, d]

    if len(sp_idx) > 0:
        sp = torch.as_tensor(sp_idx, dtype=torch.long, device=e.device)
        sp_logits = logits[:, sp]                          # [m, n_sp]
        sp_values = values_ref[sp]                         # [n_sp, d]
    else:
        sp_logits = None
        sp_values = None

    if frozen is not None:
        fk, fv, fw = frozen
        frozen_logits = scale * (probe_queries @ fk.T)     # [m, n_frozen]
        frozen_values = fv
        frozen_weights = fw
    else:
        frozen_logits = None
        frozen_values = None
        frozen_weights = None

    return {
        "queries": probe_queries.detach(),
        "max_global": max_global.detach(),
        "z_exact": z_exact.detach(),
        "target": target.detach(),
        "sp_logits": None if sp_logits is None else sp_logits.detach(),
        "sp_values": None if sp_values is None else sp_values.detach(),
        "frozen_logits": None if frozen_logits is None else frozen_logits.detach(),
        "frozen_values": None if frozen_values is None else frozen_values.detach(),
        "frozen_weights": None if frozen_weights is None else frozen_weights.detach(),
    }


def _forward_pred(
    consts: dict,
    idx: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    log_w: torch.Tensor,
    scale: float,
    exact_denominator: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate attention for a subset of probes; mirrors ``weighted_attention``."""
    target = consts["target"][idx]                       # [b, d]
    w_new = log_w.exp()

    logits_parts: list[torch.Tensor] = []
    if consts["sp_logits"] is not None:
        logits_parts.append(consts["sp_logits"][idx])
    if consts["frozen_logits"] is not None:
        logits_parts.append(consts["frozen_logits"][idx])
    logits_new = scale * (consts["queries"][idx] @ k_new.T)
    logits_parts.append(logits_new)
    all_core_logits = torch.cat(logits_parts, dim=-1)      # [b, n_core]

    if exact_denominator:
        shift = consts["max_global"][idx].unsqueeze(-1)
    else:
        shift = all_core_logits.amax(dim=-1, keepdim=True)

    num = torch.zeros_like(target)
    den_core = torch.zeros(target.shape[0], device=target.device, dtype=target.dtype)

    offset = 0
    if consts["sp_logits"] is not None:
        n_sp = consts["sp_logits"].shape[1]
        e_sp = (all_core_logits[:, offset:offset + n_sp] - shift).clamp(max=_EXP_CLAMP).exp()
        num = num + e_sp @ consts["sp_values"]
        den_core = den_core + e_sp.sum(dim=-1)
        offset += n_sp

    if consts["frozen_logits"] is not None:
        n_fr = consts["frozen_logits"].shape[1]
        e_fr = (all_core_logits[:, offset:offset + n_fr] - shift).clamp(max=_EXP_CLAMP).exp()
        fw = consts["frozen_weights"]
        ew_fr = e_fr * fw.unsqueeze(0)
        num = num + ew_fr @ consts["frozen_values"]
        den_core = den_core + ew_fr.sum(dim=-1)
        offset += n_fr

    n_new = logits_new.shape[1]
    e_new = (all_core_logits[:, offset:offset + n_new] - shift).clamp(max=_EXP_CLAMP).exp()
    ew_new = e_new * w_new.unsqueeze(0)
    num = num + ew_new @ v_new
    den_core = den_core + ew_new.sum(dim=-1)

    if exact_denominator:
        den = consts["z_exact"][idx].unsqueeze(-1)
    else:
        den = den_core.unsqueeze(-1)

    pred = num / den.clamp_min(1e-20)
    return pred, target


def _loss_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss: str,
    rel_l2_floor: float,
) -> torch.Tensor:
    sq = (pred - target).pow(2).sum(dim=-1)              # [b]
    if loss == "relative_l2":
        tnorm2 = target.pow(2).sum(dim=-1)               # [b]
        if rel_l2_floor > 0.0:
            floor = rel_l2_floor * tnorm2.median().detach()
            denom = (tnorm2 + floor).clamp_min(1e-12)
        else:
            denom = tnorm2.clamp_min(1e-12)
        return (sq / denom).mean()
    return sq.mean()


def learn_kv_coreset(
    keys: np.ndarray,
    values: np.ndarray,
    head_dim: int,
    probe_queries: np.ndarray,
    ref_pos: int,
    budget: int,
    n_sink: int,
    local_window: int,
    *,
    init: str = "kmeans",
    exact_denominator: bool = True,
    lr: float = 0.05,
    n_steps: int = 500,
    batch_size: int = 128,
    loss: str = "relative_l2",
    rel_l2_floor: float = 0.01,
    val_fraction: float = 0.1,
    early_stop_patience: int = 50,
    lr_decay_step: int = 200,
    lr_decay_gamma: float = 0.5,
    frozen: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    kmeans_subsample: int = 8192,
    device: Optional[torch.device] = None,
    seed: int = 42,
    return_history: bool = False,
):
    """
    Optimize synthetic (K', V', w') to match full attention over the reference context.

    If ``frozen`` (k, v, w) is given, those pairs are kept fixed and only the
    newly added ``budget - len(frozen)`` pairs are trained (nested/monotone).

    Returns (K_prime, V_prime, w_prime) float32 numpy arrays of sizes
    (budget, d), (budget, d), (budget,).
    """
    if device is None:
        device = torch.device("cpu")
    if budget <= 0:
        raise ValueError(f"budget must be positive; got {budget}")
    probe_queries = np.asarray(probe_queries, dtype=np.float32)
    if probe_queries.ndim != 2 or probe_queries.shape[0] == 0:
        raise ValueError("probe_queries must be a non-empty [m, d] array")

    rng = np.random.default_rng(seed)
    d = keys.shape[1]
    n_causal = ref_pos + 1
    sp_idx, cand_idx = compute_special_indices(n_causal, n_sink, local_window)
    scale = 1.0 / np.sqrt(head_dim)

    n_frozen = 0 if frozen is None else int(frozen[0].shape[0])
    n_new = budget - n_frozen
    if n_new <= 0:
        # Larger frozen set than requested budget: just truncate.
        return (
            frozen[0][:budget].astype(np.float32),
            frozen[1][:budget].astype(np.float32),
            frozen[2][:budget].astype(np.float32),
        )

    if len(cand_idx) == 0:
        k_new = torch.randn(n_new, d, device=device) * 0.02
        v_new = torch.randn(n_new, d, device=device) * 0.02
        logw0 = torch.full((n_new,), _ZERO_MASS_LOG, device=device)
        return _assemble(frozen, k_new, v_new, logw0)

    cand_keys = keys[cand_idx].astype(np.float32)
    cand_values = values[cand_idx].astype(np.float32)
    with_mass = frozen is None  # nested new pairs start at ~zero mass

    if init == "kmeans":
        k0, v0, logw0 = _kmeans_init(
            cand_keys, cand_values, n_new, with_mass, kmeans_subsample, rng, device,
        )
    elif init == "mqbeta":
        k0, v0, logw0 = _mqbeta_init(
            cand_keys, cand_values, n_new, with_mass, probe_queries, rng, device,
        )
    elif init == "first":
        k0, v0, logw0 = _first_init(
            cand_keys, cand_values, n_new, with_mass, device,
        )
    elif init == "random":
        k0, v0, logw0 = _random_init(
            cand_keys, cand_values, n_new, with_mass, rng, device,
        )
    elif init == "random_gauss":
        k0, v0, logw0 = _random_gauss_init(
            n_new, d, len(cand_keys), with_mass, rng, device,
        )
    else:
        raise ValueError(f"init must be 'kmeans', 'mqbeta', 'first', 'random', "
                         f"or 'random_gauss'; got {init!r}")

    k_new = nn.Parameter(k0.clone())
    v_new = nn.Parameter(v0.clone())
    log_w = nn.Parameter(logw0.clone())

    keys_ref = torch.as_tensor(keys[:n_causal], dtype=torch.float32, device=device)
    values_ref = torch.as_tensor(values[:n_causal], dtype=torch.float32, device=device)
    probes = torch.as_tensor(probe_queries, dtype=torch.float32, device=device)

    frozen_t = None
    if frozen is not None:
        frozen_t = (
            torch.as_tensor(frozen[0], dtype=torch.float32, device=device),
            torch.as_tensor(frozen[1], dtype=torch.float32, device=device),
            torch.as_tensor(frozen[2], dtype=torch.float32, device=device),
        )

    with torch.no_grad():
        consts = _precompute_targets(
            probes, keys_ref, values_ref, sp_idx, frozen_t, scale,
        )
    del keys_ref, values_ref

    m = probes.shape[0]
    perm = rng.permutation(m)
    n_val = max(1, int(m * val_fraction)) if m >= 10 else 0
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long, device=device)
    train_idx = torch.as_tensor(perm[n_val:], dtype=torch.long, device=device)
    if train_idx.numel() == 0:
        train_idx = torch.arange(m, device=device)

    optimizer = torch.optim.Adam([k_new, v_new, log_w], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(int(lr_decay_step), 1), gamma=lr_decay_gamma,
    )
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    best_val = float("inf")
    best = (k_new.detach().clone(), v_new.detach().clone(), log_w.detach().clone())
    stale = 0
    history = {"train_loss": [], "val_loss": []} if return_history else None

    for step_i in range(max(int(n_steps), 0)):
        if train_idx.numel() <= batch_size:
            batch = train_idx
        else:
            sel = torch.randperm(
                train_idx.numel(), generator=torch_gen, device=device,
            )[:batch_size]
            batch = train_idx[sel]

        pred, target = _forward_pred(
            consts, batch, k_new, v_new, log_w, scale, exact_denominator,
        )
        loss_t = _loss_value(pred, target, loss, rel_l2_floor)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        scheduler.step()

        if history is not None:
            history["train_loss"].append(float(loss_t.detach()))

        if val_idx.numel() > 0:
            with torch.no_grad():
                vp, vt = _forward_pred(
                    consts, val_idx, k_new, v_new, log_w, scale, exact_denominator,
                )
                val_loss = float(_loss_value(vp, vt, loss, rel_l2_floor))
            if history is not None:
                history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best = (
                    k_new.detach().clone(),
                    v_new.detach().clone(),
                    log_w.detach().clone(),
                )
                stale = 0
            else:
                stale += 1
                if stale >= early_stop_patience:
                    break

    if val_idx.numel() > 0:
        k_new, v_new, log_w = best

    result = _assemble(frozen, k_new, v_new, log_w)
    if return_history:
        return result + (history,)
    return result


def _assemble(
    frozen: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    log_w: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = k_new.detach().cpu().numpy().astype(np.float32)
    v = v_new.detach().cpu().numpy().astype(np.float32)
    w = log_w.detach().exp().cpu().numpy().astype(np.float32)
    if frozen is None:
        return k, v, w
    return (
        np.concatenate([frozen[0].astype(np.float32), k], axis=0),
        np.concatenate([frozen[1].astype(np.float32), v], axis=0),
        np.concatenate([frozen[2].astype(np.float32), w], axis=0),
    )


def reference_position(
    seq_len: int,
    query_positions: Optional[Sequence[int]],
) -> int:
    """Geometry reference = the evaluation (test) position."""
    if query_positions:
        return int(max(query_positions))
    return seq_len - 1


def build_probe_queries(
    queries: np.ndarray,
    query_positions: Optional[Sequence[int]],
    ref_pos: int,
    n_train_queries: int,
) -> np.ndarray:
    """Context queries nearest the reference position (excluding test queries)."""
    test_set = set(int(p) for p in query_positions) if query_positions else set()
    context_pos = [p for p in range(ref_pos + 1) if p not in test_set]
    if n_train_queries < len(context_pos):
        context_pos = context_pos[-n_train_queries:]
    if not context_pos:
        return np.zeros((0, queries.shape[1]), dtype=np.float32)
    return queries[context_pos].astype(np.float32)
