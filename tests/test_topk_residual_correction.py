"""
TopK + equal-size group correction analysis.

For every head on code_run, longbook_sum_eng, multi_doc_qa:
1. Oracle top-K (B/2 keys) → subset attention → err_topk
2. Oracle top-K (B/2) + equal-size groups (B/2) on residual → err_topk+groups
3. Correction = err_topk - err_topk+groups
4. Correction % = correction / err_topk * 100

Compare real V vs random V (same norms, random directions).
"""
import sys; sys.path.insert(0, ".")
import json
import numpy as np
from pathlib import Path
from src.core import compute_special_indices, softmax, relative_l2_error
from src.evaluation.data_loader import load_examples


def make_random_same_norms(M, rng):
    """Random full-rank matrix with same per-row norms."""
    n, d = M.shape
    norms = np.linalg.norm(M, axis=1, keepdims=True).astype(np.float64)
    norms = np.maximum(norms, 1e-10)
    G = rng.standard_normal((n, d)).astype(np.float64)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-10
    return (G * norms).astype(np.float32)


def svd_eff_rank(M):
    """Effective rank of M (centered)."""
    M_c = M - M.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(M_c.astype(np.float64), full_matrices=False)
    total = (S ** 2).sum()
    p = S ** 2 / total
    p = p[p > 1e-12]
    return np.exp(-np.sum(p * np.log(p)))


def run_topk_only(q, K, V, d, sp, cand, b_topk):
    """Pure oracle top-K: subset attention over special + top b_topk keys."""
    sqrt_d = np.sqrt(d)
    logits = (q.astype(np.float64) @ K.astype(np.float64).T / sqrt_d)
    w_full = softmax(logits)
    full_out = (w_full @ V.astype(np.float64)).astype(np.float32)

    n_cand = len(cand)
    cand_logits = logits[cand]
    b = min(b_topk, n_cand)
    if b < n_cand:
        topk_local = np.argpartition(cand_logits, -b)[-b:]
    else:
        topk_local = np.arange(n_cand)
    topk_global = cand[topk_local]

    # Subset attention over special + topk
    sel = np.concatenate([sp, topk_global]).astype(np.int64)
    sel_logits = logits[sel]
    sel_w = softmax(sel_logits).astype(np.float32)
    topk_out = (sel_w @ V[sel].astype(np.float32))

    return topk_out, full_out, logits, topk_local, topk_global


def run_topk_plus_groups(q, K, V, d, sp, cand, logits, topk_local,
                         topk_global, b_groups):
    """Top-K + equal-size groups on residual."""
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)
    n_cand = len(cand)
    n_topk = len(topk_local)

    # Full attention ground truth
    w_full = softmax(logits)
    full_out = (w_full @ V.astype(np.float64)).astype(np.float32)

    # Remaining candidates after topK removal
    topk_set = set(topk_local.tolist())
    remaining_local = np.array([i for i in range(n_cand) if i not in topk_set])
    n_rem = len(remaining_local)

    if n_rem == 0 or b_groups <= 0:
        # No residual — just topK
        sel = np.concatenate([sp, topk_global]).astype(np.int64)
        sel_logits = logits[sel]
        sel_w = softmax(sel_logits).astype(np.float32)
        return (sel_w @ V[sel].astype(np.float32)), full_out

    # Sort remaining by logit, split into equal-size groups
    rem_global = cand[remaining_local]
    rem_logits = logits[rem_global]
    sort_order = np.argsort(rem_logits)[::-1]
    sorted_rem = rem_global[sort_order]

    n_groups = min(b_groups, n_rem)
    groups = [np.asarray(g) for g in np.array_split(sorted_rem, n_groups) if len(g) > 0]

    # Build joint softmax: special + topK + groups
    n_sp = len(sp)
    n_g = len(groups)
    n_total = n_sp + n_topk + n_g
    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float64)

    scores[:n_sp] = logits[sp]
    out_vals[:n_sp] = V[sp].astype(np.float64)

    off = n_sp
    scores[off:off + n_topk] = logits[topk_global]
    out_vals[off:off + n_topk] = V[topk_global].astype(np.float64)

    off = n_sp + n_topk
    for i, g in enumerate(groups):
        mk = K[g].astype(np.float64).mean(axis=0)
        mv = V[g].astype(np.float64).mean(axis=0)
        scores[off + i] = float(q64 @ mk) / sqrt_d + np.log(len(g))
        out_vals[off + i] = mv

    w_hat = softmax(scores)
    out = (w_hat @ out_vals).astype(np.float32)
    return out, full_out


def main():
    d = 128
    budget = 256  # total budget: 128 topK + 128 groups
    b_topk = budget // 2
    b_groups = budget - b_topk
    rng = np.random.default_rng(42)

    tasks_meta = {}
    for task in ['code_run', 'longbook_sum_eng', 'multi_doc_qa']:
        with open(f'data/vectors/{task}/metadata.json') as f:
            meta = json.load(f)
        tasks_meta[task] = meta['selected_heads']

    # Collect all heads
    all_heads = []
    for task, heads in tasks_meta.items():
        for h in heads:
            all_heads.append((
                task, h['layer'], h['q_head'], h['kv_head'],
                h['selection_label'], h['effective_entropy'],
            ))

    print(f"Budget = {budget} (topK={b_topk} + groups={b_groups})", flush=True)
    print(f"Random V: iid Gaussian directions, same per-row norms as real V\n", flush=True)

    # Header
    print(f"{'Task':18s} {'Head':6s} {'Ent':>5s} {'V rank':>6s} "
          f"| {'TopK err':>9s} {'T+G err':>9s} {'Corr':>9s} {'Corr%':>6s} "
          f"| {'TopK(rV)':>9s} {'T+G(rV)':>9s} {'Corr(rV)':>9s} {'Corr%':>6s}",
          flush=True)
    print("-" * 125, flush=True)

    for task, layer, qh, kvh, label, ent_meta in all_heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K_real, V_real = ex['Q'], ex['K'], ex['V']
        q = Q[-1]
        sp, cand = compute_special_indices(len(K_real), 1, 0)

        # V rank
        vrank = svd_eff_rank(V_real[cand])

        # Random V
        V_rand = make_random_same_norms(V_real, rng)
        V_rand[sp] = V_real[sp]

        # === Real V ===
        topk_out_r, full_r, logits_r, topk_local, topk_global = \
            run_topk_only(q, K_real, V_real, d, sp, cand, b_topk)
        err_topk_r = relative_l2_error(topk_out_r, full_r)

        tg_out_r, _ = run_topk_plus_groups(
            q, K_real, V_real, d, sp, cand, logits_r,
            topk_local, topk_global, b_groups)
        err_tg_r = relative_l2_error(tg_out_r, full_r)

        corr_r = err_topk_r - err_tg_r
        corr_pct_r = (corr_r / err_topk_r * 100) if err_topk_r > 1e-10 else 0.0

        # === Random V (same keys, same attention, different values) ===
        topk_out_rv, full_rv, logits_rv, topk_local_rv, topk_global_rv = \
            run_topk_only(q, K_real, V_rand, d, sp, cand, b_topk)
        err_topk_rv = relative_l2_error(topk_out_rv, full_rv)

        tg_out_rv, _ = run_topk_plus_groups(
            q, K_real, V_rand, d, sp, cand, logits_rv,
            topk_local_rv, topk_global_rv, b_groups)
        err_tg_rv = relative_l2_error(tg_out_rv, full_rv)

        corr_rv = err_topk_rv - err_tg_rv
        corr_pct_rv = (corr_rv / err_topk_rv * 100) if err_topk_rv > 1e-10 else 0.0

        short_label = label.replace('_lowest', '').replace('_highest', '').replace('_median', '')
        print(f"{task:18s} {short_label:6s} {ent_meta:5.1f} {vrank:6.1f} "
              f"| {err_topk_r:9.6f} {err_tg_r:9.6f} {corr_r:9.6f} {corr_pct_r:5.1f}% "
              f"| {err_topk_rv:9.6f} {err_tg_rv:9.6f} {corr_rv:9.6f} {corr_pct_rv:5.1f}%",
              flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
