"""
Value matrix rank and its effect on EWS accuracy.

For p50, p75, p100 heads on code_run, longbook_sum_eng, multi_doc_qa:
1. Load RoPE'd Q, K, V. Exclude sink. Verify stats.
2. SVD of real V (candidates only).
3. Run EWS with:
   a) Real K, real V
   b) Real K, random V (iid Gaussian unit vectors × original norms)
   c) Random K, real V
   d) Random K, random V
   Ground truth in each case = full attention with that variant's K, V.

Random V construction: for each candidate i, draw z_i ~ N(0, I_d),
normalize to unit vector, scale by ||v_i||. This gives a full-rank
(rank d=128) matrix with exactly the same per-row norms as real V,
but random directions destroying all value structure.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import compute_special_indices, softmax, relative_l2_error
from src.evaluation.data_loader import load_examples


def svd_stats(M):
    """Return (effective_rank, dims_for_90pct, dims_for_99pct)."""
    M_c = M - M.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(M_c.astype(np.float64), full_matrices=False)
    total = (S ** 2).sum()
    cumvar = np.cumsum(S ** 2) / total
    p = S ** 2 / total
    p = p[p > 1e-12]
    eff_rank = np.exp(-np.sum(p * np.log(p)))
    d90 = int(np.searchsorted(cumvar, 0.90) + 1)
    d99 = int(np.searchsorted(cumvar, 0.99) + 1)
    return eff_rank, d90, d99


def make_random_same_norms(M, rng):
    """
    Random full-rank matrix with same per-row norms as M.

    Each row: z ~ N(0,I), normalize to unit sphere, scale by ||m_i||.
    """
    n, d = M.shape
    norms = np.linalg.norm(M, axis=1, keepdims=True).astype(np.float64)
    norms = np.maximum(norms, 1e-10)
    G = rng.standard_normal((n, d)).astype(np.float64)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-10
    return (G * norms).astype(np.float32)


def run_ews(q, K, V, d, sp, cand, budget):
    """Run EWS, return (ews_output, full_output)."""
    sqrt_d = np.sqrt(d)
    q64 = q.astype(np.float64)

    # Full attention
    logits = (q64 @ K.astype(np.float64).T / sqrt_d)
    w_all = softmax(logits)
    full_out = (w_all @ V.astype(np.float64)).astype(np.float32)

    # EWS grouping on candidates
    cand_logits = logits[cand]
    cand_w = softmax(cand_logits.astype(np.float64))
    sort_order = np.argsort(cand_w)[::-1]
    sorted_cand = cand[sort_order]
    sorted_w = cand_w[sort_order]
    n_cand = len(cand)
    num_groups = min(budget, n_cand)

    # Equal-weight split
    cumsum = np.cumsum(sorted_w)
    total = cumsum[-1]
    if total < 1e-12:
        segments = [(i * (n_cand // num_groups), (i + 1) * (n_cand // num_groups))
                    for i in range(num_groups)]
    else:
        targets = np.linspace(0, total, num_groups + 1)[1:-1]
        split_idx = np.searchsorted(cumsum, targets)
        split_idx = np.clip(split_idx, 1, n_cand - 1)
        boundaries = list(dict.fromkeys(split_idx.tolist()))
        segments = []
        prev = 0
        for s in boundaries:
            if s > prev:
                segments.append((prev, s))
            prev = s
        if prev < n_cand:
            segments.append((prev, n_cand))
        while len(segments) < num_groups:
            best = max(range(len(segments)),
                       key=lambda i: segments[i][1] - segments[i][0])
            s, e = segments[best]
            if e - s < 2:
                break
            mid = (s + e) // 2
            segments[best:best + 1] = [(s, mid), (mid, e)]

    groups = [sorted_cand[s:e] for s, e in segments]

    # Build output
    n_sp = len(sp)
    n_groups = len(groups)
    n_total = n_sp + n_groups
    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float64)

    scores[:n_sp] = logits[sp]
    out_vals[:n_sp] = V[sp].astype(np.float64)

    for i, g in enumerate(groups):
        mk = K[g].astype(np.float64).mean(axis=0)
        mv = V[g].astype(np.float64).mean(axis=0)
        scores[n_sp + i] = float(q64 @ mk) / sqrt_d + np.log(len(g))
        out_vals[n_sp + i] = mv

    w_hat = softmax(scores)
    ews_out = (w_hat @ out_vals).astype(np.float32)
    return ews_out, full_out


def main():
    d = 128
    budgets = [32, 64, 128, 256, 512]
    rng = np.random.default_rng(42)

    all_heads = [
        ('code_run', 8, 24, 6, 'p50 ent=4.09'),
        ('code_run', 15, 21, 5, 'p75 ent=4.92'),
        ('code_run', 0, 22, 5, 'p100 ent=10.56'),
        ('longbook_sum_eng', 6, 10, 2, 'p50 ent=4.50'),
        ('longbook_sum_eng', 27, 6, 1, 'p75 ent=5.98'),
        ('longbook_sum_eng', 0, 22, 5, 'p100 ent=11.01'),
        ('multi_doc_qa', 15, 22, 5, 'p50 ent=4.71'),
        ('multi_doc_qa', 27, 7, 1, 'p75 ent=5.64'),
        ('multi_doc_qa', 0, 22, 5, 'p100 ent=11.03'),
    ]

    # Print rank summary table first
    print("=" * 75, flush=True)
    print("  Value matrix rank summary (candidates only, no sink)", flush=True)
    print("=" * 75, flush=True)
    print(f"  {'Task':20s} {'Head':16s} {'Eff rank':>9s} {'90% dims':>9s} {'99% dims':>9s}", flush=True)

    rank_data = {}
    for task, layer, qh, kvh, label in all_heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        V = ex['V']
        sp, cand = compute_special_indices(len(V), 1, 0)
        er, d90, d99 = svd_stats(V[cand])
        key = (task, label)
        rank_data[key] = (er, d90, d99)
        print(f"  {task:20s} {label:16s} {er:9.1f} {d90:9d} {d99:9d}", flush=True)

    # Now run EWS comparison for each head
    for task, layer, qh, kvh, label in all_heads:
        print(f"\n{'='*75}", flush=True)
        print(f"  {task} — {label}", flush=True)
        print(f"{'='*75}", flush=True)

        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K_real, V_real = ex['Q'], ex['K'], ex['V']
        q = Q[-1]
        sp, cand = compute_special_indices(len(K_real), 1, 0)

        # Attention stats
        sqrt_d = np.sqrt(d)
        logits_r = (q.astype(np.float64) @ K_real.astype(np.float64).T / sqrt_d)
        w_r = softmax(logits_r)
        ent = -np.sum(w_r[w_r > 0] * np.log(w_r[w_r > 0]))
        print(f"  n_cand={len(cand)}, entropy={ent:.2f}, top1={w_r.max():.4f}", flush=True)

        er, d90, d99 = rank_data[(task, label)]
        print(f"  V rank: eff={er:.1f}, 90%={d90}d, 99%={d99}d", flush=True)

        # Generate random alternatives (same norms, random directions)
        K_rand = make_random_same_norms(K_real, rng)
        K_rand[sp] = K_real[sp]
        V_rand = make_random_same_norms(V_real, rng)
        V_rand[sp] = V_real[sp]

        # Verify norms match
        real_v_norms = np.linalg.norm(V_real[cand], axis=1)
        rand_v_norms = np.linalg.norm(V_rand[cand], axis=1)
        assert np.allclose(real_v_norms, rand_v_norms, atol=1e-4), "V norms mismatch"

        variants = [
            ("Real K + Real V", K_real, V_real),
            ("Real K + Rand V", K_real, V_rand),
            ("Rand K + Real V", K_rand, V_real),
            ("Rand K + Rand V", K_rand, V_rand),
        ]

        header = f"  {'':22s}" + "  ".join(f"B={b:5d}" for b in budgets)
        print(header, flush=True)

        for vname, K_test, V_test in variants:
            row = f"  {vname:22s}"
            for b in budgets:
                ews_out, full_out = run_ews(q, K_test, V_test, d, sp, cand, b)
                err = relative_l2_error(ews_out, full_out)
                row += f"  {err:.6f}"
            print(row, flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
