"""
Analyze the rank structure of value matrices and its effect on
IdealEqualWeightSplits accuracy.

1. SVD spectrum of V for each head — how low-rank is it?
2. EWS error with real V vs random V vs adversarial V
3. The adversarial V: for each EWS group, assign orthogonal
   random values so within-group mean is near zero.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import full_attention, compute_special_indices, softmax, relative_l2_error
from src.evaluation.data_loader import load_examples


def svd_analysis(V, label):
    """Analyze singular value spectrum of V."""
    # Center for proper rank analysis
    V_centered = V - V.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(V_centered.astype(np.float64), full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = np.cumsum(S ** 2) / total_var

    print(f"\n  SVD of V ({label}):")
    print(f"    Shape: {V.shape}")
    print(f"    Top-10 singular values: {S[:10].round(1)}")
    print(f"    Variance captured: "
          f"1d={cumvar[0]:.3f}, 5d={cumvar[4]:.3f}, "
          f"10d={cumvar[9]:.3f}, 20d={cumvar[19]:.3f}, "
          f"50d={cumvar[49]:.3f}, 128d={cumvar[min(127, len(cumvar)-1)]:.3f}")

    # Effective rank (Shannon entropy of normalized singular values)
    p = S ** 2 / total_var
    p = p[p > 1e-12]
    eff_rank = np.exp(-np.sum(p * np.log(p)))
    print(f"    Effective rank (exp entropy): {eff_rank:.1f}")

    # How many dims for 90%, 95%, 99%?
    for thresh in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cumvar, thresh) + 1
        print(f"    Dims for {thresh*100:.0f}% variance: {k}")

    return S, cumvar


def ews_groups(cand_logits, cand_weights, n_groups):
    """Build equal-weight groups (same logic as IdealEqualWeightSplits)."""
    sort_order = np.argsort(cand_weights)[::-1]
    sorted_weights = cand_weights[sort_order]
    n = len(sort_order)

    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    if total < 1e-12:
        # Fallback to equal-size
        return [sort_order[s:e] for s, e in
                zip(range(0, n, n // n_groups), range(n // n_groups, n + 1, n // n_groups))]

    targets = np.linspace(0, total, n_groups + 1)[1:-1]
    split_indices = np.searchsorted(cumsum, targets)
    split_indices = np.clip(split_indices, 1, n - 1)
    boundaries = list(dict.fromkeys(split_indices.tolist()))

    segments = []
    prev = 0
    for sp in boundaries:
        if sp > prev:
            segments.append((prev, sp))
        prev = sp
    if prev < n:
        segments.append((prev, n))

    while len(segments) < n_groups:
        best = max(range(len(segments)),
                   key=lambda i: segments[i][1] - segments[i][0])
        s, e = segments[best]
        if e - s < 2:
            break
        mid = (s + e) // 2
        segments[best:best + 1] = [(s, mid), (mid, e)]

    # Convert to global indices
    groups = [sort_order[s:e] for s, e in segments]
    return groups


def run_ews_with_values(q, K, V_test, logits, sp, cand, d, budget):
    """Run EWS with arbitrary value matrix."""
    sqrt_d = np.sqrt(d)
    cand_logits = logits[cand]
    cand_weights = softmax(cand_logits)

    groups = ews_groups(cand_logits, cand_weights, budget)

    n_sp = len(sp)
    n_groups = len(groups)
    n_total = n_sp + n_groups
    scores = np.empty(n_total, dtype=np.float64)
    out_vals = np.empty((n_total, d), dtype=np.float32)
    q64 = q.astype(np.float64)

    scores[:n_sp] = logits[sp].astype(np.float64)
    out_vals[:n_sp] = V_test[sp]

    for i, g in enumerate(groups):
        global_g = cand[g]
        mk = K[global_g].astype(np.float64).mean(axis=0)
        mv = V_test[global_g].astype(np.float64).mean(axis=0)
        cnt = len(g)
        scores[n_sp + i] = float(q64 @ mk) / sqrt_d + np.log(cnt)
        out_vals[n_sp + i] = mv.astype(np.float32)

    w = softmax(scores).astype(np.float32)
    return w @ out_vals.astype(np.float32)


def main():
    heads = [
        ('math_calc', 2, 13, 3, 'p50 ent=3.58'),
        ('math_calc', 30, 13, 3, 'p75 ent=4.71'),
        ('math_calc', 0, 22, 5, 'p100 ent=10.19'),
    ]

    rng = np.random.default_rng(42)
    budgets = [32, 64, 128, 256]

    for task, layer, qh, kvh, label in heads:
        ex = list(load_examples(
            Path('data/vectors'), task,
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q, K, V = ex['Q'], ex['K'], ex['V']
        d = 128; q = Q[-1]
        full_out_real, logits, weights = full_attention(q, K, V, d)
        sp, cand = compute_special_indices(len(K), 1, 0)
        n = len(K)

        print(f"\n{'='*70}")
        print(f"  {label} — {len(cand)} candidates")
        print(f"{'='*70}", flush=True)

        # 1. SVD analysis of real V
        S_real, cumvar_real = svd_analysis(V, label)

        # 2. Analyze within-group value similarity for EWS
        cand_logits = logits[cand]
        cand_weights = softmax(cand_logits)
        groups_64 = ews_groups(cand_logits, cand_weights, 64)
        cos_sims = []
        for g in groups_64:
            if len(g) < 2:
                cos_sims.append(1.0)
                continue
            global_g = cand[g]
            gv = V[global_g].astype(np.float64)
            mv = gv.mean(axis=0)
            mv_norm = np.linalg.norm(mv)
            if mv_norm < 1e-10:
                cos_sims.append(0.0)
                continue
            sims = (gv @ mv) / (np.linalg.norm(gv, axis=1) * mv_norm + 1e-10)
            cos_sims.append(sims.mean())
        cos_arr = np.array(cos_sims)
        print(f"\n  EWS 64-group value cosine-to-mean:")
        print(f"    avg={cos_arr.mean():.4f}, min={cos_arr.min():.4f}, "
              f"p25={np.percentile(cos_arr, 25):.4f}, "
              f"median={np.median(cos_arr):.4f}")

        # 3. Generate alternative V matrices
        # Random full-rank V: iid Gaussian, same norm distribution as real V
        norms_real = np.linalg.norm(V, axis=1, keepdims=True)
        V_random = rng.standard_normal(V.shape).astype(np.float32)
        V_random *= norms_real / (np.linalg.norm(V_random, axis=1, keepdims=True) + 1e-10)

        # Adversarial V: within each EWS group, assign random orthogonal
        # directions so the group mean is near zero
        V_adv = V.copy()
        groups_for_adv = ews_groups(cand_logits, cand_weights, 64)
        for g in groups_for_adv:
            if len(g) < 2:
                continue
            global_g = cand[g]
            n_g = len(g)
            rand_dirs = rng.standard_normal((n_g, d)).astype(np.float32)
            rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-10
            orig_norms = norms_real[global_g].flatten()
            V_adv[global_g] = rand_dirs * orig_norms[:, None]

        # Shuffled V: same values but randomly permuted across positions
        perm = rng.permutation(n)
        V_shuffled = V[perm]

        # 4. Compute full attention outputs for each V variant
        full_out_random = (softmax(logits) @ V_random.astype(np.float64)).astype(np.float32)
        full_out_adv = (softmax(logits) @ V_adv.astype(np.float64)).astype(np.float32)
        full_out_shuf = (softmax(logits) @ V_shuffled.astype(np.float64)).astype(np.float32)

        # 5. Run EWS with each V
        print(f"\n  EWS error comparison:", flush=True)
        header = f"  {'':15s}" + "  ".join(f"B={b:4d}" for b in budgets)
        print(header, flush=True)

        for v_name, V_test, full_ref in [
            ("Real V", V, full_out_real),
            ("Random V", V_random, full_out_random),
            ("Adversarial V", V_adv, full_out_adv),
            ("Shuffled V", V_shuffled, full_out_shuf),
        ]:
            row = f"  {v_name:15s}"
            for b in budgets:
                ews_out = run_ews_with_values(q, K, V_test, logits, sp, cand, d, b)
                err = relative_l2_error(ews_out, full_ref)
                row += f"  {err:.6f}"
            print(row, flush=True)

        # 6. SVD of random V for comparison
        svd_analysis(V_random, "Random V")

    print("\n\nDone.", flush=True)


if __name__ == "__main__":
    main()
