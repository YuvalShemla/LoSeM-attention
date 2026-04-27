"""
Analyze IdealEqualWeightSplits to understand what makes it effective.

For each head, examine:
1. Group size distribution at various budgets
2. Weight per group (should be ~equal by design)
3. Within-group key similarity (cosine sim to group mean)
4. Within-group value similarity
5. Score accuracy (mean-key score vs true mean logit)
6. Error decomposition: which groups contribute most error?
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from src.core import full_attention, compute_special_indices, softmax, relative_l2_error
from src.evaluation.data_loader import load_examples


def analyze_head(task, layer, qh, kvh, label):
    ex = list(load_examples(
        Path('data/vectors'), task,
        layer=layer, head=qh, kv_head=kvh,
        phase=None, max_examples=1, use_rope=True,
    ))[0]
    Q, K, V = ex['Q'], ex['K'], ex['V']
    d = 128
    q = Q[-1]
    full_out, logits, weights = full_attention(q, K, V, d)
    sp, cand = compute_special_indices(len(K), 1, 0)
    sqrt_d = np.sqrt(d)
    n_cand = len(cand)

    cand_logits = logits[cand]
    cand_weights = softmax(cand_logits)
    sort_order = np.argsort(cand_weights)[::-1]
    sorted_cand = cand[sort_order]
    sorted_weights = cand_weights[sort_order]

    print(f"\n{'='*70}")
    print(f"  {task} {label} — {n_cand} candidates")
    print(f"  Entropy = {-np.sum(weights[weights>0]*np.log(weights[weights>0])):.2f} nats")
    print(f"  Top-1 weight = {weights.max():.4f}")
    print(f"  Top-10 mass = {np.sort(weights)[-10:].sum():.4f}")
    print(f"  Top-100 mass = {np.sort(weights)[-100:].sum():.4f}")
    print(f"{'='*70}")

    for budget in [32, 64, 128, 256]:
        num_groups = min(budget, n_cand)
        # Build equal-weight groups
        cumsum = np.cumsum(sorted_weights)
        total = cumsum[-1]
        targets = np.linspace(0, total, num_groups + 1)[1:-1]
        split_indices = np.searchsorted(cumsum, targets)
        split_indices = np.clip(split_indices, 1, n_cand - 1)
        boundaries = list(dict.fromkeys(split_indices.tolist()))
        segments = []
        prev = 0
        for sp_idx in boundaries:
            if sp_idx > prev:
                segments.append((prev, sp_idx))
            prev = sp_idx
        if prev < n_cand:
            segments.append((prev, n_cand))
        # Subdivide if needed
        while len(segments) < num_groups:
            best = max(range(len(segments)),
                       key=lambda i: segments[i][1] - segments[i][0])
            s, e = segments[best]
            if e - s < 2:
                break
            mid = (s + e) // 2
            segments[best:best+1] = [(s, mid), (mid, e)]

        groups = [sorted_cand[s:e] for s, e in segments]
        sizes = np.array([len(g) for g in groups])
        group_weights = np.array([sorted_weights[s:e].sum() for s, e in segments])

        # Compute error for this budget
        n_sp = len(sp) if isinstance(sp, np.ndarray) else 1
        n_groups = len(groups)
        n_total = n_sp + n_groups
        scores = np.empty(n_total)
        out_vals = np.empty((n_total, d))
        sp_arr = np.array([0]) if not isinstance(sp, np.ndarray) else sp
        scores[:n_sp] = logits[sp_arr]
        out_vals[:n_sp] = V[sp_arr]

        q64 = q.astype(np.float64)
        for i, g in enumerate(groups):
            mk = K[g].astype(np.float64).mean(axis=0)
            mv = V[g].astype(np.float64).mean(axis=0)
            cnt = len(g)
            scores[n_sp + i] = float(q64 @ mk) / sqrt_d + np.log(cnt)
            out_vals[n_sp + i] = mv.astype(np.float32)

        w = softmax(scores).astype(np.float32)
        output = w @ out_vals.astype(np.float32)
        err = relative_l2_error(output, full_out)

        # Group statistics
        print(f"\n  Budget={budget}: {n_groups} groups, error={err:.6f}")
        print(f"    Group sizes: min={sizes.min()}, p25={np.percentile(sizes, 25):.0f}, "
              f"median={np.median(sizes):.0f}, p75={np.percentile(sizes, 75):.0f}, max={sizes.max()}")
        print(f"    Group weight: min={group_weights.min():.6f}, max={group_weights.max():.6f}, "
              f"std/mean={group_weights.std()/group_weights.mean():.3f}")

        # Top groups by weight
        top5_w = np.argsort(group_weights)[-5:][::-1]
        print(f"    Top-5 groups by weight:")
        for j in top5_w:
            print(f"      group {j}: size={sizes[j]}, weight={group_weights[j]:.6f}, "
                  f"rank range=[{sum(sizes[:j])}-{sum(sizes[:j+1])-1}]")

        # Last 5 groups (tail — largest groups with least weight)
        tail5 = np.argsort(sizes)[-5:][::-1]
        print(f"    Largest-5 groups (tail):")
        for j in tail5:
            print(f"      group {j}: size={sizes[j]}, weight={group_weights[j]:.6f}")

        # Within-group key cosine similarity to mean
        if budget <= 128:
            cos_sims = []
            for g in groups:
                if len(g) == 1:
                    cos_sims.append(1.0)
                    continue
                gk = K[g].astype(np.float64)
                mk = gk.mean(axis=0)
                mk_norm = np.linalg.norm(mk)
                if mk_norm < 1e-10:
                    cos_sims.append(0.0)
                    continue
                sims = (gk @ mk) / (np.linalg.norm(gk, axis=1) * mk_norm + 1e-10)
                cos_sims.append(sims.mean())
            cos_arr = np.array(cos_sims)
            print(f"    Key cosine-to-mean: avg={cos_arr.mean():.4f}, "
                  f"min={cos_arr.min():.4f}, max={cos_arr.max():.4f}")


def main():
    heads = [
        # math_calc
        ('math_calc', 31, 14, 3, 'p0 ent=0.20'),
        ('math_calc', 25, 1, 0, 'p25 ent=2.88'),
        ('math_calc', 2, 13, 3, 'p50 ent=3.58'),
        ('math_calc', 30, 13, 3, 'p75 ent=4.71'),
        ('math_calc', 0, 22, 5, 'p100 ent=10.19'),
    ]

    for task, layer, qh, kvh, label in heads:
        analyze_head(task, layer, qh, kvh, label)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
