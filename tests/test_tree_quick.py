"""Quick test of tree attention on p50 math_calc head."""
import sys; sys.path.insert(0, ".")
import numpy as np
import time, copy
from pathlib import Path
from src.core import full_attention, compute_special_indices, softmax, relative_l2_error
from src.evaluation.data_loader import load_examples
from src.algorithms.tree_attention import (
    build_tree_random_proj, save_tree, load_tree,
    remove_from_tree, tree_refine, tree_max_depth,
)


def test_head(task, layer, qh, kvh, label):
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
    cand_keys = K[cand]
    cand_vals = V[cand]

    ent = -np.sum(weights[weights > 0] * np.log(weights[weights > 0]))
    print(f"\n{'='*70}")
    print(f"  {task} {label} — {n_cand} cands, entropy={ent:.2f}")
    print(f"{'='*70}")

    t0 = time.time()
    tree = build_tree_random_proj(
        cand_keys, cand_vals, max_depth=14, seed=42,
    )
    build_time = time.time() - t0
    max_d = tree_max_depth(tree)
    print(f"  RP tree: {build_time:.1f}s, max_depth={max_d}")

    # Save for reuse
    tree_dir = Path('data/trees')
    tree_dir.mkdir(parents=True, exist_ok=True)
    tree_path = tree_dir / f"tree_rp_{task}_L{layer}H{qh}_n{n_cand}.pkl"
    save_tree(tree, tree_path)

    print(f"\n  {'budget':>6s}  {'sd=2':>10s}  {'sd=3':>10s}  "
          f"{'sd=4':>10s}  {'sd=5':>10s}")

    for budget in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        errors = []
        for sd in [2, 3, 4, 5]:
            tree_copy = copy.deepcopy(tree)

            b_topk = budget // 2
            b_cluster = budget - b_topk
            cand_logits = logits[cand]
            if b_topk < n_cand:
                topk_local = np.argpartition(
                    cand_logits, -b_topk,
                )[-b_topk:]
            else:
                topk_local = np.arange(min(b_topk, n_cand))
            topk_global = cand[topk_local]

            remove_from_tree(tree_copy, topk_local, cand_keys, cand_vals)

            clusters = tree_refine(
                tree_copy, q, sqrt_d,
                max(1, b_cluster),
                starting_depth=sd,
            )

            n_sp = len(sp)
            n_topk = len(topk_local)
            n_clust = len(clusters)
            n_total = n_sp + n_topk + n_clust
            scores = np.empty(n_total, dtype=np.float64)
            out_vals = np.empty((n_total, d), dtype=np.float32)
            q64 = q.astype(np.float64)

            scores[:n_sp] = logits[sp].astype(np.float64)
            out_vals[:n_sp] = V[sp]
            off = n_sp
            scores[off:off+n_topk] = logits[topk_global].astype(np.float64)
            out_vals[off:off+n_topk] = V[topk_global]
            off = n_sp + n_topk
            for i, (mk, mv, cnt) in enumerate(clusters):
                ml = float(q64 @ mk.astype(np.float64)) / sqrt_d
                scores[off+i] = ml + np.log(cnt)
                out_vals[off+i] = mv

            w = softmax(scores).astype(np.float32)
            output = w @ out_vals
            err = relative_l2_error(output, full_out)
            errors.append(err)

        print(f"  {budget:6d}  {errors[0]:10.6f}  {errors[1]:10.6f}  "
              f"{errors[2]:10.6f}  {errors[3]:10.6f}")


def main():
    heads = [
        ('math_calc', 2, 13, 3, 'p50 ent=3.58'),
        ('math_calc', 30, 13, 3, 'p75 ent=4.71'),
        ('math_calc', 31, 14, 3, 'p0 ent=0.20'),
    ]
    for task, layer, qh, kvh, label in heads:
        test_head(task, layer, qh, kvh, label)
    print("\nDone.")


if __name__ == "__main__":
    main()
