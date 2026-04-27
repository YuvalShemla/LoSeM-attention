"""End-to-end test of TreeAttention with pre-built flat trees."""
import sys; sys.path.insert(0, ".")
import traceback
import numpy as np
from pathlib import Path

try:
    from src.core import full_attention, compute_special_indices, relative_l2_error
    from src.evaluation.data_loader import load_examples
    from src.algorithms.tree_attention import TreeAttention
    from src.algorithms.base import AttentionInput
    print("Imports OK", flush=True)

    ex = list(load_examples(
        Path('data/vectors'), 'math_calc',
        layer=2, head=13, kv_head=3,
        phase=None, max_examples=1, use_rope=True,
    ))[0]
    Q, K, V = ex['Q'], ex['K'], ex['V']
    d = 128; q = Q[-1]
    full_out, logits, weights = full_attention(q, K, V, d)
    sp, cand = compute_special_indices(len(K), 1, 0)
    print(f"Data loaded: n_cand={len(cand)}")

    problem = AttentionInput(
        query=q, keys=K, values=V, head_dim=d,
        logits=logits, special_idx=sp, candidate_idx=cand,
    )
    rng = np.random.default_rng(42)

    algo = TreeAttention(starting_depth=3, tree_dir='data/trees')
    algo.prepare(K, V, d, seed=42)
    print("Algorithm prepared")

    # Test single budget first
    print("Running budget=128...")
    out = algo.run(problem, 128, rng)
    err = relative_l2_error(out.output, full_out)
    print(f"  B=128: err={err:.6f} actual_budget={out.actual_budget}")

    # Now sweep
    print("\nFull sweep:")
    for sd in [2, 3, 4, 5]:
        algo = TreeAttention(starting_depth=sd, tree_dir='data/trees')
        algo.prepare(K, V, d, seed=42)
        errors = []
        for budget in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
            out = algo.run(problem, budget, rng)
            err = relative_l2_error(out.output, full_out)
            errors.append(f"{err:.6f}")
        print(f"  SD={sd}: {' '.join(errors)}")

    print("\nDone.")

except Exception:
    traceback.print_exc()
