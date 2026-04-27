"""Test tree attention across budgets and starting depths, all math_calc heads."""
import sys; sys.path.insert(0, ".")
import numpy as np
import time
from pathlib import Path
from src.core import full_attention, compute_special_indices, relative_l2_error
from src.evaluation.data_loader import load_examples
from src.algorithms.tree_attention import TreeAttention
from src.algorithms.idealized_methods import IdealTopK, IdealEqualWeightSplits
from src.algorithms.base import AttentionInput

budgets = [32, 64, 128, 256, 512, 1024, 2048, 4096]

heads = [
    ('math_calc', 31, 14, 3, 'p0 ent=0.20'),
    ('math_calc', 25, 1, 0, 'p25 ent=2.88'),
    ('math_calc', 2, 13, 3, 'p50 ent=3.58'),
    ('math_calc', 30, 13, 3, 'p75 ent=4.71'),
    ('math_calc', 0, 22, 5, 'p100 ent=10.19'),
]

for task, layer, qh, kvh, label in heads:
    ex = list(load_examples(
        Path('data/vectors'), task,
        layer=layer, head=qh, kv_head=kvh,
        phase=None, max_examples=1, use_rope=True,
    ))[0]
    Q, K, V = ex['Q'], ex['K'], ex['V']
    d = 128; q = Q[-1]
    full_out, logits, weights = full_attention(q, K, V, d)
    sp, cand = compute_special_indices(len(K), 1, 0)
    problem = AttentionInput(
        query=q, keys=K, values=V, head_dim=d,
        logits=logits, special_idx=sp, candidate_idx=cand,
    )
    rng = np.random.default_rng(42)
    ent = -np.sum(weights[weights > 0] * np.log(weights[weights > 0]))

    print(f"\n{'='*80}", flush=True)
    print(f"  {label} — {len(cand)} candidates, entropy={ent:.2f}", flush=True)
    print(f"{'='*80}", flush=True)

    # Baselines: IdealTopK and IdealEqualWeightSplits
    ideal_topk = IdealTopK()
    ideal_ews = IdealEqualWeightSplits()
    ideal_topk.prepare(K, V, d, seed=42)
    ideal_ews.prepare(K, V, d, seed=42)

    row_topk = "TopK   "
    row_ews = "EWS    "
    for b in budgets:
        out_t = ideal_topk.run(problem, b, rng)
        err_t = relative_l2_error(out_t.output, full_out)
        row_topk += f"  {err_t:.6f}"
        out_e = ideal_ews.run(problem, b, rng)
        err_e = relative_l2_error(out_e.output, full_out)
        row_ews += f"  {err_e:.6f}"

    header = "       " + "  ".join(f"B={b:7d}" for b in budgets)
    print(header, flush=True)
    print(row_topk, flush=True)
    print(row_ews, flush=True)

    # Tree at different starting depths
    for sd in [3, 4]:
        algo = TreeAttention(starting_depth=sd, tree_dir='data/trees')
        algo.prepare(K, V, d, seed=42)
        row = f"Tr-D{sd}  "
        for b in budgets:
            out = algo.run(problem, b, rng)
            err = relative_l2_error(out.output, full_out)
            row += f"  {err:.6f}"
        print(row, flush=True)

print("\nDone.", flush=True)
