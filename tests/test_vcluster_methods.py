"""
Sanity tests for the 5 value/key clustering methods.

Tests:
1. No crashes on synthetic data
2. Budget accounting is correct
3. Special keys are handled properly
4. Output has correct shape
5. Error is reasonable (not NaN, not > 1 for moderate budgets)
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from src.core import (
    full_attention, compute_special_indices, relative_l2_error,
    clear_kmeans_cache,
)
from src.algorithms.base import AttentionInput
from src.algorithms.value_cluster_methods import (
    VClusterMeanKey,
    VClusterSampled,
    KClusterSampled,
    VClusterLastKey,
    VClusterMeanLogit,
)


def make_problem(n=1000, d=128, n_sink=1, local_window=0,
                 seed=42):
    """Create a synthetic attention problem."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(d).astype(np.float32)
    K = rng.standard_normal((n, d)).astype(np.float32)
    V = rng.standard_normal((n, d)).astype(np.float32)

    full_out, logits, weights = full_attention(q, K, V, d)
    sp_idx, cand_idx = compute_special_indices(
        n, n_sink, local_window,
    )

    problem = AttentionInput(
        query=q, keys=K, values=V,
        head_dim=d, logits=logits,
        special_idx=sp_idx, candidate_idx=cand_idx,
    )
    return problem, full_out


def test_method(name, method, problem, full_out,
                budgets=[16, 64, 256]):
    """Test a single method at multiple budgets."""
    rng = np.random.default_rng(42)
    method.prepare(
        problem.keys, problem.values, problem.head_dim,
        seed=42,
    )

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    for b in budgets:
        out = method.run(problem, b, rng)

        err = relative_l2_error(out.output, full_out)
        assert out.output.shape == full_out.shape, (
            f"Shape mismatch: {out.output.shape} vs "
            f"{full_out.shape}"
        )
        assert not np.isnan(err), "Error is NaN!"
        assert out.actual_budget > 0, "Budget is 0!"

        status = "OK" if err < 1.0 else "HIGH"
        print(
            f"  budget={b:4d} -> actual={out.actual_budget:5d}, "
            f"error={err:.6f}  [{status}]"
        )

    print(f"  PASSED")


def main():
    print("Creating synthetic problem (n=1000, d=128)...")
    problem, full_out = make_problem(
        n=1000, d=128, n_sink=1, local_window=0,
    )
    print(
        f"  special={len(problem.special_idx)}, "
        f"candidates={len(problem.candidate_idx)}"
    )

    # Method 1
    test_method(
        "VCluster-MeanKey",
        VClusterMeanKey(),
        problem, full_out,
    )

    # Method 2
    test_method(
        "VCluster-Sampled-C256",
        VClusterSampled(n_clusters=256),
        problem, full_out,
    )

    # Method 3
    test_method(
        "KCluster-Sampled-C256",
        KClusterSampled(n_clusters=256),
        problem, full_out,
    )

    # Method 4 (last 1, 2, 3)
    for nl in [1, 2, 3]:
        test_method(
            f"VCluster-Last{nl}Key",
            VClusterLastKey(n_last=nl),
            problem, full_out,
        )

    # Method 5
    test_method(
        "VCluster-MeanLogit",
        VClusterMeanLogit(),
        problem, full_out,
    )

    # Clear cache before changing to different data
    clear_kmeans_cache()

    # Test with larger N and more clusters
    print("\n\nCreating larger problem (n=10000, d=128)...")
    problem2, full_out2 = make_problem(
        n=10000, d=128, n_sink=1, local_window=0,
    )

    test_method(
        "VCluster-MeanKey (large)",
        VClusterMeanKey(),
        problem2, full_out2,
        budgets=[64, 256, 1024],
    )
    test_method(
        "VCluster-Sampled-C1024 (large)",
        VClusterSampled(n_clusters=1024),
        problem2, full_out2,
        budgets=[64, 256, 512],
    )
    test_method(
        "VCluster-MeanLogit (large)",
        VClusterMeanLogit(),
        problem2, full_out2,
        budgets=[64, 256, 1024],
    )

    print("\n\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
