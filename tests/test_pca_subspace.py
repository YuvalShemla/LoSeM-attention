"""Quick test of PCA subspace analysis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exploration.key_sorting_stability import (
    compute_pca_projection_stability,
    create_pca_dashboard,
)


def test():
    rng = np.random.default_rng(42)
    seq_len = 4096
    head_dim = 128
    n_test = 500

    # Keys with structure: vary in 6 directions
    k_base = rng.standard_normal(head_dim)
    k_dirs = [rng.standard_normal(head_dim) for _ in range(6)]
    K = np.zeros((seq_len, head_dim), dtype=np.float32)
    for i in range(seq_len):
        K[i] = k_base + sum(
            rng.standard_normal() * d for d in k_dirs
        ) + rng.standard_normal(head_dim) * 0.01

    # Queries vary in 4 directions
    q_base = rng.standard_normal(head_dim)
    q_dirs = [rng.standard_normal(head_dim) for _ in range(4)]
    Q = np.zeros((seq_len, head_dim), dtype=np.float32)
    for i in range(seq_len):
        Q[i] = q_base + sum(
            rng.standard_normal() * d for d in q_dirs
        ) + rng.standard_normal(head_dim) * 0.01

    print("=== Q varies in ~4 dirs, K varies in ~6 dirs ===")
    pca = compute_pca_projection_stability(
        Q, K, head_dim,
        n_test_queries=n_test,
    )

    for m in pca["methods"]:
        print(f"\n  --- {m} ---")
        for d in pca["pca_dims"]:
            near_parts = []
            far_parts = []
            for pct in pca["top_pct_values"]:
                nm = np.mean(
                    pca["nearest_recall"][m][d][pct],
                )
                fm = np.mean(
                    pca["farthest_recall"][m][d][pct],
                )
                near_parts.append(f"{pct}%={nm:.3f}")
                far_parts.append(f"{pct}%={fm:.3f}")
            rho = np.mean(pca["spearman"][m][d])
            print(
                f"    d={d:3d}: near=[{' '.join(near_parts)}]"
                f"  far=[{' '.join(far_parts)}]"
                f"  ρ={rho:.3f}"
            )

    out = Path("results/synthetic_test")
    create_pca_dashboard(
        pca, "Synthetic — Q(4d) K(6d)",
        out / "pca_two_methods.png",
    )
    print(f"\n  Saved: {out / 'pca_two_methods.png'}")


if __name__ == "__main__":
    test()
