"""Test key cluster PCA analysis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exploration.key_cluster_pca import (
    compute_key_cluster_pca,
    create_key_cluster_pca_dashboard,
)


def test():
    rng = np.random.default_rng(42)
    seq_len = 8192
    head_dim = 128

    # Keys: 4 clusters, each varying in 3 local directions
    # + small global noise.  Clustering should find these,
    # and within-cluster PCA should concentrate at ~3 dims.
    cluster_centers = [
        rng.standard_normal(head_dim) * 3
        for _ in range(4)
    ]
    cluster_dirs = [
        [rng.standard_normal(head_dim) for _ in range(3)]
        for _ in range(4)
    ]

    K = np.zeros((seq_len, head_dim), dtype=np.float32)
    K[0] = rng.standard_normal(head_dim) * 10  # sink
    for i in range(1, seq_len):
        c = i % 4
        K[i] = (
            cluster_centers[c]
            + sum(
                rng.standard_normal() * d
                for d in cluster_dirs[c]
            )
            + rng.standard_normal(head_dim) * 0.05
        )

    print("=== 4 clusters, 3 local dirs each ===")
    result = compute_key_cluster_pca(
        K, head_dim,
        n_clusters=16,  # small for speed
        n_random_trials=3,
    )

    print("\n  Cumulative variance (cluster vs random):")
    for dim in result["pca_dims"]:
        c_vals = result["cluster_cum_var"][dim]
        c_mean = np.mean(c_vals)
        r_mean = result["random_stats"][dim]["mean"]
        g_val = result["global_stats"]["cum_var"][dim]
        print(
            f"    d={dim:3d}: cluster={c_mean:.3f}  "
            f"random={r_mean:.3f}  "
            f"global={g_val:.3f}  "
            f"gain={c_mean - r_mean:+.3f}"
        )

    out = Path("results/synthetic_test")
    create_key_cluster_pca_dashboard(
        result, "Synthetic — 4 clusters × 3 dirs",
        out / "key_cluster_pca.png",
    )
    print(f"\n  Saved: {out / 'key_cluster_pca.png'}")


if __name__ == "__main__":
    test()
