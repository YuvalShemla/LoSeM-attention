"""Synthetic test for key sorting stability analysis."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exploration.key_sorting_stability import (
    compute_all_analyses,
    create_stability_dashboard,
    create_persistence_dashboard,
    create_agreement_dashboard,
)


def test_synthetic():
    rng = np.random.default_rng(42)
    seq_len = 4096
    head_dim = 128
    n_test = 500
    n_clusters = 16  # smaller for synthetic

    mean_dir = rng.standard_normal(head_dim)
    mean_dir /= np.linalg.norm(mean_dir)

    K = (rng.standard_normal((seq_len, head_dim)) * 0.5
         + rng.uniform(0, 2, (seq_len, 1)) * mean_dir
         ).astype(np.float32)

    out = Path("results/synthetic_test")

    # ── Case 1: Similar queries ──
    Q1 = (np.tile(mean_dir, (seq_len, 1))
          + rng.standard_normal((seq_len, head_dim)) * 0.1
          ).astype(np.float32)

    print("=== Case 1: Similar Queries ===")
    g1, c1, p1, a1 = compute_all_analyses(
        Q1, K, head_dim,
        n_test_queries=n_test,
        n_query_clusters=n_clusters,
    )
    for ng in [2, 4, 8]:
        gm = np.mean(g1["overlaps"][ng])
        cm = c1["median_overlaps"][ng]
        print(f"  {ng}g: global={gm:.3f}  "
              f"clustered_med={cm:.3f}  "
              f"random={1/ng:.3f}")
    print(f"  Close pers: "
          f"{np.mean(p1['first_group_persistence']):.3f}")
    print(f"  Far pers: "
          f"{np.mean(p1['last_group_persistence']):.3f}")
    for name, j in a1["jaccards"].items():
        print(f"  Agreement {name}: {np.mean(j):.3f}")

    create_stability_dashboard(
        g1, c1, "Synthetic — Similar",
        out / "similar_stability.png",
    )
    create_persistence_dashboard(
        p1, "Synthetic — Similar",
        out / "similar_persistence.png",
    )
    create_agreement_dashboard(
        a1, "Synthetic — Similar",
        out / "similar_agreement.png",
    )

    # ── Case 2: Diverse queries ──
    dirs = [rng.standard_normal(head_dim) for _ in range(4)]
    Q2 = np.zeros((seq_len, head_dim), dtype=np.float32)
    for i in range(seq_len):
        Q2[i] = dirs[i % 4] + rng.standard_normal(head_dim) * 0.3

    print("\n=== Case 2: Diverse Queries ===")
    g2, c2, p2, a2 = compute_all_analyses(
        Q2, K, head_dim,
        n_test_queries=n_test,
        n_query_clusters=n_clusters,
    )
    for ng in [2, 4, 8]:
        gm = np.mean(g2["overlaps"][ng])
        cm = c2["median_overlaps"][ng]
        print(f"  {ng}g: global={gm:.3f}  "
              f"clustered_med={cm:.3f}  "
              f"random={1/ng:.3f}")
    print(f"  Close pers: "
          f"{np.mean(p2['first_group_persistence']):.3f}")
    print(f"  Far pers: "
          f"{np.mean(p2['last_group_persistence']):.3f}")
    for name, j in a2["jaccards"].items():
        print(f"  Agreement {name}: {np.mean(j):.3f}")

    create_stability_dashboard(
        g2, c2, "Synthetic — Diverse (4 dirs)",
        out / "diverse_stability.png",
    )
    create_persistence_dashboard(
        p2, "Synthetic — Diverse (4 dirs)",
        out / "diverse_persistence.png",
    )
    create_agreement_dashboard(
        a2, "Synthetic — Diverse (4 dirs)",
        out / "diverse_agreement.png",
    )

    print("\nDone! Check results/synthetic_test/")


if __name__ == "__main__":
    test_synthetic()
