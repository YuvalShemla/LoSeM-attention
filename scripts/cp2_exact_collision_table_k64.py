"""
Exact CP2 collision probability table for k_dim=64.

Same method as k=128: synthetic unit vectors, 5M rotation
pairs, 0.001 cosine-similarity resolution. For k=64 each
CP has 2×64=128 buckets, CP2 has 128²=16384 — coarser
partition means higher collision rates.

Usage:
  python3 -m scripts.cp2_exact_collision_table_k64
"""

import numpy as np
import time
import sys


def random_orthonormal_pair(k, rng):
    v1 = rng.standard_normal(k).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(k).astype(np.float32)
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)
    return v1, v2


def main():
    k_dim = 64
    N_TABLES = 5_000_000
    BIN_STEP = 0.001
    cos_bins = np.arange(-0.500, 0.601, BIN_STEP).astype(np.float64)
    n_bins = len(cos_bins)

    print(f"CP2 Exact Collision Table (k_dim={k_dim})")
    print(f"  N={N_TABLES:,}, bins={n_bins} (step={BIN_STEP})")
    print(f"  Buckets per table: (2×{k_dim})² = {(2*k_dim)**2}")
    print(flush=True)

    cos_f32 = cos_bins.astype(np.float32)
    theta_vals = np.arccos(np.clip(cos_f32, -1 + 1e-7, 1 - 1e-7))
    sin_f32 = np.sin(theta_vals).astype(np.float32)

    collisions = np.zeros(n_bins, dtype=np.int64)
    rng = np.random.default_rng(42)
    n_verts = 2 * k_dim

    t0 = time.time()
    report_every = 500_000

    for i in range(N_TABLES):
        if i % report_every == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta = (N_TABLES - i) / max(rate, 0.01)
            print(f"  {i:>9,}/{N_TABLES:,}  "
                  f"({elapsed:.0f}s, ~{eta:.0f}s left)",
                  flush=True)

        a1, b1 = random_orthonormal_pair(k_dim, rng)
        a2, b2 = random_orthonormal_pair(k_dim, rng)

        qi1 = int(np.argmax(np.abs(a1)))
        ql1 = qi1 * 2 + (1 if a1[qi1] < 0 else 0)
        qi2 = int(np.argmax(np.abs(a2)))
        ql2 = qi2 * 2 + (1 if a2[qi2] < 0 else 0)
        q_label = ql1 * n_verts + ql2

        p1 = np.outer(cos_f32, a1) + np.outer(sin_f32, b1)
        p2 = np.outer(cos_f32, a2) + np.outer(sin_f32, b2)

        idx1 = np.argmax(np.abs(p1), axis=1)
        s1 = (p1[np.arange(n_bins), idx1] < 0).astype(np.int32)
        idx2 = np.argmax(np.abs(p2), axis=1)
        s2 = (p2[np.arange(n_bins), idx2] < 0).astype(np.int32)
        k_label = (idx1 * 2 + s1) * n_verts + (idx2 * 2 + s2)

        collisions += (k_label == q_label)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)",
          flush=True)

    p_table = collisions / float(N_TABLES)

    out_path = f"data/cp2_collision_table_k{k_dim}.npz"
    np.savez_compressed(
        out_path,
        cos_bins=cos_bins,
        p_table=p_table,
        n_tables=N_TABLES,
        k_dim=k_dim,
        collisions=collisions,
    )
    print(f"  Saved to {out_path}")

    print(f"\n  Sample rates:")
    for c in np.arange(-0.3, 0.51, 0.05):
        idx = np.argmin(np.abs(cos_bins - c))
        nc = collisions[idx]
        p = p_table[idx]
        stderr = np.sqrt(p * (1 - p) / N_TABLES) if p > 0 else 0
        print(f"    cos={cos_bins[idx]:+.3f}: "
              f"p={p:.8f} ± {stderr:.8f}  "
              f"({nc:,} collisions)")


if __name__ == "__main__":
    main()
