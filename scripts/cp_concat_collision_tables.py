"""
Generate CP-concatenation collision probability tables.

Instead of product-of-two-CPs, this uses concatenation:
  - m random rotations R_1, ..., R_m (each d×d)
  - Concatenate: z = [R_1 x, R_2 x, ..., R_m x]  (m*d dims)
  - Label = argmax(|z|) with sign → 2*m*d buckets

Bucket count C = 2*m*d matches SimHash's 2^K:
  m=1  → C=256  (SimHash K=8)
  m=2  → C=512  (SimHash K=9)
  m=4  → C=1024 (SimHash K=10)
  m=8  → C=2048 (SimHash K=11)
  m=16 → C=4096 (SimHash K=12)

Uses the Gram-Schmidt trick: each rotation only needs
2 orthonormal columns (one for query, one for the key's
perpendicular component at each angle).

Usage:
  python3 -m scripts.cp_concat_collision_tables
"""

import numpy as np
import time
import os


def random_orthonormal_pair(d, rng):
    v1 = rng.standard_normal(d).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(d).astype(np.float32)
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)
    return v1, v2


def generate_table(m, d, N, cos_bins, rng):
    """Generate collision table for m-rotation concatenation CP."""
    n_bins = len(cos_bins)
    cos_f32 = cos_bins.astype(np.float32)
    theta_vals = np.arccos(np.clip(cos_f32, -1 + 1e-7, 1 - 1e-7))
    sin_f32 = np.sin(theta_vals).astype(np.float32)

    C = 2 * m * d
    collisions = np.zeros(n_bins, dtype=np.int64)

    t0 = time.time()
    for i in range(N):
        if i % 500_000 == 0:
            el = time.time() - t0
            rate = i / max(el, 0.01)
            eta = (N - i) / max(rate, 0.01) if rate > 0 else 0
            print(f"    {i:>9,}/{N:,} "
                  f"({el:.0f}s, ~{eta:.0f}s left)", flush=True)

        # Generate m orthonormal pairs
        pairs = [random_orthonormal_pair(d, rng) for _ in range(m)]

        # Query: e_1 maps to col0 of each rotation
        # Concatenated query vector: [v_11, v_21, ..., v_m1]
        q_concat = np.concatenate([p[0] for p in pairs])  # [m*d]
        qi = int(np.argmax(np.abs(q_concat)))
        q_label = qi * 2 + (1 if q_concat[qi] < 0 else 0)

        # Keys at each angle: cos*col0 + sin*col1 for each rotation
        # Build [n_bins, m*d] key projections
        parts = []
        for v1, v2 in pairs:
            # [n_bins, d] = cos[:, None] * v1[None, :] + sin[:, None] * v2[None, :]
            parts.append(
                np.outer(cos_f32, v1) + np.outer(sin_f32, v2)
            )
        k_concat = np.concatenate(parts, axis=1)  # [n_bins, m*d]

        idx = np.argmax(np.abs(k_concat), axis=1)  # [n_bins]
        signs = k_concat[np.arange(n_bins), idx]
        k_labels = idx * 2 + (signs < 0).astype(np.int32)

        collisions += (k_labels == q_label)

    elapsed = time.time() - t0
    return collisions, elapsed


def main():
    d = 128
    N = 5_000_000
    cos_bins = np.arange(-1.000, 1.001, 0.001).astype(np.float64)

    m_values = [1, 2, 4, 8, 16]
    rng = np.random.default_rng(42)

    os.makedirs("data", exist_ok=True)

    for m in m_values:
        C = 2 * m * d
        print(f"\n{'='*50}", flush=True)
        print(f"m={m} rotations, C={C} buckets "
              f"(matches SimHash K={int(np.log2(C))})",
              flush=True)
        print(f"{'='*50}", flush=True)

        collisions, elapsed = generate_table(
            m, d, N, cos_bins, rng,
        )
        p_table = collisions / float(N)

        fname = f"data/cp_concat_collision_table_m{m}.npz"
        np.savez_compressed(
            fname,
            cos_bins=cos_bins,
            p_table=p_table,
            n_tables=N,
            n_rotations=m,
            head_dim=d,
            bucket_count=C,
            collisions=collisions,
        )
        print(f"  Saved to {fname} ({elapsed/60:.1f} min)",
              flush=True)

        for c in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
            idx = np.argmin(np.abs(cos_bins - c))
            print(f"    cos={c:+.2f}: p={p_table[idx]:.8f} "
                  f"({collisions[idx]:,} coll)", flush=True)


if __name__ == "__main__":
    main()
