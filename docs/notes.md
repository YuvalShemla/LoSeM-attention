# Research Notes

Running doc for observations, open questions, and directions that come up during experiments. Revisit periodically.

---

## Directions for Improvement

### 1. RoPE vs No-RoPE Representations

We observed that clustering is significantly easier on pre-RoPE (raw) vectors. RoPE rotates keys into position-dependent subspaces, which spreads them out and hurts nearest-neighbor / hashing-based methods. However, the worst-case distribution under RoPE might actually be more amenable to LSH — the rotation could bound the gap between best-case and worst-case hash-bucket collisions. Needs further analysis: compare LSH recall on raw vs RoPE vectors across layers and sequence lengths.

### 2. Out-of-Distribution: Q and K Live in Different Subspaces

Queries and keys are produced by different linear projections (W_Q, W_K), so they occupy different regions of the embedding space. Standard similarity-search structures (LSH, k-means, etc.) assume a single distribution. We need to think about how to handle cross-subspace retrieval — possible directions include joint indexing, learned projections that align Q/K spaces, or asymmetric hashing schemes.

### 3. Special Tokens: Sink and Local Window

The head statistics confirm that sink tokens (position 0) and the local window (last ~1024 keys) capture a large fraction of attention mass across most heads and layers. This is an advantage: if we always include these tokens, the remaining "long-range" mass is what we need to approximate. A good approximation of just the non-sink, non-local attention could compose with the exact special-token attention to give strong overall accuracy. The key question is how much mass remains in the long-range portion and how concentrated it is — the `entropy_no_sink_local` and `top1pct_mass_no_sink_local` statistics give us a direct read on this.
