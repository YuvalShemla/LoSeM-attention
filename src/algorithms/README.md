# Algorithms

Attention approximation methods. Each algorithm
implements the `AttentionAlgorithm` ABC from `base.py`.

## Problem Setup

Given a single query `q`, a causal window of keys
`K = [k_1, ..., k_N]` and values `V = [v_1, ..., v_N]`,
the exact attention output is:

    o* = sum_i softmax(q^T k_i / sqrt(d)) v_i

We approximate `o*` using fewer than N key interactions.
The evaluation runner computes `o*` (ground truth), then
calls each algorithm to produce `o_hat` and measures:

    relative L2 error = ||o_hat - o*||_2 / ||o*||_2

## Base Classes and Dataclasses

### `AttentionInput`

Everything an algorithm receives for one query position.
Created by the evaluator before calling `run()`.

```python
@dataclass
class AttentionInput:
    query: np.ndarray              # [head_dim] — the query vector
    keys: np.ndarray               # [n_causal, head_dim] — all causal keys
    values: np.ndarray             # [n_causal, head_dim] — all causal values
    head_dim: int                  # dimension per head (128 for Llama 3.1)
    logits: np.ndarray             # [n_causal] — precomputed q^T k_i / sqrt(d)
    special_idx: np.ndarray        # indices of special keys (sink + local window)
    candidate_idx: np.ndarray      # indices of non-special keys (the approximation target)
```

**Special keys** (sink token at position 0, last W tokens
in the local window) can be added to receive exact attention. The
algorithm only needs to approximate attention over the
`candidate_idx` positions. The final output should combine
special and approximated keys in a single softmax.

`special_set` property gives O(1) membership lookup.

### `AttentionOutput`

Everything an algorithm returns.

```python
@dataclass
class AttentionOutput:
    output: np.ndarray             # [head_dim] — the approximated attention output
    actual_budget: int             # number of items in the final softmax
    selected_indices: np.ndarray   # (optional) which key indices were used
```

### `AttentionAlgorithm` (ABC)

The abstract base class every method must implement.

```python
class AttentionAlgorithm(ABC):

    @property
    def name(self) -> str:
        """Display name for plots and logs.
        Include key hyperparameters in the name, e.g.
        'MultiQ-Q256-G256-hybrid-k5'."""

    @property
    def kind(self) -> str:
        """'idealized' or 'algorithm'.
        Idealized methods are auto-included in every run.
        Algorithms must be explicitly requested."""

    @property
    def sweeps_budget(self) -> bool:
        """True if the evaluation runner should call run()
        at each budget in the budget_sweep list.
        False if the method has a fixed budget."""

    def prepare(
        self,
        keys: np.ndarray,           # [seq_len, head_dim]
        values: np.ndarray,         # [seq_len, head_dim]
        head_dim: int,
        queries: np.ndarray = None, # [seq_len, head_dim] — all queries in the example
        query_positions: list = None,
        seed: int = 42,
    ) -> None:
        """Called once per example before evaluating queries.
        Use for offline precomputation: clustering, sorting,
        building data structures. Default is a no-op."""

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        """Compute the approximate attention output for one query.
        Called once per query position per budget value."""

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        """Generate all parameter combinations from YAML config.
        Returns a list of algorithm instances."""
```

#### Lifecycle

The evaluation runner calls methods in this order:

1. **`expand_from_config(cfg)`** — at startup, generates
   all instances from the YAML config (e.g., sweep over
   `top_k_sweep` and `modes`).
2. **`prepare(keys, values, ...)`** — once per example,
   receives the full sequence. Do clustering, sorting, or
   any offline work here. Note: `queries` contains ALL
   query vectors in the sequence, not just the ones being
   evaluated — this lets methods like MultiQ cluster over
   the full query distribution.
3. **`run(problem, budget, rng)`** — once per query per
   budget. Must return `AttentionOutput`.

## Files

| File | Contents |
|------|----------|
| `base.py` | `AttentionAlgorithm` ABC, `AttentionInput` / `AttentionOutput` dataclasses |
| `idealized_methods.py` | `IdealTopK`, `IdealSampling`, `IdealEqualSplits`, `IdealEqualWeightSplits` |
| `multiq_grouping.py` | MultiQ: KMeans on queries, per-centroid key ordering (C=1 = mean-query sort) |
| `kmeans_clustering.py` | KMeans on keys, per-query cluster scoring |
| `__init__.py` | `METHOD_REGISTRY` — maps string keys to `MethodSpec(cls, kind)` |

## Shared Utilities in `core.py`

Algorithms import shared primitives from `src/core.py`:

| Function | What it does |
|----------|-------------|
| `softmax(x)` | Numerically stable softmax |
| `full_attention(q, K, V, d)` | Ground truth: returns `(output, logits, weights)` |
| `subset_attention(logits, V, idx)` | Softmax + weighted sum over a subset of positions |
| `relative_l2_error(approx, truth)` | `\|\|approx - truth\|\|_2 / \|\|truth\|\|_2` |
| `compute_special_indices(n, n_sink, local_window)` | Returns `(special_idx, candidate_idx)` |
| `flat_kmeans(data, C, seed)` | K-means++ init, returns `(centroids, labels)` |
| `make_equal_groups(sorted_idx, n_groups)` | Split sorted indices into equal-sized groups |
| `hybrid_attention(q, K, V, logits, groups, top_k, d, special_idx, mode)` | Two-mode attention over pre-sorted groups (TopK or Hybrid with count-weighted softmax) |
| `entropy_nats(weights)` | Shannon entropy in nats |
| `top_k_mass(weights, k)` | Fraction of attention in top-k positions |
| `nonlocal_mask(n, n_sink, local_window)` | Boolean mask excluding sink + local tokens |
| `concentration_curve(weights)` | Cumulative mass at top X% thresholds |
| `norm_statistics(vectors)` | L2 norm mean, std, CV |
| `kv_norm_correlation(K, V)` | Pearson correlation between key and value norms |

## Idealized Methods

Always auto-included in every evaluation (kind=`"idealized"`).
They use oracle knowledge (true logits/weights) and per-query
computation. Any new algorithm should be compared against
these to gauge how close it gets to the theoretical best.

| Method | Strategy |
|--------|----------|
| `IdealTopK` | Select top-B keys by logit, renormalize softmax over the subset. Biased. |
| `IdealSampling` | Sample B keys proportional to true attention weights (with replacement until B unique). |
| `IdealEqualSplits` | Sort candidates by logit, split into B equal-sized groups, represent each by (avg_key, avg_value) with count-weighted softmax. |
| `IdealEqualWeightSplits` | Sort candidates by attention weight, split into B groups of equal total weight mass. Finer resolution where attention is concentrated. |

## Our Algorithms

| Method | Offline | Per-query |
|--------|---------|-----------|
| `MultiQGrouping` | KMeans on all Q vectors -> C centroids. For each centroid, sort all keys by centroid-key logit, partition into G equal groups. | Route query to nearest centroid, use that centroid's grouping. Apply TopK or Hybrid mode. |
| `KMeansClustering` | KMeans on all keys -> C clusters. Precompute per-cluster avg_key, avg_value, count, member indices. | Score each cluster by `q^T avg_key / sqrt(d) + log(count)`. Sort by score. Apply TopK or Hybrid mode. |

### Modes

Both MultiQ and KMeans support two modes controlled by the
`mode` and `top_k` parameters:

- **`topk`** — Expand the top-k groups into individual keys.
  Exact softmax over special + expanded keys only. Groups
  beyond top-k are discarded.
- **`hybrid`** — Expand top-k groups into individual keys.
  Represent remaining groups as centroids with count-weighted
  scores: `score = q^T avg_key / sqrt(d) + log(count)`.
  Joint softmax over special keys + individual keys + group
  representatives. With `top_k=0` this is pure grouped mode.

## Adding a New Algorithm

### Step 1: Create the file

Create `src/algorithms/my_method.py`:

```python
import numpy as np
from typing import List
from .base import (
    AttentionAlgorithm, AttentionInput, AttentionOutput,
)

class MyMethod(AttentionAlgorithm):

    def __init__(self, my_param: int = 32):
        self.my_param = my_param
        self._precomputed = None

    @property
    def name(self) -> str:
        return f"MyMethod-{self.my_param}"

    # kind defaults to "algorithm" (inherited from ABC)
    # sweeps_budget defaults to False

    def prepare(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        head_dim: int,
        queries: np.ndarray = None,
        query_positions: List[int] = None,
        seed: int = 42,
    ) -> None:
        # Offline precomputation (once per example).
        # keys/values are the FULL sequence, not just causal.
        self._precomputed = ...

    def run(
        self,
        problem: AttentionInput,
        budget: int,
        rng: np.random.Generator,
    ) -> AttentionOutput:
        # problem.keys/values are the CAUSAL window for this query.
        # problem.special_idx: always-included positions.
        # problem.candidate_idx: positions to approximate.
        # problem.logits: precomputed q^T k_i / sqrt(d).
        output = ...  # np.ndarray [head_dim]
        return AttentionOutput(
            output=output,
            actual_budget=...,  # int: items in final softmax
        )

    @staticmethod
    def expand_from_config(cfg: dict) -> list:
        instances = []
        for p in cfg.get("my_param_sweep", [32]):
            instances.append(MyMethod(my_param=p))
        return instances
```

### Step 2: Register in `__init__.py`

```python
from .my_method import MyMethod

METHOD_REGISTRY = {
    ...
    "my_method": MethodSpec(MyMethod, "algorithm"),
}
```

### Step 3: Add config in `evaluation_config.yaml`

```yaml
algorithm_configs:
  my_method:
    my_param_sweep: [16, 32, 64]
```

### Step 4: Run

```bash
python -m src.evaluation.run_evaluation --algorithms my_method
```

The evaluation runner will:
1. Auto-include all idealized methods for comparison
2. Call `expand_from_config()` to generate all instances
3. For each task/example: call `prepare()` then evaluate
   queries via `run()`
4. Plot error vs budget curves comparing your method
   against the idealized baselines
