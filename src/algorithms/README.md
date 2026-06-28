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
| `wildcat2/` | WildCat2: faithful port of microsoft/wildcat (RPNys + CompressKV + WtdAttn) on candidate keys |
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

## PQ-based Methods

### `FullAttentionPQ_topk`

Implemented in `pq_methods.py` as class `FullAttentionPQ`.
Registered under key `fullattention_pq`.

Goal: approximate full attention over all non-special
positions (`candidate_idx`) using quantized query + quantized
keys, while spending budget `B` on exact logits for
PQ-selected top candidates.

#### Scoring rule

For each subspace `s`, quantize query subvector `q_s` to its
nearest codeword `c_q[s]`. Each key `j` has stored PQ codeword
indices `c_k[j, s]`. Approximate candidate logit:

    l_hat_j = (1 / sqrt(d)) * sum_s < C_s[c_q[s]], C_s[c_k[j, s]] >

Special logits use exact `problem.logits[special_idx]`.
For budget `B`, select top-`B` candidate positions by
`l_hat_j` and replace those scores with exact logits.
Then run one joint softmax over:

- exact special scores
- exact logits for top-`B` PQ-selected candidates
- approximate candidate scores for the rest

and return `output = softmax(scores) @ values`.

#### Config

```yaml
algorithm_configs:
  fullattention_pq:
    m_sweep: [8]
    n_codes: 256
```

Notes:
- `sweeps_budget = True` (uses evaluation budget sweep).
- `actual_budget = len(special_idx) + B`.
- Current PQ storage in this repo uses `int32` codes in memory
  (4 bytes per subcode), not packed 8-bit codes.

### `WildCat2`

Registered under key `wildcat2`. Faithful port of
[microsoft/wildcat](https://github.com/microsoft/wildcat) and the WildCat paper
(Alg. 2–4): `rp_nystrom`, `compress_kv`, `weighted_attention` in
`src/algorithms/wildcat2/`. Compresses `candidate_idx` only; sink and local
window are included in WtdAttn with unit weights. Keys are centered using the
mean over all causal keys before compression. CUDA is used automatically when
available (`device: cpu` to override).

```yaml
algorithm_configs:
  wildcat2:
    num_bins: 1
    q_scale_mode: key_max     # max key norm (examples/kvcache parity); or query
```

- `sweeps_budget = True` (`B` = coreset rank `r` on candidates).
- `actual_budget = len(special_idx) + r` (or all candidates if `n_cand <= r`).
- **Budget sweeps**: each budget runs a **fresh** RPC coreset of that size (random
  pivots). Error vs budget is **not** guaranteed monotone — a larger budget is a
  new draw, not an extension of a smaller coreset. Pivots are reproducible via
  `(evaluation.seed, budget)`.
- **vs `~/projects/wildcat/examples/kvcache/measure_attention_error.py`**: that script
  uses `q_scale_mode: key_max` (max key norm), often `num_bins: 16`, and
  `compression_ratio` (not absolute `budget`). WildCat2 now defaults to
  `q_scale_mode: key_max` to match it; set `q_scale_mode: query` to use the actual
  query norm instead (WildCat.forward / paper Alg. 4).

### `WildCat3`

Registered under key `wildcat3`. This option delegates directly to the vendored
original WildCat package under `wildcat/` instead of using the `wildcat2` port.
It calls `wildcat/examples/kvcache/compress_kv_cache.py` and the original
`wildcat.weighted_attention`, matching `measure_attention_error.py` closely while
still fitting the evaluation runner's per-query API.

```yaml
algorithm_configs:
  wildcat3:
    num_bins: 1
    dtype: float32            # bfloat16/float16 also supported
```

- `sweeps_budget = True`. The requested budget is converted to the kvcache
  script's middle-segment `compression_ratio`, so with `num_bins: 1` the
  compressed middle coreset size equals the requested budget.
- `actual_budget` is the compressed KV length returned by original WildCat
  (`sink + compressed middle + window`), so with `num_bins: 1` it should be
  `len(special_idx) + requested_budget` unless the budget exceeds the middle.
- This path is useful as a ground-truth integration check against the original
  WildCat repository code. It is intentionally less integrated with
  loco-attention internals than `wildcat2`.

### `FCFrankWolfeL2`

Registered under key `fc_frank_wolfe_l2`, in `src/algorithms/fcfw_l2/`. Same
pipeline as `wildcat2` (full-causal key centering, shared `find_kernel_temperature`,
the `exp(scaling/2)` weight/value aggregation in `_finish_compress_kv`, and
`weighted_attention`); only the **coreset selection** differs.

Instead of randomly-pivoted Nystrom, the coreset is built by **fully-corrective
Frank-Wolfe** in the unit-diagonal Gaussian kernel space. It approximates the
kernel-mean target `mu = sum_n phi(k_n)` by a weighted set of selected feature
maps, minimizing `|| mu - sum_j c_j phi(k_j) ||^2`. Each step:

- **Selection (FW linear oracle):** add the key most correlated with the current
  residual, `i* = argmax_i g_i`, where `g = b - K_{:,S} c` and `b = K @ 1`.
- **Correction (l2):** re-optimize all selected weights by unconstrained least
  squares, `c = K_SS^{-1} b_S` (the fully-corrective step). This is exactly the
  WildCat weight formula, so values/weights flow through `_finish_compress_kv`
  unchanged.

```yaml
algorithm_configs:
  fc_frank_wolfe_l2:
    num_bins: 1
    q_scale_mode: key_max     # shared with wildcat2; or query
```

- `sweeps_budget = True` (`B` = coreset size `r` on candidates).
- **Deterministic** (no RNG/seed dependence) and **nested**: the first `r` picks
  are identical regardless of the target size, so the **kernel-space residual**
  is monotone non-increasing in budget (the downstream attention error is
  typically, though not strictly, monotone) — unlike `wildcat2`'s
  independent-per-budget draws. The budget sweep warm-starts a single greedy run
  to the largest budget.
- Temperature is set **precisely as in `wildcat2`** (same centering, `q_scale_mode`,
  `phi`, and `key_multiplier = sqrt(scale)/tau`).
- Cost is the same class as `rp_nystrom` (`O(n r^2)` dominated by residual
  updates) plus a one-time blocked `b = K @ 1` (`O(n^2 d)`). float32 on CUDA when
  available.

### `TensorFCFWLq`

Registered under key `tensor_fcfw_lq`, in `src/algorithms/tensor_fcfw_lq/`. The
**query-weighted** sibling of `tensor_fcfw_l2`: same value-aware coreset of the
candidate region (existing keys, synthetic `(value, mass)`, sink + local window
exact via `weighted_attention`), but it minimizes a *different norm*. Where
`tensor_fcfw_l2` minimizes the RKHS-l2 (Frobenius) error of the tensor target
`sigma = sum_i phi(k_i) (x) [1, v_i]`, this minimizes

`||sigma - sigma'||_* = E_{q in Q} | (sigma - sigma') . psi(q) |`

i.e. the **L1-over-queries** average of the unnormalized readout error, where
`psi(q)` is the attention feature of query `q` (`<psi(q), phi(k)> =
exp(scale q.k)`), so `sigma . psi(q) = [D(q), N(q)]`. The probe set `Q` is the
set of **earlier-context queries** (the same probes `learned` uses, via
`build_probe_queries` / `reference_position`).

Concretely, with `A[q,i] = exp(scale q.k_i - c)` (`m = |Q|` probes, `n`
candidates, `c` a global shift that cancels) and `M = [1, V]`:

- **Selection (FCFW):** in the *empirical attention Gram* `K~ = A^T A` (inner
  products of candidate attention profiles over `Q`), with value-aware residual
  `G = E_S M` and oracle `i* = argmax_i ||G[i,:]||^2`. This is the same
  pivoted-Cholesky-with-labels recursion as `tensor_fcfw_select`, but the kernel
  is the *data-driven* `K~` (non-unit diagonal) instead of the analytic
  Gaussian, so selection is driven by how candidates serve the actual query
  distribution. Cost `O(n r^2 + n r (1+d))` after a one-time `A^T(A M)` pass.
- **Correction (lq):** the synthetic `U` is refined to minimize the true
  `L_{2,1}` objective `sum_q ||T_q - A[:,S] U||_2` by iteratively reweighted
  least squares (`W = diag(1/||R_q||)`). `irls_iters <= 1` falls back to the
  plain query-space least-squares (L2) solve. Each reweighted solve uses a
  **truncated / ridge-damped SVD** (relative cutoff `rcond`): the value solve is
  under-determined whenever the budget `r` reaches the number of probes `m`
  (`A[:,S]` is `m x r`), so the plain normal equations are singular and would
  amplify tiny singular directions into exploding synthetic values. The
  truncated solve returns the minimum-norm fit and stays bounded for any `r`.

Unlike `tensor_fcfw_l2` there is **no temperature rescaling or key centering** —
the objective is defined directly through the real attention kernel evaluated at
`Q`, so selected keys are used verbatim and the `(value, mass)` solve is
calibrated for the kernel `weighted_attention` consumes. **Lifecycle matches
`learned`:** the coreset is built once at the reference (last test) position;
at evaluation only the exact sink + local window change with query position.
Deterministic and nested (greedy forward selection ⇒ the `lq` error over `Q` is
non-increasing in budget; warm-starts larger budgets). Supports
`exact_denominator` like the other coreset methods.

```yaml
evaluation:
  n_train_queries: 5000   # |Q| for learned + TFCFW-lq (exceed max budget)
algorithm_configs:
  tensor_fcfw_lq:
    oracle: fw              # fw = argmax ||G[i,:]||^2; or omp
    irls_iters: 5           # IRLS steps for the lq solve (<=1 => L2 surrogate)
    rcond: 1.0e-3           # relative SVD cutoff for the value solve
    exact_denominator: true
    n_sink: 1
    local_window: 1024
```

### `LearnedCoreset`

Registered under key `learned`, in `src/algorithms/learned/`. Learns **B
synthetic** triples `(k'_j, v'_j, w'_j)` by gradient descent (Adam). Only the
residual (candidate) region is learned; sink and local window stay exact and are
added at evaluation.

**Objective (matches the exact eval pipeline).** For each probe query `q`, the
approximate output is computed exactly as `run()` does it — exact sink +
local-window tokens concatenated with the learned pairs, normalized by
`weighted_attention` (with `exact_denominator` if enabled) — and matched against
the **true full attention** over the fixed reference (test) context:

```
pred(q)   = weighted_attention(q, [K_special, K'], [V_special, w'(.)V'], [1, w'], exact_denom)
target(q) = softmax(scale * q K_ref^T) V_ref
loss      = mean_q ||pred(q) - target(q)||^2 / ||target(q)||^2
```

Because training == eval, the learned pairs are forced to calibrate their
**absolute mass** (via the learnable weight `w'_j`, folded into the numerator as
`w'_j v'_j` and the denominator as `w'_j`), not just the softmax average. An
earlier self-normalized formulation left the mass uncalibrated and the candidate
contribution exploded when combined with the exact special tokens / `Z_exact`.

**Probe queries.** The latest ``evaluation.n_train_queries`` query vectors in
the causal prefix before the test position (trailing window, excluding test
queries). Shared with TFCFW-lq via ``src/algorithms/probe_queries.py``. The forward pass is fully
vectorized over the probe batch; Adam with step LR decay and early stopping on a
held-out 10% split. The relative-L2 denominator is floored (`rel_l2_floor`) so
high-entropy heads with small-norm targets do not dominate the gradient.

**Initialization (pure).** `init: kmeans` (default) clusters the candidate keys
into B centroids (values = per-cluster mean, mass = cluster size); `random`
samples candidate `(k, v)` pairs. FCFW is intentionally **not** used, so the
method is a self-contained learned baseline.

**Monotone budget sweep (nested).** A larger budget freezes the smaller budget's
trained coreset — folded into a fixed per-probe contribution, exactly like the
special tokens — and trains only the newly added pairs, initialized at near-zero
mass. The starting point reproduces the smaller coreset, so error is
non-increasing in budget (in the training/validation objective). Disable with
`nested_budget: false`.

**vs Cartridge:** Cartridge trains KV caches end-to-end through the frozen LLM
with KL / next-token distillation on synthetic conversations across all layers.
Learned is per-head, direct (relative-)L2 on attention outputs.

**vs TensorFCFWL2:** Tensor FCFW produces synthetic values analytically via a
fully-corrective tensor solve; Learned optimizes `(K', V', w')` freely with SGD
from a pure (k-means/random) start.

```yaml
evaluation:
  n_train_queries: 5000
algorithm_configs:
  learned:
    init: kmeans
    lr: 0.05
    n_steps: 500
    loss: relative_l2
    rel_l2_floor: 0.01
    nested_budget: true
    exact_denominator: true
    n_sink: 1
    local_window: 1024
```

### `TensorFCFWL2`

Registered under key `tensor_fcfw_l2`, in `src/algorithms/tensor_fcfw_l2/`. A
**value-aware** version of FCFW. Instead of approximating the kernel mean
`mu = sum_i phi(k_i)`, it approximates the **tensor**

`T = sum_i phi(k_i) (x) [1, v_i]`  (an operator `H (x) R^{1+d}`),

which is the exact query-agnostic sufficient statistic for attention: for any
query, `[D(q), N(q)] = <phi(q), T>`. Coreset keys are existing keys; the coreset
**value vectors are synthetic** — produced by the algorithm's own fully-corrective
tensor solve, not copied from any key.

- **Selection**: the FW residual is now vector-valued, `G = E_S M in R^{n x (1+d)}`
  with `M = [1, V]` and `E_S` the Nystrom kernel residual. Pick
  `i* = argmax_i ||G[i,:]||^2`. The mass channel alone reproduces plain FCFW-l2;
  the value channels make selection value-aware. Implemented by a
  pivoted-Cholesky-with-labels recursion (rank-1 updates), so cost stays
  `O(n r^2 + n r (1+d))` — the same leading term as plain FCFW, not `(1+d)x`.
- **Synthetic values**: the algorithm's own corrective solve
  `U = (K_SS^{-1} K_{S,:} (.) exp(scaling/2)) @ M`; `U[...,0]` is the synthetic
  weight, `U[...,1:]` the synthetic value. The `exp(scaling/2)` factor is the
  attention kernel's Gram in stable normalized-Gaussian form, so `U` is the true
  minimizer in the kernel `weighted_attention` consumes.

```yaml
algorithm_configs:
  tensor_fcfw_l2:
    num_bins: 1
    q_scale_mode: key_max
    oracle: fw            # or omp = argmax ||G[i,:]||^2 / E_S[i,i]
```

- Shares temperature (`find_kernel_temperature`, `q_scale_mode`), full-causal
  `kbar` centering, deterministic nested budget sweep, and `weighted_attention`
  with `wildcat2` / `fcfw_l2`. The only differences vs `fcfw_l2` are value-aware
  selection and the self-contained tensor value solve. float32 on CUDA when
  available.

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
