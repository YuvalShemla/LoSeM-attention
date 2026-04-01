# Algorithms

Attention approximation methods. Each algorithm
implements the `AttentionAlgorithm` ABC from `base.py`.

## Interface

```python
class AttentionAlgorithm(ABC):
    name: str                    # display name
    kind: str                    # "idealized" or "algorithm"
    sweeps_budget: bool          # True for budget-sweeping methods

    def prepare(keys, values, head_dim, ...)  # offline
    def run(problem, budget, rng) -> AttentionOutput
    def expand_from_config(cfg) -> list[Self]
```

## Files

| File | Contents |
|------|----------|
| `base.py` | ABC, AttentionInput/Output dataclasses |
| `../core.py` | softmax, rel_l2, hybrid_attention, equal groups |
| `idealized_methods.py` | IdealTopK, IdealSampling, IdealEqualSplits, IdealEqualWeightSplits |
| `meanq_grouping.py` | MeanQ: sort by mean-query projection |
| `multiq_grouping.py` | MultiQ: KMeans on queries, per-centroid ordering |
| `kmeans_clustering.py` | KMeans on keys, per-query cluster scoring |
| `__init__.py` | METHOD_REGISTRY |

## Adding a Method

1. Create a new file (e.g., `my_method.py`)
2. Subclass `AttentionAlgorithm`
3. Implement `name`, `run()`, `expand_from_config()`
4. Override `prepare()` if offline precomputation needed
5. Add to `METHOD_REGISTRY` in `__init__.py`

The experiment runner auto-discovers idealized methods
(kind="idealized") and requires algorithms to be
explicitly requested via `--algorithms`. Every new
algorithm will automatically be tested against the
idealized methods.
