# Head Statistics

Per-head attention statistics computed during the extraction scout pass. These characterize the attention behavior of every head in the model for a given task, and are used to select which heads to extract vectors for.

## What's in here

One JSON file per task, each containing statistics for all 1024 heads (32 layers x 32 Q heads) computed on the shortest example for that task.

```
head_statistics/
└── llama3.1_8b/
    ├── math_calc.json
    ├── code_run.json
    ├── longbook_sum_eng.json
    ├── passkey.json
    ├── multi_doc_qa.json
    └── single_doc_qa.json
```

## Format

```json
{
  "metadata": {
    "model": "meta-llama/Meta-Llama-3.1-8B",
    "backend": "cuda",
    "task": "math_calc",
    "scout_examples": ["math_calc_0"],
    "sequence_lengths": [45018],
    "selected_heads": [
      {"layer": 5, "q_head": 12, "kv_head": 1, "nonlocal_entropy": 1.23},
      {"layer": 17, "q_head": 3, "kv_head": 0, "nonlocal_entropy": 3.45},
      ...
    ],
    "head_statistics_params": {
      "n_queries": 10, "n_sink_tokens": 1, "local_window": 1024
    }
  },
  "layer_0": {
    "head_0": {
      "full_entropy": 2.74,
      "nonlocal_entropy": 8.17,
      "full_top1pct_mass": 0.97,
      "nonlocal_top1pct_mass": 0.52,
      "full_top5pct_mass": 0.99,
      "nonlocal_top5pct_mass": 0.82
    },
    "head_1": { ... }
  },
  "layer_1": { ... }
}
```

## Metrics

Each metric is averaged over the last N query positions (configured via `head_statistics.n_queries` in `extraction_config.yaml`).

| Metric | Description |
|--------|-------------|
| `full_entropy` | Shannon entropy (nats) over all causal keys |
| `nonlocal_entropy` | Entropy after excluding sink and local window, renormalized |
| `full_top1pct_mass` | Fraction of attention captured by the top 1% of keys |
| `nonlocal_top1pct_mass` | Same, computed on the nonlocal portion only |
| `full_top5pct_mass` | Fraction of attention captured by the top 5% of keys |
| `nonlocal_top5pct_mass` | Same, computed on the nonlocal portion only |

**"Nonlocal"** means: the sink token (position 0) and the local window (last 1024 positions) are masked out, and the remaining weights are renormalized before computing the metric. This isolates the long-range attention behavior from the trivially high-weight positions.

## Head Selection

The extraction pipeline ranks all 1024 heads by a chosen metric and picks heads at percentile positions. Controlled by `head_selection` in `extraction_config.yaml`:

```yaml
head_selection:
  metric: "nonlocal_entropy"         # which metric to rank by
  percentiles: [0, 25, 50, 75, 100]  # -> 5 heads
```

With the default settings, 5 heads are selected:

| Percentile | Meaning |
|-----------|---------|
| P0 | Most concentrated attention (lowest nonlocal entropy) |
| P25 | Below-average diffusion |
| P50 | Median behavior |
| P75 | Above-average diffusion |
| P100 | Most diffuse attention (highest nonlocal entropy) |

The selected heads and their metric values are recorded in the `metadata.selected_heads` field of each JSON file.

## Regenerating

These files are automatically regenerated whenever the extraction pipeline runs. To regenerate with different parameters, edit `head_statistics` in `extraction_config.yaml` and re-run:

```bash
python -m src.extraction.extract_vectors
```
