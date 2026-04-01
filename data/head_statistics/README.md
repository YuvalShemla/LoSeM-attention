# Head Statistics

Per-head attention statistics computed during the extraction scout pass. These characterize the attention behavior of every head in the model for a given task, and are used to select which heads to extract vectors for.

## What's in here

One JSON file per task (scout stats), plus optional per-example stats when `compute_all_examples: true` is set in the config.

```
head_statistics/
└── llama3.1_8b/
    ├── math_calc.json                   # scout stats (shortest example)
    ├── code_run.json
    ├── longbook_sum_eng.json
    ├── passkey.json
    ├── multi_doc_qa.json
    ├── single_doc_qa.json
    └── per_example/                     # all-examples stats (optional)
        ├── math_calc/
        │   ├── ex_000.json              # all 1024 heads, same format
        │   ├── ex_001.json
        │   ├── ex_002.json
        │   ├── ex_003.json
        │   └── ex_004.json
        ├── code_run/
        │   └── ...
        └── ...
```

## Format

Each JSON file (both scout and per-example) has the same structure:

```json
{
  "metadata": {
    "model": "meta-llama/Meta-Llama-3.1-8B",
    "backend": "cuda",
    "task": "math_calc",
    "example_id": "math_calc_0",
    "example_index": 0,
    "sequence_length": 45018,
    "n_layers": 32,
    "n_q_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "head_statistics_params": {
      "n_queries": 10, "exclude_sink_token": true, "local_window": 1024
    }
  },
  "layer_0": {
    "head_0": {
      "full_entropy": 2.74,
      "effective_entropy": 8.17,
      "full_top1pct_mass": 0.97,
      "effective_top1pct_mass": 0.52,
      "full_top5pct_mass": 0.99,
      "effective_top5pct_mass": 0.82
    },
    "head_1": { ... }
  },
  "layer_1": { ... }
}
```

Scout JSONs additionally include `scout_examples`, `sequence_lengths`, and `selected_heads` in metadata. Per-example JSONs include `example_id`, `example_index`, and `sequence_length`.

## Metrics

Each metric is averaged over the last N query positions (configured via `head_statistics.n_queries` in `extraction_config.yaml`).

| Metric | Description |
|--------|-------------|
| `full_entropy` | Shannon entropy (nats) over all causal keys |
| `effective_entropy` | Entropy after excluding sink and local window, renormalized |
| `full_top1pct_mass` | Fraction of attention captured by the top 1% of keys |
| `effective_top1pct_mass` | Same, computed on the effective portion only |
| `full_top5pct_mass` | Fraction of attention captured by the top 5% of keys |
| `effective_top5pct_mass` | Same, computed on the effective portion only |

**"Effective"** means: the configured sink tokens and local window are masked out, and the remaining weights are renormalized before computing the metric. This isolates the long-range attention behavior from the trivially high-weight positions.

## Head Selection

The extraction pipeline ranks all 1024 heads by a chosen metric and picks heads at percentile positions. Controlled by `head_selection` in `extraction_config.yaml`:

```yaml
head_selection:
  metric: "effective_entropy"         # which metric to rank by
  percentiles: [0, 25, 50, 75, 100]  # -> 5 heads
```

With the default settings, 5 heads are selected:

| Percentile | Meaning |
|-----------|---------|
| P0 | Most concentrated attention (lowest effective entropy) |
| P25 | Below-average diffusion |
| P50 | Median behavior |
| P75 | Above-average diffusion |
| P100 | Most diffuse attention (highest effective entropy) |

The selected heads and their metric values are recorded in the `metadata.selected_heads` field of each scout JSON file.

## Per-Example Stats

When `head_statistics.compute_all_examples: true` is set, the pipeline runs an additional forward pass for each of the N examples per task (all heads), saving per-example statistics to `per_example/{task}/ex_{NNN}.json`. This adds ~N forward passes per task but gives ~5120 data points per task (5 examples x 1024 heads) instead of 1024, enabling analysis of how head behavior varies across inputs.

## Regenerating

These files are automatically regenerated whenever the extraction pipeline runs. To regenerate with different parameters, edit `head_statistics` in `extraction_config.yaml` and re-run:

```bash
python -m src.extraction.extract_vectors
```
