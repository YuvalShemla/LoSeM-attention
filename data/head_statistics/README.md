# Head Statistics

Per-head attention statistics from the scout pass.
Used to select representative heads for vector extraction.

## Structure

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
      ...
    ]
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

| Metric | Description |
|--------|-------------|
| `full_entropy` | Shannon entropy over all causal keys |
| `nonlocal_entropy` | Entropy excluding sink (pos 0) and last 1024 positions |
| `full_top1pct_mass` | Fraction of attention in top 1% of keys |
| `nonlocal_top1pct_mass` | Same, excluding sink and local window |
| `full_top5pct_mass` | Fraction of attention in top 5% of keys |
| `nonlocal_top5pct_mass` | Same, excluding sink and local window |

Each metric is averaged over the last N query positions
(configured via `head_statistics.n_queries`).

## Head Selection

Five heads are selected at percentile positions
[0, 25, 50, 75, 100] of `nonlocal_entropy`:
1. **P0** — most concentrated attention (min entropy)
2. **P25** — below-average diffusion
3. **P50** — median behavior
4. **P75** — above-average diffusion
5. **P100** — most diffuse attention (max entropy)

Selection uses `nonlocal_entropy` to isolate
"global" attention behavior from the sink and local
window effects.
