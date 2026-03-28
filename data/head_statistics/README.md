# Head Statistics

Per-head attention statistics from Phase 1 extraction.
Used to select representative heads for Phase 2.

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
  "layer_17": {
    "head_0": {
      "entropy_full": 5.23,
      "entropy_no_sink_local": 4.81,
      "top100_mass_full": 0.42,
      "top100_mass_no_sink_local": 0.31
    },
    "head_1": { ... }
  },
  "layer_19": { ... }
}
```

## Metrics

| Metric | Description |
|--------|-------------|
| `entropy_full` | Shannon entropy over all causal keys |
| `entropy_no_sink_local` | Entropy excluding sink (pos 0) and last 1024 positions |
| `top100_mass_full` | Fraction of attention in top 100 keys |
| `top100_mass_no_sink_local` | Same, excluding sink and local window |

Each metric is averaged over ~50 sampled query
positions from the valid range.

## Head Selection for Phase 2

Three heads are selected:
1. **Max entropy** — most diffuse attention
2. **Min entropy** — most concentrated
3. **Median entropy** — typical behavior

Selection uses `entropy_no_sink_local` to isolate
"global" attention behavior from the sink and local
window effects.
