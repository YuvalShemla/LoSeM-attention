# Exploration

Data analysis scripts for understanding attention
patterns before running approximation experiments.

## Scripts

| File | Analysis |
|------|----------|
| `attention_concentration.py` | Top-K mass vs position, concentration curves |
| `entropy_distribution.py` | Entropy vs position, entropy histograms |
| `kv_norm_correlation.py` | Key-value norm relationship, top-keys correlation |
| `topk_vs_sampling_bias.py` | TopK vs Uniform vs Oracle at various budgets |
| `visualize_head_statistics.py` | Cross-task and per-example head statistics |
| `run_exploration.py` | CLI entry point for all analyses |

## Usage

```bash
# All plots for specific tasks
python -m src.exploration.run_exploration \
  --tasks math_calc code_run

# Specific plots
python -m src.exploration.run_exploration \
  --tasks math_calc \
  --plots entropy_distribution kv_norm_correlation

# Everything
python -m src.exploration.run_exploration --all
```

## Head Modes

The exploration config supports two head modes:

**`selected_heads`** (default) — reads selected heads
from each task's `metadata.json` and generates plots
for all 5 selected heads. Each head gets its own
subdirectory.

**`custom`** — single head specified in the config.
Useful for debugging or quick runs.

## Output

Results saved to `results/exploration_{date}/`:
```
results/exploration_2026-03-28_14-30/
├── math_calc/
│   ├── L26H12/
│   │   ├── attention_concentration.png
│   │   ├── entropy_distribution.png
│   │   ├── kv_norm_correlation.png
│   │   └── topk_vs_sampling_bias.png
│   ├── L31H24/
│   │   └── ...
│   ├── L6H4/
│   │   └── ...
│   ├── L12H14/
│   │   └── ...
│   └── L1H2/
│       └── ...
├── code_run/
│   └── ...
└── ...
```

## Configuration

All parameters live in `src/exploration/exploration_config.yaml`:

```yaml
exploration:
  seed: 42
  n_examples: 1
  n_queries: 200
  head_mode: "selected_heads"   # or "custom"
  # Used only for head_mode: "custom"
  layer: 17
  q_head: 0
  kv_head: 0
  attention_sink:
    n_sink_tokens: 1
  local_window:
    size: 1024

plots:
  - attention_concentration
  - entropy_distribution
  - kv_norm_correlation
  - topk_vs_sampling_bias

# Per-plot options
concentration:
  top_k_values: [10, 50, 100, 200, 500]
kv_norms:
  top_pct: 10.0
bias_comparison:
  budget_fractions: [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
```

## What Each Plot Shows

**Attention concentration**: How much mass the top
K keys capture. Diffuse attention = low concentration.

**Entropy distribution**: Shannon entropy across
queries. High entropy = spread-out attention. Shown
both with and without sink/local window.

**K-V norm correlation**: Whether high-norm keys
also have high-norm values. Relevant for value-
weighted sampling strategies.

**TopK vs sampling bias**: Compares truncation
(TopK), random (Uniform), and ideal (Oracle)
selection at matching budgets.
