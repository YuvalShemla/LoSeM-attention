# Exploration

Data analysis scripts for understanding attention
patterns before running approximation evaluations.

## Dashboards

Two dashboards are generated per head:

**Global Overview** (`global_dashboard.png`) — 8 panels:
- Row 0: Top-K mass, Concentration curves, Entropy vs position, Bias comparison
- Row 1: Mean-Q vs Full scatter, Query Deviation histogram
- Row 2: PCA (Q+K), t-SNE (Q+K)

**Pairwise Comparisons** (`pairwise_dashboard.png`) — 12 panels:
- Rows: QK, QQ, KK pair types
- Cols: Cosine histogram, Cosine vs distance, Dot histogram, Dot vs distance

## Aggregation

When multiple heads are analyzed, aggregated views are
generated per task and across all tasks:
- `aggregated_mean/` — mean across heads
- `aggregated_median/` — median across heads
- `aggregated_percentiles/` — p25, p75, p90
- `aggregated_variance/` — spaghetti plots + IQR bands

## Scripts

| File | Purpose |
|------|---------|
| `attention_concentration.py` | Top-K mass vs position, concentration curves |
| `entropy_distribution.py` | Entropy vs position, entropy histograms |
| `topk_vs_sampling_bias.py` | TopK vs Uniform vs Oracle at various budgets |
| `kv_norm_correlation.py` | Key-value norm relationship (standalone) |
| `pairwise_similarity.py` | QK/QQ/KK cosine + dot products with distance binning |
| `query_analysis.py` | Mean-Q scores + query deviation |
| `embedding_projections.py` | PCA and t-SNE of Q+K spaces |
| `aggregation.py` | Cross-head/task aggregation (mean, median, percentiles, variance) |
| `dashboard_global.py` | Global overview dashboard (8 panels) |
| `dashboard_pairwise.py` | Pairwise comparison dashboard (12 panels) |
| `run_exploration.py` | CLI entry point |
| `visualize_head_statistics.py` | Cross-task and per-example head statistics |

## Usage

```bash
# Specific tasks
python -m src.exploration.run_exploration \
  --tasks math_calc code_run

# All tasks
python -m src.exploration.run_exploration --all
```

## Output

```
results/exploration_{date}/
├── math_calc/
│   ├── L26H12/
│   │   ├── global_dashboard.png
│   │   └── pairwise_dashboard.png
│   ├── L31H24/
│   │   └── ...
│   ├── aggregated_mean/
│   │   ├── global_dashboard.png
│   │   └── pairwise_dashboard.png
│   ├── aggregated_median/
│   ├── aggregated_percentiles/
│   └── aggregated_variance/
├── code_run/
│   └── ...
└── all_tasks/
    ├── aggregated_mean/
    ├── aggregated_median/
    ├── aggregated_percentiles/
    └── aggregated_variance/
```

## Configuration

All parameters in `exploration_config.yaml`:

```yaml
exploration:
  seed: 42
  n_examples: 1
  n_queries: 50
  head_mode: "selected_heads"

concentration:
  top_k_values: [10, 50, 100, 200, 500]

bias_comparison:
  budget_fractions: [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]

pairwise:
  n_sample: 10000
  n_distance_bins: 15
  n_distance_ranges: 4

embedding:
  max_points_pca: 0       # 0 = all
  max_points_tsne: 3000
  tsne_perplexity: 30.0
  tsne_n_iter: 1000
```
