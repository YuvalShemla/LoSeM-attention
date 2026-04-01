# Evaluation Module

Runs attention approximation methods on extracted
`.pt` vector data and measures their accuracy against
ground truth. Produces per-task plots, per-head
breakdowns, cross-task overviews, and CSV exports.

## Files

| File | Purpose |
|------|---------|
| `run_evaluation.py` | `Evaluation` class — orchestrates the full pipeline + CLI entry point |
| `evaluation_config.yaml` | All hyperparameters: tasks, budgets, head selection, plotting |
| `data_loader.py` | Load `.pt` vector files per task/layer/head |
| `evaluator.py` | Per-query evaluation: builds `AttentionInput`, calls each method, measures error |
| `plotting.py` | Publication-quality log/linear plots, per-head comparison, cross-task overview |

## How It Works

### Pipeline Overview

```
evaluation_config.yaml
        |
        v
  Evaluation.__init__()    — parse config, resolve paths
        |
        v
  Evaluation.run()         — main loop
        |
        +-- validate_data()     — fail fast if any task has too few examples
        |
        +-- for each task:
        |     +-- resolve_heads()     — which (layer, q_head, kv_head) to evaluate
        |     +-- for each head:
        |     |     +-- load_examples()     — read .pt files from data/vectors/
        |     |     +-- for each example:
        |     |           +-- prepare() on all methods  — offline precomputation
        |     |           +-- for each query position:
        |     |                 +-- evaluate_query()
        |     |                       1. full_attention()     — ground truth o*
        |     |                       2. compute_special_indices()  — sink + local
        |     |                       3. method.run(problem, budget) for each method × budget
        |     |                       4. relative_l2_error(o_hat, o*)
        |     +-- aggregate_results()   — mean/std per method
        |     +-- plot per-task + per-head figures
        |
        +-- plot_overview()     — cross-task summary
        +-- save spec.json, run.json, results.csv
```

### What `evaluate_query()` Does

For each query position, the evaluator (`evaluator.py`):

1. Computes **ground truth**: `o*, logits, weights = full_attention(q, K, V, d)`
2. Splits positions into **special** (sink + local window)
   and **candidates** (everything else)
3. Packages into `AttentionInput(query, keys, values, head_dim, logits, special_idx, candidate_idx)`
4. For each method:
   - If `sweeps_budget=True`: calls `run(problem, b, rng)` for every budget in the sweep
   - If `sweeps_budget=False`: calls `run(problem, 0, rng)` once
5. Measures `relative_l2_error = ||o_hat - o*||_2 / ||o*||_2`
6. Optionally computes attention statistics (entropy, top-% mass)

### Method Resolution

The runner distinguishes two kinds of methods:

- **Idealized** (`kind="idealized"`): auto-included in
  every run. Represent the theoretical best at each budget.
- **Algorithms** (`kind="algorithm"`): must be explicitly
  requested via `--algorithms`. These are the methods
  being evaluated.

At startup the runner:
1. Collects all idealized methods from `METHOD_REGISTRY`
2. Calls `expand_from_config(cfg)` on each requested algorithm
   to generate all parameter combinations
3. Merges into a single methods list

### Aggregation

Results are aggregated at three levels:

1. **Per-head**: mean/std error across all queries for one head
2. **Per-task (weighted)**: weighted average across heads using
   triangular percentile weights (p50 gets 3x weight of p0/p100)
3. **Cross-task**: side-by-side subplot comparison

## Configuration

Everything lives in `evaluation_config.yaml`. Key sections:

### Evaluation Scope

```yaml
evaluation:
  seed: 42
  n_queries: 10          # last N token positions as queries
  n_examples: 1          # examples per task (strict — fails if too few)
  head_mode: "selected_heads"   # or "all_heads" or "custom"
  exclude_sink_token: true
  local_window:
    size: 1024
  budget_sweep:
    absolute: [16, 32, 64, 128, 256, 512, 1024, 2048]
```

| Parameter | What it controls |
|-----------|-----------------|
| `seed` | RNG seed for reproducibility |
| `n_queries` | How many query positions per example (taken from the end of the sequence) |
| `n_examples` | How many examples per task. **Strict**: fails immediately if fewer exist on disk |
| `head_mode` | Which attention heads to evaluate (see below) |
| `exclude_sink_token` | Whether to exclude position 0 (attention sink) from candidate keys |
| `local_window.size` | Number of trailing tokens always given exact attention |
| `budget_sweep.absolute` | List of budget values to sweep for budget-sweeping methods |
| `compute_statistics` | If true, compute per-query entropy and top-% mass alongside errors |

### Head Modes

| Mode | Description |
|------|-------------|
| `all_heads` | Every Q head (32) for each layer in `layers`. Use for broad scouts. |
| `selected_heads` | Reads from each task's `metadata.json`. Default for main evaluations. |
| `custom` | Explicit `(layer, q_head, kv_head)` list. Use for debugging. |

### Algorithm Configs

```yaml
algorithm_configs:
  multiq:
    n_query_clusters: [256]
    n_groups: 256
    modes: ["topk", "hybrid"]
    top_k_sweep: [0, 1, 2, 3, 5, 7, 10, 12]
  kmeans:
    n_clusters: 256
    modes: ["topk", "hybrid"]
    top_k_sweep: [0, 1, 2, 3, 5, 7, 10, 12]
```

Each key matches a `METHOD_REGISTRY` name. The algorithm's
`expand_from_config()` generates all combinations (e.g.,
`modes × top_k_sweep`).

### Plotting

```yaml
plotting:
  figsize: [16, 10]
  dpi: 200
  log_scale: true
  linear_scale: true
  error_bands: false
  idealized_colors:
    IdealTopK: "#d62728"
    IdealSampling: "#2ca02c"
    IdealEqualSplits: "#1f77b4"
    IdealEqualWeightSplits: "#9467bd"
  algorithm_colors:
    kmeans:
      topk: "#ffb380"
      hybrid: "#ff7f0e"
      marker: "D"
```

## Usage

```bash
# Run with specific algorithms and tasks
python -m src.evaluation.run_evaluation \
  --algorithms multiq kmeans \
  --tasks math_calc code_run \
  --name grouping_comparison_v1

# Use all tasks from config
python -m src.evaluation.run_evaluation \
  --algorithms kmeans

# Custom config file
python -m src.evaluation.run_evaluation \
  --algorithms multiq \
  --config path/to/my_config.yaml
```

## Output Structure

```
results/{name}_{date}/
├── spec.json                      # Full config snapshot + date
├── run.json                       # Timing + success/failure
├── results.csv                    # Flat table (all measurements)
├── per_task/
│   ├── math_calc/
│   │   ├── results_log.png        # Log-scale error vs budget
│   │   ├── results_linear.png     # Linear-scale error vs budget
│   │   ├── aggregated_stats.json  # Mean/std per method
│   │   ├── data_statistics.json   # Attention entropy, top-% mass
│   │   └── per_head/
│   │       ├── L26H12_p50.json    # Per-head stats
│   │       └── ...
│   └── ...
└── overview/
    ├── cross_task_summary_log.png
    ├── cross_task_summary_linear.png
    └── cross_task_stats.json
```

## Common Recipes

| Scenario | Config |
|----------|--------|
| Quick scout | `head_mode: all_heads`, `n_examples: 1`, `n_queries: 50`, `layers: [17]` |
| Full scout (all layers) | `head_mode: all_heads`, `n_examples: 1`, `n_queries: 100`, `layers: [0,17,19,31]` |
| Main evaluation | `head_mode: selected_heads`, `n_examples: 10`, `n_queries: 100` |
| Debug / test | `head_mode: custom`, `n_examples: 1`, `n_queries: 10`, `budget_sweep: [32, 256]` |
