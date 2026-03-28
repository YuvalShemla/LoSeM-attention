# Experiment Module

Handles the full experiment lifecycle: data loading,
method evaluation, result aggregation, and plotting.

## Files

| File | Purpose |
|------|---------|
| `run_experiment.py` | `Experiment` class — multi-task orchestration + CLI |
| `data_loader.py` | Load .pt vector files per task/layer/head |
| `evaluator.py` | Per-query evaluation against ground truth |
| `statistics.py` | Shared analysis: entropy, concentration, norms |
| `plotting.py` | Publication-quality plots |

## Controlling What Runs

Everything is in `src/experiment/experiment_config.yaml`.
Three knobs control the experiment scope:

| Knob | What it controls |
|------|-----------------|
| `head_mode` | Which heads to evaluate |
| `n_examples` | How many examples per task |
| `n_queries` | How many query positions per example |

### Head Modes

**`all_heads`** — Runs every Q head (32)
for each layer in the `layers` list. Good for broad
scout runs or head comparison.

```yaml
experiment:
  head_mode: "all_heads"
  layers: [0, 17, 19, 31]   # 4 layers × 32 heads = 128 heads
  n_examples: 1
```

**`selected_heads`** — Reads which (layer, head)
combos to use from each task's `metadata.json`.
This is the default for main experiments after
head selection.

```yaml
experiment:
  head_mode: "selected_heads"
  n_examples: 10
```

**`custom`** — Explicit list. Use for debug runs or
single-head analysis.

```yaml
experiment:
  head_mode: "custom"
  custom_heads:
    - {layer: 17, q_head: 5, kv_head: 0}
  n_examples: 5
```

### Strict Validation

`n_examples` is strict: if a task has fewer examples
on disk than requested, the run **fails immediately**
with a clear error message before any computation
starts. This prevents silent partial results.

### Common Recipes

| Scenario | head_mode | n_examples | n_queries | layers |
|----------|-----------|------------|-----------|--------|
| Quick scout | `all_heads` | 1 | 50 | `[17]` |
| Full scout (all layers) | `all_heads` | 1 | 100 | `[0,17,19,31]` |
| Main experiment | `selected_heads` | 10 | 100 | (from metadata) |
| Debug / test | `custom` | 1 | 10 | (from list) |

## How Experiments Work

1. Load `experiment_config.yaml` for all hyperparameters
2. **Validate data** — check every task has enough
   examples before starting any computation
3. Resolve methods: auto-include baselines + expand
   requested algorithms via `expand_from_config()`
4. For each task:
   - Resolve heads (from mode config)
   - For each (layer, head):
     - Load n_examples from .pt files
     - For each example:
       - Call `prepare()` on all methods
       - Evaluate last `n_queries` positions
   - Save per-task plots and statistics
5. Generate cross-task overview
6. Save `spec.json`, `run.json`, `results.csv`

## Output Structure

```
results/{name}_{date}/
├── spec.json           # Full config snapshot + date
├── run.json            # Timing + success/failure
├── results.csv         # Flat table (all measurements)
├── per_task/
│   ├── math_calc/
│   │   ├── results_log.png
│   │   ├── results_linear.png
│   │   ├── aggregated_stats.json
│   │   └── data_statistics.json
│   └── ...
└── overview/
    ├── cross_task_summary.png
    └── cross_task_stats.json
```
