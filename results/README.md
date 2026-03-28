# Results

Each experiment creates a timestamped folder:
`{name}_{YYYY-MM-DD_HH-MM}/`

## Structure

```
results/{name}_{date}/
├── README.md           # Auto-generated summary
├── spec.json           # What was intended (before)
├── run.json            # What happened (after)
├── results.csv         # Flat table, all measurements
├── per_task/
│   ├── math_calc/
│   │   ├── results_log.png
│   │   ├── results_linear.png
│   │   ├── aggregated_stats.json
│   │   └── data_statistics.json
│   └── code_run/
│       └── ...
└── overview/
    ├── cross_task_summary.png
    └── cross_task_stats.json
```

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| task | str | Task name |
| layer | int | Layer number |
| head | int | Q head number |
| example_id | str | Example identifier |
| query_pos | int | Query position in sequence |
| method | str | Algorithm display name |
| method_kind | str | "baseline" or "algorithm" |
| budget | int | Requested budget |
| actual_budget | int | Keys actually used |
| rel_l2_error | float | Relative L2 error |
| seed | int | Random seed |

## Quick Analysis

```python
import pandas as pd
df = pd.read_csv("results/exp_xxx/results.csv")
df.groupby(["task", "method"])[
    "rel_l2_error"
].describe()
```

## Exploration Results

Exploration runs create separate folders:
```
results/exploration_{date}/
├── math_calc/
│   ├── attention_concentration.png
│   ├── entropy_distribution.png
│   ├── kv_norm_correlation.png
│   └── topk_vs_sampling_bias.png
└── ...
```
