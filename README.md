# Local Attention

Modular framework for evaluating attention
approximation methods on long-context LLMs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract attention vectors (GPU required)
python -m src.extraction.extract_vectors

# Run an experiment
python -m src.experiment.run_experiment \
  --algorithms meanq kmeans \
  --tasks math_calc code_run \
  --name grouping_comparison_v1

# Run exploration analysis
python -m src.exploration.run_exploration --all
```

## Structure

```
local-attention/
├── src/
│   ├── core.py             # Shared math: softmax, attention, entropy, kmeans
│   ├── algorithms/         # Algorithm implementations
│   │   ├── base.py         # ABC + dataclasses
│   │   ├── baselines.py    # OracleTopK, OracleSampling, OracleGrouping
│   │   ├── meanq_grouping.py
│   │   ├── multiq_grouping.py
│   │   └── kmeans_clustering.py
│   ├── experiment/         # Runner, plotting, data loading
│   │   ├── experiment_config.yaml
│   │   ├── run_experiment.py
│   │   ├── plotting.py
│   │   ├── data_loader.py
│   │   └── evaluator.py
│   ├── exploration/        # Data analysis dashboards
│   │   ├── exploration_config.yaml
│   │   ├── run_exploration.py
│   │   ├── visualize_head_statistics.py
│   │   ├── attention_concentration.py
│   │   ├── entropy_distribution.py
│   │   ├── kv_norm_correlation.py
│   │   └── topk_vs_sampling_bias.py
│   └── extraction/         # CUDA/MLX extraction pipeline
│       ├── extract_vectors.py
│       ├── extraction_config.yaml
│       ├── load_benchmarks.py
│       ├── cuda_extract.py
│       ├── mlx_extract.py
│       └── save_utils.py
├── data/                   # Extracted vectors (not in git)
│   ├── benchmarks/         # Raw benchmark examples
│   ├── head_statistics/    # Per-head attention profiles
│   └── vectors/            # .pt attention vectors (flat: vectors/{task}/)
├── tests/                  # Minimal test suite
├── docs/methods.md         # Algorithms in math notation
└── results/                # Experiment outputs
```

## Adding a New Algorithm

1. Create `src/algorithms/my_method.py`
2. Subclass `AttentionAlgorithm` from `base.py`
3. Implement `name`, `run()`, and optionally
   `prepare()` and `expand_from_config()`
4. Register in `src/algorithms/__init__.py`
5. Add config to `src/experiment/experiment_config.yaml`
   under `algorithm_configs`
6. Run: `python -m src.experiment.run_experiment --algorithms my_method`

## Data Format

Per-layer bfloat16 `.pt` files with JSON metadata (flat layout):
```
data/vectors/math_calc/
  ex_000/
    layer_26.pt    # {Q_rope_head12: [seq,128], ...}
    example.json   # Per-example metadata
  metadata.json    # Task-level provenance + selected_heads
```

Only layers containing selected heads are saved. See `data/vectors/README.md` for the full schema.

## Baselines

Always auto-included in every experiment:
- **OracleTopK**: select top-B keys by logit (biased)
- **OracleSampling**: sample from true attention weights
  (privileged lower bound)
- **Oracle Doubling**: doubling groups on oracle-sorted
  keys (~log2(N) budget)

## Configuration

Each module has its own config file:
- `src/extraction/extraction_config.yaml` — model, tasks, extraction phases
- `src/experiment/experiment_config.yaml` — experiment scope, budgets, algorithms, plotting
- `src/exploration/exploration_config.yaml` — exploration plots, per-plot settings

## Tests

```bash
pytest tests/ -v   # < 30 seconds on synthetic data
```
