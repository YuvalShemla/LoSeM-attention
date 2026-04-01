# Source Code

## Module Overview

```
src/
├── core.py             # Shared math: softmax, attention, entropy, norms, kmeans
├── algorithms/         # Attention approximation methods
│   ├── base.py         # ABC + dataclasses
│   ├── idealized_methods.py  # IdealTopK, IdealSampling, IdealEqualSplits, IdealEqualWeightSplits
│   ├── multiq_grouping.py
│   └── kmeans_clustering.py
├── evaluation/         # Evaluation infrastructure
│   ├── evaluation_config.yaml  # Evaluation settings
│   ├── run_evaluation.py  # Evaluation class + CLI
│   ├── data_loader.py  # .pt file loading
│   ├── evaluator.py    # Per-query evaluation
│   └── plotting.py     # Publication-quality plots
├── exploration/        # Data analysis dashboards
│   ├── exploration_config.yaml  # Exploration settings
│   ├── run_exploration.py
│   ├── visualize_head_statistics.py
│   ├── attention_concentration.py
│   ├── entropy_distribution.py
│   ├── kv_norm_correlation.py
│   └── topk_vs_sampling_bias.py
└── extraction/         # CUDA/MLX extraction pipeline
    ├── extract_vectors.py
    ├── extraction_config.yaml
    ├── load_benchmarks.py
    ├── cuda_extract.py
    ├── mlx_extract.py
    └── save_utils.py
```

## Dependency Graph

```
core.py  ←── shared by everything below

python -m src.evaluation.run_evaluation
  └── evaluation/run_evaluation.py
        ├── algorithms/__init__.py (METHOD_REGISTRY)
        │     ├── idealized_methods.py
        │     ├── multiq_grouping.py
        │     └── kmeans_clustering.py
        │           └── base.py
        ├── evaluation/data_loader.py (.pt loading)
        ├── evaluation/evaluator.py
        └── evaluation/plotting.py

exploration/run_exploration.py
  ├── evaluation/data_loader.py
  ├── exploration/attention_concentration.py
  ├── exploration/entropy_distribution.py
  ├── exploration/kv_norm_correlation.py
  └── exploration/topk_vs_sampling_bias.py
```
