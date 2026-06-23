# Data

Extracted attention vectors for evaluations.

## Directory Structure

```
data/
├── benchmarks/          # Raw benchmark examples (JSON)
│   ├── infinitebench/
│   └── longbench_v2/
├── head_statistics/     # Per-head attention profiles
│   └── llama3.1_8b/
│       ├── math_calc.json
│       └── per_example/
└── vectors/             # .pt attention vectors (flat layout)
    ├── math_calc/
    │   ├── metadata.json
    │   ├── ex_000/
    │   │   ├── layer_26.pt
    │   │   └── example.json
    │   └── ...
    └── ...
```

## .pt File Format (bfloat16)

One file per example per layer (only layers with selected heads):
```
vectors/math_calc/ex_000/
  layer_01.pt
  layer_06.pt
  layer_12.pt
  layer_26.pt
  layer_31.pt
  example.json
```

Each `.pt` contains a dict of tensors for the selected heads at that layer:
```python
{
    "Q_rope_head12": tensor [seq_len, 128] bf16,
    "Q_raw_head12": tensor [seq_len, 128] bf16,
    "K_rope_kvhead3": tensor [seq_len, 128] bf16,
    "K_raw_kvhead3": tensor [seq_len, 128] bf16,
    "V_kvhead3": tensor [seq_len, 128] bf16,
}
```

## Loading

```python
from examples.kvcache.vector_data import iter_selected_heads

for ex in iter_selected_heads("data/vectors", "math_calc"):
    Q, K, V = ex["Q"], ex["K"], ex["V"]
```

See each subfolder's README for detailed schema.
