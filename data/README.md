# Data

Extracted attention vectors for experiments.

## Directory Structure

```
data/
├── benchmarks/          # Raw benchmark examples (JSON)
│   ├── infinitebench/
│   └── longbench_v2/
├── head_statistics/     # Phase 1 per-head profiles
│   └── llama3.1_8b/
└── vectors/             # .pt attention vectors
    └── llama3.1_8b/
        ├── all_heads/   # Phase 1: 1 example, all heads
        └── selected_heads/  # Phase 2: 3 heads, 10 ex
```

## .pt File Format (bfloat16)

One file per example per layer:
```
vectors/llama3.1_8b/all_heads/math_calc/ex_000/
  layer_00.pt
  layer_17.pt
  layer_19.pt
  layer_31.pt
  example.json
```

Each `.pt` contains a dict of tensors:
```python
{
    "Q_rope_head0": tensor [seq_len, 128] bf16,
    "Q_raw_head0": tensor [seq_len, 128] bf16,
    "K_rope_kvhead0": tensor [seq_len, 128] bf16,
    "K_raw_kvhead0": tensor [seq_len, 128] bf16,
    "V_kvhead0": tensor [seq_len, 128] bf16,
    # ... all heads for this layer
}
```

## Loading

```python
from src.experiment.data_loader import load_examples

for ex in load_examples(
    Path("data/vectors/llama3.1_8b"),
    task="math_calc", layer=17,
    head=0, kv_head=0,
):
    Q, K, V = ex["Q"], ex["K"], ex["V"]
```

See each subfolder's README for detailed schema.
