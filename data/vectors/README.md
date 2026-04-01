# Attention Vectors

Per-layer bfloat16 `.pt` files containing Q, K, V attention vectors extracted from Llama 3.1 8B.

## Structure

Flat layout: one directory per task, with per-example subdirectories. Only the layers containing selected heads are saved (not all 32 layers).

```
vectors/
└── math_calc/
    ├── metadata.json
    ├── ex_000/
    │   ├── example.json
    │   ├── layer_01.pt
    │   ├── layer_06.pt
    │   ├── layer_12.pt
    │   ├── layer_26.pt
    │   └── layer_31.pt
    ├── ex_001/
    │   └── ...
    └── ex_004/
```

Each task has 5 examples, and each example contains only the layers where selected heads reside. The selected heads (5 per task) are chosen by effective entropy percentile during the extraction scout pass.

## .pt File Contents

Each `layer_XX.pt` contains only the selected Q and KV heads for that layer:

```python
{
    "Q_rope_head12":   # [seq_len, 128] with RoPE
    "Q_raw_head12":    # [seq_len, 128] without RoPE
    "K_rope_kvhead3":  # [seq_len, 128] with RoPE
    "K_raw_kvhead3":   # [seq_len, 128] without RoPE
    "V_kvhead3":       # [seq_len, 128] (no RoPE on V)
}
```

## Key Naming Convention

- `Q_rope_head{i}` — Q head i with RoPE applied
- `Q_raw_head{i}` — Q head i without RoPE
- `K_rope_kvhead{j}` — KV head j with RoPE
- `K_raw_kvhead{j}` — KV head j without RoPE
- `V_kvhead{j}` — V for KV head j (no RoPE on V)

GQA mapping: Q head i maps to KV head `i // 4`.

## Selected Heads

The extraction pipeline selects 5 heads per task by effective entropy percentile:

| Percentile | Meaning |
|-----------|---------|
| P0 | Most concentrated attention (lowest effective entropy) |
| P25 | Below-average diffusion |
| P50 | Median behavior |
| P75 | Above-average diffusion |
| P100 | Most diffuse attention (highest effective entropy) |

The selected heads and their layer/head indices are in `metadata.json`:

```json
{
  "selected_heads": [
    {"layer": 26, "q_head": 12, "kv_head": 3},
    {"layer": 31, "q_head": 24, "kv_head": 6},
    {"layer": 6,  "q_head": 4,  "kv_head": 1},
    {"layer": 12, "q_head": 14, "kv_head": 3},
    {"layer": 1,  "q_head": 2,  "kv_head": 0}
  ]
}
```

## Loading

```python
import torch
tensors = torch.load(
    "layer_26.pt", map_location="cpu",
    weights_only=True,
)
Q = tensors["Q_rope_head12"].float().numpy()
K = tensors["K_rope_kvhead3"].float().numpy()
V = tensors["V_kvhead3"].float().numpy()
```

Or use the data loader:
```python
from src.experiment.data_loader import load_examples
for ex in load_examples(
    "data/vectors", "math_calc",
    layer=26, head=12, kv_head=3,
):
    Q, K, V = ex["Q"], ex["K"], ex["V"]
```

## Metadata

**`metadata.json`** (per task):
```json
{
  "task": "math_calc",
  "source": "infinitebench",
  "model": "meta-llama/Meta-Llama-3.1-8B",
  "layers_extracted": [1, 6, 12, 26, 31],
  "n_examples": 5,
  "selected_heads": [...],
  "extraction_config": {
    "max_length": 125000,
    "store_raw_vectors": true,
    "store_rope_vectors": true
  }
}
```

**`example.json`** (per example):
```json
{
  "example_id": "math_calc_40",
  "sequence_length": 45018,
  "heads_extracted": "L26H12,L31H24,L6H4,L12H14,L1H2",
  "rope_included": true,
  "raw_included": true
}
```

## Storage

With per-layer head selection (only selected heads saved at their specific layers), each example is ~275 MB. Total for 5 examples per task: ~1.4 GB per task.
