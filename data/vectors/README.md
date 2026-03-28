# Attention Vectors

Per-layer bfloat16 `.pt` files containing Q, K, V
attention vectors extracted from Llama 3.1 8B.

## Structure

```
vectors/
└── llama3.1_8b/
    ├── all_heads/         # Phase 1: 1 example, all heads
    │   ├── math_calc/
    │   │   ├── metadata.json
    │   │   └── ex_000/
    │   │       ├── layer_00.pt
    │   │       ├── layer_17.pt
    │   │       ├── layer_19.pt
    │   │       ├── layer_31.pt
    │   │       └── example.json
    │   └── ...
    └── selected_heads/    # Phase 2: 10 examples, 3 heads
        ├── math_calc/
        │   ├── metadata.json
        │   └── ex_000/ ... ex_009/
        └── ...
```

## .pt File Contents

Each `layer_XX.pt` is a dict of bfloat16 tensors:

```python
{
    "Q_rope_head0":   # [seq_len, 128] with RoPE
    "Q_raw_head0":    # [seq_len, 128] without RoPE
    "K_rope_kvhead0": # [seq_len, 128] with RoPE
    "K_raw_kvhead0":  # [seq_len, 128] without RoPE
    "V_kvhead0":      # [seq_len, 128] (no RoPE on V)
    ...               # more heads
}
```

## Key Naming Convention

- `Q_rope_head{i}` — Q head i with RoPE applied
- `Q_raw_head{i}` — Q head i without RoPE
- `K_rope_kvhead{j}` — KV head j with RoPE
- `K_raw_kvhead{j}` — KV head j without RoPE
- `V_kvhead{j}` — V for KV head j (no RoPE on V)

GQA mapping: Q head i maps to KV head `i // 4`.

## Loading

```python
import torch
tensors = torch.load(
    "layer_17.pt", map_location="cpu",
    weights_only=True,
)
# Get Q head 0, K/V from KV head 0
Q = tensors["Q_rope_head0"].float().numpy()
K = tensors["K_rope_kvhead0"].float().numpy()
V = tensors["V_kvhead0"].float().numpy()
```

Or use the data loader:
```python
from src.experiment.data_loader import load_examples
for ex in load_examples(
    vectors_dir, "math_calc",
    layer=17, head=0, kv_head=0,
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
  "extraction_date": "2026-03-18",
  "layers_extracted": [0, 17, 19, 31],
  "n_examples": 1
}
```

**`example.json`** (per example):
```json
{
  "example_id": "math_calc_0",
  "sequence_length": 45018,
  "heads_extracted": "all",
  "rope_included": true,
  "raw_included": true
}
```

## Storage Estimates

| Phase | Storage |
|-------|-------:|
| Phase 1 (all heads, 1 ex/task) | ~46 GB |
| Phase 2 (3 heads, 10 ex/task) | ~79 GB |
| **Total** | **~125 GB** |
