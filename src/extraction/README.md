# Data Extraction

Extracts Q, K, V attention vectors from Llama 3.1 8B
for use in attention approximation experiments.

## Requirements

**Hardware:** CUDA GPU or Apple Silicon (MLX).
 128K sequences are impractical on CPU.

**CUDA:**
```
pip install torch transformers accelerate datasets
```

**MLX (Apple Silicon):**
```
pip install mlx mlx-lm datasets
```

Both also need `pyyaml` (for config loading).

## How It Works

### Phase 1 — All-Heads Scout

First it will (Todo exaplin it will load the benchmarks, and add a section explaianing about the benchmarks Extracts ALL 32 Q heads + 8 KV heads across 4 layers,
for 1 example per task. Computes per-head attention
statistics (entropy, top-100 mass) to identify which
heads have the most/least concentrated attention.

```bash
python -m src.extraction.extract_vectors --phase 1
```

**Output:**
```
data/vectors/llama3.1_8b/all_heads/{task}/
  ex_000/
    layer_00.pt    # All heads, bfloat16
    layer_17.pt
    layer_19.pt
    layer_31.pt
    example.json   # Per-example metadata

data/head_statistics/llama3.1_8b/
  math_calc.json   # Per-head entropy stats
  code_run.json
  ...
```

### Phase 2 — Selected Heads

Uses Phase 1 statistics to pick 3 heads:
- **Highest entropy** (most diffuse attention)
- **Lowest entropy** (most concentrated)
- **Median entropy** (typical behavior)

Extracts only those heads, for 10 examples per task.

```bash
python -m src.extraction.extract_vectors --phase 2
```

**Output:**
```
data/vectors/llama3.1_8b/selected_heads/{task}/
  ex_000/ ... ex_009/
    layer_00.pt    # Only selected heads
    ...
```

## .pt File Format

Each `layer_XX.pt` contains a dict of bfloat16 tensors:

```python
{
    "Q_rope_head0":    # [seq_len, 128] with RoPE
    "Q_raw_head0":     # [seq_len, 128] without RoPE
    "K_rope_kvhead0":  # [seq_len, 128] with RoPE
    "K_raw_kvhead0":   # [seq_len, 128] without RoPE
    "V_kvhead0":       # [seq_len, 128] (no RoPE on V)
}
```

## Configuration

All parameters in `src/extraction/extraction_config.yaml`:
- `extraction.layers`: which layers to extract
- `extraction.max_length`: context window (131072)
- `extraction.phase1.examples_per_task`: default 1
- `extraction.phase2.examples_per_task`: default 10
- `model.*`: model name, head counts, RoPE theta

## Storage Estimates

| Phase | Total |
|-------|------:|
| Phase 1 (all heads, 1 ex/task) | ~46 GB |
| Phase 2 (3 heads, 10 ex/task) | ~79 GB |
| **Total** | **~125 GB** |

2 layers instead of 4 halves this to ~63 GB.

## Tasks

| Task | Source | ~Tokens |
|------|--------|--------:|
| math_calc | InfiniteBench | 19K |
| code_run | InfiniteBench | 75K |
| longbook_sum_eng | InfiniteBench | 161K |
| passkey | InfiniteBench | 127K |
| multi_doc_qa | LongBench v2 | 61K |
| single_doc_qa | LongBench v2 | 85K |

## Files

| File | Purpose |
|------|---------|
| `extract_vectors.py` | CLI + Phase 1/Phase 2 orchestration + head stats |
| `load_benchmarks.py` | HuggingFace dataset loaders |
| `cuda_extract.py` | CUDA backend (HF hooks) |
| `mlx_extract.py` | MLX backend (layer-by-layer) |
