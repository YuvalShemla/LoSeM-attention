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

Single-command pipeline per task:

1. **Scout pass** — extracts ALL 32 Q heads + 8 KV heads
   for the shortest example. Computes per-head attention
   statistics (entropy, top-% mass). No .pt files saved.

2. **Head selection** — picks 5 heads at percentile
   positions [0, 25, 50, 75, 100] of `nonlocal_entropy`.

3. **Vectors pass** — extracts only the selected heads
   for N examples per task (default 5). Saves .pt files.

```bash
python -m src.extraction.extract_vectors
```

**Output:**
```
data/vectors/llama3.1_8b/{task}/
  ex_000/ ... ex_004/
    layer_00.pt    # Only selected heads, bfloat16
    layer_17.pt
    ...
    example.json   # Per-example metadata
  metadata.json    # Task metadata + selected_heads

data/head_statistics/llama3.1_8b/
  math_calc.json   # Per-head entropy stats + selection info
  code_run.json
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
- `extraction.examples_per_task`: default 5
- `extraction.layers`: which layers to extract
- `extraction.max_length`: context window (131072)
- `head_selection.mode`: "auto" or explicit pairs
- `head_selection.metric`: ranking metric (default `nonlocal_entropy`)
- `head_selection.percentiles`: which percentile positions to select
- `model.*`: model name, head counts, RoPE theta

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
| `extract_vectors.py` | CLI + unified extraction orchestration + head stats |
| `load_benchmarks.py` | HuggingFace dataset loaders |
| `cuda_extract.py` | CUDA backend (HF hooks) |
| `mlx_extract.py` | MLX backend (layer-by-layer) |
