# Data Extraction

Extracts Q, K, V attention vectors from Llama 3.1 8B for use in attention approximation evaluations.

## What We Extract

For each task we extract the **raw and RoPE-applied Q, K, V projections** from every transformer layer, for a set of selected attention heads. These vectors are the inputs to the attention mechanism — having them lets us replay and approximate attention offline without running the full model.

Each example produces one `.pt` file per layer containing bfloat16 tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `Q_rope_head{i}` | `[seq_len, 128]` | Query after RoPE rotation |
| `Q_raw_head{i}` | `[seq_len, 128]` | Query before RoPE (pre-rotation) |
| `K_rope_kvhead{j}` | `[seq_len, 128]` | Key after RoPE rotation |
| `K_raw_kvhead{j}` | `[seq_len, 128]` | Key before RoPE |
| `V_kvhead{j}` | `[seq_len, 128]` | Value (RoPE is not applied to V) |

RoPE vectors are used for attention computation; raw vectors are useful for clustering and similarity analysis since RoPE rotations spread keys into position-dependent subspaces.

## How It Works

Single command, two internal passes per task:

```bash
python -m src.extraction.extract_vectors
```

1. **Scout pass** — runs the model on the shortest example with ALL 1024 heads (32 layers x 32 Q heads). No `.pt` files are saved. Computes per-head attention statistics (entropy, top-% mass) to characterize each head's attention pattern.

2. **Head selection** — ranks all heads by `effective_entropy` (entropy computed after excluding sink tokens and the local window) and picks heads at configurable percentile positions. This gives a representative spread from most concentrated to most diffuse attention.

3. **Vectors pass** — runs the model again on N examples, extracting only the selected heads. Saves `.pt` files + metadata. GPU memory is aggressively freed between examples to avoid OOM.

## Requirements

**Hardware:** CUDA GPU or Apple Silicon (MLX). 128K sequences require an A100 (40GB+).

```bash
# CUDA
pip install torch transformers accelerate datasets pyyaml

# MLX (Apple Silicon)
pip install mlx mlx-lm datasets pyyaml
```

## Configuration

Everything is controlled by `extraction_config.yaml`. The key sections:

### Tasks

The `tasks` list controls which benchmarks to extract. Each task maps to a HuggingFace dataset via `task_sources`:

```yaml
tasks:
  - math_calc            # 19K tok, arithmetic sums
  - code_run             # 75K tok, predict code output
  - longbook_sum_eng     # 120K tok, book summarization
  - passkey              # 127K tok, passkey retrieval
  - multi_doc_qa         # 61K tok, multi-document QA
  - single_doc_qa        # 85K tok, single-document QA
```

To extract a subset, just comment out tasks you don't need.

### Extraction parameters

```yaml
extraction:
  examples_per_task: 5       # how many examples per task
  max_length: 131072         # max tokens (128K context)
  store_raw_vectors: true    # include pre-RoPE Q and K
  store_rope_vectors: true   # include post-RoPE Q and K
  layers: "all"              # "all" or e.g. [0, 17, 19, 31]
```

- **`examples_per_task`** — examples are sorted shortest-first. The scout pass always uses the first (shortest) example; the vectors pass extracts this many.
- **`layers`** — `"all"` extracts all 32 layers. Use a list like `[0, 17, 31]` to save disk/time.
- **`store_raw_vectors` / `store_rope_vectors`** — set either to `false` to halve storage. Evaluations use RoPE vectors; exploration scripts sometimes use raw vectors.

### Head selection

```yaml
head_selection:
  mode: "auto"                              # or "explicit"
  metric: "effective_entropy"                # ranking metric
  percentiles: [0, 25, 50, 75, 100]         # -> 5 heads
```

- **`mode: "auto"`** — runs the scout pass, computes stats, selects heads by percentile. This is the default.
- **`mode: "explicit"`** — skips the scout pass entirely. Provide exact heads:
  ```yaml
  head_selection:
    mode: "explicit"
    explicit: [[17, 5], [19, 12], [31, 0]]  # [layer, q_head] pairs
  ```
- **`metric`** — which statistic to rank heads by. `effective_entropy` excludes sink and local window effects to capture effective attention behavior. Alternatives: `full_entropy`, `effective_top1pct_mass`, etc.
- **`percentiles`** — which positions along the ranked head list to pick. `[0, 50, 100]` would give 3 heads (min, median, max).

### Head statistics (scout pass)

```yaml
head_statistics:
  n_queries: 10            # average stats over last N query positions
  exclude_sink_token: true  # exclude position 0 (attention sink)
  local_window: 1024       # positions excluded as "local"
  top_pct_for_mass: [1, 5] # compute top-1% and top-5% mass
```

These control how the per-head statistics are computed during the scout pass. `exclude_sink_token` and `local_window` define which tokens are excluded from the effective entropy computation.

### Output paths

```yaml
output:
  vectors_subdir: "vectors"
  head_stats_subdir: "head_statistics/llama3.1_8b"
  benchmarks_subdir: "benchmarks"
```

All paths are relative to `--data-root` (default: `data/`).

### CLI options

```bash
python -m src.extraction.extract_vectors \
  --config path/to/config.yaml \   # default: extraction_config.yaml
  --data-root path/to/data/ \      # default: data/
  --hf-token YOUR_TOKEN            # or set HF_TOKEN env var
```

## Output Structure

```
data/
├── vectors/
│   ├── math_calc/
│   │   ├── ex_000/
│   │   │   ├── layer_01.pt      # selected heads only, bfloat16
│   │   │   ├── layer_06.pt
│   │   │   ├── layer_26.pt
│   │   │   └── example.json     # example metadata (id, seq_len, heads)
│   │   ├── ex_001/ ...
│   │   └── metadata.json        # task metadata + selected_heads list
│   ├── code_run/ ...
│   └── ...
├── head_statistics/llama3.1_8b/
│   ├── math_calc.json           # per-head stats + selection info
│   └── per_example/             # optional per-example stats
└── benchmarks/
    ├── math_calc.json           # raw benchmark examples
    └── ...
```

## Tasks

| Task | Benchmark | Description | ~Tokens |
|------|-----------|-------------|--------:|
| `math_calc` | InfiniteBench | Sum arithmetic expressions | 19K |
| `code_run` | InfiniteBench | Predict code output | 75K |
| `longbook_sum_eng` | InfiniteBench | Summarize a book | 120K |
| `passkey` | InfiniteBench | Retrieve a passkey from noise | 127K |
| `multi_doc_qa` | LongBench v2 | QA across multiple documents | 61K |
| `single_doc_qa` | LongBench v2 | QA on a single long document | 85K |

Additional tasks from both benchmarks are available — see the bottom of `extraction_config.yaml` for the full list.

## Files

| File | Purpose |
|------|---------|
| `extract_vectors.py` | CLI + unified extraction orchestration + head stats |
| `extraction_config.yaml` | All extraction parameters |
| `load_benchmarks.py` | HuggingFace dataset loaders + prompt formatting |
| `save_utils.py` | .pt saving, JSON sidecars, per-example loop |
| `cuda_extract.py` | CUDA backend (HF forward hooks) |
| `mlx_extract.py` | MLX backend (layer-by-layer) |
