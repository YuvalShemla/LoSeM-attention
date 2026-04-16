# Gemma 4 Attention Extraction Plan

> Analysis & implementation plan for extending LoSeM-attention to Google's Gemma 4 open-source models.
> Generated: 2026-04-04

---

## Table of Contents

1. [Gemma 4 Model Family Overview](#1-gemma-4-model-family-overview)
2. [Attention Architecture Deep Dive](#2-attention-architecture-deep-dive)
3. [Flagship Model Recommendation](#3-flagship-model-recommendation)
4. [A100 80GB Feasibility Analysis](#4-a100-80gb-feasibility-analysis)
5. [Differences from Llama 3.1 8B Extraction](#5-differences-from-llama-31-8b-extraction)
6. [CUDA Extraction File Design](#6-cuda-extraction-file-design)
7. [Notebook Plan](#7-notebook-plan)
8. [Open Questions & Risks](#8-open-questions--risks)

---

## 1. Gemma 4 Model Family Overview

Google released four Gemma 4 variants (April 2026), all multimodal, Apache 2.0 licensed:

| Model | Total Params | Active Params | Layers | Context | Sliding Window | Type |
|-------|-------------|---------------|--------|---------|----------------|------|
| **Gemma 4 31B** | 32.7B | 32.7B (dense) | 60 | 256K | 1024 | Image+Text |
| **Gemma 4 26B-A4B** | 26.5B | 3.8B (MoE) | 30 | 256K | 1024 | Image+Text |
| **Gemma 4 E4B** | 8B (4.5B effective) | 4.5B | 42 | 128K | 512 | Any-to-Any (audio+image+text) |
| **Gemma 4 E2B** | 5.1B (2.3B effective) | 2.3B | 35 | 128K | 512 | Any-to-Any (audio+image+text) |

All models are **decoder-only** causal LMs despite their multimodal capabilities. For text-only attention extraction, we bypass the vision/audio encoders entirely and work with the text decoder only.

### Weight Sizes on Disk (bfloat16 safetensors)

| Model | Disk Size | bf16 VRAM (weights only) |
|-------|-----------|--------------------------|
| 31B | ~62.5 GB | ~62.5 GB |
| 26B-A4B (MoE) | ~48.1 GB | ~48.1 GB |
| E4B | ~16 GB (estimated) | ~16 GB |
| E2B | ~10 GB (estimated) | ~10 GB |

---

## 2. Attention Architecture Deep Dive

### 2.1 Hybrid Sliding Window + Global Attention

All Gemma 4 models use a **hybrid attention pattern** that alternates between two types of attention layers:

- **Sliding attention layers**: Attend only to a local window (512 or 1024 tokens). These are cheap and fast.
- **Full (global) attention layers**: Attend to the entire sequence. These are expensive but provide long-range context.

The pattern is fixed: every N sliding layers are followed by 1 global layer, and the **final layer is always global**.

#### Per-Model Layer Patterns

| Model | Pattern | Global Layer Indices | Ratio |
|-------|---------|---------------------|-------|
| **31B** | 5 sliding + 1 global (x10) | 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 | 10/60 global (17%) |
| **26B-A4B** | 5 sliding + 1 global (x5) | 5, 11, 17, 23, 29 | 5/30 global (17%) |
| **E4B** | 5 sliding + 1 global (x7) | 5, 11, 17, 23, 29, 35, 41 | 7/42 global (17%) |
| **E2B** | 4 sliding + 1 global (x7) | 4, 9, 14, 19, 24, 29, 34 | 7/35 global (20%) |

**Implication for attention extraction**: This is fundamentally different from Llama 3.1's uniform full attention. The sliding attention layers only compute attention over the local window, meaning:
- For our long-context attention approximation research, **only the global attention layers are interesting** — they're the ones that actually attend across the full sequence
- Sliding attention layers are trivially "solved" (attention is already sparse by design)
- We should extract Q/K/V from **global layers only**, or at minimum, distinguish between the two types

### 2.2 Per-Model Head Configuration

| Model | Q Heads | KV Heads | Head Dim | Global Head Dim | Global KV Heads | GQA Ratio |
|-------|---------|----------|----------|-----------------|-----------------|-----------|
| **31B** | 32 | 16 | 256 | 512 | 4 | 2:1 (sliding), 8:1 (global) |
| **26B-A4B** | 16 | 8 | 256 | 512 | ? (likely 4) | 2:1 (sliding), 4:1+ (global) |
| **E4B** | 8 | 2 | 256 | 512 | ? (likely 1-2) | 4:1 (sliding) |
| **E2B** | 8 | 1 | 256 | 512 | ? (likely 1) | 8:1 (sliding) |

**Critical architectural detail**: Global attention layers use a **different head dimension** (512) than sliding layers (256), and have **fewer KV heads** (`num_global_key_value_heads`). For the 31B model:
- Sliding layers: 32 Q-heads x 256d, 16 KV-heads x 256d
- Global layers: 32 Q-heads x 512d, 4 KV-heads x 512d

This means the Q/K/V projection matrices have **different shapes** depending on the layer type. The extraction code must handle both.

### 2.3 Dual RoPE Configuration (Proportional RoPE / p-RoPE)

Gemma 4 uses **two separate RoPE configurations** depending on layer type:

| Layer Type | RoPE Theta | RoPE Type | Partial Rotary Factor |
|------------|-----------|-----------|----------------------|
| **Sliding** | 10,000 | Standard (default) | 1.0 (full) |
| **Global** | 1,000,000 | Proportional | 0.25 (25% of dims) |

**Proportional RoPE (p-RoPE)**: Only 25% of the head dimensions receive rotary position embeddings in global layers. The remaining 75% are position-invariant. This dramatically reduces the memory needed for the position-dependent part of the KV cache at long contexts.

**Implications for extraction**:
- We must apply the **correct RoPE variant** per layer type
- For global layers, only `head_dim * 0.25 = 128 dims` (for the 512d global heads) get RoPE; the other 384 dims are untouched
- The `Q_raw` / `Q_rope` distinction from our Llama extraction maps naturally, but we need a third category: `Q_rope_partial` or similar, to indicate that only a subset of dimensions were rotated

### 2.4 Q/K/V Norm (Post-Projection RMSNorm)

Unlike Llama 3.1, Gemma 4 applies **RMSNorm to Q, K, and V after projection** (before attention):

```python
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
value_states = self.v_norm(value_states)
```

This is important because our attention approximation algorithms work on the **actual Q/K/V that enter the dot-product attention**. We must apply these norms in the extraction to get correct vectors.

### 2.5 Shared KV Cache (KV Sharing Across Layers)

Gemma 4 E4B has 18 shared KV layers (`num_kv_shared_layers: 18`). In these layers, the K and V projections are **not recomputed** — instead, K/V from an earlier layer of the same attention type are reused.

For the 31B model, this is not documented in the config, but the blog mentions "Shared KV Cache" as a feature. This means:
- Some layers don't have their own `k_proj` and `v_proj` weights
- They share the K/V tensors from a "donor" layer
- We need to identify which layers share and which are independent

**Impact on extraction**: For shared-KV layers, we extract Q but reuse the K/V from the donor layer. This must be tracked in metadata.

### 2.6 Per-Layer Embeddings (PLE) — E4B/E2B Only

The E-series models use Per-Layer Embeddings: each decoder layer has its own small embedding lookup table that provides layer-specific token identity signals alongside the residual stream. This doesn't directly affect attention extraction (it modifies the input to each layer), but it means the hidden states arriving at each attention layer have layer-specific conditioning.

### 2.7 Mixture-of-Experts (MoE) — 26B-A4B Only

The MoE architecture applies to the **FFN/MLP blocks**, not the attention layers. Each layer has 128 expert FFN blocks, of which 8 are activated per token (plus 1 shared expert). The attention mechanism itself is standard (same hybrid sliding+global pattern). This means attention extraction is unaffected by the MoE routing.

### 2.8 Final Logit Softcapping

The 26B-A4B model has `final_logit_softcapping: 30.0`, which caps logits before the final softmax. This may also apply to attention logits (like Gemma 2's attention logit softcapping). We need to check if Gemma 4 applies softcapping to attention scores — if so, our ground-truth attention computation must include it.

---

## 3. Flagship Model Recommendation

### 3.1 Best Model for Text Comprehension, Reasoning, and Math

**The Gemma 4 31B dense model is the clear flagship for quality.**

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|-----------|-----|---------|-----|-----|
| MMLU Pro (knowledge) | **85.2%** | 82.6% | 69.4% | 60.0% |
| AIME 2026 (math) | **89.2%** | 88.3% | 42.5% | 37.5% |
| GPQA Diamond (reasoning) | **84.3%** | 82.3% | 58.6% | 43.4% |
| BigBench Hard (reasoning) | **74.4%** | 64.8% | 33.1% | 21.9% |
| LiveCodeBench v6 (coding) | **80.0%** | 77.1% | 52.0% | 44.0% |
| MRCR v2 128K (long context) | **66.4%** | 44.1% | 25.4% | 19.1% |
| LMArena Score | **1452** | 1441 | — | — |

The 31B is dominant across all categories, especially on long-context tasks (MRCR v2: 66.4% vs 44.1% for the MoE).

### 3.2 Most Popular Model (Practical Usage)

Based on HuggingFace download counts (as of April 2026):
- **31B-it**: 287K downloads (most popular)
- **26B-A4B-it**: 133K downloads
- **E4B-it**: 108K downloads
- **E2B-it**: 90K downloads

The 31B is both the highest quality and most downloaded. It will likely be the model people use for serious text comprehension, reasoning, and math tasks.

### 3.3 Recommendation for Our Research

For attention extraction research, I recommend:

1. **Primary target: Gemma 4 31B** — flagship, most interesting attention patterns, 256K context, comparable to Llama 3.1 8B's role as "the model people actually use"
2. **Secondary target: Gemma 4 E4B** — fits easily on A100, provides contrast (fewer heads, smaller model, 128K context)
3. **Tertiary/optional: Gemma 4 26B-A4B** — interesting because MoE, but the attention is standard so it may not add much insight vs 31B

---

## 4. A100 80GB Feasibility Analysis

### 4.1 Memory Budget Breakdown

An A100-80GB has 80 GB of HBM2e. Our extraction pipeline needs:

| Component | Formula | Notes |
|-----------|---------|-------|
| Model weights | `params × 2 bytes` (bf16) | Dominant cost |
| Hidden state capture (hooks) | `layers_hooked × batch × seq × hidden_size × 2` | Kept on CPU in our pipeline |
| Q/K/V projection intermediates | `batch × seq × heads × head_dim × 2` | Per-layer, freed after |
| PyTorch CUDA overhead | ~1-2 GB | Allocator, kernels |
| Input tensors | Negligible | Token IDs are int32 |

### 4.2 Per-Model Feasibility

#### Gemma 4 31B (62.5 GB weights)

| Factor | Value |
|--------|-------|
| Weights (bf16) | ~62.5 GB |
| Remaining for activations | ~17.5 GB |
| Peak activation at 80K seq | ~2-4 GB (depends on attention implementation) |
| **Verdict** | **TIGHT but feasible on A100 80GB** |

The 31B model weights alone take ~62.5 GB of the 80 GB budget. With our optimization of skipping the LM head and disabling KV cache, we should have ~15-17 GB remaining. However:

- At 80K tokens with hybrid attention, the sliding attention layers only compute over 1024-token windows, so activation memory is much smaller than Llama's full attention
- Global attention layers still need full `[seq, seq]` attention matrices, but there are only 10 global layers out of 60
- **We can extract from global layers one at a time**, freeing memory between layers

**Risk**: If the model requires >62.5 GB with framework overhead, we may need:
- 4-bit quantization (GPTQ/AWQ) to reduce to ~16 GB weights → easily fits
- Or `device_map="auto"` with CPU offloading for some layers

**Alternative**: Load as **float8** (8 GB savings) or use the base model (not -it, same weights size but no system prompt overhead).

#### Gemma 4 26B-A4B MoE (48.1 GB weights)

| Factor | Value |
|--------|-------|
| Weights (bf16) | ~48.1 GB |
| Remaining for activations | ~31.9 GB |
| **Verdict** | **Comfortable fit on A100 80GB** |

Despite having 128 experts, the MoE weights are only 48 GB in bf16. With 32 GB remaining, there's plenty of room for activations even at long contexts.

**Caveat**: During forward pass, all 128 experts must be in memory even though only 8+1 are activated per token. The 48 GB is the total loaded weight.

#### Gemma 4 E4B (estimated ~16 GB weights)

| Factor | Value |
|--------|-------|
| Weights (bf16) | ~16 GB |
| Remaining for activations | ~64 GB |
| **Verdict** | **Easily fits on A100 80GB** |

Abundant headroom. Can extract at full 128K context with no memory pressure.

#### Gemma 4 E2B (estimated ~10 GB weights)

| Factor | Value |
|--------|-------|
| Weights (bf16) | ~10 GB |
| Remaining for activations | ~70 GB |
| **Verdict** | **Trivially fits on A100 80GB** |

### 4.3 Summary Table

| Model | Weights (bf16) | A100 80GB? | Max Context (est.) | Notes |
|-------|---------------|------------|---------------------|-------|
| **31B** | 62.5 GB | Tight fit | ~80K-100K | May need quantization for >100K context |
| **26B-A4B** | 48.1 GB | Comfortable | ~128K+ | MoE doesn't affect activation memory much |
| **E4B** | ~16 GB | Easy | Full 128K | Lots of headroom |
| **E2B** | ~10 GB | Trivial | Full 128K | Lots of headroom |

### 4.4 H100 Option

An H100 80GB has the same memory as A100 80GB but with faster HBM3 bandwidth. It would be equally tight for the 31B model. For the 31B to run comfortably at 256K context, we'd likely need:
- **H100 80GB + quantization** (4-bit: ~16 GB weights → plenty of room for 256K)
- **2x A100/H100** with tensor parallelism
- Or accept the 80K-100K context limit on a single A100 80GB in bf16

For our research (which uses tasks from 19K to 120K tokens), a single A100 80GB should work for the 31B model up to ~100K tokens in bf16, which covers most of our tasks.

---

## 5. Differences from Llama 3.1 8B Extraction

### 5.1 Architecture Comparison

| Feature | Llama 3.1 8B | Gemma 4 31B | Gemma 4 E4B |
|---------|-------------|-------------|-------------|
| Layers | 32 | 60 | 42 |
| Attention type | Uniform full | Hybrid (sliding + global) | Hybrid (sliding + global) |
| Q heads | 32 | 32 | 8 |
| KV heads | 8 | 16 (sliding) / 4 (global) | 2 |
| Head dim | 128 | 256 (sliding) / 512 (global) | 256 / 512 |
| RoPE | Standard (theta=500K) | Dual: standard (10K) + proportional (1M) | Dual: standard (10K) + proportional (1M) |
| Q/K/V norm | No | Yes (RMSNorm) | Yes (RMSNorm) |
| Shared KV | No | Possible | Yes (18 layers) |
| Hidden size | 4096 | 5376 | 2560 |
| Partial rotary | No (full dim) | Yes (25% for global) | Yes (25% for global) |

### 5.2 Key Code Changes Required

1. **Layer type detection**: Must distinguish sliding vs global layers and handle differently
2. **Different head dimensions per layer type**: Global layers use 512d heads, sliding use 256d
3. **Different KV head counts per layer type**: Global layers have fewer KV heads
4. **Dual RoPE**: Apply correct RoPE based on layer type (standard vs proportional)
5. **Partial rotary embedding**: Only rotate 25% of dimensions in global layers
6. **Post-projection Q/K/V normalization**: Apply RMSNorm after projection
7. **Shared KV tracking**: Identify which layers share KV and link to donors
8. **Model class**: `AutoModelForConditionalGeneration` (multimodal) or text-only submodel
9. **Model backbone path**: Likely `model.language_model.model.layers` instead of `model.model.layers`

---

## 6. CUDA Extraction File Design

### 6.1 File: `src/extraction/gemma4_extract.py`

The new extraction backend mirrors `cuda_extract.py` but handles Gemma 4's hybrid attention:

```
gemma4_extract.py
├── load_gemma4_model()          # Load HF model, identify text backbone path
├── get_layer_type()             # Return "sliding" or "global" for a layer index
├── get_layer_config()           # Return head count, head dim, KV heads for layer type
├── _get_rope_embeddings_dual()  # Handle dual RoPE (standard vs proportional)
├── _apply_partial_rotary()      # Apply RoPE to only 25% of dims (global layers)
├── _apply_qkv_norm()           # Apply per-head RMSNorm post-projection
├── extract_layer_qkv_gemma4()  # Main extraction function
└── identify_shared_kv_layers()  # Map shared-KV layers to their donors
```

### 6.2 Key Design Decisions

**A. Which layers to extract?**

We should focus on **global attention layers** since they're the ones doing long-range attention:
- 31B: layers 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 (10 global layers)
- E4B: layers 5, 11, 17, 23, 29, 35, 41 (7 global layers)

We should also extract 2-3 sliding layers for comparison (to show the contrast).

**B. Handling different head dimensions**

The extraction output `.pt` files need to handle:
- Global layers: Q shape `[seq, 512]`, K shape `[seq, 512]`, V shape `[seq, 512]`
- Sliding layers: Q shape `[seq, 256]`, K shape `[seq, 256]`, V shape `[seq, 256]`

This means tensors in different layers will have different dimensions. The evaluation code needs to handle this.

**C. RoPE handling**

For global layers with partial rotary (25% of 512 = 128 dims):
```
Q_rope[:, :128] = rotated Q[:, :128]   # Rotated with proportional RoPE
Q_rope[:, 128:] = Q[:, 128:]            # Untouched
```

Store both `Q_raw` and `Q_rope` as before, plus a metadata flag indicating partial rotary.

**D. Forward pass through text backbone**

```python
# For multimodal Gemma4:
# model = AutoModelForConditionalGeneration
# text backbone = model.language_model.model
# decoder layers = model.language_model.model.layers

# OR load text-only (if available):
# model = AutoModelForCausalLM.from_pretrained("google/gemma-4-31B")
# text backbone = model.model
# decoder layers = model.model.layers
```

Since we're doing text-only extraction, we should try loading the base model with `AutoModelForCausalLM` first. If that doesn't work for Gemma 4's architecture, use the multimodal class and access `model.language_model`.

### 6.3 Tensor Naming Convention

Extend the current convention to include layer-type metadata:

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `Q_rope_head{i}` | `[seq, head_dim]` | Q with RoPE (full or partial) |
| `Q_raw_head{i}` | `[seq, head_dim]` | Q before RoPE |
| `K_rope_kvhead{j}` | `[seq, head_dim]` | K with RoPE |
| `K_raw_kvhead{j}` | `[seq, head_dim]` | K before RoPE |
| `V_kvhead{j}` | `[seq, head_dim]` | Value |

Additionally, `metadata.json` must record:
- `layer_type`: "sliding" or "global"
- `head_dim`: 256 or 512
- `num_q_heads`: varies by layer type
- `num_kv_heads`: varies by layer type
- `rope_type`: "standard" or "proportional"
- `partial_rotary_factor`: 1.0 or 0.25
- `is_shared_kv`: bool
- `kv_donor_layer`: int (if shared)

### 6.4 Memory Optimization Strategy

For the 31B model on A100 80GB:

1. **Skip LM head** (same as Llama): forward through text backbone only
2. **Disable KV cache**: `use_cache=False`
3. **One-layer-at-a-time projection**: After capturing hidden states via hooks, compute Q/K/V for one layer, save to CPU, free GPU memory, then process the next layer
4. **Global-layers-only by default**: Only hook the 10 global layers (not all 60)
5. **Expandable segments**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
6. **Optional quantization fallback**: If bf16 doesn't fit, use 4-bit quantized weights with full-precision Q/K/V extraction

---

## 7. Notebook Plan

### 7.1 File: `notebooks/colab_extract_gemma4.ipynb`

Structure follows the existing `colab_extract_vectors.ipynb`:

**Cell 1: GPU Check**
```python
# Validate GPU type and memory
# A100 80GB required for 31B
# Any GPU with 20+ GB for E4B
```

**Cell 2: Setup & Install**
```python
# Clone repo, install transformers>=4.45 (Gemma 4 support)
# Install safetensors, accelerate
```

**Cell 3: Configuration**
```python
# Select model: "google/gemma-4-31B" or "google/gemma-4-E4B"
# Override extraction config for Gemma 4 architecture
model_configs = {
    "gemma-4-31B": {
        "num_layers": 60,
        "global_layers": [5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
        "sliding_layers": [0,1,2,3,4, 6,7,...],  # all others
        "num_q_heads": 32,
        "num_kv_heads_sliding": 16,
        "num_kv_heads_global": 4,
        "head_dim_sliding": 256,
        "head_dim_global": 512,
        "sliding_window": 1024,
        "context_window": 262144,
    },
    "gemma-4-E4B": {
        "num_layers": 42,
        "global_layers": [5, 11, 17, 23, 29, 35, 41],
        "num_q_heads": 8,
        "num_kv_heads": 2,
        "head_dim_sliding": 256,
        "head_dim_global": 512,
        "sliding_window": 512,
        "context_window": 131072,
    },
}
```

**Cell 4: Model Loading**
```python
# Load model in bf16 with device_map="auto"
# Identify text backbone path
# Verify memory usage
```

**Cell 5: Per-Task Extraction Loop**
```python
# For each of the 6 tasks:
#   1. Load benchmark examples
#   2. Tokenize and truncate
#   3. Scout pass (all global-layer heads)
#   4. Head selection by entropy percentile
#   5. Vectors pass (selected heads only)
#   6. Save .pt files + metadata
#   7. Background tar for download
```

**Cell 6: Download & Verification**
```python
# Display tar files ready for download
# Verify tensor shapes and metadata
```

### 7.2 Running the 6 Tasks

The same 6 tasks from our Llama extraction:

| Task | Tokens | Context Fit (31B, 256K)? | Context Fit (E4B, 128K)? |
|------|--------|--------------------------|--------------------------|
| math_calc | ~19K | Yes | Yes |
| code_run | ~75K | Yes | Yes |
| longbook_sum_eng | ~120K | Yes | Yes (barely) |
| kv_retrieval | ~100K+ | Yes | Yes (may truncate) |
| multi_doc_qa | ~61K | Yes | Yes |
| single_doc_qa | ~85K | Yes | Yes |

All tasks fit within both models' context windows. The 31B model's 256K context gives ample room.

### 7.3 Evaluation Config Updates

New `gemma4_evaluation_config.yaml`:

```yaml
model:
  hf_name: "google/gemma-4-31B"
  num_layers: 60
  attention_type: "hybrid"
  layer_types:
    sliding:
      num_q_heads: 32
      num_kv_heads: 16
      head_dim: 256
      rope_theta: 10000.0
      rope_type: "standard"
    global:
      num_q_heads: 32
      num_kv_heads: 4
      head_dim: 512
      rope_theta: 1000000.0
      rope_type: "proportional"
      partial_rotary_factor: 0.25
  global_layer_indices: [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
  sliding_window: 1024
  context_window: 262144
```

---

## 8. Open Questions & Risks

### 8.1 Unresolved Questions

1. **Does Gemma 4 apply attention logit softcapping?** The 26B-A4B config shows `final_logit_softcapping: 30.0`. If this applies to attention logits (like Gemma 2), our ground-truth attention computation `softmax(QK^T/sqrt(d))` needs modification to `softmax(softcap * tanh(QK^T/sqrt(d) / softcap))`.

2. **Exact shared-KV layer mapping for 31B**: The config for E4B shows `num_kv_shared_layers: 18`, but we need to verify which layers share with which donors for the 31B model.

3. **`num_global_key_value_heads` for E4B/E2B**: The 31B config shows `num_global_key_value_heads: 4`, but this value wasn't in the E4B/E2B configs fetched. Need to check the actual config.json more carefully.

4. **Transformer version requirement**: Gemma 4 requires `transformers >= 5.5.0.dev0` based on the E4B config. We need to ensure this is released / installable.

5. **Text-only loading**: Can we load `google/gemma-4-31B` with `AutoModelForCausalLM` (text-only)? Or must we use `AutoModelForConditionalGeneration` / `AutoModelForMultimodalLM` and access the text sub-model? The base (non-IT) models may support text-only loading.

### 8.2 Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| 31B doesn't fit A100 80GB | Medium | Use 4-bit quantization or CPU offload for some layers |
| Shared KV layers break our per-layer extraction assumption | Medium | Track donor layers in metadata; extract from donors only |
| Different head dims in global vs sliding layers break evaluation code | High | Refactor evaluator to accept variable head dimensions |
| Partial rotary factor affects attention approximation quality | Low | Extract both raw and rope vectors; evaluate both |
| HuggingFace transformers version not stable for Gemma 4 | Medium | Pin to specific commit if needed |
| Q/K/V norm changes attention distribution properties | Low | Just apply the norms; the vectors we extract are correct |

### 8.3 Implementation Priority

1. **Phase 1**: Get the 31B model loading and basic extraction working (global layers, correct shapes)
2. **Phase 2**: Handle dual RoPE, partial rotary, Q/K/V norms
3. **Phase 3**: Shared KV layer handling
4. **Phase 4**: Notebook with all 6 tasks
5. **Phase 5**: Run evaluation with existing algorithms on Gemma 4 vectors

### 8.4 Expected Findings

The hybrid attention architecture will likely show:
- **Global layers**: Similar attention patterns to Llama (long-range, diffuse attention) — our approximation algorithms are relevant
- **Sliding layers**: Trivially sparse (only 1024 tokens) — our algorithms are unnecessary
- **Larger head dimensions** (256/512 vs 128): May affect LSH and clustering algorithm performance since they operate in higher-dimensional spaces
- **Fewer KV heads in global layers**: With only 4 KV heads (31B global), each KV head serves 8 Q heads — the attention patterns per KV head may be more diverse/complex

---

## Appendix A: Full Architecture Configs

### Gemma 4 31B (Text Decoder)

```
hidden_size:              5376
num_hidden_layers:        60
num_attention_heads:      32
num_key_value_heads:      16       (sliding layers)
num_global_key_value_heads: 4      (global layers)
head_dim:                 256      (sliding)
global_head_dim:          512      (global)
intermediate_size:        21504
max_position_embeddings:  262144
sliding_window:           1024
vocab_size:               262144
layer_pattern:            5×sliding + 1×global (×10)
attention_bias:           false
rms_norm_eps:             1e-6
torch_dtype:              bfloat16
```

### Gemma 4 26B-A4B (Text Decoder, MoE)

```
hidden_size:              2816
num_hidden_layers:        30
num_attention_heads:      16
num_key_value_heads:      8
head_dim:                 256
global_head_dim:          512
intermediate_size:        2112
moe_intermediate_size:    704
num_experts:              128
top_k_experts:            8
shared_experts:           1
max_position_embeddings:  262144
sliding_window:           1024
vocab_size:               262144
layer_pattern:            5×sliding + 1×global (×5)
final_logit_softcapping:  30.0
torch_dtype:              bfloat16
```

### Gemma 4 E4B (Text Decoder, PLE)

```
hidden_size:              2560
num_hidden_layers:        42
num_attention_heads:      8
num_key_value_heads:      2
head_dim:                 256
global_head_dim:          512
intermediate_size:        10240
max_position_embeddings:  131072
sliding_window:           512
vocab_size:               262144
num_kv_shared_layers:     18
layer_pattern:            5×sliding + 1×global (×7)
torch_dtype:              bfloat16
```

### Gemma 4 E2B (Text Decoder, PLE)

```
hidden_size:              1536
num_hidden_layers:        35
num_attention_heads:      8
num_key_value_heads:      1
head_dim:                 256
global_head_dim:          512
intermediate_size:        6144
max_position_embeddings:  131072
sliding_window:           512
vocab_size:               262144
layer_pattern:            4×sliding + 1×global (×7)
torch_dtype:              bfloat16
```
