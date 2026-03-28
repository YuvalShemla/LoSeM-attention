"""
MLX extraction backend for Llama models on Apple Silicon.

Loads mlx-community model in bf16, does manual layer-by-
layer forwarding with h snapshots at target layers, then
computes Q/K/V projections with and without RoPE.

Memory management: converts large MLX arrays to numpy
in chunks to avoid Metal buffer exhaustion.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _import_mlx():
    """Import MLX with clear error on failure."""
    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_load
        return mx, mlx_load
    except ImportError:
        raise ImportError(
            "MLX required: pip install mlx mlx-lm"
        )


def load_mlx_model(model_name: str):
    """Load MLX model and tokenizer."""
    mx, mlx_load = _import_mlx()
    model, tokenizer = mlx_load(model_name)
    return model, tokenizer, mx


def _to_numpy_chunked(
    arr, mx, chunk_rows: int = 4096,
) -> np.ndarray:
    """
    Convert MLX bfloat16 array to float32 numpy
    in chunks to limit Metal buffer pressure.
    """
    n = arr.shape[0]
    if n <= chunk_rows:
        mx.eval(arr)
        return np.array(
            arr.astype(mx.float32), copy=False
        )

    parts = []
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = arr[start:end].astype(mx.float32)
        mx.eval(chunk)
        parts.append(np.array(chunk, copy=False))
    return np.concatenate(parts, axis=0)


def _apply_rope_mlx(
    q, k, cos_cache, sin_cache, mx,
):
    """Apply RoPE using precomputed cos/sin."""
    def rotate_half(x):
        d2 = x.shape[-1] // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        return mx.concatenate([-x2, x1], axis=-1)

    q_embed = q * cos_cache + rotate_half(q) * sin_cache
    k_embed = k * cos_cache + rotate_half(k) * sin_cache
    return q_embed, k_embed


def _compute_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    mx,
):
    """Compute RoPE cos/sin cache matching HF Llama."""
    dim = head_dim
    inv_freq = 1.0 / (
        theta ** (
            np.arange(0, dim, 2, dtype=np.float32)
            / dim
        )
    )
    inv_freq_mx = mx.array(inv_freq)
    positions = mx.arange(seq_len).astype(mx.float32)
    # [seq_len, dim//2]
    freqs = mx.outer(positions, inv_freq_mx)
    # [seq_len, dim] — repeat for full dimension
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos_cache = mx.cos(emb)
    sin_cache = mx.sin(emb)
    return cos_cache, sin_cache


def extract_layer_qkv_mlx(
    model,
    tokenizer,
    input_ids: list,
    layers: List[int],
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    rope_theta: float = 500000.0,
    store_raw: bool = True,
    store_rope: bool = True,
    target_heads: Optional[List[int]] = None,
    target_kv_heads: Optional[List[int]] = None,
    mx=None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract Q/K/V from target layers.

    Does layer-by-layer forwarding, saving h
    snapshots at target layers. Computes Q/K/V
    projections and RoPE from snapshots.

    Returns {layer_idx: {name: numpy_array}}.
    Arrays are float32 numpy (converted from bf16).
    """
    if mx is None:
        mx, _ = _import_mlx()

    # Get model internals
    backbone = model.model
    embed_tokens = backbone.embed_tokens
    decoder_layers = backbone.layers
    final_norm = backbone.norm

    # Embed input
    x = mx.array(input_ids)[None, :]
    h = embed_tokens(x)
    mx.eval(h)

    seq_len = h.shape[1]
    layer_set = set(layers)
    snapshots = {}

    # Create causal attention mask (required for
    # correct hidden state propagation)
    import mlx.nn as nn
    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        seq_len
    ).astype(h.dtype)
    mx.eval(mask)

    # Layer-by-layer forward
    for li, layer in enumerate(decoder_layers):
        h = layer(h, mask=mask)
        mx.eval(h)
        if li in layer_set:
            snapshots[li] = h

    # Precompute RoPE cache
    cos_cache, sin_cache = _compute_rope_cache(
        seq_len, head_dim, rope_theta, mx,
    )
    mx.eval(cos_cache)
    mx.eval(sin_cache)

    q_heads_list = (
        target_heads if target_heads is not None
        else list(range(num_q_heads))
    )
    kv_heads_list = (
        target_kv_heads
        if target_kv_heads is not None
        else list(range(num_kv_heads))
    )

    results = {}
    for li in layers:
        if li not in snapshots:
            continue
        h_snap = snapshots[li]
        layer = decoder_layers[li]
        attn = layer.self_attn

        # Apply input layernorm
        h_normed = layer.input_layernorm(h_snap)

        # Raw projections
        q_all = attn.q_proj(h_normed)
        k_all = attn.k_proj(h_normed)
        v_all = attn.v_proj(h_normed)

        # Reshape: [1, seq, heads, dim]
        q_all = q_all.reshape(
            1, seq_len, num_q_heads, head_dim
        )
        k_all = k_all.reshape(
            1, seq_len, num_kv_heads, head_dim
        )
        v_all = v_all.reshape(
            1, seq_len, num_kv_heads, head_dim
        )

        # Apply RoPE
        # cos/sin: [seq, dim] -> broadcast
        q_rope, k_rope = _apply_rope_mlx(
            q_all, k_all,
            cos_cache[None, :, None, :],
            sin_cache[None, :, None, :],
            mx,
        )
        mx.eval(q_rope)
        mx.eval(k_rope)

        layer_tensors = {}
        for hi in q_heads_list:
            if store_rope:
                layer_tensors[
                    f"Q_rope_head{hi}"
                ] = _to_numpy_chunked(
                    q_rope[0, :, hi, :], mx,
                )
            if store_raw:
                layer_tensors[
                    f"Q_raw_head{hi}"
                ] = _to_numpy_chunked(
                    q_all[0, :, hi, :], mx,
                )

        for ki in kv_heads_list:
            if store_rope:
                layer_tensors[
                    f"K_rope_kvhead{ki}"
                ] = _to_numpy_chunked(
                    k_rope[0, :, ki, :], mx,
                )
            if store_raw:
                layer_tensors[
                    f"K_raw_kvhead{ki}"
                ] = _to_numpy_chunked(
                    k_all[0, :, ki, :], mx,
                )
            layer_tensors[
                f"V_kvhead{ki}"
            ] = _to_numpy_chunked(
                v_all[0, :, ki, :], mx,
            )

        results[li] = layer_tensors

    return results
