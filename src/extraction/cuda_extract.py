"""
CUDA extraction backend for Llama models.

Loads HuggingFace model in bfloat16, uses forward hooks
to capture hidden states at target layers, then computes
Q/K/V projections with and without RoPE.
"""

import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def load_cuda_model(
    model_name: str,
) -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """Load HF model on CUDA in bfloat16."""
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def _get_rope_embeddings(
    rotary_emb, seq_len: int, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get cos/sin from the model's rotary embedding."""
    # Position IDs: [1, seq_len]
    position_ids = torch.arange(
        seq_len, device=device
    ).unsqueeze(0)
    # rotary_emb returns (cos, sin) each [1, seq, dim]
    # Create a dummy tensor to pass as value_states
    # (only used for device/dtype in some versions)
    dummy = torch.zeros(
        1, 1, seq_len, 1,
        device=device, dtype=torch.bfloat16,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    return cos, sin


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. Matches HF's apply_rotary_pos_emb."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def extract_layer_qkv_cuda(
    model,
    input_ids: torch.Tensor,
    layers: List[int],
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    store_raw: bool = True,
    store_rope: bool = True,
    target_heads: Optional[List[int]] = None,
    target_kv_heads: Optional[List[int]] = None,
    per_layer_heads: Optional[Dict[int, tuple]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract Q/K/V from target layers via hooks.

    Returns {layer_idx: {tensor_name: tensor}}.
    Tensor names follow the plan convention:
      Q_rope_head{i}, Q_raw_head{i},
      K_rope_kvhead{i}, K_raw_kvhead{i},
      V_kvhead{i}
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            captured[layer_idx] = args[0].detach().cpu()
        return hook_fn

    # Register pre-forward hooks on target layers
    decoder_layers = model.model.layers
    for li in layers:
        hook = decoder_layers[li].register_forward_hook(
            make_hook(li)
        )
        hooks.append(hook)

    # Forward pass through backbone only (no gradient).
    # Use model.model (LlamaModel) instead of model
    # (LlamaForCausalLM) to skip lm_head, which would
    # allocate [batch, seq, vocab_size] (~17 GB at 70K).
    # Disable KV cache to save ~8 GB across 32 layers.
    with torch.no_grad():
        model.model(input_ids, use_cache=False)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Extract Q/K/V from captured hidden states
    results = {}
    for li in layers:
        if li not in captured:
            continue
        h = captured.pop(li).to(device)
        layer = decoder_layers[li]
        attn = layer.self_attn
        ln = layer.input_layernorm

        h_normed = ln(h)

        # Raw projections (before RoPE)
        q_raw = attn.q_proj(h_normed)
        k_raw = attn.k_proj(h_normed)
        v_all = attn.v_proj(h_normed)

        # Reshape to heads: [batch, seq, heads, dim]
        q_raw = q_raw.view(
            1, seq_len, num_q_heads, head_dim
        )
        k_raw = k_raw.view(
            1, seq_len, num_kv_heads, head_dim
        )
        v_all = v_all.view(
            1, seq_len, num_kv_heads, head_dim
        )

        # Transpose to [batch, heads, seq, dim]
        q_raw = q_raw.transpose(1, 2)
        k_raw = k_raw.transpose(1, 2)
        v_all = v_all.transpose(1, 2)

        rotary = getattr(attn, "rotary_emb", None) \
            or getattr(model.model, "rotary_emb", None)
        cos, sin = _get_rope_embeddings(
            rotary, seq_len, device,
        )
        q_rope, k_rope = _apply_rotary_pos_emb(
            q_raw, k_raw, cos, sin,
        )

        # Select which heads to store
        if per_layer_heads and li in per_layer_heads:
            q_heads, kv_heads = per_layer_heads[li]
        elif target_heads is not None:
            q_heads = target_heads
            kv_heads = (
                target_kv_heads
                if target_kv_heads is not None
                else list(range(num_kv_heads))
            )
        else:
            q_heads = list(range(num_q_heads))
            kv_heads = list(range(num_kv_heads))

        layer_tensors = {}
        for hi in q_heads:
            if store_rope:
                # [seq, dim] in bfloat16
                layer_tensors[
                    f"Q_rope_head{hi}"
                ] = q_rope[0, hi].cpu().to(
                    torch.bfloat16
                )
            if store_raw:
                layer_tensors[
                    f"Q_raw_head{hi}"
                ] = q_raw[0, hi].cpu().to(
                    torch.bfloat16
                )

        for ki in kv_heads:
            if store_rope:
                layer_tensors[
                    f"K_rope_kvhead{ki}"
                ] = k_rope[0, ki].cpu().to(
                    torch.bfloat16
                )
            if store_raw:
                layer_tensors[
                    f"K_raw_kvhead{ki}"
                ] = k_raw[0, ki].cpu().to(
                    torch.bfloat16
                )
            layer_tensors[
                f"V_kvhead{ki}"
            ] = v_all[0, ki].cpu().to(torch.bfloat16)

        results[li] = layer_tensors

        # Free GPU memory
        del h, h_normed, q_raw, k_raw, v_all
        del q_rope, k_rope, cos, sin
        torch.cuda.empty_cache()

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return results
