"""
Load self-study training queries from cartridge vectors.

Aggregates Q vectors from all 1000 self-study conversations
into a single training query matrix for use with algorithms
that accept external training queries.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional


def load_self_study_train_queries(
    cartridge_vectors_dir: Path,
    layer: int,
    q_head: int,
    kv_head: int,
    max_conversations: int = 1000,
    use_rope: bool = True,
) -> np.ndarray:
    """
    Load Q vectors from all self-study conversations for one head.

    Returns Q_train: [total_qa_tokens, head_dim] float32 numpy array.
    With 1000 conversations of ~286 tokens each, this is ~286K vectors.
    """
    q_key = f"Q_rope_head{q_head}" if use_rope else f"Q_raw_head{q_head}"
    conv_dir = Path(cartridge_vectors_dir) / "conversations"

    if not conv_dir.exists():
        raise FileNotFoundError(f"No conversations directory at {conv_dir}")

    conv_dirs = sorted(
        d for d in conv_dir.iterdir()
        if d.is_dir() and d.name.startswith("conv_")
    )[:max_conversations]

    if not conv_dirs:
        raise FileNotFoundError(f"No conv_NNNN directories in {conv_dir}")

    all_q = []
    pt_name = f"layer_{layer:02d}.pt"

    for cd in conv_dirs:
        pt_path = cd / pt_name
        if not pt_path.exists():
            continue
        tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
        if q_key in tensors:
            all_q.append(tensors[q_key].float().numpy())

    if not all_q:
        raise FileNotFoundError(
            f"No Q vectors found for L{layer} H{q_head} in {conv_dir}"
        )

    result = np.concatenate(all_q, axis=0)
    return result
