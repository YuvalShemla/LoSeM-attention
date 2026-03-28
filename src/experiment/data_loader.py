"""
Data loading for attention experiments (.pt format).

Loads per-layer bfloat16 .pt files from the extraction
pipeline. Returns standardized dicts per example.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Iterator


def load_pt_example(
    example_dir: Path,
    layer: int,
    head: Optional[int] = None,
    kv_head: Optional[int] = None,
) -> Dict:
    """
    Load one example's single layer from .pt file.

    Args:
        example_dir: path to ex_NNN/ directory
        layer: which layer to load
        head: Q head index (None = all heads)
        kv_head: KV head index (None = all)

    Returns dict with Q, K, V as float32 arrays,
    plus metadata if available.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch required for .pt loading"
        )

    pt_path = example_dir / f"layer_{layer:02d}.pt"
    if not pt_path.exists():
        return {}

    tensors = torch.load(
        pt_path, map_location="cpu",
        weights_only=True,
    )

    data = {}
    if head is not None:
        q_key = f"Q_rope_head{head}"
        if q_key in tensors:
            data["Q"] = (
                tensors[q_key].float().numpy()
            )
    if kv_head is not None:
        k_key = f"K_rope_kvhead{kv_head}"
        v_key = f"V_kvhead{kv_head}"
        if k_key in tensors:
            data["K"] = (
                tensors[k_key].float().numpy()
            )
        if v_key in tensors:
            data["V"] = (
                tensors[v_key].float().numpy()
            )

    meta_path = example_dir / "example.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    return data


def discover_examples(
    vectors_dir: Path,
    task: str,
    phase: str = "all_heads",
) -> List[Path]:
    """
    Find all example directories for a task.

    Returns sorted list of ex_NNN/ directories.
    """
    task_dir = vectors_dir / phase / task
    if not task_dir.exists():
        return []
    dirs = sorted(
        d for d in task_dir.iterdir()
        if d.is_dir() and d.name.startswith("ex_")
    )
    return dirs


def load_examples(
    vectors_dir: Path,
    task: str,
    layer: int,
    head: int,
    kv_head: int,
    phase: str = "selected_heads",
    max_examples: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Iterate examples for a task/layer/head combo.

    Yields dicts with Q, K, V, example_id, task,
    layer, head — compatible with the experiment runner.
    """
    ex_dirs = discover_examples(
        vectors_dir, task, phase,
    )
    if max_examples is not None:
        ex_dirs = ex_dirs[:max_examples]

    for ex_dir in ex_dirs:
        data = load_pt_example(
            ex_dir, layer=layer,
            head=head, kv_head=kv_head,
        )
        if "Q" not in data or "K" not in data:
            continue
        if "V" not in data:
            continue

        meta = data.get("metadata", {})
        ex_id = meta.get(
            "example_id", ex_dir.name,
        )

        yield {
            "Q": data["Q"],
            "K": data["K"],
            "V": data["V"],
            "example_id": ex_id,
            "sequence_length": data["Q"].shape[0],
            "task": task,
            "layer": layer,
            "head": head,
        }


def load_task_metadata(
    vectors_dir: Path,
    task: str,
    phase: str = "all_heads",
) -> Dict:
    """Load task-level metadata.json if present."""
    path = (
        vectors_dir / phase / task / "metadata.json"
    )
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def count_examples(
    vectors_dir: Path,
    task: str,
    phase: str = "all_heads",
) -> int:
    """Count example directories for a task."""
    return len(discover_examples(
        vectors_dir, task, phase,
    ))
