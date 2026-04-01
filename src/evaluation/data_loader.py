"""
Data loading for attention evaluations (.pt format).

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
    use_rope: bool = True,
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
    q_prefix = "Q_rope" if use_rope else "Q_raw"
    k_prefix = "K_rope" if use_rope else "K_raw"

    if head is not None:
        q_key = f"{q_prefix}_head{head}"
        if q_key in tensors:
            data["Q"] = (
                tensors[q_key].detach().float().numpy()
            )
    if kv_head is not None:
        k_key = f"{k_prefix}_kvhead{kv_head}"
        v_key = f"V_kvhead{kv_head}"
        if k_key in tensors:
            data["K"] = (
                tensors[k_key].detach().float().numpy()
            )
        if v_key in tensors:
            data["V"] = (
                tensors[v_key].detach().float().numpy()
            )

    meta_path = example_dir / "example.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    return data


def _task_dir(
    vectors_dir: Path,
    task: str,
    phase: Optional[str] = None,
) -> Path:
    """Resolve task directory: flat or phase-based."""
    if phase is None:
        # Flat layout: {vectors_dir}/{task}/
        flat = vectors_dir / task
        if flat.exists():
            return flat
        # Fall back to old layout
        for p in ["selected_heads", "all_heads"]:
            old = vectors_dir / p / task
            if old.exists():
                return old
        return flat  # default to flat (will be empty)
    return vectors_dir / phase / task


def discover_examples(
    vectors_dir: Path,
    task: str,
    phase: Optional[str] = None,
) -> List[Path]:
    """
    Find all example directories for a task.

    Returns sorted list of ex_NNN/ directories.
    phase=None uses flat layout with fallback.
    """
    task_d = _task_dir(vectors_dir, task, phase)
    if not task_d.exists():
        return []
    dirs = sorted(
        d for d in task_d.iterdir()
        if d.is_dir() and d.name.startswith("ex_")
    )
    return dirs


def load_examples(
    vectors_dir: Path,
    task: str,
    layer: int,
    head: int,
    kv_head: int,
    phase: Optional[str] = None,
    max_examples: Optional[int] = None,
    use_rope: bool = True,
) -> Iterator[Dict]:
    """
    Iterate examples for a task/layer/head combo.

    Yields dicts with Q, K, V, example_id, task,
    layer, head — compatible with the evaluation runner.
    phase=None uses flat layout with fallback.
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
            use_rope=use_rope,
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
    phase: Optional[str] = None,
) -> Dict:
    """Load task-level metadata.json if present."""
    path = _task_dir(
        vectors_dir, task, phase,
    ) / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def count_examples(
    vectors_dir: Path,
    task: str,
    phase: Optional[str] = None,
) -> int:
    """Count example directories for a task."""
    return len(discover_examples(
        vectors_dir, task, phase,
    ))
