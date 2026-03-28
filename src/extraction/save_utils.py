"""
Save utilities for extraction pipeline.

Handles .pt bfloat16 saving, JSON sidecar writing,
and per-task metadata generation.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def save_layer_pt(
    tensors: dict, path: Path,
) -> None:
    """Save tensor dict as .pt bfloat16 file."""
    import torch
    pt_data = {}
    for name, arr in tensors.items():
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr).to(
                torch.bfloat16
            )
        elif isinstance(arr, torch.Tensor):
            t = arr.to(torch.bfloat16)
        else:
            t = torch.tensor(
                arr, dtype=torch.bfloat16,
            )
        pt_data[name] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_data, path)


def save_example_json(
    example: dict,
    seq_len: int,
    heads_extracted: str,
    path: Path,
    backend: str = None,
    store_raw: bool = True,
    store_rope: bool = True,
) -> None:
    """Save per-example metadata sidecar."""
    meta = {
        "example_id": example["id"],
        "sequence_length": seq_len,
        "question": example["question"][:200],
        "answer": str(example["answer"])[:200],
        "context_chars": len(
            example.get("context", "")
        ),
        "heads_extracted": heads_extracted,
        "rope_included": store_rope,
        "raw_included": store_raw,
    }
    if backend:
        meta["backend"] = backend
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def save_task_metadata(
    task_name: str,
    source: str,
    model_name: str,
    layers: List[int],
    n_examples: int,
    out_path: Path,
    selected_heads: list = None,
    backend: str = None,
    head_statistics_params: dict = None,
    extraction_config: dict = None,
    example_ids: list = None,
) -> None:
    """Save per-task metadata JSON."""
    meta = {
        "task": task_name,
        "source": source,
        "model": model_name,
        "extraction_date": datetime.now().isoformat(),
        "layers_extracted": layers,
        "n_examples": n_examples,
    }
    if backend:
        meta["backend"] = backend
    if selected_heads is not None:
        meta["selected_heads"] = selected_heads
    if head_statistics_params:
        meta["head_statistics_params"] = (
            head_statistics_params
        )
    if extraction_config:
        meta["extraction_config"] = extraction_config
    if example_ids:
        meta["example_ids"] = example_ids
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


def extract_and_save_examples(
    candidates: list,
    extract_fn,
    out_dir: Path,
    heads_label: str = "all",
    backend: str = None,
    store_raw: bool = True,
    store_rope: bool = True,
) -> list:
    """
    Extract and save vectors for pre-tokenized candidates.

    candidates: list of (seq_len, example_dict, tokens)
    extract_fn(tokens) -> {layer_idx: {name: tensor}}
    Returns list of {"id", "seq_len"} for extracted
    examples. Aggressively frees GPU memory between
    examples to avoid OOM.
    """
    import gc
    import time
    import torch

    extracted = []
    for ei, (seq_len, ex, tokens) in enumerate(
        candidates
    ):
        print(f"    [{ei}] {ex['id']} "
              f"({seq_len} tok)")

        t0 = time.time()
        try:
            ld = extract_fn(tokens)
        except (torch.cuda.OutOfMemoryError,
                RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"    OOM at {seq_len} tok, "
                      f"skipping")
                continue
            raise
        print(f"    {time.time() - t0:.0f}s")

        edir = out_dir / f"ex_{ei:03d}"
        for li, tens in ld.items():
            save_layer_pt(
                tens, edir / f"layer_{li:02d}.pt",
            )
        save_example_json(
            ex, seq_len, heads_label,
            edir / "example.json",
            backend=backend,
            store_raw=store_raw,
            store_rope=store_rope,
        )

        del ld
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        extracted.append({
            "id": ex["id"], "seq_len": seq_len,
        })
    return extracted
