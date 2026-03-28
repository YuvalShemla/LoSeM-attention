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
        "rope_included": True,
        "raw_included": True,
    }
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
    if selected_heads is not None:
        meta["selected_heads"] = selected_heads
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


def extract_and_save_examples(
    examples: list,
    n_examples: int,
    task: str,
    tokenizer,
    max_len: int,
    extract_fn,
    out_dir: Path,
    heads_label: str = "all",
) -> list:
    """
    Shared loop: tokenize, extract, save .pt + JSON.

    extract_fn(tokens) -> {layer_idx: {name: tensor}}
    Returns list of layer_data dicts (for stats).
    """
    from .load_benchmarks import (
        format_prompt, tokenize_and_truncate,
    )
    import time
    import torch

    # Pre-tokenize all examples and sort by length
    # (shortest first) so we maximize successful
    # extractions before hitting memory limits.
    candidates = []
    for ex in examples:
        prompt = format_prompt(ex)
        tokens = tokenize_and_truncate(
            tokenizer, prompt, max_len,
        )
        candidates.append((len(tokens), ex, tokens))
    candidates.sort(key=lambda x: x[0])

    all_ld = []
    ei = 0
    for seq_len, ex, tokens in candidates:
        if ei >= n_examples:
            break
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
        )
        all_ld.append(ld)
        ei += 1
    return all_ld
