#!/usr/bin/env python3
"""
Unified extraction pipeline for attention vectors.

Auto-detects CUDA or MLX. Single-command workflow:

  Scout pass:  extract ALL heads for shortest example
               -> head statistics + head selection
  Vectors pass: extract ONLY selected heads for N examples

Usage:
  python -m src.extraction.extract_vectors
  python -m src.extraction.extract_vectors --config path/to/config.yaml
"""

import argparse
import gc
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ..core import (
    softmax, entropy_nats, top_pct_mass,
    nonlocal_mask, attention_stats_for_query,
)
from .load_benchmarks import (
    load_task, save_benchmark_examples,
    format_prompt, tokenize_and_truncate,
    TASK_CONFIG,
)
from .save_utils import (
    save_task_metadata, extract_and_save_examples,
)


# ── Head statistics ─────────────────────────────

def compute_head_statistics(
    Q_all: np.ndarray,
    K_all: np.ndarray,
    head_dim: int,
    n_queries: int = 10,
    n_sink: int = 1,
    local_window: int = 1024,
    top_pcts: List[int] = (1, 5),
) -> Dict[str, float]:
    """
    Average attention stats for one head over
    the last N query positions.
    """
    seq_len = Q_all.shape[0]
    n_q = min(n_queries, seq_len)
    if n_q <= 0:
        zero = {"full_entropy": 0.0,
                "nonlocal_entropy": 0.0}
        for pct in top_pcts:
            zero[f"full_top{pct}pct_mass"] = 1.0
            zero[f"nonlocal_top{pct}pct_mass"] = 1.0
        return zero

    start = max(0, seq_len - n_q)
    positions = list(range(start, seq_len))

    accum = {}
    for qpos in positions:
        stats = attention_stats_for_query(
            Q_all[qpos], K_all[:qpos + 1],
            head_dim, n_sink, local_window,
            top_pcts=top_pcts,
        )
        for k, v in stats.items():
            accum.setdefault(k, []).append(v)

    return {
        k: float(np.mean(v)) for k, v in accum.items()
    }


def select_heads_by_percentile(
    stats: Dict,
    metric: str = "nonlocal_entropy",
    percentiles: List[int] = (0, 25, 50, 75, 100),
) -> List[Tuple[int, int]]:
    """
    Select heads at entropy percentile positions.

    Ranks all (layer, head) pairs by metric value
    ascending. Returns deduplicated (layer, q_head)
    tuples.
    """
    entries = []
    for layer_key, heads in stats.items():
        layer = int(layer_key.split("_")[1])
        for head_key, head_stats in heads.items():
            head = int(head_key.split("_")[1])
            entries.append(
                (layer, head,
                 head_stats.get(metric, 0.0))
            )

    if not entries:
        return []

    entries.sort(key=lambda x: x[2])
    n = len(entries)
    seen = set()
    selected = []
    for pct in percentiles:
        idx = round(pct / 100 * (n - 1))
        idx = max(0, min(idx, n - 1))
        key = (entries[idx][0], entries[idx][1])
        if key not in seen:
            seen.add(key)
            selected.append(key)
    return selected


def save_head_statistics(
    stats: Dict, out_path: Path,
    metadata: Dict = None,
) -> None:
    """Save head statistics JSON with optional metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if metadata:
        data["metadata"] = metadata
    data.update(stats)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def load_head_statistics(path: Path) -> Dict:
    """Load head statistics JSON, stripping metadata."""
    with open(path) as f:
        data = json.load(f)
    data.pop("metadata", None)
    return data


# ── Backend detection ──────────────────────────

def detect_backend():
    """Auto-detect CUDA or MLX. Fail if neither."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:
        import mlx.core
        return "mlx"
    except ImportError:
        pass
    print("ERROR: Requires CUDA or MLX.")
    sys.exit(1)


def _load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model(backend: str, config: dict):
    if backend == "cuda":
        from .cuda_extract import load_cuda_model
        return load_cuda_model(
            config["model"]["hf_name"]
        )
    from .mlx_extract import load_mlx_model
    return load_mlx_model(
        config["model"]["mlx_name"]
    )


def _make_extract_fn(
    backend, model, tokenizer, layers,
    config, mx=None, **kwargs,
):
    """Return a callable(tokens) -> layer_data."""
    mcfg = config["model"]
    ext = config["extraction"]

    def fn(tokens):
        if backend == "cuda":
            import torch
            from .cuda_extract import (
                extract_layer_qkv_cuda,
            )
            ids = torch.tensor(
                [tokens], device="cuda",
            )
            return extract_layer_qkv_cuda(
                model, ids, layers,
                num_q_heads=mcfg["num_q_heads"],
                num_kv_heads=mcfg["num_kv_heads"],
                head_dim=mcfg["head_dim"],
                store_raw=ext["store_raw_vectors"],
                store_rope=ext[
                    "store_rope_vectors"
                ],
                **kwargs,
            )
        from .mlx_extract import (
            extract_layer_qkv_mlx,
        )
        return extract_layer_qkv_mlx(
            model, tokenizer, tokens, layers,
            num_q_heads=mcfg["num_q_heads"],
            num_kv_heads=mcfg["num_kv_heads"],
            head_dim=mcfg["head_dim"],
            rope_theta=mcfg["rope"]["theta"],
            store_raw=ext["store_raw_vectors"],
            store_rope=ext["store_rope_vectors"],
            mx=mx, **kwargs,
        )
    return fn


def _resolve_layers(config):
    """Resolve layers config: list of ints or 'all'."""
    raw = config["extraction"]["layers"]
    if raw == "all":
        return list(range(config["model"]["num_layers"]))
    return list(raw)


def _head_stats_from_data(layer_data, config):
    """Compute per-head stats from extracted data."""
    n_q = config["model"]["num_q_heads"]
    gqa = n_q // config["model"]["num_kv_heads"]
    hdim = config["model"]["head_dim"]
    hs_cfg = config.get("head_statistics", {})
    n_queries = hs_cfg.get("n_queries", 10)
    n_sink = hs_cfg.get("n_sink_tokens", 1)
    local_w = hs_cfg.get("local_window", 1024)
    top_pcts = hs_cfg.get("top_pct_for_mass", [1, 5])

    stats = {}
    for li, tens in layer_data.items():
        ls = {}
        for hi in range(n_q):
            qk = f"Q_rope_head{hi}"
            kk = f"K_rope_kvhead{hi // gqa}"
            if qk not in tens or kk not in tens:
                continue
            Q, K = tens[qk], tens[kk]
            if isinstance(Q, np.ndarray):
                Qf = Q.astype(np.float32)
                Kf = K.astype(np.float32)
            else:
                Qf = Q.detach().float().numpy()
                Kf = K.detach().float().numpy()
            ls[f"head_{hi}"] = (
                compute_head_statistics(
                    Qf, Kf, hdim,
                    n_queries=n_queries,
                    n_sink=n_sink,
                    local_window=local_w,
                    top_pcts=top_pcts,
                )
            )
        stats[f"layer_{li}"] = ls
    return stats


def _tokenize_and_sort(examples, tokenizer, max_len):
    """Pre-tokenize examples and sort by length."""
    candidates = []
    for ex in examples:
        prompt = format_prompt(ex)
        tokens = tokenize_and_truncate(
            tokenizer, prompt, max_len,
        )
        candidates.append(
            (len(tokens), ex, tokens)
        )
    candidates.sort(key=lambda x: x[0])
    return candidates


def run(config: dict, data_root: Path):
    """Unified extraction: scout -> select -> extract."""
    backend = detect_backend()
    res = _load_model(backend, config)
    if backend == "mlx":
        model, tokenizer, mx = res
    else:
        model, tokenizer = res
        mx = None

    ext = config["extraction"]
    layers = _resolve_layers(config)
    n_ex = ext.get("examples_per_task", 5)
    tasks = config.get("tasks", list(TASK_CONFIG))
    mcfg = config["model"]
    gqa = mcfg["num_q_heads"] // mcfg["num_kv_heads"]
    sel_cfg = config.get("head_selection", {})

    out_cfg = config.get("output", {})
    vdir = data_root / out_cfg.get(
        "vectors_subdir", "vectors/llama3.1_8b"
    )
    sdir = data_root / out_cfg.get(
        "head_stats_subdir",
        "head_statistics/llama3.1_8b",
    )
    bdir = data_root / out_cfg.get(
        "benchmarks_subdir", "benchmarks"
    )

    hs_cfg = config.get("head_statistics", {})
    hs_params = {
        "n_queries": hs_cfg.get("n_queries", 10),
        "n_sink_tokens": hs_cfg.get(
            "n_sink_tokens", 1,
        ),
        "local_window": hs_cfg.get(
            "local_window", 1024,
        ),
        "top_pct_for_mass": hs_cfg.get(
            "top_pct_for_mass", [1, 5],
        ),
    }

    mode = sel_cfg.get("mode", "auto")
    print(f"Extraction: {len(tasks)} tasks, "
          f"{len(layers)} layers, {n_ex} ex/task")
    print(f"Backend: {backend}, "
          f"head selection: {mode}")

    for task in tasks:
        print(f"\n  Task: {task}")

        examples = load_task(task)
        if not examples:
            print("    No examples, skipping")
            continue
        save_benchmark_examples(
            examples[:n_ex], task, bdir,
        )

        candidates = _tokenize_and_sort(
            examples, tokenizer,
            ext["max_length"],
        )

        # ── Scout pass (auto mode) ──
        if mode == "auto":
            print("    Scout pass: all heads")
            scout_fn = _make_extract_fn(
                backend, model, tokenizer,
                layers, config, mx=mx,
            )
            scout_seq_len, scout_ex, scout_tokens = (
                candidates[0]
            )
            print(f"    Scout: {scout_ex['id']} "
                  f"({scout_seq_len} tok)")

            import time
            t0 = time.time()
            layer_data = scout_fn(scout_tokens)
            print(f"    Scout extraction: "
                  f"{time.time() - t0:.0f}s")

            stats = _head_stats_from_data(
                layer_data, config,
            )

            del layer_data
            gc.collect()
            if backend == "cuda":
                import torch
                torch.cuda.empty_cache()

            selected = select_heads_by_percentile(
                stats,
                sel_cfg.get(
                    "metric", "nonlocal_entropy"
                ),
                sel_cfg.get(
                    "percentiles",
                    [0, 25, 50, 75, 100],
                ),
            )

            selected_meta = []
            for l, h in selected:
                lk = f"layer_{l}"
                hk = f"head_{h}"
                val = 0.0
                if lk in stats and hk in stats[lk]:
                    val = stats[lk][hk].get(
                        sel_cfg.get(
                            "metric",
                            "nonlocal_entropy",
                        ), 0.0,
                    )
                selected_meta.append({
                    "layer": l, "q_head": h,
                    "kv_head": h // gqa,
                    "nonlocal_entropy": val,
                })

            head_stats_meta = {
                "model": mcfg["hf_name"],
                "backend": backend,
                "task": task,
                "scout_examples": [scout_ex["id"]],
                "sequence_lengths": [scout_seq_len],
                "n_layers": mcfg["num_layers"],
                "n_q_heads": mcfg["num_q_heads"],
                "n_kv_heads": mcfg["num_kv_heads"],
                "head_dim": mcfg["head_dim"],
                "head_statistics_params": hs_params,
                "selected_heads": selected_meta,
                "extraction_date": (
                    datetime.now().isoformat()
                ),
            }
            save_head_statistics(
                stats, sdir / f"{task}.json",
                metadata=head_stats_meta,
            )
            print(f"    Selected {len(selected)} heads: "
                  + ", ".join(
                      f"L{l}H{h}"
                      for l, h in selected
                  ))
        else:
            explicit = sel_cfg.get("explicit", [])
            selected = [
                (l, h) for l, h in explicit
            ]
            selected_meta = [
                {"layer": l, "q_head": h,
                 "kv_head": h // gqa}
                for l, h in selected
            ]

        if not selected:
            print("    No heads selected, skipping")
            continue

        # ── Vectors pass: selected heads only ──
        pairs = [
            (l, h, h // gqa)
            for l, h in selected
        ]
        tgt_q = sorted(set(
            h for _, h, _ in pairs
        ))
        tgt_kv = sorted(set(
            k for _, _, k in pairs
        ))
        hdesc = ",".join(
            f"L{l}H{h}" for l, h, _ in pairs
        )
        print(f"    Vectors pass: {hdesc}")

        vec_fn = _make_extract_fn(
            backend, model, tokenizer,
            layers, config, mx=mx,
            target_heads=tgt_q,
            target_kv_heads=tgt_kv,
        )

        out = vdir / task
        extracted = extract_and_save_examples(
            examples, n_ex, task, tokenizer,
            ext["max_length"], vec_fn, out,
            heads_label=hdesc,
            backend=backend,
            store_raw=ext["store_raw_vectors"],
            store_rope=ext["store_rope_vectors"],
        )

        save_task_metadata(
            task, TASK_CONFIG[task]["source"],
            mcfg["hf_name"], layers,
            len(extracted),
            out / "metadata.json",
            backend=backend,
            head_statistics_params=hs_params,
            extraction_config={
                "max_length": ext["max_length"],
                "store_raw_vectors": ext[
                    "store_raw_vectors"
                ],
                "store_rope_vectors": ext[
                    "store_rope_vectors"
                ],
            },
            example_ids=[
                e["id"] for e in extracted
            ],
            selected_heads=[
                {"layer": l, "q_head": h,
                 "kv_head": k}
                for l, h, k in pairs
            ],
        )

    print("\nExtraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention vectors.",
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parent
            / "extraction_config.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(
            Path(__file__).parent.parent / "data"
        ),
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token for gated models "
        "(or set HF_TOKEN env var).",
    )
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    config = _load_config(Path(args.config))
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Attention Vector Extraction")
    print("=" * 60)

    run(config, data_root)


if __name__ == "__main__":
    main()
