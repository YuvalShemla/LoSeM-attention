#!/usr/bin/env python3
"""
Unified extraction pipeline for attention vectors.

Auto-detects CUDA or MLX. Two phases:

  Phase 1: All heads, few examples -> head statistics.
  Phase 2: Selected heads, more examples.
           Supports "from_stats", "all", or explicit
           [layer, head] pairs via config.

Usage:
  python -m src.extraction.extract_vectors --phase 1
  python -m src.extraction.extract_vectors --phase 2
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from ..core import (
    softmax, entropy_nats, top_pct_mass,
    no_sink_local_mask, attention_stats_for_query,
)
from .load_benchmarks import (
    load_task, save_benchmark_examples, TASK_CONFIG,
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
        zero = {"entropy_full": 0.0,
                "entropy_no_sink_local": 0.0}
        for pct in top_pcts:
            label = f"top{pct}pct"
            zero[f"{label}_mass_full"] = 1.0
            zero[f"{label}_mass_no_sink_local"] = 1.0
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


def select_heads_for_phase2(
    stats: Dict,
    metric: str = "entropy_no_sink_local",
) -> List[Tuple[int, int]]:
    """
    Select 3 heads: max, min, median entropy.

    stats: {layer_N: {head_M: {metric: value}}}
    Returns list of (layer, head) tuples.
    """
    entries = []
    for layer_key, heads in stats.items():
        layer = int(layer_key.split("_")[1])
        for head_key, head_stats in heads.items():
            head = int(head_key.split("_")[1])
            val = head_stats.get(metric, 0.0)
            entries.append((layer, head, val))

    if not entries:
        return []

    entries.sort(key=lambda x: x[2])
    selected = []
    selected.append(
        (entries[0][0], entries[0][1])
    )
    mid = len(entries) // 2
    selected.append(
        (entries[mid][0], entries[mid][1])
    )
    selected.append(
        (entries[-1][0], entries[-1][1])
    )
    return selected


def save_head_statistics(
    stats: Dict, out_path: Path,
) -> None:
    """Save head statistics JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_head_statistics(path: Path) -> Dict:
    """Load head statistics JSON."""
    with open(path) as f:
        return json.load(f)


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


def _resolve_heads_for_task(
    phase_cfg, task, config, stats_dir, gqa,
):
    """
    Resolve which heads to extract for a task.

    Returns (extract_kwargs, label, pairs) or None.
    pairs: list of (layer, q_head, kv_head) or None.
    """
    heads = phase_cfg.get("heads", "all")

    if heads == "all":
        return {}, "all", None

    if heads == "from_stats":
        sp = stats_dir / f"{task}.json"
        if not sp.exists():
            print("    No stats file — run Phase 1")
            return None
        hs_cfg = config.get("head_statistics", {})
        metric = hs_cfg.get(
            "selection_metric",
            "entropy_no_sink_local",
        )
        sel = select_heads_for_phase2(
            load_head_statistics(sp),
            metric=metric,
        )
        if not sel:
            return None
        pairs = [(l, h, h // gqa) for l, h in sel]
        tgt_q = list(set(h for _, h, _ in pairs))
        tgt_kv = list(set(k for _, _, k in pairs))
        hdesc = ",".join(
            f"L{l}H{h}" for l, h, _ in pairs
        )
        return {
            "target_heads": tgt_q,
            "target_kv_heads": tgt_kv,
        }, hdesc, pairs

    pairs = [(l, h, h // gqa) for l, h in heads]
    tgt_q = sorted(set(h for _, h, _ in pairs))
    tgt_kv = sorted(set(k for _, _, k in pairs))
    hdesc = ",".join(
        f"L{l}H{h}" for l, h, _ in pairs
    )
    return {
        "target_heads": tgt_q,
        "target_kv_heads": tgt_kv,
    }, hdesc, pairs


def run(phase: str, config: dict, data_root: Path):
    """Single entry point for both extraction phases."""
    backend = detect_backend()
    res = _load_model(backend, config)
    if backend == "mlx":
        model, tokenizer, mx = res
    else:
        model, tokenizer = res
        mx = None

    ext = config["extraction"]
    layers = _resolve_layers(config)
    phase_cfg = config[f"phase{phase}"]
    n_ex = phase_cfg["examples_per_task"]
    tasks = config.get("tasks", list(TASK_CONFIG))
    gqa = (
        config["model"]["num_q_heads"]
        // config["model"]["num_kv_heads"]
    )

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

    subdir = (
        "all_heads" if phase == "1"
        else "selected_heads"
    )

    print(f"Phase {phase}: {len(tasks)} tasks, "
          f"{len(layers)} layers, {n_ex} ex/task")
    print(f"Backend: {backend}")

    all_stats = {}
    for task in tasks:
        print(f"\n  Task: {task}")

        resolved = _resolve_heads_for_task(
            phase_cfg, task, config, sdir, gqa,
        )
        if resolved is None:
            continue
        head_kwargs, hdesc, pairs = resolved
        print(f"    Heads: {hdesc}")

        extract_fn = _make_extract_fn(
            backend, model, tokenizer,
            layers, config, mx=mx,
            **head_kwargs,
        )

        examples = load_task(task)
        if not examples:
            print("    No examples, skipping")
            continue
        save_benchmark_examples(
            examples[:n_ex], task, bdir,
        )

        out = vdir / subdir / task
        lds = extract_and_save_examples(
            examples, n_ex, task, tokenizer,
            ext["max_length"], extract_fn, out,
            heads_label=hdesc,
        )

        if phase == "1" and lds:
            all_stats[task] = (
                _head_stats_from_data(lds[0], config)
            )

        save_task_metadata(
            task, TASK_CONFIG[task]["source"],
            config["model"]["hf_name"], layers,
            min(n_ex, len(examples)),
            out / "metadata.json",
            **({"selected_heads": [
                {"layer": l, "q_head": h,
                 "kv_head": k}
                for l, h, k in pairs
            ]} if pairs else {}),
        )

    if phase == "1":
        for task, stats in all_stats.items():
            save_head_statistics(
                stats, sdir / f"{task}.json",
            )

    print(f"\nPhase {phase} complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention vectors.",
    )
    parser.add_argument(
        "--phase", type=str, required=True,
        choices=["1", "2"],
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

    run(args.phase, config, data_root)


if __name__ == "__main__":
    main()
