#!/usr/bin/env python3
"""
Unified extraction pipeline for attention vectors.

Auto-detects CUDA or MLX. Runs in two phases:
  Phase 1: All heads, 1 example/task, head stats.
  Phase 2: 3 selected heads, 10 examples/task.

Usage:
  python -m data_extraction.extract_vectors --phase 1
  python -m data_extraction.extract_vectors --phase 2
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from .load_benchmarks import (
    load_task, save_benchmark_examples, TASK_CONFIG,
)
from .head_statistics import (
    compute_head_statistics, select_heads_for_phase2,
    save_head_statistics, load_head_statistics,
)
from .save_utils import (
    save_task_metadata, extract_and_save_examples,
)


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


def _head_stats_from_data(layer_data, config):
    """Compute per-head stats from extracted data."""
    n_q = config["model"]["num_q_heads"]
    gqa = n_q // config["model"]["num_kv_heads"]
    hdim = config["model"]["head_dim"]
    hs_cfg = config.get("head_statistics", {})
    n_queries = hs_cfg.get("n_queries", 10)
    n_sink = hs_cfg.get("n_sink_tokens", 1)
    local_w = hs_cfg.get("local_window", 1024)
    top_k = hs_cfg.get("top_k_for_mass", 100)

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
                Qf = Q.float().numpy()
                Kf = K.float().numpy()
            ls[f"head_{hi}"] = (
                compute_head_statistics(
                    Qf, Kf, hdim,
                    n_queries=n_queries,
                    n_sink=n_sink,
                    local_window=local_w,
                    top_k_for_mass=top_k,
                )
            )
        stats[f"layer_{li}"] = ls
    return stats


def run_phase1(config: dict, data_root: Path):
    """Phase 1: all heads, 1 example per task."""
    backend = detect_backend()
    res = _load_model(backend, config)
    if backend == "mlx":
        model, tokenizer, mx = res
    else:
        model, tokenizer = res
        mx = None

    ext = config["extraction"]
    layers = ext["layers"]
    n_ex = config["phase1"]["examples_per_task"]
    tasks = config.get("tasks", list(TASK_CONFIG))

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

    print(f"Phase 1: {len(tasks)} tasks, "
          f"{len(layers)} layers, all heads")
    print(f"Backend: {backend}")

    extract_fn = _make_extract_fn(
        backend, model, tokenizer,
        layers, config, mx=mx,
    )

    all_stats = {}
    for task in tasks:
        print(f"\n  Task: {task}")
        examples = load_task(task)
        if not examples:
            print("    No examples, skipping")
            continue
        save_benchmark_examples(
            examples[:n_ex], task, bdir,
        )
        out = vdir / "all_heads" / task
        lds = extract_and_save_examples(
            examples, n_ex, task, tokenizer,
            ext["max_length"], extract_fn, out,
        )
        if lds:
            all_stats[task] = (
                _head_stats_from_data(lds[0], config)
            )
        save_task_metadata(
            task, TASK_CONFIG[task]["source"],
            config["model"]["hf_name"], layers,
            min(n_ex, len(examples)),
            out / "metadata.json",
        )

    for task, stats in all_stats.items():
        save_head_statistics(
            stats, sdir / f"{task}.json",
        )
    print("\nPhase 1 complete.")


def run_phase2(config: dict, data_root: Path):
    """Phase 2: selected heads, 10 examples/task."""
    backend = detect_backend()
    res = _load_model(backend, config)
    if backend == "mlx":
        model, tokenizer, mx = res
    else:
        model, tokenizer = res
        mx = None

    ext = config["extraction"]
    layers = ext["layers"]
    n_ex = config["phase2"]["examples_per_task"]
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

    n_heads = config["phase2"].get(
        "n_heads_to_select", 3,
    )
    print(f"Phase 2: {n_ex} ex/task, "
          f"{n_heads} heads")
    print(f"Backend: {backend}")

    for task in tasks:
        print(f"\n  Task: {task}")
        sp = sdir / f"{task}.json"
        if not sp.exists():
            print("    Run Phase 1 first")
            continue
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
            continue
        pairs = [(l, h, h // gqa) for l, h in sel]
        print(f"    Heads: {sel}")

        tgt_q = list(set(h for _, h, _ in pairs))
        tgt_kv = list(set(k for _, _, k in pairs))

        extract_fn = _make_extract_fn(
            backend, model, tokenizer,
            layers, config, mx=mx,
            target_heads=tgt_q,
            target_kv_heads=tgt_kv,
        )

        examples = load_task(task)
        if not examples:
            continue
        save_benchmark_examples(
            examples[:n_ex], task, bdir,
        )
        hdesc = ",".join(
            f"L{l}H{h}" for l, h, _ in pairs
        )
        out = vdir / "selected_heads" / task
        extract_and_save_examples(
            examples, n_ex, task, tokenizer,
            ext["max_length"], extract_fn, out,
            heads_label=hdesc,
        )
        save_task_metadata(
            task, TASK_CONFIG[task]["source"],
            config["model"]["hf_name"], layers,
            min(n_ex, len(examples)),
            out / "metadata.json",
            selected_heads=[
                {"layer": l, "q_head": h,
                 "kv_head": k}
                for l, h, k in pairs
            ],
        )
    print("\nPhase 2 complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract attention vectors.",
    )
    parser.add_argument(
        "--phase", type=int, required=True,
        choices=[1, 2],
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
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Attention Vector Extraction")
    print("=" * 60)

    if args.phase == 1:
        run_phase1(config, data_root)
    else:
        run_phase2(config, data_root)


if __name__ == "__main__":
    main()
