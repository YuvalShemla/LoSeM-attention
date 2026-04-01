"""
End-to-end smoke test on synthetic .pt data.

Creates temporary .pt files mimicking the extraction
pipeline output, writes a minimal config, then runs
the evaluation runner and verifies results.
"""

import json
import tempfile
import numpy as np
import pytest
import yaml
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent / "src")
)


def _create_synthetic_pt_data(
    vdir: Path,
    task: str = "test_task",
    n_examples: int = 1,
    seq_len: int = 200,
    head_dim: int = 128,
):
    """Create synthetic .pt data matching the schema."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    rng = np.random.default_rng(42)
    task_dir = vdir / "all_heads" / task

    for ei in range(n_examples):
        ex_dir = task_dir / f"ex_{ei:03d}"
        ex_dir.mkdir(parents=True)

        Q = torch.from_numpy(
            rng.standard_normal(
                (seq_len, head_dim)
            ).astype(np.float32)
        ).to(torch.bfloat16)
        K = torch.from_numpy(
            rng.standard_normal(
                (seq_len, head_dim)
            ).astype(np.float32)
        ).to(torch.bfloat16)
        V = torch.from_numpy(
            rng.standard_normal(
                (seq_len, head_dim)
            ).astype(np.float32)
        ).to(torch.bfloat16)

        torch.save({
            "Q_rope_head0": Q,
            "K_rope_kvhead0": K,
            "V_kvhead0": V,
        }, ex_dir / "layer_17.pt")

        with open(ex_dir / "example.json", "w") as f:
            json.dump({
                "example_id": f"{task}_{ei}",
                "sequence_length": seq_len,
            }, f)

    with open(
        task_dir / "metadata.json", "w",
    ) as f:
        json.dump({
            "task": task,
            "source": "synthetic",
            "model": "test",
            "layers_extracted": [17],
            "n_examples": n_examples,
        }, f)


def _write_test_config(tmpdir, vdir):
    """Write a minimal config for testing."""
    cfg = {
        "model": {
            "hf_name": "test",
            "num_layers": 32,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        "data": {
            "vectors_dir": str(vdir),
            "results_dir": str(
                Path(tmpdir) / "results"
            ),
        },
        "tasks": ["test_task"],
        "evaluation": {
            "seed": 42,
            "n_queries": 5,
            "n_examples": 1,
            "compute_statistics": True,
            "head_mode": "custom",
            "layers": [17],
            "custom_heads": [
                {"layer": 17, "q_head": 0,
                 "kv_head": 0},
            ],
            "exclude_sink_token": True,
            "local_window": {"size": 10},
            "budget_sweep": {
                "absolute": [32],
            },
        },
        "algorithm_configs": {
            "kmeans": {
                "n_clusters": 8,
                "modes": ["hybrid"],
                "top_k_sweep": [0, 3],
            },
        },
    }
    cfg_path = Path(tmpdir) / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return str(cfg_path)


def test_smoke_run():
    """Full evaluation pipeline on synthetic .pt data."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from src.evaluation.run_evaluation import Evaluation

    with tempfile.TemporaryDirectory() as tmpdir:
        vdir = Path(tmpdir) / "vectors"
        _create_synthetic_pt_data(
            vdir, task="test_task",
        )
        cfg_path = _write_test_config(tmpdir, vdir)

        exp = Evaluation(
            config_path=cfg_path,
            name="test",
        )
        exp.run(algo_names=["kmeans"])

        out = exp.out_dir
        assert (out / "spec.json").exists()
        assert (out / "run.json").exists()
        assert (out / "results.csv").exists()

        # Check per-task output
        task_dir = out / "per_task" / "test_task"
        assert task_dir.exists()
        assert (
            task_dir / "aggregated_stats.json"
        ).exists()

        # Check attention statistics were computed
        with open(
            task_dir / "data_statistics.json"
        ) as f:
            ds = json.load(f)
        assert "attention_statistics" in ds
        astats = ds["attention_statistics"]
        assert "full_entropy_mean" in astats
        assert "effective_entropy_mean" in astats

        # Check spec has tasks + date + head_mode
        with open(out / "spec.json") as f:
            spec = json.load(f)
        assert "test_task" in spec["tasks"]
        assert "date" in spec
        assert spec["head_mode"] == "custom"

        # Check CSV has required columns
        import csv
        with open(out / "results.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        required = [
            "task", "layer", "head",
            "method_kind",
        ]
        for col in required:
            assert col in row, (
                f"Missing column: {col}"
            )


def test_math_utils_consistency():
    """softmax and ground truth are numerically stable."""
    from src.core import (
        softmax, full_attention,
        relative_l2_error,
    )

    rng = np.random.default_rng(42)
    head_dim = 128
    seq_len = 100
    q = rng.standard_normal(head_dim).astype(
        np.float32
    )
    K = rng.standard_normal(
        (seq_len, head_dim)
    ).astype(np.float32)
    V = rng.standard_normal(
        (seq_len, head_dim)
    ).astype(np.float32)

    out, logits, weights = full_attention(
        q, K, V, head_dim,
    )

    assert abs(weights.sum() - 1.0) < 1e-6
    err = relative_l2_error(out, out)
    assert err < 1e-10
    assert out.shape == (head_dim,)
    assert logits.shape == (seq_len,)


def test_statistics_module():
    """Statistics functions produce valid results."""
    from src.core import (
        entropy_nats, top_k_mass, softmax,
        concentration_curve, norm_statistics,
        kv_norm_correlation,
    )

    rng = np.random.default_rng(42)

    # Uniform distribution should have max entropy
    n = 100
    uniform_w = np.ones(n) / n
    ent = entropy_nats(uniform_w)
    assert abs(ent - np.log(n)) < 1e-5

    # Top-K mass for uniform
    assert abs(top_k_mass(uniform_w, 50) - 0.5) < 0.01

    # Concentration curve
    curve = concentration_curve(uniform_w)
    assert abs(curve["top_50pct"] - 0.5) < 0.02

    # Norm statistics
    vecs = rng.standard_normal((200, 128)).astype(
        np.float32,
    )
    stats = norm_statistics(vecs)
    assert stats["norm_mean"] > 0
    assert stats["norm_cv"] > 0

    # K-V correlation
    K = rng.standard_normal((100, 128)).astype(
        np.float32,
    )
    V = rng.standard_normal((100, 128)).astype(
        np.float32,
    )
    corr = kv_norm_correlation(K, V)
    assert -1.0 <= corr <= 1.0
