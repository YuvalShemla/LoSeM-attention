"""
Test data loading utilities (.pt format only).

Tests .pt save/load round-trip, example discovery,
and the load_examples iterator.
"""

import json
import tempfile
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent / "src")
)


def test_pt_roundtrip():
    """Bfloat16 .pt save/load preserves data."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    head_dim = 128
    seq_len = 100
    tensor = torch.randn(
        seq_len, head_dim
    ).to(torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "layer_17.pt"
        torch.save(
            {"Q_rope_head0": tensor}, path,
        )
        loaded = torch.load(
            path, map_location="cpu",
            weights_only=True,
        )
        recovered = loaded["Q_rope_head0"]
        assert recovered.dtype == torch.bfloat16
        assert recovered.shape == (seq_len, head_dim)
        assert torch.equal(tensor, recovered)


def test_pt_example_loading():
    """load_pt_example returns correct arrays."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from src.evaluation.data_loader import (
        load_pt_example,
    )

    head_dim = 128
    seq_len = 50
    q = torch.randn(seq_len, head_dim).to(
        torch.bfloat16,
    )
    k = torch.randn(seq_len, head_dim).to(
        torch.bfloat16,
    )
    v = torch.randn(seq_len, head_dim).to(
        torch.bfloat16,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ex_dir = Path(tmpdir) / "ex_000"
        ex_dir.mkdir()
        torch.save({
            "Q_rope_head0": q,
            "K_rope_kvhead0": k,
            "V_kvhead0": v,
        }, ex_dir / "layer_17.pt")

        # Write example.json
        with open(ex_dir / "example.json", "w") as f:
            json.dump({
                "example_id": "test_0",
                "sequence_length": seq_len,
            }, f)

        data = load_pt_example(
            ex_dir, layer=17, head=0, kv_head=0,
        )
        assert data["Q"].shape == (seq_len, head_dim)
        assert data["K"].shape == (seq_len, head_dim)
        assert data["V"].shape == (seq_len, head_dim)
        assert data["Q"].dtype == np.float32
        assert data["metadata"]["example_id"] == (
            "test_0"
        )


def test_discover_examples():
    """discover_examples finds ex_NNN directories."""
    from src.evaluation.data_loader import (
        discover_examples,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        vdir = Path(tmpdir)
        task_dir = vdir / "all_heads" / "math_calc"
        (task_dir / "ex_000").mkdir(parents=True)
        (task_dir / "ex_001").mkdir()
        (task_dir / "ex_002").mkdir()
        (task_dir / "metadata.json").touch()

        dirs = discover_examples(
            vdir, "math_calc", "all_heads",
        )
        assert len(dirs) == 3
        assert dirs[0].name == "ex_000"
        assert dirs[2].name == "ex_002"


def test_load_examples_iterator():
    """load_examples yields correct dicts."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from src.evaluation.data_loader import (
        load_examples,
    )

    head_dim = 128
    seq_len = 30

    with tempfile.TemporaryDirectory() as tmpdir:
        vdir = Path(tmpdir)
        task_dir = (
            vdir / "all_heads" / "math_calc"
        )
        for i in range(3):
            ex_dir = task_dir / f"ex_{i:03d}"
            ex_dir.mkdir(parents=True)
            torch.save({
                "Q_rope_head0": torch.randn(
                    seq_len, head_dim,
                ).to(torch.bfloat16),
                "K_rope_kvhead0": torch.randn(
                    seq_len, head_dim,
                ).to(torch.bfloat16),
                "V_kvhead0": torch.randn(
                    seq_len, head_dim,
                ).to(torch.bfloat16),
            }, ex_dir / "layer_17.pt")

        examples = list(load_examples(
            vdir, "math_calc", layer=17,
            head=0, kv_head=0,
            phase="all_heads",
        ))
        assert len(examples) == 3
        for ex in examples:
            assert "Q" in ex
            assert "K" in ex
            assert "V" in ex
            assert ex["task"] == "math_calc"
            assert ex["layer"] == 17


def test_count_examples():
    """count_examples returns correct count."""
    from src.evaluation.data_loader import (
        count_examples,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        vdir = Path(tmpdir)
        task_dir = vdir / "all_heads" / "test_task"
        for i in range(5):
            (task_dir / f"ex_{i:03d}").mkdir(
                parents=True,
            )
        assert count_examples(
            vdir, "test_task", "all_heads",
        ) == 5
