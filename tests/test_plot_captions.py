"""Tests for evaluation figure captions."""

from src.evaluation.caption import (
    format_eval_config_caption,
    format_probe_method_variants,
)


def test_probe_method_variants_tensor_fcfw_lq():
    config = {
        "evaluation": {
            "train_q_strategy": "kvsculpt",
            "n_train_queries": 5000,
            "n_synthetic": 1000,
            "exact_denominator": True,
        },
        "algorithm_configs": {
            "tensor_fcfw_lq": {
                "oracle": "residual_lq",
                "irls_iters": 5,
                "rcond": 1e-3,
            },
        },
    }
    text = format_probe_method_variants(config, ["tensor_fcfw_lq"])
    assert "TFCFW-lq: oracle=residual_lq" in text
    assert "irls=5" in text
    assert "Q=kvsculpt" in text
    assert "exact_d=True" in text


def test_eval_config_caption_includes_variants():
    config = {
        "evaluation": {
            "local_window": {"size": 1024},
            "exclude_sink_token": True,
            "n_queries": 1,
            "n_examples": 1,
            "head_mode": "selected_heads",
            "seed": 42,
            "train_q_strategy": "kvsculpt",
            "n_train_queries": 5000,
            "exact_denominator": True,
        },
        "algorithm_configs": {
            "tensor_fcfw_lq": {"oracle": "omp"},
        },
    }
    cap = format_eval_config_caption(config, ["tensor_fcfw_lq"])
    assert "local=1024" in cap
    assert "TFCFW-lq: oracle=omp" in cap
