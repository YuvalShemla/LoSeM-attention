"""Probe-query set ``Q`` shared by ``learned`` and ``tensor_fcfw_lq``."""

from __future__ import annotations

from typing import Dict, FrozenSet, List

# Fallback when ``expand_from_config`` runs outside the evaluation runner.
DEFAULT_N_TRAIN_QUERIES = 5000

PROBE_Q_ALGORITHM_KEYS: FrozenSet[str] = frozenset({
    "learned",
    "tensor_fcfw_lq",
})


def n_train_queries_list(cfg: dict) -> List[int]:
    """Normalize ``n_train_queries`` from an algorithm config dict."""
    n = cfg.get("n_train_queries", DEFAULT_N_TRAIN_QUERIES)
    if isinstance(n, int):
        return [n]
    return [int(x) for x in n]


def n_train_queries_int(cfg: dict) -> int:
    return n_train_queries_list(cfg)[0]


def inject_evaluation_probe_q(
    algo_name: str,
    algo_cfg: dict,
    evaluation_cfg: dict,
) -> dict:
    """Merge ``evaluation.n_train_queries`` into probe-Q algorithm configs."""
    cfg = dict(algo_cfg)
    if algo_name in PROBE_Q_ALGORITHM_KEYS and "n_train_queries" not in cfg:
        n_train = evaluation_cfg.get("n_train_queries")
        if n_train is not None:
            cfg["n_train_queries"] = n_train
    return cfg
