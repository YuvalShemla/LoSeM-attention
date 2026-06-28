"""Training / probe query set ``Q`` shared by ``learned``, ``tensor_fcfw_lq``, and ``kvsculpt``."""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np

# Fallback when ``expand_from_config`` runs outside the evaluation runner.
DEFAULT_N_TRAIN_QUERIES = 5000
DEFAULT_N_SYNTHETIC = 128
DEFAULT_TRAIN_Q_STRATEGY = "trailing"
DEFAULT_ROPE_THETA = 500_000.0

TRAIN_Q_STRATEGIES: FrozenSet[str] = frozenset({"trailing", "kvsculpt"})

PROBE_Q_ALGORITHM_KEYS: FrozenSet[str] = frozenset({
    "learned",
    "tensor_fcfw_lq",
    "kvsculpt",
})


def validate_train_q_strategy(strategy: str) -> str:
    s = str(strategy)
    if s not in TRAIN_Q_STRATEGIES:
        raise ValueError(
            f"train_q_strategy must be one of {sorted(TRAIN_Q_STRATEGIES)}; got {s!r}",
        )
    return s


def n_train_queries_list(cfg: dict) -> List[int]:
    """Normalize ``n_train_queries`` from an algorithm config dict."""
    n = cfg.get("n_train_queries", DEFAULT_N_TRAIN_QUERIES)
    if isinstance(n, int):
        return [n]
    return [int(x) for x in n]


def n_train_queries_int(cfg: dict) -> int:
    return n_train_queries_list(cfg)[0]


def build_probe_queries(
    queries: np.ndarray,
    query_positions: Optional[Sequence[int]],
    ref_pos: int,
    n_train_queries: int,
) -> np.ndarray:
    """Latest ``n_train_queries`` query vectors in the causal prefix before ``ref_pos``.

    Uses the trailing position window
    ``[max(0, ref_pos + 1 - n_train), ref_pos]``, excluding held-out test
    query positions.
    """
    test_set = set(int(p) for p in query_positions) if query_positions else set()
    n_train = max(int(n_train_queries), 0)
    if n_train == 0:
        return np.zeros((0, queries.shape[1]), dtype=np.float32)
    win_start = max(0, ref_pos + 1 - n_train)
    context_pos = [
        p for p in range(win_start, ref_pos + 1)
        if p not in test_set
    ]
    if not context_pos:
        return np.zeros((0, queries.shape[1]), dtype=np.float32)
    return queries[context_pos].astype(np.float32)


def apply_rope(
    x: np.ndarray,
    positions: np.ndarray,
    *,
    head_dim: int,
    rope_theta: float = DEFAULT_ROPE_THETA,
    rope_dims: Optional[int] = None,
) -> np.ndarray:
    """Apply standard half-split RoPE to the leading ``rope_dims`` dimensions."""
    n, d = x.shape
    half = d // 2
    if rope_dims is None:
        rope_dims = d
    n_pairs = min(rope_dims // 2, half)

    x_out = x.astype(np.float64, copy=True)
    positions = positions.astype(np.float64)

    for i in range(n_pairs):
        freq = 1.0 / (rope_theta ** (2.0 * i / head_dim))
        angles = positions * freq
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        d0, d1 = i, i + half
        x0 = x_out[:, d0].copy()
        x1 = x_out[:, d1].copy()
        x_out[:, d0] = x0 * cos_a - x1 * sin_a
        x_out[:, d1] = x0 * sin_a + x1 * cos_a

    return x_out.astype(np.float32)


def inverse_rope(
    x: np.ndarray,
    positions: np.ndarray,
    *,
    head_dim: int,
    rope_theta: float = DEFAULT_ROPE_THETA,
    rope_dims: Optional[int] = None,
) -> np.ndarray:
    """Invert RoPE to recover position-independent content vectors."""
    n, d = x.shape
    half = d // 2
    if rope_dims is None:
        rope_dims = d
    n_pairs = min(rope_dims // 2, half)

    x_out = x.astype(np.float64, copy=True)
    positions = positions.astype(np.float64)

    for i in range(n_pairs):
        freq = 1.0 / (rope_theta ** (2.0 * i / head_dim))
        angles = positions * freq
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        d0, d1 = i, i + half
        x0p = x_out[:, d0].copy()
        x1p = x_out[:, d1].copy()
        x_out[:, d0] = x0p * cos_a + x1p * sin_a
        x_out[:, d1] = -x0p * sin_a + x1p * cos_a

    return x_out.astype(np.float32)


def build_kvsculpt_train_queries(
    queries: np.ndarray,
    ref_pos: int,
    special_idx: np.ndarray,
    n_synthetic: int,
    *,
    head_dim: int,
    rope_theta: float = DEFAULT_ROPE_THETA,
    seed: int = 42,
) -> np.ndarray:
    """
    Retain-zone queries (real, at original positions) plus synthetic future queries.

    Synthetic queries uniformly subsample content vectors from the causal prefix and
    re-apply RoPE at positions ``ref_pos+1, ...``.
    """
    queries = np.asarray(queries, dtype=np.float32)
    retain_pos = np.asarray(special_idx, dtype=np.int64)
    retain_pos = retain_pos[(retain_pos >= 0) & (retain_pos <= ref_pos)]
    parts: List[np.ndarray] = []
    if retain_pos.size > 0:
        parts.append(queries[retain_pos])

    n_synth = max(int(n_synthetic), 0)
    if n_synth > 0 and ref_pos >= 0:
        rng = np.random.default_rng(seed)
        pool = np.arange(ref_pos + 1, dtype=np.int64)
        if pool.size > 0:
            pick = rng.choice(
                pool,
                size=min(n_synth, pool.size),
                replace=pool.size < n_synth,
            )
            content = inverse_rope(
                queries[pick],
                pick.astype(np.float64),
                head_dim=head_dim,
                rope_theta=rope_theta,
            )
            future_pos = np.arange(
                ref_pos + 1, ref_pos + 1 + pick.size, dtype=np.float64,
            )
            parts.append(
                apply_rope(
                    content,
                    future_pos,
                    head_dim=head_dim,
                    rope_theta=rope_theta,
                ),
            )

    if not parts:
        return np.zeros((0, queries.shape[1]), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_train_queries(
    strategy: str,
    queries: np.ndarray,
    query_positions: Optional[Sequence[int]],
    ref_pos: int,
    *,
    special_idx: Optional[np.ndarray] = None,
    n_train_queries: int = DEFAULT_N_TRAIN_QUERIES,
    n_synthetic: int = DEFAULT_N_SYNTHETIC,
    head_dim: int = 128,
    rope_theta: float = DEFAULT_ROPE_THETA,
    seed: int = 42,
) -> np.ndarray:
    """Build training / probe set ``Q`` for probe-Q algorithms."""
    strategy = validate_train_q_strategy(strategy)
    if strategy == "trailing":
        return build_probe_queries(
            queries, query_positions, ref_pos, n_train_queries,
        )
    if special_idx is None:
        raise ValueError("special_idx is required for train_q_strategy='kvsculpt'")
    return build_kvsculpt_train_queries(
        queries,
        ref_pos,
        special_idx,
        n_synthetic,
        head_dim=head_dim,
        rope_theta=rope_theta,
        seed=seed,
    )


def prepare_probe_queries(
    queries: np.ndarray,
    query_positions: Optional[Sequence[int]],
    head_dim: int,
    n_sink: int,
    local_window: int,
    train_q_strategy: str,
    n_train_queries: int,
    n_synthetic: int,
    rope_theta: float,
    seed: int,
) -> tuple[int, np.ndarray]:
    """
    Reference position and training query set ``Q`` for one example/head.

    Returns ``(ref_pos, probe_queries)``. The same array is used for optimization
    and for ``plot_probe_training_error``.
    """
    from .learned.learn_coreset import reference_position
    from ..core import compute_special_indices

    ref_pos = reference_position(len(queries), query_positions)
    n_causal = ref_pos + 1
    sp_idx, _ = compute_special_indices(n_causal, n_sink, local_window)
    probe_q = build_train_queries(
        train_q_strategy,
        queries,
        query_positions,
        ref_pos,
        special_idx=sp_idx,
        n_train_queries=n_train_queries,
        n_synthetic=n_synthetic,
        head_dim=head_dim,
        rope_theta=rope_theta,
        seed=seed,
    )
    return ref_pos, probe_q


def inject_evaluation_probe_q(
    algo_name: str,
    algo_cfg: dict,
    evaluation_cfg: dict,
) -> dict:
    """Merge evaluation-level probe-Q settings into algorithm configs."""
    cfg = dict(algo_cfg)
    if algo_name not in PROBE_Q_ALGORITHM_KEYS:
        return cfg

    if "n_train_queries" not in cfg:
        n_train = evaluation_cfg.get("n_train_queries")
        if n_train is not None:
            cfg["n_train_queries"] = n_train

    if "train_q_strategy" not in cfg:
        strategy = evaluation_cfg.get("train_q_strategy")
        if strategy is not None:
            cfg["train_q_strategy"] = strategy

    if "n_synthetic" not in cfg:
        n_synth = evaluation_cfg.get("n_synthetic")
        if n_synth is not None:
            cfg["n_synthetic"] = n_synth

    if "rope_theta" not in cfg:
        rope_theta = evaluation_cfg.get("rope_theta")
        if rope_theta is not None:
            cfg["rope_theta"] = rope_theta

    if "exact_denominator" not in cfg:
        exact_d = evaluation_cfg.get("exact_denominator")
        if exact_d is not None:
            cfg["exact_denominator"] = exact_d

    return cfg


def inject_evaluation_device(
    algo_cfg: dict,
    evaluation_cfg: dict,
) -> dict:
    """Merge evaluation-level ``device`` into algorithm configs."""
    cfg = dict(algo_cfg)
    if "device" not in cfg:
        device = evaluation_cfg.get("device")
        if device is not None:
            cfg["device"] = device
    return cfg
