"""Figure caption helpers (no matplotlib dependency)."""

from __future__ import annotations

from typing import Dict, List, Optional


def format_probe_method_variants(
    config: Optional[Dict] = None,
    algo_names: Optional[List[str]] = None,
) -> str:
    """Variant / hyperparameter caption for probe-Q budget-sweep methods."""
    if not config:
        return ""
    ev = config.get("evaluation")
    if not isinstance(ev, dict):
        ev = {}
    algo_cfgs = config.get("algorithm_configs", {})
    if not isinstance(algo_cfgs, dict):
        algo_cfgs = {}
    active = set(algo_names or algo_cfgs.keys())

    lines: List[str] = []

    if "tensor_fcfw_lq" in active:
        cfg = algo_cfgs.get("tensor_fcfw_lq", {})
        parts = []
        oracle = cfg.get("oracle")
        if oracle:
            parts.append(f"oracle={oracle}")
        if cfg.get("irls_iters") is not None:
            parts.append(f"irls={cfg['irls_iters']}")
        if cfg.get("scoring_irls_iters") is not None:
            parts.append(f"score_irls={cfg['scoring_irls_iters']}")
        if cfg.get("correction_irls_iters") is not None:
            parts.append(f"corr_irls={cfg['correction_irls_iters']}")
        if cfg.get("correction_period") is not None:
            parts.append(f"corr_period={cfg['correction_period']}")
        if cfg.get("lbfgs_steps"):
            parts.append(f"lbfgs={cfg['lbfgs_steps']}")
        if cfg.get("rcond") is not None:
            parts.append(f"rcond={cfg['rcond']}")
        if parts:
            lines.append("TFCFW-lq: " + " · ".join(parts))

    if "learned" in active:
        cfg = algo_cfgs.get("learned", {})
        parts = []
        if cfg.get("init"):
            parts.append(f"init={cfg['init']}")
        if cfg.get("loss"):
            parts.append(f"loss={cfg['loss']}")
        if cfg.get("n_steps") is not None:
            parts.append(f"steps={cfg['n_steps']}")
        if parts:
            lines.append("Learned: " + " · ".join(parts))

    if "kvsculpt" in active:
        cfg = algo_cfgs.get("kvsculpt", {})
        parts = []
        if cfg.get("n_k_steps") is not None:
            parts.append(f"k_steps={cfg['n_k_steps']}")
        if cfg.get("v_solve_every") is not None:
            parts.append(f"v_every={cfg['v_solve_every']}")
        if parts:
            lines.append("KVSculpt: " + " · ".join(parts))

    shared = []
    q_strat = ev.get("train_q_strategy")
    if q_strat and active & {"tensor_fcfw_lq", "learned", "kvsculpt"}:
        shared.append(f"Q={q_strat}")
    if ev.get("n_train_queries") is not None and shared:
        shared.append(f"|Q|={ev['n_train_queries']}")
    if ev.get("n_synthetic") is not None and shared:
        shared.append(f"synth={ev['n_synthetic']}")
    if "exact_denominator" in ev and shared:
        shared.append(f"exact_d={ev['exact_denominator']}")
    if shared:
        lines.append("Probe train: " + " · ".join(shared))

    return "\n".join(lines)


def format_eval_config_caption(
    config: Optional[Dict] = None,
    algo_names: Optional[List[str]] = None,
) -> str:
    """
    Short caption of important evaluation knobs for figure titles.

    Emphasizes local window size (and sink handling), query count,
    head mode, and seed so plots are self-describing when compared.
    Optionally appends probe-method variant lines (TFCFW-lq oracle, etc.).
    """
    if not config:
        return ""
    ev = config.get("evaluation")
    if not isinstance(ev, dict):
        return ""
    lw = ev.get("local_window") or {}
    if isinstance(lw, dict):
        local_size = lw.get("size", "?")
    else:
        local_size = "?"
    sink = (
        1 if ev.get("exclude_sink_token", True) else 0
    )
    nq = ev.get("n_queries", "?")
    ne = ev.get("n_examples", "?")
    hm = ev.get("head_mode", "") or ""
    if hm == "selected_heads":
        hm_s = "selected"
    elif hm == "all_heads":
        hm_s = "all"
    elif hm == "custom":
        hm_s = "custom"
    else:
        hm_s = hm[:16] if hm else "?"
    seed = ev.get("seed", "?")
    nk = bool(ev.get("normalize_keys_to_median_norm", False))
    nq_norm = bool(ev.get("normalize_queries_to_median_norm", False))
    nk_s = "on" if nk else "off"
    nq_s = "on" if nq_norm else "off"
    base = (
        f"local={local_size} · sink={sink} · n_q={nq} · "
        f"ex={ne} · heads={hm_s} · seed={seed} · "
        f"normK={nk_s} · normQ={nq_s}"
    )
    variants = format_probe_method_variants(config, algo_names)
    if variants:
        return f"{base}\n{variants}"
    return base
