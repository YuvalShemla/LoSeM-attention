"""
Non-algorithm-specific sorted p_i distribution plot.

Builds a task-level curve by aggregating query-wise softmax probabilities:
  - p_(i): i-th largest token probability in a query
  - cumulative mass: sum_{j<=i} p_(j)
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plotting import save_figure, setup_style


class PiDistributionAggregator:
    """Streaming accumulator for rank-sorted p_i and cumulative mass."""

    def __init__(self) -> None:
        self._sum_p = np.zeros(0, dtype=np.float64)
        self._sum_cum = np.zeros(0, dtype=np.float64)
        self._count = np.zeros(0, dtype=np.int64)
        self._rel_pos_sum = 0.0
        self._rel_pos_count = 0
        self._ref_level_sums: Dict[str, float] = {}
        self._ref_level_counts: Dict[str, int] = {}
        self._q_norm_values = []
        self._k_norm_values = []
        self.n_queries = 0
        self.max_rank = 0

    def _ensure_len(self, n: int) -> None:
        if n <= len(self._sum_p):
            return
        grow = n - len(self._sum_p)
        self._sum_p = np.pad(self._sum_p, (0, grow), mode="constant")
        self._sum_cum = np.pad(self._sum_cum, (0, grow), mode="constant")
        self._count = np.pad(self._count, (0, grow), mode="constant")

    @staticmethod
    def _sorted_probs_from_logits(logits: np.ndarray) -> np.ndarray:
        lg = np.asarray(logits, dtype=np.float64).ravel()
        if lg.size == 0:
            return np.zeros(0, dtype=np.float64)
        m = float(np.max(lg))
        e = np.exp(lg - m)
        z = float(np.sum(e))
        if z <= 0.0:
            return np.zeros(0, dtype=np.float64)
        p = e / z
        p.sort()
        return p[::-1]

    def add_logits(
        self,
        logits: np.ndarray,
        rel_pos_wrt_q: Optional[float] = None,
        ref_levels: Optional[Dict[str, float]] = None,
        q_norm: Optional[float] = None,
        k_norm_values: Optional[np.ndarray] = None,
    ) -> None:
        """Add one query via its causal logits vector."""
        p = self._sorted_probs_from_logits(logits)
        n = int(p.size)
        if n == 0:
            return
        self._ensure_len(n)
        c = np.cumsum(p)
        self._sum_p[:n] += p
        self._sum_cum[:n] += c
        self._count[:n] += 1
        self.n_queries += 1
        self.max_rank = max(self.max_rank, n)
        if rel_pos_wrt_q is not None and np.isfinite(rel_pos_wrt_q):
            self._rel_pos_sum += float(rel_pos_wrt_q)
            self._rel_pos_count += 1
        if ref_levels:
            for k, v in ref_levels.items():
                vf = float(v)
                if not np.isfinite(vf):
                    continue
                self._ref_level_sums[k] = (
                    self._ref_level_sums.get(k, 0.0) + vf
                )
                self._ref_level_counts[k] = (
                    self._ref_level_counts.get(k, 0) + 1
                )
        if q_norm is not None and np.isfinite(float(q_norm)):
            self._q_norm_values.append(float(q_norm))
        if k_norm_values is not None:
            kn = np.asarray(k_norm_values, dtype=np.float64).ravel()
            if kn.size > 0:
                kn = kn[np.isfinite(kn)]
                if kn.size > 0:
                    self._k_norm_values.extend(kn.tolist())

    def finalize(self) -> Dict:
        """Return JSON-serializable aggregate curves."""
        valid = self._count > 0
        if not np.any(valid):
            return {
                "n_queries": 0,
                "max_rank": 0,
                "rank": [],
                "p_mean": [],
                "cum_mass_mean": [],
                "count_per_rank": [],
                "p1_rel_pos_wrt_q_mean": None,
                "p1_rel_pos_wrt_q_count": 0,
            }
        p_mean = np.zeros_like(self._sum_p)
        c_mean = np.zeros_like(self._sum_cum)
        p_mean[valid] = self._sum_p[valid] / self._count[valid]
        c_mean[valid] = self._sum_cum[valid] / self._count[valid]
        last = int(np.max(np.where(valid)[0])) + 1
        rel_mean = None
        if self._rel_pos_count > 0:
            rel_mean = self._rel_pos_sum / float(self._rel_pos_count)
        ref_level_means: Dict[str, float] = {}
        for k, s in self._ref_level_sums.items():
            c = int(self._ref_level_counts.get(k, 0))
            if c > 0:
                ref_level_means[k] = float(s) / float(c)
        q_norm_median = None
        if self._q_norm_values:
            q_norm_median = float(np.median(self._q_norm_values))
        k_norm_median = None
        if self._k_norm_values:
            k_norm_median = float(np.median(self._k_norm_values))
        return {
            "n_queries": int(self.n_queries),
            "max_rank": int(self.max_rank),
            "rank": np.arange(1, last + 1, dtype=np.int64).tolist(),
            "p_mean": p_mean[:last].tolist(),
            "cum_mass_mean": c_mean[:last].tolist(),
            "count_per_rank": self._count[:last].tolist(),
            "p1_rel_pos_wrt_q_mean": rel_mean,
            "p1_rel_pos_wrt_q_count": int(self._rel_pos_count),
            "reference_levels_mean": ref_level_means,
            "q_norm_median": q_norm_median,
            "k_norm_median": k_norm_median,
        }


def plot_pi_distribution(
    pi_stats: Dict,
    out_dir: Path,
    plot_cfg: Optional[Dict] = None,
    *,
    title: str = "",
    config_caption: str = "",
    filename: str = "pi_distribution",
) -> None:
    """Plot mean sorted p_i (log-log) + mean cumulative mass (secondary y)."""
    if not pi_stats:
        return
    rank = np.asarray(pi_stats.get("rank", []), dtype=np.float64)
    p_mean = np.asarray(pi_stats.get("p_mean", []), dtype=np.float64)
    c_mean = np.asarray(
        pi_stats.get("cum_mass_mean", []),
        dtype=np.float64,
    )
    if rank.size == 0 or p_mean.size != rank.size or c_mean.size != rank.size:
        return

    cfg = plot_cfg or {}
    figsize = tuple(cfg.get("figsize", [16, 10]))
    dpi = int(cfg.get("dpi", 200))

    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(
        rank,
        np.maximum(p_mean, 1e-300),
        color="C0",
        lw=2.0,
        label=r"mean sorted $p_i$",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank i (after sorting p_i descending)")
    ax.set_ylabel(r"$p_i$ (mean across queries)")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")

    ax2 = ax.twinx()
    ax2.plot(
        rank,
        np.clip(c_mean, 0.0, 1.0),
        color="#d62728",
        lw=2.0,
        linestyle="-",
        label=r"mean cumulative mass $\sum_{j\leq i} p_j$",
        zorder=4,
    )
    ax2.set_ylabel("cumulative mass")
    ax2.set_ylim(0.0, 1.02)

    n_q = int(pi_stats.get("n_queries", 0))
    max_rank = int(pi_stats.get("max_rank", rank.size))
    parts = []
    if title:
        parts.append(title)
    parts.append(
        f"Global sorted p_i profile · n_queries={n_q} · max_rank={max_rank}"
    )
    if config_caption:
        parts.append(config_caption)
    ax.set_title("\n".join(parts), fontsize=13, fontweight="bold")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    save_figure(fig, Path(out_dir) / f"{filename}.png", dpi=dpi)


def _per_head_sort_key(item: Dict) -> tuple:
    ent = item.get("effective_entropy")
    if ent is None:
        e = float("inf")
    else:
        try:
            e = float(ent)
            if not np.isfinite(e):
                e = float("inf")
        except (TypeError, ValueError):
            e = float("inf")
    return (
        e,
        int(item.get("layer", 10**9)),
        int(item.get("q_head", 10**9)),
    )


def plot_pi_distribution_per_head(
    per_head_stats: Dict[str, Dict],
    out_dir: Path,
    plot_cfg: Optional[Dict] = None,
    *,
    title: str = "",
    config_caption: str = "",
    filename: str = "pi_distribution_per_head",
) -> None:
    """
    One subplot per head: sorted p_i (log-log) + cumulative mass.
    """
    if not per_head_stats:
        return

    rows = []
    for _, entry in per_head_stats.items():
        stats = entry.get("stats", {})
        rank = np.asarray(stats.get("rank", []), dtype=np.float64)
        p_mean = np.asarray(stats.get("p_mean", []), dtype=np.float64)
        c_mean = np.asarray(
            stats.get("cum_mass_mean", []),
            dtype=np.float64,
        )
        if rank.size == 0:
            continue
        if p_mean.size != rank.size or c_mean.size != rank.size:
            continue
        rows.append({
            "layer": int(entry.get("layer", 0)),
            "q_head": int(entry.get("q_head", 0)),
            "kv_head": int(entry.get("kv_head", 0)),
            "selection_label": str(
                entry.get("selection_label", ""),
            ),
            "effective_entropy": entry.get("effective_entropy"),
            "n_queries": int(stats.get("n_queries", 0)),
            "max_rank": int(stats.get("max_rank", rank.size)),
            "p1_rel_pos_wrt_q_mean": stats.get(
                "p1_rel_pos_wrt_q_mean",
            ),
            "p1_rel_pos_wrt_q_count": int(
                stats.get("p1_rel_pos_wrt_q_count", 0),
            ),
            "reference_levels_mean": dict(
                stats.get("reference_levels_mean", {}),
            ),
            "q_norm_median": stats.get("q_norm_median"),
            "k_norm_median": stats.get("k_norm_median"),
            "rank": rank,
            "p_mean": p_mean,
            "c_mean": c_mean,
        })
    if not rows:
        return

    rows = sorted(rows, key=_per_head_sort_key)
    cfg = plot_cfg or {}
    base_figsize = tuple(cfg.get("figsize", [16, 10]))
    dpi = int(cfg.get("dpi", 200))
    n = len(rows)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols

    setup_style()
    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(base_figsize[0], base_figsize[1] * rows_n / 2),
        squeeze=False,
    )
    for i, item in enumerate(rows):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        rank = item["rank"]
        p_mean = item["p_mean"]
        c_mean = item["c_mean"]
        ax.plot(
            rank,
            np.maximum(p_mean, 1e-300),
            color="C0",
            lw=1.6,
            zorder=3,
            label=r"$p_i$",
        )
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2, linestyle="--")
        ax2 = ax.twinx()
        ax2.plot(
            rank,
            np.clip(c_mean, 0.0, 1.0),
            color="#d62728",
            lw=1.5,
            zorder=4,
            label=r"$\sum_{j\leq i} p_j$",
        )
        ax2.set_ylim(0.0, 1.02)
        ref_levels = item.get("reference_levels_mean", {}) or {}
        ref_specs = [
            ("e^0/Z", "C0", ":", 0.65),
            ("e^1/Z", "C0", ":", 0.8),
            ("e^sqrt(d)/Z", "C0", ":", 0.95),
        ]
        for key, color, ls, a in ref_specs:
            yv = ref_levels.get(key)
            if yv is None:
                continue
            yf = float(yv)
            if not np.isfinite(yf) or yf <= 0.0:
                continue
            ax.axhline(
                yf,
                color=color,
                linestyle=ls,
                linewidth=1.0,
                alpha=a,
                zorder=2.4,
            )
        p1 = float(p_mean[0]) if p_mean.size > 0 else float("nan")
        rel_mean = item.get("p1_rel_pos_wrt_q_mean")
        rel_count = int(item.get("p1_rel_pos_wrt_q_count", 0))
        if np.isfinite(p1):
            lines = [f"$p_1$={p1:.4e}"]
            if rel_mean is not None and rel_count > 0:
                rel_i = int(round(float(rel_mean)))
                lines.append(f"rel pos wrt q: {rel_i:+d}")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                zorder=6,
                bbox=dict(
                    facecolor="white",
                    alpha=0.72,
                    edgecolor="none",
                ),
            )
        ax.tick_params(axis="both", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7, colors="#d62728")
        ttl = f"L{item['layer']}H{item['q_head']}"
        if item["selection_label"]:
            ttl += f" ({item['selection_label']})"
        if item["effective_entropy"] is not None:
            ttl += f"\nent={float(item['effective_entropy']):.2f}"
        qn = item.get("q_norm_median")
        kn = item.get("k_norm_median")
        if qn is not None or kn is not None:
            q_s = f"{float(qn):.3f}" if qn is not None else "n/a"
            k_s = f"{float(kn):.3f}" if kn is not None else "n/a"
            ttl += f"\nmed ||q||={q_s}, med ||k||={k_s}"
        ttl += (
            f"\nn={item['n_queries']} q · rank<={item['max_rank']}"
        )
        ax.set_title(ttl, fontsize=9)
        if r == rows_n - 1:
            ax.set_xlabel("rank i (p_i sorted descending)", fontsize=8)
        if c == 0:
            ax.set_ylabel(r"$p_i$ (log)", fontsize=8)
            ax2.set_ylabel("cum mass", fontsize=8, color="#d62728")
        else:
            ax.set_ylabel("")
            ax2.set_ylabel("")

    for i in range(n, rows_n * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    parts = ["Per-Head sorted p_i and cumulative mass"]
    if title:
        parts.insert(0, title)
    if config_caption:
        parts.append(config_caption)
    parts.append(
        "produced: " + datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    fig.suptitle(
        "\n".join(parts),
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.02, 0.03, 1.0, 0.95))
    save_figure(fig, Path(out_dir) / f"{filename}.png", dpi=dpi)
