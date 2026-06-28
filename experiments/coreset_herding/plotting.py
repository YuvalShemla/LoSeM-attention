"""
Plotting for the coreset herding experiment.

Three plots:
  1. Residual norm vs atoms (main result, per distribution)
  2. d_eff vs residual scatter (hypothesis H2)
  3. End-to-end attention error vs coreset size (real data)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


COLORS = {
    "Subset Herding": "#6c8cff",
    "Synthetic Herding": "#ff6b6b",
    "Uniform Sampling": "#8b8fa3",
    "Leverage Sampling": "#4ecdc4",
}
MARKERS = {
    "Subset Herding": "o",
    "Synthetic Herding": "s",
    "Uniform Sampling": "^",
    "Leverage Sampling": "D",
}
ATTN_COLORS = {
    "Subset Herding": "#6c8cff",
    "Oracle TopK": "#ffa726",
    "Uniform": "#8b8fa3",
}


def plot_residuals(
    all_results: Dict[str, Dict[str, List[float]]],
    d_effs: Dict[str, float],
    save_path: Path,
) -> None:
    """Residual norm vs number of atoms (log-log).

    One panel per key distribution. Reference lines for
    O(1/sqrt(T)) and O(1/T) convergence rates.
    """
    n_plots = len(all_results)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5.5 * n_plots, 5)
    )
    if n_plots == 1:
        axes = [axes]

    for ax, (dist, results) in zip(axes, all_results.items()):
        d_eff = d_effs[dist]

        for method, residuals in results.items():
            T = np.arange(1, len(residuals) + 1)
            ax.plot(
                T,
                residuals,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=3,
                linewidth=1.8,
                alpha=0.85,
                label=method,
            )

        max_T = max(len(r) for r in results.values())
        T_ref = np.arange(1, max_T + 1, dtype=float)
        r0 = max(list(results.values())[0][0], 1e-6)
        ax.plot(
            T_ref,
            r0 / np.sqrt(T_ref),
            "--",
            color="gray",
            alpha=0.3,
            linewidth=1,
            label=r"$O(1/\sqrt{T})$",
        )
        ax.plot(
            T_ref,
            r0 / T_ref,
            ":",
            color="gray",
            alpha=0.3,
            linewidth=1,
            label=r"$O(1/T)$",
        )

        ax.set_xlabel("Number of atoms T")
        ax.set_ylabel(r"Residual norm $\|r\|$")
        title = f"{dist}\n$d_{{\\mathrm{{eff}}}} = {d_eff:.0f}$"
        ax.set_title(title, fontsize=12)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_deff_vs_residual(
    all_results: Dict[str, Dict[str, List[float]]],
    d_effs: Dict[str, float],
    T_check: int,
    save_path: Path,
) -> None:
    """Scatter of d_eff vs residual at a fixed atom count.

    Tests hypothesis H2: d_eff should predict difficulty.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    first = True

    for dist, results in all_results.items():
        d_eff = d_effs[dist]
        for method in ["Subset Herding", "Synthetic Herding"]:
            if method not in results:
                continue
            r = results[method]
            idx = min(T_check - 1, len(r) - 1)
            label = method if first else None
            ax.scatter(
                d_eff,
                r[idx],
                c=COLORS[method],
                s=100,
                zorder=5,
                label=label,
            )
            short = dist.split("(")[0].strip()
            ax.annotate(
                short,
                (d_eff, r[idx]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )
        first = False

    ax.set_xlabel(r"$d_{\mathrm{eff}}$", fontsize=13)
    ax.set_ylabel(
        f"Residual norm at T={T_check}", fontsize=13
    )
    ax.set_title(
        "Does effective dimension predict coreset difficulty?"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eigenvalue_spectra(
    eig_data: Dict[str, Tuple[float, np.ndarray]],
    save_path: Path,
) -> None:
    """Eigenvalue spectrum of the exponential kernel."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (d_eff, eigs) in eig_data.items():
        sorted_eigs = np.sort(eigs)[::-1]
        ax.plot(
            sorted_eigs[:100],
            linewidth=2,
            label=f"{name} ($d_{{eff}}$={d_eff:.0f})",
        )
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.set_title("Exponential Kernel Eigenvalue Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_error(
    attn_errors: Dict[str, Dict[int, float]],
    coreset_sizes: List[int],
    save_path: Path,
) -> None:
    """End-to-end attention error vs coreset size."""
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"Subset Herding": "o-", "Oracle TopK": "s--",
               "Uniform": "^:"}

    for method, size_err in attn_errors.items():
        sizes = sorted(size_err.keys())
        vals = [size_err[s] for s in sizes]
        ax.plot(
            sizes,
            vals,
            markers.get(method, "o-"),
            color=ATTN_COLORS.get(method, "gray"),
            linewidth=2,
            markersize=6,
            label=method,
        )

    ax.set_xlabel("Coreset size (atoms)", fontsize=12)
    ax.set_ylabel("Relative L2 Attention Error", fontsize=12)
    ax.set_title(
        "End-to-End Attention Error on Real Data"
    )
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
