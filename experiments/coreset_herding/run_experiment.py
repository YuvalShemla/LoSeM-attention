"""
Coreset Herding Experiment — Research Plan Section 1.2

Tests Frank-Wolfe / kernel herding for building attention
coresets, comparing subset vs synthetic atom selection on
four key distributions.

Usage:
    python -m experiments.coreset_herding.run_experiment
    python -m experiments.coreset_herding.run_experiment \\
        --n 2000 --max_atoms 80 --tau 5

Results saved to experiments/coreset_herding/results/.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from .distributions import build_distributions
from .gram import compute_gram, effective_dimension
from .herding import herding_subset, herding_synthetic
from .baselines import uniform_sampling, leverage_sampling
from .evaluation import measure_attention_errors
from .plotting import (
    plot_residuals,
    plot_deff_vs_residual,
    plot_eigenvalue_spectra,
    plot_attention_error,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Coreset herding experiment"
    )
    p.add_argument(
        "--n", type=int, default=2000,
        help="Keys per distribution",
    )
    p.add_argument(
        "--d", type=int, default=128,
        help="Key dimension",
    )
    p.add_argument(
        "--max_atoms", type=int, default=80,
        help="Max atoms for subset/baselines",
    )
    p.add_argument(
        "--max_synth", type=int, default=40,
        help="Max atoms for synthetic (slower)",
    )
    p.add_argument(
        "--tau", type=float, default=5.0,
        help="Kernel temperature exp(tau * k^T k'). "
        "Higher tau amplifies structure; tau=1/sqrt(d) "
        "matches standard attention scaling.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip_real", action="store_true",
        help="Skip real Llama data",
    )
    p.add_argument(
        "--results_dir", type=str, default=None,
    )
    return p.parse_args()


def run_all_methods(
    K, G, max_atoms, max_synth, tau, rng
):
    """Run all four methods on one distribution."""
    results = {}
    extras = {}

    print("  Subset herding...", end=" ", flush=True)
    t0 = time.time()
    res, indices = herding_subset(K, G, max_atoms, tau)
    results["Subset Herding"] = res
    extras["subset_indices"] = indices
    print(f"{time.time() - t0:.1f}s")

    print(
        "  Synthetic herding...", end=" ", flush=True
    )
    t0 = time.time()
    results["Synthetic Herding"] = herding_synthetic(
        K, G, max_synth, tau=tau, rng=rng
    )
    print(f"{time.time() - t0:.1f}s")

    print(
        "  Uniform sampling...", end=" ", flush=True
    )
    t0 = time.time()
    results["Uniform Sampling"] = uniform_sampling(
        K, G, max_atoms, rng, tau
    )
    print(f"{time.time() - t0:.1f}s")

    print(
        "  Leverage sampling...", end=" ", flush=True
    )
    t0 = time.time()
    results["Leverage Sampling"] = leverage_sampling(
        K, G, max_atoms, rng, tau
    )
    print(f"{time.time() - t0:.1f}s")

    return results, extras


def print_summary(all_results, d_effs):
    """Print summary table and hypothesis checks."""
    header = (
        f"{'Distribution':<24s}  {'d_eff':>6s}  "
        f"{'Method':<22s}  {'r(T=10)':>9s}  "
        f"{'r(T=30)':>9s}"
    )
    print(f"\n{header}")
    print("-" * 80)
    for dist, results in all_results.items():
        d_eff = d_effs[dist]
        for method, residuals in results.items():
            r10 = residuals[min(9, len(residuals) - 1)]
            r30 = residuals[min(29, len(residuals) - 1)]
            print(
                f"{dist:<24s}  {d_eff:6.0f}  "
                f"{method:<22s}  {r10:9.5f}  "
                f"{r30:9.5f}"
            )
        print()

    print("=" * 60)
    print("HYPOTHESIS 1: Synthetic vs Subset gap")
    print("=" * 60)
    for dist, results in all_results.items():
        sub = results["Subset Herding"]
        syn = results["Synthetic Herding"]
        T = min(len(sub), len(syn)) - 1
        ratio = sub[T] / max(syn[T], 1e-10)
        if ratio > 1.05:
            tag = "SYNTHETIC wins"
        elif ratio < 0.95:
            tag = "SUBSET wins"
        else:
            tag = "~TIE"
        print(
            f"  {dist:<24s}  "
            f"subset/synth = {ratio:.2f}  -> {tag}"
        )

    print()
    print("=" * 60)
    print("HYPOTHESIS 2: d_eff predicts difficulty")
    print("=" * 60)
    pairs = sorted(d_effs.items(), key=lambda x: x[1])
    for dist, deff in pairs:
        sub = all_results[dist]["Subset Herding"]
        r_last = sub[-1]
        print(
            f"  {dist:<24s}  d_eff={deff:6.0f}  "
            f"r_final={r_last:.5f}"
        )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    results_dir = Path(
        args.results_dir
        or Path(__file__).parent / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Coreset Herding Experiment")
    print(
        f"  n={args.n}  d={args.d}  tau={args.tau}  "
        f"max_atoms={args.max_atoms}"
    )
    print("=" * 60)

    dists, real_data = build_distributions(
        args.n,
        args.d,
        rng,
        include_real=not args.skip_real,
    )
    print(f"\n{len(dists)} distributions loaded.\n")

    gram_data = {}
    d_effs = {}
    eig_data = {}

    for name, K in dists.items():
        t0 = time.time()
        G = compute_gram(K, tau=args.tau)
        d_eff, eigs = effective_dimension(G, lam=1.0)
        gram_data[name] = (K, G)
        d_effs[name] = d_eff
        eig_data[name] = (d_eff, eigs)
        top3 = eigs[-3:][::-1]
        print(
            f"  {name:<24s}  d_eff={d_eff:7.1f}  "
            f"top eigs=[{top3[0]:.0f}, {top3[1]:.0f}, "
            f"{top3[2]:.0f}]  ({time.time()-t0:.1f}s)"
        )

    plot_eigenvalue_spectra(
        eig_data, results_dir / "eigenvalue_spectra.png"
    )

    all_results = {}
    all_extras = {}
    for name, (K, G) in gram_data.items():
        print(
            f"\n--- {name} (d_eff={d_effs[name]:.0f}) ---"
        )
        results, extras = run_all_methods(
            K, G,
            args.max_atoms, args.max_synth,
            args.tau, rng,
        )
        all_results[name] = results
        all_extras[name] = extras

    plot_residuals(
        all_results, d_effs,
        results_dir / "residual_vs_atoms.png",
    )
    plot_deff_vs_residual(
        all_results, d_effs, T_check=20,
        save_path=results_dir / "deff_vs_residual.png",
    )

    print_summary(all_results, d_effs)

    # --- Attention error on real data ---
    if (
        real_data is not None
        and "Real (Llama)" in all_extras
    ):
        real_idx = all_extras["Real (Llama)"].get(
            "subset_indices", []
        )
        if len(real_idx) > 0:
            K_r, V_r, Q_r = real_data
            sizes = [5, 10, 20, 30, 40, 50, 60]
            print("\nMeasuring attention error...")
            attn_errors = measure_attention_errors(
                K_r, V_r, Q_r, real_idx, sizes, rng
            )
            plot_attention_error(
                attn_errors, sizes,
                results_dir / "attention_error.png",
            )
            print("Attention error at T=30:")
            for m, se in attn_errors.items():
                if 30 in se:
                    print(f"  {m:<20s}: {se[30]:.4f}")

    raw = {
        "config": {
            "n": args.n, "d": args.d,
            "tau": args.tau, "seed": args.seed,
            "max_atoms": args.max_atoms,
            "max_synth": args.max_synth,
        },
        "d_effs": d_effs,
        "residuals": {
            dist: {m: v for m, v in res.items()}
            for dist, res in all_results.items()
        },
    }
    with open(results_dir / "raw_results.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
