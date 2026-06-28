# Coreset Herding Experiment

First experiment from the research plan (Section 1.2): test the
Frank-Wolfe / kernel herding algorithm for building attention
coresets, comparing subset vs synthetic atom selection.

## Background

The softmax denominator is B(q,K) = sum_i exp(q^T k_i) = <psi(q), sigma>
where sigma = sum_i psi(k_i) and psi is the exponential kernel feature
map. Approximating sigma/n by a small weighted sum is the approximate
Caratheodory problem. Frank-Wolfe solves it one atom at a time.

At each step the algorithm needs a Linear Minimization Oracle (LMO):

    k* = argmax_{||k||<=1} sum_i beta_i exp(k_i^T k)

Two strategies:

- **Subset LMO**: scan existing keys, pick the best one. Gives a
  subset coreset (original tokens, RoPE-compatible).
- **Synthetic LMO**: gradient ascent over the unit ball from k-means
  centroid restarts. Gives a synthetic coreset (new keys, potentially
  smaller).

## Hypotheses (from the plan)

1. Synthetic beats subset most on clustered and spherical keys
2. Convergence tracks d_eff (effective dimension) not ambient d
3. Centroid-seeded gradient ascent matches expensive grid search

## Structure

```
experiments/coreset_herding/
    README.md               # this file
    distributions.py        # generate / load key distributions
    gram.py                 # Gram matrix and effective dimension
    herding.py              # Frank-Wolfe herding (subset + synthetic)
    baselines.py            # uniform and leverage-score sampling
    evaluation.py           # end-to-end attention error measurement
    plotting.py             # all plots
    run_experiment.py       # main entry point
    results/                # created on run
```

## Usage

From the repo root:

```bash
python -m experiments.coreset_herding.run_experiment
```

Options:

```
--n 2000           number of keys per distribution
--max_atoms 80     max coreset atoms for subset/baselines
--max_synth 40     max atoms for synthetic (slower)
--seed 42          random seed
--skip_real        skip real Llama data
--results_dir ...  override output directory
```

Results (plots, tables, raw data) are saved to
`experiments/coreset_herding/results/`.

## Expected Results

On log-log residual-vs-atoms plots:

- Uniform sampling: slow O(1/sqrt(T)) decay
- Subset herding: steady O(1/T) or better
- Synthetic herding: same or faster, especially on clustered data
- Clustered keys have lowest d_eff and converge fastest
- Gaussian keys have highest d_eff and converge slowest
