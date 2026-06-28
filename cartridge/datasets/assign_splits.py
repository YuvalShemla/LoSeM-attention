#!/usr/bin/env python3
"""
Assign train/test split to conversation example.json files.

Adds a "split" field ("train" or "test") to each conversation's
example.json. 100 conversations are randomly selected as test,
the remaining 900 as train. Uses a fixed seed for reproducibility.

Usage:
  python cartridge/datasets/assign_splits.py
  python cartridge/datasets/assign_splits.py --dataset multi_doc_qa_sanofi
  python cartridge/datasets/assign_splits.py --n-test 200  # custom split
"""

import argparse
import json
import numpy as np
from pathlib import Path

SEED = 42
DATASETS_DIR = Path(__file__).parent


def assign_split(dataset_name, n_test=100):
    vectors_dir = DATASETS_DIR / dataset_name / "vectors" / "conversations"
    if not vectors_dir.exists():
        print(f"  SKIP {dataset_name}: no vectors/conversations/")
        return

    conv_dirs = sorted(
        d for d in vectors_dir.iterdir()
        if d.is_dir() and d.name.startswith("conv_")
    )
    n_conv = len(conv_dirs)
    if n_conv == 0:
        print(f"  SKIP {dataset_name}: no conversations")
        return

    n_test = min(n_test, n_conv)
    n_train = n_conv - n_test

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_conv)
    test_set = set(perm[:n_test].tolist())

    updated = 0
    for ci, conv_dir in enumerate(conv_dirs):
        meta_path = conv_dir / "example.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["split"] = "test" if ci in test_set else "train"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        updated += 1

    print(f"  {dataset_name}: {updated} conversations updated "
          f"({n_train} train, {n_test} test)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Specific dataset (default: all)")
    parser.add_argument("--n-test", type=int, default=100)
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [
            d.name for d in sorted(DATASETS_DIR.iterdir())
            if d.is_dir() and (d / "vectors").exists()
        ]

    print(f"Assigning splits (seed={SEED}, n_test={args.n_test}):")
    for ds in datasets:
        assign_split(ds, args.n_test)
    print("Done.")


if __name__ == "__main__":
    main()
