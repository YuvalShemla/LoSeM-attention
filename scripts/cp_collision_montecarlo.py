"""
CP-SNIS vs SimHash-SNIS (MagicPIG) comparison on code_run.

Phase 1: Monte Carlo measurement of CP2 collision probability
         as a function of cosine similarity.
         3000 tables on ~8K subsampled keys. (~10 min)

Phase 2: Per-head comparison on all 5 available heads.
         Build 400 CP2 tables + SimHash tables per head,
         run SNIS at various L values, compare errors.
         (~30 min)

Total: ~40 min.

Usage:
  python3 -m scripts.cp_collision_montecarlo
"""

import numpy as np
import torch
import time
import os
import json
import sys

# ══════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════

MC_TABLES = 3000          # tables for collision rate measurement
MC_KEYS_PER_BIN = 300     # subsampled keys per cosine-sim bin
CP_TABLES = 400           # tables per head for actual hashing
CP_K_DIM = 128            # full cross-polytope dimension
SIMHASH_K_VALUES = [9, 10, 11]
SIMHASH_MIN_HITS = 2
L_SWEEP = [50, 100, 150, 200, 300, 400]
LOCAL_WINDOW = 100
HEAD_DIM = 128
BIN_WIDTH = 0.01
BIN_EDGES = np.arange(-0.40, 0.42, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

HEADS = [
    (0,  "head22", "kvhead5"),
    (8,  "head24", "kvhead6"),
    (12, "head26", "kvhead6"),
    (15, "head21", "kvhead5"),
    (31, "head14", "kvhead3"),
]


# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

def random_orthogonal(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float64)


def cp_labels_batch(proj):
    """CP bucket labels for [n, k] float32 matrix → [n] int32."""
    abs_proj = np.abs(proj)
    idx = np.argmax(abs_proj, axis=1)
    signs = proj[np.arange(len(proj)), idx]
    return 2 * idx + (signs < 0).astype(np.int32)


def cp_label_1d(proj):
    """CP bucket label for a single [k] vector → int."""
    abs_proj = np.abs(proj)
    idx = int(np.argmax(abs_proj))
    return 2 * idx + (1 if proj[idx] < 0 else 0)


def cosine_similarities(keys_c, query_c):
    """Compute cosine similarities between centered keys and query."""
    q64 = query_c.astype(np.float64)
    k64 = keys_c.astype(np.float64)
    q_norm = np.linalg.norm(q64)
    k_norms = np.linalg.norm(k64, axis=1)
    return (k64 @ q64) / (q_norm * k_norms + 1e-10)


def snis_output(logits_retr, values_retr, inclusion_probs,
                logits_special, values_special):
    """SNIS attention output."""
    corrected = logits_retr - np.log(
        np.clip(inclusion_probs, 1e-30, 1.0)
    )
    all_logits = np.concatenate([logits_special, corrected])
    all_values = np.concatenate([values_special, values_retr])
    max_l = float(np.max(all_logits))
    exp_l = np.exp(all_logits - max_l)
    w = exp_l / exp_l.sum()
    return (w[:, None] * all_values).sum(axis=0)


def true_attention_output(logits, values):
    """Full softmax attention."""
    max_l = float(np.max(logits))
    exp_l = np.exp(logits - max_l)
    w = exp_l / exp_l.sum()
    return (w[:, None] * values).sum(axis=0)


def error_metrics(approx, true):
    """Compute L1, L2, and cosine similarity."""
    diff = approx - true
    l1 = float(np.abs(diff).sum())
    l2 = float(np.sqrt((diff ** 2).sum()))
    true_norm = float(np.sqrt((true ** 2).sum()))
    cos = float(np.dot(approx, true)) / (
        float(np.sqrt((approx ** 2).sum())) * true_norm + 1e-15
    )
    return {
        "l1": l1,
        "l2": l2,
        "l1_rel": l1 / (np.abs(true).sum() + 1e-15),
        "l2_rel": l2 / (true_norm + 1e-15),
        "cosine_sim": cos,
    }


# ── Inclusion probability formulas ────────────────────

def cp_inclusion_prob(cos_sims, L, mc_bins, mc_rates):
    """CP inclusion prob from MC collision table."""
    cos_clipped = np.clip(cos_sims, mc_bins[0], mc_bins[-1])
    p_table = np.interp(cos_clipped, mc_bins, mc_rates)
    p_table = np.clip(p_table, 1e-30, 1.0)
    return np.clip(1.0 - np.power(1.0 - p_table, L), 1e-30, 1.0)


def simhash_inclusion_prob(cos_sims, K, L, min_hits=2):
    """SimHash inclusion prob from exact formula."""
    cos_clipped = np.clip(cos_sims, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos_clipped)
    p_bit = 1.0 - theta / np.pi
    p_table = np.power(p_bit, K)

    if min_hits == 1:
        return np.clip(
            1.0 - np.power(1.0 - p_table, L), 1e-30, 1.0,
        )

    # min_hits == 2
    q = 1.0 - p_table
    prob = (
        1.0
        - np.power(q, L)
        - L * p_table * np.power(q, L - 1)
    )
    return np.clip(prob, 1e-30, 1.0)


# ══════════════════════════════════════════════════════
# Phase 1: Monte Carlo collision table
# ══════════════════════════════════════════════════════

def phase1_mc_collision_table(seed=42):
    """Measure CP2 collision rates on subsampled keys."""
    print("=" * 60)
    print("PHASE 1: Monte Carlo CP2 Collision Table")
    print("=" * 60)

    # Load layer 31 keys
    data = torch.load(
        "data/vectors/code_run/ex_000/layer_31.pt",
        weights_only=True,
    )
    keys_all = data["K_rope_kvhead3"].detach().float().numpy()
    query = data["Q_rope_head14"][-1].detach().float().numpy()
    n_all, d = keys_all.shape

    # Center
    key_mean = np.mean(
        keys_all, axis=0, dtype=np.float64,
    ).astype(np.float32)
    keys_c_all = keys_all - key_mean
    query_c = query - key_mean

    # Cosine similarities
    cos_sims_all = cosine_similarities(keys_c_all, query_c)
    print(f"  n_keys={n_all}, d={d}")
    print(f"  cos_sim range: [{cos_sims_all.min():.4f}, "
          f"{cos_sims_all.max():.4f}]")

    # Bin and subsample
    bin_idx_all = np.digitize(cos_sims_all, BIN_EDGES) - 1
    bin_idx_all = np.clip(bin_idx_all, 0, len(BIN_CENTERS) - 1)

    rng = np.random.default_rng(seed)
    selected = np.zeros(n_all, dtype=bool)
    for b in range(len(BIN_CENTERS)):
        in_bin = np.where(bin_idx_all == b)[0]
        if len(in_bin) <= MC_KEYS_PER_BIN:
            selected[in_bin] = True
        else:
            chosen = rng.choice(
                in_bin, MC_KEYS_PER_BIN, replace=False,
            )
            selected[chosen] = True

    sel_idx = np.where(selected)[0]
    keys_c = keys_c_all[sel_idx].astype(np.float64)
    bin_indices = bin_idx_all[sel_idx]
    n = len(sel_idx)

    keys_per_bin = np.bincount(
        bin_indices, minlength=len(BIN_CENTERS),
    )
    nonempty = keys_per_bin[keys_per_bin > 0]
    print(f"  Subsampled: {n} keys ({n/n_all*100:.1f}%)")
    print(f"  Non-empty bins: {len(nonempty)}, "
          f"median keys/bin: {int(np.median(nonempty))}")

    # MC: hash subsampled keys with many tables
    collision_counts = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    q64 = query_c.astype(np.float64)
    k = CP_K_DIM

    t0 = time.time()
    for l in range(MC_TABLES):
        if l % 200 == 0:
            elapsed = time.time() - t0
            rate = l / max(elapsed, 0.01)
            eta = (MC_TABLES - l) / max(rate, 0.01) if rate > 0 else 0
            print(f"  Table {l}/{MC_TABLES}  "
                  f"({elapsed:.1f}s, ~{eta:.0f}s left)")

        Q1 = random_orthogonal(d, rng)
        Q2 = random_orthogonal(d, rng)
        R1, R2 = Q1[:k], Q2[:k]

        # Hash keys
        z1 = (keys_c @ R1.T).astype(np.float32)
        z2 = (keys_c @ R2.T).astype(np.float32)
        b1 = cp_labels_batch(z1)
        b2 = cp_labels_batch(z2)
        labels_k = b1 * (2 * k) + b2

        # Hash query
        z1q = (R1 @ q64).astype(np.float32)
        z2q = (R2 @ q64).astype(np.float32)
        label_q = cp_label_1d(z1q) * (2 * k) + cp_label_1d(z2q)

        collisions = (labels_k == label_q)
        for b in range(len(BIN_CENTERS)):
            mask = (bin_indices == b)
            collision_counts[b] += (collisions & mask).sum()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    total_counts = keys_per_bin * MC_TABLES
    mc_rates = np.where(
        total_counts > 0,
        collision_counts / total_counts.astype(np.float64),
        0.0,
    )

    # Print MC table
    print(f"\n  {'cos':>6} {'p_table':>10} {'n_keys':>7} "
          f"{'n_coll':>9} {'thm1':>10} {'ratio':>8}")
    print("  " + "-" * 56)
    for i, c in enumerate(BIN_CENTERS):
        if total_counts[i] == 0:
            continue
        thm = float(np.power(
            float(k), -2.0 * (1 - c) / (1 + c + 1e-10),
        ))
        r = mc_rates[i] / thm if thm > 1e-15 else float('nan')
        print(f"  {c:6.3f} {mc_rates[i]:10.7f} "
              f"{keys_per_bin[i]:7d} {collision_counts[i]:9d} "
              f"{thm:10.7f} {r:8.4f}")

    # Save
    mc_table = {
        "bins": BIN_CENTERS.tolist(),
        "rates": mc_rates.tolist(),
        "k_dim": CP_K_DIM,
        "n_tables": MC_TABLES,
        "keys_per_bin": keys_per_bin.tolist(),
        "collision_counts": collision_counts.tolist(),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/mc_collision_table.json", "w") as f:
        json.dump(mc_table, f, indent=2)
    print(f"\n  Saved to results/mc_collision_table.json")

    return BIN_CENTERS, mc_rates


# ══════════════════════════════════════════════════════
# Phase 2: Per-head comparison
# ══════════════════════════════════════════════════════

def phase2_per_head(mc_bins, mc_rates, seed=42):
    """Run CP-SNIS and SimHash-SNIS on each head."""
    print("\n" + "=" * 60)
    print("PHASE 2: Per-Head Comparison")
    print("=" * 60)

    rng_master = np.random.default_rng(seed + 1000)

    # Pre-generate CP rotation matrices (shared across heads)
    print(f"\nGenerating {CP_TABLES} CP2 rotation pairs...")
    t0 = time.time()
    d = HEAD_DIM
    k = CP_K_DIM
    rotations_R1 = []  # [CP_TABLES][k, d] float64
    rotations_R2 = []
    for l in range(CP_TABLES):
        Q1 = random_orthogonal(d, rng_master)
        Q2 = random_orthogonal(d, rng_master)
        rotations_R1.append(Q1[:k])
        rotations_R2.append(Q2[:k])
    print(f"  Done in {time.time()-t0:.1f}s")

    # Pre-generate SimHash hyperplanes (shared across heads)
    max_K = max(SIMHASH_K_VALUES)
    n_hyp = max_K * CP_TABLES  # reuse same L count
    print(f"Generating SimHash hyperplanes "
          f"(max_K={max_K}, L={CP_TABLES})...")
    hyperplanes = rng_master.standard_normal(
        (n_hyp, d),
    ).astype(np.float32)
    norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
    hyperplanes /= np.maximum(norms, 1e-10)
    print(f"  Shape: {hyperplanes.shape}")

    all_results = []

    for layer_id, q_head_name, kv_head_name in HEADS:
        print(f"\n{'─'*60}")
        print(f"Layer {layer_id}: {q_head_name} / {kv_head_name}")
        print(f"{'─'*60}")

        # Load data
        data = torch.load(
            f"data/vectors/code_run/ex_000/"
            f"layer_{layer_id:02d}.pt",
            weights_only=True,
        )
        keys = data[f"K_rope_{kv_head_name}"].detach().float().numpy()
        queries = data[f"Q_rope_{q_head_name}"].detach().float().numpy()
        values = data[f"V_{kv_head_name}"].detach().float().numpy()
        n = len(keys)

        # Last query
        query = queries[-1]  # position n-1
        sqrt_d = np.sqrt(float(d))

        # Center keys
        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        keys_c = (keys - key_mean).astype(np.float64)
        query_c = (query - key_mean).astype(np.float64)

        # Logits (using original, non-centered keys)
        logits = (
            keys.astype(np.float64) @ query.astype(np.float64)
        ).astype(np.float64) / sqrt_d

        # Special / candidate split
        # Sink: position 0
        # Local window: last LOCAL_WINDOW positions
        local_start = max(1, n - LOCAL_WINDOW)
        special_idx = np.concatenate([
            [0],
            np.arange(local_start, n),
        ]).astype(int)
        candidate_idx = np.arange(1, local_start).astype(int)
        n_cand = len(candidate_idx)

        print(f"  n={n}, special={len(special_idx)}, "
              f"candidates={n_cand}")

        # True attention
        true_out = true_attention_output(
            logits.astype(np.float64),
            values.astype(np.float64),
        )

        # Cosine similarities for candidates
        cos_sims = cosine_similarities(
            keys_c[candidate_idx],
            query_c,
        ).astype(np.float64)

        # ── CP hashing ────────────────────────────────
        print(f"  Hashing {n} keys with {CP_TABLES} CP2 "
              f"tables...")
        t0 = time.time()
        n_verts = 2 * k
        cp_key_labels = np.empty(
            (n, CP_TABLES), dtype=np.int32,
        )
        cp_query_labels = np.empty(CP_TABLES, dtype=np.int32)

        for l in range(CP_TABLES):
            if l % 100 == 0 and l > 0:
                elapsed = time.time() - t0
                eta = (CP_TABLES - l) / (l / elapsed)
                print(f"    Table {l}/{CP_TABLES} "
                      f"({elapsed:.0f}s, ~{eta:.0f}s left)")

            R1 = rotations_R1[l]
            R2 = rotations_R2[l]

            # Keys
            z1 = (keys_c @ R1.T).astype(np.float32)
            z2 = (keys_c @ R2.T).astype(np.float32)
            b1 = cp_labels_batch(z1)
            b2 = cp_labels_batch(z2)
            cp_key_labels[:, l] = b1 * n_verts + b2

            # Query
            z1q = (R1 @ query_c).astype(np.float32)
            z2q = (R2 @ query_c).astype(np.float32)
            cp_query_labels[l] = (
                cp_label_1d(z1q) * n_verts
                + cp_label_1d(z2q)
            )

        cp_time = time.time() - t0
        print(f"  CP hashing done in {cp_time:.1f}s")

        # CP collision counts for candidates
        cand_cp_labels = cp_key_labels[candidate_idx]
        cand_cp_matches = (
            cand_cp_labels
            == cp_query_labels[np.newaxis, :]
        )  # [n_cand, CP_TABLES]

        # ── SimHash hashing ───────────────────────────
        print(f"  Hashing with SimHash...")
        t0 = time.time()

        # Project all keys onto all hyperplanes at once
        proj_all = (
            keys_c.astype(np.float32) @ hyperplanes.T
        )  # [n, max_K * CP_TABLES]
        sign_bits_all = (proj_all >= 0)

        # Query projection
        proj_q = (
            query_c.astype(np.float32) @ hyperplanes.T
        )
        sign_q_all = (proj_q >= 0)

        # Build per-K hash codes
        simhash_codes = {}  # K → (key_codes [n, L], q_codes [L])
        for K in SIMHASH_K_VALUES:
            # Use first K*CP_TABLES hyperplanes for this K
            n_hyp_k = K * CP_TABLES
            sb_k = sign_bits_all[:, :n_hyp_k].reshape(
                n, CP_TABLES, K,
            )
            sq_k = sign_q_all[:n_hyp_k].reshape(CP_TABLES, K)
            powers = (
                1 << np.arange(K, dtype=np.uint32)
            )[np.newaxis, np.newaxis, :]
            key_codes = (
                sb_k.astype(np.uint32) * powers
            ).sum(axis=2).astype(np.uint32)
            powers_q = (
                1 << np.arange(K, dtype=np.uint32)
            )[np.newaxis, :]
            q_codes = (
                sq_k.astype(np.uint32) * powers_q
            ).sum(axis=1).astype(np.uint32)
            simhash_codes[K] = (key_codes, q_codes)

        sh_time = time.time() - t0
        print(f"  SimHash hashing done in {sh_time:.1f}s")

        # ── Run comparisons at each L ─────────────────
        head_results = []

        for L in L_SWEEP:
            if L > CP_TABLES:
                continue

            # --- CP-SNIS ---
            cp_match_counts = cand_cp_matches[:, :L].sum(
                axis=1,
            )
            cp_retrieved_mask = cp_match_counts >= 1
            cp_retr_local = np.where(cp_retrieved_mask)[0]

            if len(cp_retr_local) > 0:
                cp_retr_global = candidate_idx[cp_retr_local]
                cp_cos = cos_sims[cp_retr_local]
                cp_inc = cp_inclusion_prob(
                    cp_cos, L, mc_bins, mc_rates,
                )
                cp_out = snis_output(
                    logits[cp_retr_global],
                    values[cp_retr_global],
                    cp_inc,
                    logits[special_idx],
                    values[special_idx],
                )
                cp_budget = len(special_idx) + len(cp_retr_local)
                cp_err = error_metrics(cp_out, true_out)
            else:
                cp_budget = len(special_idx)
                cp_out = true_attention_output(
                    logits[special_idx], values[special_idx],
                )
                cp_err = error_metrics(cp_out, true_out)

            row = {
                "method": "CP-SNIS",
                "K": CP_K_DIM, "L": L,
                "budget": int(cp_budget),
                **{f"cp_{k}": v for k, v in cp_err.items()},
            }
            head_results.append(row)

            # --- SimHash-SNIS for each K ---
            for K in SIMHASH_K_VALUES:
                key_codes, q_codes = simhash_codes[K]
                cand_codes = key_codes[candidate_idx, :L]
                cand_matches = (
                    cand_codes
                    == q_codes[:L][np.newaxis, :]
                )
                match_counts = cand_matches.sum(axis=1)
                sh_mask = match_counts >= SIMHASH_MIN_HITS
                sh_retr_local = np.where(sh_mask)[0]

                if len(sh_retr_local) > 0:
                    sh_retr_global = candidate_idx[
                        sh_retr_local
                    ]
                    sh_cos = cos_sims[sh_retr_local]
                    sh_inc = simhash_inclusion_prob(
                        sh_cos, K, L, SIMHASH_MIN_HITS,
                    )
                    sh_out = snis_output(
                        logits[sh_retr_global],
                        values[sh_retr_global],
                        sh_inc,
                        logits[special_idx],
                        values[special_idx],
                    )
                    sh_budget = (
                        len(special_idx) + len(sh_retr_local)
                    )
                    sh_err = error_metrics(sh_out, true_out)
                else:
                    sh_budget = len(special_idx)
                    sh_out = true_attention_output(
                        logits[special_idx],
                        values[special_idx],
                    )
                    sh_err = error_metrics(sh_out, true_out)

                row = {
                    "method": f"SimHash-SNIS",
                    "K": K, "L": L,
                    "budget": int(sh_budget),
                    **{f"sh_{k}": v
                       for k, v in sh_err.items()},
                }
                head_results.append(row)

            # --- IdealTopK at CP's budget ---
            if len(cp_retr_local) > 0:
                B = cp_budget
                top_idx = np.argsort(logits)[-B:]
                ideal_out = true_attention_output(
                    logits[top_idx], values[top_idx],
                )
                ideal_err = error_metrics(ideal_out, true_out)
                row = {
                    "method": "IdealTopK",
                    "K": "-", "L": L,
                    "budget": int(B),
                    **{f"ideal_{k}": v
                       for k, v in ideal_err.items()},
                }
                head_results.append(row)

        # Print results for this head
        print(f"\n  {'Method':<18} {'K':>4} {'L':>5} "
              f"{'Budget':>7} {'L1_rel':>10} {'L2_rel':>10} "
              f"{'CosSim':>10}")
        print(f"  {'-'*70}")

        for r in head_results:
            m = r["method"]
            k_val = r["K"]
            l_val = r["L"]
            b = r["budget"]
            prefix = m.split("-")[0].lower().replace(
                "idealtopk", "ideal",
            ).replace("simhash", "sh").replace("cp", "cp")

            # Find the right error keys
            if "cp_l1_rel" in r:
                l1r = r["cp_l1_rel"]
                l2r = r["cp_l2_rel"]
                cs = r["cp_cosine_sim"]
            elif "sh_l1_rel" in r:
                l1r = r["sh_l1_rel"]
                l2r = r["sh_l2_rel"]
                cs = r["sh_cosine_sim"]
            elif "ideal_l1_rel" in r:
                l1r = r["ideal_l1_rel"]
                l2r = r["ideal_l2_rel"]
                cs = r["ideal_cosine_sim"]
            else:
                continue

            print(f"  {m:<18} {str(k_val):>4} {l_val:>5} "
                  f"{b:>7} {l1r:>10.6f} {l2r:>10.6f} "
                  f"{cs:>10.8f}")

        all_results.append({
            "layer": layer_id,
            "q_head": q_head_name,
            "kv_head": kv_head_name,
            "n_keys": int(n),
            "n_special": int(len(special_idx)),
            "n_candidates": int(n_cand),
            "results": head_results,
        })

    # Save all results
    with open("results/cp_vs_simhash_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/cp_vs_simhash_comparison.json")

    return all_results


# ══════════════════════════════════════════════════════
# Phase 3: Summary
# ══════════════════════════════════════════════════════

def phase3_summary(all_results):
    print("\n" + "=" * 60)
    print("SUMMARY: Average across heads")
    print("=" * 60)

    # Group by (method, K, L)
    from collections import defaultdict
    grouped = defaultdict(list)

    for head in all_results:
        for r in head["results"]:
            key = (r["method"], str(r["K"]), r["L"])

            # Extract the right error
            if "cp_l1_rel" in r:
                grouped[key].append({
                    "budget": r["budget"],
                    "l1_rel": r["cp_l1_rel"],
                    "l2_rel": r["cp_l2_rel"],
                    "cos": r["cp_cosine_sim"],
                })
            elif "sh_l1_rel" in r:
                grouped[key].append({
                    "budget": r["budget"],
                    "l1_rel": r["sh_l1_rel"],
                    "l2_rel": r["sh_l2_rel"],
                    "cos": r["sh_cosine_sim"],
                })
            elif "ideal_l1_rel" in r:
                grouped[key].append({
                    "budget": r["budget"],
                    "l1_rel": r["ideal_l1_rel"],
                    "l2_rel": r["ideal_l2_rel"],
                    "cos": r["ideal_cosine_sim"],
                })

    print(f"\n{'Method':<18} {'K':>4} {'L':>5} "
          f"{'Avg Budget':>10} {'Avg L1_rel':>10} "
          f"{'Avg L2_rel':>10} {'Avg CosSim':>10}")
    print("-" * 72)

    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[2])):
        method, k_val, l_val = key
        vals = grouped[key]
        avg_b = np.mean([v["budget"] for v in vals])
        avg_l1 = np.mean([v["l1_rel"] for v in vals])
        avg_l2 = np.mean([v["l2_rel"] for v in vals])
        avg_cos = np.mean([v["cos"] for v in vals])
        print(f"{method:<18} {k_val:>4} {l_val:>5} "
              f"{avg_b:>10.0f} {avg_l1:>10.6f} "
              f"{avg_l2:>10.6f} {avg_cos:>10.8f}")

    # Paste-ready MC table
    print("\n\n# === PASTE-READY COLLISION TABLE ===")
    try:
        with open("results/mc_collision_table.json") as f:
            mc = json.load(f)
        rates = mc["rates"]
        print(f"# k_dim={mc['k_dim']}, L={mc['n_tables']} MC tables")
        print(f"{mc['k_dim']}: np.array([")
        for i in range(0, len(rates), 5):
            chunk = rates[i:i+5]
            line = ", ".join(f"{v:.7f}" for v in chunk)
            print(f"    {line},")
        print("]),")
    except Exception:
        pass


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    t_start = time.time()

    mc_bins, mc_rates = phase1_mc_collision_table()
    all_results = phase2_per_head(mc_bins, mc_rates)
    phase3_summary(all_results)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
