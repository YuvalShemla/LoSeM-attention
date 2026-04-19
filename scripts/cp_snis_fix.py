"""
Fix CP-SNIS by trying better inclusion probability methods.

The raw MC lookup table has too much noise at low cos_sim,
causing catastrophic SNIS errors. We try:

  1. MC lookup (baseline — known to be bad)
  2. Parametric: p = alpha * k^(-2*rho), alpha fit from MC
  3. Split-table: first 200 tables measure per-key p_table,
     last 200 tables do retrieval (direct, unbiased)
  4. SimHash K_eff: fit an effective K to match MC rates,
     then use (1 - theta/pi)^K_eff (exact SimHash formula
     with effective hash depth)

All compared against SimHash-SNIS (K=9, min_hits=2).

Usage:
  python3 -m scripts.cp_snis_fix
"""

import numpy as np
import torch
import time
import os
import json


# ══════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════

CP_TABLES = 400
CP_K_DIM = 128
HEAD_DIM = 128
LOCAL_WINDOW = 100
SIMHASH_K = 9
SIMHASH_MIN_HITS = 2

# Split: first half for measurement, second half for retrieval
MEASURE_TABLES = 200
RETRIEVE_TABLES = 200

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
    abs_proj = np.abs(proj)
    idx = np.argmax(abs_proj, axis=1)
    signs = proj[np.arange(len(proj)), idx]
    return 2 * idx + (signs < 0).astype(np.int32)


def cp_label_1d(proj):
    abs_proj = np.abs(proj)
    idx = int(np.argmax(abs_proj))
    return 2 * idx + (1 if proj[idx] < 0 else 0)


def cosine_sims(keys_c, query_c):
    q64 = query_c.astype(np.float64)
    k64 = keys_c.astype(np.float64)
    qn = np.linalg.norm(q64)
    kn = np.linalg.norm(k64, axis=1)
    return (k64 @ q64) / (qn * kn + 1e-10)


def snis_output(logits_r, values_r, inc_probs,
                logits_s, values_s):
    corrected = logits_r - np.log(
        np.clip(inc_probs, 1e-30, 1.0),
    )
    all_l = np.concatenate([logits_s, corrected])
    all_v = np.concatenate([values_s, values_r])
    mx = float(np.max(all_l))
    e = np.exp(all_l - mx)
    w = e / e.sum()
    return (w[:, None] * all_v).sum(axis=0)


def true_attn(logits, values):
    mx = float(np.max(logits))
    e = np.exp(logits - mx)
    w = e / e.sum()
    return (w[:, None] * values).sum(axis=0)


def err(approx, true_out):
    d = approx - true_out
    tn = np.sqrt((true_out**2).sum())
    an = np.sqrt((approx**2).sum())
    return {
        "l1_rel": float(np.abs(d).sum() / (np.abs(true_out).sum() + 1e-15)),
        "l2_rel": float(np.sqrt((d**2).sum()) / (tn + 1e-15)),
        "cos": float(np.dot(approx, true_out) / (an * tn + 1e-15)),
    }


# ── Inclusion probability methods ─────────────────────

def method_mc_lookup(cos_sims_r, L, mc_bins, mc_rates):
    """Raw MC table lookup."""
    cs = np.clip(cos_sims_r, mc_bins[0], mc_bins[-1])
    pt = np.interp(cs, mc_bins, mc_rates)
    pt = np.clip(pt, 1e-30, 1.0)
    return np.clip(1 - (1 - pt)**L, 1e-30, 1.0)


def method_parametric(cos_sims_r, L, alpha, k_dim=128):
    """Parametric: p = alpha * k^(-2*rho)."""
    cs = np.clip(cos_sims_r, -0.999, 0.999)
    rho = (1.0 - cs) / (1.0 + cs)
    pt = alpha * np.power(float(k_dim), -2.0 * rho)
    pt = np.clip(pt, 1e-30, 1.0)
    return np.clip(1 - (1 - pt)**L, 1e-30, 1.0)


def method_simhash_keff(cos_sims_r, L, K_eff):
    """SimHash formula with effective K."""
    cs = np.clip(cos_sims_r, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cs)
    p_bit = 1.0 - theta / np.pi
    pt = np.power(p_bit, K_eff)
    pt = np.clip(pt, 1e-30, 1.0)
    return np.clip(1 - (1 - pt)**L, 1e-30, 1.0)


def method_m2(cos_sims_r, L, A=139.93, K=23.54):
    """M2: A * (1-theta/pi)^K — best 2-param fit."""
    cs = np.clip(cos_sims_r, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cs)
    p_bit = 1.0 - theta / np.pi
    pt = A * np.power(p_bit, K)
    pt = np.clip(pt, 1e-30, 1.0)
    return np.clip(1 - (1 - pt)**L, 1e-30, 1.0)


_EXACT_TABLE = None
def method_exact_table(cos_sims_r, L):
    """Exact lookup from 5M-table .npz file."""
    global _EXACT_TABLE
    if _EXACT_TABLE is None:
        d = np.load("data/cp2_collision_table.npz")
        _EXACT_TABLE = (
            d["cos_bins"].astype(np.float64),
            d["p_table"].astype(np.float64),
        )
    cos_bins, p_tab = _EXACT_TABLE
    cs = np.clip(cos_sims_r, cos_bins[0], cos_bins[-1])
    pt = np.interp(cs, cos_bins, p_tab)
    pt = np.clip(pt, 1e-30, 1.0)
    return np.clip(1 - (1 - pt)**L, 1e-30, 1.0)


def method_direct(collision_counts, L_measure, L_retrieve):
    """Direct per-key measurement from separate tables."""
    pt = collision_counts / float(L_measure)
    pt = np.clip(pt, 1e-30, 1.0)
    return np.clip(1 - (1 - pt)**L_retrieve, 1e-30, 1.0)


def simhash_inc(cos_sims_r, K, L, min_hits=2):
    """SimHash exact inclusion probability."""
    cs = np.clip(cos_sims_r, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cs)
    p_bit = 1.0 - theta / np.pi
    pt = np.power(p_bit, K)
    q = 1.0 - pt
    if min_hits == 1:
        return np.clip(1 - q**L, 1e-30, 1.0)
    prob = 1.0 - q**L - L * pt * q**(L - 1)
    return np.clip(prob, 1e-30, 1.0)


# ══════════════════════════════════════════════════════
# Fit parametric models from MC data
# ══════════════════════════════════════════════════════

def fit_alpha(mc_bins, mc_rates, k_dim=128):
    """Fit alpha in p = alpha * k^(-2*rho) from MC data."""
    valid = mc_rates > 0
    bins_v = mc_bins[valid]
    rates_v = mc_rates[valid]

    rho = (1.0 - bins_v) / (1.0 + bins_v)
    thm1 = np.power(float(k_dim), -2.0 * rho)

    # alpha = median(mc_rate / thm1)
    ratios = rates_v / thm1
    alpha = float(np.median(ratios))
    print(f"  Parametric fit: alpha = {alpha:.6f} "
          f"(range {ratios.min():.4f} - {ratios.max():.4f})")
    return alpha


def fit_keff(mc_bins, mc_rates):
    """Fit K_eff in p = (1 - theta/pi)^K_eff from MC data."""
    valid = (mc_rates > 0) & (mc_bins > -0.3) & (mc_bins < 0.4)
    bins_v = mc_bins[valid]
    rates_v = mc_rates[valid]

    theta = np.arccos(np.clip(bins_v, -1 + 1e-7, 1 - 1e-7))
    log_p_bit = np.log(1.0 - theta / np.pi)
    log_rate = np.log(rates_v)

    # log(rate) = K_eff * log(p_bit)
    # Least squares: K_eff = sum(log_rate * log_p_bit) / sum(log_p_bit^2)
    K_eff = float(
        np.sum(log_rate * log_p_bit)
        / np.sum(log_p_bit ** 2)
    )
    print(f"  SimHash K_eff fit: K_eff = {K_eff:.2f}")
    return K_eff


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # Load MC table
    with open("results/mc_collision_table.json") as f:
        mc = json.load(f)
    mc_bins = np.array(mc["bins"])
    mc_rates = np.array(mc["rates"])

    print("=" * 65)
    print("CP-SNIS: Comparing Inclusion Probability Methods")
    print("=" * 65)

    # Fit parametric models
    alpha = fit_alpha(mc_bins, mc_rates)
    K_eff = fit_keff(mc_bins, mc_rates)

    # Pre-generate rotations
    rng = np.random.default_rng(42)
    d = HEAD_DIM
    k = CP_K_DIM
    n_verts = 2 * k

    print(f"\nGenerating {CP_TABLES} CP2 rotation pairs...")
    rotations = []
    for l in range(CP_TABLES):
        Q1 = random_orthogonal(d, rng)
        Q2 = random_orthogonal(d, rng)
        rotations.append((Q1[:k], Q2[:k]))
    print("  Done.")

    # SimHash hyperplanes
    n_hyp = SIMHASH_K * CP_TABLES
    hyperplanes = rng.standard_normal(
        (n_hyp, d),
    ).astype(np.float32)
    norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
    hyperplanes /= np.maximum(norms, 1e-10)

    all_results = []

    for layer_id, q_head_name, kv_head_name in HEADS:
        print(f"\n{'─'*65}")
        print(f"Layer {layer_id}: {q_head_name} / {kv_head_name}")
        print(f"{'─'*65}")

        data = torch.load(
            f"data/vectors/code_run/ex_000/"
            f"layer_{layer_id:02d}.pt",
            weights_only=True,
        )
        keys = data[f"K_rope_{kv_head_name}"].detach().float().numpy()
        queries = data[f"Q_rope_{q_head_name}"].detach().float().numpy()
        values = data[f"V_{kv_head_name}"].detach().float().numpy()
        n = len(keys)
        sqrt_d = np.sqrt(float(d))

        key_mean = np.mean(
            keys, axis=0, dtype=np.float64,
        ).astype(np.float32)
        keys_c = (keys - key_mean).astype(np.float64)
        query = queries[-1]
        query_c = (query - key_mean).astype(np.float64)

        logits = (
            keys.astype(np.float64) @ query.astype(np.float64)
        ) / sqrt_d

        local_start = max(1, n - LOCAL_WINDOW)
        special_idx = np.concatenate([
            [0], np.arange(local_start, n),
        ]).astype(int)
        candidate_idx = np.arange(1, local_start).astype(int)

        true_out = true_attn(logits, values.astype(np.float64))

        # ── CP hashing ────────────────────────────────
        print(f"  Hashing {n} keys with {CP_TABLES} CP2 tables...")
        t0 = time.time()
        cp_labels_keys = np.empty((n, CP_TABLES), dtype=np.int32)
        cp_labels_q = np.empty(CP_TABLES, dtype=np.int32)

        for l in range(CP_TABLES):
            R1, R2 = rotations[l]
            z1 = (keys_c @ R1.T).astype(np.float32)
            z2 = (keys_c @ R2.T).astype(np.float32)
            cp_labels_keys[:, l] = (
                cp_labels_batch(z1) * n_verts
                + cp_labels_batch(z2)
            )
            z1q = (R1 @ query_c).astype(np.float32)
            z2q = (R2 @ query_c).astype(np.float32)
            cp_labels_q[l] = (
                cp_label_1d(z1q) * n_verts
                + cp_label_1d(z2q)
            )
        print(f"  Done in {time.time()-t0:.1f}s")

        # Candidate collision matrix
        cand_labels = cp_labels_keys[candidate_idx]
        cand_matches = (
            cand_labels == cp_labels_q[np.newaxis, :]
        )  # [n_cand, CP_TABLES] bool

        # Candidate cosine similarities
        cand_cos = cosine_sims(
            keys_c[candidate_idx], query_c,
        )

        # ── SimHash hashing ───────────────────────────
        proj = (
            keys_c[candidate_idx].astype(np.float32)
            @ hyperplanes.T
        )
        signs = (proj >= 0).reshape(
            len(candidate_idx), CP_TABLES, SIMHASH_K,
        )
        powers = (
            1 << np.arange(SIMHASH_K, dtype=np.uint32)
        )[None, None, :]
        sh_codes = (
            signs.astype(np.uint32) * powers
        ).sum(axis=2).astype(np.uint32)

        proj_q = (
            query_c.astype(np.float32) @ hyperplanes.T
        )
        signs_q = (proj_q >= 0).reshape(
            CP_TABLES, SIMHASH_K,
        )
        sh_q = (
            signs_q.astype(np.uint32)
            * (1 << np.arange(SIMHASH_K, dtype=np.uint32))[None, :]
        ).sum(axis=1).astype(np.uint32)

        # ── Compare methods at L=200 ─────────────────
        # Split tables: 0..199 = measure, 200..399 = retrieve
        L = RETRIEVE_TABLES

        # CP: retrieval from tables 200-399
        retr_matches = cand_matches[:, MEASURE_TABLES:]
        retr_counts = retr_matches.sum(axis=1)
        cp_retr_mask = retr_counts >= 1
        cp_retr_local = np.where(cp_retr_mask)[0]

        # CP: measurement from tables 0-199
        meas_matches = cand_matches[:, :MEASURE_TABLES]
        meas_counts = meas_matches.sum(axis=1)  # [n_cand]

        # SimHash: retrieval from tables 200-399
        sh_cand = sh_codes[:, MEASURE_TABLES:]
        sh_q_sub = sh_q[MEASURE_TABLES:]
        sh_match_counts = (
            sh_cand == sh_q_sub[np.newaxis, :]
        ).sum(axis=1)
        sh_retr_mask = sh_match_counts >= SIMHASH_MIN_HITS
        sh_retr_local = np.where(sh_retr_mask)[0]

        cp_budget = len(special_idx) + len(cp_retr_local)
        sh_budget = len(special_idx) + len(sh_retr_local)

        print(f"  CP retrieved: {len(cp_retr_local)} "
              f"(budget={cp_budget})")
        print(f"  SimHash K={SIMHASH_K} retrieved: "
              f"{len(sh_retr_local)} (budget={sh_budget})")

        results_head = {}

        # Method 1: MC lookup
        if len(cp_retr_local) > 0:
            cos_r = cand_cos[cp_retr_local]
            inc = method_mc_lookup(cos_r, L, mc_bins, mc_rates)
            out = snis_output(
                logits[candidate_idx[cp_retr_local]],
                values[candidate_idx[cp_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["CP MC-lookup"] = {
                "budget": cp_budget, **err(out, true_out),
            }

        # Method 2: Parametric (alpha * k^(-2*rho))
        if len(cp_retr_local) > 0:
            cos_r = cand_cos[cp_retr_local]
            inc = method_parametric(cos_r, L, alpha)
            out = snis_output(
                logits[candidate_idx[cp_retr_local]],
                values[candidate_idx[cp_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["CP Parametric"] = {
                "budget": cp_budget, **err(out, true_out),
            }

        # Method 3: SimHash K_eff
        if len(cp_retr_local) > 0:
            cos_r = cand_cos[cp_retr_local]
            inc = method_simhash_keff(cos_r, L, K_eff)
            out = snis_output(
                logits[candidate_idx[cp_retr_local]],
                values[candidate_idx[cp_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["CP K_eff"] = {
                "budget": cp_budget, **err(out, true_out),
            }

        # Method 3b: Exact table (5M MC)
        if len(cp_retr_local) > 0:
            cos_r = cand_cos[cp_retr_local]
            inc = method_exact_table(cos_r, L)
            out = snis_output(
                logits[candidate_idx[cp_retr_local]],
                values[candidate_idx[cp_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["CP Exact"] = {
                "budget": cp_budget, **err(out, true_out),
            }

        # Method 3c: M2 fit (A * (1-theta/pi)^K)
        if len(cp_retr_local) > 0:
            cos_r = cand_cos[cp_retr_local]
            inc = method_m2(cos_r, L)
            out = snis_output(
                logits[candidate_idx[cp_retr_local]],
                values[candidate_idx[cp_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["CP M2"] = {
                "budget": cp_budget, **err(out, true_out),
            }

        # Method 4: Direct per-key measurement
        if len(cp_retr_local) > 0:
            meas_c = meas_counts[cp_retr_local]
            inc = method_direct(meas_c, MEASURE_TABLES, L)
            out = snis_output(
                logits[candidate_idx[cp_retr_local]],
                values[candidate_idx[cp_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["CP Direct"] = {
                "budget": cp_budget, **err(out, true_out),
            }

        # Method 5: SimHash-SNIS (reference)
        if len(sh_retr_local) > 0:
            cos_r = cand_cos[sh_retr_local]
            inc = simhash_inc(
                cos_r, SIMHASH_K, L, SIMHASH_MIN_HITS,
            )
            out = snis_output(
                logits[candidate_idx[sh_retr_local]],
                values[candidate_idx[sh_retr_local]],
                inc,
                logits[special_idx],
                values[special_idx],
            )
            results_head["SimHash K=9"] = {
                "budget": sh_budget, **err(out, true_out),
            }

        # Method 6: IdealTopK at CP's budget
        top_idx = np.argsort(logits)[-cp_budget:]
        ideal_out = true_attn(logits[top_idx], values[top_idx])
        results_head["IdealTopK"] = {
            "budget": cp_budget, **err(ideal_out, true_out),
        }

        # Print
        print(f"\n  {'Method':<20} {'Budget':>7} "
              f"{'L1_rel':>10} {'L2_rel':>10} {'CosSim':>12}")
        print(f"  {'-'*63}")
        for name, r in results_head.items():
            print(f"  {name:<20} {r['budget']:>7} "
                  f"{r['l1_rel']:>10.6f} "
                  f"{r['l2_rel']:>10.6f} "
                  f"{r['cos']:>12.8f}")

        all_results.append({
            "layer": layer_id,
            "q_head": q_head_name,
            "kv_head": kv_head_name,
            "results": results_head,
        })

    # ── Summary ───────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY: Average across 5 heads (L=200, split-table)")
    print(f"{'='*65}")

    from collections import defaultdict
    avg = defaultdict(lambda: {"l1": [], "l2": [], "cos": [], "b": []})
    for h in all_results:
        for name, r in h["results"].items():
            avg[name]["l1"].append(r["l1_rel"])
            avg[name]["l2"].append(r["l2_rel"])
            avg[name]["cos"].append(r["cos"])
            avg[name]["b"].append(r["budget"])

    print(f"\n{'Method':<20} {'Avg Bgt':>8} "
          f"{'Avg L1':>10} {'Avg L2':>10} {'Avg Cos':>12}")
    print("-" * 64)
    for name in ["CP MC-lookup", "CP Parametric",
                  "CP K_eff", "CP Exact", "CP M2",
                  "CP Direct",
                  "SimHash K=9", "IdealTopK"]:
        if name not in avg:
            continue
        a = avg[name]
        print(f"{name:<20} {np.mean(a['b']):>8.0f} "
              f"{np.mean(a['l1']):>10.6f} "
              f"{np.mean(a['l2']):>10.6f} "
              f"{np.mean(a['cos']):>12.8f}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/cp_snis_fix_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total = time.time() - t_start
    print(f"\nTotal: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
