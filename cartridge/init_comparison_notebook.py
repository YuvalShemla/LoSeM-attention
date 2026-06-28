# %% [markdown]
# # KV Cache Compression: Initialization & Optimizer Comparison
# ## US Presidential Debates — Llama 3.1 8B (44,998 context tokens)
#
# **Sections:**
# 1. Setup & data loading
# 2. Init-only comparison (no training) vs IdealTopK & vAttention
# 3. Adam post-training — all 7 inits
# 4. L-BFGS (KVSculpt-style) post-training — K-only + ridge V solve
# 4b. Full-Softmax L-BFGS — same loss as Adam, learns K+V+w
# 5. Adam vs KVSculpt-LBFGS vs Full-LBFGS comparison
# 6. exact_denominator ablation (True vs False)
# 7. Training query source comparison (self-study vs De-RoPE vs context)

# %% [markdown]
# ## Section 1: Setup & Data Loading

# %%
# ── 1a. Clone repo + install ──
import os, subprocess, sys
REPO_URL = "https://github.com/YuvalShemla/LoSeM-attention.git"
BRANCH = "experiment/self-study-cartridge-clean"
REPO_DIR = "LoSeM-attention"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, ".")

import torch, numpy as np, json, time, gc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB)")

# %%
# ── 1b. Constants ──
HEAD_DIM = 128
N_SINK = 1
LOCAL_WINDOW = 128
SEED = 42
ROPE_THETA = 500_000.0
BUDGETS = [128, 512, 1024, 2048, 4096]
DATASET = "us_presidential_debates"

HEADS = [
    {"label": "p0",   "layer": 31, "q_head": 14, "kv_head": 3, "entropy": 0.14},
    {"label": "p25",  "layer": 12, "q_head": 25, "kv_head": 6, "entropy": 3.98},
    {"label": "p50",  "layer": 15, "q_head": 22, "kv_head": 5, "entropy": 4.71},
    {"label": "p75",  "layer": 27, "q_head":  7, "kv_head": 1, "entropy": 5.64},
    {"label": "p100", "layer":  0, "q_head": 22, "kv_head": 5, "entropy": 11.03},
]

# Colors for init methods
INIT_COLORS = {
    "RandomGauss":  "#d62728",
    "RandomKeys":   "#ff7f0e",
    "FirstTokens":  "#bcbd22",
    "KMeans":       "#2ca02c",
    "MQBeta":       "#1f77b4",
    "TFCFW-omp":    "#9467bd",
    "KVSculpt":     "#17becf",
}
INIT_ORDER = list(INIT_COLORS.keys())

# Adam training config
ADAM_CFG = dict(
    lr=0.01, n_steps=3000, batch_size=256,
    early_stop_patience=500, lr_decay_step=500, lr_decay_gamma=0.5,
    val_fraction=0.1, loss="relative_l2", rel_l2_floor=0.01,
)

# L-BFGS training config (KVSculpt-style)
LBFGS_CFG = dict(
    n_k_steps=100, v_solve_every=5,
    lbfgs_lr=0.5, lbfgs_inner_iter=10,
    ridge_lambda=1e-3,
)

VECTORS_DIR = Path(f"cartridge/datasets/{DATASET}/vectors")
print(f"Dataset: {DATASET}")
print(f"Budgets: {BUDGETS}")
print(f"Heads: {[h['label'] for h in HEADS]}")

# %%
# ── 1c. Data loading functions ──
from src.core import full_attention, compute_special_indices, relative_l2_error

def load_head_context(vectors_dir, layer, q_head, kv_head):
    """Load context K/V/Q (both rope and raw) for one head."""
    pt = torch.load(vectors_dir / "context" / f"layer_{layer:02d}.pt",
                    map_location="cpu", weights_only=True)
    return {
        "K_rope": pt[f"K_rope_kvhead{kv_head}"].float().numpy(),
        "K_raw":  pt[f"K_raw_kvhead{kv_head}"].float().numpy(),
        "V":      pt[f"V_kvhead{kv_head}"].float().numpy(),
        "Q_rope": pt[f"Q_rope_head{q_head}"].float().numpy(),
        "Q_raw":  pt[f"Q_raw_head{q_head}"].float().numpy(),
    }

def load_conv_vectors(vectors_dir, conv_idx, layer, q_head, kv_head):
    """Load Q/K/V for one conversation's QA portion."""
    pt_path = vectors_dir / "conversations" / f"conv_{conv_idx:04d}" / f"layer_{layer:02d}.pt"
    if not pt_path.exists():
        return None
    pt = torch.load(pt_path, map_location="cpu", weights_only=True)
    return {
        "Q_rope": pt[f"Q_rope_head{q_head}"].float().numpy(),
        "K_rope": pt[f"K_rope_kvhead{kv_head}"].float().numpy(),
        "V":      pt[f"V_kvhead{kv_head}"].float().numpy(),
    }

def load_conv_metadata(vectors_dir, conv_idx):
    """Load conversation example.json metadata."""
    path = vectors_dir / "conversations" / f"conv_{conv_idx:04d}" / "example.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def get_train_test_conv_indices(vectors_dir, seed=42):
    """Get train/test split from example.json 'split' field, or create 80/20."""
    conv_parent = vectors_dir / "conversations"
    all_convs = sorted(int(d.name.split("_")[1])
                       for d in conv_parent.iterdir()
                       if d.is_dir() and d.name.startswith("conv_"))
    train_idx, test_idx = [], []
    for ci in all_convs:
        meta = load_conv_metadata(vectors_dir, ci)
        if meta and meta.get("split") == "test":
            test_idx.append(ci)
        else:
            train_idx.append(ci)
    # Fallback if no split field
    if not test_idx:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_convs))
        n_test = len(all_convs) // 5
        test_idx = sorted([all_convs[i] for i in perm[:n_test]])
        train_idx = sorted([all_convs[i] for i in perm[n_test:]])
    return train_idx, test_idx

def load_question_queries(vectors_dir, conv_indices, layer, q_head, kv_head):
    """Load question-only Q vectors from conversations."""
    all_q = []
    for ci in conv_indices:
        meta = load_conv_metadata(vectors_dir, ci)
        if meta is None:
            continue
        q_tokens = meta.get("question_tokens", 0)
        if q_tokens == 0:
            continue
        vecs = load_conv_vectors(vectors_dir, ci, layer, q_head, kv_head)
        if vecs is None:
            continue
        all_q.append(vecs["Q_rope"][:q_tokens])
    if all_q:
        return np.concatenate(all_q, axis=0).astype(np.float32)
    return np.zeros((0, HEAD_DIM), dtype=np.float32)

def build_test_problems(vectors_dir, test_conv_indices, layer, q_head, kv_head,
                        K_ctx, V_ctx, max_total=100):
    """Build test problems: (q, K_full, V_full, full_out, special_idx, candidate_idx)."""
    from src.algorithms.base import AttentionInput
    ctx_len = K_ctx.shape[0]
    problems = []
    for ci in test_conv_indices:
        meta = load_conv_metadata(vectors_dir, ci)
        if meta is None:
            continue
        q_tokens = meta.get("question_tokens", 0)
        if q_tokens == 0:
            continue
        vecs = load_conv_vectors(vectors_dir, ci, layer, q_head, kv_head)
        if vecs is None:
            continue
        # Use last question token
        qi = q_tokens - 1
        qpos = ctx_len + qi
        q = vecs["Q_rope"][qi]
        K_full = np.concatenate([K_ctx, vecs["K_rope"][:qi+1]], axis=0)
        V_full = np.concatenate([V_ctx, vecs["V"][:qi+1]], axis=0)
        n_causal = qpos + 1
        full_out, logits, _ = full_attention(q, K_full[:n_causal], V_full[:n_causal], HEAD_DIM)
        lw_start = max(N_SINK, n_causal - LOCAL_WINDOW)
        special_start = min(lw_start, ctx_len)
        special_idx = np.concatenate([
            np.arange(N_SINK, dtype=np.intp),
            np.arange(special_start, n_causal, dtype=np.intp),
        ])
        candidate_idx = np.arange(N_SINK, special_start, dtype=np.intp)
        problem = AttentionInput(
            query=q, keys=K_full[:n_causal], values=V_full[:n_causal],
            head_dim=HEAD_DIM, logits=logits,
            special_idx=special_idx, candidate_idx=candidate_idx)
        problems.append((problem, full_out))
    # Subsample
    if len(problems) > max_total:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(problems), max_total, replace=False)
        problems = [problems[i] for i in sorted(sel)]
    return problems

# %%
# ── 1d. Evaluation helpers ──
from src.algorithms.wildcat2.weighted_attention import weighted_attention

def evaluate_coreset_on_problems(K_c, V_c, w_c, test_problems,
                                  exact_denominator=False):
    """Evaluate a (K,V,w) coreset on precomputed test problems.
    Returns mean relative L2 error."""
    errors = []
    dev = device
    for problem, full_out in test_problems:
        sp_idx = problem.special_idx
        n_sp = len(sp_idx)
        scale = 1.0 / np.sqrt(HEAD_DIM)

        keys_all = torch.as_tensor(problem.keys, dtype=torch.float32, device=dev).unsqueeze(0)
        vals_all = torch.as_tensor(problem.values, dtype=torch.float32, device=dev).unsqueeze(0)

        k_c_t = torch.as_tensor(K_c, dtype=torch.float32, device=dev).unsqueeze(0)
        v_c_t = torch.as_tensor(V_c, dtype=torch.float32, device=dev).unsqueeze(0)
        w_c_t = torch.as_tensor(w_c, dtype=torch.float32, device=dev).unsqueeze(0)
        v_eff = v_c_t * w_c_t.unsqueeze(-1)

        if n_sp > 0:
            sp_k = keys_all[:, sp_idx, :]
            sp_v = vals_all[:, sp_idx, :]
            sp_w = torch.ones((1, n_sp), dtype=torch.float32, device=dev)
            core_k = torch.cat([sp_k, k_c_t], dim=1)
            core_v = torch.cat([sp_v, v_eff], dim=1)
            core_w = torch.cat([sp_w, w_c_t], dim=-1)
        else:
            core_k, core_v, core_w = k_c_t, v_eff, w_c_t

        q_t = torch.as_tensor(problem.query, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
        vmin = vals_all.amin(dim=-2, keepdim=True)
        vmax = vals_all.amax(dim=-2, keepdim=True)

        all_logits = None
        if exact_denominator and problem.logits is not None:
            all_logits = torch.as_tensor(problem.logits, dtype=torch.float32, device=dev)

        out_t = weighted_attention(q_t, core_k, core_v, core_w, scale, vmin, vmax,
                                   all_logits=all_logits)
        output = out_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        err = relative_l2_error(output, full_out)
        errors.append(err)
    return float(np.mean(errors))

def evaluate_baselines(test_problems, budgets):
    """Compute IdealTopK and vAttention errors for all budgets."""
    from src.algorithms.idealized_methods import IdealTopK, VAttentionOracle
    rng = np.random.default_rng(SEED)
    topk = IdealTopK()
    vattn = VAttentionOracle()
    results = {"IdealTopK": {}, "vAttention": {}}
    for b in budgets:
        topk_errs, vattn_errs = [], []
        for problem, full_out in test_problems:
            out = topk.run(problem, b, rng)
            topk_errs.append(relative_l2_error(out.output, full_out))
            out = vattn.run(problem, b, rng)
            vattn_errs.append(relative_l2_error(out.output, full_out))
        results["IdealTopK"][b] = float(np.mean(topk_errs))
        results["vAttention"][b] = float(np.mean(vattn_errs))
    return results

# %%
# ── 1e. Plotting helpers ──
def plot_error_vs_budget(results, baselines, budgets, title, out_path=None):
    """5 subplots (per head). results[head_label][method][budget] = error."""
    head_labels = sorted(results.keys(),
                         key=lambda l: next(h["entropy"] for h in HEADS if h["label"]==l))
    n = len(head_labels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, hl in enumerate(head_labels):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        head = next(h for h in HEADS if h["label"] == hl)

        # Baselines
        if hl in baselines:
            bl = baselines[hl]
            for bl_name, bl_style in [("IdealTopK", "r--"), ("vAttention", "r-")]:
                if bl_name in bl:
                    x = [b for b in budgets if b in bl[bl_name]]
                    y = [bl[bl_name][b] for b in x]
                    ax.plot(x, y, bl_style, marker="x" if "TopK" in bl_name else "+",
                            lw=2, ms=8, label=bl_name, zorder=10)

        # Methods
        for method in INIT_ORDER:
            if method not in results[hl]:
                continue
            x = [b for b in budgets if b in results[hl][method]]
            y = [results[hl][method][b] for b in x]
            if x:
                ax.plot(x, y, color=INIT_COLORS.get(method, "gray"),
                        marker="o", lw=2, ms=5, label=method)

        ax.set_title(f"{hl} (ent={head['entropy']:.2f})", fontsize=10)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Budget"); ax.grid(True, alpha=0.3)
        if c == 0: ax.set_ylabel("Rel L2 Error")
        if i == 0: ax.legend(fontsize=6, loc="upper right")

    for i in range(len(head_labels), rows*cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show(); plt.close(fig)

def plot_training_curves(histories, budgets, head_label, entropy, out_path=None):
    """Per-budget subplot showing val (solid) and train (dashed) loss per init."""
    n_b = len(budgets)
    fig, axes = plt.subplots(1, n_b, figsize=(5*n_b, 4), squeeze=False)
    fig.suptitle(f"Training Curves — {head_label} (ent={entropy:.2f})",
                 fontsize=14, fontweight="bold")
    for bi, b in enumerate(budgets):
        ax = axes[0][bi]
        for method in INIT_ORDER:
            if method not in histories or b not in histories[method]:
                continue
            h = histories[method][b]
            color = INIT_COLORS.get(method, "gray")
            if h.get("val_loss"):
                ax.plot(h["val_loss"], color=color, label=method, lw=1.5)
            if h.get("train_loss"):
                ax.plot(h["train_loss"], color=color, ls="--", alpha=0.4, lw=0.8)
        ax.set_title(f"Budget = {b}", fontsize=11)
        ax.set_xlabel("Step"); ax.set_yscale("log")
        if bi == 0: ax.set_ylabel("Loss (rel L2)")
        if bi == n_b - 1: ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show(); plt.close(fig)

# %%
# ── 1f. Load data for all heads ──
print("Loading data for all heads...")
train_idx, test_idx = get_train_test_conv_indices(VECTORS_DIR)
print(f"Train convs: {len(train_idx)}, Test convs: {len(test_idx)}")

head_data = {}
for head in HEADS:
    label = head["label"]
    layer, q_head, kv_head = head["layer"], head["q_head"], head["kv_head"]
    t0 = time.time()

    ctx = load_head_context(VECTORS_DIR, layer, q_head, kv_head)
    ctx_len = ctx["K_rope"].shape[0]
    sp_idx, cand_idx = compute_special_indices(ctx_len, N_SINK, LOCAL_WINDOW)

    Q_train = load_question_queries(VECTORS_DIR, train_idx, layer, q_head, kv_head)
    test_problems = build_test_problems(VECTORS_DIR, test_idx, layer, q_head, kv_head,
                                        ctx["K_rope"], ctx["V"])

    head_data[label] = {
        **ctx, "ctx_len": ctx_len,
        "sp_idx": sp_idx, "cand_idx": cand_idx,
        "Q_train": Q_train, "test_problems": test_problems,
        "layer": layer, "q_head": q_head, "kv_head": kv_head,
    }
    print(f"  {label}: ctx={ctx_len}, train_q={Q_train.shape[0]}, "
          f"test={len(test_problems)}, {time.time()-t0:.1f}s")

# %% [markdown]
# ## Section 2: Initialization-Only Comparison

# %%
# ── 2a. Initialization functions ──
from src.core import flat_kmeans

def init_random_gauss(budget, d, n_cand, rng):
    K = rng.standard_normal((budget, d)).astype(np.float32) * 0.1
    V = rng.standard_normal((budget, d)).astype(np.float32) * 0.1
    w = np.full(budget, max(n_cand / budget, 1e-8), dtype=np.float32)
    return K, V, w

def init_random_keys(budget, K_ctx, V_ctx, cand_idx, rng):
    n = len(cand_idx)
    pick = rng.choice(n, min(budget, n), replace=False)
    K = K_ctx[cand_idx[pick]].astype(np.float32)
    V = V_ctx[cand_idx[pick]].astype(np.float32)
    w = np.full(len(pick), max(n / budget, 1e-8), dtype=np.float32)
    if len(pick) < budget:
        pad = budget - len(pick)
        K = np.concatenate([K, np.zeros((pad, K.shape[1]), np.float32)])
        V = np.concatenate([V, np.zeros((pad, V.shape[1]), np.float32)])
        w = np.concatenate([w, np.ones(pad, np.float32)])
    return K, V, w

def init_first_tokens(budget, K_ctx, V_ctx, cand_idx):
    n = len(cand_idx)
    pick = min(budget, n)
    K = K_ctx[cand_idx[:pick]].copy().astype(np.float32)
    V = V_ctx[cand_idx[:pick]].copy().astype(np.float32)
    w = np.full(pick, max(n / budget, 1e-8), dtype=np.float32)
    if pick < budget:
        pad = budget - pick
        K = np.concatenate([K, np.zeros((pad, K.shape[1]), np.float32)])
        V = np.concatenate([V, np.zeros((pad, V.shape[1]), np.float32)])
        w = np.concatenate([w, np.ones(pad, np.float32)])
    return K, V, w

def init_kmeans(budget, K_ctx, V_ctx, cand_idx, rng):
    ck = K_ctx[cand_idx].astype(np.float32)
    cv = V_ctx[cand_idx].astype(np.float32)
    n = len(cand_idx)
    n_clusters = min(budget, n)
    centroids, labels = flat_kmeans(ck, n_clusters, seed=int(rng.integers(1<<30)))
    v_new = np.zeros((n_clusters, ck.shape[1]), np.float32)
    counts = np.zeros(n_clusters, np.float64)
    for c in range(n_clusters):
        mask = labels == c
        cnt = int(mask.sum())
        counts[c] = cnt
        if cnt > 0: v_new[c] = cv[mask].mean(axis=0)
    mass = counts / max(counts.sum(), 1.0) * float(n)
    w = np.clip(mass, 1e-8, None).astype(np.float32)
    K = centroids.astype(np.float32)
    if n_clusters < budget:
        pad = budget - n_clusters
        K = np.concatenate([K, rng.standard_normal((pad, ck.shape[1])).astype(np.float32)*0.02])
        v_new = np.concatenate([v_new, rng.standard_normal((pad, ck.shape[1])).astype(np.float32)*0.02])
        w = np.concatenate([w, np.ones(pad, np.float32)])
    return K, v_new, w

def init_mqbeta(budget, K_ctx, V_ctx, cand_idx, Q_train, rng):
    from src.algorithms.mq_beta_cluster import _weighted_kmeans
    ck = K_ctx[cand_idx].astype(np.float64)
    cv = V_ctx[cand_idx].astype(np.float32)
    n = len(cand_idx)
    d = ck.shape[1]
    n_clusters = min(budget, n)
    Q = Q_train.astype(np.float64)
    sqrt_d = np.sqrt(d)
    if len(Q) == 0:
        # Fallback to plain kmeans if no training queries
        return init_kmeans(budget, K_ctx, V_ctx, cand_idx, rng)
    # M_Q
    M_Q = Q.T @ Q + 1e-6 * np.eye(d)
    eigvals, eigvecs = np.linalg.eigh(M_Q)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_eig = np.sqrt(eigvals)
    # Rho importance — numerically stable per-row softmax
    BATCH = 500
    rho = np.zeros(n, np.float64)
    for b0 in range(0, len(Q), BATCH):
        b1 = min(b0+BATCH, len(Q))
        logits = (Q[b0:b1] @ ck.T) / sqrt_d
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        np.clip(logits_shifted, -50, 0, out=logits_shifted)  # prevent underflow
        rho += np.sum(np.exp(logits_shifted), axis=0)
    # Clean up any NaN/Inf
    rho = np.nan_to_num(rho, nan=1.0, posinf=1.0, neginf=0.0)
    rho = np.maximum(rho, 1e-30)
    # Weighted k-means in M_Q space
    K_z = (ck @ eigvecs * sqrt_eig[None, :]).astype(np.float32)
    _, labels = _weighted_kmeans(K_z, rho, n_clusters, seed=int(rng.integers(1<<30)), n_iter=50)
    # Cluster stats
    K_new = np.zeros((n_clusters, d), np.float32)
    V_new = np.zeros((n_clusters, d), np.float32)
    counts = np.zeros(n_clusters, np.float64)
    for c in range(n_clusters):
        mask = labels == c
        cnt = int(mask.sum())
        counts[c] = cnt
        if cnt > 0:
            K_new[c] = K_ctx[cand_idx[mask]].mean(axis=0)
            V_new[c] = cv[mask].mean(axis=0)
    mass = counts / max(counts.sum(), 1.0) * float(n)
    w = np.clip(mass, 1e-8, None).astype(np.float32)
    if n_clusters < budget:
        pad = budget - n_clusters
        K_new = np.concatenate([K_new, rng.standard_normal((pad, d)).astype(np.float32)*0.02])
        V_new = np.concatenate([V_new, rng.standard_normal((pad, d)).astype(np.float32)*0.02])
        w = np.concatenate([w, np.ones(pad, np.float32)])
    return K_new, V_new, w

def init_tfcfw_omp(budget, K_ctx, V_ctx, cand_idx, sp_idx, Q_train, head_dim):
    """TFCFW-lq with omp oracle — no L-BFGS, just selection + IRLS values."""
    from src.algorithms.tensor_fcfw_lq.select_lq import select_lq_coreset
    n_cand = len(cand_idx)
    if n_cand == 0 or budget <= 0:
        return (np.zeros((budget, head_dim), np.float32),
                np.zeros((budget, head_dim), np.float32),
                np.ones(budget, np.float32))
    scale = 1.0 / np.sqrt(head_dim)
    n_causal = max(cand_idx.max()+1, sp_idx.max()+1) if len(sp_idx) > 0 else cand_idx.max()+1
    keys_t = torch.as_tensor(K_ctx[:n_causal], dtype=torch.float32, device=device)
    vals_t = torch.as_tensor(V_ctx[:n_causal], dtype=torch.float32, device=device)
    probes = torch.as_tensor(Q_train, dtype=torch.float32, device=device)
    sp_t = torch.as_tensor(sp_idx, dtype=torch.long, device=device)
    cand_t = torch.as_tensor(cand_idx, dtype=torch.long, device=device)
    sel_idx, syn_vals, weights, _ = select_lq_coreset(
        probes, keys_t, vals_t, sp_t, cand_t, scale,
        budget=budget, oracle="omp", irls_iters=5, rcond=1e-3,
    )
    K_sel = keys_t[cand_t[sel_idx]].cpu().numpy().astype(np.float32)
    V_sel = syn_vals.cpu().numpy().astype(np.float32)
    w_sel = weights.cpu().numpy().astype(np.float32)
    if len(K_sel) < budget:
        pad = budget - len(K_sel)
        K_sel = np.concatenate([K_sel, np.zeros((pad, head_dim), np.float32)])
        V_sel = np.concatenate([V_sel, np.zeros((pad, head_dim), np.float32)])
        w_sel = np.concatenate([w_sel, np.ones(pad, np.float32)])
    return K_sel, V_sel, w_sel

def init_kvsculpt(budget, K_ctx, V_ctx, cand_idx, sp_idx, Q_train, head_dim):
    """KVSculpt init: top-k by accumulated attention mass."""
    scale = 1.0 / np.sqrt(head_dim)
    n_cand = len(cand_idx)
    if n_cand == 0 or budget <= 0:
        return (np.zeros((budget, head_dim), np.float32),
                np.zeros((budget, head_dim), np.float32),
                np.ones(budget, np.float32))
    Q_t = torch.as_tensor(Q_train, dtype=torch.float32, device=device)
    K_cand = torch.as_tensor(K_ctx[cand_idx], dtype=torch.float32, device=device)
    V_cand = torch.as_tensor(V_ctx[cand_idx], dtype=torch.float32, device=device)
    n_causal = max(cand_idx.max()+1, sp_idx.max()+1) if len(sp_idx) > 0 else cand_idx.max()+1
    K_all = torch.as_tensor(K_ctx[:n_causal], dtype=torch.float32, device=device)
    # Attention mass per candidate
    scores = scale * (Q_t @ K_all.T)
    attn = torch.softmax(scores, dim=-1)
    cand_t = torch.as_tensor(cand_idx, dtype=torch.long, device=device)
    importance = attn.index_select(dim=1, index=cand_t).sum(dim=0)
    pick = min(budget, n_cand)
    top_local = torch.topk(importance, k=pick, largest=True).indices
    K_init = K_cand[top_local].cpu().numpy().astype(np.float32)
    V_init = V_cand[top_local].cpu().numpy().astype(np.float32)
    w_init = np.ones(pick, dtype=np.float32)
    if pick < budget:
        pad = budget - pick
        K_init = np.concatenate([K_init, np.zeros((pad, head_dim), np.float32)])
        V_init = np.concatenate([V_init, np.zeros((pad, head_dim), np.float32)])
        w_init = np.concatenate([w_init, np.ones(pad, np.float32)])
    return K_init, V_init, w_init

def compute_all_inits(head_label, budget, rng):
    """Compute all 7 initializations for a given head and budget."""
    hd = head_data[head_label]
    K_ctx, V_ctx = hd["K_rope"], hd["V"]
    cand_idx, sp_idx = hd["cand_idx"], hd["sp_idx"]
    Q_train = hd["Q_train"]
    n_cand = len(cand_idx)
    d = HEAD_DIM
    inits = {}
    inits["RandomGauss"] = init_random_gauss(budget, d, n_cand, rng)
    inits["RandomKeys"] = init_random_keys(budget, K_ctx, V_ctx, cand_idx, rng)
    inits["FirstTokens"] = init_first_tokens(budget, K_ctx, V_ctx, cand_idx)
    inits["KMeans"] = init_kmeans(budget, K_ctx, V_ctx, cand_idx, rng)
    inits["MQBeta"] = init_mqbeta(budget, K_ctx, V_ctx, cand_idx, Q_train, rng)
    inits["TFCFW-omp"] = init_tfcfw_omp(budget, K_ctx, V_ctx, cand_idx, sp_idx, Q_train, d)
    inits["KVSculpt"] = init_kvsculpt(budget, K_ctx, V_ctx, cand_idx, sp_idx, Q_train, d)
    return inits

# %%
# ── 2b. Run init-only evaluation ──
print("\n" + "="*60)
print("Section 2: Init-Only Comparison")
print("="*60)

results_init = defaultdict(lambda: defaultdict(dict))
baselines_all = {}

for head in HEADS:
    label = head["label"]
    hd = head_data[label]
    t0 = time.time()
    print(f"\n  {label} (ent={head['entropy']:.2f})...")

    # Baselines
    baselines_all[label] = evaluate_baselines(hd["test_problems"], BUDGETS)

    rng = np.random.default_rng(SEED)
    for budget in BUDGETS:
        inits = compute_all_inits(label, budget, rng)
        for method, (K_c, V_c, w_c) in inits.items():
            err = evaluate_coreset_on_problems(K_c, V_c, w_c, hd["test_problems"])
            results_init[label][method][budget] = err
        # Print progress
        best = min(results_init[label][m][budget] for m in inits)
        print(f"    B={budget}: best={best:.6f}", end="")
    print(f"  [{time.time()-t0:.0f}s]")

# %%
# ── 2c. Plot init-only results ──
plot_error_vs_budget(
    results_init, baselines_all, BUDGETS,
    f"Init-Only Comparison — {DATASET}\nNo training, {len(head_data[HEADS[0]['label']]['test_problems'])} test queries",
    out_path="init_only_comparison.png",
)

# %% [markdown]
# ## Section 3: Adam Post-Training

# %%
# ── 3a. Adam training function ──
from src.algorithms.learned.learn_coreset import (
    _precompute_targets, _forward_pred, _loss_value, compute_special_indices,
)
import torch.nn as nn

def train_adam_from_init(K_init, V_init, w_init, Q_train, K_ctx, V_ctx,
                         head_dim, ref_pos, n_sink, local_window,
                         cfg, exact_denominator=False, dev=None, seed=42):
    """Adam optimization from external (K, V, w) initialization.
    Returns (K_trained, V_trained, w_trained, history)."""
    if dev is None: dev = device
    scale = 1.0 / np.sqrt(head_dim)
    n_causal = ref_pos + 1
    sp_idx, cand_idx = compute_special_indices(n_causal, n_sink, local_window)

    probes = torch.as_tensor(Q_train, dtype=torch.float32, device=dev)
    keys_ref = torch.as_tensor(K_ctx[:n_causal], dtype=torch.float32, device=dev)
    vals_ref = torch.as_tensor(V_ctx[:n_causal], dtype=torch.float32, device=dev)

    # Convert init to torch parameters
    k_new = nn.Parameter(torch.as_tensor(K_init, dtype=torch.float32, device=dev).clone())
    v_new = nn.Parameter(torch.as_tensor(V_init, dtype=torch.float32, device=dev).clone())
    log_w = nn.Parameter(torch.log(torch.as_tensor(
        np.clip(w_init, 1e-8, None), dtype=torch.float32, device=dev).clone()))

    # Precompute targets
    with torch.no_grad():
        consts = _precompute_targets(probes, keys_ref, vals_ref, sp_idx, None, scale)
    del keys_ref, vals_ref

    # Train/val split
    rng = np.random.default_rng(seed)
    m = probes.shape[0]
    perm = rng.permutation(m)
    n_val = max(1, int(m * cfg["val_fraction"])) if m >= 10 else 0
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long, device=dev)
    train_idx = torch.as_tensor(perm[n_val:], dtype=torch.long, device=dev)
    if train_idx.numel() == 0:
        train_idx = torch.arange(m, device=dev)

    optimizer = torch.optim.Adam([k_new, v_new, log_w], lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["lr_decay_step"], gamma=cfg["lr_decay_gamma"])
    torch_gen = torch.Generator(device=dev)
    torch_gen.manual_seed(seed)

    best_val = float("inf")
    best = (k_new.detach().clone(), v_new.detach().clone(), log_w.detach().clone())
    stale = 0
    history = {"train_loss": [], "val_loss": []}
    batch_size = cfg["batch_size"]

    for step in range(cfg["n_steps"]):
        if train_idx.numel() <= batch_size:
            batch = train_idx
        else:
            sel = torch.randperm(train_idx.numel(), generator=torch_gen, device=dev)[:batch_size]
            batch = train_idx[sel]

        pred, target = _forward_pred(consts, batch, k_new, v_new, log_w, scale, exact_denominator)
        loss_t = _loss_value(pred, target, cfg["loss"], cfg["rel_l2_floor"])

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        scheduler.step()

        history["train_loss"].append(float(loss_t.detach()))

        if val_idx.numel() > 0:
            with torch.no_grad():
                vp, vt = _forward_pred(consts, val_idx, k_new, v_new, log_w, scale, exact_denominator)
                val_loss = float(_loss_value(vp, vt, cfg["loss"], cfg["rel_l2_floor"]))
            history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best = (k_new.detach().clone(), v_new.detach().clone(), log_w.detach().clone())
                stale = 0
            else:
                stale += 1
                if stale >= cfg["early_stop_patience"]:
                    break

    if val_idx.numel() > 0:
        k_new, v_new, log_w = best

    K_out = k_new.detach().cpu().numpy().astype(np.float32)
    V_out = v_new.detach().cpu().numpy().astype(np.float32)
    w_out = log_w.detach().exp().cpu().numpy().astype(np.float32)

    # Cleanup
    del consts, probes, k_new, v_new, log_w
    torch.cuda.empty_cache()

    return K_out, V_out, w_out, history

# %%
# ── 3b. Run Adam training for all inits ──
print("\n" + "="*60)
print("Section 3: Adam Post-Training (3000 steps)")
print("="*60)

results_adam = defaultdict(lambda: defaultdict(dict))
histories_adam = defaultdict(lambda: defaultdict(dict))

for head in HEADS:
    label = head["label"]
    hd = head_data[label]
    ref_pos = hd["ctx_len"] - 1
    print(f"\n  {label} (ent={head['entropy']:.2f})...")

    rng = np.random.default_rng(SEED)
    for budget in BUDGETS:
        inits = compute_all_inits(label, budget, rng)
        for method, (K_init, V_init, w_init) in inits.items():
            t0 = time.time()
            K_t, V_t, w_t, hist = train_adam_from_init(
                K_init, V_init, w_init,
                hd["Q_train"], hd["K_rope"], hd["V"],
                HEAD_DIM, ref_pos, N_SINK, LOCAL_WINDOW,
                ADAM_CFG, exact_denominator=False, seed=SEED+budget,
            )
            err = evaluate_coreset_on_problems(K_t, V_t, w_t, hd["test_problems"])
            results_adam[label][method][budget] = err
            histories_adam[label][method][budget] = hist
            steps = len(hist["train_loss"])
            print(f"    {method} B={budget}: err={err:.6f} ({steps} steps, {time.time()-t0:.0f}s)")
        gc.collect(); torch.cuda.empty_cache()

# %%
# ── 3c. Plot Adam results ──
plot_error_vs_budget(
    results_adam, baselines_all, BUDGETS,
    f"After Adam Training — {DATASET}\n3000 steps, lr=0.01, exact_denom=False",
    out_path="adam_comparison.png",
)

# %%
# ── 3d. Plot training curves ──
for head in HEADS:
    label = head["label"]
    if label in histories_adam:
        plot_training_curves(
            histories_adam[label], BUDGETS, label, head["entropy"],
            out_path=f"training_curves_adam_{label}.png",
        )

# %% [markdown]
# ## Section 4: L-BFGS (KVSculpt-style) Post-Training

# %%
# ── 4a. L-BFGS training function ──
def train_lbfgs_from_init(K_init, V_init, w_init, Q_train, K_ctx, V_ctx,
                           head_dim, ref_pos, n_sink, local_window, sp_idx,
                           cfg, exact_denominator=False, dev=None, seed=42):
    """L-BFGS optimization (KVSculpt-style): L-BFGS on keys, ridge V solve.
    Returns (K_trained, V_trained, w_trained, history)."""
    if dev is None: dev = device
    scale = 1.0 / np.sqrt(head_dim)
    n_causal = ref_pos + 1

    Q_t = torch.as_tensor(Q_train, dtype=torch.float32, device=dev)
    K_full = torch.as_tensor(K_ctx[:n_causal], dtype=torch.float32, device=dev)
    V_full = torch.as_tensor(V_ctx[:n_causal], dtype=torch.float32, device=dev)

    # Special token keys/values (retained exactly)
    sp_t = torch.as_tensor(sp_idx, dtype=torch.long, device=dev)
    k_ret = K_full[sp_t]
    v_ret = V_full[sp_t]

    # Fold mass into values for init, then use unit weights
    w_t = torch.as_tensor(np.clip(w_init, 1e-8, None), dtype=torch.float32, device=dev)
    k_c = nn.Parameter(torch.as_tensor(K_init, dtype=torch.float32, device=dev).clone())
    v_c = torch.as_tensor(V_init, dtype=torch.float32, device=dev).clone() * w_t.unsqueeze(-1)

    # Compute targets
    with torch.no_grad():
        target_y, target_lse = _lbfgs_targets(Q_t, K_full, V_full, scale)

    history = {"train_loss": []}

    for step in range(cfg["n_k_steps"]):
        # Ridge V solve
        if step % cfg["v_solve_every"] == 0:
            with torch.no_grad():
                v_c = _ridge_v_solve(Q_t, k_c, k_ret, v_ret, target_y, scale,
                                      cfg["ridge_lambda"])

        # L-BFGS step on keys
        optimizer = torch.optim.LBFGS(
            [k_c], lr=cfg["lbfgs_lr"], max_iter=cfg["lbfgs_inner_iter"],
            line_search_fn="strong_wolfe",
        )
        def closure():
            optimizer.zero_grad()
            pred_y, pred_lse = _lbfgs_pred(Q_t, k_c, k_ret, v_c, v_ret, scale)
            loss = (pred_y - target_y).pow(2).mean() + (pred_lse - target_lse).pow(2).mean()
            loss.backward()
            return loss
        loss_val = optimizer.step(closure)
        history["train_loss"].append(float(loss_val))

    # Final V solve
    with torch.no_grad():
        v_c = _ridge_v_solve(Q_t, k_c, k_ret, v_ret, target_y, scale, cfg["ridge_lambda"])

    K_out = k_c.detach().cpu().numpy().astype(np.float32)
    V_out = v_c.detach().cpu().numpy().astype(np.float32)
    w_out = np.ones(K_out.shape[0], dtype=np.float32)

    del Q_t, K_full, V_full, k_c, v_c
    torch.cuda.empty_cache()
    return K_out, V_out, w_out, history

def _lbfgs_targets(queries, keys, values, scale):
    scores = scale * (queries @ keys.T)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    target = attn @ values
    return target, lse

def _lbfgs_pred(queries, k_c, k_ret, v_c, v_ret, scale):
    k_cat = torch.cat([k_c, k_ret], dim=0)
    v_cat = torch.cat([v_c, v_ret], dim=0)
    scores = scale * (queries @ k_cat.T)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    pred = attn @ v_cat
    return pred, lse

def _ridge_v_solve(queries, k_c, k_ret, v_ret, target_y, scale, ridge_lambda):
    k_cat = torch.cat([k_c, k_ret], dim=0)
    scores = scale * (queries @ k_cat.T)
    attn = torch.softmax(scores, dim=-1)
    n_c = k_c.shape[0]
    a_c = attn[:, :n_c]  # [m, n_c]
    a_r = attn[:, n_c:]  # [m, n_sp]
    residual = target_y - a_r @ v_ret  # [m, d]
    ata = a_c.T @ a_c + ridge_lambda * torch.eye(n_c, device=a_c.device, dtype=a_c.dtype)
    atb = a_c.T @ residual
    v_c = torch.linalg.solve(ata, atb)
    return v_c

# %%
# ── 4b. Run L-BFGS training ──
print("\n" + "="*60)
print("Section 4: L-BFGS Post-Training (100 K-steps)")
print("="*60)

results_lbfgs = defaultdict(lambda: defaultdict(dict))
histories_lbfgs = defaultdict(lambda: defaultdict(dict))

for head in HEADS:
    label = head["label"]
    hd = head_data[label]
    ref_pos = hd["ctx_len"] - 1
    print(f"\n  {label} (ent={head['entropy']:.2f})...")

    rng = np.random.default_rng(SEED)
    for budget in BUDGETS:
        inits = compute_all_inits(label, budget, rng)
        for method, (K_init, V_init, w_init) in inits.items():
            t0 = time.time()
            K_t, V_t, w_t, hist = train_lbfgs_from_init(
                K_init, V_init, w_init,
                hd["Q_train"], hd["K_rope"], hd["V"],
                HEAD_DIM, ref_pos, N_SINK, LOCAL_WINDOW, hd["sp_idx"],
                LBFGS_CFG, exact_denominator=False, seed=SEED+budget,
            )
            err = evaluate_coreset_on_problems(K_t, V_t, w_t, hd["test_problems"])
            results_lbfgs[label][method][budget] = err
            histories_lbfgs[label][method][budget] = hist
            print(f"    {method} B={budget}: err={err:.6f} ({time.time()-t0:.0f}s)")
        gc.collect(); torch.cuda.empty_cache()

# %%
# ── 4c. Plot L-BFGS results ──
plot_error_vs_budget(
    results_lbfgs, baselines_all, BUDGETS,
    f"After L-BFGS Training — {DATASET}\n100 K-steps, ridge V solve, exact_denom=False",
    out_path="lbfgs_comparison.png",
)

# %%
# ── 4d. Plot L-BFGS training curves ──
for head in HEADS:
    label = head["label"]
    if label in histories_lbfgs:
        plot_training_curves(
            histories_lbfgs[label], BUDGETS, label, head["entropy"],
            out_path=f"training_curves_lbfgs_{label}.png",
        )

# %% [markdown]
# ## Section 4b: Full-Softmax L-BFGS
# Same loss as Adam (relative L2 on full softmax output), same learnable params (K, V, w),
# but using L-BFGS optimizer instead of Adam. Compute-matched: 300 steps × 10 inner iters
# ≈ 3000 gradient evaluations, same as Adam's 3000 steps.

# %%
# ── 4b-a. Full-softmax L-BFGS training function ──
def train_lbfgs_full_softmax(K_init, V_init, w_init, Q_train, K_ctx, V_ctx,
                              head_dim, ref_pos, n_sink, local_window,
                              cfg, exact_denominator=False, dev=None, seed=42):
    """L-BFGS optimizing the SAME loss as Adam (relative L2 on full softmax output).
    Learns K, V, and w (all three), just like Adam.
    300 outer steps × max_iter=10 ≈ 3000 gradient evals to match Adam compute.
    Returns (K_trained, V_trained, w_trained, history)."""
    if dev is None: dev = device
    scale = 1.0 / np.sqrt(head_dim)
    n_causal = ref_pos + 1
    sp_idx, cand_idx = compute_special_indices(n_causal, n_sink, local_window)

    probes = torch.as_tensor(Q_train, dtype=torch.float32, device=dev)
    keys_ref = torch.as_tensor(K_ctx[:n_causal], dtype=torch.float32, device=dev)
    vals_ref = torch.as_tensor(V_ctx[:n_causal], dtype=torch.float32, device=dev)

    k_new = nn.Parameter(torch.as_tensor(K_init, dtype=torch.float32, device=dev).clone())
    v_new = nn.Parameter(torch.as_tensor(V_init, dtype=torch.float32, device=dev).clone())
    log_w = nn.Parameter(torch.log(torch.as_tensor(
        np.clip(w_init, 1e-8, None), dtype=torch.float32, device=dev).clone()))

    with torch.no_grad():
        consts = _precompute_targets(probes, keys_ref, vals_ref, sp_idx, None, scale)
    del keys_ref, vals_ref

    # Use full batch (L-BFGS needs deterministic closure)
    m = probes.shape[0]
    rng_np = np.random.default_rng(seed)
    perm = rng_np.permutation(m)
    n_val = max(1, int(m * cfg["val_fraction"])) if m >= 10 else 0
    val_idx = torch.as_tensor(perm[:n_val], dtype=torch.long, device=dev)
    train_idx = torch.as_tensor(perm[n_val:], dtype=torch.long, device=dev)
    if train_idx.numel() == 0:
        train_idx = torch.arange(m, device=dev)

    # 300 outer steps × 10 inner = ~3000 gradient evals (matching Adam)
    n_outer = 300
    max_inner = 10
    optimizer = torch.optim.LBFGS(
        [k_new, v_new, log_w], lr=0.5, max_iter=max_inner,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-9,
    )

    best_val = float("inf")
    best = (k_new.detach().clone(), v_new.detach().clone(), log_w.detach().clone())
    history = {"train_loss": [], "val_loss": []}
    stale = 0
    patience = 50  # early stop if no improvement for 50 outer steps

    for step in range(n_outer):
        def closure():
            optimizer.zero_grad()
            pred, target = _forward_pred(consts, train_idx, k_new, v_new, log_w,
                                          scale, exact_denominator)
            loss = _loss_value(pred, target, cfg["loss"], cfg["rel_l2_floor"])
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        history["train_loss"].append(float(loss_val))

        if val_idx.numel() > 0:
            with torch.no_grad():
                vp, vt = _forward_pred(consts, val_idx, k_new, v_new, log_w,
                                        scale, exact_denominator)
                val_loss = float(_loss_value(vp, vt, cfg["loss"], cfg["rel_l2_floor"]))
            history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best = (k_new.detach().clone(), v_new.detach().clone(), log_w.detach().clone())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if val_idx.numel() > 0:
        k_new, v_new, log_w = best

    K_out = k_new.detach().cpu().numpy().astype(np.float32)
    V_out = v_new.detach().cpu().numpy().astype(np.float32)
    w_out = log_w.detach().exp().cpu().numpy().astype(np.float32)

    del consts, probes, k_new, v_new, log_w
    torch.cuda.empty_cache()
    return K_out, V_out, w_out, history

# %%
# ── 4b-b. Run full-softmax L-BFGS ──
print("\n" + "="*60)
print("Section 4b: Full-Softmax L-BFGS (300 steps × 10 inner)")
print("="*60)

results_lbfgs_full = defaultdict(lambda: defaultdict(dict))
histories_lbfgs_full = defaultdict(lambda: defaultdict(dict))

for head in HEADS:
    label = head["label"]
    hd = head_data[label]
    ref_pos = hd["ctx_len"] - 1
    print(f"\n  {label} (ent={head['entropy']:.2f})...")

    rng = np.random.default_rng(SEED)
    for budget in BUDGETS:
        inits = compute_all_inits(label, budget, rng)
        for method, (K_init, V_init, w_init) in inits.items():
            t0 = time.time()
            K_t, V_t, w_t, hist = train_lbfgs_full_softmax(
                K_init, V_init, w_init,
                hd["Q_train"], hd["K_rope"], hd["V"],
                HEAD_DIM, ref_pos, N_SINK, LOCAL_WINDOW,
                ADAM_CFG, exact_denominator=False, seed=SEED+budget,
            )
            err = evaluate_coreset_on_problems(K_t, V_t, w_t, hd["test_problems"])
            results_lbfgs_full[label][method][budget] = err
            histories_lbfgs_full[label][method][budget] = hist
            steps = len(hist["train_loss"])
            print(f"    {method} B={budget}: err={err:.6f} ({steps} steps, {time.time()-t0:.0f}s)")
        gc.collect(); torch.cuda.empty_cache()

# %%
# ── 4b-c. Plot full-softmax L-BFGS results ──
plot_error_vs_budget(
    results_lbfgs_full, baselines_all, BUDGETS,
    f"After Full-Softmax L-BFGS — {DATASET}\n300 steps×10 inner, same loss as Adam, exact_denom=False",
    out_path="lbfgs_full_softmax_comparison.png",
)

# %%
# ── 4b-d. Plot training curves ──
for head in HEADS:
    label = head["label"]
    if label in histories_lbfgs_full:
        plot_training_curves(
            histories_lbfgs_full[label], BUDGETS, label, head["entropy"],
            out_path=f"training_curves_lbfgs_full_{label}.png",
        )

# %% [markdown]
# ## Section 5: Adam vs L-BFGS Comparison

# %%
def plot_optimizer_comparison(results_dict, baselines, budgets, title, out_path=None):
    """Compare multiple optimizers. results_dict[optimizer_name][head][method][budget]=err."""
    opt_styles = {"Adam": ("-", "o"), "KVSculpt-LBFGS": ("--", "s"), "Full-LBFGS": (":", "D")}
    head_labels = sorted(list(list(results_dict.values())[0].keys()),
                         key=lambda l: next(h["entropy"] for h in HEADS if h["label"]==l))
    n = len(head_labels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, hl in enumerate(head_labels):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        head = next(h for h in HEADS if h["label"] == hl)

        if hl in baselines:
            bl = baselines[hl]
            for bn, bs in [("IdealTopK", "r--"), ("vAttention", "r-")]:
                if bn in bl:
                    x = [b for b in budgets if b in bl[bn]]
                    y = [bl[bn][b] for b in x]
                    ax.plot(x, y, bs, marker="x" if "TopK" in bn else "+",
                            lw=2, ms=8, label=bn, zorder=10)

        for method in INIT_ORDER:
            color = INIT_COLORS.get(method, "gray")
            for opt_name, (ls, marker) in opt_styles.items():
                if opt_name not in results_dict:
                    continue
                res = results_dict[opt_name]
                if method in res.get(hl, {}):
                    x = [b for b in budgets if b in res[hl][method]]
                    y = [res[hl][method][b] for b in x]
                    if x:
                        ax.plot(x, y, color=color, marker=marker, lw=2, ms=5,
                                ls=ls, label=f"{method} ({opt_name})")

        ax.set_title(f"{hl} (ent={head['entropy']:.2f})", fontsize=10)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Budget"); ax.grid(True, alpha=0.3)
        if c == 0: ax.set_ylabel("Rel L2 Error")
        if i == 0: ax.legend(fontsize=4, loc="upper right", ncol=3)

    for i in range(len(head_labels), rows*cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show(); plt.close(fig)

# All three optimizers
plot_optimizer_comparison(
    {"Adam": results_adam, "KVSculpt-LBFGS": results_lbfgs, "Full-LBFGS": results_lbfgs_full},
    baselines_all, BUDGETS,
    f"Adam vs KVSculpt-LBFGS vs Full-LBFGS — {DATASET}\n"
    f"Adam: 3000 steps | KVSculpt: 100 K-steps | Full-LBFGS: 300×10 steps",
    out_path="optimizer_comparison.png",
)

# %%
# Per-init comparison: for each init, which optimizer is best?
for method in ["MQBeta", "TFCFW-omp", "KVSculpt", "KMeans"]:
    head_labels = sorted(results_adam.keys(),
                         key=lambda l: next(h["entropy"] for h in HEADS if h["label"]==l))
    fig, axes = plt.subplots(1, len(head_labels), figsize=(5*len(head_labels), 4), squeeze=False)
    fig.suptitle(f"Optimizer Comparison — {method} init", fontsize=13, fontweight="bold")
    opt_data = {"Adam": results_adam, "KVSculpt-LBFGS": results_lbfgs, "Full-LBFGS": results_lbfgs_full}
    opt_colors = {"Adam": "#1f77b4", "KVSculpt-LBFGS": "#ff7f0e", "Full-LBFGS": "#2ca02c"}
    opt_ls = {"Adam": "-", "KVSculpt-LBFGS": "--", "Full-LBFGS": ":"}
    for i, hl in enumerate(head_labels):
        ax = axes[0][i]
        head = next(h for h in HEADS if h["label"] == hl)
        for opt_name, res in opt_data.items():
            if method in res.get(hl, {}):
                x = [b for b in BUDGETS if b in res[hl][method]]
                y = [res[hl][method][b] for b in x]
                if x: ax.plot(x, y, color=opt_colors[opt_name], ls=opt_ls[opt_name],
                              marker="o", lw=2, ms=5, label=opt_name)
        # Baselines
        if hl in baselines_all:
            bl = baselines_all[hl]
            if "IdealTopK" in bl:
                x = [b for b in BUDGETS if b in bl["IdealTopK"]]
                y = [bl["IdealTopK"][b] for b in x]
                ax.plot(x, y, "r--", marker="x", lw=1, ms=6, label="IdealTopK", alpha=0.5)
        ax.set_title(f"{hl} (ent={head['entropy']:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("Budget")
        if i == 0: ax.set_ylabel("Rel L2 Error"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = f"optimizer_per_init_{method}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show(); plt.close(fig)

# %% [markdown]
# ## Section 6: exact_denominator Ablation

# %%
print("\n" + "="*60)
print("Section 6: exact_denominator=True vs False")
print("="*60)

TOP_METHODS = ["MQBeta", "TFCFW-omp", "KVSculpt"]
results_exact = defaultdict(lambda: defaultdict(dict))

for head in HEADS:
    label = head["label"]
    hd = head_data[label]
    ref_pos = hd["ctx_len"] - 1
    print(f"\n  {label}...")

    rng = np.random.default_rng(SEED)
    for budget in BUDGETS:
        inits = compute_all_inits(label, budget, rng)
        for method in TOP_METHODS:
            if method not in inits:
                continue
            K_init, V_init, w_init = inits[method]
            K_t, V_t, w_t, _ = train_adam_from_init(
                K_init, V_init, w_init,
                hd["Q_train"], hd["K_rope"], hd["V"],
                HEAD_DIM, ref_pos, N_SINK, LOCAL_WINDOW,
                ADAM_CFG, exact_denominator=True, seed=SEED+budget,
            )
            err = evaluate_coreset_on_problems(K_t, V_t, w_t, hd["test_problems"],
                                                exact_denominator=True)
            results_exact[label][method][budget] = err
            print(f"    {method} B={budget}: exact=True err={err:.6f}")
    gc.collect(); torch.cuda.empty_cache()

# %%
# Plot exact_denom comparison
head_labels = sorted(results_exact.keys(),
                     key=lambda l: next(h["entropy"] for h in HEADS if h["label"]==l))
fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
fig.suptitle(f"exact_denominator: True vs False — {DATASET}", fontsize=13, fontweight="bold")
for i, hl in enumerate(head_labels):
    r, c = divmod(i, 3)
    ax = axes[r][c]
    head = next(h for h in HEADS if h["label"] == hl)
    for method in TOP_METHODS:
        color = INIT_COLORS.get(method, "gray")
        # False (from Section 3)
        if method in results_adam.get(hl, {}):
            x = [b for b in BUDGETS if b in results_adam[hl][method]]
            y = [results_adam[hl][method][b] for b in x]
            if x: ax.plot(x, y, color=color, marker="o", lw=2, label=f"{method} (False)")
        # True
        if method in results_exact.get(hl, {}):
            x = [b for b in BUDGETS if b in results_exact[hl][method]]
            y = [results_exact[hl][method][b] for b in x]
            if x: ax.plot(x, y, color=color, marker="s", lw=2, ls="--", label=f"{method} (True)")
    ax.set_title(f"{hl} (ent={head['entropy']:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Budget")
    if c == 0: ax.set_ylabel("Rel L2 Error")
    if i == 0: ax.legend(fontsize=7)
for i in range(len(head_labels), 6):
    r, c = divmod(i, 3)
    axes[r][c].set_visible(False)
plt.tight_layout()
plt.savefig("exact_denom_comparison.png", dpi=200, bbox_inches="tight")
print("Saved: exact_denom_comparison.png")
plt.show(); plt.close(fig)

# %% [markdown]
# ## Section 7: Training Query Source Comparison
# Self-study vs De-RoPE vs Context queries

# %%
# ── 7a. Generate De-RoPE queries ──
from src.algorithms.probe_queries import apply_rope

def build_derope_queries(Q_raw_context, ctx_len, n_queries, head_dim, rope_theta, seed=42):
    """Synthetic future queries from raw (pre-RoPE) context Q via apply_rope."""
    rng = np.random.default_rng(seed)
    n_available = Q_raw_context.shape[0]
    n_pick = min(n_queries, n_available)
    pick = rng.choice(n_available, n_pick, replace=False)
    content = Q_raw_context[pick].astype(np.float32)
    future_pos = np.arange(ctx_len, ctx_len + n_pick, dtype=np.float64)
    return apply_rope(content, future_pos, head_dim=head_dim, rope_theta=rope_theta)

print("\n" + "="*60)
print("Section 7: Training Query Source Comparison")
print("="*60)

QUERY_DEPENDENT_METHODS = ["MQBeta", "TFCFW-omp", "KVSculpt"]
results_qsource = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for head in HEADS:
    label = head["label"]
    hd = head_data[label]
    ref_pos = hd["ctx_len"] - 1
    n_train = hd["Q_train"].shape[0]
    print(f"\n  {label} (train_q={n_train})...")

    # Three query sources
    Q_selfstudy = hd["Q_train"]
    Q_derope = build_derope_queries(hd["Q_raw"], hd["ctx_len"], n_train, HEAD_DIM, ROPE_THETA)
    Q_context = hd["Q_rope"][max(0, hd["ctx_len"]-n_train):hd["ctx_len"]].astype(np.float32)

    sources = {"SelfStudy": Q_selfstudy, "DeRoPE": Q_derope, "Context": Q_context}

    for src_name, Q_src in sources.items():
        rng = np.random.default_rng(SEED)
        for budget in BUDGETS:
            for method in QUERY_DEPENDENT_METHODS:
                # Compute init with this query source
                K_ctx, V_ctx = hd["K_rope"], hd["V"]
                cand_idx, sp_idx = hd["cand_idx"], hd["sp_idx"]
                if method == "MQBeta":
                    K_i, V_i, w_i = init_mqbeta(budget, K_ctx, V_ctx, cand_idx, Q_src, rng)
                elif method == "TFCFW-omp":
                    K_i, V_i, w_i = init_tfcfw_omp(budget, K_ctx, V_ctx, cand_idx, sp_idx, Q_src, HEAD_DIM)
                elif method == "KVSculpt":
                    K_i, V_i, w_i = init_kvsculpt(budget, K_ctx, V_ctx, cand_idx, sp_idx, Q_src, HEAD_DIM)
                # Train with Adam using same query source
                K_t, V_t, w_t, _ = train_adam_from_init(
                    K_i, V_i, w_i, Q_src, K_ctx, V_ctx,
                    HEAD_DIM, ref_pos, N_SINK, LOCAL_WINDOW,
                    ADAM_CFG, exact_denominator=False, seed=SEED+budget)
                err = evaluate_coreset_on_problems(K_t, V_t, w_t, hd["test_problems"])
                results_qsource[label][method][src_name][budget] = err
                print(f"    {method}/{src_name} B={budget}: {err:.6f}")
        gc.collect(); torch.cuda.empty_cache()

# %%
# ── 7b. Plot query source comparison ──
SRC_COLORS = {"SelfStudy": "#1f77b4", "DeRoPE": "#ff7f0e", "Context": "#2ca02c"}
SRC_STYLES = {"SelfStudy": "-", "DeRoPE": "--", "Context": ":"}

for method in QUERY_DEPENDENT_METHODS:
    head_labels = sorted(results_qsource.keys(),
                         key=lambda l: next(h["entropy"] for h in HEADS if h["label"]==l))
    n = len(head_labels)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    fig.suptitle(f"Query Source Comparison — {method}", fontsize=13, fontweight="bold")

    for i, hl in enumerate(head_labels):
        ax = axes[0][i]
        head = next(h for h in HEADS if h["label"] == hl)
        for src_name in ["SelfStudy", "DeRoPE", "Context"]:
            if src_name in results_qsource[hl].get(method, {}):
                x = [b for b in BUDGETS if b in results_qsource[hl][method][src_name]]
                y = [results_qsource[hl][method][src_name][b] for b in x]
                if x:
                    ax.plot(x, y, color=SRC_COLORS[src_name], ls=SRC_STYLES[src_name],
                            marker="o", lw=2, ms=5, label=src_name)
        ax.set_title(f"{hl} (ent={head['entropy']:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("Budget")
        if i == 0: ax.set_ylabel("Rel L2 Error"); ax.legend(fontsize=8)

    plt.tight_layout()
    out = f"query_source_{method}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show(); plt.close(fig)

# %% [markdown]
# ## Summary
# All results saved as PNG plots. Key outputs:
# - `init_only_comparison.png` — Section 2
# - `adam_comparison.png` — Section 3
# - `training_curves_adam_*.png` — Section 3 training curves
# - `lbfgs_comparison.png` — Section 4 (KVSculpt-style: K-only + ridge V)
# - `lbfgs_full_softmax_comparison.png` — Section 4b (full-softmax L-BFGS: K+V+w)
# - `optimizer_comparison.png` — Section 5 (all 3 optimizers)
# - `optimizer_per_init_*.png` — Section 5 (per-init optimizer comparison)
# - `exact_denom_comparison.png` — Section 6
# - `query_source_*.png` — Section 7
