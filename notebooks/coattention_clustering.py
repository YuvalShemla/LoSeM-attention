# %% [markdown]
# # Co-Attention Key Clustering for Sparse Attention
#
# **Goal**: Test whether clustering keys by their co-attention patterns
# gives better clusters than KMeans on raw key vectors.
#
# **Two similarity matrices tested**:
# 1. **Attention co-occurrence**: W = A^T A, A[i,j] = softmax(q_i · k_j / √d)
# 2. **Logit co-occurrence**: W = L^T L, L[i,j] = q_i · k_j / √d (no softmax)
#
# **Key ablations**: causal vs bidirectional, with/without local window removal.
#
# **Method**: Randomized SVD of the (implicit) attention/logit matrix to get
# key embeddings, then KMeans. Equivalent to eigendecomposing W but without
# ever materializing the 80K×80K matrix (~25 GB). Runs in ~20s per config on CPU.

# %%
# Setup
# !pip install numpy scipy scikit-learn pyyaml torch psutil  # uncomment if needed

# Clone repo (uncomment on Colab):
# !git clone <YOUR_REPO_URL> LoSeM-attention
# %cd LoSeM-attention

# If data is on Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/LoSeM-attention/data/vectors data/

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import time
from pathlib import Path

from src.core import (
    full_attention, compute_special_indices,
    relative_l2_error, flat_kmeans, softmax as np_softmax,
)

# PQIndex: import with fallback in case __init__.py references untracked files
import types
try:
    from src.algorithms.pq_topk import PQIndex
except (ImportError, ModuleNotFoundError):
    import src
    pkg = types.ModuleType("src.algorithms")
    pkg.__path__ = [str(Path("src/algorithms").resolve())]
    pkg.__package__ = "src.algorithms"
    sys.modules["src.algorithms"] = pkg
    src.algorithms = pkg
    from src.algorithms.pq_topk import PQIndex
    print("Note: loaded PQIndex via fallback (some algorithm modules missing from git)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import psutil
ram_gb = psutil.virtual_memory().total / 1e9
print(f"System RAM: {ram_gb:.1f} GB")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
# Load Data — code_run p75 head (L15H21, kv_head=5, entropy=4.92)
# Pre-extracted single-head file avoids loading the full dataset
d = 128
sqrt_d = np.sqrt(d)

data_path = Path("data/notebook_data/code_run_p75_L15H21.npz")
npz = np.load(data_path)
Q_np, K_np, V_np = npz["Q"], npz["K"], npz["V"]
N = Q_np.shape[0]
print(f"Loaded from {data_path}: Q={Q_np.shape}, K={K_np.shape}, V={V_np.shape}")

N_TEST = 10
N_TRAIN = N - N_TEST

Q_train_np = Q_np[:N_TRAIN]
Q_test_np = Q_np[N_TRAIN:]

print(f"Sequence length: N = {N:,}")
print(f"Training queries: {N_TRAIN:,}")
print(f"Test queries: {N_TEST}")

# %% [markdown]
# ## Core: Randomized SVD for Co-Attention Embeddings
#
# Computes the top-k right singular vectors of the implicit M matrix
# (attention or logit) with:
# - **Power iterations** for accuracy on slowly-decaying spectra
# - **Frobenius norm estimation** to measure energy captured vs lost
# - Sink token excluded, optional local window removal

# %%
# Core: Randomized SVD for co-attention embeddings

def _apply_M(Q_train_t, K_all_t, X, mode, causal, local_window, batch_size,
             compute_dcol=False, compute_frob=False):
    """Compute Y = M_cand @ X without materializing M. Optionally D_col and ||M||_F^2."""
    N_q, N_k = Q_train_t.shape[0], K_all_t.shape[0]
    N_cand = N_k - 1
    s_d = K_all_t.shape[1] ** 0.5
    k = X.shape[1]

    Y = torch.zeros(N_q, k, device=K_all_t.device, dtype=torch.float32)
    D_col = torch.zeros(N_cand, device=K_all_t.device, dtype=torch.float32) if compute_dcol else None
    frob_sq = 0.0

    for start in range(0, N_q, batch_size):
        end = min(start + batch_size, N_q)
        q_batch = Q_train_t[start:end]

        if causal:
            K_used = K_all_t[:end]
            scores = q_batch @ K_used.T / s_d
            q_pos = torch.arange(start, end, device=scores.device).unsqueeze(1)
            k_pos = torch.arange(end, device=scores.device).unsqueeze(0)
            cmask = k_pos <= q_pos
            if mode == "attention":
                scores = scores.masked_fill(~cmask, float("-inf"))
                Mc = torch.softmax(scores, dim=-1)[:, 1:].float()
            else:
                Mc = (scores * cmask.float())[:, 1:].float()
            if local_window > 0:
                ckp = torch.arange(1, end, device=scores.device).unsqueeze(0)
                dist = q_pos - ckp
                lm = (dist >= 0) & (dist < local_window)
                Mc = Mc.masked_fill(lm, 0.0)
            nb = end - 1
            Y[start:end] = Mc @ X[:nb]
            if compute_dcol:
                D_col[:nb] += Mc.sum(dim=0)
            if compute_frob:
                frob_sq += (Mc ** 2).sum().item()
        else:
            scores = q_batch @ K_all_t.T / s_d
            if mode == "attention":
                Mc = torch.softmax(scores, dim=-1)[:, 1:].float()
            else:
                Mc = scores[:, 1:].float()
            if local_window > 0:
                q_pos = torch.arange(start, end, device=scores.device).unsqueeze(1)
                ckp = torch.arange(1, N_k, device=scores.device).unsqueeze(0)
                dist = (q_pos - ckp).abs()
                lm = dist < local_window
                Mc = Mc.masked_fill(lm, 0.0)
            Y[start:end] = Mc @ X
            if compute_dcol:
                D_col += Mc.sum(dim=0)
            if compute_frob:
                frob_sq += (Mc ** 2).sum().item()

    return Y, D_col, frob_sq


def _apply_Mt(Q_train_t, K_all_t, Y, mode, causal, local_window, batch_size):
    """Compute Z = M_cand^T @ Y without materializing M."""
    N_q, N_k = Q_train_t.shape[0], K_all_t.shape[0]
    N_cand = N_k - 1
    s_d = K_all_t.shape[1] ** 0.5
    k = Y.shape[1]

    Z = torch.zeros(N_cand, k, device=K_all_t.device, dtype=torch.float32)

    for start in range(0, N_q, batch_size):
        end = min(start + batch_size, N_q)
        q_batch = Q_train_t[start:end]

        if causal:
            K_used = K_all_t[:end]
            scores = q_batch @ K_used.T / s_d
            q_pos = torch.arange(start, end, device=scores.device).unsqueeze(1)
            k_pos = torch.arange(end, device=scores.device).unsqueeze(0)
            cmask = k_pos <= q_pos
            if mode == "attention":
                scores = scores.masked_fill(~cmask, float("-inf"))
                Mc = torch.softmax(scores, dim=-1)[:, 1:].float()
            else:
                Mc = (scores * cmask.float())[:, 1:].float()
            if local_window > 0:
                ckp = torch.arange(1, end, device=scores.device).unsqueeze(0)
                dist = q_pos - ckp
                lm = (dist >= 0) & (dist < local_window)
                Mc = Mc.masked_fill(lm, 0.0)
            nb = end - 1
            Z[:nb] += Mc.T @ Y[start:end]
        else:
            scores = q_batch @ K_all_t.T / s_d
            if mode == "attention":
                Mc = torch.softmax(scores, dim=-1)[:, 1:].float()
            else:
                Mc = scores[:, 1:].float()
            if local_window > 0:
                q_pos = torch.arange(start, end, device=scores.device).unsqueeze(1)
                ckp = torch.arange(1, N_k, device=scores.device).unsqueeze(0)
                dist = (q_pos - ckp).abs()
                lm = dist < local_window
                Mc = Mc.masked_fill(lm, 0.0)
            Z += Mc.T @ Y[start:end]

    return Z


def coattention_embeddings(
    Q_train_t, K_all_t, mode="attention", causal=True,
    local_window=0, rank=512, oversample=128, n_power_iter=2,
    batch_size=512, verbose=True,
):
    """
    Compute key embeddings from co-attention via randomized SVD
    with power iterations for accuracy.

    Power iterations: Y = (M M^T)^p M @ Omega. Each iteration
    adds 2 passes through the data. With p=2, total = 7 passes
    (~10s per config on A100 for 80K sequences).

    Also computes ||M||_F^2 to measure energy captured.

    Returns:
        V_emb: [N_cand, rank] right singular vectors (unweighted)
        S: [rank] singular values
        D_col: [N_cand] column sums
        frob_sq: float, ||M||_F^2 (total energy)
    """
    N_q = Q_train_t.shape[0]
    N_k = K_all_t.shape[0]
    N_cand = N_k - 1
    k = rank + oversample

    lw_str = f", lw={local_window}" if local_window > 0 else ""
    if verbose:
        print(f"  Randomized SVD ({mode}, {'causal' if causal else 'bidir'}{lw_str})")
        print(f"  N_q={N_q:,}, N_cand={N_cand:,}, rank={rank}, "
              f"oversample={oversample}, power_iter={n_power_iter}")

    t0 = time.time()
    args = (Q_train_t, K_all_t)
    kwargs = dict(mode=mode, causal=causal, local_window=local_window,
                  batch_size=batch_size)

    # Pass 1: Y = M @ Omega (also compute D_col and ||M||_F^2)
    omega = torch.randn(N_cand, k, device=K_all_t.device, dtype=torch.float32)
    Y, D_col, frob_sq = _apply_M(*args, omega, **kwargs,
                                  compute_dcol=True, compute_frob=True)
    del omega
    if verbose:
        print(f"  Pass 1 done ({time.time()-t0:.1f}s), ||M||_F^2 = {frob_sq:.2e}")

    # Power iterations: Y = M @ (M^T @ Y), with QR for stability
    for p in range(n_power_iter):
        tp = time.time()
        Y, _ = torch.linalg.qr(Y)
        Z = _apply_Mt(*args, Y, **kwargs)
        Z, _ = torch.linalg.qr(Z)
        Y, _, _ = _apply_M(*args, Z, **kwargs)
        del Z
        if verbose:
            print(f"  Power iter {p+1}/{n_power_iter} ({time.time()-tp:.1f}s)")

    # QR
    Q_basis, _ = torch.linalg.qr(Y)
    del Y

    # Final pass: B = Q_basis^T @ M
    # Reuse _apply_Mt with Q_basis transposed... actually B = Q^T M
    # which is M^T Q columns = _apply_Mt(Q_basis)
    B_t = _apply_Mt(*args, Q_basis, **kwargs)  # [N_cand, k]
    del Q_basis
    B = B_t.T  # [k, N_cand]
    del B_t

    if verbose:
        print(f"  Projection done ({time.time()-t0:.1f}s total)")

    # SVD of B [k x N_cand]
    U_B, S, Vt = torch.linalg.svd(B, full_matrices=False)
    V_emb = Vt[:rank].T  # [N_cand, rank] — unweighted

    # Energy analysis
    energy_captured = float((S[:rank] ** 2).sum().item())
    energy_svd_total = float((S ** 2).sum().item())
    if verbose:
        pct_of_svd = energy_captured / max(energy_svd_total, 1e-30) * 100
        pct_of_frob = energy_captured / max(frob_sq, 1e-30) * 100
        print(f"  Top-5 singular values: {S[:5].cpu().numpy()}")
        print(f"  Energy: rank-{rank} captures {pct_of_svd:.2f}% of SVD energy, "
              f"{pct_of_frob:.2f}% of total ||M||_F^2")
        if pct_of_frob < 95:
            print(f"  WARNING: only {pct_of_frob:.1f}% of total energy captured. "
                  f"Consider increasing rank.")
        print(f"  Total time: {time.time() - t0:.1f}s")

    return (V_emb.cpu().numpy(), S[:rank].cpu().numpy(),
            D_col.cpu().numpy(), frob_sq)


# %%
# Clustering & evaluation helpers
def kmeans_cluster(features, n_clusters, seed=42, n_iter=50):
    features = np.ascontiguousarray(features, dtype=np.float32)
    centroids, labels = flat_kmeans(features, n_clusters, seed=seed, n_iter=n_iter)
    return centroids, labels


def normalize_for_spectral(V_emb, D_col):
    """D^{-1/2} scaling + row L2 normalization (spectral-style)."""
    D_inv_sqrt = np.zeros_like(D_col)
    nz = D_col > 1e-12
    D_inv_sqrt[nz] = 1.0 / np.sqrt(D_col[nz])
    V_norm = V_emb * D_inv_sqrt[:, None]
    row_norms = np.maximum(np.linalg.norm(V_norm, axis=1, keepdims=True), 1e-12)
    V_norm = V_norm / row_norms
    n_zero = int(np.sum(~nz))
    if n_zero > 0:
        print(f"    Warning: {n_zero} keys with near-zero D_col (assigned zero embedding)")
    return V_norm


def spectral_cluster_from_eigvecs(V_emb, D_col, n_clusters, seed=42):
    V_norm = normalize_for_spectral(V_emb, D_col)
    return kmeans_cluster(V_norm, n_clusters, seed=seed)


def oracle_cluster(logits_cand, n_clusters):
    n_cand = len(logits_cand)
    sort_order = np.argsort(logits_cand)[::-1]
    labels = np.zeros(n_cand, dtype=np.int32)
    gs, rem, pos = n_cand // n_clusters, n_cand % n_clusters, 0
    for c in range(n_clusters):
        sz = gs + (1 if c < rem else 0)
        labels[sort_order[pos:pos + sz]] = c
        pos += sz
    return labels


def assign_new_keys(new_keys, centroids):
    dists = np.sum((new_keys[:, None] - centroids[None, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int32)


def within_cluster_variance(labels, logits_cand, n_clust):
    n = len(labels)
    sum_l = np.bincount(labels, weights=logits_cand, minlength=n_clust)
    sum_l2 = np.bincount(labels, weights=logits_cand ** 2, minlength=n_clust)
    cnts = np.bincount(labels, minlength=n_clust).astype(np.float64)
    active = cnts > 1
    mean_l = np.zeros(n_clust)
    var_l = np.zeros(n_clust)
    mean_l[active] = sum_l[active] / cnts[active]
    var_l[active] = np.maximum(sum_l2[active] / cnts[active] - mean_l[active] ** 2, 0.0)
    return float(np.sum(cnts * var_l) / n)


def cluster_diagnostics(labels, keys, values, logits_cand, n_clust):
    """Per-cluster quality: logit var, key/value distance, size balance."""
    n = len(labels)
    dim = keys.shape[1]
    cnts = np.bincount(labels, minlength=n_clust).astype(np.float64)
    active = cnts > 0

    # Logit variance
    sum_l = np.bincount(labels, weights=logits_cand, minlength=n_clust)
    sum_l2 = np.bincount(labels, weights=logits_cand ** 2, minlength=n_clust)
    mean_l = np.zeros(n_clust); var_l = np.zeros(n_clust)
    mean_l[active] = sum_l[active] / cnts[active]
    var_l[active] = np.maximum(sum_l2[active] / cnts[active] - mean_l[active] ** 2, 0.0)
    logit_var = float(np.sum(cnts * var_l) / n)

    # Key/Value distance to centroid (vectorized per dimension)
    def avg_dist_to_centroid(data, labels, cnts, n_clust, active):
        data_f = data.astype(np.float64)
        sums = np.zeros((n_clust, dim), dtype=np.float64)
        for j in range(dim):
            sums[:, j] = np.bincount(labels, weights=data_f[:, j], minlength=n_clust)
        # E[||x - mu||^2] = E[||x||^2] - ||mu||^2
        sq_norms = np.sum(data_f ** 2, axis=1)
        sum_sq = np.bincount(labels, weights=sq_norms, minlength=n_clust)
        centroid_sq = np.zeros(n_clust)
        centroid_sq[active] = np.sum((sums[active] / cnts[active, None]) ** 2, axis=1)
        mean_sq = np.zeros(n_clust)
        mean_sq[active] = sum_sq[active] / cnts[active]
        per_cluster = np.maximum(mean_sq - centroid_sq, 0.0)
        return float(np.sum(cnts * per_cluster) / n)

    key_dist = avg_dist_to_centroid(keys, labels, cnts, n_clust, active)
    val_dist = avg_dist_to_centroid(values, labels, cnts, n_clust, active)

    active_cnts = cnts[active]
    size_mean = float(np.mean(active_cnts))
    size_cv = float(np.std(active_cnts) / size_mean) if size_mean > 0 else 0.0

    return {
        "logit_var": logit_var,
        "key_dist": key_dist,
        "value_dist": val_dist,
        "size_cv": size_cv,
        "n_empty": int(np.sum(~active)),
    }


def evaluate_query(q, K, V, logits, special_idx, candidate_idx,
                   labels_cand, n_clust, topk_local, topk_global,
                   full_out, order=0):
    q64 = q.astype(np.float64)
    dim = V.shape[1]
    ck = K[candidate_idx].astype(np.float64)
    cv = V[candidate_idx].astype(np.float64)
    cl = logits[candidate_idx].astype(np.float64)

    k_s = np.zeros((n_clust, dim), np.float64)
    v_s = np.zeros((n_clust, dim), np.float64)
    for j in range(dim):
        k_s[:, j] = np.bincount(labels_cand, weights=ck[:, j], minlength=n_clust)
        v_s[:, j] = np.bincount(labels_cand, weights=cv[:, j], minlength=n_clust)
    cnts = np.bincount(labels_cand, minlength=n_clust).astype(np.float64)
    if order == 2:
        sl = np.bincount(labels_cand, weights=cl, minlength=n_clust)
        sl2 = np.bincount(labels_cand, weights=cl ** 2, minlength=n_clust)

    # Remove topK
    tl = labels_cand[topk_local]
    for j in range(dim):
        k_s[:, j] -= np.bincount(tl, weights=ck[topk_local, j], minlength=n_clust)
        v_s[:, j] -= np.bincount(tl, weights=cv[topk_local, j], minlength=n_clust)
    tc = np.bincount(tl, minlength=n_clust).astype(np.float64)
    cr = cnts - tc
    if order == 2:
        sl -= np.bincount(tl, weights=cl[topk_local], minlength=n_clust)
        sl2 -= np.bincount(tl, weights=cl[topk_local] ** 2, minlength=n_clust)

    act = np.where(cr > 0)[0]
    nsp = len(special_idx); ntk = len(topk_local); nact = len(act)
    ntot = nsp + ntk + nact
    scores = np.empty(ntot, np.float64)
    ovals = np.empty((ntot, dim), np.float32)

    scores[:nsp] = logits[special_idx].astype(np.float64)
    ovals[:nsp] = V[special_idx]
    o = nsp
    scores[o:o+ntk] = logits[topk_global].astype(np.float64)
    ovals[o:o+ntk] = V[topk_global]
    o = nsp + ntk
    for i, c in enumerate(act):
        nc = cr[c]; mk = k_s[c] / nc; ml = float(q64 @ mk) / sqrt_d
        if order == 2:
            mn = sl[c] / nc; vl = max(sl2[c] / nc - mn ** 2, 0.0)
            scores[o+i] = ml + vl / 2 + np.log(nc)
        else:
            scores[o+i] = ml + np.log(nc)
        ovals[o+i] = (v_s[c] / nc).astype(np.float32)

    w = np_softmax(scores).astype(np.float32)
    return relative_l2_error(w @ ovals, full_out)


# %% [markdown]
# ## Step 1: Compute Co-Attention Embeddings
#
# 8 configurations: {attention, logit} × {causal, bidir} × {no window, window=128}
#
# Each takes ~20s on CPU via randomized SVD.

# %%
# Compute co-attention embeddings (8 configs)
Q_train_t = torch.from_numpy(Q_train_np).to(device)
K_all_t = torch.from_numpy(K_np).to(device)

# Batch size: larger on GPU for faster throughput
if device.type == "cuda":
    gpu_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    BS = 2048 if gpu_gb >= 60 else 1024 if gpu_gb >= 30 else 512
else:
    BS = 512
LOCAL_WINDOW = 128
print(f"Batch size: {BS}")

configs = [
    ("attention", True,  0,            "attn_causal"),
    ("attention", True,  LOCAL_WINDOW,  "attn_causal_nolocal"),
    ("attention", False, 0,            "attn_bidir"),
    ("attention", False, LOCAL_WINDOW,  "attn_bidir_nolocal"),
    ("logit",     True,  0,            "logit_causal"),
    ("logit",     True,  LOCAL_WINDOW,  "logit_causal_nolocal"),
    ("logit",     False, 0,            "logit_bidir"),
    ("logit",     False, LOCAL_WINDOW,  "logit_bidir_nolocal"),
]

N_CLUSTERS = 512
RANK = 2048  # high rank — energy check will tell us if it's enough
N_POWER_ITER = 2  # power iterations for accuracy
N_CAND = N - 1
N_CAND_TRAIN = N_TRAIN - 1
K_cand_train = K_np[1:N_TRAIN]
K_cand_test = K_np[N_TRAIN:]

print("=" * 60)
print(f"Computing embeddings for {len(configs)} configurations")
print("=" * 60)

# Store embeddings and D for diagnostics
all_embeddings = {}

for mode, causal, lw, label in configs:
    print(f"\n--- {label} ---")
    V_emb, S, D, frob_sq = coattention_embeddings(
        Q_train_t, K_all_t, mode=mode, causal=causal,
        local_window=lw, rank=RANK, oversample=128,
        n_power_iter=N_POWER_ITER, batch_size=BS,
    )
    all_embeddings[label] = (V_emb, S, D, frob_sq)

del Q_train_t, K_all_t
if device.type == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## Step 2: Cluster

# %%
# Clustering: KMeans baseline
print("\n" + "=" * 60)
print(f"Clustering {N_CAND:,} candidate keys into {N_CLUSTERS} clusters")
print("=" * 60)

# Baseline: KMeans on raw keys
print("\n[Baseline] KMeans on raw keys...")
t0 = time.time()
centroids_key, labels_key_train = kmeans_cluster(K_cand_train, N_CLUSTERS, seed=SEED)
labels_key_test = assign_new_keys(K_cand_test, centroids_key)
labels_key_all = np.concatenate([[0], labels_key_train, labels_key_test])
print(f"  Done in {time.time()-t0:.1f}s")

clusterings = {"KMeans-Keys": labels_key_all}

# %% [markdown]
# ### M_Q Methods: Query-Covariance Weighted KMeans
#
# M_Q = Q^T Q captures the query distribution. Transforming keys by
# M_Q^{1/2} and running KMeans is **mathematically equivalent** to
# KMeans on the full logit profile (q_1^T k, ..., q_N^T k) but in
# only d=128 dimensions. Directly minimizes within-cluster logit variance.
#
# **Exact** for bidirectional logits: z_i = M_Q^{1/2} k_i, M_Q = Q^T Q.
#
# **Proxy** for causal: the true causal metric is position-dependent
# M_j = Σ_{i≥j} q_i q_i^T (each key has a different metric), which
# breaks standard KMeans. We approximate by varying the query subset
# (LastHalf, L5000, L1000, L100) — these are proxies, not exact causal.
#
# The co-attention SVD is where causal/nolocal masking is exact per-key.

# %%
# M_Q clustering
print("\n--- M_Q: Query-Covariance KMeans ---")

def mq_transform_and_cluster(Q_subset, K_cand, n_clusters, label, seed=SEED):
    """Transform keys by M_Q^{1/2} and run KMeans."""
    t0 = time.time()
    # M_Q = Q^T Q  [d x d]
    Q_f = Q_subset.astype(np.float64)
    M_Q = Q_f.T @ Q_f  # [128, 128]

    # Eigendecompose: M_Q = V Λ V^T
    eigenvalues, eigenvectors = np.linalg.eigh(M_Q)
    # M_Q^{1/2} = V Λ^{1/2} V^T
    # z_i = Λ^{1/2} V^T k_i  (rotation invariant, same KMeans result)
    sqrt_eig = np.sqrt(np.maximum(eigenvalues, 0.0))

    # Transform: z = K @ V * sqrt(eigenvalues)
    z = K_cand.astype(np.float64) @ eigenvectors * sqrt_eig[None, :]
    z = z.astype(np.float32)

    centroids, labels = kmeans_cluster(z, n_clusters, seed=seed)

    # Effective rank: how many eigenvalue directions matter?
    cumvar = np.cumsum(eigenvalues[::-1]) / max(np.sum(eigenvalues), 1e-30)
    eff_rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    eff_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    print(f"  [{label}] M_Q effective rank: {eff_rank_90} (90%), "
          f"{eff_rank_99} (99%) — done in {time.time()-t0:.1f}s")

    return centroids, labels, z


def mq_full_pipeline(Q_subset, K_train, K_test, n_clusters, label, seed=SEED):
    """Compute M_Q, transform keys, cluster, assign test keys."""
    Q_f = Q_subset.astype(np.float64)
    M_Q = Q_f.T @ Q_f
    eig_vals, eig_vecs = np.linalg.eigh(M_Q)
    sqrt_eig = np.sqrt(np.maximum(eig_vals, 0.0))

    def transform(K):
        return (K.astype(np.float64) @ eig_vecs * sqrt_eig[None, :]).astype(np.float32)

    z_train = transform(K_train)
    z_test = transform(K_test)
    centroids, lab_train = kmeans_cluster(z_train, n_clusters, seed=seed)
    lab_test = assign_new_keys(z_test, centroids)

    # Effective rank
    cumvar = np.cumsum(eig_vals[::-1]) / max(np.sum(eig_vals), 1e-30)
    r90 = int(np.searchsorted(cumvar, 0.90)) + 1
    r99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"  [{label}] eff rank: {r90} (90%), {r99} (99%)")

    return np.concatenate([[0], lab_train, lab_test])


# Note on causal/nolocal for M_Q:
# Position-dependent M_Q(j) = Σ_{i≥j} q_i q_i^T would give each key
# a different transform, breaking standard KMeans. Instead, we proxy
# by varying the query subset — late queries approximate the causal
# distribution for most keys. The local window correction is negligible
# (128 out of 80K queries = 0.16%).
# The co-attention SVD is where causal/nolocal truly matters — it
# computes actual per-key masked attention patterns.

for mq_label, Q_sub in [
    ("MQ-All",      Q_train_np),
    ("MQ-LastHalf", Q_train_np[N_TRAIN // 2:]),       # causal-like proxy
    ("MQ-L5000",    Q_train_np[-5000:]),
    ("MQ-L1000",    Q_train_np[-1000:]),
    ("MQ-L100",     Q_train_np[-100:]),
]:
    t0 = time.time()
    clusterings[mq_label] = mq_full_pipeline(
        Q_sub, K_cand_train, K_cand_test, N_CLUSTERS, mq_label)
    print(f"    Done in {time.time()-t0:.1f}s")

# %%
# Co-attention SVD clusterings
print("\n--- Co-Attention SVD Embeddings ---")
for label, (V_emb, S, D, frob_sq) in all_embeddings.items():
    V_train = V_emb[:N_CAND_TRAIN]
    V_test = V_emb[N_CAND_TRAIN:]
    D_train = D[:N_CAND_TRAIN]

    # Weight eigenvectors by singular values (correct objective).
    # z_j = Σ * V_j gives the rank-r representation of column j.
    # KMeans on z_j minimizes within-cluster profile distance.
    V_train_w = (V_train * S[None, :]).astype(np.float32)
    V_test_w = (V_test * S[None, :]).astype(np.float32)

    # Weighted PCA + KMeans (correct objective)
    name = f"SVD-{label}"
    c_emb, lab_tr = kmeans_cluster(V_train_w, N_CLUSTERS, seed=SEED)
    if V_test_w.shape[0] > 0 and np.any(np.abs(V_test_w) > 1e-12):
        lab_te = assign_new_keys(V_test_w, c_emb)
    else:
        lab_te = assign_new_keys(K_cand_test, centroids_key)
    clusterings[name] = np.concatenate([[0], lab_tr, lab_te])

    # Unweighted (for comparison — shows the effect of weighting)
    name_uw = f"SVD-{label}-unwtd"
    c_uw, lab_uw = kmeans_cluster(V_train.astype(np.float32), N_CLUSTERS, seed=SEED)
    if V_test.shape[0] > 0 and np.any(np.abs(V_test) > 1e-12):
        lab_te_uw = assign_new_keys(V_test.astype(np.float32), c_uw)
    else:
        lab_te_uw = assign_new_keys(K_cand_test, centroids_key)
    clusterings[name_uw] = np.concatenate([[0], lab_uw, lab_te_uw])

    # Spectral-style normalized SVD: D^{-1/2} + row-normalize + KMeans
    # (Not true Laplacian spectral clustering — uses SVD embeddings
    # with D-normalization as a heuristic proxy.)
    name_s = f"NormSVD-{label}"
    V_train_spec = normalize_for_spectral(V_train, D_train)
    c_spec, lab_s = kmeans_cluster(V_train_spec.astype(np.float32), N_CLUSTERS, seed=SEED)
    # Assign test keys in the SAME spectral-normalized space
    D_test = D[N_CAND_TRAIN:]
    V_test_spec = normalize_for_spectral(V_test, D_test)
    if V_test_spec.shape[0] > 0 and np.any(np.abs(V_test_spec) > 1e-12):
        lab_te_spec = assign_new_keys(V_test_spec.astype(np.float32), c_spec)
    else:
        lab_te_spec = assign_new_keys(K_cand_test, centroids_key)
    clusterings[name_s] = np.concatenate([[0], lab_s, lab_te_spec])

print(f"\n{len(clusterings)} clustering methods ready.")

# %% [markdown]
# ## Step 3: Diagnostics — Variance, Key Distance, Value Distance
#
# For each test query, measures:
# - **Logit variance**: within-cluster variance of q·k/√d (lower = better)
# - **Key dist²**: avg squared L2 distance to cluster centroid in key space
# - **Value dist²**: same for values (governs mean-value approximation quality)
# - **Size CV**: cluster size coefficient of variation (0 = perfectly balanced)

# %%
# Diagnostics: variance, key/value distance, size balance
print("\n" + "=" * 60)
print("Cluster Quality Diagnostics (avg over test queries)")
print("=" * 60)

variance_results = {}
diag_results = {}

for t_idx in range(N_TEST):
    q_pos = N_TRAIN + t_idx
    q = Q_test_np[t_idx]
    logits_all = (q.astype(np.float64) @ K_np[:q_pos + 1].T) / sqrt_d
    cand = np.arange(1, q_pos + 1, dtype=np.int64)
    logits_cand = logits_all[cand].astype(np.float64)
    cand_keys = K_np[cand]
    cand_vals = V_np[cand]

    # Oracle
    lab_o = oracle_cluster(logits_cand, N_CLUSTERS)
    variance_results.setdefault("Oracle", []).append(
        within_cluster_variance(lab_o, logits_cand, N_CLUSTERS))
    diag_results.setdefault("Oracle", []).append(
        cluster_diagnostics(lab_o, cand_keys, cand_vals, logits_cand, N_CLUSTERS))

    # All methods
    for name, labels_all in clusterings.items():
        lab_c = labels_all[cand]
        n_eff = lab_c.max() + 1
        variance_results.setdefault(name, []).append(
            within_cluster_variance(lab_c, logits_cand, n_eff))
        diag_results.setdefault(name, []).append(
            cluster_diagnostics(lab_c, cand_keys, cand_vals, logits_cand, n_eff))

# Print diagnostics table
oracle_var = np.mean(variance_results["Oracle"])
all_names = ["Oracle", "KMeans-Keys"] + [n for n in clusterings if n != "KMeans-Keys"]

print(f"\n{'Method':45s}  {'Logit Var':>10s}  {'vs Oracle':>9s}  "
      f"{'Key Dist²':>10s}  {'Val Dist²':>10s}  {'Size CV':>8s}")
print("-" * 100)
for name in all_names:
    if name not in diag_results:
        continue
    dl = diag_results[name]
    v = np.mean(variance_results[name])
    kd = np.mean([x["key_dist"] for x in dl])
    vd = np.mean([x["value_dist"] for x in dl])
    cv = np.mean([x["size_cv"] for x in dl])
    vr = v / oracle_var if oracle_var > 0 else 0
    print(f"{name:45s}  {v:10.4f}  {vr:8.1f}x  "
          f"{kd:10.2f}  {vd:10.2f}  {cv:8.2f}")

# %% [markdown]
# ## Step 4: Attention Error Evaluation
#
# 512 PQ approximate topK + 512 cluster reps.
# PQ topK is shared across all methods for fair comparison.

# %%
# Evaluate: PQ topK + cluster reps → attention error
print("\n" + "=" * 60)
print("Building PQ index")
print("=" * 60)

pq = PQIndex(m=8, n_codes=256, seed=SEED)
pq.fit(K_np)

B_TOPK = 512
B_CLUSTERS = 512

print(f"\nEvaluating: {B_TOPK} PQ topK + {B_CLUSTERS} clusters, {N_TEST} queries")

error_results_1st = {}
error_results_2nd = {}
topk_mass_pct = []  # attention mass captured by PQ topK alone
residual_mass_pct = []  # remaining mass that clusters must approximate

for t_idx in range(N_TEST):
    q_pos = N_TRAIN + t_idx
    q = Q_test_np[t_idx]
    full_out, logits_all, _ = full_attention(q, K_np[:q_pos+1], V_np[:q_pos+1], d)
    sp = np.array([0], dtype=np.int64)
    cand = np.arange(1, q_pos + 1, dtype=np.int64)
    n_cand = len(cand)

    # Full attention weights for mass analysis
    all_weights = np_softmax(logits_all).astype(np.float64)

    # Shared PQ topK
    cand_mask = np.zeros(N, dtype=bool)
    cand_mask[cand] = True
    tg = pq.approximate_topk(q, B_TOPK, candidate_mask=cand_mask)
    tg = tg[tg <= q_pos]
    g2l = np.full(q_pos + 1, -1, dtype=np.int64)
    g2l[cand] = np.arange(n_cand)
    tl = g2l[tg]; v = tl >= 0; tl = tl[v]; tg = tg[v]

    # Mass captured by topK + special
    fixed_mass = float(all_weights[sp].sum() + all_weights[tg].sum())
    topk_mass_pct.append(fixed_mass * 100)
    residual_mass_pct.append((1.0 - fixed_mass) * 100)

    # Oracle
    lab_o = oracle_cluster(logits_all[cand], B_CLUSTERS)
    for order, res in [(0, error_results_1st), (2, error_results_2nd)]:
        res.setdefault("Oracle", []).append(
            evaluate_query(q, K_np[:q_pos+1], V_np[:q_pos+1], logits_all,
                           sp, cand, lab_o, B_CLUSTERS, tl, tg, full_out, order))

    # All methods
    for name, labels_all in clusterings.items():
        lab_c = labels_all[cand]
        n_eff = lab_c.max() + 1
        for order, res in [(0, error_results_1st), (2, error_results_2nd)]:
            res.setdefault(name, []).append(
                evaluate_query(q, K_np[:q_pos+1], V_np[:q_pos+1], logits_all,
                               sp, cand, lab_c, n_eff, tl, tg, full_out, order))

    print(f"  Query {t_idx}: topK+special captures {fixed_mass*100:.1f}% "
          f"of attention mass, residual {(1-fixed_mass)*100:.1f}%")

print(f"\n  Avg topK+special mass: {np.mean(topk_mass_pct):.1f}%")
print(f"  Avg residual mass (clusters must approximate): {np.mean(residual_mass_pct):.1f}%")

# %%
# Results — sorted ranking
print("\n" + "=" * 100)
print("FULL RESULTS — sorted by 1st-order error (best → worst)")
print("=" * 100)

# Build combined table for all methods
all_methods_eval = ["Oracle"] + list(clusterings.keys())
rows = []
oracle_var_mean = np.mean(variance_results.get("Oracle", [1.0]))
km_err_1st = np.mean(error_results_1st.get("KMeans-Keys", [1.0]))
km_err_2nd = np.mean(error_results_2nd.get("KMeans-Keys", [1.0]))
km_var = np.mean(variance_results.get("KMeans-Keys", [1.0]))

for name in all_methods_eval:
    if name not in error_results_1st:
        continue
    e1 = np.mean(error_results_1st[name])
    e2 = np.mean(error_results_2nd[name])
    v = np.mean(variance_results.get(name, [0]))
    kd = np.mean([x["key_dist"] for x in diag_results.get(name, [{"key_dist": 0}])])
    vd = np.mean([x["value_dist"] for x in diag_results.get(name, [{"value_dist": 0}])])
    cv = np.mean([x["size_cv"] for x in diag_results.get(name, [{"size_cv": 0}])])
    rows.append((name, e1, e2, v, kd, vd, cv))

# Sort by 1st-order error
rows.sort(key=lambda r: r[1])

print(f"\n{'#':>3s}  {'Method':45s}  {'Err 1st':>9s}  {'Err 2nd':>9s}  "
      f"{'vs KM 1st':>9s}  {'Logit Var':>10s}  {'Var/Orc':>8s}  "
      f"{'Key D²':>8s}  {'Val D²':>8s}  {'SzCV':>5s}")
print("-" * 160)
for rank_i, (name, e1, e2, v, kd, vd, cv) in enumerate(rows, 1):
    vs_km = e1 / km_err_1st if km_err_1st > 0 else 0
    vr = v / oracle_var_mean if oracle_var_mean > 0 else 0
    tag = ""
    if name == "Oracle":
        tag = " *** ORACLE"
    elif name == "KMeans-Keys":
        tag = " *** BASELINE"
    elif vs_km < 1.0:
        tag = " << BETTER than KMeans"
    print(f"{rank_i:3d}  {name:45s}  {e1:9.6f}  {e2:9.6f}  "
          f"{vs_km:9.2f}x  {v:10.4f}  {vr:7.1f}x  "
          f"{kd:8.2f}  {vd:8.2f}  {cv:5.2f}{tag}")

# Summary: group by method family
print("\n" + "=" * 80)
print("SUMMARY BY METHOD FAMILY (best variant per family)")
print("=" * 80)

families = {}
for name, e1, e2, v, kd, vd, cv in rows:
    if name == "Oracle":
        fam = "Oracle"
    elif name == "KMeans-Keys":
        fam = "KMeans-Keys"
    elif name.startswith("MQ-"):
        fam = "M_Q (query covariance)"
    elif "unweighted" in name:
        fam = "SVD unweighted"
    elif name.startswith("SVD-attn"):
        fam = "SVD attention co-occ"
    elif name.startswith("SVD-logit"):
        fam = "SVD logit co-occ"
    elif name.startswith("NormSVD-"):
        fam = "NormSVD (D-normalized)"
    else:
        fam = name
    if fam not in families or e1 < families[fam][1]:
        families[fam] = (name, e1, e2, v)

print(f"\n{'Family':30s}  {'Best variant':40s}  {'Err 1st':>9s}  {'vs KM':>7s}  {'Logit Var':>10s}")
print("-" * 105)
for fam in ["Oracle", "M_Q (query covariance)", "SVD logit co-occ",
            "SVD attention co-occ", "NormSVD (D-normalized)", "SVD unweighted", "KMeans-Keys"]:
    if fam not in families:
        continue
    name, e1, e2, v = families[fam]
    vs_km = e1 / km_err_1st if km_err_1st > 0 else 0
    print(f"{fam:30s}  {name:40s}  {e1:9.6f}  {vs_km:6.2f}x  {v:10.4f}")

# %%
# Plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Main methods only (skip reduced-rank for clarity)
    main = ["Oracle", "KMeans-Keys"] + [
        n for n in clusterings if n != "KMeans-Keys"
        and "-R32" not in n and "-R64" not in n
        and "-unwtd" not in n  # skip unweighted for clarity
    ]

    def mcol(name):
        if name == "Oracle": return "gold"
        if "KMeans" in name: return "tab:orange"
        if name.startswith("MQ-"): return "tab:red"
        if "NormSVD" in name: return "tab:cyan"
        if "nolocal" in name: return "darkgreen" if "logit" in name else "darkblue"
        if "logit" in name: return "tab:green"
        if "attn" in name: return "tab:blue"
        return "tab:gray"

    cols = [mcol(m) for m in main]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    data = [
        (axes[0,0], [np.mean(variance_results.get(m,[0])) for m in main],
         "Avg Within-Cluster Logit Variance", "Logit Variance (lower = better)"),
        (axes[0,1], [np.mean(error_results_1st.get(m,[0])) for m in main],
         "Mean Relative L2 Error", "Attention Error 1st order"),
        (axes[1,0], [np.mean([x["key_dist"] for x in diag_results.get(m,[{"key_dist":0}])]) for m in main],
         "Avg Key L2 Dist² to Centroid", "Key Space Tightness"),
        (axes[1,1], [np.mean([x["value_dist"] for x in diag_results.get(m,[{"value_dist":0}])]) for m in main],
         "Avg Value L2 Dist² to Centroid", "Value Space Tightness"),
    ]
    for ax, vals, xlabel, title in data:
        ax.barh(range(len(main)), vals, color=cols)
        ax.set_yticks(range(len(main)))
        ax.set_yticklabels(main, fontsize=7)
        ax.set_xlabel(xlabel); ax.set_title(title)
        ax.invert_yaxis()

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/coattention_clustering.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved results/coattention_clustering.png")

    # Scatter: variance vs error
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, m in enumerate(main):
        if m == "Oracle": continue
        v = np.mean(variance_results.get(m, [0]))
        e = np.mean(error_results_1st.get(m, [0]))
        ax.scatter(v, e, c=cols[i], s=80, zorder=5)
        ax.annotate(m, (v, e), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.scatter(np.mean(variance_results["Oracle"]), np.mean(error_results_1st["Oracle"]),
               c="gold", s=200, marker="*", zorder=6, label="Oracle")
    ax.set_xlabel("Logit Variance"); ax.set_ylabel("Attention Error (1st)")
    ax.set_title("Does lower logit variance → lower error?")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/coattention_var_vs_error.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved results/coattention_var_vs_error.png")
except ImportError:
    print("matplotlib not available")
