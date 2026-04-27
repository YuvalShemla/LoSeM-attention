"""
Final comparison: all methods with TRUE normalizer.
Runs twice: once with original keys/queries, once with normalized.
All on code_run, all 5 heads.
"""
import sys; sys.path.insert(0, ".")
import json
import numpy as np
from pathlib import Path
from src.core import full_attention, compute_special_indices, softmax, relative_l2_error, flat_kmeans
from src.evaluation.data_loader import load_examples

d = 128; sqrt_d = np.sqrt(d); seed = 42
budgets = [32, 64, 128, 256, 512, 1024, 2048, 4096]

with open('data/vectors/code_run/metadata.json') as f:
    meta = json.load(f)
heads = [(h['layer'], h['q_head'], h['kv_head'],
          h['selection_label'], h['effective_entropy'])
         for h in meta['selected_heads']]


def run_all_methods(Q_use, K_use, V, label_prefix, heads_list):
    for layer, qh, kvh, label, ent_meta in heads_list:
        ex = list(load_examples(
            Path('data/vectors'), 'code_run',
            layer=layer, head=qh, kv_head=kvh,
            phase=None, max_examples=1, use_rope=True,
        ))[0]
        Q_orig, K_orig = ex['Q'], ex['K']
        V_head = ex['V']

        if label_prefix == 'NORMALIZED':
            k_norms = np.linalg.norm(K_orig, axis=1, keepdims=True)
            q_norms = np.linalg.norm(Q_orig, axis=1, keepdims=True)
            K_h = (K_orig / np.maximum(k_norms, 1e-10)).astype(np.float32)
            Q_h = (Q_orig / np.maximum(q_norms, 1e-10)).astype(np.float32)
        else:
            K_h = K_orig
            Q_h = Q_orig

        q = Q_h[-1]; q64 = q.astype(np.float64)
        full_out, logits, _ = full_attention(q, K_h, V_head, d)
        sp, cand = compute_special_indices(len(K_h), 1, 0)
        ck = K_h[cand].astype(np.float64); cv = V_head[cand].astype(np.float64)
        n_cand = len(cand); cand_logits = logits[cand]
        l_max = logits.astype(np.float64).max()
        D_shifted = np.exp(logits.astype(np.float64) - l_max).sum()
        rng = np.random.default_rng(seed)
        ent = -np.sum(softmax(logits)[softmax(logits)>0]*np.log(softmax(logits)[softmax(logits)>0]))

        # QClust prototypes
        Q_100 = Q_h[-101:-1].astype(np.float64)
        qc, _ = flat_kmeans(Q_100.astype(np.float32), 8, seed=seed, n_iter=30)
        qc = qc.astype(np.float64)

        short = label.replace('_lowest', '').replace('_highest', '').replace('_median', '')
        print(f"\n  {label_prefix} — code_run {short} (ent={ent:.2f})", flush=True)
        header = f"  {'Method':35s}" + "".join(f"  B={b:>5d}" for b in budgets)
        print(header, flush=True)
        print("  " + "-" * (37 + 9 * len(budgets)), flush=True)

        results = {}
        for b in budgets:
            b_topk = b // 2; b_cluster = b - b_topk
            bt = min(b_topk, n_cand)
            topk_local = np.argpartition(cand_logits, -bt)[-bt:]
            topk_global = cand[topk_local]

            # IdealTopK (full budget, subset attn)
            sel = np.concatenate([sp, cand[np.argpartition(cand_logits, -min(b,n_cand))[-min(b,n_cand):]]]).astype(np.int64)
            results.setdefault('IdealTopK(subsetAttn,B=topK)', []).append(
                relative_l2_error((softmax(logits[sel]) @ V_head[sel].astype(np.float64)).astype(np.float32), full_out))

            # EWS
            cand_w = softmax(cand_logits.astype(np.float64))
            so = np.argsort(cand_w)[::-1]
            sorted_cand = cand[so]; sorted_w = cand_w[so]
            ng = min(b, n_cand)
            cumsum = np.cumsum(sorted_w); total = cumsum[-1]
            targets = np.linspace(0, total, ng+1)[1:-1]
            si = np.clip(np.searchsorted(cumsum, targets), 1, n_cand-1)
            bounds = list(dict.fromkeys(si.tolist()))
            segs = []; prev = 0
            for s in bounds:
                if s > prev: segs.append((prev, s)); prev = s
            if prev < n_cand: segs.append((prev, n_cand))
            while len(segs) < ng:
                best = max(range(len(segs)), key=lambda i: segs[i][1]-segs[i][0])
                s, e = segs[best]
                if e-s < 2: break
                mid = (s+e)//2; segs[best:best+1] = [(s, mid), (mid, e)]
            groups = [sorted_cand[s:e] for s, e in segs]
            n_sp = len(sp); n_g = len(groups); n_t = n_sp+n_g
            sc = np.empty(n_t); ov = np.empty((n_t, d))
            sc[:n_sp] = logits[sp]; ov[:n_sp] = V_head[sp]
            for i, g in enumerate(groups):
                mk = K_h[g].astype(np.float64).mean(axis=0)
                sc[n_sp+i] = float(q64 @ mk)/sqrt_d + np.log(len(g))
                ov[n_sp+i] = V_head[g].astype(np.float64).mean(axis=0)
            results.setdefault('EWS(oracleGroups,B=groups)', []).append(
                relative_l2_error((softmax(sc) @ ov).astype(np.float32), full_out))

            # Helper for cluster methods with true norm
            def do_cluster(labels, nc):
                ks = np.zeros((nc, d), dtype=np.float64)
                vs = np.zeros((nc, d), dtype=np.float64)
                for j in range(d):
                    ks[:, j] = np.bincount(labels, weights=ck[:, j], minlength=nc)
                    vs[:, j] = np.bincount(labels, weights=cv[:, j], minlength=nc)
                cnts = np.bincount(labels, minlength=nc).astype(np.float64)
                sl = labels[topk_local]
                for j in range(d):
                    ks[:, j] -= np.bincount(sl, weights=ck[topk_local, j], minlength=nc)
                    vs[:, j] -= np.bincount(sl, weights=cv[topk_local, j], minlength=nc)
                cr = cnts - np.bincount(sl, minlength=nc).astype(np.float64)
                active = np.where(cr > 0)[0]
                N = np.zeros(d, dtype=np.float64)
                for i in range(len(sp)):
                    N += np.exp(logits[sp[i]].astype(np.float64) - l_max) * V_head[sp[i]].astype(np.float64)
                for i in range(len(topk_local)):
                    N += np.exp(logits[topk_global[i]].astype(np.float64) - l_max) * V_head[topk_global[i]].astype(np.float64)
                for c in active:
                    nc_ = cr[c]; mk = ks[c]/nc_; mv = vs[c]/nc_
                    N += nc_ * np.exp(float(q64 @ mk)/sqrt_d - l_max) * mv
                return relative_l2_error((N/D_shifted).astype(np.float32), full_out)

            # vAttention TRUE norm
            remaining = np.ones(n_cand, dtype=bool); remaining[topk_local] = False
            rem_pos = np.where(remaining)[0]; n_s = len(rem_pos)
            n_sample = min(b_cluster, n_s)
            sampled_pos = rng.choice(rem_pos, size=n_sample, replace=False) if n_sample > 0 else np.array([], dtype=int)
            sampled_global = cand[sampled_pos]
            N_va = np.zeros(d, dtype=np.float64)
            for i in range(len(sp)):
                N_va += np.exp(logits[sp[i]].astype(np.float64) - l_max) * V_head[sp[i]].astype(np.float64)
            for i in range(len(topk_local)):
                N_va += np.exp(logits[topk_global[i]].astype(np.float64) - l_max) * V_head[topk_global[i]].astype(np.float64)
            if n_sample > 0:
                w_is = float(n_s) / float(n_sample)
                for i in range(n_sample):
                    N_va += w_is * np.exp(logits[sampled_global[i]].astype(np.float64) - l_max) * V_head[sampled_global[i]].astype(np.float64)
            results.setdefault('vAttn(topK+unifSample,trueD)', []).append(
                relative_l2_error((N_va/D_shifted).astype(np.float32), full_out))

            # OracleClust
            sort_order = np.argsort(cand_logits)[::-1]
            olab = np.zeros(n_cand, dtype=np.int32)
            gs = n_cand // b_cluster; rem = n_cand % b_cluster; pos = 0
            for c in range(b_cluster):
                sz = gs + (1 if c < rem else 0)
                olab[sort_order[pos:pos+sz]] = c; pos += sz
            results.setdefault('OracleClust(logitRank,trueD)', []).append(do_cluster(olab, b_cluster))

            # KeyClust
            _, klab = flat_kmeans(K_h[cand], b_cluster, seed=seed, n_iter=50)
            results.setdefault('KeyClust(KMkeys,trueD)', []).append(do_cluster(klab, b_cluster))

            # ValClust
            _, vlab = flat_kmeans(V_head[cand], b_cluster, seed=seed, n_iter=50)
            results.setdefault('ValClust(KMvalues,trueD)', []).append(do_cluster(vlab, b_cluster))

            # QClust8
            proto = (ck @ qc.T) / sqrt_d
            _, qlab = flat_kmeans(proto.astype(np.float32), b_cluster, seed=seed, n_iter=50)
            results.setdefault('QClust8(logitProf100,trueD)', []).append(do_cluster(qlab, b_cluster))

            # ValClust→KeySub (alternating power-of-2 growth)
            val_key_schedule = [
                (4,8), (4,16), (8,16), (8,32), (16,32), (16,64), (32,64), (32,128),
            ]
            best_vk = min(val_key_schedule, key=lambda vk: abs(vk[0]*vk[1] - b_cluster))
            n_vc, n_ksub = best_vk
            _, vlabels = flat_kmeans(V_head[cand], n_vc, seed=seed, n_iter=50)
            flat_vk = np.full(n_cand, -1, dtype=np.int32)
            sub_id = 0
            for vc in range(n_vc):
                members = np.where(vlabels == vc)[0]
                nk = min(n_ksub, len(members))
                if nk < 2:
                    flat_vk[members] = sub_id; sub_id += 1; continue
                n_unique = len(np.unique(V_head[cand[members]], axis=0))
                if n_unique < 2:
                    flat_vk[members] = sub_id; sub_id += 1; continue
                _, ksub = flat_kmeans(K_h[cand[members]], nk, seed=seed+vc, n_iter=30)
                for k in range(nk):
                    mask = ksub == k
                    if mask.any(): flat_vk[members[mask]] = sub_id; sub_id += 1
            results.setdefault('ValKeyClust(V->K,trueD)', []).append(do_cluster(flat_vk, sub_id))

        for name in ['EWS(oracleGroups,B=groups)', 'OracleClust(logitRank,trueD)',
                      'QClust8(logitProf100,trueD)', 'KeyClust(KMkeys,trueD)',
                      'ValKeyClust(V->K,trueD)', 'ValClust(KMvalues,trueD)',
                      'vAttn(topK+unifSample,trueD)', 'IdealTopK(subsetAttn,B=topK)']:
            row = f"  {name:35s}"
            for err in results[name]:
                row += f"  {err:7.5f}"
            print(row, flush=True)


# Run 1: Original keys/queries
print(f"\n{'#'*90}")
print(f"  RUN 1: ORIGINAL keys and queries")
print(f"{'#'*90}", flush=True)
run_all_methods(None, None, None, 'ORIGINAL', heads)

# Run 2: Normalized keys/queries
print(f"\n{'#'*90}")
print(f"  RUN 2: NORMALIZED keys and queries (unit norm)")
print(f"{'#'*90}", flush=True)
run_all_methods(None, None, None, 'NORMALIZED', heads)

print("\nDone.", flush=True)
