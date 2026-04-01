# Attention Approximation Methods

## Problem

Given a query q, keys K = [k_1, ..., k_N], values
V = [v_1, ..., v_N], the exact attention output is:

    o* = sum_i softmax(q^T k_i / sqrt(d)) v_i

We approximate o* using only B << N key interactions.

**Error metric:**

    ||o_hat - o*||_2 / ||o*||_2


## Idealized Methods

These represent the best achievable accuracy at a given
budget. They use oracle knowledge (true logits/weights)
and spend per-query computation on grouping. Any new
algorithm should be compared against these idealized
methods to gauge how close it gets to the theoretical
best.

All idealized methods are automatically included in
every evaluation run.


### IdealTopK

Select the B keys with largest logits q^T k_i / sqrt(d).
Renormalize softmax over the selected subset.


### IdealSampling

Sample B keys proportional to the true attention weights
(with replacement). Simple average of sampled values
(unbiased Monte Carlo estimator).


### IdealEqualSplits

1. For each query, sort all non-special keys by logit (descending)
2. Split into B equal-sized groups
3. Represent each group by (avg_key, avg_value)
4. Score: q^T avg_key / sqrt(d) + log(group_count)
5. Softmax over special keys + all group reps

The log(count) term in the score compensates for the fact
that a group of C keys should carry C times the attention
mass of a single key with the same average logit.

Budget: scales with B (the number of groups requested).


### IdealEqualWeightSplits

1. For each query, sort all non-special keys by attention
   weight (descending)
2. Split into B groups so each group captures approximately
   equal total attention weight mass. High-weight keys get
   more groups (finer resolution where it matters).
3. Represent each group by (avg_key, avg_value)
4. Score: q^T avg_key / sqrt(d) + log(group_count)
5. Softmax over special keys + all group reps

This is a smarter per-query grouping strategy that
allocates groups proportional to attention mass, giving
finer resolution where it matters most.

Budget: scales with B (the number of groups requested).


All idealized methods require knowing the true attention
distribution and are not possible when attempting to
speed up N^2 attention in practice.


### Grouping Methods

**Pure grouped (top_k = 0 in hybrid mode):**
All G groups as representatives. No individual keys.
This is just hybrid with k=0, not a separate mode.

**Hybrid mode (top_k = k, mode = "hybrid"):**
Expand top-k groups individually. Represent remaining
(G - k) groups as centroids. Count-weighted softmax over
all items:
- Special keys: exact logits, count = 1
- Expanded keys: exact logits, count = 1
- Group reps: score = q^T avg_key/sqrt(d) + log(count),
  value = avg_value

To demonstrate usefulness of groups - Compare to typical TopK attention:
**TopK mode (top_k = k, mode = "topk"):**
Expand the top-k groups into individual keys. Exact
softmax attention over special + expanded keys only.


### Count-Weighted Softmax

For a group of C keys with average key mu_k and average
value mu_v, the representative score includes a log(C)
bias:

    score_group = q^T mu_k / sqrt(d) + log(C)

This approximates: sum_{i in group} exp(q^T k_i / sqrt(d))
by C * exp(q^T mu_k / sqrt(d)), using the identity
log(C * x) = log(x) + log(C) inside the softmax.


## Special Keys

In many implementations, positions 0 (attention sink) and the last W positions
(local window) always receive exact attention. The
approximation applies only to the remaining candidate keys.
The final softmax is computed jointly over special keys
and approximated keys/groups.



## Our Methods

### MultiQ Grouping

Cluster all query vectors to discover representative
"prototype queries", then pre-sort keys for each
prototype. At inference time, route each query to its
nearest prototype and use that prototype's pre-sorted
key ordering.

1. Run KMeans on **all** query vectors (entire sequence)
   -> C centroids (prototype queries)
2. For each centroid c_j: sort all N keys by
   c_j^T k_i / sqrt(d), partition into G equal groups
3. Per query: route to nearest centroid
   (argmax c_j^T q), use that centroid's grouping
4. Apply TopK or Hybrid mode

**Cost:** Offline O(C * N log N). Per-query O(C + G).

Setting C=1 reduces to sorting keys by the mean query
(a single global ordering). Higher C adapts to query
diversity — different query clusters get different key
orderings optimized for their attention pattern.


### KMeans Clustering

1. Run flat KMeans on keys -> C clusters
2. Precompute per-cluster: avg_key, avg_value, count,
   member indices
3. Per query: score each cluster by
   q^T avg_key_c / sqrt(d) + log(count_c)
4. Sort clusters by score, treat as groups
5. Apply TopK or Hybrid mode

**Cost:** Offline O(N * C * n_iter). Per-query O(C).
