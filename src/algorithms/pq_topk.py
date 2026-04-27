"""
Product Quantization (PQ) and IVF-PQ for approximate top-k.

PQIndex: flat PQ scan — splits d-dim keys into m subvectors,
         quantizes each, approximates q·k via lookup table sums.

IVFPQIndex: coarse KMeans on full keys (Voronoi cells) + PQ
            within each cell. Probe nearest cells for top-k,
            un-probed cells provide cluster residuals for free.
"""

import numpy as np
from ..core import flat_kmeans


class PQIndex:
    """
    Flat Product Quantization index for approximate MIPS.

    Parameters:
        m: number of subspaces (must divide d)
        n_codes: codebook size per subspace (default 256)
        n_iter: KMeans iterations for codebook training
    """

    def __init__(self, m: int = 8, n_codes: int = 256,
                 n_iter: int = 30, seed: int = 42):
        self.m = m
        self.n_codes = n_codes
        self.n_iter = n_iter
        self.seed = seed
        self.codebooks = None
        self.codes = None

    def fit(self, keys: np.ndarray):
        """Build PQ codebooks and encode all keys."""
        n, d = keys.shape
        m = self.m
        assert d % m == 0, f"d={d} not divisible by m={m}"
        dsub = d // m

        codebooks = np.empty(
            (m, self.n_codes, dsub), dtype=np.float32,
        )
        codes = np.empty((n, m), dtype=np.int32)

        for i in range(m):
            sub = keys[:, i * dsub:(i + 1) * dsub]
            nc = min(self.n_codes, n)
            centroids, labels = flat_kmeans(
                sub, nc,
                seed=self.seed + i,
                n_iter=self.n_iter,
            )
            codebooks[i, :nc] = centroids
            codes[:, i] = labels

        self.codebooks = codebooks
        self.codes = codes
        self.dsub = dsub

    def approximate_topk(
        self, query: np.ndarray, k: int,
        candidate_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Return global indices of approximate top-k keys.

        candidate_mask: [n] bool, True = eligible.
        Returns: [k] int64 global indices.
        """
        m = self.m
        dsub = self.dsub

        lut = np.empty(
            (m, self.n_codes), dtype=np.float32,
        )
        for i in range(m):
            q_sub = query[i * dsub:(i + 1) * dsub]
            lut[i] = self.codebooks[i] @ q_sub

        approx_ip = np.zeros(
            len(self.codes), dtype=np.float32,
        )
        for i in range(m):
            approx_ip += lut[i][self.codes[:, i]]

        if candidate_mask is not None:
            approx_ip[~candidate_mask] = -np.inf

        n_valid = int(np.sum(approx_ip > -np.inf))
        k_use = min(k, n_valid)
        if k_use <= 0:
            return np.array([], dtype=np.int64)

        topk_idx = np.argpartition(
            approx_ip, -k_use,
        )[-k_use:]

        return topk_idx.astype(np.int64)


class IVFPQIndex:
    """
    IVF-PQ index: coarse KMeans on full keys + PQ within.

    The coarse quantizer partitions keys into C Voronoi
    cells. Per query, probe the nearest `nprobe` cells
    and use PQ to find approximate top-k within them.

    Un-probed cells provide cluster residuals:
      - centroid = mean key of the cell
      - value_sum = precomputed sum of values in the cell
      - count = number of keys in the cell

    Parameters:
        n_cells: number of coarse Voronoi cells
        m: PQ subspaces
        n_codes: PQ codebook size per subspace
    """

    def __init__(self, n_cells: int = 1024,
                 m: int = 8, n_codes: int = 256,
                 n_iter_coarse: int = 50,
                 n_iter_pq: int = 30,
                 seed: int = 42):
        self.n_cells = n_cells
        self.m = m
        self.n_codes = n_codes
        self.n_iter_coarse = n_iter_coarse
        self.n_iter_pq = n_iter_pq
        self.seed = seed

        # Filled by fit()
        self.centroids = None      # [C, d] coarse centroids
        self.cell_labels = None    # [n] cell assignment
        self.cell_members = None   # list of arrays per cell
        self.cell_value_sums = None  # [C, d]
        self.cell_counts = None    # [C]
        self.pq = None             # PQIndex for fine search

    def fit(self, keys: np.ndarray, values: np.ndarray):
        """
        Build coarse quantizer + PQ + precompute cell stats.

        keys: [n, d]
        values: [n, d]
        """
        n, d = keys.shape
        C = min(self.n_cells, n)

        # Coarse KMeans on full keys
        centroids, cell_labels = flat_kmeans(
            keys, C,
            seed=self.seed,
            n_iter=self.n_iter_coarse,
        )
        self.centroids = centroids
        self.cell_labels = cell_labels

        # Per-cell member lists
        self.cell_members = [
            np.where(cell_labels == c)[0].astype(np.int64)
            for c in range(C)
        ]

        # Per-cell value sums and counts
        self.cell_value_sums = np.zeros(
            (C, d), dtype=np.float64,
        )
        vals_f = values.astype(np.float64)
        for j in range(d):
            self.cell_value_sums[:, j] = np.bincount(
                cell_labels, weights=vals_f[:, j],
                minlength=C,
            )
        self.cell_counts = np.bincount(
            cell_labels, minlength=C,
        ).astype(np.float64)

        # Per-cell key sums (for mean key of residuals)
        self.cell_key_sums = np.zeros(
            (C, d), dtype=np.float64,
        )
        keys_f = keys.astype(np.float64)
        for j in range(d):
            self.cell_key_sums[:, j] = np.bincount(
                cell_labels, weights=keys_f[:, j],
                minlength=C,
            )

        # PQ on full keys for fine-grained scoring
        self.pq = PQIndex(
            m=self.m, n_codes=self.n_codes,
            n_iter=self.n_iter_pq,
            seed=self.seed + 999,
        )
        self.pq.fit(keys)

    def search(
        self, query: np.ndarray, k: int, nprobe: int,
        candidate_mask: np.ndarray = None,
    ):
        """
        Find approximate top-k from probed cells.

        Returns:
            topk_idx: [k] global indices of top-k keys
            probed_cells: set of cell IDs that were probed
            unprobed_cells: list of (cell_id, count, mean_key,
                            mean_value) for un-probed non-empty cells
        """
        d = self.centroids.shape[1]
        C = len(self.centroids)

        # Score all cells by query · centroid
        cell_scores = self.centroids @ query
        nprobe = min(nprobe, C)
        probed_cell_ids = np.argpartition(
            cell_scores, -nprobe,
        )[-nprobe:]
        probed_set = set(probed_cell_ids.tolist())

        # Collect candidate keys from probed cells
        probed_keys_idx = []
        for c in probed_cell_ids:
            members = self.cell_members[c]
            if candidate_mask is not None:
                members = members[candidate_mask[members]]
            probed_keys_idx.append(members)

        if probed_keys_idx:
            all_probed = np.concatenate(probed_keys_idx)
        else:
            all_probed = np.array([], dtype=np.int64)

        # PQ top-k within probed keys only
        if len(all_probed) > 0 and k > 0:
            probe_mask = np.zeros(
                len(self.pq.codes), dtype=bool,
            )
            probe_mask[all_probed] = True
            if candidate_mask is not None:
                probe_mask &= candidate_mask
            topk_idx = self.pq.approximate_topk(
                query, min(k, len(all_probed)),
                candidate_mask=probe_mask,
            )
        else:
            topk_idx = np.array([], dtype=np.int64)

        # Un-probed non-empty cells (for cluster residuals)
        unprobed_info = []
        for c in range(C):
            if c in probed_set:
                continue
            cnt = self.cell_counts[c]
            if cnt == 0:
                continue
            # Filter out special keys from this cell
            if candidate_mask is not None:
                members = self.cell_members[c]
                cand_members = members[candidate_mask[members]]
                cand_cnt = len(cand_members)
                if cand_cnt == 0:
                    continue
            else:
                cand_cnt = int(cnt)

            mean_k = (
                self.cell_key_sums[c] / cnt
            ).astype(np.float32)
            mean_v = (
                self.cell_value_sums[c] / cnt
            ).astype(np.float32)
            unprobed_info.append(
                (c, cand_cnt, mean_k, mean_v)
            )

        return topk_idx, probed_set, unprobed_info
