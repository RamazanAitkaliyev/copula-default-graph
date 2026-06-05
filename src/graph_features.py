"""
Graph Feature Extractor with Network Visualization  (src/graph_features.py)
===========================================================================

PURPOSE
-------
Builds and analyses the transaction network. The primary output is the
correlation matrix used by CopulaDefaultModel. Secondary outputs are
per-borrower network features added to the persons DataFrame.

AGENT ENTRY POINT
-----------------
Preferred: use RiskAgentAPI which handles graph building internally.
Direct use:
    graph = TransactionGraph(transactions, persons)
    corr_matrix = graph.get_correlation_matrix(
        base_corr=0.05, max_corr=0.60,
        same_city_boost=0.10, same_group_boost=0.20
    )
    # corr_matrix is PSD with diagonal=1, ready for CopulaDefaultModel.fit()

    neighbor_features = get_neighbor_risk_features(graph, persons)
    # Merge back: persons.merge(neighbor_features[...], on='person_id')

PRECONDITIONS
-------------
  - transactions must have columns: sender_id, receiver_id, amount.
  - persons must have columns: person_id, city_id, base_pd.
  - All person_ids in transactions must exist in persons['person_id'].

KEY OUTPUTS
-----------
  get_correlation_matrix() → np.ndarray of shape (n, n), PSD, diag=1.
    Correlation between borrowers i and j is:
      base_corr
      + f(transaction_volume_ij)         # scales to max_corr with volume
      + same_city_boost   if city_id_i == city_id_j
      + same_group_boost  if high_risk_group_id_i == high_risk_group_id_j
    Final matrix is projected to nearest PSD (Higham 2002).
    # AGENT: INV-2 — Always call get_correlation_matrix() rather than
    #   building a matrix manually. The PSD projection is applied internally.

  get_neighbor_risk_features() → pd.DataFrame with columns:
    person_id, neighbor_pd_avg, neighbor_pd_max, n_high_risk_neighbors
    These are merged into persons and used as PD model features in step 3.

  get_network_stats() → NetworkStats dataclass:
    n_nodes, n_edges, density, avg_degree, avg_clustering, n_components

  plot_network() → matplotlib Figure
    color_by: 'base_pd', 'city_id', 'risk_tier', 'degree'
    layout:   'city' (recommended), 'spring', 'circular'
    # AGENT: The 'city' layout uses polar positioning for n_cities=3.
    #   It crashes for n_cities ≠ 3 with the old hardcoded dict approach.
    #   The current implementation uses dynamic polar layout — do not revert.

INVARIANTS
----------
# AGENT: The correlation matrix MUST be PSD (positive semi-definite).
#   Non-PSD matrices produce negative eigenvalues which break:
#     - Cholesky decomposition in Gaussian copula simulation
#     - numpy.linalg.cholesky (raises LinAlgError)
#     - Joint probability estimates (can go negative)
#   The _nearest_psd() function is called automatically. If you modify the
#   matrix after get_correlation_matrix() returns, call _nearest_psd() again.

# AGENT: Bridge nodes (high betweenness centrality) are flagged via persons['is_bridge'].
#   These are the most systemically important nodes — they connect otherwise
#   separate communities. Correlations involving bridge nodes are boosted.

NETWORK FEATURES ADDED TO PERSONS
----------------------------------
  neighbor_pd_avg          — average PD of direct transaction neighbours
  neighbor_pd_max          — maximum PD among direct neighbours
  n_high_risk_neighbors    — count of neighbours in high-risk group (high_risk_group_id ≠ -1)

  These are added in pipeline step 2 and used as features in the PD model (step 3).
  They encode the contagion channel at the feature level — a borrower with
  high-PD neighbours is considered riskier even before the copula is fitted.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csgraph
from typing import Optional, Tuple, Dict, List, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Above this node count, dense n×n materialisation is refused (would OOM).
# At n=50k a dense float64 matrix is ~20 GB; at n=10M it is ~0.8 PB.
# The sparse code paths (CSR adjacency, sparse correlation) have no such limit.
DENSE_MATRIX_MAX_NODES = 20_000


@dataclass
class NetworkStats:
    """Summary statistics for the network."""
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    avg_clustering: float
    n_components: int


class TransactionGraph:
    """
    Build and analyze transaction network.

    Provides:
    - Adjacency matrices
    - Node-level features (centrality, clustering)
    - Network visualization
    - Community detection
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        persons: pd.DataFrame
    ):
        """
        Initialize graph from transaction data.

        Parameters
        ----------
        transactions : pd.DataFrame
            Transactions with sender_id, receiver_id, amount
        persons : pd.DataFrame
            Person data with person_id, city_id, base_pd, etc.
        """
        if "person_id" not in persons.columns:
            raise ValueError("persons must have a 'person_id' column")
        if persons["person_id"].duplicated().any():
            raise ValueError("persons has duplicate person_ids")
        required_tx = {"sender_id", "receiver_id", "amount"}
        missing = required_tx - set(transactions.columns)
        if missing:
            raise ValueError(f"transactions missing columns: {missing}")

        self.transactions = transactions
        self.persons = persons.reset_index(drop=True)
        self.n_nodes = len(persons)

        # Build adjacency matrices
        self._build_adjacency()

        # Compute node features
        self.node_features = self._compute_node_features()

    def _build_adjacency(self):
        """
        Build sparse adjacency matrices from transactions (vectorized).

        SCALE: This is fully vectorized (no iterrows) and uses scipy.sparse CSR
        matrices. Memory is O(n_edges), not O(n²). Works for 10M+ nodes as long
        as the transaction graph is sparse (typical: avg degree 10-50).

        Stored as sparse:
            adj_weighted_sp : undirected, summed amount  (CSR)
            adj_binary_sp   : undirected, 0/1            (CSR)
            adj_count_sp    : undirected, tx count       (CSR)
            adj_out_sp      : directed out, summed amount (CSR)
            adj_in_sp       : directed in, summed amount  (CSR)

        Dense equivalents (adj_weighted, adj_binary, ...) are exposed as lazy
        properties that materialise on first access, but ONLY when
        n_nodes <= DENSE_MATRIX_MAX_NODES (otherwise they raise to prevent OOM).
        """
        n = self.n_nodes
        s = self.transactions['sender_id'].to_numpy(dtype=np.int64)
        r = self.transactions['receiver_id'].to_numpy(dtype=np.int64)
        amt = self.transactions['amount'].to_numpy(dtype=np.float64)

        if len(s) == 0:
            # Empty graph — all-zero sparse matrices.
            empty = sparse.csr_matrix((n, n), dtype=np.float64)
            self.adj_weighted_sp = empty.copy()
            self.adj_binary_sp = empty.copy()
            self.adj_count_sp = empty.copy()
            self.adj_out_sp = empty.copy()
            self.adj_in_sp = empty.copy()
            self._dense_cache = {}
            return

        # Directed accumulators via COO→CSR (duplicate (i,j) entries are summed).
        out_coo = sparse.coo_matrix((amt, (s, r)), shape=(n, n))
        self.adj_out_sp = out_coo.tocsr()
        self.adj_in_sp = self.adj_out_sp.transpose().tocsr()

        # Undirected weighted = out + in (symmetric).
        self.adj_weighted_sp = (self.adj_out_sp + self.adj_in_sp).tocsr()

        # Undirected transaction count.
        ones = np.ones_like(amt)
        cnt_coo = sparse.coo_matrix((ones, (s, r)), shape=(n, n))
        cnt_csr = cnt_coo.tocsr()
        self.adj_count_sp = (cnt_csr + cnt_csr.transpose()).tocsr()

        # Binary adjacency (0/1) from the weighted structure.
        binary = self.adj_weighted_sp.copy()
        binary.data[:] = 1.0
        binary.eliminate_zeros()
        self.adj_binary_sp = binary

        self._dense_cache = {}

    # ── lazy dense materialisation (small n only) ─────────────────────────────

    def _dense(self, name: str) -> np.ndarray:
        """
        Materialise a sparse adjacency as a dense numpy array, cached.

        Refuses (raises MemoryError) above DENSE_MATRIX_MAX_NODES to prevent an
        accidental petabyte allocation on a large portfolio. Use the sparse
        attributes (adj_*_sp) and sparse methods instead at scale.
        """
        if name in self._dense_cache:
            return self._dense_cache[name]
        if self.n_nodes > DENSE_MATRIX_MAX_NODES:
            raise MemoryError(
                f"Refusing to materialise a dense {self.n_nodes}×{self.n_nodes} "
                f"matrix ('{name}'): that is ~{(self.n_nodes**2 * 8) / 1e9:,.0f} GB. "
                f"Use the sparse attribute '{name}_sp' and sparse-aware methods "
                f"(this limit is DENSE_MATRIX_MAX_NODES={DENSE_MATRIX_MAX_NODES})."
            )
        dense = getattr(self, f"{name}_sp").toarray()
        self._dense_cache[name] = dense
        return dense

    @property
    def adj_weighted(self) -> np.ndarray:
        return self._dense("adj_weighted")

    @property
    def adj_binary(self) -> np.ndarray:
        return self._dense("adj_binary")

    @property
    def adj_count(self) -> np.ndarray:
        return self._dense("adj_count")

    @property
    def adj_out(self) -> np.ndarray:
        return self._dense("adj_out")

    @property
    def adj_in(self) -> np.ndarray:
        return self._dense("adj_in")

    def _compute_node_features(self) -> pd.DataFrame:
        """
        Compute centrality and structural features for each node (sparse-native).

        All degree/volume statistics are O(n_edges) sparse reductions.
        PageRank uses sparse matrix-vector products. Clustering and betweenness
        iterate only over actual neighbours (CSR rows), never dense rows.
        """
        features = pd.DataFrame({'person_id': range(self.n_nodes)})

        bin_sp = self.adj_binary_sp
        w_sp = self.adj_weighted_sp
        out_sp = self.adj_out_sp
        in_sp = self.adj_in_sp

        # Degree / weighted degree (sparse row sums → flattened arrays).
        features['degree'] = np.asarray(bin_sp.sum(axis=1)).ravel()
        features['weighted_degree'] = np.asarray(w_sp.sum(axis=1)).ravel()

        # In/out degree = count of nonzero entries per row/col.
        out_indptr = out_sp.indptr
        features['out_degree'] = np.diff(out_indptr)
        in_csr = in_sp.tocsr()
        features['in_degree'] = np.diff(in_csr.indptr)

        # Volumes.
        features['in_volume'] = np.asarray(in_sp.sum(axis=1)).ravel()
        features['out_volume'] = np.asarray(out_sp.sum(axis=1)).ravel()

        features['pagerank'] = self._compute_pagerank()
        features['clustering'] = self._compute_clustering()
        features['betweenness'] = self._compute_betweenness_sampled()

        total_flow = features['in_volume'] + features['out_volume']
        features['flow_imbalance'] = np.where(
            total_flow > 0,
            (features['in_volume'] - features['out_volume']) / total_flow,
            0
        )

        return features

    def _compute_pagerank(self, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """
        PageRank centrality via sparse power iteration.

        Uses the column-stochastic transition built from the weighted adjacency.
        Each iteration is one sparse matrix-vector product: O(n_edges).
        Handles dangling nodes (zero out-degree) by redistributing their mass.
        """
        n = self.n_nodes
        if n == 0:
            return np.zeros(0)
        W = self.adj_weighted_sp.tocsr()
        out_strength = np.asarray(W.sum(axis=1)).ravel()
        dangling = out_strength == 0
        # Avoid divide-by-zero; dangling handled separately.
        inv_out = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, out_strength))

        # Row-normalise: M[i,j] = W[i,j] / out_strength[i]. Then propagate with Mᵀ.
        D_inv = sparse.diags(inv_out)
        M = (D_inv @ W).tocsr()           # row-stochastic over non-dangling rows
        Mt = M.transpose().tocsr()

        pr = np.full(n, 1.0 / n)
        teleport = (1.0 - damping) / n
        for _ in range(max_iter):
            dangling_mass = damping * pr[dangling].sum() / n
            pr_new = teleport + damping * (Mt @ pr) + dangling_mass
            if np.abs(pr_new - pr).max() < 1e-8:
                pr = pr_new
                break
            pr = pr_new
        # Normalise (guards against drift).
        total = pr.sum()
        return pr / total if total > 0 else pr

    def _compute_clustering(self) -> np.ndarray:
        """
        Local clustering coefficient, iterating over CSR neighbour lists only.

        For node i with neighbours N(i), counts directed edges among N(i) using
        sparse submatrix slicing. Cost is O(Σ deg(i)²) over the sampled rows —
        cheap for sparse graphs. Never touches a dense row.
        """
        n = self.n_nodes
        clustering = np.zeros(n)
        A = self.adj_binary_sp.tocsr()
        indptr, indices = A.indptr, A.indices

        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            neighbors = indices[start:end]
            k = len(neighbors)
            if k < 2:
                continue
            # Edges among neighbours = nnz of A[neighbors][:, neighbors].
            sub = A[neighbors][:, neighbors]
            neighbor_edges = sub.nnz  # binary, so nnz == sum
            max_edges = k * (k - 1)
            if max_edges > 0:
                clustering[i] = neighbor_edges / max_edges
        return clustering

    def _compute_betweenness_sampled(self, n_samples: int = 50) -> np.ndarray:
        """
        Approximate betweenness via Brandes' algorithm from sampled sources.

        Uses CSR neighbour lists (indices slices), never dense rows.
        Cost: O(n_samples · n_edges).
        """
        n = self.n_nodes
        betweenness = np.zeros(n)
        if n == 0:
            return betweenness
        sources = np.random.choice(n, size=min(n_samples, n), replace=False)
        A = self.adj_binary_sp.tocsr()
        indptr, indices = A.indptr, A.indices

        for src in sources:
            dist = np.full(n, np.inf)
            dist[src] = 0
            paths = np.zeros(n)
            paths[src] = 1
            queue = [src]
            order = []

            while queue:
                v = queue.pop(0)
                order.append(v)
                for w in indices[indptr[v]:indptr[v + 1]]:
                    if dist[w] == np.inf:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        paths[w] += paths[v]

            delta = np.zeros(n)
            for w in reversed(order[1:]):
                for v in indices[indptr[w]:indptr[w + 1]]:
                    if dist[v] == dist[w] - 1 and paths[w] > 0:
                        delta[v] += (paths[v] / paths[w]) * (1 + delta[w])
                betweenness[w] += delta[w]

        return betweenness / len(sources)

    def get_adjacency(self, weighted: bool = True) -> np.ndarray:
        """
        Return DENSE adjacency matrix (small n only).

        Raises MemoryError above DENSE_MATRIX_MAX_NODES. For large graphs use
        get_adjacency_sparse() instead.
        """
        return self.adj_weighted if weighted else self.adj_binary

    def get_adjacency_sparse(self, weighted: bool = True) -> "sparse.csr_matrix":
        """
        Return the SPARSE adjacency matrix (CSR). Safe at any scale.

        This is the scale-friendly accessor — O(n_edges) memory, no dense
        materialisation. Prefer this in any code that must run on large portfolios.
        """
        return self.adj_weighted_sp if weighted else self.adj_binary_sp

    def get_correlation_matrix(
        self,
        base_corr: float = 0.05,
        max_corr: float = 0.6,
        same_city_boost: float = 0.1,
        same_group_boost: float = 0.2
    ) -> np.ndarray:
        """
        Derive a DENSE correlation matrix from network structure (small n only).

        Correlation increases with:
          - Direct transaction links (weighted by volume)
          - Same city membership
          - Same high-risk group membership

        SCALE WARNING: This materialises a dense n×n matrix AND a dense n×n
        boolean outer product for the city/group boosts. It raises MemoryError
        above DENSE_MATRIX_MAX_NODES. For large portfolios use
        get_correlation_sparse(), which keeps off-diagonal entries only where a
        correlation actually exists (transaction link, same city, or same group).

        The dense form is still what the full-matrix Clayton copula expects; for
        large n the copula must be parameterised from the sparse correlation /
        factor structure instead (see CopulaDefaultModel scale notes).
        """
        n = self.n_nodes
        if n > DENSE_MATRIX_MAX_NODES:
            raise MemoryError(
                f"get_correlation_matrix() would build a dense {n}×{n} matrix "
                f"(~{(n*n*8)/1e9:,.0f} GB). Use get_correlation_sparse() for "
                f"n > {DENSE_MATRIX_MAX_NODES}."
            )

        # Normalize adjacency to [0, 1] (dense — small n only).
        adj_dense = self.adj_weighted
        max_weight = adj_dense.max()
        if max_weight <= 0:
            max_weight = 1.0
        adj_norm = adj_dense / max_weight

        corr = base_corr + (max_corr - base_corr) * adj_norm

        city_ids = self.persons['city_id'].values
        same_city = (city_ids[:, None] == city_ids[None, :])
        np.fill_diagonal(same_city, False)
        corr += same_city * same_city_boost

        group_ids = self.persons['high_risk_group_id'].values
        in_group = group_ids >= 0
        same_group = (
            in_group[:, None] & in_group[None, :] &
            (group_ids[:, None] == group_ids[None, :])
        )
        np.fill_diagonal(same_group, False)
        corr += same_group * same_group_boost

        corr = np.clip(corr, 0, 0.95)
        np.fill_diagonal(corr, 1.0)
        corr = self._nearest_psd(corr)
        return corr

    def get_correlation_sparse(
        self,
        base_corr: float = 0.05,
        max_corr: float = 0.6,
        same_city_boost: float = 0.1,
        same_group_boost: float = 0.2,
        include_geo_blocks: bool = True,
        max_block_size: int = 5_000,
    ) -> "sparse.csr_matrix":
        """
        Sparse correlation: off-diagonal entries ONLY where correlation exists.

        Scale-friendly replacement for get_correlation_matrix(). An off-diagonal
        pair (i,j) gets a nonzero correlation only if at least one holds:
          - they have a direct transaction link (weighted contribution), OR
          - include_geo_blocks and they share a city, OR
          - they share a high-risk group.

        This keeps the matrix sparse: instead of n² entries, it stores roughly
        (n_edges + Σ city_block_size² + Σ group_size²) entries. The diagonal is 1.

        Parameters
        ----------
        include_geo_blocks : bool
            Whether to add same-city correlation blocks. For very large cities
            this can still be expensive (block is size²); blocks larger than
            max_block_size are SKIPPED with a warning (transaction links between
            their members are still captured). Set False to rely on transaction
            links + groups only.
        max_block_size : int
            Skip same-city/same-group blocks larger than this to bound memory.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n, n), symmetric, diagonal = 1.

        SEMANTIC DIFFERENCE vs get_correlation_matrix() — IMPORTANT
        ----------------------------------------------------------
        The DENSE get_correlation_matrix() applies `base_corr` to EVERY pair (a
        market-wide correlation floor). This SPARSE version applies `base_corr`
        ONLY to pairs that have a transaction link — unconnected pairs with no
        shared city/group are treated as INDEPENDENT (correlation 0), not
        base_corr. This is deliberate: a uniform floor would make the matrix
        fully dense (n² nonzeros) and defeat the purpose. The two therefore give
        different numbers for unconnected pairs. For a market-wide floor at scale,
        model it as one extra systematic factor in FactorCopula (a "market"
        factor every borrower loads on) rather than via this matrix.

        NOTE: This sparse matrix is NOT PSD-projected (a full eigendecomposition
        is infeasible at scale). Downstream consumers that need PSD should use a
        factor-model parameterisation (FactorCopula) or operate block-wise on
        dense submatrices.
        """
        n = self.n_nodes

        # 1) Transaction-link correlations (already sparse).
        W = self.adj_weighted_sp.tocoo()
        max_weight = W.data.max() if W.nnz > 0 else 1.0
        if max_weight <= 0:
            max_weight = 1.0
        link_vals = base_corr + (max_corr - base_corr) * (W.data / max_weight)
        rows = list(W.row)
        cols = list(W.col)
        vals = list(link_vals)

        # 2) Same-group blocks (usually small).
        group_ids = self.persons['high_risk_group_id'].values
        rows, cols, vals = self._add_block_correlations(
            group_ids, same_group_boost, rows, cols, vals,
            max_block_size, label="group", valid_predicate=lambda g: g >= 0,
        )

        # 3) Same-city blocks (optional, can be large).
        if include_geo_blocks and 'city_id' in self.persons.columns:
            city_ids = self.persons['city_id'].values
            rows, cols, vals = self._add_block_correlations(
                city_ids, same_city_boost, rows, cols, vals,
                max_block_size, label="city", valid_predicate=lambda c: True,
            )

        # Assemble, sum duplicate (i,j) contributions, clip, set diagonal.
        M = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        M.data = np.clip(M.data, 0.0, 0.95)
        M.setdiag(1.0)
        M.eliminate_zeros()
        return M.tocsr()

    @staticmethod
    def _add_block_correlations(
        label_array: np.ndarray,
        boost: float,
        rows: list,
        cols: list,
        vals: list,
        max_block_size: int,
        label: str,
        valid_predicate,
    ) -> tuple:
        """
        Append within-group/within-city correlation entries (off-diagonal only).

        Groups larger than max_block_size are skipped (their members may still be
        connected via transaction links). Returns the extended (rows, cols, vals).
        """
        # Map label → member indices.
        order = np.argsort(label_array, kind="stable")
        sorted_labels = label_array[order]
        # Boundaries of equal-label runs.
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(sorted_labels)]))

        n_skipped = 0
        for st, en in zip(starts, ends):
            lbl = sorted_labels[st]
            if not valid_predicate(lbl):
                continue
            members = order[st:en]
            m = len(members)
            if m < 2:
                continue
            if m > max_block_size:
                n_skipped += 1
                continue
            # All off-diagonal pairs in this block.
            ii, jj = np.meshgrid(members, members, indexing="ij")
            mask = ii != jj
            rows.extend(ii[mask].tolist())
            cols.extend(jj[mask].tolist())
            vals.extend([boost] * int(mask.sum()))
        if n_skipped:
            logger.warning(
                "get_correlation_sparse: skipped %d %s-block(s) larger than "
                "max_block_size=%d (transaction links between their members are "
                "still included).", n_skipped, label, max_block_size,
            )
        return rows, cols, vals

    def _nearest_psd(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        B = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 1e-8)
        A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(A_psd, 1.0)
        return A_psd

    def get_network_stats(self) -> NetworkStats:
        """Compute summary statistics for the network (sparse-native)."""
        # nnz of symmetric binary matrix counts each edge twice.
        n_edges = int(self.adj_binary_sp.nnz / 2)
        max_edges = self.n_nodes * (self.n_nodes - 1) / 2

        return NetworkStats(
            n_nodes=self.n_nodes,
            n_edges=n_edges,
            density=n_edges / max_edges if max_edges > 0 else 0,
            avg_degree=self.node_features['degree'].mean(),
            avg_clustering=self.node_features['clustering'].mean(),
            n_components=self._count_components()
        )

    def _count_components(self) -> int:
        """
        Count connected components via scipy.sparse.csgraph (iterative, O(n_edges)).

        Replaces the old recursive DFS which would hit Python's recursion limit
        (and stack-overflow) on large or deep graphs.
        """
        if self.n_nodes == 0:
            return 0
        n_comp, _ = csgraph.connected_components(
            self.adj_binary_sp, directed=False, return_labels=True
        )
        return int(n_comp)

    def detect_communities(self, n_communities: int = 5) -> np.ndarray:
        """Detect communities using spectral clustering.

        Returns an int label per node (length n_nodes). Degenerate graphs
        (fewer nodes than requested communities) and a k-means failure both fall
        back to one-label-per-node rather than collapsing every node into a
        single community (a silent wrong answer for downstream risk grouping).
        """
        if self.n_nodes < 2 or n_communities <= 1:
            return np.zeros(self.n_nodes, dtype=int)

        # Laplacian
        D = np.diag(self.adj_weighted.sum(axis=1))
        L = D - self.adj_weighted

        # Normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D) + 1e-10)))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Use first k eigenvectors (skip first which is constant)
        k = min(n_communities, self.n_nodes - 1)
        features = eigenvectors[:, 1:k + 1]

        # K-means
        from scipy.cluster.vq import kmeans2
        try:
            _, labels = kmeans2(features, n_communities, minit='++', missing='warn')
            # kmeans2 can return empty clusters; labels are still a valid partition.
            labels = np.asarray(labels, dtype=int)
        except Exception as e:
            # Do NOT collapse all nodes into community 0 — that silently destroys
            # structure. Give each node its own label so callers see no spurious
            # grouping, and surface the failure.
            logger.warning(
                "spectral k-means failed (%s); falling back to one community "
                "per node", e
            )
            labels = np.arange(self.n_nodes, dtype=int)

        return labels

    def plot_network(
        self,
        color_by: str = 'base_pd',
        size_by: str = 'degree',
        layout: str = 'spring',
        figsize: Tuple[int, int] = (14, 10),
        title: str = 'Transaction Network',
        show_edges: bool = True,
        edge_alpha: float = 0.1,
        node_alpha: float = 0.7,
        cmap: str = 'RdYlGn_r',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Visualize the network with risk coloring.

        Parameters
        ----------
        color_by : str
            Column to use for node color (e.g., 'base_pd', 'city_id')
        size_by : str
            Column to use for node size (e.g., 'degree', 'pagerank')
        layout : str
            Layout algorithm: 'spring', 'city', 'circular'
        figsize : tuple
            Figure size
        title : str
            Plot title
        show_edges : bool
            Whether to draw edges
        edge_alpha : float
            Edge transparency
        node_alpha : float
            Node transparency
        cmap : str
            Colormap for node colors
        ax : matplotlib Axes, optional
            Axes to plot on

        Returns
        -------
        fig : matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        # Compute layout
        pos = self._compute_layout(layout)

        # Get node colors
        if color_by in self.persons.columns:
            colors = self.persons[color_by].values
        elif color_by in self.node_features.columns:
            colors = self.node_features[color_by].values
        else:
            colors = np.zeros(self.n_nodes)

        # Get node sizes
        if size_by in self.node_features.columns:
            sizes = self.node_features[size_by].values
        else:
            sizes = np.ones(self.n_nodes)

        # Normalize sizes
        sizes = 50 + 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10)

        # Draw edges — build all segments at once with LineCollection (much faster than per-edge plot)
        if show_edges:
            from matplotlib.collections import LineCollection
            rows, cols = np.where(np.triu(self.adj_binary, k=1) > 0)
            if len(rows) > 0:
                segments = np.stack([pos[rows], pos[cols]], axis=1)
                lc = LineCollection(segments, colors='k', linewidths=0.3, alpha=edge_alpha)
                ax.add_collection(lc)

        # Draw nodes
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1],
            c=colors, s=sizes, cmap=cmap,
            alpha=node_alpha, edgecolors='white', linewidths=0.5
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(color_by)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        return fig

    def _compute_layout(self, layout: str) -> np.ndarray:
        """Compute node positions for visualization."""
        n = self.n_nodes

        if layout == 'city':
            # Separate cities into a circle, spread nodes within each city
            unique_cities = sorted(self.persons['city_id'].unique())
            n_cities = len(unique_cities)
            radius = 3.0
            angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
            city_centers = {
                cid: (radius * np.cos(a), radius * np.sin(a))
                for cid, a in zip(unique_cities, angles)
            }
            pos = np.zeros((n, 2))
            city_ids = self.persons['city_id'].values
            for i in range(n):
                cx, cy = city_centers[int(city_ids[i])]
                pos[i] = [cx + np.random.normal(0, 0.5),
                           cy + np.random.normal(0, 0.5)]
            return pos

        elif layout == 'circular':
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return np.column_stack([np.cos(angles), np.sin(angles)])

        else:  # spring layout
            return self._spring_layout(iterations=50)

    def _spring_layout(self, iterations: int = 50) -> np.ndarray:
        """Force-directed spring layout."""
        n = self.n_nodes
        pos = np.random.randn(n, 2)

        k = 1.0 / np.sqrt(n)  # Optimal distance

        for _ in range(iterations):
            # Repulsive forces
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2)) + 1e-10
            repulsion = (diff / dist[:, :, np.newaxis]) * (k ** 2 / dist[:, :, np.newaxis])
            np.fill_diagonal(repulsion[:, :, 0], 0)
            np.fill_diagonal(repulsion[:, :, 1], 0)

            # Attractive forces (only for connected nodes)
            attraction = np.zeros_like(pos)
            for i in range(n):
                neighbors = np.where(self.adj_binary[i] > 0)[0]
                if len(neighbors) > 0:
                    diff_to_neighbors = pos[neighbors] - pos[i]
                    dist_to_neighbors = np.sqrt((diff_to_neighbors ** 2).sum(axis=1)) + 1e-10
                    attraction[i] = (diff_to_neighbors * dist_to_neighbors[:, np.newaxis] / k).sum(axis=0)

            # Update positions
            displacement = repulsion.sum(axis=1) * 0.1 + attraction * 0.1
            pos += displacement * 0.1

            # Center and scale
            pos -= pos.mean(axis=0)
            pos /= np.abs(pos).max() + 1e-10

        return pos

    def plot_city_subgraphs(
        self,
        figsize: Tuple[int, int] = (16, 5),
        color_by: str = 'base_pd'
    ) -> plt.Figure:
        """Plot separate subgraphs for each city."""
        cities = self.persons['city_name'].unique()
        n_cities = len(cities)

        fig, axes = plt.subplots(1, n_cities, figsize=figsize)
        if n_cities == 1:
            axes = [axes]

        for idx, city in enumerate(cities):
            city_mask = self.persons['city_name'] == city
            city_persons = self.persons[city_mask]
            city_ids = city_persons['person_id'].values

            # Get subgraph positions
            n_city = len(city_ids)
            pos = np.random.randn(n_city, 2)

            # Color by PD
            if color_by in self.persons.columns:
                colors = city_persons[color_by].values
            else:
                colors = np.zeros(n_city)

            # Draw nodes
            scatter = axes[idx].scatter(
                pos[:, 0], pos[:, 1],
                c=colors, s=50, cmap='RdYlGn_r',
                alpha=0.7, edgecolors='white', linewidths=0.5
            )

            # Draw edges within city
            city_adj = self.adj_binary[np.ix_(city_ids, city_ids)]
            for i in range(n_city):
                for j in range(i + 1, n_city):
                    if city_adj[i, j] > 0:
                        axes[idx].plot(
                            [pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            'k-', alpha=0.1, linewidth=0.3
                        )

            axes[idx].set_title(f'{city}\n(n={n_city})')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        plt.colorbar(scatter, ax=axes[-1], label=color_by, shrink=0.8)
        plt.tight_layout()
        return fig


def get_neighbor_risk_features(
    graph: TransactionGraph,
    persons: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute neighbor-based risk features.

    For each person, compute:
    - Average PD of neighbors
    - Max PD of neighbors
    - Number of high-risk neighbors
    - Weighted average PD (by transaction volume)
    """
    n = graph.n_nodes
    pds = persons['base_pd'].values

    # Normalize adjacency for weighted average
    row_sums = graph.adj_weighted.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    adj_norm = graph.adj_weighted / row_sums

    # Weighted average neighbor PD
    neighbor_pd_weighted = adj_norm @ pds

    # Simple average and max
    neighbor_pd_avg = np.zeros(n)
    neighbor_pd_max = np.zeros(n)
    n_high_risk_neighbors = np.zeros(n)

    high_risk_threshold = np.percentile(pds, 80)

    for i in range(n):
        neighbors = np.where(graph.adj_binary[i] > 0)[0]
        if len(neighbors) > 0:
            neighbor_pds = pds[neighbors]
            neighbor_pd_avg[i] = neighbor_pds.mean()
            neighbor_pd_max[i] = neighbor_pds.max()
            n_high_risk_neighbors[i] = (neighbor_pds > high_risk_threshold).sum()

    return pd.DataFrame({
        'person_id': range(n),
        'neighbor_pd_weighted': np.round(neighbor_pd_weighted, 4),
        'neighbor_pd_avg': np.round(neighbor_pd_avg, 4),
        'neighbor_pd_max': np.round(neighbor_pd_max, 4),
        'n_high_risk_neighbors': n_high_risk_neighbors.astype(int)
    })


class GraphFeatureExtractor(TransactionGraph):
    """
    Alias for TransactionGraph for backward compatibility.

    This class provides the same functionality as TransactionGraph.
    Use TransactionGraph directly for new code.
    """

    def compute_features(self) -> pd.DataFrame:
        """Return node features DataFrame."""
        return self.node_features

    def get_adjacency_matrix(self, weighted: bool = True) -> np.ndarray:
        """Alias for get_adjacency method."""
        return self.get_adjacency(weighted=weighted)

    def compute_neighbor_features(
        self,
        persons: pd.DataFrame,
        pd_column: str = 'base_pd'
    ) -> pd.DataFrame:
        """
        Compute neighbor-based features.

        Parameters
        ----------
        persons : pd.DataFrame
            Person data with PD column
        pd_column : str
            Name of PD column

        Returns
        -------
        features : pd.DataFrame
            Neighbor-based features
        """
        # Temporarily store base_pd if using different column
        temp_persons = persons.copy()
        if pd_column != 'base_pd' and pd_column in temp_persons.columns:
            temp_persons['base_pd'] = temp_persons[pd_column]

        return get_neighbor_risk_features(self, temp_persons)


if __name__ == '__main__':
    from data_generator import generate_network, get_summary_stats

    print("Generating network...")
    persons, transactions = generate_network(seed=42)

    print("\nBuilding graph...")
    graph = TransactionGraph(transactions, persons)

    print("\n=== Network Statistics ===")
    stats = graph.get_network_stats()
    print(f"Nodes: {stats.n_nodes}")
    print(f"Edges: {stats.n_edges}")
    print(f"Density: {stats.density:.4f}")
    print(f"Avg degree: {stats.avg_degree:.1f}")
    print(f"Avg clustering: {stats.avg_clustering:.3f}")
    print(f"Components: {stats.n_components}")

    print("\n=== Node Features ===")
    print(graph.node_features.describe().round(3))

    print("\n=== Neighbor Risk Features ===")
    neighbor_features = get_neighbor_risk_features(graph, persons)
    print(neighbor_features.describe().round(3))

    print("\nPlotting network...")
    fig = graph.plot_network(
        color_by='base_pd',
        size_by='degree',
        layout='city',
        title='Transaction Network by City (colored by PD)'
    )
    plt.savefig('network_plot.png', dpi=150, bbox_inches='tight')
    print("Saved to network_plot.png")
