"""
Graph Feature Extractor with Network Visualization

Builds and analyzes the transaction network:
- Adjacency matrices (binary and weighted)
- Centrality measures (degree, PageRank, betweenness)
- Clustering and community detection
- Network visualization with risk coloring
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
        """Build adjacency matrices from transactions."""
        n = self.n_nodes

        # Weighted by transaction amount (undirected)
        self.adj_weighted = np.zeros((n, n))

        # Binary adjacency
        self.adj_binary = np.zeros((n, n))

        # Transaction count
        self.adj_count = np.zeros((n, n))

        # Directed (for flow analysis)
        self.adj_out = np.zeros((n, n))
        self.adj_in = np.zeros((n, n))

        for _, tx in self.transactions.iterrows():
            s = int(tx['sender_id'])
            r = int(tx['receiver_id'])
            amount = tx['amount']

            # Undirected
            self.adj_weighted[s, r] += amount
            self.adj_weighted[r, s] += amount
            self.adj_binary[s, r] = 1
            self.adj_binary[r, s] = 1
            self.adj_count[s, r] += 1
            self.adj_count[r, s] += 1

            # Directed
            self.adj_out[s, r] += amount
            self.adj_in[r, s] += amount

    def _compute_node_features(self) -> pd.DataFrame:
        """Compute centrality and structural features for each node."""
        features = pd.DataFrame({'person_id': range(self.n_nodes)})

        # Degree centrality
        features['degree'] = self.adj_binary.sum(axis=1)
        features['weighted_degree'] = self.adj_weighted.sum(axis=1)

        # In/out degree (for flow analysis)
        features['in_degree'] = (self.adj_in > 0).sum(axis=0)
        features['out_degree'] = (self.adj_out > 0).sum(axis=1)
        features['in_volume'] = self.adj_in.sum(axis=0)
        features['out_volume'] = self.adj_out.sum(axis=1)

        # PageRank
        features['pagerank'] = self._compute_pagerank()

        # Clustering coefficient
        features['clustering'] = self._compute_clustering()

        # Betweenness (sampled for efficiency)
        features['betweenness'] = self._compute_betweenness_sampled()

        # Flow imbalance (net receiver vs net sender)
        total_flow = features['in_volume'] + features['out_volume']
        features['flow_imbalance'] = np.where(
            total_flow > 0,
            (features['in_volume'] - features['out_volume']) / total_flow,
            0
        )

        return features

    def _compute_pagerank(self, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """Compute PageRank centrality."""
        n = self.n_nodes
        out_degree = self.adj_weighted.sum(axis=1)
        out_degree[out_degree == 0] = 1

        M = self.adj_weighted / out_degree[:, np.newaxis]
        pr = np.ones(n) / n

        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * (M.T @ pr)
            if np.abs(pr_new - pr).max() < 1e-8:
                break
            pr = pr_new

        return pr

    def _compute_clustering(self) -> np.ndarray:
        """Compute local clustering coefficient."""
        n = self.n_nodes
        clustering = np.zeros(n)
        A = self.adj_binary

        for i in range(n):
            neighbors = np.where(A[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue

            # Count edges between neighbors
            neighbor_edges = A[np.ix_(neighbors, neighbors)].sum()
            max_edges = k * (k - 1)
            clustering[i] = neighbor_edges / max_edges if max_edges > 0 else 0

        return clustering

    def _compute_betweenness_sampled(self, n_samples: int = 50) -> np.ndarray:
        """Approximate betweenness via sampling."""
        n = self.n_nodes
        betweenness = np.zeros(n)
        sources = np.random.choice(n, size=min(n_samples, n), replace=False)
        A = self.adj_binary

        for s in sources:
            dist = np.full(n, np.inf)
            dist[s] = 0
            paths = np.zeros(n)
            paths[s] = 1
            queue = [s]
            order = []

            while queue:
                v = queue.pop(0)
                order.append(v)
                for w in np.where(A[v] > 0)[0]:
                    if dist[w] == np.inf:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        paths[w] += paths[v]

            delta = np.zeros(n)
            for w in reversed(order[1:]):
                for v in np.where(A[w] > 0)[0]:
                    if dist[v] == dist[w] - 1 and paths[w] > 0:
                        delta[v] += (paths[v] / paths[w]) * (1 + delta[w])
                betweenness[w] += delta[w]

        return betweenness / len(sources)

    def get_adjacency(self, weighted: bool = True) -> np.ndarray:
        """Return adjacency matrix."""
        return self.adj_weighted if weighted else self.adj_binary

    def get_correlation_matrix(
        self,
        base_corr: float = 0.05,
        max_corr: float = 0.6,
        same_city_boost: float = 0.1,
        same_group_boost: float = 0.2
    ) -> np.ndarray:
        """
        Derive correlation matrix from network structure.

        Correlation increases with:
        - Direct transaction links (weighted by volume)
        - Same city membership
        - Same high-risk group membership
        """
        n = self.n_nodes

        # Normalize adjacency to [0, 1]
        max_weight = self.adj_weighted.max()
        if max_weight <= 0:
            max_weight = 1.0
        adj_norm = self.adj_weighted / max_weight

        # Base correlation from transaction links
        corr = base_corr + (max_corr - base_corr) * adj_norm

        # Same city boost — vectorized outer comparison
        city_ids = self.persons['city_id'].values
        same_city = (city_ids[:, None] == city_ids[None, :])
        np.fill_diagonal(same_city, False)
        corr += same_city * same_city_boost

        # Same group boost — vectorized (only for members with valid group id >= 0)
        group_ids = self.persons['high_risk_group_id'].values
        in_group = group_ids >= 0
        same_group = (
            in_group[:, None] & in_group[None, :] &
            (group_ids[:, None] == group_ids[None, :])
        )
        np.fill_diagonal(same_group, False)
        corr += same_group * same_group_boost

        # Cap correlation
        corr = np.clip(corr, 0, 0.95)

        # Set diagonal to 1
        np.fill_diagonal(corr, 1.0)

        # Ensure positive semi-definite
        corr = self._nearest_psd(corr)

        return corr

    def _nearest_psd(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        B = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 1e-8)
        A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(A_psd, 1.0)
        return A_psd

    def get_network_stats(self) -> NetworkStats:
        """Compute summary statistics for the network."""
        n_edges = int(self.adj_binary.sum() / 2)
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
        """Count connected components."""
        visited = set()
        components = 0

        def dfs(node):
            visited.add(node)
            for neighbor in np.where(self.adj_binary[node] > 0)[0]:
                if neighbor not in visited:
                    dfs(neighbor)

        for node in range(self.n_nodes):
            if node not in visited:
                dfs(node)
                components += 1

        return components

    def detect_communities(self, n_communities: int = 5) -> np.ndarray:
        """Detect communities using spectral clustering."""
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
            _, labels = kmeans2(features, n_communities, minit='++')
        except Exception:
            labels = np.zeros(self.n_nodes, dtype=int)

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
