"""
src.analytics — Data Scientist role view: graphs & clusters.

A NAVIGATION facade re-exporting the flat modules a Data Scientist owns on the
graph/cluster side (`from src.analytics import TransferClusterer`).
Implementation stays in `src/*.py`. See ROLES.md.

Owns: the transaction graph, geo + transfer clusters, and the anchor/dependent
pattern (якорный человек). The copula models live in `src.copula` (same role).
"""
from ..graph_features import TransactionGraph, get_neighbor_risk_features
from ..geo_clusters import GeoClusterer, GeoClusterConfig
from ..transfer_clusters import TransferClusterer, TransferClusterConfig

__all__ = [
    "TransactionGraph", "get_neighbor_risk_features",
    "GeoClusterer", "GeoClusterConfig",
    "TransferClusterer", "TransferClusterConfig",
]
