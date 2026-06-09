"""
Stage 20 — Graph features & correlation matrix  (owner: Data Scientist · graph)
===============================================================================

Build the transaction graph and derive the borrower-borrower correlation
matrix that the copula consumes. This stage hosts arpym Tier-1 concept #2:
optional **Marčenko-Pastur spectrum shrinkage** (random-matrix denoising) of the
correlation matrix, exposed via ``denoise=True``.

Inputs (artifacts):  persons_scored, transactions
Outputs (artifacts): corr_matrix (npy), network_stats (json),
                     and spectrum_diagnostics (json) when denoising is on
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "20_graph"


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    base_corr: float = 0.05,
    max_corr: float = 0.6,
    same_city_boost: float = 0.1,
    same_group_boost: float = 0.2,
    denoise: bool = False,
    denoise_t_bar: Optional[int] = None,
    denoise_method: str = "mp_edge",
) -> StageResult:
    """
    Derive the correlation matrix from the transaction network.

    Parameters
    ----------
    base_corr, max_corr, same_city_boost, same_group_boost : float
        Correlation construction knobs (see
        ``graph_features.get_correlation_matrix``).
    denoise : bool, default False
        Apply Marčenko-Pastur spectrum shrinkage before the PSD projection
        (arpym concept #2). Improves conditioning and out-of-sample stability.
    denoise_t_bar : int, optional
        Effective sample size for the MP aspect ratio (heuristic if omitted).
    denoise_method : {"mp_edge", "hist_mse"}
        Signal/noise selection rule.
    """
    store.require("persons_scored", "transactions")
    persons = store.read_df("persons_scored")
    transactions = store.read_df("transactions")
    res = StageResult(stage=STAGE, ok=True)

    from src.graph_features import TransactionGraph
    graph = TransactionGraph(transactions, persons)

    stats = graph.get_network_stats()
    res.outputs.append(store.write_json("network_stats", {
        "n_nodes": stats.n_nodes, "n_edges": stats.n_edges,
        "density": stats.density, "avg_degree": stats.avg_degree,
        "avg_clustering": stats.avg_clustering, "n_components": stats.n_components,
    }))

    # Correlation matrix, with optional random-matrix denoising.
    corr = graph.get_correlation_matrix(
        base_corr=base_corr, max_corr=max_corr,
        same_city_boost=same_city_boost, same_group_boost=same_group_boost,
        denoise=denoise, denoise_t_bar=denoise_t_bar, denoise_method=denoise_method,
    )
    res.outputs.append(store.write_array("corr_matrix", corr))

    # Conditioning diagnostics — and a side-by-side when denoising is on.
    res.metrics["denoised"] = denoise
    res.metrics["cond_number"] = round(float(np.linalg.cond(corr)), 1)
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)]
    res.metrics["avg_offdiag_corr"] = round(float(off.mean()), 4)

    if denoise:
        # Record how much the denoiser changed conditioning vs the raw matrix.
        corr_raw = graph.get_correlation_matrix(
            base_corr=base_corr, max_corr=max_corr,
            same_city_boost=same_city_boost, same_group_boost=same_group_boost,
            denoise=False,
        )
        res.outputs.append(store.write_json("spectrum_diagnostics", {
            "method": denoise_method,
            "cond_raw": float(np.linalg.cond(corr_raw)),
            "cond_denoised": float(np.linalg.cond(corr)),
            "frobenius_change": float(np.linalg.norm(corr - corr_raw)),
        }))

    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s, denoise=True).summary())
