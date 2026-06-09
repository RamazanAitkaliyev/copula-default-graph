"""
Stage 25 — Factor-loading estimation  (owner: Data Scientist · copula)
=====================================================================

Optional stage between graph (20) and copula (30). Fits an ``(n, k)`` factor
loading matrix from the transaction-graph correlation using the ARPM low-rank
diagonal estimator (``low_rank_corr.fit_factor_loadings``) — turning the factor
copula's loadings from a hand-set assumption into a data-fitted quantity.

The loadings are written as an artifact so a multi-factor copula can consume
them; this stage does not itself replace stage 30 (which fits the dense Clayton
copula by default). To use the fitted loadings, build a ``MultiFactorCopula``
with ``betas = store.read_array("factor_loadings")``.

Inputs (artifacts):  corr_matrix
Outputs (artifacts): factor_loadings (npy), loading_diagnostics (json)
"""

from __future__ import annotations

import numpy as np

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "25_loadings"


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    k_factors: int = 2,
) -> StageResult:
    """
    Fit factor loadings from the correlation matrix.

    Parameters
    ----------
    k_factors : int
        Number of systematic factors to fit.
    """
    store.require("corr_matrix")
    corr = store.read_array("corr_matrix")
    res = StageResult(stage=STAGE, ok=True)

    from src.low_rank_corr import fit_factor_loadings
    beta = fit_factor_loadings(corr, k_factors=k_factors)
    res.outputs.append(store.write_array("factor_loadings", beta))

    row_sumsq = np.sum(beta ** 2, axis=1)
    # Reconstruction fidelity vs the input correlation (off-diagonal Frobenius).
    recon = beta @ beta.T
    np.fill_diagonal(recon, 1.0)
    n = corr.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    frob = float(np.linalg.norm((recon - corr)[offdiag]))

    res.outputs.append(store.write_json("loading_diagnostics", {
        "k_factors": k_factors,
        "n_borrowers": int(beta.shape[0]),
        "avg_loading": float(np.mean(beta)),
        "max_row_sumsq": float(np.max(row_sumsq)),     # < 1 ⇒ copula-ready
        "n_loaded_borrowers": int(np.sum(row_sumsq > 1e-9)),
        "offdiag_frobenius_vs_corr": frob,
    }))
    res.metrics["k_factors"] = k_factors
    res.metrics["max_row_sumsq"] = round(float(np.max(row_sumsq)), 4)
    res.metrics["copula_ready"] = bool(np.max(row_sumsq) < 1.0)
    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s).summary())
