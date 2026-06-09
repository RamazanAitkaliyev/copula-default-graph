"""
Low-Rank Diagonal Conditional Correlation  (src/low_rank_corr.py)
=================================================================

Approximate a correlation matrix by a **low-rank + diagonal** structure

    C ≈ β βᵀ + diag(1 − rowSumSq(β))           (unit diagonal preserved)

where ``β`` has shape ``(n, k)``: each borrower gets a ``k``-dimensional vector
of **factor loadings**. This is the structural form of a ``k``-factor copula —
``k`` systematic factors plus an idiosyncratic term — and the row-norm
constraint ``Σ_k β_ik² < 1`` is exactly the positive-idiosyncratic-variance
condition the platform's factor copulas require.

Self-contained port of arpym ``estimation.low_rank_diag_conditional_corr``
(+ ``conditional_pc``). numpy / scipy only — no arpym dependency.

WHY THIS MATTERS HERE
---------------------
``MultiFactorCopula.fit`` takes ``betas`` of shape ``(n, K)`` but today you SET
them by hand (equal loadings = "equally important"). This estimator **fits**
them from a target correlation matrix (e.g. the transaction-graph correlation),
so the factor copula's structure comes from data instead of assumption. Use
``fit_factor_loadings`` to go matrix → loadings, then feed them straight to
``MultiFactorCopula``.

Optionally a linear constraint ``D β = 0`` can be imposed (``conditional`` mode):
loadings are forced orthogonal to the rows of ``D`` — e.g. to remove a known
direction (a market-wide mode) from the systematic factors.

MATH
----
Alternating projection:
  1. eigendecompose the current ``β βᵀ`` (or its conditional version under D),
  2. keep the top ``k`` directions → new ``β``,
  3. scale any row with ‖β_i‖ > 1 back below 1 (keeps idiosyncratic var > 0),
  4. rebuild ``C = β βᵀ + I − diag(β βᵀ)`` (unit diagonal),
  5. repeat until the loadings stop moving.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from numpy.linalg import eig, matrix_rank, solve, svd


class LowRankResult(NamedTuple):
    """Output of :func:`low_rank_diag_conditional_corr`."""
    c2_lrd: np.ndarray   # (n, n) low-rank+diagonal correlation approximation
    beta: np.ndarray     # (n, k) fitted factor loadings
    distance: float      # final row-wise loading change at convergence
    n_iter: int          # iterations used
    constraint_ok: bool  # True if the D@beta = 0 constraint was satisfied


def conditional_pc(sigma2: np.ndarray, d: np.ndarray):
    """
    Conditional principal directions / variances of a symmetric matrix.

    Returns the principal directions of ``sigma2`` restricted to the subspace
    orthogonal to the rows of ``d`` (the "conditional" PCs). Port of arpym
    ``conditional_pc``.

    Parameters
    ----------
    sigma2 : array, shape (n, n) — symmetric matrix.
    d : array, shape (k, n) — constraint rows; PCs are made orthogonal to these.

    Returns
    -------
    lam2_d : array, shape (n, 1) — conditional variances (leading eigenvalues).
    e_d : array, shape (n, n) — conditional principal directions (columns).
    """
    n_bar = sigma2.shape[0]
    lam2_d = np.empty((n_bar, 1))
    e_d = np.empty((n_bar, n_bar))

    a_n = d
    for n in range(n_bar):
        # Orthogonal projector onto the complement of span(rows of a_n).
        p = np.eye(n_bar) - a_n.T @ solve(a_n @ a_n.T, a_n)
        s2 = p @ sigma2 @ p
        # Leading principal direction of the projected matrix (via SVD for
        # numerical stability + a deterministic sign convention).
        w, _ = eig(s2)
        w = np.real(w)
        order = np.argsort(-w)
        eigval = w[order]
        _, _, vt = svd(s2)
        eigvec = vt.T
        cols = eigvec.shape[1]
        signs = np.sign(eigvec[np.argmax(np.abs(eigvec), axis=0), np.arange(cols)])
        eigvec = eigvec * signs[np.newaxis, :]
        e_d[:, n] = np.real(eigvec[:, 0])
        lam2_d[n] = eigval[0]

    return lam2_d, e_d


def low_rank_diag_conditional_corr(
    c2: np.ndarray,
    d: Optional[np.ndarray] = None,
    k_bar: int = 1,
    eta: float = 0.01,
    gamma: float = 0.1,
    max_iter: int = 1000,
) -> LowRankResult:
    """
    Fit a low-rank + diagonal approximation of a correlation matrix.

    Parameters
    ----------
    c2 : array, shape (n, n)
        Target correlation matrix (symmetric, ideally unit diagonal).
    d : array, shape (m, n), optional
        Linear constraint rows; the fitted loadings are made orthogonal to them
        (``d @ beta = 0``). If None (or all-zero), the unconstrained low-rank fit
        is computed.
    k_bar : int, default 1
        Number of systematic factors (rank of ``beta``). Must satisfy
        ``k_bar <= n - rank(d)``.
    eta : float, default 0.01
        Convergence tolerance on the mean row-wise loading change.
    gamma : float, default 0.1
        Safety margin when scaling over-length loading rows below unit norm.
    max_iter : int, default 1000
        Maximum alternating-projection iterations.

    Returns
    -------
    LowRankResult
        ``c2_lrd`` (approximation), ``beta`` (n×k loadings), ``distance``,
        ``n_iter``, ``constraint_ok``.
    """
    c2 = np.asarray(c2, dtype=float)
    n_bar = c2.shape[0]

    if d is None:
        d = np.zeros((1, n_bar))
    d = np.atleast_2d(np.asarray(d, dtype=float))
    conditional = 1 if np.sum(np.abs(d.flatten())) != 0 else 0

    if k_bar > n_bar - matrix_rank(d):
        raise ValueError("k_bar must be <= n - rank(d)")

    eps1 = 1e-9
    constraint = 0
    c2_lrd = c2.copy()
    distance = 0.0
    n_iter = 0

    # Initialise beta from the top-k eigenpairs of c2.
    lam2, e = eig(c2)
    lam2 = np.real(lam2)
    order = np.argsort(lam2)[::-1]
    lam = np.real(np.sqrt(np.maximum(lam2[order][:k_bar], 0.0)))
    e_ord = np.real(e[:, order])
    beta = e_ord[:, :k_bar] @ np.diagflat(np.maximum(lam, eps1))
    c = c2.copy()

    for j in range(max_iter):
        # Systematic part with the current diagonal restored.
        a = c - np.eye(n_bar) + np.diagflat(np.diag(beta @ beta.T))

        if conditional == 1:
            lam2_c, e_c = conditional_pc(a, d)
            lam2_k = lam2_c[:k_bar].ravel()
            e_k = e_c[:, :k_bar]
            lam = np.sqrt(np.maximum(lam2_k, 0.0))
        else:
            lam2_a, e_a = eig(a)
            lam2_a = np.real(lam2_a)
            order = np.argsort(lam2_a)[::-1]
            e_k = np.real(e_a[:, order][:, :k_bar])
            lam = np.sqrt(np.maximum(lam2_a[order][:k_bar], 0.0))

        beta_new = e_k @ np.diagflat(np.maximum(lam, eps1))

        # Scale rows whose loading norm exceeds 1 back below 1 (keeps the
        # idiosyncratic variance 1 - ‖β_i‖² strictly positive).
        row_norm = np.sqrt(np.sum(beta_new ** 2, axis=1))
        over = row_norm > 1
        if np.any(over):
            beta_new[over, :] = beta_new[over, :] / (row_norm[over, None] * (1 + gamma))

        # Reconstruct with unit diagonal.
        c = beta_new @ beta_new.T + np.eye(n_bar) - np.diag(np.diag(beta_new @ beta_new.T))

        distance = (1.0 / n_bar) * np.sum(np.sqrt(np.sum((beta_new - beta) ** 2, axis=1)))
        beta = np.real(beta_new)

        if distance <= eta:
            c2_lrd = np.real(c)
            c2_lrd = (c2_lrd + c2_lrd.T) / 2.0
            n_iter = j
            tol = np.max(np.abs(d @ beta))
            if tol < 1e-9:
                constraint = 1
            break
        else:
            c2_lrd = np.real(c)
            c2_lrd = (c2_lrd + c2_lrd.T) / 2.0
            n_iter = j

    return LowRankResult(c2_lrd, beta, float(distance), int(n_iter), bool(constraint))


def fit_factor_loadings(
    corr: np.ndarray,
    k_factors: int = 1,
    constraint: Optional[np.ndarray] = None,
    non_negative: bool = True,
    max_row_norm: float = 0.999,
    **kwargs,
) -> np.ndarray:
    """
    Convenience: fit ``(n, k)`` factor loadings from a correlation matrix.

    Thin wrapper returning just the ``beta`` loading matrix — the exact input
    ``MultiFactorCopula.fit(marginal_pds, factor_matrix, betas=beta)`` wants.
    Row norms are capped STRICTLY below 1 (``max_row_norm``) so that
    ``Σ_k β_ik² < 1`` always holds (positive idiosyncratic variance) — the
    loadings are copula-ready even when the input correlation has unit-norm
    directions (e.g. a near-identity matrix, where the raw fit can land a row
    exactly at norm 1 and make ``MultiFactorCopula`` reject it).

    Parameters
    ----------
    corr : array, shape (n, n) — target correlation matrix.
    k_factors : int — number of systematic factors.
    constraint : array, shape (m, n), optional — ``D`` for ``D @ beta = 0``.
    non_negative : bool, default True
        Return the absolute value of the loadings. The low-rank fit yields
        SIGNED loadings (eigenvector directions), but ``MultiFactorCopula`` (a
        Vasicek model) requires non-negative loadings — the asset-correlation
        loading ``√ρ`` is non-negative by construction. Since that copula uses
        ``β_i·β_j`` only for borrowers sharing a factor, taking magnitudes keeps
        the implied within-factor correlation while satisfying the copula's
        sign convention. Set False to keep the raw signed loadings (e.g. if you
        want the full ``β βᵀ`` correlation reconstruction with cross-sign terms).
    max_row_norm : float, default 0.999
        Upper bound on each loading row's Euclidean norm. Rows exceeding it are
        scaled down to this value, guaranteeing ``Σ_k β_ik² <= max_row_norm² < 1``.

    Returns
    -------
    beta : array, shape (n, k_factors) — fitted loadings, non-negative by default
        and with every row norm < 1 (copula-ready).
    """
    result = low_rank_diag_conditional_corr(corr, d=constraint, k_bar=k_factors, **kwargs)
    beta = np.abs(result.beta) if non_negative else result.beta

    # Cap row norms strictly below 1 so idiosyncratic variance stays positive.
    row_norm = np.sqrt(np.sum(beta ** 2, axis=1))
    over = row_norm > max_row_norm
    if np.any(over):
        beta = beta.copy()
        beta[over, :] = beta[over, :] * (max_row_norm / row_norm[over, None])
    return beta
