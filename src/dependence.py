"""
Copula Dependence Measures  (src/dependence.py)
===============================================

Rank-based dependence measures that look at the *copula* (the dependence
structure) rather than linear correlation — so they capture non-linear and
tail dependence that Pearson correlation misses.

Self-contained ports of arpym ``statistics.schweizer_wolff`` and the
measurement core of ``statistics.invariance_test_copula`` (the plotting is
dropped to keep the project's numpy/scipy/pandas-only footprint).

CORRECTNESS NOTE (intentional divergence from arpym)
----------------------------------------------------
This port FIXES a normalisation bug in arpym's ``schweizer_wolff``. The original
sums the empirical-vs-independence copula difference over grid indices
``i, k = 0 .. j-1`` with thresholds ``i/j`` — which leaves the empirical copula
and the independence copula ``(i·k)/j²`` evaluated on misaligned grids. The
result is correct near perfect dependence but badly wrong near independence: on
i.i.d. data the original returns values well above 1.0 (e.g. ≈1.12), even though
the Schweizer-Wolff measure is defined to lie in [0, 1] with σ=0 for
independence. This port sums over ``i, k = 1 .. g`` (thresholds in ``(0, 1]``),
which aligns the two copulas and reproduces the correct values:
σ≈0 for independent data and σ→1 for comonotone data (both verified in tests).

Functions
---------
- ``schweizer_wolff(x, p=None)`` — the Schweizer-Wolff measure σ ∈ [0, 1]:
  the (normalised) L1 distance between the empirical copula and independence.
  σ=0 ⇔ independent, σ=1 ⇔ perfect monotone dependence. Unlike Spearman/Kendall
  it is sensitive to *non-monotone* dependence too.
- ``copula_invariance_test(eps, lag_bar, ...)`` — Schweizer-Wolff dependence
  between a series and its own lag, for lags 1..lag_bar. For an i.i.d. invariant
  the dependence should be ≈ 0 at every lag; a bar that stands out flags
  residual serial structure (a failed invariance assumption).

SCALE NOTE
----------
The exact Schweizer-Wolff measure builds a ``j×j`` empirical-copula grid, so it
is O(j²) memory / O(j² log j) time. For large samples pass ``max_grid`` (default
2000): the grades are evaluated on a coarser ``g×g`` grid (g = min(j, max_grid))
which is the standard practical approximation and keeps memory bounded. The
vectorised implementation avoids the original's Python triple loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _grades(x: np.ndarray, p: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Copula grades (marginal CDF values) of a 2-column scenario matrix.

    Returns ``u`` of shape (j, 2) with each column the probability-integral
    transform of the corresponding marginal, clipped off {0, 1}.
    """
    x = np.asarray(x, dtype=float)
    j_bar, n_bar = x.shape
    if p is None:
        p = np.ones(j_bar) / j_bar
    p = np.asarray(p, dtype=float)

    u = np.zeros((j_bar, n_bar))
    for n in range(n_bar):
        order = np.argsort(x[:, n], kind="mergesort")
        # Cumulative probability up to and including each sorted point.
        cum = np.cumsum(p[order])
        # Assign the CDF value back to each original position.
        ranks_cdf = np.empty(j_bar)
        ranks_cdf[order] = cum
        u[:, n] = ranks_cdf
    eps = np.spacing(1)
    u[u >= 1] = 1 - eps
    u[u <= 0] = eps
    return u


def schweizer_wolff(
    x: np.ndarray,
    p: Optional[np.ndarray] = None,
    max_grid: int = 2000,
) -> float:
    """
    Schweizer-Wolff dependence measure of a bivariate sample.

    Parameters
    ----------
    x : array, shape (j, 2)
        Bivariate scenarios (two columns).
    p : array, shape (j,), optional
        Scenario probabilities (uniform by default).
    max_grid : int, default 2000
        Cap on the empirical-copula grid resolution (see SCALE NOTE). If the
        sample has more than ``max_grid`` points the copula is evaluated on a
        ``max_grid × max_grid`` grid (a standard approximation).

    Returns
    -------
    sw : float in [0, 1]
        0 ⇔ independence, 1 ⇔ perfect monotone dependence.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("schweizer_wolff expects x of shape (j, 2)")
    j_bar = x.shape[0]
    if j_bar < 2:
        return 0.0
    # A (near-)constant column has a degenerate copula — dependence is undefined.
    # Return 0 (no detectable dependence) rather than a discretisation artifact.
    if np.ptp(x[:, 0]) <= 1e-12 or np.ptp(x[:, 1]) <= 1e-12:
        return 0.0
    if p is None:
        p = np.ones(j_bar) / j_bar
    p = np.asarray(p, dtype=float)

    u = _grades(x, p)

    # Grid resolution: exact (g = j) for small samples, capped otherwise.
    g = min(j_bar, max_grid)
    thr = np.arange(1, g + 1) / g  # thresholds in (0, 1]

    # Indicator matrices: which scenarios fall below each grid threshold.
    # below0[s, i] = u[s,0] <= thr[i]  → (j, g)
    below0 = u[:, 0][:, None] <= thr[None, :]
    below1 = u[:, 1][:, None] <= thr[None, :]

    # Empirical copula on the grid: C[i,k] = Σ_s p_s · 1{u0<=thr_i} · 1{u1<=thr_k}
    # = (p·below0)ᵀ @ below1, weighting rows by probability.
    cdf_u = (below0 * p[:, None]).T @ below1  # (g, g)

    # Independence copula on the grid: (i/g)·(k/g) using 1-based indices.
    i_idx = np.arange(1, g + 1)
    indep = np.outer(i_idx, i_idx) / (g ** 2)

    sw = 12.0 / (g ** 2) * np.sum(np.abs(cdf_u - indep))
    return float(np.clip(sw, 0.0, 1.0))


def copula_invariance_test(
    eps: np.ndarray,
    lag_bar: int,
    max_grid: int = 2000,
) -> np.ndarray:
    """
    Schweizer-Wolff serial-dependence across lags (plot-free invariance test).

    Computes the SW dependence between ``eps`` and a copy of itself shifted by
    ``l`` observations, for ``l = 1 .. lag_bar``. For an i.i.d. invariant every
    value should be close to 0; a lag with materially higher dependence flags
    residual serial structure (the series is not an invariant).

    Parameters
    ----------
    eps : array, shape (t,)
        Candidate-invariant time series.
    lag_bar : int
        Maximum lag to test.
    max_grid : int
        Forwarded to ``schweizer_wolff`` (scale cap).

    Returns
    -------
    sw : array, shape (lag_bar,)
        SW dependence at each lag (index 0 = lag 1).
    """
    eps = np.asarray(eps, dtype=float).ravel()
    t_bar = eps.shape[0]
    if lag_bar < 1:
        raise ValueError("lag_bar must be >= 1")
    if lag_bar >= t_bar:
        raise ValueError("lag_bar must be < len(eps)")

    sw = np.zeros(lag_bar)
    for l in range(lag_bar):
        lag = l + 1
        paired = np.column_stack((eps[lag:], eps[:-lag]))
        sw[l] = schweizer_wolff(paired, max_grid=max_grid)
    return sw
