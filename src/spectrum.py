"""
Spectrum Shrinkage (Random-Matrix Denoising)  (src/spectrum.py)
===============================================================

PURPOSE
-------
Denoise a sample covariance / correlation matrix by **Marčenko-Pastur (MP)
spectrum shrinkage**. An empirical covariance estimated from a finite sample
has many small eigenvalues that are pure noise — their distribution is
described, in the high-dimensional limit, by the MP law. This routine finds the
number of "signal" eigenvalues ``k_bar`` that lie above the noise bulk, then
replaces every eigenvalue below the cut with their common average (an isotropic
noise floor) and rebuilds the matrix.

Effect: the top ``k_bar`` principal directions are kept intact; the noisy bulk
is flattened. The result is better conditioned and much more stable
out-of-sample — exactly what the copula correlation matrix and any downstream
PSD operation want.

This is a self-contained port of arpym ``estimation.spectrum_shrink``. The only
substantive change: arpym pulls the MP density from the ``skrmt`` package; here
the MP density is implemented in closed form with numpy, so the project keeps
its numpy / scipy / scikit-learn-only footprint.

WHERE IT PLUGS IN
-----------------
- ``graph_features.TransactionGraph.get_correlation_matrix(denoise=True)``
  applies it to the network-derived correlation matrix before the PSD
  projection. The shrunk matrix is then handed to the copula as usual.
- It can be applied to any covariance/correlation estimate you produce.

MATH
----
For ``q = t_bar / n`` (samples per variable) and noise variance ``sigma2``, the
MP density on ``[lam_minus, lam_plus]`` is
    lam_pm = sigma2 * (1 ± sqrt(1/q))**2
    f(lam) = sqrt((lam_plus - lam)*(lam - lam_minus)) / (2*pi*sigma2*lam/q)
``k_bar`` is chosen to minimise the MSE between the empirical eigenvalue
histogram (of the candidate noise tail) and the fitted MP density.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import interpolate, linalg


class ShrinkResult(NamedTuple):
    """Output bundle of :func:`spectrum_shrink`."""
    sigma2_out: np.ndarray   # (n, n) denoised matrix
    lam_out: np.ndarray      # (n,) eigenvalues after shrinkage (descending)
    k_bar: int               # number of signal eigenvalues retained
    err: float               # MSE of the best MP fit
    x_mp: np.ndarray         # (100,) MP density support
    y_mp: np.ndarray         # (100,) MP density values
    dist: np.ndarray         # (n-1,) MSE per candidate cut (NaN where invalid)


def marchenko_pastur_pdf(x: np.ndarray, q: float, sigma2: float) -> np.ndarray:
    """
    Marčenko-Pastur probability density (closed form), evaluated at ``x``.

    Parameters
    ----------
    x : array
        Points at which to evaluate the density.
    q : float
        Aspect ratio ``t_bar / n`` (observations per variable). Must be > 0.
    sigma2 : float
        Variance scale of the noise (mean of the noise eigenvalues).

    Returns
    -------
    array, same shape as ``x`` — density values (0 outside ``[lam_-, lam_+]``).
    """
    x = np.asarray(x, dtype=float)
    if q <= 0 or sigma2 <= 0:
        return np.zeros_like(x)

    lam_minus = sigma2 * (1.0 - np.sqrt(1.0 / q)) ** 2
    lam_plus = sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2

    pdf = np.zeros_like(x)
    inside = (x > lam_minus) & (x < lam_plus)
    xi = x[inside]
    pdf[inside] = np.sqrt((lam_plus - xi) * (xi - lam_minus)) / (2.0 * np.pi * sigma2 * xi / q)
    return pdf


def mp_support(q: float, sigma2: float) -> tuple[float, float]:
    """Return ``(lam_minus, lam_plus)``, the edges of the MP noise bulk."""
    lam_minus = sigma2 * (1.0 - np.sqrt(1.0 / q)) ** 2
    lam_plus = sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2
    return lam_minus, lam_plus


def _k_bar_mp_edge(lam: np.ndarray, t_bar: int) -> int:
    """
    Signal count = number of eigenvalues above the MP upper edge ``lam_plus``.

    This is the standard Laloux/Bouchaud RMT denoising rule. The noise variance
    is estimated self-consistently: assume the top ``k`` eigenvalues are signal,
    fit the MP edge to the remaining bulk, and count how many eigenvalues exceed
    that edge; iterate to a fixed point. Robust and parameter-free, unlike the
    histogram-MSE search which is noisy on small matrices.
    """
    i_bar = len(lam)
    k = 0
    for _ in range(i_bar):
        tail = lam[k:]
        if len(tail) < 2:
            break
        sigma2 = float(np.mean(tail))
        q = t_bar / len(tail)
        _, lam_plus = mp_support(q, sigma2)
        k_new = int(np.sum(lam > lam_plus))
        if k_new == k or k_new >= i_bar:
            k = k_new
            break
        k = k_new
    # Convert "count of signal eigenvalues" to the arpym cut index (last signal).
    return max(k - 1, 0)


def _finalise_shrink(
    sigma2_in: np.ndarray,
    lam: np.ndarray,
    e: np.ndarray,
    k_bar: int,
    err: float,
    t_bar: int,
    i_bar: int,
    dist: np.ndarray,
) -> ShrinkResult:
    """
    Flatten the noise bulk and rebuild the matrix from the shrunk spectrum.

    Shared by both selection methods. The eigenvalues with index ``> k_bar`` are
    replaced by their mean (isotropic noise floor); the matrix is reconstructed
    as ``E · diag(lam_out) · Eᵀ``. Also returns the fitted MP density for plots.
    """
    lam_out = lam.copy()
    if k_bar + 1 < i_bar:
        lam_noise = float(np.mean(lam[k_bar + 1:]))
        lam_out[k_bar + 1:] = lam_noise
    else:
        lam_noise = 0.0  # everything is signal — nothing to flatten

    sigma2_out = e @ np.diagflat(lam_out) @ e.T

    # MP aspect ratio q = observations / variables = t_bar / n_noise (matching
    # both mp_support's sqrt(1/q) edge formula and the hist_mse branch's
    # q = t_bar / len(lam_tail)). The number of noise variables is the count of
    # flattened eigenvalues, i_bar - (k_bar + 1).
    n_noise = i_bar - k_bar - 1
    if n_noise > 0 and lam_noise > 0:
        q_out = t_bar / n_noise
        lo, hi = mp_support(q_out, lam_noise)
        x_mp = np.linspace(lo, hi, 100)
        y_mp = marchenko_pastur_pdf(x_mp, q_out, lam_noise)
    else:
        x_mp = np.zeros(100)
        y_mp = np.zeros(100)

    return ShrinkResult(sigma2_out, lam_out, int(k_bar), float(err), x_mp, y_mp, dist)


def spectrum_shrink(
    sigma2_in: np.ndarray,
    t_bar: int,
    method: str = "mp_edge",
) -> ShrinkResult:
    """
    Marčenko-Pastur spectrum shrinkage of a covariance/correlation matrix.

    Parameters
    ----------
    sigma2_in : array, shape (n, n)
        Symmetric covariance or correlation matrix to denoise.
    t_bar : int
        Number of observations used to estimate ``sigma2_in`` (the effective
        sample size). Drives the MP aspect ratio.
    method : {"mp_edge", "hist_mse"}, default "mp_edge"
        How to choose the number of signal eigenvalues:

        * ``"mp_edge"`` — keep eigenvalues above the Marčenko-Pastur upper edge
          (standard RMT rule, robust, parameter-free). Recommended.
        * ``"hist_mse"`` — the original arpym criterion: pick the cut minimising
          the MSE between the noise-tail histogram and the fitted MP density.
          Kept for exact parity with ``arpym.estimation.spectrum_shrink``; it can
          be unstable on small matrices.

    Returns
    -------
    ShrinkResult
        Named tuple with the denoised matrix and diagnostics. See
        :class:`ShrinkResult`.

    Notes
    -----
    If the matrix is tiny (``n < 3``) there is no meaningful bulk to shrink and
    the input is returned unchanged with ``k_bar = n``.
    """
    if method not in ("mp_edge", "hist_mse"):
        raise ValueError(f"method must be 'mp_edge' or 'hist_mse', got {method!r}")

    sigma2_in = np.asarray(sigma2_in, dtype=float)
    i_bar = sigma2_in.shape[0]

    # Eigendecomposition (ascending), then reverse to descending.
    lam, e = linalg.eigh(sigma2_in)
    lam = lam[::-1].copy()
    e = e[:, ::-1].copy()

    # Sign convention: largest-magnitude entry of each eigenvector is positive.
    ind = np.argmax(np.abs(e), axis=0)
    flip = np.diag(e[ind, :]) < 0
    e[:, flip] = -e[:, flip]

    # Degenerate small matrices: nothing to shrink.
    if i_bar < 3:
        return ShrinkResult(
            sigma2_out=sigma2_in.copy(), lam_out=lam, k_bar=i_bar, err=0.0,
            x_mp=np.zeros(100), y_mp=np.zeros(100), dist=np.full(max(i_bar - 1, 0), np.nan),
        )

    # ── selection of the signal/noise cut k_bar ───────────────────────────────
    if method == "mp_edge":
        k_bar = _k_bar_mp_edge(lam, t_bar)
        err = 0.0
        dist = np.full(i_bar - 1, np.nan)
        return _finalise_shrink(sigma2_in, lam, e, k_bar, err, t_bar, i_bar, dist)

    # method == "hist_mse": original arpym histogram-MSE search.
    dist = np.full(i_bar - 1, np.nan)
    for k in range(i_bar - 1):
        lam_tail = lam[k + 1:]
        if len(lam_tail) < 2:
            continue
        lam_noise = np.mean(lam_tail)
        if lam_noise <= 0:
            continue
        q = t_bar / len(lam_tail)

        # Dense MP grid for interpolation.
        lo, hi = mp_support(q, lam_noise)
        x_grid = np.linspace(lo, hi, 1000)
        mp_grid = marchenko_pastur_pdf(x_grid, q, lam_noise)
        # Extend grid so interpolation covers the histogram range.
        if q > 1:
            x_grid = np.r_[0.0, x_grid[0], x_grid]
            mp_grid = np.r_[0.0, mp_grid[0], mp_grid]
        l_max = np.max(lam_tail)
        if l_max > x_grid[-1]:
            x_grid = np.r_[x_grid, x_grid[-1], l_max]
            mp_grid = np.r_[mp_grid, 0.0, 0.0]

        # Empirical histogram of the candidate noise eigenvalues.
        hgram, edges = np.histogram(lam_tail, bins="auto", density=True)
        if len(edges) < 2:
            continue
        bin_size = np.diff(edges)[0]
        x_bin = edges[:-1] + bin_size / 2.0

        interp = interpolate.interp1d(x_grid, mp_grid, fill_value="extrapolate")
        mp_at_bins = interp(x_bin)
        dist[k] = np.mean((mp_at_bins - hgram) ** 2)

    if np.all(np.isnan(dist)):
        # No valid fit (e.g. degenerate spectrum) — return input unchanged.
        return ShrinkResult(
            sigma2_out=sigma2_in.copy(), lam_out=lam, k_bar=i_bar, err=0.0,
            x_mp=np.zeros(100), y_mp=np.zeros(100), dist=dist,
        )

    k_bar = int(np.nanargmin(dist))
    err = float(np.nanmin(dist))
    return _finalise_shrink(sigma2_in, lam, e, k_bar, err, t_bar, i_bar, dist)


def denoise_correlation(corr_in: np.ndarray, t_bar: int, method: str = "mp_edge") -> np.ndarray:
    """
    Convenience wrapper: MP-shrink a CORRELATION matrix and re-impose unit diag.

    Spectrum shrinkage operates on the full matrix; for a correlation matrix the
    diagonal can drift slightly off 1 after reconstruction. This rescales the
    output back to a proper correlation matrix (unit diagonal) and symmetrises.

    Parameters
    ----------
    corr_in : array, shape (n, n) — correlation matrix to denoise.
    t_bar : int — effective sample size.
    method : {"mp_edge", "hist_mse"}, default "mp_edge"
        Signal/noise cut selection rule, forwarded to :func:`spectrum_shrink`.

    Returns
    -------
    array, shape (n, n) — denoised correlation matrix, symmetric, unit diagonal.
    """
    res = spectrum_shrink(corr_in, t_bar, method=method)
    out = res.sigma2_out
    d = np.sqrt(np.clip(np.diag(out), 1e-12, None))
    out = out / np.outer(d, d)
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 1.0)
    return out
