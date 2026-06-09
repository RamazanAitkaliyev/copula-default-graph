"""
Conditional Flexible Probabilities  (src/conditional_fp.py)
===========================================================

The **rigorous** Flexible-Probabilities (FP) family from arpym: given a state
(risk-driver) series ``z`` and a target state ``z_star`` ("today's regime"),
assign each historical scenario a probability that reflects how relevant it is
to that target — so estimators (correlation, copula θ, moments) become
*regime-conditional*.

Two estimators, ported self-contained (numpy / scipy + the project-local
``relative_entropy.min_rel_entropy_sp``; no arpym / statsmodels dependency):

* ``crisp_fp``        — **crisp** conditioning: a hard window of width ``alpha``
                        around ``z_star`` (probabilities are uniform inside the
                        window, zero outside). Port of arpym ``crisp_fp``.
* ``conditional_fp``  — **smooth** conditioning: start from the crisp window,
                        read off its conditional mean & variance, then find the
                        minimum-relative-entropy distribution over ALL scenarios
                        that matches those two moments. Port of arpym
                        ``conditional_fp``.

WHY THIS, NOT THE KERNEL IN flexible_probs.py
---------------------------------------------
``flexible_probs.gaussian_kernel_weights`` is a smooth *kernel approximation*:
weights fall off like a Gaussian in ``|z - z_star|``. It is simple and fast but
it does not pin the conditional moments. ``conditional_fp`` instead **matches the
conditional mean and variance exactly** (via entropy pooling), which is the arpym
definition and what model-validation expects. Both remain available; pick the
kernel for speed, this for rigour.

WHERE IT PLUGS IN
-----------------
``FlexibleProbsCalibrator.fit(..., method="conditional_fp")`` uses these to build
the regime-conditional probability vector that re-weights the correlation matrix
and copula θ. See ``src/flexible_probs.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .relative_entropy import min_rel_entropy_sp


def quantile_smooth(
    c_bar,
    x: np.ndarray,
    p: Optional[np.ndarray] = None,
    h: Optional[float] = None,
) -> np.ndarray:
    """
    Kernel-smoothed quantile(s) of a scenario set (arpym ``quantile_smooth``).

    Parameters
    ----------
    c_bar : scalar or array, shape (k_bar,)
        Confidence level(s) in [0, 1] at which to evaluate the quantile.
    x : array, shape (j_bar,)
        Scenarios.
    p : array, shape (j_bar,), optional
        Scenario probabilities (uniform by default).
    h : float, optional
        Kernel bandwidth (Silverman-like default if omitted).

    Returns
    -------
    array, shape (k_bar,) of smoothed quantiles.
    """
    c_bar = np.atleast_1d(c_bar).astype(float)
    j_bar = x.shape[0]
    k_bar = c_bar.shape[0]

    if p is None:
        p = np.ones(j_bar) / j_bar
    order = np.argsort(x)
    x_sort = np.asarray(x)[order]
    p_sort = np.asarray(p)[order]

    # Cumulative sums of sorted probabilities (length j_bar + 1, leading 0).
    u_sort = np.concatenate(([0.0], np.cumsum(p_sort)))

    if h is None:
        h = 0.25 * (j_bar ** (-0.2))

    q = np.zeros(k_bar)
    for k in range(k_bar):
        w = np.diff(stats.norm.cdf(u_sort, c_bar[k], h))
        total = w.sum()
        if total > 0:
            w = w / total
        q[k] = x_sort @ w
    return np.squeeze(q)


def crisp_fp(
    z: np.ndarray,
    z_star,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crisp flexible probabilities: a hard probability window around ``z_star``.

    For each target ``z_star[k]`` a window ``[z_lb, z_ub]`` is chosen so it
    encloses a probability mass of ``alpha``; scenarios inside the window get
    equal probability, scenarios outside get zero.

    Parameters
    ----------
    z : array, shape (t_bar,)
        State / risk-driver scenarios.
    z_star : scalar or array, shape (k_bar,)
        Target state(s) ("today's regime").
    alpha : float
        Fraction of scenarios to keep in the window (window mass), in (0, 1].

    Returns
    -------
    p : array, shape (t_bar,) if k_bar == 1 else (t_bar, k_bar)
        Crisp probabilities (columns sum to 1).
    z_lb, z_ub : array, shape (k_bar,)
        Lower / upper window edges.
    """
    z = np.asarray(z, dtype=float)
    z_star = np.atleast_1d(z_star).astype(float)
    t_bar = z.shape[0]
    k_bar = z_star.shape[0]

    p_uniform = np.ones(t_bar) / t_bar
    order = np.argsort(z)
    z_sort = z[order]
    u_sort = np.concatenate(([0.0], np.cumsum(p_uniform[order])))

    # Empirical CDF of z at each target (interpolated), with an extrapolated
    # left anchor so targets below the minimum are handled smoothly.
    if t_bar >= 3:
        z_0 = z_sort[0] - (z_sort[1] - z_sort[0]) * u_sort[1] / (u_sort[2] - u_sort[1])
    else:
        z_0 = z_sort[0] - 1e-6
    z_aug = np.concatenate(([z_0], z_sort))

    cdf_z_star = np.zeros(k_bar)
    for k in range(k_bar):
        cidx = int(np.searchsorted(z_aug, z_star[k], side="right"))
        if cidx == 0:
            cdf_z_star[k] = 0.0
        elif cidx >= t_bar + 1:
            cdf_z_star[k] = 1.0
        else:
            denom = (z_aug[cidx] - z_aug[cidx - 1]) or 1e-12
            cdf_z_star[k] = u_sort[cidx - 1] + (u_sort[cidx] - u_sort[cidx - 1]) * \
                (z_star[k] - z_aug[cidx - 1]) / denom

    z_lb = np.zeros(k_bar)
    z_ub = np.zeros(k_bar)
    pp = np.zeros((k_bar, t_bar))
    for k in range(k_bar):
        if z_star[k] <= quantile_smooth(alpha / 2, z):
            z_lb[k] = np.min(z)
            z_ub[k] = quantile_smooth(alpha, z)
        elif z_star[k] >= quantile_smooth(1 - alpha / 2, z):
            z_lb[k] = quantile_smooth(1 - alpha, z)
            z_ub[k] = np.max(z)
        else:
            z_lb[k] = quantile_smooth(cdf_z_star[k] - alpha / 2, z)
            z_ub[k] = quantile_smooth(cdf_z_star[k] + alpha / 2, z)

        in_window = (z <= z_ub[k]) & (z >= z_lb[k])
        n_in = in_window.sum()
        if n_in > 0:
            pp[k, in_window] = 1.0 / n_in
        else:
            # Window too narrow to enclose any scenario (very small alpha or a
            # tightly clustered z). Fall back to the single nearest scenario so
            # the result is a valid distribution rather than an all-zero vector
            # (which would silently degrade conditional_fp to the prior).
            nearest = int(np.argmin(np.abs(z - z_star[k])))
            pp[k, nearest] = 1.0
            z_lb[k] = z_ub[k] = z[nearest]

    return np.squeeze(pp.T), np.squeeze(z_lb), np.squeeze(z_ub)


def conditional_fp(
    z: np.ndarray,
    z_star,
    alpha: float,
    p_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Conditional (smooth) flexible probabilities via entropy-pooling moment match.

    Builds the crisp window, reads its conditional mean and variance, then finds
    the minimum-relative-entropy distribution over ALL scenarios (relative to
    ``p_prior``) that reproduces those two moments. The result is smooth (every
    scenario keeps some mass) yet exactly matches the conditional first and
    second moments — the arpym definition of conditional FP.

    Parameters
    ----------
    z : array, shape (t_bar,)
        State scenarios.
    z_star : scalar or array, shape (k_bar,)
        Target state(s).
    alpha : float
        Crisp-window mass (relevance bandwidth) in (0, 1].
    p_prior : array, shape (t_bar,), optional
        Prior probabilities (uniform by default).

    Returns
    -------
    p : array, shape (t_bar,) if k_bar == 1 else (t_bar, k_bar)
        Conditional flexible probabilities (columns sum to 1).
    """
    z = np.asarray(z, dtype=float)
    z_star = np.atleast_1d(z_star).astype(float)
    t_bar = z.shape[0]
    k_bar = z_star.shape[0]

    if p_prior is None:
        p_prior = np.ones(t_bar) / t_bar
    p_prior = np.asarray(p_prior, dtype=float)

    # Crisp probabilities per target (shape (k_bar, t_bar)).
    p_crisp = np.atleast_2d(crisp_fp(z, z_star, alpha)[0].T)
    if p_crisp.shape[0] != k_bar:  # squeeze edge case for k_bar == 1
        p_crisp = p_crisp.reshape(k_bar, t_bar)
    p_crisp[p_crisp == 0] = 1e-20

    p_out = np.zeros((k_bar, t_bar))
    for k in range(k_bar):
        pk = p_crisp[k] / p_crisp[k].sum()
        # Conditional moments under the crisp window.
        m_z = pk @ z
        s2_z = pk @ (z ** 2) - m_z ** 2
        # Entropy-pool to a smooth distribution matching mean & variance.
        a_ineq = np.atleast_2d(z ** 2)
        b_ineq = np.atleast_1d(m_z ** 2 + s2_z)
        a_eq = np.atleast_2d(z)
        b_eq = np.atleast_1d(m_z)
        p_out[k] = min_rel_entropy_sp(p_prior, a_ineq, b_ineq, a_eq, b_eq)

    return np.squeeze(p_out.T)


def effective_scenarios(p: np.ndarray) -> float:
    """
    Effective number of scenarios (perplexity) of a probability vector.

    ``ens = exp(-Σ p_i log p_i)``. A diffuse vector → ens ≈ len(p); a sharply
    concentrated one → ens ≈ 1. Useful to warn when a regime view is so tight
    that almost no historical scenarios remain informative.
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))
