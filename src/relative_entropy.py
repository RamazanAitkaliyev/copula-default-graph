"""
Relative-Entropy Minimisation  (src/relative_entropy.py)
========================================================

PURPOSE
-------
Minimum-relative-entropy ("entropy pooling", Meucci) posterior probabilities:
given a prior distribution ``p_pri`` over scenarios and a set of linear views
(equality / inequality constraints on the expectation of view variables),
find the posterior ``p`` that is closest to the prior in Kullback-Leibler
divergence while satisfying the views.

    minimise   sum_j  p_j * log(p_j / p_pri_j)
    s.t.       sum_j p_j = 1,   p_j >= 0          (always enforced)
               Z_eq   @ p == mu_view_eq           (equality views)
               Z_ineq @ p <= mu_view_ineq         (inequality views)

This is a **self-contained reimplementation** of arpym
``views.min_rel_entropy_sp`` — same math, but the constraint-normalisation
step uses numpy instead of ``statsmodels.DescrStatsW`` so that the project's
dependency footprint stays at numpy / pandas / scipy / scikit-learn only.

WHO USES THIS
-------------
- ``src/credit_transitions.py`` — imposes the monotonicity constraint on the
  rows of a credit transition matrix (default probability must not decrease as
  credit quality worsens). This is the concrete consumer today.
- It is also a general scenario-reweighting engine: stress overlays and macro
  views can be expressed as linear constraints and applied to any set of
  scenario probabilities (see arpym Tier-1 concept #3, entropy pooling).

MATH NOTES
----------
The optimum lies in the exponential family
    p(theta) ∝ p_pri * exp(theta @ Z),
where ``theta`` are Lagrange multipliers found by minimising the (smooth,
convex) dual Lagrangian. Equality-only problems use a Newton trust-region
solver; problems with inequalities use SLSQP with the multipliers for the
inequality block constrained to be non-negative.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import eye as sparse_eye


def _weighted_mean_std(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row mean and (population) standard deviation of the view matrix.

    ``z`` has shape (k_bar, j_bar): one row per view variable, one column per
    scenario. Returns ``(mean, std)`` each of shape (k_bar,). A zero std (a
    constant view variable) is floored to 1.0 so normalisation never divides by
    zero — the centred values are zero in that case anyway.
    """
    m_z = z.mean(axis=1)
    s_z = z.std(axis=1)  # population std (ddof=0), matching arpym's covariance diag
    s_z = np.where(s_z <= 0, 1.0, s_z)
    return m_z, s_z


def min_rel_entropy_sp(
    p_pri: np.ndarray,
    z_ineq: Optional[np.ndarray] = None,
    mu_view_ineq: Optional[np.ndarray] = None,
    z_eq: Optional[np.ndarray] = None,
    mu_view_eq: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Minimise relative entropy to ``p_pri`` subject to linear views.

    The constraints ``p_j >= 0`` and ``sum_j p_j = 1`` are always enforced
    implicitly by the exponential-family parameterisation.

    Parameters
    ----------
    p_pri : array, shape (j_bar,)
        Prior probabilities (need not be normalised; they are used as a base
        measure). Must be strictly positive where you want posterior support.
    z_ineq : array, shape (l_bar, j_bar), optional
        View matrix for inequality constraints ``z_ineq @ p <= mu_view_ineq``.
    mu_view_ineq : array, shape (l_bar,), optional
        Right-hand side of the inequality views.
    z_eq : array, shape (m_bar, j_bar), optional
        View matrix for equality constraints ``z_eq @ p == mu_view_eq``.
    mu_view_eq : array, shape (m_bar,), optional
        Right-hand side of the equality views.
    normalize : bool, default True
        Standardise each view variable (subtract mean, divide by std) before
        solving. Improves conditioning of the dual problem; the posterior is
        unchanged in exact arithmetic.

    Returns
    -------
    p_bar : array, shape (j_bar,)
        Posterior probabilities, summing to 1.
    """
    p_pri = np.asarray(p_pri, dtype=float)

    # No views -> posterior equals (normalised) prior.
    if z_ineq is None and z_eq is None:
        total = p_pri.sum()
        return p_pri / total if total > 0 else p_pri

    if z_ineq is None:
        z = np.atleast_2d(z_eq)
        mu_view = np.atleast_1d(mu_view_eq).astype(float)
        l_bar = 0
        m_bar = len(mu_view)
    elif z_eq is None:
        z = np.atleast_2d(z_ineq)
        mu_view = np.atleast_1d(mu_view_ineq).astype(float)
        l_bar = len(mu_view)
        m_bar = 0
    else:
        z = np.concatenate((np.atleast_2d(z_ineq), np.atleast_2d(z_eq)), axis=0)
        mu_view = np.concatenate(
            (np.atleast_1d(mu_view_ineq), np.atleast_1d(mu_view_eq))
        ).astype(float)
        l_bar = len(np.atleast_1d(mu_view_ineq))
        m_bar = len(np.atleast_1d(mu_view_eq))

    z = z.astype(float)

    # Standardise constraints for numerical conditioning.
    if normalize:
        m_z, s_z = _weighted_mean_std(z)
        z = (z - m_z[:, None]) / s_z[:, None]
        mu_view = (mu_view - m_z) / s_z

    log_p_pri = np.log(p_pri)

    def exp_family(theta: np.ndarray) -> np.ndarray:
        x = theta @ z + log_p_pri
        phi = logsumexp(x)
        p = np.exp(x - phi)
        p[p < 1e-32] = 1e-32
        return p / p.sum()

    def lagrangian(theta: np.ndarray) -> float:
        x = theta @ z + log_p_pri
        return float(logsumexp(x) - theta @ mu_view)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return z @ exp_family(theta) - mu_view

    def hessian(theta: np.ndarray) -> np.ndarray:
        p = exp_family(theta)
        z_bar = z.T - z @ p
        return (z_bar.T * p) @ z_bar

    k_bar = l_bar + m_bar
    theta0 = np.zeros(k_bar)

    if l_bar == 0:
        # Equality-only: smooth unconstrained dual -> Newton trust-region.
        res = minimize(
            lagrangian, theta0, method="trust-ncg",
            jac=gradient, hess=hessian, options={"gtol": 1e-10},
        )
    else:
        # Inequality block multipliers must be non-negative -> SLSQP.
        alpha = -sparse_eye(l_bar, k_bar)
        constraints = {"type": "ineq", "fun": lambda theta: alpha @ theta}
        res = minimize(
            lagrangian, theta0, method="SLSQP",
            jac=gradient, constraints=constraints,
            options={"ftol": 1e-10, "disp": False, "maxiter": 1000},
        )

    # Robustness: a degenerate view (e.g. a near-constant crisp window in
    # conditional_fp) can make the dual ill-conditioned and leave the solver
    # short of convergence with a non-finite / unusable theta. Rather than
    # silently returning a posterior that does not satisfy the views, fall back
    # to the (normalised) prior — the least-committal valid distribution.
    theta_hat = np.asarray(res.x, dtype=float)
    if not res.success and not np.all(np.isfinite(theta_hat)):
        total = p_pri.sum()
        return p_pri / total if total > 0 else p_pri

    # np.atleast_1d (not np.squeeze) so a single-scenario prior (j_bar == 1)
    # returns a 1-D length-1 vector, never a 0-D array that breaks p.sum().
    return np.atleast_1d(np.squeeze(exp_family(theta_hat)))
