"""
Credit Transition-Matrix Estimation  (src/credit_transitions.py)
================================================================

PURPOSE
-------
Estimate an annual credit-rating **transition matrix** from observed history,
the rigorous way: via the continuous-time **generator** matrix, with optional
exponential time-weighting (half-life) of older observations, and a final
**entropy-regularised monotonicity** correction so that the cumulative
default/downgrade probability behaves monotonically across rating quality.

This is a self-contained port of arpym ``estimation.fit_trans_matrix_credit``
(generator construction + ``min_rel_entropy_sp`` monotonicity step). It uses
only numpy / scipy and the project-local ``relative_entropy.min_rel_entropy_sp``
— no external arpym or ``statsmodels`` dependency.

WHY THIS, NOT THE SIMPLE ESTIMATOR IN rating_engine.py
------------------------------------------------------
``rating_engine.estimate_transition_matrix`` is a quick discrete-time MLE
(ratio of transition counts to obligor counts). It is fine for a baseline, but
it does NOT:

  1. estimate a proper continuous-time generator ``G`` (so projecting to other
     horizons via ``expm(G·Δt)`` is only approximate there), nor
  2. weight exposure by the actual time-at-risk (business-day count / 252)
     in each period, nor
  3. enforce that posterior rows respect a monotonicity ordering.

This module does all three. Both estimators are kept: use the simple one for a
fast default, this one for model-validation-grade estimates from real data.

INPUTS (cohort / duration format)
---------------------------------
``dates``   : array (t_bar,) of period boundary dates (``datetime64[D]`` or
              anything ``numpy.busday_count`` accepts). Length ``t_bar``.
``n_oblig`` : array (t_bar, c_bar) — number of obligors in each rating ``i`` at
              the start of each period ``t`` (the population at risk).
``n_cum``   : array (t_bar, c_bar, c_bar) — CUMULATIVE count of transitions
              observed from rating ``i`` to rating ``j`` up to and including
              period ``t``.
``tau_hl``  : optional half-life in YEARS for exponential decay of older
              observations. ``None`` => all history weighted equally.

OUTPUT
------
``p`` : array (c_bar, c_bar) — annual transition matrix. Rows sum to 1, the
        last state is absorbing (default), and rows satisfy the monotonicity
        constraint.

KEY MATH
--------
Off-diagonal generator entries (equal-weight case):
    g[i,j] = N_cum[i,j] / sum_t ( n_oblig[t,i] * Δτ_t ),   Δτ_t in years
Diagonal:
    g[i,i] = - sum_{j≠i} g[i,j]
Prior transition matrix:
    P_prior = expm(G)
Monotonicity step (per row, arpym):
    P[c,:] = argmin_p  KL(p || P_prior[c,:])  s.t. cumulative ordering holds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import expm

from .relative_entropy import min_rel_entropy_sp

# Business days per year — arpym convention for converting day counts to years.
BUSINESS_DAYS_PER_YEAR = 252.0


def _busday_years(d0, d1) -> float:
    """Business-day gap between two dates expressed in years (252 convention)."""
    return np.busday_count(np.datetime64(d0, "D"), np.datetime64(d1, "D")) / BUSINESS_DAYS_PER_YEAR


def estimate_generator(
    dates: np.ndarray,
    n_oblig: np.ndarray,
    n_cum: np.ndarray,
    tau_hl: Optional[float] = None,
) -> np.ndarray:
    """
    Estimate the continuous-time generator matrix ``G`` from cohort history.

    See module docstring for the input format. Returns a ``(c_bar, c_bar)``
    generator with non-negative off-diagonals and zero row sums. The last state
    is treated as absorbing (its row is all zeros).
    """
    dates = np.asarray(dates)
    n_oblig = np.asarray(n_oblig, dtype=float)
    n_cum = np.asarray(n_cum, dtype=float)

    t_bar = len(dates)
    c_bar = n_cum.shape[1]
    if t_bar < 2:
        raise ValueError("need at least two dates to estimate transitions")

    # Per-period (non-cumulative) transition counts.
    m_num = np.zeros_like(n_cum)
    m_num[0] = n_cum[0]
    if t_bar > 1:
        m_num[1:] = np.diff(n_cum, axis=0)

    num = np.zeros((c_bar, c_bar))
    den = np.zeros((c_bar, c_bar))
    g = np.zeros((c_bar, c_bar))

    ln2 = np.log(2.0)

    for i in range(c_bar):
        for j in range(c_bar):
            if i == j:
                continue
            if tau_hl is None:
                # Equal weighting: total transitions over total time-at-risk.
                num[i, j] = n_cum[-1, i, j]
                for t in range(1, t_bar):
                    den[i, j] += n_oblig[t, i] * _busday_years(dates[t - 1], dates[t])
                g[i, j] = num[i, j] / den[i, j] if den[i, j] > 0 else 0.0
            else:
                # Exponential decay: weight by closeness to the last date.
                decay_rate = ln2 / tau_hl
                for t in range(t_bar):
                    num[i, j] += m_num[t, i, j] * np.exp(-decay_rate * _busday_years(dates[t], dates[-1]))
                for t in range(1, t_bar):
                    w_now = np.exp(-decay_rate * _busday_years(dates[t], dates[-1]))
                    w_prev = np.exp(-decay_rate * _busday_years(dates[t - 1], dates[-1]))
                    den[i, j] += n_oblig[t - 1, i] * (w_now - w_prev)
                g[i, j] = decay_rate * num[i, j] / den[i, j] if den[i, j] != 0 else 0.0

    # Off-diagonals must be non-negative (numerical guard for the decay branch).
    g = np.maximum(g, 0.0)
    np.fill_diagonal(g, 0.0)
    for i in range(c_bar):
        g[i, i] = -np.sum(g[i, :])

    # Absorbing default state: no outflow.
    g[-1, :] = 0.0
    return g


def _apply_monotonicity(p_prior: np.ndarray) -> np.ndarray:
    """
    arpym monotonicity correction via sequential relative-entropy minimisation.

    For each non-absorbing row ``c`` the posterior is the KL-closest distribution
    to ``p_prior[c]`` subject to an ordering (monotonicity) inequality whose
    sign flips as we move down the rating ladder. The final row is the absorbing
    default row ``[0, …, 0, 1]``.
    """
    c_bar = p_prior.shape[0]

    # Probability (equality) constraint: row sums to 1.
    a_eq = np.ones((1, c_bar))
    b_eq = np.array([1.0])

    # Monotonicity (inequality) constraint matrix: upper-bidiagonal differences.
    a_base = np.diagflat(np.ones(c_bar - 1), 1) - np.diagflat(np.ones(c_bar), 0)
    a_base = a_base[:-1]  # (c_bar-1, c_bar)
    b_ineq = np.zeros(c_bar - 1)

    a_ineq = {0: a_base.copy()}
    p = np.zeros((c_bar - 1, c_bar))
    for c in range(c_bar - 1):
        p[c, :] = min_rel_entropy_sp(
            p_prior[c, :], a_ineq[c], b_ineq, a_eq, b_eq, normalize=False
        )
        a_next = a_ineq[c].copy()
        a_next[c, :] = -a_next[c, :]
        a_ineq[c + 1] = a_next

    # Append absorbing default row.
    last = np.r_[np.zeros(c_bar - 1), 1.0]
    return np.r_[p, last[None, :]]


def fit_trans_matrix_credit(
    dates: np.ndarray,
    n_oblig: np.ndarray,
    n_cum: np.ndarray,
    tau_hl: Optional[float] = None,
    monotonic: bool = True,
) -> np.ndarray:
    """
    Estimate an annual credit transition matrix (rigorous, generator-based).

    Parameters
    ----------
    dates, n_oblig, n_cum, tau_hl
        See module docstring.
    monotonic : bool, default True
        Apply the entropy-regularised monotonicity correction. Set ``False`` to
        return the raw ``expm(G)`` prior (useful for diagnostics / comparison).

    Returns
    -------
    p : array, shape (c_bar, c_bar)
        Annual transition matrix; rows sum to 1; last state absorbing.
    """
    g = estimate_generator(dates, n_oblig, n_cum, tau_hl)
    p_prior = expm(g)
    # Numerical hygiene: clip tiny negatives, renormalise rows.
    p_prior = np.clip(p_prior, 0.0, 1.0)
    p_prior = p_prior / p_prior.sum(axis=1, keepdims=True)

    if not monotonic:
        p_prior[-1, :] = 0.0
        p_prior[-1, -1] = 1.0
        return p_prior

    return _apply_monotonicity(p_prior)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build the cohort arrays from a tidy migration-event log
# ──────────────────────────────────────────────────────────────────────────────

def cohort_arrays_from_events(
    events,
    n_ratings: int,
    period_col: str = "period",
    from_col: str = "from_state",
    to_col: str = "to_state",
    count_col: Optional[str] = None,
    population: Optional[np.ndarray] = None,
):
    """
    Turn a tidy migration-event table into ``(dates, n_oblig, n_cum)`` arrays.

    This is the ergonomic on-ramp for real data: most shops have a long table of
    "in period P, X obligors moved from rating I to rating J", not the 3-D cube
    the estimator wants.

    Parameters
    ----------
    events : pandas.DataFrame
        One row per (period, from_state, to_state) with an optional count.
        ``period`` values are sorted to define the period order; states are
        1-indexed (1..n_ratings) to match ``rating_engine``.
    n_ratings : int
        Number of rating states ``c_bar``.
    population : array (t_bar, n_ratings), optional
        Obligors at risk in each rating at the start of each period. If omitted,
        it is inferred as the row-wise total outflow per (period, from_state) —
        a serviceable proxy when an explicit population snapshot is unavailable.

    Returns
    -------
    dates : array (t_bar,) of consecutive integer day-offsets (one per period;
            252 business days apart) suitable for ``np.busday_count``.
    n_oblig : array (t_bar, n_ratings)
    n_cum   : array (t_bar, n_ratings, n_ratings)
    """
    import pandas as pd

    df = events.copy()
    periods = sorted(df[period_col].unique())
    t_bar = len(periods)
    c_bar = n_ratings

    # Per-period transition cube.
    m = np.zeros((t_bar, c_bar, c_bar))
    for t, per in enumerate(periods):
        sub = df[df[period_col] == per]
        for _, row in sub.iterrows():
            i = int(row[from_col]) - 1
            j = int(row[to_col]) - 1
            cnt = float(row[count_col]) if count_col else 1.0
            if 0 <= i < c_bar and 0 <= j < c_bar:
                m[t, i, j] += cnt

    n_cum = np.cumsum(m, axis=0)

    if population is not None:
        n_oblig = np.asarray(population, dtype=float)
    else:
        # Proxy: obligors at risk in rating i during period t = total outflow.
        n_oblig = m.sum(axis=2)

    # Synthetic equally-spaced dates: exactly one "year" per period under the
    # 252-business-day convention the estimator uses. Spacing by 252 BUSINESS
    # days (not 365 calendar days) makes _busday_years(dates[t-1], dates[t]) == 1.0
    # so each period contributes exactly 1.0 year of time-at-risk. (365 calendar
    # days is ~261 business days ≈ 1.036 yr, which would bias every generator
    # rate ~3.6% low.)
    base = np.datetime64("2000-01-03", "D")  # a Monday
    dates = np.array(
        [np.busday_offset(base, int(t * int(BUSINESS_DAYS_PER_YEAR)), roll="forward")
         for t in range(t_bar)],
        dtype="datetime64[D]",
    )
    return dates, n_oblig, n_cum
