"""
Rating Migration Engine

Converts continuous PD scores into discrete credit rating states and models
how those ratings migrate over time using a Markov transition matrix.

Ratings (8 states):
    1=AAA  2=AA  3=A  4=BBB  5=BB  6=B  7=CCC  8=Default

Key capabilities:
- Map PD → rating bucket
- Estimate annual transition matrix from historical migration counts
  (self-contained reimplementation of arpym fit_trans_matrix_credit /
   project_trans_matrix, no external dependency on arpym)
- Project transition matrix to any horizon (monthly, quarterly, annual)
- Simulate correlated joint rating paths via t-copula on transitions
- Summarise portfolio rating distribution and migration risk
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from scipy import stats

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Rating constants
# ──────────────────────────────────────────────────────────────────────────────

RATING_LABELS: List[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
N_RATINGS: int = len(RATING_LABELS)  # 8

# PD upper-bound thresholds that define each rating bucket.
# A borrower with PD ≤ threshold maps to that bucket.
# Thresholds tuned to broadly match S&P long-run average PDs.
PD_THRESHOLDS: np.ndarray = np.array([
    0.001,   # AAA
    0.003,   # AA
    0.010,   # A
    0.030,   # BBB
    0.080,   # BB
    0.200,   # B
    0.500,   # CCC
    1.000,   # Default
])

# Approximate long-run average annual transition matrix (S&P-like).
# Row i = current rating, column j = next-year rating.
# Last row/col = absorbing default state.
_DEFAULT_TRANSITION: np.ndarray = np.array([
    [0.9081, 0.0833, 0.0068, 0.0006, 0.0008, 0.0002, 0.0002, 0.0000],
    [0.0070, 0.9065, 0.0779, 0.0064, 0.0006, 0.0013, 0.0002, 0.0001],
    [0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0026, 0.0001, 0.0006],
    [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018],
    [0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.0884, 0.0100, 0.0106],
    [0.0000, 0.0011, 0.0024, 0.0043, 0.0648, 0.8346, 0.0407, 0.0521],
    [0.0022, 0.0000, 0.0022, 0.0130, 0.0238, 0.1124, 0.6443, 0.2021],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RatingProfile:
    """Rating information for a single borrower."""
    person_id: int
    current_rating: int          # 1-indexed rating state (1=AAA … 8=Default)
    current_rating_label: str    # "AAA" … "Default"
    marginal_pd: float
    one_year_migration: np.ndarray   # shape (8,) — prob of each rating in 1 yr
    upgrade_prob: float
    downgrade_prob: float
    default_prob_1yr: float
    default_prob_3yr: float


@dataclass
class PortfolioRatingDistribution:
    """Portfolio-level rating distribution summary."""
    counts: Dict[str, int]
    fractions: Dict[str, float]
    weighted_avg_pd: float
    migration_risk_score: float   # expected fraction of portfolio that migrates


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────

def pd_to_rating(pd_value: float) -> int:
    """Map a PD in [0, 1] to a rating integer 1–8."""
    pd_value = float(np.clip(pd_value, 0.0, 1.0))
    for state, threshold in enumerate(PD_THRESHOLDS, start=1):
        if pd_value <= threshold:
            return state
    return N_RATINGS  # Default


def pd_array_to_ratings(pds: np.ndarray) -> np.ndarray:
    """Vectorised pd_to_rating over an array."""
    return np.array([pd_to_rating(p) for p in pds], dtype=int)


def _generator_from_transition(p: np.ndarray) -> np.ndarray:
    """
    Compute the continuous-time generator matrix G from annual transition P.

    G = logm(P) projected to be a valid generator:
    - off-diagonal entries ≥ 0
    - row sums = 0
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = np.real(logm(p))

    # Project off-diagonals to be non-negative
    g_proj = g.copy()
    np.fill_diagonal(g_proj, 0.0)
    g_proj = np.maximum(g_proj, 0.0)
    # Restore diagonal so rows sum to zero
    np.fill_diagonal(g_proj, -g_proj.sum(axis=1))
    return g_proj


def project_transition_matrix(
    p_annual: np.ndarray,
    delta_t: float = 1.0,
) -> np.ndarray:
    """
    Project an annual transition matrix to horizon delta_t (in years).

    Uses the matrix-exponential of the generator:  P(Δt) = expm(G · Δt)

    Parameters
    ----------
    p_annual : (c, c) annual transition matrix
    delta_t  : horizon in years (e.g. 0.25 = quarterly, 3.0 = 3 years)

    Returns
    -------
    (c, c) projected transition matrix with absorbing default row.
    """
    n = p_annual.shape[0]
    g = _generator_from_transition(p_annual)
    p_dt = expm(g * delta_t)
    p_dt = np.clip(p_dt, 0.0, 1.0)
    # Enforce absorbing default state
    p_dt[-1, :] = 0.0
    p_dt[-1, -1] = 1.0
    # Re-normalise rows
    row_sums = p_dt.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    p_dt = p_dt / row_sums
    return p_dt


def estimate_transition_matrix(
    historical_counts: Optional[np.ndarray] = None,
    tau_hl_years: Optional[float] = None,
) -> np.ndarray:
    """
    Estimate annual transition matrix from historical migration counts.

    If no data is provided, returns the industry-standard S&P-like baseline.

    Parameters
    ----------
    historical_counts : (t_bar, c, c) array of migration counts per period, optional.
        historical_counts[t, i, j] = number of borrowers that migrated
        from rating i to rating j in period t.
    tau_hl_years : half-life for exponential time-weighting (older data counts less).

    Returns
    -------
    (c, c) estimated annual transition matrix.
    """
    if historical_counts is None:
        return _DEFAULT_TRANSITION.copy()

    t_bar, c, _ = historical_counts.shape
    g = np.zeros((c, c))

    for i in range(c - 1):  # skip absorbing default state
        n_total = historical_counts[:, i, :].sum()
        if n_total == 0:
            g[i, :] = _DEFAULT_TRANSITION[i, :]
            continue

        for j in range(c):
            if i == j:
                continue
            if tau_hl_years is None:
                g[i, j] = historical_counts[:, i, j].sum() / max(n_total, 1)
            else:
                # Exponential decay: recent periods weight more
                decay = np.array([
                    np.exp(-np.log(2) / tau_hl_years * (t_bar - 1 - t))
                    for t in range(t_bar)
                ])
                g[i, j] = (historical_counts[:, i, j] * decay).sum() / \
                           max((historical_counts[:, i, :].sum(axis=1) * decay).sum(), 1e-10)

    # Diagonal = 1 - sum of off-diagonals (stay probability)
    for i in range(c - 1):
        g[i, i] = max(0.0, 1.0 - g[i, :].sum() + g[i, i])

    # Absorbing default
    g[-1, :] = 0.0
    g[-1, -1] = 1.0

    # Re-normalise
    row_sums = g.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return g / row_sums


def simulate_correlated_rating_paths(
    initial_ratings: np.ndarray,
    transition_matrix: np.ndarray,
    n_periods: int = 4,
    n_simulations: int = 1000,
    correlation_matrix: Optional[np.ndarray] = None,
    nu: float = 6.0,
) -> np.ndarray:
    """
    Simulate correlated joint rating migration paths via t-copula.

    Follows the arpym simulate_markov_chain_multiv approach:
    draw uniform grades from a multivariate-t copula, then use
    the CDF of the transition row as a quantile function.

    Parameters
    ----------
    initial_ratings : (n,) integer array of current ratings (1-indexed)
    transition_matrix : (c, c) one-period transition matrix
    n_periods : number of periods to simulate
    n_simulations : Monte Carlo paths
    correlation_matrix : (n, n) copula correlation, defaults to identity
    nu : t-copula degrees of freedom (lower = heavier tails)

    Returns
    -------
    paths : (n_simulations, n_periods+1, n) integer rating states
    """
    n = len(initial_ratings)
    c = transition_matrix.shape[0]

    if correlation_matrix is None:
        correlation_matrix = np.eye(n)

    # Clip correlation matrix to be PSD
    rho = np.clip(correlation_matrix, -0.999, 0.999)
    np.fill_diagonal(rho, 1.0)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-8)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(rho, 1.0)

    paths = np.zeros((n_simulations, n_periods + 1, n), dtype=int)
    paths[:, 0, :] = initial_ratings[np.newaxis, :]  # broadcast initial state

    # Pre-compute CDF thresholds for each rating row
    thresholds = np.zeros((c, c + 1))
    for r in range(c):
        thresholds[r, :] = np.r_[0.0, np.cumsum(transition_matrix[r, :])]

    for period in range(n_periods):
        # Sample from multivariate-t copula — shape (n_simulations, n)
        try:
            z = stats.multivariate_t.rvs(
                loc=np.zeros(n), shape=rho, df=nu, size=n_simulations
            )
        except Exception:
            # Fallback: Gaussian copula
            z = np.random.multivariate_normal(np.zeros(n), rho, size=n_simulations)
            grades = stats.norm.cdf(z)
        else:
            grades = stats.t.cdf(z, df=nu)

        # Map each borrower's grade to new rating via inverse CDF of transition row
        for sim in range(n_simulations):
            for borrower in range(n):
                current_state = paths[sim, period, borrower] - 1  # 0-indexed
                current_state = int(np.clip(current_state, 0, c - 1))
                grade = grades[sim, borrower]
                new_state = int(np.searchsorted(thresholds[current_state, 1:], grade))
                new_state = int(np.clip(new_state, 0, c - 1))
                paths[sim, period + 1, borrower] = new_state + 1  # back to 1-indexed

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# RatingEngine class
# ──────────────────────────────────────────────────────────────────────────────

class RatingEngine:
    """
    Convert continuous PD scores to discrete rating states and model migrations.

    Usage
    -----
        engine = RatingEngine()
        engine.fit(persons_df)
        profile = engine.get_rating_profile(person_id=42)
        dist    = engine.portfolio_distribution()
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        historical_counts: Optional[np.ndarray] = None,
        tau_hl_years: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        transition_matrix : override the annual transition matrix directly.
        historical_counts : (t, c, c) array to estimate transition matrix from data.
        tau_hl_years : exponential half-life for time-weighting historical data.
        """
        if transition_matrix is not None:
            self.transition_annual = np.array(transition_matrix, dtype=float)
        else:
            self.transition_annual = estimate_transition_matrix(
                historical_counts, tau_hl_years
            )

        # Pre-compute projected matrices at common horizons
        self.transition_1yr  = project_transition_matrix(self.transition_annual, 1.0)
        self.transition_3yr  = project_transition_matrix(self.transition_annual, 3.0)
        self.transition_qtr  = project_transition_matrix(self.transition_annual, 0.25)

        self.persons: Optional[pd.DataFrame] = None
        self.ratings: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, persons: pd.DataFrame, pd_col: str = "model_pd") -> "RatingEngine":
        """
        Assign ratings to all borrowers.

        Parameters
        ----------
        persons : DataFrame with at least one PD column.
        pd_col  : column to use for PD (falls back to 'base_pd' if missing).
        """
        if "person_id" not in persons.columns:
            raise ValueError("persons must have a 'person_id' column")
        if persons["person_id"].duplicated().any():
            raise ValueError("persons has duplicate person_ids")
        if pd_col not in persons.columns:
            if "base_pd" not in persons.columns:
                raise ValueError(f"Neither '{pd_col}' nor 'base_pd' found in persons columns")
            pd_col = "base_pd"

        self.persons = persons.copy().reset_index(drop=True)
        self.pd_col = pd_col
        pds = self.persons[pd_col].values

        if np.any(pds < 0) or np.any(pds > 1):
            raise ValueError(f"'{pd_col}' contains values outside [0, 1]")
        self.ratings = pd_array_to_ratings(pds)
        self.persons["rating_state"] = self.ratings
        self.persons["rating_label"] = [RATING_LABELS[r - 1] for r in self.ratings]
        self._is_fitted = True
        logger.info(
            "RatingEngine fitted: %d borrowers, "
            "rating distribution: %s",
            len(persons),
            dict(pd.Series(self.ratings).map(lambda r: RATING_LABELS[r - 1]).value_counts()),
        )
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

    # ── per-borrower ──────────────────────────────────────────────────────────

    def get_rating_profile(self, person_id: int) -> RatingProfile:
        """Return a RatingProfile for a single borrower."""
        self._check_fitted()
        # Use positional index from values array to avoid DataFrame-index aliasing
        positions = np.where(self.persons["person_id"].values == person_id)[0]
        if len(positions) == 0:
            raise ValueError(f"person_id {person_id} not found in fitted persons")
        pos = int(positions[0])
        state = int(self.ratings[pos])
        pd_val = float(self.persons.iloc[pos][self.pd_col])

        row_1yr = self.transition_1yr[state - 1]
        row_3yr = self.transition_3yr[state - 1]

        # upgrade: any rating better (lower index) than current, excluding Default col
        upgrade_prob   = float(row_1yr[:state - 1].sum()) if state > 1 else 0.0
        # downgrade: any rating worse (higher index) than current, excluding Default col
        downgrade_prob = float(row_1yr[state:N_RATINGS - 1].sum()) if state < N_RATINGS - 1 else 0.0

        return RatingProfile(
            person_id=person_id,
            current_rating=state,
            current_rating_label=RATING_LABELS[state - 1],
            marginal_pd=pd_val,
            one_year_migration=row_1yr,
            upgrade_prob=upgrade_prob,
            downgrade_prob=downgrade_prob,
            default_prob_1yr=float(row_1yr[-1]),
            default_prob_3yr=float(row_3yr[-1]),
        )

    def migration_table(self, person_id: int) -> pd.DataFrame:
        """Return a tidy DataFrame of 1-year migration probabilities."""
        profile = self.get_rating_profile(person_id)
        return pd.DataFrame({
            "to_rating": RATING_LABELS,
            "probability": profile.one_year_migration,
        }).assign(
            current_rating=profile.current_rating_label,
            person_id=person_id,
        )

    # ── portfolio ─────────────────────────────────────────────────────────────

    def portfolio_distribution(self) -> PortfolioRatingDistribution:
        """Summarise current rating distribution across the portfolio."""
        self._check_fitted()
        series = pd.Series(self.ratings).map(lambda r: RATING_LABELS[r - 1])
        counts = {lbl: int(series.eq(lbl).sum()) for lbl in RATING_LABELS}
        n_total = len(self.ratings)
        fractions = {lbl: counts[lbl] / n_total for lbl in RATING_LABELS}

        weighted_pd = float(
            (self.persons[self.pd_col] *
             self.persons.get("income", pd.Series(np.ones(n_total)))).sum() /
            self.persons.get("income", pd.Series(np.ones(n_total))).sum()
        )

        # Migration risk: expected fraction of portfolio that changes rating
        stay_probs = np.array([
            self.transition_1yr[r - 1, r - 1] for r in self.ratings
        ])
        migration_risk = float(1.0 - stay_probs.mean())

        return PortfolioRatingDistribution(
            counts=counts,
            fractions=fractions,
            weighted_avg_pd=weighted_pd,
            migration_risk_score=migration_risk,
        )

    def expected_migrations(self, n_periods: int = 4) -> pd.DataFrame:
        """
        Expected rating one-step ahead for each borrower over n_periods quarters.

        Returns a DataFrame of shape (n_borrowers, n_periods+1) of
        expected rating states (float, weighted average).
        """
        self._check_fitted()
        n = len(self.ratings)
        p_qtr = self.transition_qtr
        result = np.zeros((n, n_periods + 1))
        result[:, 0] = self.ratings.astype(float)
        state_vals = np.arange(1, N_RATINGS + 1, dtype=float)

        for period in range(1, n_periods + 1):
            p_t = np.linalg.matrix_power(
                np.round(p_qtr, 10), period
            )
            for i, r in enumerate(self.ratings):
                result[i, period] = p_t[r - 1] @ state_vals

        cols = ["t0"] + [f"q+{q}" for q in range(1, n_periods + 1)]
        df = pd.DataFrame(result, columns=cols)
        df.insert(0, "person_id", self.persons["person_id"].values)
        return df

    def simulate_portfolio_paths(
        self,
        n_simulations: int = 500,
        n_periods: int = 4,
        correlation_matrix: Optional[np.ndarray] = None,
        nu: float = 6.0,
    ) -> np.ndarray:
        """
        Monte Carlo simulation of correlated rating paths for entire portfolio.

        Returns
        -------
        paths : (n_simulations, n_periods+1, n_borrowers) integer rating states
        """
        self._check_fitted()
        return simulate_correlated_rating_paths(
            initial_ratings=self.ratings,
            transition_matrix=self.transition_qtr,
            n_periods=n_periods,
            n_simulations=n_simulations,
            correlation_matrix=correlation_matrix,
            nu=nu,
        )

    def portfolio_default_rate_distribution(
        self,
        n_simulations: int = 1000,
        horizon_periods: int = 4,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate portfolio default rates at the given horizon.

        Returns
        -------
        default_rates : (n_simulations,) array of simulated default fractions
        """
        paths = self.simulate_portfolio_paths(
            n_simulations=n_simulations,
            n_periods=horizon_periods,
            correlation_matrix=correlation_matrix,
        )
        # last period, check if any period hit default (state 8)
        ever_defaulted = (paths == N_RATINGS).any(axis=1)  # (n_sim, n_borrowers)
        return ever_defaulted.mean(axis=1)

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame with person_id, rating_label, rating_state, pd, and 1yr default prob."""
        self._check_fitted()
        profiles = [self.get_rating_profile(pid)
                    for pid in self.persons["person_id"].values]
        return pd.DataFrame([{
            "person_id":       p.person_id,
            "rating_label":    p.current_rating_label,
            "rating_state":    p.current_rating,
            "marginal_pd":     round(p.marginal_pd, 4),
            "upgrade_prob":    round(p.upgrade_prob, 4),
            "downgrade_prob":  round(p.downgrade_prob, 4),
            "default_1yr":     round(p.default_prob_1yr, 4),
            "default_3yr":     round(p.default_prob_3yr, 4),
        } for p in profiles])


if __name__ == "__main__":
    from data_generator import generate_network
    from pd_model import IndividualPDModel

    persons, transactions = generate_network(seed=42)
    model = IndividualPDModel("gradient_boosting")
    model.fit(persons, "default")
    persons["model_pd"] = model.predict_proba(persons)

    engine = RatingEngine()
    engine.fit(persons, pd_col="model_pd")

    dist = engine.portfolio_distribution()
    print("\nPortfolio rating distribution:")
    for lbl in RATING_LABELS:
        bar = "█" * int(dist.fractions[lbl] * 50)
        print(f"  {lbl:8s} {dist.counts[lbl]:4d}  {dist.fractions[lbl]:.1%}  {bar}")
    print(f"\n  Weighted avg PD:    {dist.weighted_avg_pd:.3%}")
    print(f"  Migration risk:     {dist.migration_risk_score:.1%}")

    profile = engine.get_rating_profile(persons["person_id"].iloc[0])
    print(f"\nProfile — person {profile.person_id}:")
    print(f"  Rating: {profile.current_rating_label}  PD={profile.marginal_pd:.3%}")
    print(f"  1yr default prob: {profile.default_prob_1yr:.4%}")
    print(f"  3yr default prob: {profile.default_prob_3yr:.4%}")
    print(f"  Upgrade prob:     {profile.upgrade_prob:.3%}")
    print(f"  Downgrade prob:   {profile.downgrade_prob:.3%}")
