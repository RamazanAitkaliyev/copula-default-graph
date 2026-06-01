"""
Flexible Probability / Regime-Aware Copula Calibration

Implements the arpym conditional_fp approach for weighting historical
observations by how closely they resemble the current macro regime.

Core idea
---------
Instead of treating all historical data points equally (uniform probabilities),
we assign higher probability mass to observations recorded under conditions
similar to today's macro environment (e.g. similar stress level, similar
default rate regime, similar credit spread).

This means:
- In calm markets: copula correlation calibrated on calm-period data → θ ≈ low
- In stressed markets: calibrated on stressed-period data → θ ≈ high
- No manual pd_multiplier needed — the model tightens automatically

The result is a regime-conditional copula theta and correlation matrix, which
feeds directly into CopulaDefaultModel.

Approach (self-contained, no arpym dependency)
----------------------------------------------
1. Define a macro stress indicator Z (e.g. rolling default rate,
   portfolio avg PD, VIX-like proxy).
2. For each current stress level z*, compute a kernel-smoothed probability
   vector over historical observations:
       p_i ∝ K((z_i - z*) / bandwidth)
   where K is a Gaussian kernel.
3. Use these flexible probabilities to compute a weighted correlation matrix
   and a weighted copula theta — replacing the simple average.
4. Optionally also produce a low-rank decomposition of the weighted
   correlation matrix (systematic + idiosyncratic components).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """Current market regime characterisation."""
    stress_score: float       # 0 (calm) to 1 (extreme stress)
    label: str                # "calm" | "moderate" | "stressed" | "crisis"
    effective_n_obs: float    # effective number of observations used
    regime_weights: np.ndarray  # (t,) probability weights on historical periods


@dataclass
class RegimeAdjustedCopula:
    """Output of flexible-probability copula calibration."""
    theta: float              # regime-adjusted Clayton theta
    avg_correlation: float    # weighted average pairwise correlation
    correlation_matrix: np.ndarray  # (n, n) regime-adjusted correlation
    regime: RegimeState
    # Low-rank decomposition: corr ≈ beta @ beta.T + diag
    loadings: Optional[np.ndarray] = None   # (n, k) systematic factor loadings
    idio_var: Optional[np.ndarray] = None   # (n,) idiosyncratic variances


# ──────────────────────────────────────────────────────────────────────────────
# Kernel / flexible probability helpers
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_kernel_weights(
    z_history: np.ndarray,
    z_current: float,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Gaussian kernel weights centred at z_current.

    p_i ∝ exp( -(z_i - z*)² / (2·h²) )

    Parameters
    ----------
    z_history  : (t,) time series of macro indicator values
    z_current  : current value of the macro indicator
    bandwidth  : kernel bandwidth h. Defaults to Silverman's rule: 1.06·σ·t^(-1/5)

    Returns
    -------
    weights : (t,) array summing to 1
    """
    t = len(z_history)
    if bandwidth is None:
        bandwidth = 1.06 * z_history.std() * t ** (-0.2)
    bandwidth = max(bandwidth, 1e-6)

    raw = np.exp(-0.5 * ((z_history - z_current) / bandwidth) ** 2)
    total = raw.sum()
    if total < 1e-12:
        return np.ones(t) / t
    return raw / total


def exponential_decay_weights(
    t: int,
    half_life: float,
) -> np.ndarray:
    """
    Time-decay weights: recent observations get higher weight.

    w_i ∝ exp(-λ · (t-1-i))  where λ = log(2) / half_life

    Parameters
    ----------
    t         : number of observations
    half_life : number of periods for weight to halve

    Returns
    -------
    weights : (t,) array summing to 1, most recent = index t-1
    """
    lam = np.log(2) / max(half_life, 1e-6)
    ages = np.arange(t - 1, -1, -1, dtype=float)  # 0 = most recent
    w = np.exp(-lam * ages)
    return w / w.sum()


def combine_weights(
    *weight_arrays: np.ndarray,
    method: str = "product",
) -> np.ndarray:
    """
    Combine multiple weight arrays (kernel + decay + custom).

    method : "product" — multiply then normalise (AND logic)
             "average" — average then normalise
    """
    assert len(weight_arrays) >= 1
    t = len(weight_arrays[0])
    if method == "product":
        combined = np.ones(t)
        for w in weight_arrays:
            combined *= np.maximum(w, 1e-20)
    else:
        combined = np.mean(np.stack(weight_arrays, axis=0), axis=0)

    total = combined.sum()
    if total < 1e-12:
        return np.ones(t) / t
    return combined / total


def effective_n(weights: np.ndarray) -> float:
    """Effective number of observations: 1 / sum(p²)."""
    return float(1.0 / (weights ** 2).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Macro stress indicator construction
# ──────────────────────────────────────────────────────────────────────────────

def build_stress_indicator(
    persons_snapshots: Optional[pd.DataFrame] = None,
    portfolio_avg_pds: Optional[np.ndarray] = None,
    external_indicator: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a scalar stress time series Z from available data.

    Priority: external_indicator > portfolio_avg_pds > persons_snapshots.

    Parameters
    ----------
    persons_snapshots : DataFrame with column 'avg_pd' per time period
    portfolio_avg_pds : (t,) pre-computed portfolio average PD series
    external_indicator : (t,) external series (e.g. VIX, credit spread)

    Returns
    -------
    z : (t,) normalised stress indicator in [0, 1]
    """
    if external_indicator is not None:
        z = np.asarray(external_indicator, dtype=float)
    elif portfolio_avg_pds is not None:
        z = np.asarray(portfolio_avg_pds, dtype=float)
    elif persons_snapshots is not None and "avg_pd" in persons_snapshots.columns:
        z = persons_snapshots["avg_pd"].values.astype(float)
    else:
        raise ValueError("Provide at least one of: external_indicator, "
                         "portfolio_avg_pds, persons_snapshots.")

    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-10:
        return np.full(len(z), 0.5)
    return (z - z_min) / (z_max - z_min)


def classify_regime(stress_score: float) -> str:
    """Map a [0, 1] stress score to a human-readable label."""
    if stress_score < 0.25:
        return "calm"
    elif stress_score < 0.50:
        return "moderate"
    elif stress_score < 0.75:
        return "stressed"
    else:
        return "crisis"


# ──────────────────────────────────────────────────────────────────────────────
# Low-rank correlation decomposition
# ──────────────────────────────────────────────────────────────────────────────

def low_rank_decomposition(
    corr: np.ndarray,
    n_factors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose correlation matrix into low-rank + diagonal (idiosyncratic).

    corr ≈ β·βᵀ + diag(δ²)

    where β is (n, k) loadings on k systematic factors.

    Parameters
    ----------
    corr      : (n, n) correlation matrix (must be PSD)
    n_factors : number of systematic factors k

    Returns
    -------
    beta     : (n, k) factor loadings
    idio_var : (n,) idiosyncratic variances
    """
    n = corr.shape[0]
    k = min(n_factors, n - 1)

    eigvals, eigvecs = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Systematic: top k eigenvectors
    lam_k = np.maximum(eigvals[:k], 0.0)
    beta = eigvecs[:, :k] * np.sqrt(lam_k)[np.newaxis, :]  # (n, k)

    # Idiosyncratic: diagonal of corr - beta @ beta.T
    systematic = beta @ beta.T
    idio = np.diag(corr - systematic)
    idio_var = np.maximum(idio, 1e-6)

    return beta, idio_var


# ──────────────────────────────────────────────────────────────────────────────
# FlexibleProbsCalibrator
# ──────────────────────────────────────────────────────────────────────────────

class FlexibleProbsCalibrator:
    """
    Calibrate copula parameters conditional on the current market regime.

    Usage
    -----
        calib = FlexibleProbsCalibrator()
        calib.fit(historical_pd_series, historical_corr_series)
        regime_copula = calib.calibrate(current_stress_score=0.7,
                                        base_corr_matrix=corr_now)
        # Use regime_copula.theta and regime_copula.correlation_matrix
        # in CopulaDefaultModel.fit()
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        half_life_periods: Optional[float] = None,
        n_factors: int = 5,
        copula_type: str = "clayton",
    ) -> None:
        """
        Parameters
        ----------
        bandwidth         : kernel bandwidth for stress conditioning (auto = Silverman)
        half_life_periods : time-decay half-life in periods (None = no time decay)
        n_factors         : number of systematic factors in low-rank decomposition
        copula_type       : "clayton" | "gaussian" | "student_t"
        """
        self.bandwidth = bandwidth
        self.half_life = half_life_periods
        self.n_factors = n_factors
        self.copula_type = copula_type

        self._stress_history: Optional[np.ndarray] = None
        self._avg_corr_history: Optional[np.ndarray] = None
        self._n_history: int = 0

    def fit(
        self,
        stress_history: np.ndarray,
        avg_corr_history: Optional[np.ndarray] = None,
    ) -> "FlexibleProbsCalibrator":
        """
        Store historical stress and correlation data for conditioning.

        Parameters
        ----------
        stress_history   : (t,) normalised stress indicator [0, 1]
        avg_corr_history : (t,) average off-diagonal correlation at each period.
                           If None, defaults to a linear mapping from stress.
        """
        stress_history = np.asarray(stress_history, dtype=float)
        if len(stress_history) < 2:
            raise ValueError("stress_history needs at least 2 observations")
        if np.all(np.isnan(stress_history)):
            raise ValueError("stress_history is all NaNs")
        if avg_corr_history is not None:
            avg_corr_history = np.asarray(avg_corr_history, dtype=float)
            if len(avg_corr_history) != len(stress_history):
                raise ValueError("avg_corr_history must have same length as stress_history")

        if stress_history.std() < 1e-10:
            logger.warning("stress_history is constant; kernel weights will be uniform")

        self._stress_history = stress_history
        t = len(self._stress_history)
        self._n_history = t

        if avg_corr_history is not None:
            self._avg_corr_history = np.asarray(avg_corr_history, dtype=float)
        else:
            # Synthetic: correlation scales linearly with stress from 0.03 to 0.40
            self._avg_corr_history = 0.03 + 0.37 * self._stress_history

        return self

    def _compute_weights(self, current_stress: float) -> np.ndarray:
        """Combine kernel and optional time-decay weights."""
        kernel_w = gaussian_kernel_weights(
            self._stress_history, current_stress, self.bandwidth
        )
        if self.half_life is not None:
            decay_w = exponential_decay_weights(self._n_history, self.half_life)
            return combine_weights(kernel_w, decay_w, method="product")
        return kernel_w

    def _corr_to_theta(self, avg_corr: float) -> float:
        """Convert average correlation to Clayton theta via Kendall's tau."""
        tau = (2.0 / np.pi) * np.arcsin(np.clip(avg_corr, -0.99, 0.99))
        tau = np.clip(tau, 0.01, 0.95)
        return max(0.05, 2 * tau / (1 - tau))

    def calibrate(
        self,
        current_stress: float,
        base_corr_matrix: np.ndarray,
        decompose: bool = True,
    ) -> RegimeAdjustedCopula:
        """
        Produce regime-adjusted copula parameters.

        Parameters
        ----------
        current_stress    : current stress score in [0, 1]
        base_corr_matrix  : (n, n) base correlation matrix from graph
        decompose         : whether to compute low-rank decomposition

        Returns
        -------
        RegimeAdjustedCopula with regime-adjusted theta + corr matrix
        """
        if self._stress_history is None:
            raise RuntimeError("Call fit() first.")

        weights = self._compute_weights(current_stress)

        # Weighted average correlation
        weighted_avg_corr = float(weights @ self._avg_corr_history)

        # Scale the base correlation matrix to match weighted average
        n = base_corr_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        current_avg = base_corr_matrix[mask].mean()

        if current_avg > 1e-6:
            scale = weighted_avg_corr / current_avg
        else:
            scale = 1.0

        adj_corr = base_corr_matrix.copy()
        # Scale off-diagonals only
        adj_corr[mask] = np.clip(adj_corr[mask] * scale, 0.0, 0.95)
        np.fill_diagonal(adj_corr, 1.0)

        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(adj_corr)
        eigvals = np.maximum(eigvals, 1e-8)
        adj_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(adj_corr, 1.0)

        # Copula theta
        theta = self._corr_to_theta(weighted_avg_corr)

        # Regime state
        regime = RegimeState(
            stress_score=current_stress,
            label=classify_regime(current_stress),
            effective_n_obs=effective_n(weights),
            regime_weights=weights,
        )

        # Optional low-rank decomposition
        beta, idio = None, None
        if decompose:
            beta, idio = low_rank_decomposition(adj_corr, self.n_factors)

        logger.info(
            "Regime calibration: stress=%.2f (%s), "
            "theta=%.4f, avg_corr=%.4f, eff_n=%.1f",
            current_stress, regime.label, theta,
            weighted_avg_corr, regime.effective_n_obs,
        )

        return RegimeAdjustedCopula(
            theta=theta,
            avg_correlation=weighted_avg_corr,
            correlation_matrix=adj_corr,
            regime=regime,
            loadings=beta,
            idio_var=idio,
        )

    def calibrate_for_scenarios(
        self,
        stress_scenarios: np.ndarray,
        base_corr_matrix: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute calibrated theta and avg_corr for a range of stress scenarios.

        Useful for building a stress table showing how copula tightens.

        Parameters
        ----------
        stress_scenarios : (k,) array of stress levels to evaluate (e.g. [0.1, 0.3, 0.5, 0.7, 0.9])

        Returns
        -------
        DataFrame with columns: stress_score, regime, theta, avg_correlation, eff_n_obs
        """
        rows = []
        for s in stress_scenarios:
            result = self.calibrate(float(s), base_corr_matrix, decompose=False)
            rows.append({
                "stress_score":    round(float(s), 2),
                "regime":          result.regime.label,
                "theta":           round(result.theta, 4),
                "avg_correlation": round(result.avg_correlation, 4),
                "eff_n_obs":       round(result.regime.effective_n_obs, 1),
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build calibrator from a persons DataFrame time series
# ──────────────────────────────────────────────────────────────────────────────

def build_calibrator_from_portfolio(
    avg_pd_series: np.ndarray,
    avg_corr_series: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
    half_life_periods: Optional[float] = 12.0,
    n_factors: int = 5,
) -> FlexibleProbsCalibrator:
    """
    Convenience constructor: build and fit a calibrator from a portfolio's
    historical average PD series.

    Parameters
    ----------
    avg_pd_series    : (t,) time series of portfolio average PD
    avg_corr_series  : (t,) optional corresponding average correlation series
    half_life_periods: time-decay half-life (default 12 periods = 1 year if monthly)
    """
    stress = build_stress_indicator(portfolio_avg_pds=avg_pd_series)
    calib = FlexibleProbsCalibrator(
        bandwidth=bandwidth,
        half_life_periods=half_life_periods,
        n_factors=n_factors,
    )
    calib.fit(stress, avg_corr_series)
    return calib


if __name__ == "__main__":
    import numpy as np
    from src.data_generator import generate_network
    from src.graph_features import TransactionGraph

    np.random.seed(42)
    persons, transactions = generate_network(seed=42)
    graph = TransactionGraph(transactions, persons)
    corr = graph.get_correlation_matrix()

    # Simulate 24 months of history with increasing stress
    t = 24
    avg_pd_history = 0.05 + 0.15 * np.linspace(0, 1, t) + 0.02 * np.random.randn(t)

    calib = build_calibrator_from_portfolio(avg_pd_history)

    print("Stress scenario table:")
    table = calib.calibrate_for_scenarios(
        np.array([0.1, 0.25, 0.50, 0.75, 0.90]), corr
    )
    print(table.to_string(index=False))

    # Current calibration at moderate stress
    result = calib.calibrate(current_stress=0.5, base_corr_matrix=corr)
    print(f"\nCurrent regime: {result.regime.label}")
    print(f"Adjusted theta: {result.theta:.4f}")
    print(f"Avg correlation: {result.avg_correlation:.4f}")
    print(f"Effective n_obs: {result.regime.effective_n_obs:.1f}")
    if result.loadings is not None:
        print(f"Factor loadings shape: {result.loadings.shape}")
