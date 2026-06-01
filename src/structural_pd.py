"""
Structural (Merton) Probability of Default

Implements the Merton (1974) structural credit model as a second, independent
PD signal alongside the statistical gradient-boosting model.

Core idea
---------
A firm defaults when its asset value V falls below its debt D at maturity T.
Using the Black-Scholes framework:

    PD = N( -(log(V/D) + (μ - σ²/2)·T) / (σ·√T) )

where N(·) is the standard normal CDF.

Since we don't observe asset value or asset volatility directly, we back them
out from observable equity value and equity volatility using the KMV / Merton
iterative system:

    Equity  = V·N(d1) - D·e^(-r·T)·N(d2)
    σ_E·E   = V·σ_V·N(d1)        (Ito's lemma)

For retail borrowers without traded equity, we approximate asset value and
asset volatility from observable features (income, debt, credit utilisation)
following the spirit of the Merton model.

Two modes
---------
1. market_mode (traded firms):  provide equity_value + equity_vol + debt
2. proxy_mode  (retail):        derive proxies from person features

The two PDs are then blended with the statistical model PD via a
configurable alpha weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MertonParams:
    """Calibrated Merton model parameters for one borrower."""
    asset_value: float
    asset_vol: float
    debt_face_value: float
    horizon_years: float
    risk_free_rate: float
    distance_to_default: float
    merton_pd: float


# ──────────────────────────────────────────────────────────────────────────────
# Pure Merton formulas
# ──────────────────────────────────────────────────────────────────────────────

def _d1(V: float, D: float, r: float, sigma: float, T: float) -> float:
    return (np.log(V / D) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(V: float, D: float, r: float, sigma: float, T: float) -> float:
    return _d1(V, D, r, sigma, T) - sigma * np.sqrt(T)


def merton_pd_direct(
    asset_value: float,
    debt: float,
    asset_vol: float,
    T: float = 1.0,
    r: float = 0.02,
) -> Tuple[float, float]:
    """
    Compute Merton PD given asset value and asset volatility directly.

    Returns
    -------
    (pd, distance_to_default)
    """
    if asset_value <= 0 or debt <= 0 or asset_vol <= 0:
        return 1.0, 0.0

    d2_val = _d2(asset_value, debt, r, asset_vol, T)
    dd = d2_val
    pd = float(stats.norm.cdf(-d2_val))
    return np.clip(pd, 1e-6, 1.0 - 1e-6), dd


def calibrate_from_equity(
    equity_value: float,
    equity_vol: float,
    debt: float,
    T: float = 1.0,
    r: float = 0.02,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MertonParams:
    """
    Back out asset value and asset volatility from observable equity data.

    Solves the KMV / Merton system iteratively:
        E  = V·N(d1) - D·e^{-rT}·N(d2)
        σ_E·E = V·σ_V·N(d1)

    Parameters
    ----------
    equity_value : market cap
    equity_vol   : annualised equity volatility
    debt         : face value of debt at maturity T
    T            : horizon in years
    r            : risk-free rate
    """
    # Initial guess: asset value = equity + debt, asset vol = equity vol / 2
    V = equity_value + debt
    sigma_V = equity_vol * equity_value / max(V, 1e-6)

    for _ in range(max_iter):
        d1_val = _d1(V, debt, r, sigma_V, T)
        d2_val = d1_val - sigma_V * np.sqrt(T)

        N_d1 = float(stats.norm.cdf(d1_val))
        N_d2 = float(stats.norm.cdf(d2_val))

        equity_model = V * N_d1 - debt * np.exp(-r * T) * N_d2
        sigma_V_new = equity_vol * equity_value / max(V * N_d1, 1e-6)

        V_new = equity_value + debt * np.exp(-r * T) * N_d2

        if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
            break
        V = V_new
        sigma_V = sigma_V_new

    pd_val, dd = merton_pd_direct(V, debt, sigma_V, T, r)

    return MertonParams(
        asset_value=V,
        asset_vol=sigma_V,
        debt_face_value=debt,
        horizon_years=T,
        risk_free_rate=r,
        distance_to_default=dd,
        merton_pd=pd_val,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Proxy Merton for retail borrowers (no traded equity)
# ──────────────────────────────────────────────────────────────────────────────

def _proxy_asset_value(income: float, debt_to_income: float) -> float:
    """
    Approximate asset value as present value of future income stream.
    Assumes a simplified perpetuity: V ≈ income / discount_rate
    and total debt = income × debt_to_income.
    """
    discount_rate = 0.08  # 8% personal discount rate
    V = income * 12 / discount_rate   # annualise monthly income, capitalise
    return max(V, 1.0)


def _proxy_asset_vol(
    credit_utilisation: float,
    missed_payments: int,
    employment_years: float,
) -> float:
    """
    Approximate asset volatility from observable stress signals.

    Higher utilisation / missed payments / shorter tenure → higher vol.
    Range roughly [0.10, 0.60] matching retail asset-vol estimates.
    """
    base_vol = 0.15
    util_add = credit_utilisation * 0.25
    miss_add = min(missed_payments, 10) * 0.02
    emp_sub = min(employment_years, 20) * 0.005
    return float(np.clip(base_vol + util_add + miss_add - emp_sub, 0.08, 0.65))


def _proxy_debt(income: float, debt_to_income: float) -> float:
    """Total debt face value ≈ annual income × DTI ratio."""
    return max(income * 12 * debt_to_income, 1.0)


def compute_proxy_merton_pd(
    income: float,
    debt_to_income: float,
    credit_utilisation: float,
    missed_payments: int,
    employment_years: float,
    T: float = 1.0,
    r: float = 0.02,
) -> MertonParams:
    """
    Compute Merton PD for a retail borrower using observable feature proxies.

    Parameters
    ----------
    income            : monthly income
    debt_to_income    : total debt / annual income
    credit_utilisation: fraction of credit limit used (0–1)
    missed_payments   : number of historical missed payments
    employment_years  : tenure at current employer
    T                 : horizon in years
    r                 : risk-free rate
    """
    V = _proxy_asset_value(income, debt_to_income)
    D = _proxy_debt(income, debt_to_income)
    sigma_V = _proxy_asset_vol(credit_utilisation, missed_payments, employment_years)

    pd_val, dd = merton_pd_direct(V, D, sigma_V, T, r)

    return MertonParams(
        asset_value=V,
        asset_vol=sigma_V,
        debt_face_value=D,
        horizon_years=T,
        risk_free_rate=r,
        distance_to_default=dd,
        merton_pd=pd_val,
    )


# ──────────────────────────────────────────────────────────────────────────────
# StructuralPDModel: vectorised over a DataFrame
# ──────────────────────────────────────────────────────────────────────────────

class StructuralPDModel:
    """
    Compute Merton structural PD for every borrower in a DataFrame.

    Combines with the statistical model PD to produce a blended signal:

        blended_pd = alpha * merton_pd + (1 - alpha) * statistical_pd

    The divergence between the two signals (|merton_pd - statistical_pd|)
    is itself a useful early-warning indicator: when the Merton model
    suggests higher risk than the statistical model, the market-implied
    deterioration precedes the financial-statement deterioration.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        T: float = 1.0,
        r: float = 0.02,
    ) -> None:
        """
        Parameters
        ----------
        alpha : weight on Merton PD in the blend (0 = pure statistical, 1 = pure Merton)
        T     : PD horizon in years
        r     : risk-free rate
        """
        self.alpha = alpha
        self.T = T
        self.r = r
        self._results: Optional[pd.DataFrame] = None

    def fit_transform(
        self,
        persons: pd.DataFrame,
        statistical_pd_col: str = "model_pd",
    ) -> pd.DataFrame:
        """
        Compute Merton PD for all borrowers and blend with statistical PD.

        Returns a copy of persons with extra columns:
            merton_pd          — structural PD
            distance_to_default — Merton DD (higher = safer)
            asset_vol_proxy    — implied asset volatility
            blended_pd         — alpha*merton + (1-alpha)*statistical
            pd_signal_divergence — |merton_pd - statistical_pd|
        """
        required = {"income", "debt_to_income", "credit_utilization",
                    "missed_payments", "employment_years"}
        missing = required - set(persons.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if statistical_pd_col not in persons.columns:
            statistical_pd_col = "base_pd"

        results = persons.copy()
        n = len(results)

        merton_pds   = np.zeros(n)
        dds          = np.zeros(n)
        asset_vols   = np.zeros(n)

        for i, row in results.iterrows():
            params = compute_proxy_merton_pd(
                income=float(row["income"]),
                debt_to_income=float(row["debt_to_income"]),
                credit_utilisation=float(row["credit_utilization"]),
                missed_payments=int(row["missed_payments"]),
                employment_years=float(row["employment_years"]),
                T=self.T,
                r=self.r,
            )
            idx = results.index.get_loc(i)
            merton_pds[idx] = params.merton_pd
            dds[idx]        = params.distance_to_default
            asset_vols[idx] = params.asset_vol

        stat_pds = results[statistical_pd_col].values.astype(float)

        results["merton_pd"]            = np.round(merton_pds, 4)
        results["distance_to_default"]  = np.round(dds, 4)
        results["asset_vol_proxy"]      = np.round(asset_vols, 4)
        results["blended_pd"]           = np.round(
            self.alpha * merton_pds + (1 - self.alpha) * stat_pds, 4
        )
        results["pd_signal_divergence"] = np.round(
            np.abs(merton_pds - stat_pds), 4
        )

        self._results = results
        return results

    def get_early_warnings(
        self,
        divergence_threshold: float = 0.10,
    ) -> pd.DataFrame:
        """
        Return borrowers where Merton PD significantly exceeds statistical PD.

        These are cases where market-implied stress precedes financial-statement
        deterioration — a classic early warning signal.
        """
        if self._results is None:
            raise RuntimeError("Call fit_transform() first.")

        flags = self._results[
            self._results["pd_signal_divergence"] >= divergence_threshold
        ].copy()

        flags = flags.sort_values("pd_signal_divergence", ascending=False)
        return flags[[
            "person_id", "merton_pd", "blended_pd",
            "pd_signal_divergence", "distance_to_default",
            "asset_vol_proxy",
        ]]

    def summary_stats(self) -> pd.DataFrame:
        """Return distribution statistics of both PD signals."""
        if self._results is None:
            raise RuntimeError("Call fit_transform() first.")

        cols = ["merton_pd", "blended_pd", "distance_to_default",
                "pd_signal_divergence"]
        present = [c for c in cols if c in self._results.columns]
        desc = self._results[present].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ).T
        return desc


if __name__ == "__main__":
    from data_generator import generate_network
    from pd_model import IndividualPDModel
    import warnings; warnings.filterwarnings("ignore")

    print("Generating data...")
    persons, _ = generate_network(seed=42)
    stat_model = IndividualPDModel("gradient_boosting")
    stat_model.fit(persons, "default")
    persons["model_pd"] = stat_model.predict_proba(persons)

    print("Computing Merton structural PD...")
    struct_model = StructuralPDModel(alpha=0.35)
    enriched = struct_model.fit_transform(persons, "model_pd")

    print("\nPD signal summary:")
    print(struct_model.summary_stats().to_string())

    warnings_df = struct_model.get_early_warnings(divergence_threshold=0.05)
    print(f"\nEarly warnings ({len(warnings_df)} borrowers with divergence > 5%):")
    print(warnings_df.head(10).to_string(index=False))
