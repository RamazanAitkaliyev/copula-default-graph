"""
Metric Comparison Harness  (src/metric_comparison.py)
=====================================================

PURPOSE
-------
Tests which metrics are representative and finds where they disagree.
The primary output is divergence_flags() — the early-warning signal that
RAROC is blind to network correlation while sortino_copula is not.

AGENT ENTRY POINT
-----------------
Preferred: use RiskAgentAPI.flag_divergences() and RiskAgentAPI.rank_metrics().
Direct use:
    comp = MetricComparator(calc)
    comp.rank_correlation()          # Spearman matrix — identify redundant metrics
    comp.divergence_flags(z=1.5)     # RAROC vs Sortino early-warning flags
    comp.disagreements('raroc', 'sortino_copula', top_n=20)

PRECONDITIONS
-------------
  - Requires a fully initialised RiskRatioCalculator (copula must be fitted).
  - borrower_table() is computed lazily and cached. The first call is expensive
    (O(n²) due to per-borrower loss_cov indexing). Subsequent calls are O(1).
    Use invalidate_cache=True only when the underlying data has changed.

KEY METHODS AND THEIR OUTPUT
-----------------------------
  borrower_table()
    → pd.DataFrame: one row per borrower, columns = metric names.
      Includes 'numerator_negative' bool flag.
      # AGENT: Do NOT use this table to aggregate by segment. Always call
      #   segment_table(col) which uses the correct block-sum aggregation.

  rank_correlation(level='borrower')
    → pd.DataFrame: square Spearman correlation matrix (metrics × metrics).
      Interpretation:
        ρ ≥ 0.95 → near-redundant pair (same information)
        ρ ∈ [0.7, 0.95] → related but not identical
        ρ < 0.7  → genuinely different signal (both worth keeping)
      # AGENT: At borrower level, CoV_L0 == CoV_L1 because off-diagonal LossCov
      #   terms cancel for n=1. The divergence between L0 and L1 appears at
      #   segment/portfolio level. Run rank_correlation(level='segment') for that.

  divergence_flags(z_threshold=1.5)
    → pd.DataFrame sorted by |z_score| descending. Columns:
        person_id, raroc, sortino_copula, raroc_rank, sortino_rank,
        rank_gap, z_score, flag_type
      flag_type values:
        "hidden_network_risk"   — z > threshold: good RAROC, bad Sortino
        "diversified_low_value" — z < -threshold: bad RAROC, good Sortino
      # AGENT: z_score is signed. Positive z = RAROC rank better than Sortino rank.
      #   The |z_score| column (not z_score) is used for severity ranking.
      #   Access with: flags.sort_values('z_score', key=abs, ascending=False)

  disagreements(metric_a, metric_b, top_n=20)
    → pd.DataFrame with columns: id, metric_a, metric_b, rank_a, rank_b, rank_gap.
      Sorted by rank_gap descending. These are the analytically interesting cases
      for manual analyst review.

RETURNS
-------
  All methods return pd.DataFrame. Empty DataFrame = no data (not an error).
  Convert to agent-safe format with _df_to_list() from src/agents.py.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .risk_adjusted_metrics import RiskRatioCalculator, available_metrics

logger = logging.getLogger(__name__)

# Metrics that are well-defined for ranking (exclude simulation-only and flags)
_RANKING_METRICS = [
    m for m in available_metrics()
    if m not in ("sortino_simulated",)  # requires with_sim; excluded from default ranking
]


class MetricComparator:
    """
    Compare all registered risk-adjusted metrics on the same population.

    Parameters
    ----------
    calc : RiskRatioCalculator
        A fully initialised calculator (copula must be fitted).
    """

    def __init__(self, calc: RiskRatioCalculator) -> None:
        self._calc = calc
        self._borrower_cache: Optional[pd.DataFrame] = None

    # ── internal ────────────────────────────────────────────────────────────

    def _get_borrower_table(self) -> pd.DataFrame:
        """Lazily compute and cache per-borrower metric table."""
        if self._borrower_cache is None:
            self._borrower_cache = self._calc.per_borrower()
        return self._borrower_cache

    # ── public API ───────────────────────────────────────────────────────────

    def borrower_table(
        self,
        metrics: Optional[List[str]] = None,
        invalidate_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one row per borrower and one column per metric.

        All metrics are computed from the same underlying primitives so values
        are mutually comparable. 'numerator_negative' column flags borrowers
        where signed metrics (RAROC, Sharpe, Sortino) may be misleading —
        use coefficient_of_variation* for riskiness ranking in those cases.
        """
        if invalidate_cache:
            self._borrower_cache = None
        df = self._get_borrower_table()
        if metrics is not None:
            cols = ["person_id"] + [m for m in metrics if m in df.columns] + ["numerator_negative"]
            return df[[c for c in cols if c in df.columns]]
        return df

    def segment_table(
        self,
        segment_col: str,
        metrics: Optional[List[str]] = None,
        *,
        with_sim: bool = False,
    ) -> pd.DataFrame:
        """
        Return metrics aggregated by a segment column.

        Parameters
        ----------
        segment_col : str
            Column in persons (e.g. 'city_name', 'risk_archetype').
        metrics : list, optional
            Subset of metric names. Defaults to all registered.
        with_sim : bool
            If True, also compute sortino_simulated.
        """
        return self._calc.by_segment(segment_col, metrics=metrics, with_sim=with_sim)

    def rank_correlation(
        self,
        metrics: Optional[List[str]] = None,
        level: str = "borrower",
        segment_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Spearman rank-correlation matrix between metrics.

        Pairs near 1.0 are redundant (they order the population identically).
        Low or negative correlations mean the two metrics encode different risk.
        This is the primary tool for deciding which metrics to keep.

        Parameters
        ----------
        metrics : list, optional
            Subset. Defaults to the ranking-safe set (excludes sortino_simulated).
        level : str
            'borrower' (default) or 'segment' (requires segment_col).
        segment_col : str, optional
            Required when level='segment'.

        Returns
        -------
        pd.DataFrame
            Square symmetric correlation matrix with metrics as index and columns.
        """
        metric_names = metrics or _RANKING_METRICS

        if level == "segment":
            if segment_col is None:
                raise ValueError("segment_col is required when level='segment'")
            df = self.segment_table(segment_col, metrics=metric_names)
        else:
            df = self.borrower_table(metrics=metric_names)

        # Filter to only the numeric metric columns that exist
        cols = [m for m in metric_names if m in df.columns]
        values = df[cols].values.astype(float)

        n_cols = len(cols)
        corr_matrix = np.eye(n_cols)

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                xi = values[:, i]
                xj = values[:, j]
                # Mask out NaN rows for this pair
                mask = np.isfinite(xi) & np.isfinite(xj)
                if mask.sum() < 3:
                    c = np.nan
                else:
                    c, _ = spearmanr(xi[mask], xj[mask])
                    c = float(c) if np.isfinite(c) else np.nan
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

        return pd.DataFrame(corr_matrix, index=cols, columns=cols)

    def disagreements(
        self,
        metric_a: str,
        metric_b: str,
        top_n: int = 20,
        level: str = "borrower",
        segment_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Find the units where metric_a and metric_b most disagree on ranking.

        These are the analytically interesting cases: they reveal borrowers /
        segments that look very different depending on which metric you use,
        and are the prime candidates for investigation by risk analysts.

        Returns a DataFrame sorted by rank disagreement, descending.
        """
        if level == "segment":
            if segment_col is None:
                raise ValueError("segment_col required for segment-level disagreements")
            df = self.segment_table(segment_col, metrics=[metric_a, metric_b])
            id_col = "segment"
        else:
            df = self.borrower_table(metrics=[metric_a, metric_b])
            id_col = "person_id"

        if metric_a not in df.columns or metric_b not in df.columns:
            return pd.DataFrame()

        mask = df[metric_a].notna() & df[metric_b].notna()
        df = df[mask].copy()
        if df.empty:
            return df

        df["rank_a"] = df[metric_a].rank(ascending=False, na_option="bottom")
        df["rank_b"] = df[metric_b].rank(ascending=False, na_option="bottom")
        df["rank_gap"] = (df["rank_a"] - df["rank_b"]).abs()

        cols = [id_col, metric_a, metric_b, "rank_a", "rank_b", "rank_gap"]
        return (df[[c for c in cols if c in df.columns]]
                .sort_values("rank_gap", ascending=False)
                .head(top_n)
                .reset_index(drop=True))

    def divergence_flags(
        self,
        z_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Flag borrowers where RAROC and sortino_copula strongly diverge.

        RAROC is blind to network correlation (capital = k·EAD); sortino_copula
        inflates its denominator for concentrated/contagious borrowers. When a
        borrower looks good on RAROC but bad on sortino_copula, it is likely
        embedded in a high-risk network cluster — the kind of early-warning
        signal that standalone credit analysis misses.

        Parameters
        ----------
        z_threshold : float
            Number of standard deviations away from mean rank-gap to flag.

        Returns
        -------
        pd.DataFrame
            Flagged borrowers, sorted by divergence severity.
            Columns: person_id, raroc, sortino_copula,
                     raroc_rank, sortino_rank, rank_gap, z_score, flag_type
        """
        df = self.borrower_table(metrics=["raroc", "sortino_copula"])
        if "raroc" not in df.columns or "sortino_copula" not in df.columns:
            return pd.DataFrame()

        mask = df["raroc"].notna() & df["sortino_copula"].notna()
        df = df[mask].copy()
        if df.empty:
            return df

        df["raroc_rank"] = df["raroc"].rank(ascending=False, na_option="bottom")
        df["sortino_rank"] = df["sortino_copula"].rank(ascending=False, na_option="bottom")
        df["rank_gap"] = df["raroc_rank"] - df["sortino_rank"]

        gap_mean = df["rank_gap"].mean()
        gap_std = df["rank_gap"].std()

        if gap_std < 1e-10:
            return pd.DataFrame()

        df["z_score"] = (df["rank_gap"] - gap_mean) / gap_std

        flagged = df[df["z_score"].abs() >= z_threshold].copy()

        def _flag_type(z: float) -> str:
            if z >= z_threshold:
                # Good RAROC, bad Sortino → network risk hidden from RAROC
                return "hidden_network_risk"
            else:
                # Bad RAROC, good Sortino → diversified despite low individual value
                return "diversified_low_value"

        flagged["flag_type"] = flagged["z_score"].apply(_flag_type)

        cols = ["person_id", "raroc", "sortino_copula",
                "raroc_rank", "sortino_rank", "rank_gap", "z_score", "flag_type"]
        return (flagged[[c for c in cols if c in flagged.columns]]
                .sort_values("z_score", key=np.abs, ascending=False)
                .reset_index(drop=True))
