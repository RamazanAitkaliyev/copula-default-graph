"""
Customer Risk Profile

Single entry-point that combines every layer of the framework into a
one-page risk profile for any borrower — the kind a risk analyst would
review in a credit committee or relationship-manager briefing.

Profile fields
--------------
Identity
    person_id, city, risk_archetype

Rating & PD signals
    current_rating (AAA … Default)
    statistical_pd    — gradient-boosting model
    merton_pd         — structural Merton model
    blended_pd        — alpha-weighted combination
    pd_signal_divergence — early-warning flag
    distance_to_default

Migration outlook (1-year horizon)
    upgrade_prob, downgrade_prob, default_1yr, default_3yr
    top_destination_rating  — most likely rating 1yr from now

Network risk
    contagion_vulnerability  — how much PD rises if neighbours default
    systemic_importance      — how much PD of others rises if this one defaults
    n_connections, top_5_contagion_neighbours

Business value
    client_sharpe, raroc, cltv_risk_adjusted
    segment (Stars / Cash Cows / Question Marks / Dogs)
    contagion_adjusted_sharpe

Composite assessment
    composite_risk_score
    risk_tier (low / medium / high / critical)
    recommended_action
    narrative (one plain-English paragraph)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CustomerRiskProfile:
    """Complete risk profile for a single customer."""

    # Identity
    person_id: int
    city: str
    archetype: str

    # PD signals
    statistical_pd: float
    merton_pd: float
    blended_pd: float
    distance_to_default: float
    pd_signal_divergence: float
    early_warning: bool               # True if divergence > threshold

    # Rating
    current_rating: str               # "AAA" … "Default"
    current_rating_state: int         # 1 … 8
    upgrade_prob: float
    downgrade_prob: float
    default_prob_1yr: float
    default_prob_3yr: float
    most_likely_rating_1yr: str

    # Network
    contagion_vulnerability: float
    systemic_importance: float
    n_connections: int
    top_neighbours: List[Dict[str, Any]]   # [{person_id, joint_pd, city}]

    # Business value
    expected_revenue: float
    expected_loss: float
    client_sharpe: float
    raroc: float
    cltv_risk_adjusted: float
    contagion_adjusted_sharpe: float
    segment: str                       # Stars / Cash Cows / Question Marks / Dogs

    # Risk-adjusted metrics (from RiskRatioCalculator; 0.0 if not available)
    coefficient_of_variation: float = 0.0
    profile_raroc: float = 0.0         # named profile_raroc to avoid shadowing existing raroc field
    sortino_copula: float = 0.0

    # Composite
    composite_risk_score: float = 0.0
    risk_tier: str = "low"             # low / medium / high / critical
    recommended_action: str = ""
    narrative: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "top_neighbours"}
        d["top_neighbours"] = self.top_neighbours
        return d

    def to_series(self) -> pd.Series:
        d = self.to_dict()
        d["top_neighbours"] = str(d["top_neighbours"])
        return pd.Series(d)


# ──────────────────────────────────────────────────────────────────────────────
# Action / narrative rules
# ──────────────────────────────────────────────────────────────────────────────

_ACTION_RULES = {
    ("critical", True):   "ESCALATE IMMEDIATELY — Flag for Credit Committee review. "
                          "Merton model signals structural deterioration ahead of financials.",
    ("critical", False):  "ESCALATE — Place on watch list. Initiate intensive monitoring "
                          "and request updated financial information.",
    ("high", True):       "REVIEW — Merton divergence signals early stress. Schedule "
                          "relationship manager call within 2 weeks.",
    ("high", False):      "REVIEW — High composite risk. Reassess exposure limits and "
                          "consider covenant triggers.",
    ("medium", True):     "MONITOR CLOSELY — Merton divergence present. Increase "
                          "monitoring frequency to monthly.",
    ("medium", False):    "MONITOR — Maintain standard quarterly review cycle. "
                          "No immediate action required.",
    ("low", True):        "WATCH — Minor divergence. Continue standard monitoring.",
    ("low", False):       "MAINTAIN — No immediate risk action required. "
                          "Standard annual review sufficient.",
}


def _recommend_action(tier: str, early_warning: bool) -> str:
    return _ACTION_RULES.get((tier, early_warning),
                             "MONITOR — Review per standard cycle.")


def _build_narrative(profile_data: dict) -> str:
    """Produce a one-paragraph plain-English summary."""
    pid   = profile_data["person_id"]
    city  = profile_data["city"]
    arch  = profile_data["archetype"]
    rtg   = profile_data["current_rating"]
    bpd   = profile_data["blended_pd"]
    tier  = profile_data["risk_tier"]
    cv    = profile_data["contagion_vulnerability"]
    si    = profile_data["systemic_importance"]
    d1    = profile_data["default_prob_1yr"]
    seg   = profile_data["segment"]
    sharpe = profile_data["client_sharpe"]
    ew    = profile_data["early_warning"]
    div   = profile_data["pd_signal_divergence"]
    dd    = profile_data["distance_to_default"]

    ew_sentence = (
        f" The Merton structural model shows a {div:.1%} divergence from the "
        f"statistical estimate (distance-to-default = {dd:.2f}), indicating "
        f"market-implied deterioration that may not yet be visible in financial statements."
        if ew else ""
    )

    contagion_sentence = ""
    if cv > 0.002:
        contagion_sentence = (
            f" This borrower has elevated contagion vulnerability ({cv:.4f}): "
            f"defaults among connected counterparties would materially increase their PD."
        )
    if si > 0.005:
        contagion_sentence += (
            f" Their systemic importance ({si:.4f}) is above average — "
            f"their own default would raise PDs across their network."
        )

    return (
        f"Customer {pid} ({arch} archetype, {city}) is currently rated {rtg} "
        f"with a blended probability of default of {bpd:.2%} and a 1-year "
        f"default probability of {d1:.2%}. The overall risk tier is {tier.upper()}."
        f"{ew_sentence}"
        f"{contagion_sentence} "
        f"From a business-value perspective, this client falls in the '{seg}' "
        f"segment with a Client Sharpe ratio of {sharpe:.2f}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# CustomerProfiler — main class
# ──────────────────────────────────────────────────────────────────────────────

class CustomerProfiler:
    """
    Combines all framework layers into per-customer risk profiles.

    Usage
    -----
        profiler = CustomerProfiler()
        profiler.fit(
            persons=persons_df,
            transactions=transactions_df,
            graph=graph,
            copula=fitted_copula,
            pd_model=fitted_pd_model,
            rating_engine=fitted_rating_engine,    # optional
            structural_model=fitted_struct_model,  # optional
            client_value_calc=fitted_cvc,          # optional
            individual_risks=individual_risks_df,  # from RiskAnalyzer
        )
        profile = profiler.get_profile(person_id=42)
        report  = profiler.profile_report(person_id=42)   # printable string
        df      = profiler.all_profiles()                 # DataFrame of all customers
    """

    def __init__(
        self,
        early_warning_threshold: float = 0.05,
    ) -> None:
        self.early_warning_threshold = early_warning_threshold
        self._fitted = False

    def fit(
        self,
        persons: pd.DataFrame,
        transactions: pd.DataFrame,
        graph,
        copula,
        pd_model=None,
        rating_engine=None,
        structural_model=None,
        client_value_calc=None,
        individual_risks: Optional[pd.DataFrame] = None,
        risk_ratio_calc=None,
    ) -> "CustomerProfiler":
        """
        Attach all model outputs. Any component can be None — the profiler
        gracefully falls back to available data.

        Parameters
        ----------
        persons          : base persons DataFrame (must have person_id, base_pd / model_pd)
        transactions     : transaction DataFrame
        graph            : fitted TransactionGraph
        copula           : fitted CopulaDefaultModel
        pd_model         : fitted IndividualPDModel (optional)
        rating_engine    : fitted RatingEngine (optional)
        structural_model : StructuralPDModel.fit_transform() result DataFrame (optional)
        client_value_calc: fitted ClientValueCalculator (optional)
        individual_risks : DataFrame from RiskAnalyzer.compute_individual_risks() (optional)
        risk_ratio_calc  : fitted RiskRatioCalculator (optional); populates CoV/RAROC/Sortino
        """
        self._persons = persons.copy().reset_index(drop=True)
        self._transactions = transactions
        self._graph = graph
        self._copula = copula
        self._n = len(persons)

        # Determine which PD column to use
        self._stat_pd_col = "model_pd" if "model_pd" in persons.columns else "base_pd"

        # Optional enriched DataFrames
        self._structural_df = structural_model   # already-computed DataFrame
        self._rating_engine = rating_engine
        self._cvc = client_value_calc
        self._individual_risks = individual_risks

        # Pre-compute client metrics if calculator provided
        self._client_metrics: Optional[pd.DataFrame] = None
        if self._cvc is not None:
            try:
                self._client_metrics = self._cvc.compute_contagion_adjusted_sharpe()
                self._segments = self._cvc.segment_clients()
            except Exception as e:
                logger.warning("Client value computation failed: %s", e)
                self._client_metrics = None
                self._segments = None

        # Pre-compute per-borrower metric table from RiskRatioCalculator if provided
        self._metric_df: Optional[pd.DataFrame] = None
        if risk_ratio_calc is not None:
            try:
                self._metric_df = risk_ratio_calc.per_borrower(
                    metrics=["coefficient_of_variation", "raroc", "sortino_copula"]
                )
            except Exception as e:
                logger.warning("Risk ratio per-borrower computation failed: %s", e)

        # Cache adjacency for neighbour lookups
        self._adj_binary = graph.adj_binary   # (n, n)

        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _stat_pd(self, idx: int) -> float:
        return float(self._persons.iloc[idx].get(self._stat_pd_col, 0.1))

    def _merton_data(self, person_id: int) -> dict:
        if self._structural_df is None:
            return {"merton_pd": 0.0, "distance_to_default": 0.0,
                    "blended_pd": self._stat_pd(self._idx(person_id)),
                    "pd_signal_divergence": 0.0}
        row = self._structural_df[
            self._structural_df["person_id"] == person_id
        ]
        if row.empty:
            spd = self._stat_pd(self._idx(person_id))
            return {"merton_pd": 0.0, "distance_to_default": 0.0,
                    "blended_pd": spd, "pd_signal_divergence": 0.0}
        r = row.iloc[0]
        return {
            "merton_pd":            float(r.get("merton_pd", 0.0)),
            "distance_to_default":  float(r.get("distance_to_default", 0.0)),
            "blended_pd":           float(r.get("blended_pd",
                                                  r.get(self._stat_pd_col, 0.1))),
            "pd_signal_divergence": float(r.get("pd_signal_divergence", 0.0)),
        }

    def _idx(self, person_id: int) -> int:
        """Row index in self._persons for given person_id."""
        matches = self._persons.index[
            self._persons["person_id"] == person_id
        ].tolist()
        if not matches:
            raise ValueError(f"person_id {person_id} not found")
        return matches[0]

    def _rating_data(self, person_id: int) -> dict:
        if self._rating_engine is None:
            return {
                "current_rating": "N/A", "current_rating_state": -1,
                "upgrade_prob": 0.0, "downgrade_prob": 0.0,
                "default_prob_1yr": self._stat_pd(self._idx(person_id)),
                "default_prob_3yr": self._stat_pd(self._idx(person_id)) * 2.5,
                "most_likely_rating_1yr": "N/A",
            }
        try:
            p = self._rating_engine.get_rating_profile(person_id)
            most_likely_idx = int(np.argmax(p.one_year_migration))
            from src.rating_engine import RATING_LABELS
            return {
                "current_rating": p.current_rating_label,
                "current_rating_state": p.current_rating,
                "upgrade_prob": p.upgrade_prob,
                "downgrade_prob": p.downgrade_prob,
                "default_prob_1yr": p.default_prob_1yr,
                "default_prob_3yr": p.default_prob_3yr,
                "most_likely_rating_1yr": RATING_LABELS[most_likely_idx],
            }
        except (ValueError, KeyError, IndexError) as e:
            logger.warning("Rating profile failed for %d: %s", person_id, e)
            return {
                "current_rating": "N/A", "current_rating_state": -1,
                "upgrade_prob": 0.0, "downgrade_prob": 0.0,
                "default_prob_1yr": 0.0, "default_prob_3yr": 0.0,
                "most_likely_rating_1yr": "N/A",
            }

    def _network_data(self, person_id: int) -> dict:
        idx = self._idx(person_id)
        neighbours = np.where(self._adj_binary[idx] > 0)[0]
        n_conn = len(neighbours)

        # Contagion scores from copula (cached on _copula)
        vuln = 0.0
        imp = 0.0
        if self._copula.is_fitted:
            # Compute for this one borrower
            try:
                nbrs = np.where(self._copula.correlation_matrix[idx] > 0.1)[0]
                nbrs = nbrs[nbrs != idx]
                if len(nbrs) > 0:
                    sample = nbrs[:50] if len(nbrs) > 50 else nbrs
                    uplifts = []
                    for j in sample:
                        cond = self._copula.conditional_default_probability(idx, int(j))
                        uplift = cond - self._copula.marginal_pds[idx]
                        uplifts.append(uplift * self._copula.marginal_pds[j])
                    vuln = float(np.mean(uplifts)) if uplifts else 0.0

                    impacts = []
                    for j in sample:
                        cond = self._copula.conditional_default_probability(int(j), idx)
                        impacts.append(cond - self._copula.marginal_pds[j])
                    imp = float(np.mean(impacts)) if impacts else 0.0
            except Exception as e:
                logger.debug("Contagion calc error for %d: %s", person_id, e)

        # Top-5 neighbours by joint PD
        top_nbrs = []
        if self._copula.is_fitted and len(neighbours) > 0:
            sample_nbrs = neighbours[:30]
            joint_pds = []
            for j in sample_nbrs:
                try:
                    jp = self._copula.joint_default_probability(idx, int(j))
                    joint_pds.append((int(j), float(jp)))
                except Exception as e:
                    # Skip this neighbour but record why — a silent pass here
                    # would hide a real copula indexing error.
                    logger.debug(
                        "joint_default_probability(%d, %d) failed: %s", idx, j, e
                    )
            joint_pds.sort(key=lambda x: x[1], reverse=True)
            for j, jp in joint_pds[:5]:
                p_row = self._persons.iloc[j]
                top_nbrs.append({
                    "person_id": int(p_row["person_id"]),
                    "joint_pd": round(jp, 4),
                    "city": str(p_row.get("city_name", "?")),
                    "individual_pd": round(float(p_row.get(self._stat_pd_col, 0.0)), 4),
                })

        return {
            "contagion_vulnerability": round(vuln, 5),
            "systemic_importance": round(imp, 5),
            "n_connections": n_conn,
            "top_neighbours": top_nbrs,
        }

    def _business_data(self, person_id: int) -> dict:
        defaults = {
            "expected_revenue": 0.0,
            "expected_loss": 0.0,
            "client_sharpe": 0.0,
            "raroc": 0.0,
            "cltv_risk_adjusted": 0.0,
            "contagion_adjusted_sharpe": 0.0,
            "segment": "Unknown",
        }
        if self._client_metrics is None:
            return defaults
        row = self._client_metrics[
            self._client_metrics["person_id"] == person_id
        ]
        if row.empty:
            return defaults
        r = row.iloc[0]

        seg = "Unknown"
        if self._segments is not None:
            seg_row = self._segments[self._segments["person_id"] == person_id]
            if not seg_row.empty:
                seg = str(seg_row.iloc[0].get("segment", "Unknown"))

        return {
            "expected_revenue":         round(float(r.get("expected_revenue", 0)), 2),
            "expected_loss":            round(float(r.get("expected_loss", 0)), 4),
            "client_sharpe":            round(float(r.get("client_sharpe", 0)), 3),
            "raroc":                    round(float(r.get("raroc", 0)), 3),
            "cltv_risk_adjusted":       round(float(r.get("cltv_risk_adjusted", 0)), 2),
            "contagion_adjusted_sharpe": round(
                float(r.get("contagion_adjusted_sharpe", 0)), 3),
            "segment": seg,
        }

    def _composite_data(self, person_id: int) -> dict:
        """Look up composite score and tier from pre-computed individual_risks."""
        defaults = {"composite_risk_score": 0.0, "risk_tier": "low"}
        if self._individual_risks is None:
            return defaults
        row = self._individual_risks[
            self._individual_risks["person_id"] == person_id
        ]
        if row.empty:
            return defaults
        r = row.iloc[0]
        return {
            "composite_risk_score": round(float(r.get("composite_risk_score", 0)), 4),
            "risk_tier": str(r.get("risk_tier", "low")),
        }

    def _metric_data(self, person_id: int) -> dict:
        """Look up CoV / RAROC / Sortino from pre-computed per-borrower metric table."""
        defaults = {
            "coefficient_of_variation": 0.0,
            "profile_raroc": 0.0,
            "sortino_copula": 0.0,
        }
        if self._metric_df is None:
            return defaults
        row = self._metric_df[self._metric_df["person_id"] == person_id]
        if row.empty:
            return defaults
        r = row.iloc[0]
        import math
        def safe(val, fallback=0.0):
            v = float(r.get(val, fallback))
            return fallback if math.isnan(v) else round(v, 5)
        return {
            "coefficient_of_variation": safe("coefficient_of_variation"),
            "profile_raroc": safe("raroc"),
            "sortino_copula": safe("sortino_copula"),
        }

    # ── public API ────────────────────────────────────────────────────────────

    def get_profile(self, person_id: int) -> CustomerRiskProfile:
        """
        Build and return a complete CustomerRiskProfile for one borrower.
        """
        self._check_fitted()
        idx = self._idx(person_id)
        row = self._persons.iloc[idx]

        stat_pd = self._stat_pd(idx)
        merton  = self._merton_data(person_id)
        rating  = self._rating_data(person_id)
        network = self._network_data(person_id)
        biz     = self._business_data(person_id)
        comp    = self._composite_data(person_id)
        metrics = self._metric_data(person_id)

        early_warning = merton["pd_signal_divergence"] >= self.early_warning_threshold

        action = _recommend_action(comp["risk_tier"], early_warning)

        profile_data = {
            "person_id":    person_id,
            "city":         str(row.get("city_name", "?")),
            "archetype":    str(row.get("risk_archetype", "?")),
            "statistical_pd": round(stat_pd, 4),
            **merton,
            **rating,
            **network,
            **biz,
            **comp,
            **metrics,
            "early_warning":        early_warning,
            "recommended_action":   action,
        }
        profile_data["narrative"] = _build_narrative(profile_data)

        return CustomerRiskProfile(**{
            k: profile_data[k] for k in CustomerRiskProfile.__dataclass_fields__
        })

    def profile_report(self, person_id: int) -> str:
        """Return a formatted multi-section text report for one borrower."""
        p = self.get_profile(person_id)
        divider = "─" * 60

        lines = [
            f"\n{'═'*60}",
            f"  CUSTOMER RISK PROFILE  —  ID {p.person_id}",
            f"{'═'*60}",
            "",
            f"  City: {p.city}   |   Archetype: {p.archetype}",
            f"  Risk Tier: {p.risk_tier.upper()}   |   Rating: {p.current_rating}",
            "",
            divider,
            "  PD SIGNALS",
            divider,
            f"  Statistical PD  :  {p.statistical_pd:.3%}",
            f"  Merton PD       :  {p.merton_pd:.3%}   "
            f"(Distance-to-Default: {p.distance_to_default:.2f})",
            f"  Blended PD      :  {p.blended_pd:.3%}",
            f"  Signal Divergence: {p.pd_signal_divergence:.3%}"
            + ("  ⚠ EARLY WARNING" if p.early_warning else ""),
            "",
            divider,
            "  RATING MIGRATION (1-YEAR)",
            divider,
            f"  Current Rating  :  {p.current_rating}",
            f"  Upgrade Prob    :  {p.upgrade_prob:.2%}",
            f"  Downgrade Prob  :  {p.downgrade_prob:.2%}",
            f"  Default Prob 1yr:  {p.default_prob_1yr:.3%}",
            f"  Default Prob 3yr:  {p.default_prob_3yr:.3%}",
            f"  Most Likely (1yr): {p.most_likely_rating_1yr}",
            "",
            divider,
            "  NETWORK RISK",
            divider,
            f"  Connections     :  {p.n_connections}",
            f"  Contagion Vuln  :  {p.contagion_vulnerability:.5f}",
            f"  Systemic Imp    :  {p.systemic_importance:.5f}",
        ]

        if p.top_neighbours:
            lines.append("  Top Contagion Neighbours:")
            for nb in p.top_neighbours:
                lines.append(
                    f"    → ID {nb['person_id']:4d}  "
                    f"Joint PD={nb['joint_pd']:.4f}  "
                    f"City={nb['city']}  "
                    f"Ind PD={nb['individual_pd']:.3%}"
                )

        lines += [
            "",
            divider,
            "  BUSINESS VALUE",
            divider,
            f"  Segment         :  {p.segment}",
            f"  Expected Revenue:  {p.expected_revenue:,.2f}",
            f"  Expected Loss   :  {p.expected_loss:.4f}",
            f"  Client Sharpe   :  {p.client_sharpe:.3f}",
            f"  RAROC           :  {p.raroc:.3f}",
            f"  CLTV (adj)      :  {p.cltv_risk_adjusted:,.2f}",
            f"  Contagion Sharpe:  {p.contagion_adjusted_sharpe:.3f}",
            "",
            divider,
            "  RISK-ADJUSTED METRICS",
            divider,
        ]

        _na = "n/a"
        def _fmt(v, fmt=".4f"):
            return format(v, fmt) if v != 0.0 else _na
        lines += [
            f"  CoV (copula)    :  {_fmt(p.coefficient_of_variation)}",
            f"  RAROC (ratio)   :  {_fmt(p.profile_raroc)}",
            f"  Sortino (copula):  {_fmt(p.sortino_copula)}",
        ]

        lines += [
            "",
            divider,
            "  COMPOSITE ASSESSMENT",
            divider,
            f"  Composite Score :  {p.composite_risk_score:.4f}",
            f"  Risk Tier       :  {p.risk_tier.upper()}",
            f"  Recommended     :  {p.recommended_action}",
            "",
            divider,
            "  NARRATIVE",
            divider,
            "\n".join(
                "  " + line
                for line in _wrap(p.narrative, width=56)
            ),
            f"\n{'═'*60}\n",
        ]
        return "\n".join(lines)

    def all_profiles(self, max_workers: int = 1) -> pd.DataFrame:
        """
        Build profiles for all borrowers and return as a DataFrame.

        For large portfolios this is the batch output used for dashboards.
        """
        self._check_fitted()
        rows = []
        for pid in self._persons["person_id"].values:
            try:
                p = self.get_profile(int(pid))
                d = p.to_dict()
                d.pop("top_neighbours", None)
                d.pop("narrative", None)
                rows.append(d)
            except Exception as e:
                logger.warning("Profile failed for %d: %s", pid, e)
        return pd.DataFrame(rows)

    def watchlist(
        self,
        tiers: Optional[List[str]] = None,
        include_early_warnings: bool = True,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Return the top-N highest-risk borrowers as a prioritised watchlist.

        Parameters
        ----------
        tiers  : filter to specific risk tiers (default: ['critical', 'high'])
        include_early_warnings : include any early-warning flagged borrowers
        top_n  : maximum rows to return
        """
        if tiers is None:
            tiers = ["critical", "high"]

        if self._individual_risks is not None:
            df = self._individual_risks.copy()
        else:
            df = self.all_profiles()

        # Apply tier filter
        in_tier = df["risk_tier"].isin(tiers) if "risk_tier" in df.columns else pd.Series(True, index=df.index)

        # Include early warnings from structural divergence
        ew_flag = pd.Series(False, index=df.index)
        if include_early_warnings and self._structural_df is not None:
            ew_pids = set(
                self._structural_df[
                    self._structural_df.get("pd_signal_divergence", pd.Series(0)) >=
                    self.early_warning_threshold
                ]["person_id"].values
            )
            ew_flag = df["person_id"].isin(ew_pids)

        mask = in_tier | ew_flag
        result = df[mask].copy()

        sort_col = "composite_risk_score" if "composite_risk_score" in result.columns else "marginal_pd"
        result = result.sort_values(sort_col, ascending=False).head(top_n)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 56) -> List[str]:
    """Naive word-wrap."""
    words = text.split()
    lines, current = [], []
    length = 0
    for w in words:
        if length + len(w) + 1 > width:
            lines.append(" ".join(current))
            current, length = [w], len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        lines.append(" ".join(current))
    return lines


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np; np.random.seed(42)

    from src.data_generator import generate_network
    from src.graph_features import TransactionGraph
    from src.copula_model import CopulaDefaultModel
    from src.risk_metrics import RiskAnalyzer
    from src.pd_model import IndividualPDModel
    from src.rating_engine import RatingEngine
    from src.structural_pd import StructuralPDModel
    from src.client_value_metrics import ClientValueCalculator

    print("Building full pipeline...")
    persons, transactions = generate_network(seed=42)

    # PD model
    pd_model = IndividualPDModel("gradient_boosting")
    pd_model.fit(persons, "default")
    persons["model_pd"] = pd_model.predict_proba(persons)

    # Graph + copula
    graph = TransactionGraph(transactions, persons)
    corr  = graph.get_correlation_matrix()
    copula = CopulaDefaultModel("clayton")
    copula.fit(persons["model_pd"].values, corr)

    # Risk analyzer
    exposures = persons["income"].values / persons["income"].mean()
    analyzer = RiskAnalyzer(copula, graph, persons, exposures=exposures, lgd=0.45)
    individual_risks = analyzer.compute_individual_risks()

    # Rating engine
    engine = RatingEngine()
    engine.fit(persons, "model_pd")

    # Structural PD
    struct_model = StructuralPDModel(alpha=0.35)
    persons_enriched = struct_model.fit_transform(persons, "model_pd")

    # Client value
    cvc = ClientValueCalculator(copula, persons, transactions, lgd=0.45)

    # Profiler
    profiler = CustomerProfiler(early_warning_threshold=0.05)
    profiler.fit(
        persons=persons_enriched,
        transactions=transactions,
        graph=graph,
        copula=copula,
        pd_model=pd_model,
        rating_engine=engine,
        structural_model=persons_enriched,
        client_value_calc=cvc,
        individual_risks=individual_risks,
    )

    # Print two sample profiles
    sample_ids = individual_risks.head(3)["person_id"].tolist()
    for pid in sample_ids:
        print(profiler.profile_report(pid))
