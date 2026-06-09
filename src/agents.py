"""
Agent-Facing API  (src/agents.py)
==================================
A single, safe, validated entry point for AI agents interacting with the
copula default-graph risk framework.

PURPOSE
-------
AI agents should use `RiskAgentAPI` rather than calling individual modules
directly. This layer:
  1. Enforces correct call order (pipeline steps cannot be called before
     their prerequisites).
  2. Returns structured, typed result objects — no raw DataFrames or numpy
     arrays leaking out, reducing hallucination risk.
  3. Provides human-readable `.summary` strings on every result so agents
     can relay findings to users without further processing.
  4. Collects `.warnings` on every call so agents know when results are
     degraded (e.g., falling back to EAD proxy, negative numerator).
  5. Validates all inputs before any computation starts.

USAGE
-----
    from src.agents import RiskAgentAPI

    api = RiskAgentAPI()          # synthetic data, default config
    api.run_pipeline()            # runs all 13 steps

    # Query anything after run_pipeline():
    r = api.query_borrower(776)
    print(r.summary)
    print(r.data)                 # dict with all profile fields

    r = api.segment_metrics("city_name")
    print(r.data)                 # list of dicts, one per segment

    r = api.flag_divergences(z_threshold=1.5)
    print(r.summary)              # "73 hidden_network_risk / 79 diversified_low_value"

    r = api.run_stress(pd_multiplier=2.0, correlation_boost=0.2)
    print(r.data["change"]["expected_loss"])   # fractional change

    r = api.rank_metrics()
    print(r.data)                 # Spearman correlation dict of dicts

IMPORTANT INVARIANTS (see AGENTS.md for the full list)
-------------------------------------------------------
  - Every method raises AgentError (not KeyError/ValueError/AttributeError)
    with a descriptive message so agents get clean exception handling.
  - Results are always JSON-serialisable (floats, ints, strings, lists, dicts).
    np.nan is converted to None. np.ndarray is converted to list.
  - All metric values include a `numerator_negative` flag when relevant.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_warnings.filterwarnings("ignore")


# ─── result container ─────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """
    Structured return type for every RiskAgentAPI method.

    Attributes
    ----------
    ok : bool
        True if the call succeeded. False if it failed gracefully.
    data : dict | list | None
        The primary result payload. Always JSON-serialisable.
    summary : str
        One-paragraph plain-English interpretation agents can relay to users.
    warnings : list[str]
        Non-fatal issues (fallback paths, degraded data, sign-flip conditions).
    error : str | None
        Set only when ok=False. Contains the human-readable reason.
    """
    ok: bool
    data: Any
    summary: str
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class AgentError(RuntimeError):
    """
    Raised by RiskAgentAPI when a precondition is violated or input is invalid.

    Always has a descriptive message. Agents should catch this type specifically
    to distinguish framework errors from unexpected Python errors.

    Example:
        try:
            r = api.query_borrower(999)
        except AgentError as e:
            print(f"Framework error: {e}")
    """


# ─── helpers ──────────────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """Convert numpy types to JSON-serialisable Python types."""
    # bool must be checked BEFORE int: in Python bool is a subclass of int, so
    # isinstance(True, int) is True and a bare int check would coerce True -> 1.
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return [_safe(row) for row in v.to_dict(orient="records")]
    if isinstance(v, pd.Series):
        return _safe(v.to_dict())
    return v


def _df_to_list(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of JSON-safe dicts."""
    return [_safe(row) for row in df.to_dict(orient="records")]


# ─── main API class ────────────────────────────────────────────────────────────

class RiskAgentAPI:
    """
    Safe, validated, agent-friendly entry point to the risk framework.

    STATE MACHINE
    -------------
    The API tracks which pipeline steps have run. Methods that depend on
    earlier steps raise AgentError with a helpful "run X first" message
    rather than crashing with a cryptic AttributeError.

    States (checked via `self._state`):
      "empty"     → nothing done yet
      "data"      → generate_network() done
      "graph"     → TransactionGraph built
      "pd_model"  → IndividualPDModel fitted, persons['model_pd'] exists
      "copula"    → CopulaDefaultModel fitted
      "pipeline"  → full pipeline complete (all 13 steps)

    PARAMETERS
    ----------
    config : PipelineConfig, optional
        Full pipeline configuration. Defaults to DEFAULT_CONFIG.
    persons : pd.DataFrame, optional
        Pre-existing persons data. If None, synthetic data is generated.
    transactions : pd.DataFrame, optional
        Pre-existing transactions data. Required if persons is provided.
    seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        config=None,
        persons: Optional[pd.DataFrame] = None,
        transactions: Optional[pd.DataFrame] = None,
        seed: int = 42,
    ) -> None:
        # Import here to avoid circular imports and to keep this module
        # importable even if optional dependencies are missing.
        from .config import DEFAULT_CONFIG
        self._cfg = config or DEFAULT_CONFIG
        self._seed = seed
        self._state = "empty"

        # Pipeline objects (populated as steps run)
        self._persons: Optional[pd.DataFrame] = None
        self._transactions: Optional[pd.DataFrame] = None
        self._graph = None
        self._copula = None
        self._analyzer = None
        self._client_calc = None
        self._calc = None          # RiskRatioCalculator
        self._comp = None          # MetricComparator
        self._profiler = None
        self._rating_engine = None
        self._struct_model = None
        self._pd_model = None
        self._portfolio_result = None
        self._stress_result = None
        self._regime_result = None

        # Cluster-analysis objects (populated by run_cluster_analysis())
        self._geo_clusterer = None
        self._transfer_clusterer = None
        self._mfc = None              # MultiFactorCopula
        self._cluster_calc = None     # RiskRatioCalculator over the multi-factor copula
        self._cluster_metrics = None  # ClusterRiskMetrics

        if persons is not None:
            self._validate_persons(persons)
            self._persons = persons.copy()
            if transactions is None:
                raise AgentError(
                    "transactions DataFrame is required when persons is provided."
                )
            self._transactions = transactions.copy()
            self._state = "data"

    # ── validation helpers ────────────────────────────────────────────────────

    def _validate_persons(self, df: pd.DataFrame) -> None:
        required = {"person_id", "city_name", "base_pd"}
        missing = required - set(df.columns)
        if missing:
            raise AgentError(
                f"persons DataFrame is missing required columns: {sorted(missing)}. "
                f"Required columns: {sorted(required)}."
            )
        if df["person_id"].duplicated().any():
            raise AgentError("persons['person_id'] contains duplicate values. Must be unique.")
        if not df["base_pd"].between(0, 1).all():
            raise AgentError("persons['base_pd'] contains values outside [0, 1].")

    def _require_state(self, min_state: str, method_name: str) -> None:
        order = ["empty", "data", "graph", "pd_model", "copula", "pipeline"]
        current_idx = order.index(self._state)
        required_idx = order.index(min_state)
        if current_idx < required_idx:
            step_map = {
                "data":     "run_pipeline() or provide persons/transactions in __init__",
                "graph":    "run_pipeline()",
                "pd_model": "run_pipeline()",
                "copula":   "run_pipeline()",
                "pipeline": "run_pipeline()",
            }
            raise AgentError(
                f"Cannot call {method_name}() before state='{min_state}'. "
                f"Current state='{self._state}'. "
                f"Run {step_map.get(min_state, 'run_pipeline()')} first."
            )

    # ── pipeline execution ────────────────────────────────────────────────────

    def run_pipeline(self, verbose: bool = False) -> AgentResult:
        """
        Execute the full 13-step pipeline.

        This is the recommended first call. After it completes, all query
        methods are available. Runs in ~30-60 seconds on 1 000 borrowers.

        Parameters
        ----------
        verbose : bool
            If True, prints step progress to stdout.

        Returns
        -------
        AgentResult
            data: dict with pipeline summary statistics
            summary: human-readable description of what was computed

        AGENT NOTE: Run this once per session. Re-running re-fits all models
        from scratch with a new random seed (same seed → same results).
        """
        import warnings as w
        w.filterwarnings("ignore")
        np.random.seed(self._seed)
        warnings_list = []

        def _log(msg):
            if verbose:
                print(f"  {msg}")

        try:
            # Step 1: data
            if self._state == "empty":
                from .data_generator import generate_network
                _log("Step 1: generating synthetic network...")
                self._persons, self._transactions = generate_network(seed=self._seed)
                self._state = "data"

            # Step 2: graph
            from .graph_features import TransactionGraph, get_neighbor_risk_features
            _log("Step 2: building transaction graph...")
            self._graph = TransactionGraph(self._transactions, self._persons)
            neighbor_features = get_neighbor_risk_features(self._graph, self._persons)
            self._persons = self._persons.merge(
                neighbor_features[["person_id", "neighbor_pd_avg",
                                   "neighbor_pd_max", "n_high_risk_neighbors"]],
                on="person_id", how="left"
            ).fillna(0)
            self._state = "graph"

            # Step 3: PD model
            from .pd_model import IndividualPDModel
            _log("Step 3: fitting PD model...")
            self._pd_model = IndividualPDModel(
                model_type="gradient_boosting",
                feature_columns=[
                    "age", "income", "employment_years", "debt_to_income",
                    "num_credit_lines", "missed_payments", "credit_utilization",
                    "account_age_months",
                ],
            )
            pd_metrics = self._pd_model.fit(self._persons, target_col="default",
                                             validation_split=0.2)
            self._persons["model_pd"] = self._pd_model.predict_proba(self._persons)
            self._state = "pd_model"

            # Step 4+5: correlation matrix + copula
            from .copula_model import CopulaDefaultModel
            _log("Step 4-5: building correlation matrix and fitting copula...")
            corr_matrix = self._graph.get_correlation_matrix(
                base_corr=self._cfg.copula.base_correlation,
                max_corr=self._cfg.copula.max_correlation,
                same_city_boost=self._cfg.copula.same_city_boost,
                same_group_boost=self._cfg.copula.same_group_boost,
            )
            self._copula = CopulaDefaultModel(self._cfg.copula.copula_type)
            self._copula.fit(self._persons["model_pd"].values, corr_matrix)
            self._state = "copula"

            # Step 6: risk analysis
            from .risk_metrics import RiskAnalyzer
            _log("Step 6: computing portfolio risk...")
            exposures = self._persons["income"].values / self._persons["income"].mean()
            self._analyzer = RiskAnalyzer(
                self._copula, self._graph, self._persons,
                exposures=exposures, lgd=self._cfg.risk.lgd
            )
            individual_risks = self._analyzer.compute_individual_risks()
            self._portfolio_result = self._analyzer.compute_portfolio_risks(
                n_simulations=self._cfg.copula.default_n_simulations
            )

            # Step 7: stress test (cached, lazy)
            self._stress_result = self._analyzer.stress_test(
                pd_multiplier=self._cfg.risk.default_pd_multiplier,
                correlation_boost=self._cfg.risk.default_correlation_boost,
            )

            # Step 8: client value metrics
            from .client_value_metrics import ClientValueCalculator
            _log("Step 8: computing client value metrics...")
            self._client_calc = ClientValueCalculator(
                self._copula, self._persons, self._transactions,
                lgd=self._cfg.risk.lgd
            )
            self._client_calc.compute_contagion_adjusted_sharpe()

            # Step 8b: risk-adjusted metric family
            from .risk_adjusted_metrics import RiskRatioCalculator
            from .metric_comparison import MetricComparator
            _log("Step 8b: computing risk-adjusted metric family...")
            _persons_enriched = self._client_calc.persons
            _ead = _persons_enriched["exposure_at_default"].values
            try:
                self._calc = RiskRatioCalculator(
                    self._copula, _persons_enriched,
                    exposures=_ead,
                    lgd=self._cfg.risk.lgd,
                    hurdle_rate=self._cfg.risk.hurdle_rate,
                    risk_free_rate=self._cfg.risk.risk_free_rate,
                    capital_ratio=self._cfg.risk.capital_ratio,
                )
                self._comp = MetricComparator(self._calc)
            except Exception as e:
                warnings_list.append(f"RiskRatioCalculator failed: {e}. Metric queries disabled.")

            # Step 9: rating engine
            from .rating_engine import RatingEngine
            _log("Step 9: fitting rating engine...")
            self._rating_engine = RatingEngine()
            self._rating_engine.fit(self._persons, pd_col="model_pd")

            # Step 10: Merton structural PD
            from .structural_pd import StructuralPDModel
            _log("Step 10: computing Merton structural PD...")
            struct = StructuralPDModel(alpha=0.35, T=1.0, r=0.02)
            self._persons = struct.fit_transform(self._persons,
                                                  statistical_pd_col="model_pd")
            self._struct_model = struct

            # Step 11: flexible probabilities
            from .flexible_probs import build_calibrator_from_portfolio
            _log("Step 11: calibrating regime-aware copula...")
            t_hist = 36
            avg_pd_history = (
                self._persons["model_pd"].mean()
                * (0.5 + np.linspace(0, 1, t_hist))
                + 0.01 * np.random.randn(t_hist)
            )
            calib = build_calibrator_from_portfolio(avg_pd_history, half_life_periods=12.0)
            current_stress = float(np.clip(
                (self._persons["model_pd"].mean() - 0.05) / 0.20, 0.0, 1.0
            ))
            self._regime_result = calib.calibrate(current_stress, corr_matrix)

            # Step 12: customer profiles
            from .customer_profile import CustomerProfiler
            _log("Step 12: building customer profiles...")
            self._profiler = CustomerProfiler(early_warning_threshold=0.05)
            _rrCalc = self._calc
            self._profiler.fit(
                persons=self._persons,
                transactions=self._transactions,
                graph=self._graph,
                copula=self._copula,
                pd_model=self._pd_model,
                rating_engine=self._rating_engine,
                structural_model=self._persons,
                client_value_calc=self._client_calc,
                individual_risks=individual_risks,
                risk_ratio_calc=_rrCalc,
            )

            self._state = "pipeline"

            # Build summary
            n = len(self._persons)
            el = self._portfolio_result.expected_loss
            var95 = self._portfolio_result.var_95
            theta = self._copula.params.theta
            ltd = self._copula.tail_dependence()
            auc = pd_metrics.get("val_auc", float("nan"))

            n_neg = int((self._calc.eprofit < 0).sum()) if self._calc else "n/a"
            summary = (
                f"Pipeline complete. Portfolio: {n} borrowers. "
                f"PD model val AUC={auc:.4f}. "
                f"Clayton θ={theta:.4f}, lower-tail dep={ltd:.4f}. "
                f"E[Loss]={el:.4f}, VaR95={var95:.4f}. "
                f"Borrowers with negative expected profit: {n_neg}/{n}."
            )
            if warnings_list:
                summary += " WARNINGS: " + "; ".join(warnings_list)

            return AgentResult(
                ok=True,
                data=_safe({
                    "n_borrowers": n,
                    "pd_model_val_auc": auc,
                    "copula_theta": theta,
                    "lower_tail_dependence": ltd,
                    "expected_loss": el,
                    "var_95": var95,
                    "es_95": self._portfolio_result.es_95,
                    "n_negative_profit": n_neg,
                }),
                summary=summary,
                warnings=warnings_list,
            )

        except AgentError:
            raise
        except Exception as exc:
            return AgentResult(
                ok=False,
                data=None,
                summary=f"Pipeline failed at state='{self._state}': {exc}",
                error=str(exc),
            )

    # ── query methods ─────────────────────────────────────────────────────────

    def query_borrower(self, person_id: int) -> AgentResult:
        """
        Return the full risk profile for a single borrower.

        Parameters
        ----------
        person_id : int
            The person_id to look up. Must exist in the persons DataFrame.

        Returns
        -------
        AgentResult
            data: dict with all CustomerRiskProfile fields plus risk-adjusted metrics.
            summary: one-paragraph narrative from the profiler.

        AGENT NOTE: Use `api.query_borrower(person_id)` rather than directly
        calling profiler.profile_report(). This method validates the ID and
        returns structured data; the profiler returns a formatted string.
        """
        self._require_state("pipeline", "query_borrower")
        if not isinstance(person_id, (int, np.integer)):
            raise AgentError(f"person_id must be an integer, got {type(person_id).__name__}.")

        ids = self._persons["person_id"].values
        if person_id not in ids:
            raise AgentError(
                f"person_id={person_id} not found. "
                f"Valid range: {int(ids.min())}–{int(ids.max())}."
            )

        warnings_list = []
        profile = self._profiler.get_profile(int(person_id))

        # Also pull risk-adjusted metrics for this borrower
        metric_data = {}
        if self._calc is not None:
            try:
                idx_arr = np.where(ids == person_id)[0]
                inp = self._calc._inputs_for(idx_arr)
                from .risk_adjusted_metrics import available_metrics, compute_metric
                for m in available_metrics():
                    if m != "sortino_simulated":
                        try:
                            metric_data[m] = _safe(compute_metric(m, inp))
                        except Exception:
                            metric_data[m] = None
                metric_data["numerator_negative"] = bool(
                    inp.expected_profit - inp.hurdle_rate * inp.capital < 0
                )
                if metric_data.get("numerator_negative"):
                    warnings_list.append(
                        "Signed metrics (RAROC, Sortino) may be misleading: "
                        "E[Profit] < hurdle*Capital. Use CoV for riskiness ranking."
                    )
            except Exception as e:
                warnings_list.append(f"Risk-adjusted metrics unavailable: {e}")

        # Rating profile
        rating_data = {}
        if self._rating_engine is not None:
            try:
                rp = self._rating_engine.get_rating_profile(int(person_id))
                rating_data = _safe({
                    "rating": rp.current_rating_label,
                    "upgrade_prob": rp.upgrade_prob,
                    "downgrade_prob": rp.downgrade_prob,
                    "default_1yr": rp.default_1yr,
                    "default_3yr": rp.default_3yr,
                })
            except Exception:
                pass

        data = {
            "person_id": int(person_id),
            "city": profile.city,
            "risk_archetype": profile.archetype,
            "statistical_pd": _safe(profile.statistical_pd),
            "merton_pd": _safe(profile.merton_pd),
            "blended_pd": _safe(profile.blended_pd),
            "pd_signal_divergence": _safe(profile.pd_signal_divergence),
            "early_warning": bool(profile.early_warning),
            "distance_to_default": _safe(profile.distance_to_default),
            "current_rating": profile.current_rating,
            "composite_risk_score": _safe(profile.composite_risk_score),
            "risk_tier": profile.risk_tier,
            "recommended_action": profile.recommended_action,
            "contagion_vulnerability": _safe(profile.contagion_vulnerability),
            "systemic_importance": _safe(profile.systemic_importance),
            "n_connections": profile.n_connections,
            "client_sharpe": _safe(profile.client_sharpe),
            "raroc": _safe(profile.profile_raroc),
            "segment": profile.segment,
            "narrative": profile.narrative,
            "rating": rating_data,
            "risk_adjusted_metrics": metric_data,
        }

        summary = (
            f"Borrower {person_id} in {profile.city} ({profile.archetype}): "
            f"PD={profile.statistical_pd:.3f}, rating={profile.current_rating}, "
            f"tier={profile.risk_tier}. "
            f"Action: {profile.recommended_action}. "
            f"{profile.narrative[:180]}..."
        )

        return AgentResult(ok=True, data=data, summary=summary, warnings=warnings_list)

    def segment_metrics(
        self,
        segment_col: str,
        metrics: Optional[List[str]] = None,
    ) -> AgentResult:
        """
        Return risk-adjusted metrics aggregated by any persons column.

        Uses the mathematically correct block-sum aggregation (never averages
        per-borrower ratios). See AGENTS.md INV-6.

        Parameters
        ----------
        segment_col : str
            Column to group by. Must exist in persons DataFrame.
            Typical values: 'city_name', 'risk_archetype', 'high_risk_group_id'.
        metrics : list[str], optional
            Subset of metric names. Defaults to all registered metrics.
            Use `api.available_metrics()` to list valid names.

        Returns
        -------
        AgentResult
            data: list of dicts, one per segment, with columns:
                  segment, n, exposure, exposure_share, expected_profit,
                  expected_loss, loss_std_indep, loss_std_copula,
                  diversification_ratio, <metric columns>, numerator_negative
            summary: which segments are best/worst by copula-Sortino.

        AGENT NOTE: diversification_ratio ≥ 1 always. A value near 1.0 means
        the segment is highly correlated internally (concentration risk).
        Values of 5–6 indicate healthy diversification for typical retail books.
        """
        self._require_state("pipeline", "segment_metrics")
        if segment_col not in self._persons.columns:
            valid = [c for c in self._persons.columns if self._persons[c].dtype in
                     ("object", "int64", "int32") and self._persons[c].nunique() < 50]
            raise AgentError(
                f"'{segment_col}' not found in persons columns. "
                f"Suitable columns: {valid}"
            )
        if self._calc is None:
            raise AgentError("RiskRatioCalculator not available. run_pipeline() may have failed.")

        df = self._calc.by_segment(segment_col, metrics=metrics)
        data = _df_to_list(df)

        # Find best and worst by sortino_copula
        warnings_list = []
        summary_parts = [f"Metrics by {segment_col} ({len(data)} segments):"]
        if "sortino_copula" in df.columns and not df["sortino_copula"].isna().all():
            valid = df["sortino_copula"].dropna()
            if len(valid):
                best_idx = valid.idxmax()
                worst_idx = valid.idxmin()
                best_seg = df.loc[best_idx, "segment"]
                worst_seg = df.loc[worst_idx, "segment"]
                summary_parts.append(
                    f"Best by Sortino_L1: {best_seg} ({valid[best_idx]:.3f}). "
                    f"Worst: {worst_seg} ({valid[worst_idx]:.3f})."
                )
        neg_segs = df[df.get("numerator_negative", pd.Series(False))]["segment"].tolist() \
            if "numerator_negative" in df.columns else []
        if neg_segs:
            warnings_list.append(
                f"Segments with negative expected profit (signed metrics misleading): {neg_segs}. "
                f"Use coefficient_of_variation_copula for riskiness ranking instead."
            )

        return AgentResult(
            ok=True,
            data=data,
            summary=" ".join(summary_parts),
            warnings=warnings_list,
        )

    def flag_divergences(self, z_threshold: float = 1.5) -> AgentResult:
        """
        Return borrowers where RAROC and Sortino_copula strongly diverge.

        This is the primary early-warning deliverable. RAROC is correlation-blind
        (capital = k·EAD). Sortino_copula inflates its denominator for borrowers
        embedded in high-correlation clusters. Divergence = hidden risk.

        Parameters
        ----------
        z_threshold : float
            Flag borrowers whose rank-gap z-score exceeds this threshold.
            Default 1.5 (moderate). Use 2.0 for high-confidence flags only.
            Lower values (1.0) produce more flags but more false positives.

        Returns
        -------
        AgentResult
            data: list of dicts sorted by |z_score| descending. Each dict has:
                  person_id, raroc, sortino_copula,
                  raroc_rank, sortino_rank, rank_gap, z_score, flag_type
            summary: counts of each flag type with interpretation.

        Flag types:
            "hidden_network_risk"   — good RAROC, bad Sortino.
                                      Reduce exposure or add collateral.
            "diversified_low_value" — bad RAROC, good Sortino.
                                      Retain for portfolio diversification benefit.
        """
        self._require_state("pipeline", "flag_divergences")
        if self._comp is None:
            raise AgentError("MetricComparator not available. run_pipeline() may have failed.")
        if not isinstance(z_threshold, (int, float)) or z_threshold <= 0:
            raise AgentError(f"z_threshold must be a positive number, got {z_threshold}.")

        flags = self._comp.divergence_flags(z_threshold=z_threshold)
        n_total = len(flags)
        n_hidden = int((flags["flag_type"] == "hidden_network_risk").sum()) if n_total else 0
        n_div = int((flags["flag_type"] == "diversified_low_value").sum()) if n_total else 0

        summary = (
            f"{n_total} borrowers flagged at z≥{z_threshold}: "
            f"{n_hidden} hidden_network_risk (good RAROC, bad Sortino — "
            f"reduce exposure or add collateral); "
            f"{n_div} diversified_low_value (bad RAROC, good Sortino — "
            f"retain for diversification benefit)."
        )

        warnings_list = []
        if n_total == 0:
            warnings_list.append(
                f"No flags found at z≥{z_threshold}. "
                f"Try lowering z_threshold (e.g. 1.0) or check that "
                f"run_pipeline() completed successfully."
            )

        return AgentResult(
            ok=True,
            data=_df_to_list(flags),
            summary=summary,
            warnings=warnings_list,
        )

    def rank_metrics(self, level: str = "borrower") -> AgentResult:
        """
        Return the Spearman rank-correlation matrix between all metrics.

        Use this to decide which metrics are redundant (ρ ≈ 1.0) vs which
        encode genuinely different risk signals (ρ < 0.7).

        Parameters
        ----------
        level : str
            'borrower' (default, 1000 data points, most informative) or
            'segment' (requires segment_col — use segment_metrics() instead).

        Returns
        -------
        AgentResult
            data: dict of dicts — data[metric_a][metric_b] = spearman_rho.
                  All values in [-1, 1]. Diagonal = 1.0.
            summary: pairs with low correlation (< 0.7) i.e. unique information.

        Interpretation:
            CoV vs RAROC ≈ 0.48  → encode different signals (keep both)
            Sharpe vs Sortino ≈ 0.997 → nearly redundant at borrower level
        """
        self._require_state("pipeline", "rank_metrics")
        if self._comp is None:
            raise AgentError("MetricComparator not available.")
        if level not in ("borrower", "segment"):
            raise AgentError(f"level must be 'borrower' or 'segment', got '{level}'.")

        rc = self._comp.rank_correlation(level=level)
        data = _safe(rc.to_dict())

        # Find pairs with genuinely different information (low correlation)
        low_corr_pairs = []
        cols = list(rc.columns)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if j > i:
                    v = float(rc.loc[a, b])
                    if np.isfinite(v) and abs(v) < 0.7:
                        low_corr_pairs.append(f"{a} vs {b}: {v:+.3f}")

        summary = (
            f"Rank correlation at {level} level. "
            + (f"Pairs with unique information (|ρ|<0.7): {'; '.join(low_corr_pairs)}."
               if low_corr_pairs
               else "All metric pairs are highly correlated — consider removing redundant ones.")
        )

        return AgentResult(ok=True, data=data, summary=summary)

    def run_stress(
        self,
        pd_multiplier: float = 2.0,
        correlation_boost: float = 0.20,
    ) -> AgentResult:
        """
        Run a stress scenario: multiply all PDs and boost correlations.

        Uses the `_stressed_copula()` context manager which restores the
        copula to its base state after the test — no permanent mutation.

        Parameters
        ----------
        pd_multiplier : float
            Factor to multiply all PDs by (default 2.0 = severe recession).
            Must be > 0. Values > 5 may push PDs above 1 (clamped to 1).
        correlation_boost : float
            Additive boost to all pairwise correlations (default 0.20).
            Must be in [0, 0.5]. Values > 0.5 force near-comonotone behaviour.

        Returns
        -------
        AgentResult
            data: dict with keys 'base', 'stressed', 'change' — each a dict
                  with 'expected_loss', 'var_95', 'es_95'.
            summary: percentage changes for each metric.

        AGENT NOTE: The 'change' sub-dict contains fractional (not percentage)
        changes. Multiply by 100 for percentage. A change of 0.47 = +47%.
        """
        self._require_state("pipeline", "run_stress")
        if pd_multiplier <= 0:
            raise AgentError(f"pd_multiplier must be positive, got {pd_multiplier}.")
        if not 0 <= correlation_boost <= 0.5:
            raise AgentError(f"correlation_boost must be in [0, 0.5], got {correlation_boost}.")

        result = self._analyzer.stress_test(
            pd_multiplier=pd_multiplier,
            correlation_boost=correlation_boost,
        )
        data = _safe(result)

        summary_parts = []
        for metric in ["expected_loss", "var_95", "es_95"]:
            chg = result["change"].get(metric, float("nan"))
            if np.isfinite(chg):
                summary_parts.append(f"{metric}: {chg:+.1%}")

        summary = (
            f"Stress scenario (PD×{pd_multiplier}, corr+{correlation_boost:.0%}): "
            + ", ".join(summary_parts) + "."
        )

        warnings_list = []
        el_chg = result["change"].get("expected_loss", 0)
        if el_chg > 1.0:
            warnings_list.append(
                f"Expected loss more than doubled under stress (+{el_chg:.0%}). "
                f"Review correlation concentration (HHI) in portfolio."
            )

        return AgentResult(ok=True, data=data, summary=summary, warnings=warnings_list)

    def query_segment(self, segment_col: str, segment_value: Any) -> AgentResult:
        """
        Return all metrics for a specific segment value.

        Convenience wrapper around segment_metrics() that filters to one row.

        Parameters
        ----------
        segment_col : str
            Column to filter on (e.g. 'city_name').
        segment_value : any
            Value to match (e.g. 'Gamma').

        Returns
        -------
        AgentResult
            data: dict with all metric fields for the requested segment.

        Example:
            r = api.query_segment("city_name", "Gamma")
            print(r.data["sortino_copula"])   # Gamma's copula-Sortino ratio
        """
        r = self.segment_metrics(segment_col)
        if not r.ok:
            return r
        matches = [row for row in r.data if str(row.get("segment")) == str(segment_value)]
        if not matches:
            all_segs = [row.get("segment") for row in r.data]
            raise AgentError(
                f"Segment '{segment_value}' not found in column '{segment_col}'. "
                f"Available: {all_segs}."
            )
        data = matches[0]
        tier = "loss-making" if data.get("numerator_negative") else "profitable"
        summary = (
            f"Segment '{segment_value}' ({segment_col}): "
            f"n={data.get('n')}, exposure_share={data.get('exposure_share', 0):.1%}, "
            f"RAROC={data.get('raroc', 'n/a')}, "
            f"Sortino_L1={data.get('sortino_copula', 'n/a')}, "
            f"CoV_L1={data.get('coefficient_of_variation_copula', 'n/a')}, "
            f"div_ratio={data.get('diversification_ratio', 'n/a')}, "
            f"status={tier}."
        )
        return AgentResult(ok=True, data=data, summary=summary, warnings=r.warnings)

    def top_risks(self, n: int = 20, sort_by: str = "composite_risk_score") -> AgentResult:
        """
        Return the top-N riskiest borrowers.

        Parameters
        ----------
        n : int
            Number of borrowers to return (default 20, max 200).
        sort_by : str
            Column to rank by. Options:
              'composite_risk_score' (default) — combines PD, network, contagion
              'marginal_pd'                    — pure PD rank
              'contagion_vulnerability'        — most affected if neighbours default
              'systemic_importance'            — most dangerous if this one defaults

        Returns
        -------
        AgentResult
            data: list of dicts with person_id, city_name, risk_archetype,
                  marginal_pd, composite_risk_score, risk_tier, etc.
        """
        self._require_state("pipeline", "top_risks")
        valid_sorts = {"composite_risk_score", "marginal_pd",
                       "contagion_vulnerability", "systemic_importance"}
        if sort_by not in valid_sorts:
            raise AgentError(
                f"sort_by='{sort_by}' not recognised. "
                f"Valid options: {sorted(valid_sorts)}."
            )
        if not 1 <= n <= 200:
            raise AgentError(f"n must be between 1 and 200, got {n}.")

        individual_risks = self._analyzer.compute_individual_risks()
        top = (individual_risks
               .sort_values(sort_by, ascending=False)
               .head(n)
               [["person_id", "city_name", "risk_archetype",
                 "marginal_pd", "contagion_vulnerability",
                 "systemic_importance", "composite_risk_score", "risk_tier"]])

        data = _df_to_list(top)
        n_critical = sum(1 for r in data if r.get("risk_tier") == "critical")
        summary = (
            f"Top {n} riskiest borrowers by {sort_by}: "
            f"{n_critical} critical tier, "
            f"avg PD={np.mean([r.get('marginal_pd', 0) for r in data]):.3f}."
        )

        return AgentResult(ok=True, data=data, summary=summary)

    def available_metrics(self) -> AgentResult:
        """
        Return the list of all registered metric names.

        Returns
        -------
        AgentResult
            data: list of metric name strings.
            summary: brief description of each metric.

        Use this to discover what names are valid for segment_metrics(metrics=[...]).
        """
        from .risk_adjusted_metrics import available_metrics as _av
        names = _av()
        descriptions = {
            "coefficient_of_variation":        "σ_L0 / E[Loss]. Pure riskiness (independence). Always ≥ 0.",
            "coefficient_of_variation_copula": "σ_L1 / E[Loss]. Copula-aware riskiness. Inflates for contagious clusters.",
            "raroc":                           "E[Profit] / Capital. Profitability per unit capital. Correlation-blind.",
            "sharpe_indep":                    "(E[Profit] − rf·Revenue) / σ_L0. Risk-free opportunity cost benchmark.",
            "sortino_indep":                   "(E[Profit] − h·Capital) / σ_L0. Hurdle-rate benchmark, independence.",
            "sortino_copula":                  "(E[Profit] − h·Capital) / σ_L1. PRIMARY METRIC. Copula-aware.",
            "sortino_simulated":               "(E[Profit] − h·Capital) / σ_L2. Monte Carlo tail. Expensive.",
        }
        data = [{"name": n, "description": descriptions.get(n, "—")} for n in names]
        summary = (
            f"{len(names)} registered metrics: {', '.join(names)}. "
            "Use 'sortino_copula' as primary. Use 'coefficient_of_variation_copula' "
            "when profit is negative."
        )
        return AgentResult(ok=True, data=data, summary=summary)

    def regime_status(self) -> AgentResult:
        """
        Return the current portfolio stress regime and regime-adjusted copula theta.

        The flexible-probability calibrator computes a kernel-weighted average
        of historical scenarios most similar to the current macro state. The
        result is a theta that automatically tightens in stress and loosens in calm.

        Returns
        -------
        AgentResult
            data: dict with regime_label, stress_level, base_theta,
                  regime_theta, tighter (bool).
        """
        self._require_state("pipeline", "regime_status")
        if self._regime_result is None:
            raise AgentError("Regime calibration not available.")

        regime = self._regime_result
        base_theta = float(self._copula.params.theta)
        regime_theta = float(regime.theta)
        tighter = regime_theta > base_theta

        data = _safe({
            "regime_label": regime.regime.label,
            "stress_level": float(np.clip(
                (self._persons["model_pd"].mean() - 0.05) / 0.20, 0.0, 1.0
            )),
            "base_theta": base_theta,
            "regime_theta": regime_theta,
            "tighter": tighter,
        })

        summary = (
            f"Regime: {regime.regime.label}. "
            f"Base θ={base_theta:.4f} → regime-adjusted θ={regime_theta:.4f} "
            f"({'tighter — use regime theta for ICAAP' if tighter else 'looser — base theta conservative'})."
        )

        return AgentResult(ok=True, data=data, summary=summary)

    def portfolio_summary(self) -> AgentResult:
        """
        Return a compact summary of full portfolio risk metrics.

        Returns
        -------
        AgentResult
            data: dict with expected_loss, var_95, var_99, es_95, es_99,
                  default_correlation, concentration_hhi, tail_risk_ratio.
        """
        self._require_state("pipeline", "portfolio_summary")
        p = self._portfolio_result
        data = _safe({
            "expected_loss": p.expected_loss,
            "var_95": p.var_95,
            "var_99": p.var_99,
            "es_95": p.es_95,
            "es_99": p.es_99,
            "default_correlation": p.default_correlation,
            "concentration_hhi": p.concentration_index,
            "tail_risk_ratio": p.tail_risk_ratio,
        })
        summary = (
            f"Portfolio: E[Loss]={p.expected_loss:.4f}, "
            f"VaR95={p.var_95:.4f}, ES95={p.es_95:.4f}, "
            f"default_corr={p.default_correlation:.4f}, "
            f"HHI={p.concentration_index:.6f}, "
            f"tail_ratio(ES/VaR)={p.tail_risk_ratio:.4f}."
        )
        return AgentResult(ok=True, data=data, summary=summary)

    # ── multi-dimensional clusters & anchors ──────────────────────────────────
    #
    # These methods add the geo + transfer-graph cluster layer on top of the
    # base pipeline. They are independent of the 13-step run_pipeline() and only
    # require that PDs exist (state >= 'pd_model'); call run_cluster_analysis()
    # once, then query with the other cluster_* methods.

    def run_cluster_analysis(
        self,
        eps_km: float = 8.0,
        min_samples: int = 5,
        resolution: float = 1.0,
        min_cluster_size: int = 4,
        beta_geo: float = 0.30,
        beta_transfer: float = 0.30,
    ) -> AgentResult:
        """
        Build multi-dimensional risk clusters and the anchor/dependent structure.

        Pipeline: geo clusters (DBSCAN on lat/lon, falls back to city if no
        coordinates) → transfer communities (Louvain) + anchor detection
        (якорный человек) → multi-factor copula (geo ⟂ transfer, equal loadings)
        → cluster risk metrics + anchor-contagion uplift.

        Requires state >= 'pd_model'. Idempotent — safe to re-run with new knobs.

        Returns AgentResult.data = {
            n_geo_clusters, n_transfer_clusters, n_anchors,
            variance_inflation_vs_independence, top_anchor (or None)
        }
        """
        self._require_state("pd_model", "run_cluster_analysis")
        warnings: List[str] = []
        try:
            import numpy as _np
            from .geo_clusters import GeoClusterer, GeoClusterConfig
            from .transfer_clusters import TransferClusterer, TransferClusterConfig
            from .multi_factor_copula import MultiFactorCopula
            from .risk_adjusted_metrics import RiskRatioCalculator
            from .cluster_metrics import ClusterRiskMetrics

            persons = self._persons.copy()
            pd_col = "model_pd" if "model_pd" in persons.columns else "base_pd"

            # 1) geo clusters
            gc = GeoClusterer(
                GeoClusterConfig(eps_km=eps_km, min_samples=min_samples)
            ).fit(persons)
            persons = gc.assign(persons)
            if gc.method_ != "dbscan":
                warnings.append(
                    f"No usable geo coordinates — geo clusters fell back to "
                    f"'{gc.method_}' (city grouping)."
                )

            # 2) transfer communities + anchors
            tc = TransferClusterer(
                TransferClusterConfig(
                    resolution=resolution, min_cluster_size=min_cluster_size
                )
            ).fit(persons, self._transactions)
            persons = tc.assign(persons)

            # 3) multi-factor copula (geo + transfer, equal loadings)
            pds = persons[pd_col].to_numpy()
            factor_matrix = persons[
                ["geo_cluster_id", "transfer_cluster_id"]
            ].to_numpy()
            mfc = MultiFactorCopula().fit(
                pds, factor_matrix, betas=[beta_geo, beta_transfer]
            )

            # 4) cluster metrics
            if "exposure_at_default" in persons.columns:
                ead = persons["exposure_at_default"].to_numpy()
            elif "income" in persons.columns:
                ead = persons["income"].to_numpy()
            else:
                ead = _np.full(len(persons), 10000.0)
                warnings.append("No EAD/income column — using flat EAD=10000.")
            calc = RiskRatioCalculator(
                mfc, persons, exposures=ead, lgd=self._cfg.risk.lgd
            )
            crm = ClusterRiskMetrics(calc, persons)

            # stash for the query methods
            self._persons = persons
            self._geo_clusterer = gc
            self._transfer_clusterer = tc
            self._mfc = mfc
            self._cluster_calc = calc
            self._cluster_metrics = crm

            dr = mfc.simulate_default_rate(2000, seed=self._seed)
            indep_std = _np.sqrt((pds * (1 - pds)).sum()) / len(pds)
            infl = float((dr.std() / indep_std) ** 2) if indep_std > 0 else float("nan")

            anchors = tc.anchors_table()
            n_geo = int(persons.loc[persons["geo_cluster_id"] >= 0,
                                    "geo_cluster_id"].nunique())
            n_tx = int(persons.loc[persons["transfer_cluster_id"] >= 0,
                                   "transfer_cluster_id"].nunique())
            top_anchor = None
            if len(anchors):
                a0 = anchors.iloc[0]
                top_anchor = {
                    "anchor_person_id": int(a0["anchor_person_id"]),
                    "transfer_cluster_id": int(a0["transfer_cluster_id"]),
                    "n_dependents": int(a0["n_dependents"]),
                    "cluster_fragility": float(a0["cluster_fragility"]),
                }

            data = {
                "n_geo_clusters": n_geo,
                "n_transfer_clusters": n_tx,
                "n_anchors": int(len(anchors)),
                "variance_inflation_vs_independence": round(infl, 1),
                "top_anchor": top_anchor,
            }
            summary = (
                f"Built {n_geo} geo clusters and {n_tx} transfer communities; "
                f"found {len(anchors)} anchor(s). The multi-factor copula "
                f"(geo + transfer) inflates portfolio loss variance "
                f"{infl:.0f}× over the independence assumption."
            )
            if top_anchor:
                summary += (
                    f" Most fragile anchored cluster: #{top_anchor['transfer_cluster_id']} "
                    f"(anchor person {top_anchor['anchor_person_id']}, "
                    f"{top_anchor['n_dependents']} dependents, "
                    f"fragility {top_anchor['cluster_fragility']:.2f})."
                )
            return AgentResult(ok=True, data=data, summary=summary, warnings=warnings)
        except Exception as e:  # graceful failure per the agent contract
            return AgentResult(
                ok=False, data=None, summary="Cluster analysis failed.",
                warnings=warnings, error=str(e),
            )

    def geo_clusters(self) -> AgentResult:
        """Per-geo-cluster risk metrics (EAD/EL/σ/CoV/RAROC/Sortino). Run
        run_cluster_analysis() first."""
        return self._cluster_segment_result("geo")

    def transfer_clusters(self) -> AgentResult:
        """Per-transfer-community risk metrics. Run run_cluster_analysis() first."""
        return self._cluster_segment_result("transfer")

    def anchors(self) -> AgentResult:
        """List detected anchors (якорный человек) with dependents and fragility."""
        if getattr(self, "_transfer_clusterer", None) is None:
            return AgentResult(
                ok=False, data=None, summary="No cluster analysis yet.",
                error="Call run_cluster_analysis() first.",
            )
        tbl = self._transfer_clusterer.anchors_table()
        return AgentResult(
            ok=True, data=_df_to_list(tbl),
            summary=(f"{len(tbl)} anchor(s) detected. Each is a person whose "
                     f"default would likely cascade to its dependents."),
        )

    def fragile_clusters(self, top_n: int = 10) -> AgentResult:
        """
        Rank clusters by anchor-contagion uplift: how much a cluster's expected
        loss rises if its anchor defaults (the quantified якорный-человек risk).
        """
        if getattr(self, "_cluster_metrics", None) is None:
            return AgentResult(
                ok=False, data=None, summary="No cluster analysis yet.",
                error="Call run_cluster_analysis() first.",
            )
        tbl = self._cluster_metrics.anchor_contagion_table()
        if len(tbl):
            tbl = tbl.head(top_n)
            worst = tbl.iloc[0]
            summary = (
                f"Top {len(tbl)} most anchor-dependent clusters. Worst: cluster "
                f"#{int(worst['transfer_cluster_id'])} loss rises "
                f"{worst['uplift_ratio']:.2f}× "
                f"({worst['el_unconditional']:.0f} → {worst['el_anchor_default']:.0f}) "
                f"if its anchor defaults."
            )
        else:
            summary = "No anchored clusters large enough for contagion analysis."
        return AgentResult(ok=True, data=_df_to_list(tbl), summary=summary)

    def cluster_report(self, cluster_id: int, dimension: str = "transfer") -> AgentResult:
        """
        Full report for one cluster: members, anchor, fragility, and the
        anchor-contagion uplift. dimension = 'transfer' (default) or 'geo'.
        """
        if getattr(self, "_persons", None) is None or \
                getattr(self, "_cluster_metrics", None) is None:
            return AgentResult(
                ok=False, data=None, summary="No cluster analysis yet.",
                error="Call run_cluster_analysis() first.",
            )
        col = "transfer_cluster_id" if dimension == "transfer" else "geo_cluster_id"
        members = self._persons[self._persons[col] == cluster_id]
        if len(members) == 0:
            return AgentResult(
                ok=False, data=None,
                summary=f"No members in {dimension} cluster {cluster_id}.",
                error="Unknown cluster id.",
            )
        contagion = self._cluster_metrics.anchor_contagion_table()
        crow = contagion[contagion["transfer_cluster_id"] == cluster_id]
        data = {
            "dimension": dimension,
            "cluster_id": int(cluster_id),
            "n_members": int(len(members)),
            "member_person_ids": _safe(members["person_id"].to_numpy()),
            "anchor_contagion": (_df_to_list(crow)[0] if len(crow) else None),
        }
        if "is_anchor" in members.columns and members["is_anchor"].any():
            arow = members[members["is_anchor"]].iloc[0]
            data["anchor_person_id"] = int(arow["person_id"])
            data["cluster_fragility"] = float(arow.get("cluster_fragility", float("nan")))
        return AgentResult(
            ok=True, data=data,
            summary=(f"{dimension.title()} cluster {cluster_id}: "
                     f"{len(members)} members."),
        )

    def _cluster_segment_result(self, dimension: str) -> AgentResult:
        if getattr(self, "_cluster_metrics", None) is None:
            return AgentResult(
                ok=False, data=None, summary="No cluster analysis yet.",
                error="Call run_cluster_analysis() first.",
            )
        if dimension == "geo":
            tbl = self._cluster_metrics.geo_metrics()
        else:
            tbl = self._cluster_metrics.transfer_metrics()
        return AgentResult(
            ok=True, data=_df_to_list(tbl),
            summary=(f"Risk metrics for {len(tbl)} {dimension} clusters "
                     f"(EAD, expected loss, σ, CoV, RAROC, Sortino)."),
        )

    # ── ARPM factor-loading estimation & conditional-FP regime weighting ──────
    #
    # These expose two ARPM ports (see src/low_rank_corr.py and
    # src/conditional_fp.py) through the agent façade. Both are diagnostic by
    # default and never mutate the live copula unless explicitly told to.

    def fit_factor_loadings(
        self,
        k_factors: int = 2,
        denoise: bool = True,
        apply: bool = False,
    ) -> AgentResult:
        """
        Fit factor-copula loadings from the transaction-graph correlation matrix.

        Uses the ARPM low-rank diagonal estimator
        (``low_rank_corr.fit_factor_loadings``) to turn the graph-derived
        correlation into an (n, k) loading matrix — the input
        ``MultiFactorCopula`` wants — so the factor structure is fitted from data
        instead of configured by hand.

        Parameters
        ----------
        k_factors : int
            Number of systematic factors to fit.
        denoise : bool
            Apply Marčenko-Pastur spectrum shrinkage to the correlation matrix
            before fitting (recommended — cleaner loadings).
        apply : bool
            If True, refit the live copula as a MultiFactorCopula using the
            fitted loadings. If False (default), only report the fit — the live
            model is left untouched (diagnostic mode).

        Returns
        -------
        AgentResult
            data: dict with k_factors, n_borrowers, avg_loading,
                  max_row_sumsq, implied_avg_within_factor_corr, applied (bool).
        """
        self._require_state("graph", "fit_factor_loadings")
        if self._graph is None:
            raise AgentError("Transaction graph not available; run the pipeline first.")

        from .low_rank_corr import fit_factor_loadings as _fit_loadings

        corr = self._graph.get_correlation_matrix(denoise=denoise)
        beta = _fit_loadings(corr, k_factors=k_factors)

        row_sumsq = np.sum(beta ** 2, axis=1)
        # Implied within-factor correlation ≈ mean β_i·β_j for loaded borrowers.
        loaded = beta[row_sumsq > 1e-9]
        implied_corr = float(np.mean(loaded @ loaded.T)) if len(loaded) > 1 else 0.0

        applied = False
        if apply:
            from .multi_factor_copula import MultiFactorCopula
            pds = self._persons["model_pd"].to_numpy()
            # One systematic-factor column per fitted factor; every borrower loads
            # on all of them (the loadings carry the per-borrower weighting).
            factor_matrix = np.tile(np.arange(k_factors), (len(pds), 1))
            self._copula = MultiFactorCopula().fit(pds, factor_matrix, betas=beta)
            self._state = "copula" if self._state in ("graph", "pd_model") else self._state
            applied = True

        data = _safe({
            "k_factors": k_factors,
            "denoised": denoise,
            "n_borrowers": int(beta.shape[0]),
            "avg_loading": float(np.mean(beta)),
            "max_row_sumsq": float(np.max(row_sumsq)),
            "implied_avg_within_factor_corr": implied_corr,
            "applied": applied,
        })
        summary = (
            f"Fitted {k_factors}-factor loadings from "
            f"{'denoised ' if denoise else ''}graph correlation: "
            f"avg loading={data['avg_loading']:.3f}, "
            f"max Σβ²={data['max_row_sumsq']:.3f} (<1 ⇒ copula-ready). "
            + ("Applied to the live copula (now multi-factor)."
               if applied else "Diagnostic only — live copula unchanged (apply=True to use).")
        )
        return AgentResult(ok=True, data=data, summary=summary)

    def regime_weights(
        self,
        current_stress: Optional[float] = None,
        method: str = "conditional_fp",
        alpha: float = 0.25,
        n_history: int = 250,
    ) -> AgentResult:
        """
        Compute regime-conditional scenario weights for the current stress level.

        Uses the rigorous ARPM conditional-FP estimator
        (``conditional_fp.conditional_fp``) — crisp window + entropy-pooling
        moment match — to weight a synthetic stress history by relevance to the
        current macro state, and reports how concentrated those weights are.

        Parameters
        ----------
        current_stress : float, optional
            Current stress level in [0, 1]. Defaults to the portfolio's
            model_pd-implied stress (same proxy as ``regime_status``).
        method : {"conditional_fp", "kernel"}
            Weighting estimator (see ``FlexibleProbsCalibrator``).
        alpha : float
            Crisp-window mass for the conditional-FP method.
        n_history : int
            Length of the synthetic stress history to weight.

        Returns
        -------
        AgentResult
            data: dict with current_stress, method, effective_scenarios,
                  concentration_ratio, regime_theta, base_theta.
        """
        self._require_state("pd_model", "regime_weights")
        from .conditional_fp import effective_scenarios
        from .flexible_probs import FlexibleProbsCalibrator

        if current_stress is None:
            current_stress = float(np.clip(
                (self._persons["model_pd"].mean() - 0.05) / 0.20, 0.0, 1.0
            ))

        # Synthetic stress history (monotone-correlation proxy) to weight.
        rng = np.random.default_rng(0)
        stress_history = np.clip(rng.beta(2, 5, n_history), 0.0, 1.0)

        calib = FlexibleProbsCalibrator(weighting_method=method, conditional_alpha=alpha)
        calib.fit(stress_history)
        weights = calib._compute_weights(current_stress)
        weights = np.asarray(weights, dtype=float).ravel()
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        ess = effective_scenarios(weights)
        base_theta = float(self._copula.params.theta) if self._copula is not None else float("nan")

        # Regime theta at this stress (reuse the calibrator's corr→theta path).
        base_corr = np.eye(4) + 0.1 * (np.ones((4, 4)) - np.eye(4))
        regime = calib.calibrate(current_stress, base_corr, decompose=False)

        data = _safe({
            "current_stress": current_stress,
            "method": method,
            "effective_scenarios": ess,
            "n_history": n_history,
            "concentration_ratio": ess / n_history,
            "regime_theta": float(regime.theta),
            "base_theta": base_theta,
        })
        summary = (
            f"Regime weights ({method}) at stress={current_stress:.2f}: "
            f"effective scenarios={ess:.0f}/{n_history} "
            f"({data['concentration_ratio']:.0%} of history informative), "
            f"regime θ={float(regime.theta):.3f}."
        )
        return AgentResult(ok=True, data=data, summary=summary)

    def calibrate_copula_from_data(
        self,
        events: "pd.DataFrame",
        family: str = "auto",
        segment_col: Optional[str] = None,
        apply: bool = False,
    ) -> AgentResult:
        """
        Calibrate copula parameters from an OBSERVED default panel (Plan 07).

        Estimates empirical default dependence (Kendall τ, Schweizer-Wolff,
        default correlation, observed vs independent joint-default) and maps it
        to Gaussian / Student-t / Clayton parameters, recommending a family by
        goodness-of-fit. Diagnostic by default — it does not change the live
        copula unless ``apply=True``.

        Parameters
        ----------
        events : DataFrame
            Long default panel: borrower / period / default[/ model_pd / segment].
        family : {"auto", "gaussian", "student_t", "clayton"}
            Force a family or auto-recommend.
        segment_col : str, optional
            Break the empirical measures out by this column too.
        apply : bool
            If True and the recommended family is Clayton, refit the live dense
            copula's θ to the calibrated value (other families are reported only,
            since the live model is a Clayton CopulaDefaultModel by default).

        Returns
        -------
        AgentResult
            data: dict with recommended_family, recommended_params,
                  empirical measures, family_table (list of dicts), applied.
        """
        from .copula_calibration import build_default_panel, calibrate_copula

        panel = build_default_panel(events, segment_col=segment_col)
        result = calibrate_copula(panel, family=family, segment_col=segment_col)

        applied = False
        if apply and result.recommended_family == "clayton" and self._copula is not None:
            theta = result.recommended_params.get("theta")
            if theta is not None and hasattr(self._copula, "params"):
                self._copula.params.theta = float(theta)
                applied = True

        data = _safe({
            "recommended_family": result.recommended_family,
            "recommended_params": result.recommended_params,
            "empirical": result.empirical,
            "family_table": result.family_table.to_dict(orient="records"),
            "warnings": result.warnings,
            "applied": applied,
        })
        summary = (
            f"Calibrated copula from {len(panel)} borrower-period observations: "
            f"recommend {result.recommended_family} "
            f"({', '.join(f'{k}={v:.3f}' for k, v in result.recommended_params.items())}), "
            f"empirical default-corr={result.empirical.get('default_corr', 0):.3f}. "
            + ("Applied θ to the live copula." if applied
               else "Diagnostic only — live copula unchanged.")
        )
        return AgentResult(ok=True, data=data, summary=summary,
                           warnings=result.warnings)

    def state(self) -> str:
        """
        Return current pipeline state.

        Possible values:
          'empty'    → nothing done
          'data'     → data generated
          'graph'    → transaction graph built
          'pd_model' → PD model fitted
          'copula'   → copula fitted
          'pipeline' → full pipeline complete
        """
        return self._state

    def persons(self) -> Optional[pd.DataFrame]:
        """
        Return the current persons DataFrame (with all enrichments applied so far).

        Returns None if no data has been loaded yet.
        """
        return self._persons.copy() if self._persons is not None else None
