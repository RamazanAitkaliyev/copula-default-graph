"""
Copula Default Graph  —  Network-Based Credit Risk Framework
=============================================================

AGENT QUICK-START (read AGENTS.md for the full contract)
---------------------------------------------------------
For AI agents, the recommended entry point is RiskAgentAPI:

    from src.agents import RiskAgentAPI

    api = RiskAgentAPI()
    api.run_pipeline()                      # runs all 13 steps (~30-60s)

    r = api.query_borrower(776)             # full risk profile
    r = api.segment_metrics("city_name")    # metrics by city
    r = api.flag_divergences()              # RAROC vs Sortino early warnings
    r = api.run_stress(pd_multiplier=2.0)   # stress scenario
    r = api.portfolio_summary()             # VaR / ES / HHI

    print(r.summary)    # human-readable interpretation
    print(r.data)       # JSON-safe structured result
    print(r.warnings)   # non-fatal issues to mention

FRAMEWORK OVERVIEW
------------------
Three-layer architecture:

  Layer 1 — Graph + PD model
    generate_network()       → persons + transactions DataFrames
    TransactionGraph         → correlation matrix + network features
    IndividualPDModel        → model_pd per borrower (GBM, AUC-validated)

  Layer 2 — Copula + loss covariance
    CopulaDefaultModel       → joint default probability matrix P(D_i ∩ D_j)
    RiskRatioCalculator      → loss-covariance matrix + 7 risk-adjusted metrics

  Layer 3 — Analytics + reports
    RiskAnalyzer             → VaR, ES, contagion scores, stress test
    MetricComparator         → rank correlation + divergence flags
    RatingEngine             → PD → AAA…Default + migration probabilities
    StructuralPDModel        → Merton second signal + early warnings
    FlexibleProbsCalibrator  → regime-aware copula calibration
    CustomerProfiler         → per-borrower one-page risk report

KEY INVARIANT (never violate):
  Segment metrics MUST use block-sum of loss_cov matrix — NEVER average
  per-borrower metric values. Use calc.by_segment(col), not df.groupby(col).mean().

For full invariant list and common-mistake guide: see AGENTS.md.
For ready-to-use agent system prompts: see PROMPTS.md.
"""

from .data_generator import generate_network, CityConfig, get_summary_stats
from .graph_features import TransactionGraph, get_neighbor_risk_features
from .copula_model import CopulaDefaultModel, CopulaParams, compare_copulas
from .risk_metrics import RiskAnalyzer, PortfolioRiskResult, FraudRingDetector
from .pd_model import IndividualPDModel, PDModelEnsemble
from .client_value_metrics import ClientValueCalculator, ClientPortfolioAnalyzer
from .rating_engine import RatingEngine, RatingProfile, PortfolioRatingDistribution, RATING_LABELS
from .structural_pd import StructuralPDModel, MertonParams, compute_proxy_merton_pd
from .flexible_probs import FlexibleProbsCalibrator, RegimeAdjustedCopula, RegimeState, build_calibrator_from_portfolio
from .customer_profile import CustomerProfiler, CustomerRiskProfile
from .risk_adjusted_metrics import (
    RiskRatioCalculator,
    MetricInputs,
    available_metrics,
    register_metric,
    compute_metric,
)
from .metric_comparison import MetricComparator
from .agents import RiskAgentAPI, AgentResult, AgentError
from .loaders import (
    ColumnMapping,
    DataValidationError,
    load_persons,
    load_transactions,
    validate_persons,
    validate_transactions,
    reindex_to_contiguous,
    describe_persons,
)
from .factor_copula import FactorCopula, FactorCopulaParams, build_factor_id
from .config import (
    NetworkConfig,
    CopulaConfig,
    RiskConfig,
    VisualizationConfig,
    PipelineConfig,
    DEFAULT_CONFIG,
)

__all__ = [
    # Data generation
    'generate_network',
    'CityConfig',
    'get_summary_stats',
    # Graph analysis
    'TransactionGraph',
    'get_neighbor_risk_features',
    # Copula modeling
    'CopulaDefaultModel',
    'CopulaParams',
    'compare_copulas',
    # Risk analysis
    'RiskAnalyzer',
    'PortfolioRiskResult',
    'FraudRingDetector',
    # PD model
    'IndividualPDModel',
    'PDModelEnsemble',
    # Client value
    'ClientValueCalculator',
    'ClientPortfolioAnalyzer',
    # Rating engine
    'RatingEngine',
    'RatingProfile',
    'PortfolioRatingDistribution',
    'RATING_LABELS',
    # Structural PD
    'StructuralPDModel',
    'MertonParams',
    'compute_proxy_merton_pd',
    # Flexible probabilities / regime
    'FlexibleProbsCalibrator',
    'RegimeAdjustedCopula',
    'RegimeState',
    'build_calibrator_from_portfolio',
    # Customer profiles
    'CustomerProfiler',
    'CustomerRiskProfile',
    # Risk-adjusted metric family
    'RiskRatioCalculator',
    'MetricInputs',
    'available_metrics',
    'register_metric',
    'compute_metric',
    'MetricComparator',
    # Agent-facing API
    'RiskAgentAPI',
    'AgentResult',
    'AgentError',
    # Data loading & validation
    'ColumnMapping',
    'DataValidationError',
    'load_persons',
    'load_transactions',
    'validate_persons',
    'validate_transactions',
    'reindex_to_contiguous',
    'describe_persons',
    # Factor copula (scales to 10M+)
    'FactorCopula',
    'FactorCopulaParams',
    'build_factor_id',
    # Configuration
    'NetworkConfig',
    'CopulaConfig',
    'RiskConfig',
    'VisualizationConfig',
    'PipelineConfig',
    'DEFAULT_CONFIG',
]
