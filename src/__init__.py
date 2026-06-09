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

ROLE-BASED NAVIGATION (for teams — see ROLES.md)
------------------------------------------------
The flat modules are also grouped into role-oriented facade subpackages so each
team can import from its own domain (no files were moved; these only re-export):

    from src.data_eng  import load_persons, ColumnMapping        # Data Engineer
    from src.ml        import IndividualPDModel                   # ML Engineer
    from src.analytics import TransactionGraph, TransferClusterer # Data Scientist (graph)
    from src.copula    import MultiFactorCopula, FactorCopula     # Data Scientist (copula)
    from src.risk      import RiskRatioCalculator, RiskAnalyzer   # Risk Analyst

KEY INVARIANT (never violate):
  Segment metrics MUST use block-sum of loss_cov matrix — NEVER average
  per-borrower metric values. Use calc.by_segment(col), not df.groupby(col).mean().

For full invariant list and common-mistake guide: see AGENTS.md.
For ready-to-use agent system prompts: see PROMPTS.md.
For who-owns-what: see ROLES.md. For architecture: see ARCHITECTURE.md.
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
from .multi_factor_copula import MultiFactorCopula, MultiFactorCopulaParams
from .geo_clusters import GeoClusterer, GeoClusterConfig
from .transfer_clusters import TransferClusterer, TransferClusterConfig
from .cluster_metrics import ClusterRiskMetrics, ClusterMetricsResult
from .relative_entropy import min_rel_entropy_sp
from .credit_transitions import (
    fit_trans_matrix_credit,
    estimate_generator,
    cohort_arrays_from_events,
)
from .spectrum import (
    spectrum_shrink,
    denoise_correlation,
    marchenko_pastur_pdf,
    mp_support,
    ShrinkResult,
)
from .conditional_fp import (
    conditional_fp,
    crisp_fp,
    quantile_smooth,
    effective_scenarios,
)
from .low_rank_corr import (
    low_rank_diag_conditional_corr,
    fit_factor_loadings,
    conditional_pc,
    LowRankResult,
)
from .dependence import (
    schweizer_wolff,
    copula_invariance_test,
)
from .copula_calibration import (
    build_default_panel,
    empirical_dependence_measures,
    calibrate_copula,
    CalibrationResult,
    clayton_theta_from_tau,
    gaussian_rho_from_tau,
    default_correlation,
)
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
    'MultiFactorCopula',
    'MultiFactorCopulaParams',
    'GeoClusterer',
    'GeoClusterConfig',
    'TransferClusterer',
    'TransferClusterConfig',
    'ClusterRiskMetrics',
    'ClusterMetricsResult',
    # Entropy pooling / relative-entropy minimisation (arpym port)
    'min_rel_entropy_sp',
    # Credit transition-matrix estimation (arpym fit_trans_matrix_credit port)
    'fit_trans_matrix_credit',
    'estimate_generator',
    'cohort_arrays_from_events',
    # Random-matrix spectrum shrinkage (arpym spectrum_shrink port)
    'spectrum_shrink',
    'denoise_correlation',
    'marchenko_pastur_pdf',
    'mp_support',
    'ShrinkResult',
    # Conditional flexible probabilities (arpym conditional_fp / crisp_fp ports)
    'conditional_fp',
    'crisp_fp',
    'quantile_smooth',
    'effective_scenarios',
    # Low-rank diagonal correlation / factor-loading estimation (arpym port)
    'low_rank_diag_conditional_corr',
    'fit_factor_loadings',
    'conditional_pc',
    'LowRankResult',
    # Copula dependence measures (arpym schweizer_wolff / invariance test ports)
    'schweizer_wolff',
    'copula_invariance_test',
    # Empirical copula calibration (Plan 07)
    'build_default_panel',
    'empirical_dependence_measures',
    'calibrate_copula',
    'CalibrationResult',
    'clayton_theta_from_tau',
    'gaussian_rho_from_tau',
    'default_correlation',
    # Configuration
    'NetworkConfig',
    'CopulaConfig',
    'RiskConfig',
    'VisualizationConfig',
    'PipelineConfig',
    'DEFAULT_CONFIG',
]
