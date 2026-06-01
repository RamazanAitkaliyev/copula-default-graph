"""
Copula Default Graph - Network-Based Credit Risk Analysis

A framework for analyzing credit default risk using:
- Individual probability of default (PD) from features
- Transaction network graphs
- Copula dependency modeling
- Multi-level risk metrics (individual, group, portfolio)

Key Questions Answered:
1. Who is individually risky? (marginal PD)
2. Who is dangerous to others? (systemic importance)
3. Who is vulnerable to others? (contagion vulnerability)
4. What is the portfolio tail risk? (VaR, ES with dependencies)

Main Components:
- data_generator: Synthetic network generation (1000 persons, 3 cities)
- graph_features: Transaction graph analysis and visualization
- copula_model: Joint default probability modeling (Clayton copula)
- risk_metrics: Comprehensive risk analysis at all levels

Usage:
    from src import generate_network, TransactionGraph
    from src import CopulaDefaultModel, RiskAnalyzer

    # Generate data
    persons, transactions = generate_network(seed=42)

    # Build graph
    graph = TransactionGraph(transactions, persons)

    # Fit copula
    copula = CopulaDefaultModel('clayton')
    corr_matrix = graph.get_correlation_matrix()
    copula.fit(persons['base_pd'].values, corr_matrix)

    # Analyze risks
    analyzer = RiskAnalyzer(copula, graph, persons)
    individual_risks = analyzer.compute_individual_risks()
    portfolio_risks = analyzer.compute_portfolio_risks()
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
    # Configuration
    'NetworkConfig',
    'CopulaConfig',
    'RiskConfig',
    'VisualizationConfig',
    'PipelineConfig',
    'DEFAULT_CONFIG',
]
