"""
src.risk — Risk Analyst role view (metrics, portfolio, stress, ratings).

A NAVIGATION facade re-exporting the flat modules a Risk Analyst owns
(`from src.risk import RiskRatioCalculator`). Implementation stays in
`src/*.py`. See ROLES.md.

Owns: the numbers the bank acts on — expected loss, VaR/ES, risk-adjusted
metrics at every level, cluster contagion, stress tests, ratings, watchlists.

Contract consumed: a fitted copula (from src.copula) + persons + EAD/LGD.
Key invariant: segment/cluster variance is the BLOCK-SUM of the loss-covariance
matrix, never an average of per-borrower ratios (INV-6).
"""
from ..risk_adjusted_metrics import (
    RiskRatioCalculator,
    MetricInputs,
    register_metric,
    available_metrics,
    compute_metric,
    LOSS_COV_DENSE_MAX_NODES,
)
from ..cluster_metrics import ClusterRiskMetrics, ClusterMetricsResult
from ..risk_metrics import (
    RiskAnalyzer,
    PortfolioRiskResult,
    FraudRingDetector,
)
from ..client_value_metrics import ClientValueCalculator, ClientPortfolioAnalyzer
from ..metric_comparison import MetricComparator
from ..rating_engine import (
    RatingEngine,
    RatingProfile,
    PortfolioRatingDistribution,
    RATING_LABELS,
)
from ..customer_profile import CustomerProfiler, CustomerRiskProfile

__all__ = [
    "RiskRatioCalculator", "MetricInputs", "register_metric", "available_metrics",
    "compute_metric", "LOSS_COV_DENSE_MAX_NODES",
    "ClusterRiskMetrics", "ClusterMetricsResult",
    "RiskAnalyzer", "PortfolioRiskResult", "FraudRingDetector",
    "ClientValueCalculator", "ClientPortfolioAnalyzer",
    "MetricComparator",
    "RatingEngine", "RatingProfile", "PortfolioRatingDistribution", "RATING_LABELS",
    "CustomerProfiler", "CustomerRiskProfile",
]
