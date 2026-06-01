"""
Risk Metrics Framework

Computes risk metrics at three levels:
1. INDIVIDUAL: Per-person risk scores and rankings
2. GROUP: High-risk cluster analysis
3. PORTFOLIO: VaR, ES, concentration metrics

These metrics answer:
- Who is individually risky?
- Who is dangerous to others (systemic)?
- Who is vulnerable to others (contagion)?
- What is the portfolio tail risk?
"""

from __future__ import annotations

import contextlib
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .copula_model import CopulaDefaultModel
    from .graph_features import TransactionGraph

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared loss-calculation mixin (eliminates duplication between
# PortfolioRiskMetrics and RiskAnalyzer)
# ──────────────────────────────────────────────────────────────────────────────

class _PortfolioLossMixin:
    """Shared helpers for portfolio loss computation."""

    def _compute_losses(self, defaults: np.ndarray) -> np.ndarray:
        """Compute portfolio loss per simulation row."""
        return (defaults * self.exposures * self.lgd).sum(axis=1)

    def _estimate_default_correlation(
        self, defaults: np.ndarray, n_pairs: int = 500
    ) -> float:
        """Estimate average default correlation from simulations."""
        rng = np.random.default_rng()
        n = defaults.shape[1]
        pairs = min(n_pairs, n * (n - 1) // 2)
        idx = rng.integers(0, n, size=(pairs, 2))
        # Remove self-pairs
        idx = idx[idx[:, 0] != idx[:, 1]]
        if len(idx) == 0:
            return 0.0
        corrs = []
        for i, j in idx:
            di, dj = defaults[:, i], defaults[:, j]
            if di.std() > 0 and dj.std() > 0:
                c = float(np.corrcoef(di, dj)[0, 1])
                if not np.isnan(c):
                    corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    def _compute_concentration(self) -> float:
        """Herfindahl concentration index of exposures."""
        total = self.exposures.sum()
        if total <= 0:
            return 0.0
        shares = self.exposures / total
        return float((shares ** 2).sum())


@dataclass
class IndividualRiskProfile:
    """Risk profile for a single person."""
    person_id: int
    marginal_pd: float
    contagion_vulnerability: float
    systemic_importance: float
    network_exposure: float
    composite_risk_score: float
    risk_tier: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class GroupRiskMetrics:
    """Risk metrics for a group/cluster."""
    group_id: int
    members: List[int]
    size: int
    avg_pd: float
    max_pd: float
    internal_correlation: float
    expected_group_loss: float
    joint_default_probability: float


@dataclass
class PortfolioRiskResult:
    """Portfolio-level risk metrics result container."""
    expected_loss: float
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    default_correlation: float
    concentration_index: float
    tail_risk_ratio: float  # ES/VaR
    contagion_adjustment: float = 0.0


class PortfolioRiskMetrics(_PortfolioLossMixin):
    """
    Calculate portfolio-level risk metrics using copula model.

    This is the class expected by main.py for computing VaR, ES,
    and other portfolio metrics.
    """

    def __init__(
        self,
        copula_model,
        lgd: float = 0.45,
        exposures: Optional[np.ndarray] = None
    ):
        """
        Initialize portfolio risk calculator.

        Parameters
        ----------
        copula_model : CopulaDefaultModel
            Fitted copula model
        lgd : float
            Loss given default (0-1)
        exposures : np.ndarray, optional
            Exposure at default per person. Defaults to equal weights.
        """
        self.copula = copula_model
        self.lgd = lgd
        self.n = copula_model.n
        self.exposures = exposures if exposures is not None else np.ones(self.n)

    def compute_all_metrics(
        self,
        n_simulations: int = 10000
    ) -> PortfolioRiskResult:
        """
        Compute all portfolio risk metrics.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations

        Returns
        -------
        PortfolioRiskResult
            Container with all risk metrics
        """
        # Simulate correlated defaults
        defaults = self.copula.simulate_defaults(n_simulations)

        # Compute losses
        losses = self._compute_losses(defaults)

        # Risk metrics
        expected_loss = losses.mean()
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        es_95 = losses[losses >= var_95].mean() if (losses >= var_95).any() else var_95
        es_99 = losses[losses >= var_99].mean() if (losses >= var_99).any() else var_99

        # Default correlation
        default_corr = self._estimate_default_correlation(defaults)

        # Concentration index (Herfindahl)
        concentration = self._compute_concentration()

        # Tail risk ratio
        tail_ratio = es_95 / var_95 if var_95 > 0 else 1.0

        # Contagion adjustment (difference from independent case)
        independent_el = (self.copula.marginal_pds * self.exposures * self.lgd).sum()
        contagion_adjustment = expected_loss - independent_el

        return PortfolioRiskResult(
            expected_loss=expected_loss,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            default_correlation=default_corr,
            concentration_index=concentration,
            tail_risk_ratio=tail_ratio,
            contagion_adjustment=contagion_adjustment
        )

    def loss_distribution(self, n_simulations: int = 10000) -> np.ndarray:
        """
        Get full loss distribution from Monte Carlo.

        Parameters
        ----------
        n_simulations : int
            Number of simulations

        Returns
        -------
        losses : np.ndarray
            Array of simulated portfolio losses
        """
        defaults = self.copula.simulate_defaults(n_simulations)
        return self._compute_losses(defaults)

class RiskAnalyzer(_PortfolioLossMixin):
    """
    Comprehensive risk analysis combining PD model, graph, and copula.

    Usage:
        analyzer = RiskAnalyzer(copula_model, graph, persons)
        individual_risks = analyzer.compute_individual_risks()
        group_risks = analyzer.compute_group_risks()
        portfolio_risks = analyzer.compute_portfolio_risks()
    """

    def __init__(
        self,
        copula_model,
        graph,
        persons: pd.DataFrame,
        exposures: Optional[np.ndarray] = None,
        lgd: float = 0.45
    ):
        """
        Initialize risk analyzer.

        Parameters
        ----------
        copula_model : CopulaDefaultModel
            Fitted copula model
        graph : TransactionGraph
            Transaction graph
        persons : pd.DataFrame
            Person data with base_pd, city, etc.
        exposures : np.ndarray, optional
            Exposure at default per person. Defaults to equal weights.
        lgd : float
            Loss given default (0-1)
        """
        self.copula = copula_model
        self.graph = graph
        self.persons = persons
        self.n = len(persons)
        self.exposures = exposures if exposures is not None else np.ones(self.n)
        self.lgd = lgd

        # Cache commonly used values
        self._vulnerability = None
        self._importance = None
        self._network_exposure = None

    # =========================================================================
    # INDIVIDUAL RISK METRICS
    # =========================================================================

    def compute_individual_risks(self) -> pd.DataFrame:
        """
        Compute comprehensive risk metrics for each individual.

        Returns DataFrame with:
        - marginal_pd: Base probability of default
        - contagion_vulnerability: How much PD increases if neighbors default
        - systemic_importance: How much others' PD increases if this person defaults
        - network_exposure: Weighted average neighbor PD
        - composite_risk_score: Weighted combination of above
        - risk_tier: Categorical risk level
        """
        # Use copula marginal PDs (model-predicted), fall back to base_pd
        marginal_pds = (self.copula.marginal_pds
                        if self.copula.is_fitted
                        else self.persons['base_pd'].values)

        # Contagion metrics from copula
        vulnerability = self._get_vulnerability()
        importance = self._get_systemic_importance()

        # Network exposure (weighted avg neighbor PD)
        network_exposure = self._compute_network_exposure()

        # Composite risk score (normalized weighted sum)
        composite = self._compute_composite_score(
            marginal_pds, vulnerability, importance, network_exposure
        )

        # Risk tiers
        tiers = self._assign_risk_tiers(composite)

        results = pd.DataFrame({
            'person_id': range(self.n),
            'city_name': self.persons['city_name'].values,
            'risk_archetype': self.persons['risk_archetype'].values,
            'marginal_pd': np.round(marginal_pds, 4),
            'contagion_vulnerability': np.round(vulnerability, 4),
            'systemic_importance': np.round(importance, 4),
            'network_exposure': np.round(network_exposure, 4),
            'composite_risk_score': np.round(composite, 4),
            'risk_tier': tiers,
            'in_high_risk_group': self.persons['high_risk_group_id'].values >= 0,
            'is_bridge': self.persons['is_bridge'].values,
        })

        return results.sort_values('composite_risk_score', ascending=False)

    def _get_vulnerability(self) -> np.ndarray:
        """Get or compute contagion vulnerability."""
        if self._vulnerability is None:
            self._vulnerability = self.copula.contagion_vulnerability()
        return self._vulnerability

    def _get_systemic_importance(self) -> np.ndarray:
        """Get or compute systemic importance."""
        if self._importance is None:
            self._importance = self.copula.systemic_importance()
        return self._importance

    def _compute_network_exposure(self) -> np.ndarray:
        """Compute network-based risk exposure."""
        pds = (self.copula.marginal_pds
               if self.copula.is_fitted
               else self.persons['base_pd'].values)
        adj = self.graph.adj_weighted

        # Normalize adjacency (avoid division by zero)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        adj_norm = adj / row_sums

        return adj_norm @ pds

    def _compute_composite_score(
        self,
        marginal_pd: np.ndarray,
        vulnerability: np.ndarray,
        importance: np.ndarray,
        network_exposure: np.ndarray
    ) -> np.ndarray:
        """
        Compute composite risk score.

        Weights:
        - 40% marginal PD (individual risk)
        - 25% network exposure (neighborhood risk)
        - 20% vulnerability (contagion risk)
        - 15% systemic importance (impact risk)
        """
        # Normalize each component to [0, 1]
        def normalize(x):
            if x.max() - x.min() < 1e-10:
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        score = (
            0.40 * normalize(marginal_pd) +
            0.25 * normalize(network_exposure) +
            0.20 * normalize(vulnerability) +
            0.15 * normalize(importance)
        )

        return score

    def _assign_risk_tiers(self, composite_scores: np.ndarray) -> np.ndarray:
        """Assign risk tiers based on composite score percentiles (vectorised)."""
        from scipy.stats import rankdata
        percentiles = rankdata(composite_scores, method='average') / len(composite_scores)

        tiers = np.full(self.n, 'low', dtype=object)
        tiers[percentiles > 0.60] = 'medium'
        tiers[percentiles > 0.85] = 'high'
        tiers[percentiles > 0.95] = 'critical'

        return tiers

    def get_top_risks(self, n: int = 20, by: str = 'composite_risk_score') -> pd.DataFrame:
        """Get top N riskiest individuals."""
        individual_risks = self.compute_individual_risks()
        return individual_risks.nlargest(n, by)

    def get_bridge_analysis(self) -> pd.DataFrame:
        """Analyze bridge individuals who connect cities."""
        individual_risks = self.compute_individual_risks()
        bridges = individual_risks[individual_risks['is_bridge']]

        summary = pd.DataFrame({
            'metric': [
                'n_bridges',
                'avg_marginal_pd',
                'avg_systemic_importance',
                'avg_network_exposure',
            ],
            'value': [
                len(bridges),
                bridges['marginal_pd'].mean(),
                bridges['systemic_importance'].mean(),
                bridges['network_exposure'].mean(),
            ]
        })

        return summary

    # =========================================================================
    # GROUP RISK METRICS
    # =========================================================================

    def compute_group_risks(self) -> List[GroupRiskMetrics]:
        """
        Compute risk metrics for each high-risk group.

        Returns list of GroupRiskMetrics for each identified group.
        """
        group_ids = self.persons['high_risk_group_id'].unique()
        group_ids = [g for g in group_ids if g >= 0]

        group_metrics = []

        for gid in group_ids:
            members = self.persons[
                self.persons['high_risk_group_id'] == gid
            ]['person_id'].values.tolist()

            metrics = self._compute_group_metrics(gid, members)
            group_metrics.append(metrics)

        # Sort by expected loss
        group_metrics.sort(key=lambda x: x.expected_group_loss, reverse=True)

        return group_metrics

    def _compute_group_metrics(
        self, group_id: int, members: List[int]
    ) -> GroupRiskMetrics:
        """Compute metrics for a single group."""
        pds = (self.copula.marginal_pds
               if self.copula.is_fitted
               else self.persons['base_pd'].values)
        member_pds = pds[members]

        # Internal correlation — vectorised upper-triangle extraction
        corr_matrix = self.copula.correlation_matrix
        m_idx = np.array(members)
        group_corr = corr_matrix[np.ix_(m_idx, m_idx)]
        upper = np.triu(group_corr, k=1)
        mask = upper != 0
        avg_internal_corr = float(upper[mask].mean()) if mask.any() else 0.0

        # Joint default probability (Clayton copula lower-tail approximation)
        # P(all default) ≈ prod(PD_i)^(1/corr_factor) clipped to [0,1]
        corr_factor = max(1.0 + avg_internal_corr * (len(members) - 1), 1e-6)
        log_joint = np.log(member_pds + 1e-10).sum() / corr_factor
        joint_pd = np.clip(np.exp(log_joint), 0.0, 1.0)

        # Expected loss from this group
        group_exposures = self.exposures[members]
        expected_loss = (member_pds * group_exposures * self.lgd).sum()

        return GroupRiskMetrics(
            group_id=group_id,
            members=members,
            size=len(members),
            avg_pd=float(member_pds.mean()),
            max_pd=float(member_pds.max()),
            internal_correlation=float(avg_internal_corr),
            expected_group_loss=float(expected_loss),
            joint_default_probability=float(np.clip(joint_pd, 0, 1))
        )

    def get_group_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all groups."""
        groups = self.compute_group_risks()

        return pd.DataFrame([
            {
                'group_id': g.group_id,
                'size': g.size,
                'avg_pd': round(g.avg_pd, 4),
                'max_pd': round(g.max_pd, 4),
                'internal_correlation': round(g.internal_correlation, 4),
                'expected_loss': round(g.expected_group_loss, 4),
            }
            for g in groups
        ])

    # =========================================================================
    # PORTFOLIO RISK METRICS
    # =========================================================================

    def compute_portfolio_risks(
        self, n_simulations: int = 10000
    ) -> PortfolioRiskResult:
        """
        Compute portfolio-level risk metrics.

        Uses Monte Carlo simulation with copula-correlated defaults.
        """
        # Simulate correlated defaults
        defaults = self.copula.simulate_defaults(n_simulations)

        # Compute losses per simulation
        losses = self._compute_losses(defaults)

        # Risk metrics
        expected_loss = losses.mean()
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        es_95 = losses[losses >= var_95].mean() if (losses >= var_95).any() else var_95
        es_99 = losses[losses >= var_99].mean() if (losses >= var_99).any() else var_99

        # Default correlation
        default_corr = self._estimate_default_correlation(defaults)

        # Concentration index (Herfindahl)
        concentration = self._compute_concentration()

        # Tail risk ratio
        tail_ratio = es_95 / var_95 if var_95 > 0 else 1.0

        return PortfolioRiskResult(
            expected_loss=expected_loss,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            default_correlation=default_corr,
            concentration_index=concentration,
            tail_risk_ratio=tail_ratio
        )

    def get_loss_distribution(self, n_simulations: int = 10000) -> np.ndarray:
        """Get full loss distribution from Monte Carlo."""
        defaults = self.copula.simulate_defaults(n_simulations)
        return self._compute_losses(defaults)

    @staticmethod
    def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
        """Project a symmetric matrix to a positive semi-definite matrix."""
        sym = (matrix + matrix.T) / 2
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, 1e-8)
        psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        psd = (psd + psd.T) / 2
        np.fill_diagonal(psd, 1.0)
        return psd

    def portfolio_summary(self) -> pd.DataFrame:
        """Get summary of portfolio risk metrics."""
        metrics = self.compute_portfolio_risks()

        return pd.DataFrame({
            'metric': [
                'Expected Loss',
                'VaR 95%',
                'VaR 99%',
                'Expected Shortfall 95%',
                'Expected Shortfall 99%',
                'Default Correlation',
                'Concentration Index',
                'Tail Risk Ratio (ES/VaR)',
            ],
            'value': [
                round(metrics.expected_loss, 4),
                round(metrics.var_95, 4),
                round(metrics.var_99, 4),
                round(metrics.es_95, 4),
                round(metrics.es_99, 4),
                round(metrics.default_correlation, 4),
                round(metrics.concentration_index, 4),
                round(metrics.tail_risk_ratio, 4),
            ]
        })

    # =========================================================================
    # STRESS TESTING
    # =========================================================================

    @contextlib.contextmanager
    def _stressed_copula(self, pd_multiplier: float, correlation_boost: float):
        """Context manager that temporarily applies stress to the copula then restores it."""
        original_pds  = self.copula.marginal_pds.copy()
        original_corr = self.copula.correlation_matrix.copy()
        try:
            stressed_pds  = np.clip(original_pds * pd_multiplier, 0, 0.99)
            stressed_corr = np.clip(original_corr + correlation_boost, 0, 0.99)
            np.fill_diagonal(stressed_corr, 1.0)
            stressed_corr = self._nearest_psd(stressed_corr)
            self.copula.marginal_pds       = stressed_pds
            self.copula.correlation_matrix = stressed_corr
            yield
        finally:
            self.copula.marginal_pds       = original_pds
            self.copula.correlation_matrix = original_corr

    def stress_test(
        self,
        pd_multiplier: float = 2.0,
        correlation_boost: float = 0.2,
        n_simulations: int = 5000,
    ) -> Dict:
        """
        Run stress test with elevated PDs and correlations.

        Parameters
        ----------
        pd_multiplier : float
            Multiply all PDs by this factor
        correlation_boost : float
            Add this to all correlations
        n_simulations : int
            Monte Carlo paths per scenario

        Returns
        -------
        dict with base and stressed metrics
        """
        # Compute base metrics before any modifications
        base_metrics = self.compute_portfolio_risks(n_simulations=n_simulations)

        # Apply stress inside context manager — always restored even on exception
        with self._stressed_copula(pd_multiplier, correlation_boost):
            stressed_metrics = self.compute_portfolio_risks(n_simulations=n_simulations)

        return {
            'base': {
                'expected_loss': base_metrics.expected_loss,
                'var_95': base_metrics.var_95,
                'es_95': base_metrics.es_95,
            },
            'stressed': {
                'expected_loss': stressed_metrics.expected_loss,
                'var_95': stressed_metrics.var_95,
                'es_95': stressed_metrics.es_95,
            },
            'change': {
                'expected_loss': stressed_metrics.expected_loss / base_metrics.expected_loss - 1,
                'var_95': stressed_metrics.var_95 / base_metrics.var_95 - 1,
                'es_95': stressed_metrics.es_95 / base_metrics.es_95 - 1,
            }
        }


class FraudRingDetector:
    """
    Detect suspicious fraud rings in transaction networks.

    Fraud rings are characterized by:
    - Dense internal connections
    - Circular money flows
    - High joint default probabilities
    - Unusual transaction patterns
    """

    def __init__(
        self,
        graph,
        copula_model,
        persons: pd.DataFrame
    ):
        """
        Initialize fraud detector.

        Parameters
        ----------
        graph : TransactionGraph
            Transaction graph
        copula_model : CopulaDefaultModel
            Fitted copula model
        persons : pd.DataFrame
            Person data
        """
        self.graph = graph
        self.copula = copula_model
        self.persons = persons
        self.n = len(persons)

    def detect_suspicious_clusters(
        self,
        min_cluster_size: int = 3,
        joint_pd_threshold: float = 0.15,
        density_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect suspicious clusters based on high joint default probability
        and dense connections.

        Parameters
        ----------
        min_cluster_size : int
            Minimum number of members to consider
        joint_pd_threshold : float
            Minimum average joint PD to flag as suspicious
        density_threshold : float
            Minimum internal density (edges / possible edges)

        Returns
        -------
        suspicious : list of dict
            List of suspicious clusters with metadata
        """
        suspicious_clusters = []

        # Use community detection to find clusters
        communities = self.graph.detect_communities(n_communities=10)

        for comm_id in np.unique(communities):
            members = np.where(communities == comm_id)[0]

            if len(members) < min_cluster_size:
                continue

            # Compute cluster metrics
            cluster_metrics = self._analyze_cluster(members)

            # Check suspicion criteria
            is_suspicious = (
                cluster_metrics['avg_joint_pd'] > joint_pd_threshold or
                cluster_metrics['internal_density'] > density_threshold or
                cluster_metrics['circular_flow_score'] > 0.3
            )

            if is_suspicious:
                suspicion_score = (
                    cluster_metrics['avg_joint_pd'] * 2 +
                    cluster_metrics['internal_density'] +
                    cluster_metrics['circular_flow_score'] * 3 +
                    cluster_metrics['avg_pd'] * 2
                )

                suspicious_clusters.append({
                    'cluster_id': int(comm_id),
                    'members': members.tolist(),
                    'size': len(members),
                    'avg_pd': cluster_metrics['avg_pd'],
                    'avg_joint_pd': cluster_metrics['avg_joint_pd'],
                    'internal_density': cluster_metrics['internal_density'],
                    'circular_flow_score': cluster_metrics['circular_flow_score'],
                    'suspicion_score': suspicion_score
                })

        # Sort by suspicion score
        suspicious_clusters.sort(key=lambda x: x['suspicion_score'], reverse=True)

        return suspicious_clusters

    def _analyze_cluster(self, members: np.ndarray) -> Dict:
        """Compute metrics for a cluster."""
        n_members = len(members)
        all_pds = (self.copula.marginal_pds
                   if self.copula.is_fitted
                   else self.persons['base_pd'].values)
        pds = all_pds[members]

        # Average marginal PD
        avg_pd = pds.mean()

        # Average joint PD (sample pairs for efficiency)
        joint_pds = []
        n_pairs = min(50, n_members * (n_members - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(n_members, size=2, replace=False)
            jp = self.copula.joint_default_probability(members[i], members[j])
            joint_pds.append(jp)
        avg_joint_pd = np.mean(joint_pds) if joint_pds else 0

        # Internal density
        adj = self.graph.adj_binary[np.ix_(members, members)]
        possible_edges = n_members * (n_members - 1)
        actual_edges = adj.sum()
        internal_density = actual_edges / possible_edges if possible_edges > 0 else 0

        # Circular flow score (detect cyclic patterns)
        circular_flow_score = self._detect_circular_patterns(members)

        return {
            'avg_pd': avg_pd,
            'avg_joint_pd': avg_joint_pd,
            'internal_density': internal_density,
            'circular_flow_score': circular_flow_score
        }

    def _detect_circular_patterns(self, members: np.ndarray) -> float:
        """
        Detect circular money flow patterns.

        Returns score 0-1 indicating likelihood of circular flows.
        """
        n = len(members)
        if n < 3:
            return 0.0

        # Count 3-cycles (A→B→C→A) using matrix multiplication:
        # A³[i,i] counts the number of directed 3-cycles passing through i.
        adj_directed = (self.graph.adj_out[np.ix_(members, members)] > 0).astype(float)

        # A² = paths of length 2; A³ = paths of length 3; diag(A³) = closed 3-cycles
        a2 = adj_directed @ adj_directed
        a3_diag = (a2 * adj_directed.T).sum(axis=1)  # equiv to diag(A²·Aᵀ) = diag(A³)
        n_cycles = a3_diag.sum() / 6.0  # each 3-cycle counted 6 times (2 dirs × 3 starts)

        # Maximum possible 3-cycles in a complete directed graph on n nodes
        max_cycles = n * (n - 1) * (n - 2) / 6.0
        return float(np.clip(n_cycles / max_cycles, 0.0, 1.0)) if max_cycles > 0 else 0.0

    def detect_circular_flows(
        self,
        min_cycle_length: int = 3,
        max_cycle_length: int = 6
    ) -> List[List[int]]:
        """
        Find circular transaction flows (A -> B -> C -> A).

        Parameters
        ----------
        min_cycle_length : int
            Minimum cycle length to detect
        max_cycle_length : int
            Maximum cycle length to detect

        Returns
        -------
        cycles : list of list
            List of detected cycles (each is list of person_ids)
        """
        cycles = []
        adj = (self.graph.adj_out > 0).astype(int)

        # DFS-based cycle detection
        visited = set()

        def find_cycles_from(start: int, path: List[int]):
            if len(path) > max_cycle_length:
                return

            current = path[-1]
            neighbors = np.where(adj[current] > 0)[0]

            for neighbor in neighbors:
                if neighbor == start and len(path) >= min_cycle_length:
                    cycles.append(path.copy())
                elif neighbor not in path and neighbor not in visited:
                    path.append(neighbor)
                    find_cycles_from(start, path)
                    path.pop()

        # Sample starting nodes for efficiency
        sample_size = min(100, self.n)
        start_nodes = np.random.choice(self.n, size=sample_size, replace=False)

        for start in start_nodes:
            if start not in visited:
                find_cycles_from(start, [start])
                visited.add(start)

        return cycles

    def get_fraud_indicators(self, person_id: int) -> Dict:
        """
        Get fraud indicators for a specific person.

        Returns
        -------
        indicators : dict
            Various fraud risk indicators
        """
        # Transaction patterns
        out_txs = self.graph.adj_out[person_id]
        in_txs = self.graph.adj_in[person_id]

        # Reciprocity (sends to same people who send to them)
        both_directions = (out_txs > 0) & (in_txs > 0)
        reciprocity = both_directions.sum() / max((out_txs > 0).sum(), 1)

        # Transaction concentration (sends to few people repeatedly)
        out_counts = self.graph.adj_count[person_id]
        if out_counts.sum() > 0:
            concentration = (out_counts ** 2).sum() / (out_counts.sum() ** 2)
        else:
            concentration = 0

        # Network position
        neighbors = np.where(self.graph.adj_binary[person_id] > 0)[0]
        all_pds = (self.copula.marginal_pds
                   if self.copula.is_fitted
                   else self.persons['base_pd'].values)
        avg_neighbor_pd = float(all_pds[neighbors].mean()) if len(neighbors) > 0 else 0.0

        return {
            'person_id': person_id,
            'reciprocity': reciprocity,
            'transaction_concentration': concentration,
            'avg_neighbor_pd': avg_neighbor_pd,
            'n_connections': len(neighbors),
            'in_volume': in_txs.sum(),
            'out_volume': out_txs.sum(),
            'volume_imbalance': abs(in_txs.sum() - out_txs.sum()) / max(in_txs.sum() + out_txs.sum(), 1)
        }


class ContagionStressTester:
    """
    Stress testing for contagion scenarios.

    Simulates default cascades and identifies critical nodes
    that could trigger systemic failures.
    """

    def __init__(self, copula_model, graph):
        """
        Initialize stress tester.

        Parameters
        ----------
        copula_model : CopulaDefaultModel
            Fitted copula model
        graph : TransactionGraph
            Transaction graph
        """
        self.copula = copula_model
        self.graph = graph
        self.n = copula_model.n

    def identify_critical_nodes(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Identify nodes whose default would cause largest cascade.

        Parameters
        ----------
        top_k : int
            Number of top critical nodes to return

        Returns
        -------
        critical : list of (node_id, cascade_multiplier)
            Nodes sorted by cascade impact
        """
        cascade_impacts = []

        for node in range(self.n):
            # Simulate cascade from this node
            cascade_result = self.simulate_cascade([node], contagion_rounds=5)
            multiplier = cascade_result['total_defaults'] / 1.0  # Per initial default

            cascade_impacts.append((node, multiplier))

        # Sort by cascade multiplier
        cascade_impacts.sort(key=lambda x: x[1], reverse=True)

        return cascade_impacts[:top_k]

    def simulate_cascade(
        self,
        initial_defaults: List[int],
        contagion_rounds: int = 5,
        contagion_threshold: float = 0.5
    ) -> Dict:
        """
        Simulate default cascade from initial defaults.

        Parameters
        ----------
        initial_defaults : list
            Initial defaulting nodes
        contagion_rounds : int
            Maximum rounds of contagion
        contagion_threshold : float
            Probability threshold for contagion default

        Returns
        -------
        result : dict
            Cascade simulation results
        """
        defaulted = set(initial_defaults)
        defaults_per_round = [len(initial_defaults)]

        for round_num in range(contagion_rounds):
            new_defaults = set()

            for node in range(self.n):
                if node in defaulted:
                    continue

                # Check contagion from defaulted neighbors
                defaulted_neighbors = [
                    d for d in defaulted
                    if self.graph.adj_binary[node, d] > 0
                ]

                if not defaulted_neighbors:
                    continue

                # Compute contagion probability
                # Higher if more neighbors defaulted and stronger connections
                contagion_prob = 0
                for neighbor in defaulted_neighbors:
                    cond_prob = self.copula.conditional_default_probability(node, neighbor)
                    weight = self.graph.adj_weighted[node, neighbor]
                    max_weight = self.graph.adj_weighted[node].max()
                    if max_weight > 0:
                        weight_factor = weight / max_weight
                    else:
                        weight_factor = 0.5
                    contagion_prob = max(contagion_prob, cond_prob * weight_factor)

                if contagion_prob > contagion_threshold:
                    if np.random.random() < contagion_prob:
                        new_defaults.add(node)

            if not new_defaults:
                break

            defaulted.update(new_defaults)
            defaults_per_round.append(len(new_defaults))

        return {
            'initial_defaults': initial_defaults,
            'total_defaults': len(defaulted),
            'cascade_multiplier': len(defaulted) / len(initial_defaults),
            'defaults_per_round': defaults_per_round,
            'n_rounds': len(defaults_per_round),
            'defaulted_nodes': list(defaulted)
        }

    def stress_scenario(
        self,
        shock_nodes: List[int],
        contagion_rounds: int = 5,
        pd_multiplier: float = 2.0
    ) -> Dict:
        """
        Run stress scenario with specific shock nodes.

        Parameters
        ----------
        shock_nodes : list
            Nodes that experience initial shock
        contagion_rounds : int
            Rounds of contagion propagation
        pd_multiplier : float
            Factor to multiply PDs during stress

        Returns
        -------
        result : dict
            Stress scenario results
        """
        # Store original PDs
        original_pds = self.copula.marginal_pds.copy()

        # Apply stress to shock nodes
        stressed_pds = original_pds.copy()
        for node in shock_nodes:
            stressed_pds[node] = min(original_pds[node] * pd_multiplier, 0.99)

        self.copula.marginal_pds = stressed_pds

        # Simulate cascade
        cascade_result = self.simulate_cascade(
            shock_nodes, contagion_rounds=contagion_rounds
        )

        # Restore original PDs
        self.copula.marginal_pds = original_pds

        return cascade_result

    def systematic_stress_test(
        self,
        shock_fraction: float = 0.1,
        n_scenarios: int = 100
    ) -> Dict:
        """
        Run multiple stress scenarios with random shocks.

        Parameters
        ----------
        shock_fraction : float
            Fraction of nodes to shock in each scenario
        n_scenarios : int
            Number of scenarios to simulate

        Returns
        -------
        summary : dict
            Summary statistics across scenarios
        """
        n_shock = max(1, int(self.n * shock_fraction))
        total_defaults = []
        cascade_multipliers = []

        for _ in range(n_scenarios):
            shock_nodes = np.random.choice(self.n, size=n_shock, replace=False).tolist()
            result = self.simulate_cascade(shock_nodes)
            total_defaults.append(result['total_defaults'])
            cascade_multipliers.append(result['cascade_multiplier'])

        return {
            'n_scenarios': n_scenarios,
            'shock_fraction': shock_fraction,
            'avg_total_defaults': np.mean(total_defaults),
            'max_total_defaults': np.max(total_defaults),
            'avg_cascade_multiplier': np.mean(cascade_multipliers),
            'max_cascade_multiplier': np.max(cascade_multipliers),
            'default_rate_mean': np.mean(total_defaults) / self.n,
            'default_rate_99pct': np.percentile(total_defaults, 99) / self.n
        }


if __name__ == '__main__':
    from data_generator import generate_network
    from graph_features import TransactionGraph
    from copula_model import CopulaDefaultModel

    print("Generating network...")
    persons, transactions = generate_network(seed=42)

    print("Building graph...")
    graph = TransactionGraph(transactions, persons)

    print("Fitting copula model...")
    corr_matrix = graph.get_correlation_matrix()
    copula = CopulaDefaultModel('clayton')
    copula.fit(persons['base_pd'].values, corr_matrix)

    print("\nInitializing risk analyzer...")
    analyzer = RiskAnalyzer(copula, graph, persons)

    print("\n" + "=" * 60)
    print("INDIVIDUAL RISK ANALYSIS")
    print("=" * 60)
    individual = analyzer.compute_individual_risks()
    print(f"\nRisk tier distribution:")
    print(individual['risk_tier'].value_counts().to_string())

    print(f"\nTop 10 riskiest individuals:")
    print(analyzer.get_top_risks(10)[[
        'person_id', 'city_name', 'risk_archetype',
        'marginal_pd', 'composite_risk_score', 'risk_tier'
    ]].to_string(index=False))

    print("\n" + "=" * 60)
    print("GROUP RISK ANALYSIS")
    print("=" * 60)
    print("\nHigh-risk group summary:")
    print(analyzer.get_group_summary().to_string(index=False))

    print("\n" + "=" * 60)
    print("PORTFOLIO RISK ANALYSIS")
    print("=" * 60)
    print("\nPortfolio metrics:")
    print(analyzer.portfolio_summary().to_string(index=False))

    print("\n" + "=" * 60)
    print("STRESS TEST")
    print("=" * 60)
    stress_results = analyzer.stress_test(pd_multiplier=2.0, correlation_boost=0.2)
    print("\nBase vs Stressed:")
    for metric in ['expected_loss', 'var_95', 'es_95']:
        base = stress_results['base'][metric]
        stressed = stress_results['stressed'][metric]
        change = stress_results['change'][metric]
        print(f"  {metric}: {base:.4f} -> {stressed:.4f} ({change:+.1%})")
