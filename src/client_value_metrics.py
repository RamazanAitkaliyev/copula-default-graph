"""
Client Value Metrics with Risk Adjustment

Creates a "Client Sharpe Ratio" and related metrics:
- Risk-adjusted client value (RACV)
- Client RAROC (Risk-Adjusted Return on Capital)
- Portfolio optimization for client selection
- Expected profit accounting for default risk

Key insight: Not all revenue is equal - high-revenue clients with high
default risk may be less valuable than moderate-revenue low-risk clients.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class ClientMetrics:
    """Container for client-level metrics."""
    person_id: int
    expected_revenue: float
    expected_loss: float
    expected_profit: float
    risk_adjusted_value: float  # "Client Sharpe"
    raroc: float
    cltv_risk_adjusted: float


class ClientValueCalculator:
    """
    Calculate risk-adjusted client value metrics.

    Combines:
    - Revenue/CLTV estimates
    - Default probability
    - Expected loss given default
    - Network contagion effects
    """

    def __init__(
        self,
        copula_model,
        persons: pd.DataFrame,
        transactions: pd.DataFrame,
        lgd: float = 0.45,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize client value calculator.

        Parameters
        ----------
        copula_model : CopulaContagionModel
            Fitted copula model with default probabilities
        persons : pd.DataFrame
            Person data
        transactions : pd.DataFrame
            Transaction history
        lgd : float
            Loss given default (fraction of exposure lost)
        risk_free_rate : float
            Risk-free rate for Sharpe-like calculations
        """
        self.copula = copula_model
        self.persons = persons.copy()
        self.transactions = transactions
        self.lgd = lgd
        self.risk_free_rate = risk_free_rate

        # Compute revenue metrics
        self._compute_revenue_metrics()

    def _compute_revenue_metrics(self):
        """Compute client revenue and exposure metrics from transactions."""
        # Revenue = fees/interest from incoming transactions
        # Exposure = outgoing transaction volume (credit exposure)

        incoming = self.transactions.groupby('receiver_id')['amount'].sum()
        outgoing = self.transactions.groupby('sender_id')['amount'].sum()

        self.persons['transaction_volume_in'] = self.persons['person_id'].map(incoming).fillna(0)
        self.persons['transaction_volume_out'] = self.persons['person_id'].map(outgoing).fillna(0)

        # Simple revenue model: fee on transactions
        fee_rate = 0.02  # 2% fee
        self.persons['estimated_revenue'] = (
            self.persons['transaction_volume_in'] * fee_rate +
            self.persons['transaction_volume_out'] * fee_rate * 0.5
        )

        # Exposure at default (credit line approximation)
        self.persons['exposure_at_default'] = (
            self.persons['transaction_volume_out'] * 0.3 +  # Credit exposure
            self.persons['income'] * 0.1  # Credit line based on income
        )

        # CLTV estimate (simplified: 3-year projection)
        years = 3
        retention_rate = 0.85
        self.persons['cltv'] = sum(
            self.persons['estimated_revenue'] * (retention_rate ** y)
            for y in range(years)
        )

    def compute_client_sharpe(self) -> pd.DataFrame:
        """
        Compute Client Sharpe Ratio for each client.

        Client Sharpe = (Expected Profit - Risk-Free Return) / Risk

        Where:
        - Expected Profit = Revenue × (1 - PD) - Expected Loss
        - Expected Loss = PD × EAD × LGD
        - Risk = Std(Profit) ≈ EAD × LGD × √(PD × (1-PD))
        """
        results = []

        for _, person in self.persons.iterrows():
            pid = int(person['person_id'])
            pd_val = self.copula.marginal_pds[pid] if pid < len(self.copula.marginal_pds) else 0.1

            revenue = person['estimated_revenue']
            ead = person['exposure_at_default']
            cltv = person['cltv']

            # Expected loss = PD × EAD × LGD
            expected_loss = pd_val * ead * self.lgd

            # E[Profit] = (1-PD)×Revenue + PD×(Revenue - EAD×LGD)
            #           = Revenue - PD×EAD×LGD = Revenue - E[Loss]
            expected_profit = revenue - expected_loss

            # Profit volatility (risk)
            # Variance = EAD² × LGD² × PD × (1-PD) (from Bernoulli)
            profit_std = ead * self.lgd * np.sqrt(pd_val * (1 - pd_val) + 0.01)

            # Client Sharpe Ratio
            excess_return = expected_profit - revenue * self.risk_free_rate
            sharpe = excess_return / profit_std if profit_std > 0 else 0

            # RAROC (Risk-Adjusted Return on Capital)
            # Capital = some fraction of EAD for regulatory purposes
            capital = ead * 0.08  # 8% capital requirement
            raroc = expected_profit / capital if capital > 0 else 0

            # Risk-adjusted CLTV
            # Discount CLTV by survival probability
            survival_3yr = (1 - pd_val) ** 3
            cltv_risk_adjusted = cltv * survival_3yr - expected_loss * 3

            results.append(ClientMetrics(
                person_id=pid,
                expected_revenue=revenue,
                expected_loss=expected_loss,
                expected_profit=expected_profit,
                risk_adjusted_value=sharpe,
                raroc=raroc,
                cltv_risk_adjusted=cltv_risk_adjusted
            ))

        # Convert to DataFrame
        metrics_df = pd.DataFrame([
            {
                'person_id': m.person_id,
                'expected_revenue': m.expected_revenue,
                'expected_loss': m.expected_loss,
                'expected_profit': m.expected_profit,
                'client_sharpe': m.risk_adjusted_value,
                'raroc': m.raroc,
                'cltv_risk_adjusted': m.cltv_risk_adjusted
            }
            for m in results
        ])

        return metrics_df

    def compute_contagion_adjusted_sharpe(self) -> pd.DataFrame:
        """
        Compute Client Sharpe with contagion adjustment.

        Considers that high-risk neighbors increase this client's risk.
        """
        base_metrics = self.compute_client_sharpe()

        # Get contagion scores
        contagion_scores = self.copula.contagion_risk_score()

        # Adjust risk for contagion
        base_metrics['contagion_score'] = contagion_scores
        base_metrics['contagion_adjusted_loss'] = (
            base_metrics['expected_loss'] * (1 + contagion_scores * 2)
        )

        # Recalculate Sharpe with contagion-adjusted risk
        base_metrics['contagion_adjusted_sharpe'] = (
            (base_metrics['expected_revenue'] - base_metrics['contagion_adjusted_loss'] -
             base_metrics['expected_revenue'] * self.risk_free_rate) /
            (base_metrics['contagion_adjusted_loss'] + 1)
        )

        return base_metrics

    def portfolio_optimization(
        self,
        target_return: Optional[float] = None,
        max_risk: Optional[float] = None,
        max_clients: Optional[int] = None
    ) -> Dict:
        """
        Optimize client portfolio selection.

        Find optimal client weights to maximize portfolio Sharpe ratio
        subject to constraints.

        Parameters
        ----------
        target_return : float, optional
            Minimum required return
        max_risk : float, optional
            Maximum portfolio risk
        max_clients : int, optional
            Maximum number of clients to select

        Returns
        -------
        result : dict
            Optimal weights and portfolio metrics
        """
        metrics = self.compute_client_sharpe()
        n = len(metrics)

        expected_returns = metrics['expected_profit'].values
        risks = metrics['expected_loss'].values + 1  # Avoid zero

        # Correlation from copula
        corr = self.copula.correlation_matrix

        # Portfolio optimization: maximize Sharpe
        def portfolio_risk(weights):
            # Portfolio variance with correlation
            var = weights @ (risks[:, np.newaxis] * corr * risks) @ weights
            return np.sqrt(var)

        def portfolio_return(weights):
            return weights @ expected_returns

        def neg_sharpe(weights):
            ret = portfolio_return(weights)
            risk = portfolio_risk(weights) + 0.01
            return -(ret - self.risk_free_rate * weights.sum()) / risk

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: portfolio_return(w) - target_return
            })

        if max_risk is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_risk - portfolio_risk(w)
            })

        # Bounds: 0 to 1 for each weight
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess: equal weights
        w0 = np.ones(n) / n

        # Optimize
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        optimal_weights = result.x

        # If max_clients specified, keep only top weighted
        if max_clients is not None and max_clients < n:
            top_idx = np.argsort(optimal_weights)[-max_clients:]
            mask = np.zeros(n)
            mask[top_idx] = 1
            optimal_weights = optimal_weights * mask
            optimal_weights = optimal_weights / optimal_weights.sum()

        return {
            'optimal_weights': optimal_weights,
            'selected_clients': np.where(optimal_weights > 0.01)[0],
            'portfolio_return': portfolio_return(optimal_weights),
            'portfolio_risk': portfolio_risk(optimal_weights),
            'portfolio_sharpe': -neg_sharpe(optimal_weights),
            'n_clients': (optimal_weights > 0.01).sum()
        }

    def client_ranking(self, method: str = 'sharpe') -> pd.DataFrame:
        """
        Rank clients by risk-adjusted value.

        Parameters
        ----------
        method : str
            'sharpe' - Client Sharpe ratio
            'raroc' - Risk-adjusted return on capital
            'cltv' - Risk-adjusted CLTV
            'contagion' - Contagion-adjusted Sharpe

        Returns
        -------
        ranking : pd.DataFrame
            Clients sorted by chosen metric
        """
        if method == 'contagion':
            metrics = self.compute_contagion_adjusted_sharpe()
            sort_col = 'contagion_adjusted_sharpe'
        else:
            metrics = self.compute_client_sharpe()
            sort_col = {
                'sharpe': 'client_sharpe',
                'raroc': 'raroc',
                'cltv': 'cltv_risk_adjusted'
            }.get(method, 'client_sharpe')

        return metrics.sort_values(sort_col, ascending=False)

    def segment_clients(self, n_segments: int = 4) -> pd.DataFrame:
        """
        Segment clients into risk-return buckets.

        Returns DataFrame with segment labels:
        - "Stars": High return, Low risk
        - "Question Marks": High return, High risk
        - "Cash Cows": Low return, Low risk
        - "Dogs": Low return, High risk
        """
        metrics = self.compute_client_sharpe()

        # Calculate percentiles
        revenue_median = metrics['expected_revenue'].median()
        risk_median = metrics['expected_loss'].median()

        def assign_segment(row):
            high_return = row['expected_revenue'] > revenue_median
            high_risk = row['expected_loss'] > risk_median

            if high_return and not high_risk:
                return 'Stars'
            elif high_return and high_risk:
                return 'Question Marks'
            elif not high_return and not high_risk:
                return 'Cash Cows'
            else:
                return 'Dogs'

        metrics['segment'] = metrics.apply(assign_segment, axis=1)

        # Add quantile-based segments
        metrics['risk_quantile'] = pd.qcut(metrics['expected_loss'], n_segments, labels=False)
        metrics['return_quantile'] = pd.qcut(metrics['expected_revenue'], n_segments, labels=False)

        return metrics


class ClientPortfolioAnalyzer:
    """
    Analyze a portfolio of clients as if it were a financial portfolio.
    """

    def __init__(self, client_value_calc: ClientValueCalculator):
        self.calc = client_value_calc

    def efficient_frontier(self, n_points: int = 20) -> pd.DataFrame:
        """
        Compute efficient frontier of client portfolios.

        Returns
        -------
        frontier : pd.DataFrame
            Risk-return pairs on the efficient frontier
        """
        metrics = self.calc.compute_client_sharpe()

        min_return = metrics['expected_profit'].min()
        max_return = metrics['expected_profit'].max()

        target_returns = np.linspace(min_return * 1.1, max_return * 0.9, n_points)

        frontier_points = []
        for target in target_returns:
            try:
                result = self.calc.portfolio_optimization(target_return=target)
                frontier_points.append({
                    'target_return': target,
                    'portfolio_return': result['portfolio_return'],
                    'portfolio_risk': result['portfolio_risk'],
                    'portfolio_sharpe': result['portfolio_sharpe'],
                    'n_clients': result['n_clients']
                })
            except Exception:
                continue

        return pd.DataFrame(frontier_points)

    def attribution_analysis(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Analyze contribution of each client to portfolio metrics.
        """
        metrics = self.calc.compute_client_sharpe()
        n = len(metrics)

        # Marginal contribution
        contributions = []
        threshold = 1.0 / (2 * n) if n > 0 else 0.0
        for i in range(n):
            if weights[i] > threshold:
                contrib = {
                    'person_id': metrics.iloc[i]['person_id'],
                    'weight': weights[i],
                    'return_contribution': weights[i] * metrics.iloc[i]['expected_profit'],
                    'risk_contribution': weights[i] * metrics.iloc[i]['expected_loss'],
                    'sharpe_contribution': weights[i] * metrics.iloc[i]['client_sharpe']
                }
                contributions.append(contrib)

        if not contributions:
            return pd.DataFrame(columns=['person_id', 'weight', 'return_contribution',
                                         'risk_contribution', 'sharpe_contribution'])
        return pd.DataFrame(contributions).sort_values('return_contribution', ascending=False)


if __name__ == '__main__':
    from data_generator import generate_network
    from graph_features import TransactionGraph
    from copula_model import CopulaDefaultModel

    print("Generating data...")
    persons, transactions = generate_network(seed=42)

    print("Building graph + copula...")
    graph = TransactionGraph(transactions, persons)
    corr = graph.get_correlation_matrix()
    copula = CopulaDefaultModel('clayton')
    copula.fit(persons['base_pd'].values, corr)

    print("Computing client value metrics...")
    calc = ClientValueCalculator(copula, persons, transactions)
    ranking = calc.client_ranking(method='contagion')
    print(ranking.head(10).to_string(index=False))

    print("\n" + "=" * 50)
    print("CLIENT VALUE METRICS")
    print("=" * 50)

    # Calculate client metrics
    calc = ClientValueCalculator(copula, persons, transactions)

    # Basic Sharpe
    print("\nClient Sharpe Ratios (Top 10):")
    sharpe_metrics = calc.compute_client_sharpe()
    print(sharpe_metrics.nlargest(10, 'client_sharpe')[
        ['person_id', 'expected_revenue', 'expected_loss', 'client_sharpe', 'raroc']
    ])

    # Contagion-adjusted
    print("\nContagion-Adjusted Sharpe (Top 10):")
    contagion_metrics = calc.compute_contagion_adjusted_sharpe()
    print(contagion_metrics.nlargest(10, 'contagion_adjusted_sharpe')[
        ['person_id', 'contagion_score', 'contagion_adjusted_sharpe']
    ])

    # Client segmentation
    print("\nClient Segmentation:")
    segments = calc.segment_clients()
    print(segments['segment'].value_counts())

    # Portfolio optimization
    print("\n" + "=" * 50)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 50)

    result = calc.portfolio_optimization(max_clients=20)
    print(f"Optimal portfolio Sharpe: {result['portfolio_sharpe']:.3f}")
    print(f"Selected clients: {result['n_clients']}")
    print(f"Portfolio return: {result['portfolio_return']:.2f}")
    print(f"Portfolio risk: {result['portfolio_risk']:.2f}")

    # Efficient frontier
    print("\nEfficient Frontier:")
    analyzer = ClientPortfolioAnalyzer(calc)
    frontier = analyzer.efficient_frontier(n_points=10)
    print(frontier)

    # Summary stats
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Average Client Sharpe: {sharpe_metrics['client_sharpe'].mean():.3f}")
    print(f"Average RAROC: {sharpe_metrics['raroc'].mean():.3f}")
    print(f"Average Risk-Adjusted CLTV: ${sharpe_metrics['cltv_risk_adjusted'].mean():,.2f}")
    print(f"Total Expected Revenue: ${sharpe_metrics['expected_revenue'].sum():,.2f}")
    print(f"Total Expected Loss: ${sharpe_metrics['expected_loss'].sum():,.2f}")
    print(f"Net Expected Profit: ${sharpe_metrics['expected_profit'].sum():,.2f}")
