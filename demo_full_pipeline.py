#!/usr/bin/env python3
"""
Full Pipeline Demonstration: Copula-Based Default Contagion Analysis

This script demonstrates the complete workflow from data generation to risk analysis,
showing how each component works and how the models behave with synthetic data.

Run with: python demo_full_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Import our modules
from src import (
    generate_network,
    get_summary_stats,
    TransactionGraph,
    CopulaDefaultModel,
    compare_copulas,
    RiskAnalyzer,
    PipelineConfig,
)
from src.risk_metrics import FraudRingDetector, ContagionStressTester


def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def main():
    """Run the full demonstration pipeline."""

    print_header("COPULA-BASED DEFAULT CONTAGION ANALYSIS", "█")
    print(f"Demonstration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # STEP 1: DATA GENERATION
    # =========================================================================
    print_header("STEP 1: SYNTHETIC DATA GENERATION")

    print("Generating synthetic network with:")
    print("  - 1000 persons across 3 cities")
    print("  - 4 high-risk groups (clusters of risky individuals)")
    print("  - 15 bridge individuals (connect different cities)")
    print("  - ~8000 transactions")

    persons, transactions = generate_network(seed=42)
    stats = get_summary_stats(persons, transactions)

    print_subheader("Network Summary")
    print(f"  Total persons: {stats['n_persons']}")
    print(f"  Total transactions: {stats['n_transactions']}")
    print(f"  High-risk groups: {stats['n_high_risk_groups']}")
    print(f"  High-risk group members: {stats['n_high_risk_group_members']}")
    print(f"  Bridge individuals: {stats['n_bridges']}")

    print_subheader("By City")
    for city, count in stats['persons_per_city'].items():
        avg_pd = stats['pd_by_city'][city]
        print(f"  {city}: {count} persons, avg PD = {avg_pd:.2%}")

    print_subheader("By Risk Archetype")
    for archetype, count in stats['risk_archetype_counts'].items():
        avg_pd = stats['pd_by_archetype'][archetype]
        print(f"  {archetype}: {count} persons, avg PD = {avg_pd:.2%}")

    print_subheader("Transaction Patterns")
    print(f"  Within-city transactions: {stats['within_city_tx_pct']:.1%}")
    print(f"  Within high-risk group transactions: {stats['within_group_tx_pct']:.1%}")

    # Show sample persons
    print_subheader("Sample Persons (first 10)")
    sample_cols = ['person_id', 'city_name', 'risk_archetype', 'base_pd',
                   'high_risk_group_id', 'is_bridge']
    print(persons[sample_cols].head(10).to_string())

    # =========================================================================
    # STEP 2: GRAPH CONSTRUCTION
    # =========================================================================
    print_header("STEP 2: TRANSACTION GRAPH CONSTRUCTION")

    print("Building graph from transaction data...")
    print("  - Creating adjacency matrices (binary and weighted)")
    print("  - Computing centrality measures (degree, PageRank, betweenness)")
    print("  - Deriving correlation matrix from network structure")

    graph = TransactionGraph(transactions, persons)
    net_stats = graph.get_network_stats()

    print_subheader("Network Statistics")
    print(f"  Nodes: {net_stats.n_nodes}")
    print(f"  Edges: {net_stats.n_edges}")
    print(f"  Density: {net_stats.density:.4f} (fraction of possible edges)")
    print(f"  Average degree: {net_stats.avg_degree:.1f} (connections per person)")
    print(f"  Average clustering: {net_stats.avg_clustering:.3f}")
    print(f"  Connected components: {net_stats.n_components}")

    # Derive correlation matrix
    corr_matrix = graph.get_correlation_matrix(
        base_corr=0.05,        # Minimum correlation
        max_corr=0.60,         # Maximum correlation
        same_city_boost=0.10,  # Extra correlation for same city
        same_group_boost=0.20, # Extra correlation for same group
    )

    # Compute correlation statistics
    n = corr_matrix.shape[0]
    off_diag = corr_matrix[~np.eye(n, dtype=bool)]

    print_subheader("Correlation Matrix Statistics")
    print(f"  Average pairwise correlation: {off_diag.mean():.4f}")
    print(f"  Correlation range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    print(f"  Std of correlations: {off_diag.std():.4f}")

    # Show node features
    print_subheader("Node Features (centrality measures)")
    print(graph.node_features.describe().round(3).to_string())

    # =========================================================================
    # STEP 3: COPULA MODEL COMPARISON
    # =========================================================================
    print_header("STEP 3: COPULA MODEL COMPARISON")

    print("Comparing 5 copula types to understand their behavior:")
    print("  - Gaussian: No tail dependence (for normal times)")
    print("  - Student-t: Symmetric tail dependence")
    print("  - Clayton: LOWER tail dependence (defaults cluster in crisis)")
    print("  - Gumbel: UPPER tail dependence (survival clustering)")
    print("  - Frank: No tail dependence, symmetric")

    marginal_pds = persons['base_pd'].values
    comparison = compare_copulas(marginal_pds, corr_matrix, n_simulations=5000)

    print_subheader("Copula Comparison Results")
    print(f"{'Copula':<12} {'Theta':>10} {'Lower λ':>10} {'Upper λ':>10} {'Default Rate':>12}")
    print("-" * 56)

    for copula_type, metrics in comparison.items():
        if 'error' in metrics:
            print(f"{copula_type:<12} ERROR: {metrics['error']}")
        else:
            print(f"{copula_type:<12} {metrics['theta']:>10.4f} "
                  f"{metrics['tail_dependence']:>10.4f} "
                  f"{metrics['tail_dependence_upper']:>10.4f} "
                  f"{metrics['sim_default_rate']:>12.4f}")

    print("\nInterpretation:")
    print("  - Lower tail dependence (λ_L): Probability of joint default in extreme stress")
    print("  - Clayton has the highest λ_L, making it best for credit risk")
    print("  - Gaussian and Frank have λ_L = 0 (no tail clustering)")

    # =========================================================================
    # STEP 4: FIT CLAYTON COPULA (PRIMARY MODEL)
    # =========================================================================
    print_header("STEP 4: FITTING CLAYTON COPULA MODEL")

    print("Clayton copula is ideal for credit risk because:")
    print("  1. Defaults tend to cluster during economic downturns")
    print("  2. When one person defaults, connected people are more likely to default")
    print("  3. Clayton captures this 'lower tail dependence'")

    copula = CopulaDefaultModel('clayton')
    copula.fit(marginal_pds, corr_matrix)

    print_subheader("Fitted Parameters")
    print(f"  Copula type: {copula.copula_type}")
    print(f"  Theta parameter: {copula.params.theta:.4f}")
    print(f"  Lower tail dependence λ_L: {copula.tail_dependence('lower'):.4f}")
    print(f"  Upper tail dependence λ_U: {copula.tail_dependence('upper'):.4f}")

    # Show joint default probabilities
    print_subheader("Joint Default Probability Examples")
    print("Comparing independent vs copula-adjusted joint probabilities:")
    print()

    # Select a few pairs with different characteristics
    example_pairs = [
        (0, 1, "Same city, low correlation"),
        (0, 100, "Different city, medium correlation"),
    ]

    # Find high correlation pair
    for i in range(100):
        for j in range(i+1, 100):
            if corr_matrix[i, j] > 0.4:
                example_pairs.append((i, j, "High correlation pair"))
                break
        else:
            continue
        break

    print(f"{'Pair':<15} {'PD_i':>8} {'PD_j':>8} {'Independent':>12} {'Copula':>12} {'Ratio':>8}")
    print("-" * 65)

    for i, j, desc in example_pairs:
        pd_i = marginal_pds[i]
        pd_j = marginal_pds[j]
        indep = pd_i * pd_j
        joint = copula.joint_default_probability(i, j)
        ratio = joint / indep if indep > 0 else 0
        print(f"({i}, {j}) {desc:<12} {pd_i:>8.4f} {pd_j:>8.4f} {indep:>12.6f} {joint:>12.6f} {ratio:>8.2f}x")

    print("\nInterpretation:")
    print("  - Ratio > 1 means defaults are more likely to occur together")
    print("  - Higher correlation → higher joint default probability")

    # =========================================================================
    # STEP 5: MONTE CARLO SIMULATION
    # =========================================================================
    print_header("STEP 5: MONTE CARLO SIMULATION")

    n_simulations = 10000
    print(f"Running {n_simulations:,} Monte Carlo simulations...")
    print("  - Generate correlated uniform random variables from copula")
    print("  - Convert to defaults: U_i < PD_i means person i defaults")
    print("  - Compute portfolio losses for each scenario")

    defaults = copula.simulate_defaults(n_simulations)

    # Analyze simulation results
    default_counts = defaults.sum(axis=1)
    default_rates = defaults.mean(axis=1)

    print_subheader("Simulation Results")
    print(f"  Total simulations: {n_simulations:,}")
    print(f"  Average defaults per simulation: {default_counts.mean():.1f}")
    print(f"  Std of defaults: {default_counts.std():.1f}")
    print(f"  Min defaults: {default_counts.min()}")
    print(f"  Max defaults: {default_counts.max()}")
    print(f"  Average default rate: {default_rates.mean():.4f}")

    # Distribution of defaults
    print_subheader("Default Distribution (percentiles)")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(default_counts, p)
        print(f"  {p}th percentile: {val:.0f} defaults")

    # =========================================================================
    # STEP 6: INDIVIDUAL RISK ANALYSIS
    # =========================================================================
    print_header("STEP 6: INDIVIDUAL RISK ANALYSIS")

    print("Computing individual risk metrics:")
    print("  - Marginal PD: Individual default probability")
    print("  - Contagion vulnerability: How much PD increases if neighbors default")
    print("  - Systemic importance: How much neighbors' PD increases if this person defaults")
    print("  - Network exposure: Weighted average PD of neighbors")
    print("  - Composite risk score: Weighted combination of all factors")

    # Define exposures (loan amounts, credit limits, etc.)
    exposures = persons['income'].values / 1000  # Normalize for display

    analyzer = RiskAnalyzer(
        copula_model=copula,
        graph=graph,
        persons=persons,
        exposures=exposures,
        lgd=0.45,  # 45% loss given default
    )

    individual_risks = analyzer.compute_individual_risks()

    print_subheader("Risk Tier Distribution")
    tier_counts = individual_risks['risk_tier'].value_counts()
    for tier in ['low', 'medium', 'high', 'critical']:
        count = tier_counts.get(tier, 0)
        pct = count / len(individual_risks) * 100
        print(f"  {tier.capitalize()}: {count} persons ({pct:.1f}%)")

    print_subheader("Top 15 Riskiest Individuals")
    top_risks = individual_risks.head(15)
    display_cols = ['person_id', 'city_name', 'risk_archetype', 'marginal_pd',
                    'contagion_vulnerability', 'systemic_importance', 'composite_risk_score', 'risk_tier']
    print(top_risks[display_cols].to_string())

    print_subheader("Risk by City")
    city_risk = individual_risks.groupby('city_name').agg({
        'marginal_pd': 'mean',
        'contagion_vulnerability': 'mean',
        'systemic_importance': 'mean',
        'composite_risk_score': 'mean'
    }).round(4)
    print(city_risk.to_string())

    print_subheader("Risk by Archetype")
    arch_risk = individual_risks.groupby('risk_archetype').agg({
        'marginal_pd': 'mean',
        'contagion_vulnerability': 'mean',
        'systemic_importance': 'mean',
        'composite_risk_score': 'mean'
    }).round(4)
    print(arch_risk.to_string())

    # =========================================================================
    # STEP 7: PORTFOLIO RISK ANALYSIS
    # =========================================================================
    print_header("STEP 7: PORTFOLIO RISK ANALYSIS")

    print("Computing portfolio-level risk metrics:")
    print("  - Expected Loss (EL): Average loss across all scenarios")
    print("  - VaR: Loss that won't be exceeded with X% confidence")
    print("  - Expected Shortfall (ES): Average loss in worst X% scenarios")
    print("  - Default correlation: How correlated are defaults?")

    portfolio = analyzer.compute_portfolio_risks(n_simulations=10000)

    print_subheader("Portfolio Risk Metrics")
    print(f"  Expected Loss (EL): {portfolio.expected_loss:.2f}")
    print(f"  VaR 95%: {portfolio.var_95:.2f} (95% of losses below this)")
    print(f"  VaR 99%: {portfolio.var_99:.2f} (99% of losses below this)")
    print(f"  Expected Shortfall 95%: {portfolio.es_95:.2f} (avg loss in worst 5%)")
    print(f"  Expected Shortfall 99%: {portfolio.es_99:.2f} (avg loss in worst 1%)")
    print(f"  Default correlation: {portfolio.default_correlation:.4f}")
    print(f"  Concentration index: {portfolio.concentration_index:.4f}")
    print(f"  Tail risk ratio (ES/VaR): {portfolio.tail_risk_ratio:.2f}")

    print("\nInterpretation:")
    print(f"  - 95% of the time, losses will be below {portfolio.var_95:.2f}")
    print(f"  - When losses exceed VaR95, they average {portfolio.es_95:.2f}")
    print(f"  - Tail risk ratio > 1.5 indicates heavy tail risk")

    # =========================================================================
    # STEP 8: STRESS TESTING
    # =========================================================================
    print_header("STEP 8: STRESS TESTING")

    print("Stress testing the portfolio under adverse conditions:")
    print("  - Double all individual PDs (economic crisis)")
    print("  - Increase all correlations by 0.2 (contagion spreads)")

    stress_results = analyzer.stress_test(
        pd_multiplier=2.0,
        correlation_boost=0.20,
    )

    print_subheader("Stress Test Results")
    print("\nBase scenario:")
    print(f"  Expected Loss: {stress_results['base']['expected_loss']:.2f}")
    print(f"  VaR 95%: {stress_results['base']['var_95']:.2f}")
    print(f"  ES 95%: {stress_results['base']['es_95']:.2f}")

    print("\nStressed scenario (PD×2, correlation+0.2):")
    print(f"  Expected Loss: {stress_results['stressed']['expected_loss']:.2f}")
    print(f"  VaR 95%: {stress_results['stressed']['var_95']:.2f}")
    print(f"  ES 95%: {stress_results['stressed']['es_95']:.2f}")

    print("\nChange from base to stressed:")
    print(f"  Expected Loss: {stress_results['change']['expected_loss']:+.1%}")
    print(f"  VaR 95%: {stress_results['change']['var_95']:+.1%}")
    print(f"  ES 95%: {stress_results['change']['es_95']:+.1%}")

    # =========================================================================
    # STEP 9: CONTAGION CASCADE ANALYSIS
    # =========================================================================
    print_header("STEP 9: CONTAGION CASCADE ANALYSIS")

    print("Simulating what happens when key nodes default...")
    print("  - Select high-risk individuals as 'shock' nodes")
    print("  - Simulate default cascade through the network")
    print("  - Measure cascade multiplier")

    stress_tester = ContagionStressTester(copula, graph)

    # Find high-risk nodes to shock
    high_risk_ids = persons[persons['risk_archetype'] == 'high']['person_id'].head(5).tolist()

    cascade_result = stress_tester.stress_scenario(
        shock_nodes=high_risk_ids,
        pd_multiplier=3.0,
    )

    print_subheader("Cascade Analysis Results")
    print(f"  Shock nodes: {len(high_risk_ids)} high-risk individuals")
    print(f"  Initial defaults: {cascade_result['initial_defaults']}")
    print(f"  Total defaults after cascade: {cascade_result['total_defaults']}")
    print(f"  Cascade multiplier: {cascade_result['cascade_multiplier']:.2f}x")

    # =========================================================================
    # STEP 10: FRAUD RING DETECTION
    # =========================================================================
    print_header("STEP 10: FRAUD RING DETECTION")

    print("Detecting suspicious clusters in the network...")
    print("  - Looking for dense clusters with high joint default probability")
    print("  - Checking for circular transaction patterns")

    detector = FraudRingDetector(graph, copula, persons)
    suspicious = detector.detect_suspicious_clusters(
        min_cluster_size=3,
        joint_pd_threshold=0.10,
        density_threshold=0.3,
    )

    print_subheader("Suspicious Clusters Found")
    if suspicious:
        for i, cluster in enumerate(suspicious[:5]):
            print(f"\n  Cluster {cluster['cluster_id']}:")
            print(f"    Size: {cluster['size']} members")
            print(f"    Internal density: {cluster['internal_density']:.3f}")
            print(f"    Avg joint default prob: {cluster['avg_joint_pd']:.4f}")
            print(f"    Avg PD: {cluster['avg_pd']:.4f}")
            print(f"    Circular flow score: {cluster['circular_flow_score']:.3f}")
            print(f"    Suspicion score: {cluster['suspicion_score']:.3f}")
    else:
        print("  No highly suspicious clusters detected")

    # =========================================================================
    # STEP 11: VISUALIZATION
    # =========================================================================
    print_header("STEP 11: GENERATING VISUALIZATIONS")

    # Create output directory if needed
    import os
    os.makedirs('output', exist_ok=True)

    # 1. Network visualization
    print("  - Creating network visualization...")
    fig = graph.plot_network(
        color_by='base_pd',
        size_by='degree',
        layout='city',
        title='Transaction Network by City (color = PD, size = degree)',
        figsize=(14, 10),
    )
    plt.savefig('output/network_by_pd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/network_by_pd.png")

    # 2. Loss distribution
    print("  - Creating loss distribution plot...")
    losses = analyzer.get_loss_distribution(n_simulations=10000)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(losses, bins=50, density=True, alpha=0.7, color='steelblue', label='Loss Distribution')
    ax.axvline(portfolio.var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95% = {portfolio.var_95:.1f}')
    ax.axvline(portfolio.es_95, color='red', linestyle='--', linewidth=2, label=f'ES 95% = {portfolio.es_95:.1f}')
    ax.axvline(portfolio.expected_loss, color='green', linestyle='-', linewidth=2, label=f'Expected Loss = {portfolio.expected_loss:.1f}')
    ax.set_xlabel('Portfolio Loss', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Portfolio Loss Distribution with Risk Metrics', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.savefig('output/loss_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/loss_distribution.png")

    # 3. Risk heatmap by city and archetype
    print("  - Creating risk heatmap...")
    pivot = individual_risks.pivot_table(
        values='composite_risk_score',
        index='city_name',
        columns='risk_archetype',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Risk Archetype', fontsize=12)
    ax.set_ylabel('City', fontsize=12)
    ax.set_title('Average Composite Risk Score by City and Archetype', fontsize=14)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax, label='Composite Risk Score')
    plt.savefig('output/risk_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/risk_heatmap.png")

    # 4. Save top risks to CSV
    print("  - Saving top risks to CSV...")
    top_risks.to_csv('output/top_risks.csv', index=False)
    print("    Saved: output/top_risks.csv")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY", "█")

    print("Key Findings:")
    print()
    print(f"  1. NETWORK STRUCTURE")
    print(f"     - {net_stats.n_nodes} persons, {net_stats.n_edges} transaction links")
    print(f"     - Average {net_stats.avg_degree:.1f} connections per person")
    print(f"     - {stats['within_city_tx_pct']:.0%} of transactions within same city")
    print()
    print(f"  2. DEFAULT RISK")
    print(f"     - Average PD: {marginal_pds.mean():.2%}")
    print(f"     - PD range: [{marginal_pds.min():.2%}, {marginal_pds.max():.2%}]")
    print(f"     - Tail dependence (Clayton): {copula.tail_dependence():.4f}")
    print()
    print(f"  3. PORTFOLIO RISK")
    print(f"     - Expected Loss: {portfolio.expected_loss:.2f}")
    print(f"     - VaR 95%: {portfolio.var_95:.2f}")
    print(f"     - ES 95%: {portfolio.es_95:.2f}")
    print(f"     - Tail risk ratio: {portfolio.tail_risk_ratio:.2f}")
    print()
    print(f"  4. STRESS RESILIENCE")
    print(f"     - Under stress (PD×2, corr+0.2):")
    print(f"       Expected Loss increases by {stress_results['change']['expected_loss']:+.0%}")
    print()
    print(f"  5. CONTAGION RISK")
    print(f"     - Cascade multiplier: {cascade_result['cascade_multiplier']:.2f}x")
    print(f"     - Critical persons (high systemic importance): {(individual_risks['risk_tier'] == 'critical').sum()}")
    print()

    print(f"Outputs saved to: output/")
    print(f"  - network_by_pd.png: Network visualization")
    print(f"  - loss_distribution.png: Loss distribution with VaR/ES")
    print(f"  - risk_heatmap.png: Risk by city and archetype")
    print(f"  - top_risks.csv: Top 15 riskiest individuals")
    print()
    print(f"Demonstration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_header("END OF DEMONSTRATION", "█")


if __name__ == '__main__':
    main()
