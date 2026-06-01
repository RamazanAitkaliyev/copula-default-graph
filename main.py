#!/usr/bin/env python3
"""
Copula Default Graph — end-to-end pipeline.

Steps:
  1.  Generate synthetic network (persons + transactions)
  2.  Build transaction graph and extract network features
  3.  Train Individual PD model (gradient boosting on person features)
  4.  Derive correlation matrix from graph structure
  5.  Fit Clayton copula on model-predicted PDs + correlation matrix
  6.  Run risk analysis (individual / group / portfolio)
  7.  Stress test (2× PD, boosted correlations)
  8.  Client value metrics (Sharpe, RAROC, contagion-adjusted)
  9.  Rating engine (PD → discrete rating + migration outlook)
  10. Structural PD (Merton model as second signal + early warnings)
  11. Flexible probabilities (regime-aware copula calibration)
  12. Customer profiles (per-borrower one-page risk report)
  13. Save outputs (charts + CSVs)

Run: python main.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import generate_network, get_summary_stats
from src.graph_features import TransactionGraph, get_neighbor_risk_features
from src.copula_model import CopulaDefaultModel, compare_copulas
from src.risk_metrics import RiskAnalyzer, ContagionStressTester, FraudRingDetector
from src.pd_model import IndividualPDModel
from src.client_value_metrics import ClientValueCalculator
from src.rating_engine import RatingEngine, RATING_LABELS
from src.structural_pd import StructuralPDModel
from src.flexible_probs import build_calibrator_from_portfolio
from src.customer_profile import CustomerProfiler

warnings.filterwarnings("ignore")


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main() -> None:
    np.random.seed(42)
    os.makedirs("output", exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — DATA GENERATION
    # ------------------------------------------------------------------
    _section("[1/13] Generating synthetic network data")
    persons, transactions = generate_network(seed=42)
    stats = get_summary_stats(persons, transactions)
    print(f"  Persons:                  {stats['n_persons']}")
    print(f"  Transactions:             {stats['n_transactions']}")
    print(f"  High-risk groups:         {stats['n_high_risk_groups']}")
    print(f"  High-risk group members:  {stats['n_high_risk_group_members']}")
    print(f"  Bridge individuals:       {stats['n_bridges']}")
    print()
    for city, count in stats['persons_per_city'].items():
        avg_pd = stats['pd_by_city'][city]
        print(f"    {city:8s}: {count} persons, avg base PD = {avg_pd:.2%}")

    # ------------------------------------------------------------------
    # STEP 2 — TRANSACTION GRAPH
    # ------------------------------------------------------------------
    _section("[2/13] Building transaction graph")
    graph = TransactionGraph(transactions, persons)
    net_stats = graph.get_network_stats()
    print(f"  Nodes:          {net_stats.n_nodes}")
    print(f"  Edges:          {net_stats.n_edges}")
    print(f"  Density:        {net_stats.density:.5f}")
    print(f"  Avg degree:     {net_stats.avg_degree:.2f}")
    print(f"  Avg clustering: {net_stats.avg_clustering:.4f}")
    print(f"  Components:     {net_stats.n_components}")

    # Network features for augmenting person data
    neighbor_features = get_neighbor_risk_features(graph, persons)
    persons = persons.merge(
        neighbor_features[['person_id', 'neighbor_pd_avg', 'neighbor_pd_max',
                            'n_high_risk_neighbors']],
        on='person_id', how='left'
    ).fillna(0)

    # ------------------------------------------------------------------
    # STEP 3 — INDIVIDUAL PD MODEL (trained on synthetic labels)
    # ------------------------------------------------------------------
    _section("[3/13] Training Individual PD model")
    pd_model = IndividualPDModel(model_type='gradient_boosting', feature_columns=[
        'age', 'income', 'employment_years', 'debt_to_income',
        'num_credit_lines', 'missed_payments', 'credit_utilization',
        'account_age_months',
    ])
    pd_metrics = pd_model.fit(persons, target_col='default', validation_split=0.2)
    print(f"  Train AUC: {pd_metrics['train_auc']:.4f}")
    print(f"  Val   AUC: {pd_metrics['val_auc']:.4f}")
    print(f"  Default rate (train): {pd_metrics['default_rate_train']:.3%}")

    # Replace hard-coded base_pd with model predictions
    persons['model_pd'] = pd_model.predict_proba(persons)
    print(f"\n  Model PD range: [{persons['model_pd'].min():.4f}, {persons['model_pd'].max():.4f}]")
    print(f"  Base  PD range: [{persons['base_pd'].min():.4f}, {persons['base_pd'].max():.4f}]")

    # Feature importance
    print("\n  Top-5 features by importance:")
    for feat, imp in pd_model.feature_importance_.head(5).items():
        print(f"    {feat:30s} {imp:.4f}")

    # ------------------------------------------------------------------
    # STEP 4 — CORRELATION MATRIX
    # ------------------------------------------------------------------
    _section("[4/13] Deriving correlation matrix from graph")
    corr_matrix = graph.get_correlation_matrix(
        base_corr=0.05,
        max_corr=0.6,
        same_city_boost=0.1,
        same_group_boost=0.2,
    )
    n = corr_matrix.shape[0]
    off_diag = corr_matrix[~np.eye(n, dtype=bool)]
    print(f"  Matrix shape:              {corr_matrix.shape}")
    print(f"  Average pairwise corr:     {off_diag.mean():.4f}")
    print(f"  Max pairwise corr:         {off_diag.max():.4f}")
    print(f"  Fraction > 0.2:            {(off_diag > 0.2).mean():.2%}")

    # ------------------------------------------------------------------
    # STEP 5 — COPULA MODEL
    # ------------------------------------------------------------------
    _section("[5/13] Fitting copula models")

    # Quick comparison of all 5 copula types
    comparison = compare_copulas(
        persons['model_pd'].values, corr_matrix, n_simulations=500
    )
    print(f"  {'Copula':<12} {'Theta':>8}  {'LTD':>6}  {'UTD':>6}  {'Sim DR':>7}")
    print(f"  {'-'*50}")
    for ctype, m in comparison.items():
        if 'error' in m:
            print(f"  {ctype:<12}  ERROR: {m['error'][:30]}")
        else:
            print(f"  {ctype:<12} {m['theta']:>8.4f}  {m['tail_dependence']:>6.4f}"
                  f"  {m['tail_dependence_upper']:>6.4f}  {m['sim_default_rate']:>7.4f}")

    # Use Clayton (best for default clustering)
    copula = CopulaDefaultModel("clayton")
    copula.fit(persons["model_pd"].values, corr_matrix)
    print(f"\n  Selected: Clayton  theta={copula.params.theta:.4f}"
          f"  lower-tail-dep={copula.tail_dependence():.4f}")

    # ------------------------------------------------------------------
    # STEP 6 — RISK ANALYSIS
    # ------------------------------------------------------------------
    _section("[6/13] Running risk analysis")
    exposures = persons["income"].values / persons["income"].mean()
    analyzer = RiskAnalyzer(copula, graph, persons, exposures=exposures, lgd=0.45)

    individual_risks = analyzer.compute_individual_risks()
    group_summary = analyzer.get_group_summary()
    portfolio = analyzer.compute_portfolio_risks(n_simulations=10000)

    print(f"  Portfolio Expected Loss:  {portfolio.expected_loss:.4f}")
    print(f"  VaR 95%:                  {portfolio.var_95:.4f}")
    print(f"  VaR 99%:                  {portfolio.var_99:.4f}")
    print(f"  ES 95%:                   {portfolio.es_95:.4f}")
    print(f"  ES 99%:                   {portfolio.es_99:.4f}")
    print(f"  Default correlation:      {portfolio.default_correlation:.4f}")
    print(f"  Concentration (HHI):      {portfolio.concentration_index:.6f}")
    print(f"  Tail risk ratio ES/VaR:   {portfolio.tail_risk_ratio:.4f}")

    print("\n  Risk tier distribution:")
    tier_counts = individual_risks['risk_tier'].value_counts()
    for tier in ['critical', 'high', 'medium', 'low']:
        n_tier = tier_counts.get(tier, 0)
        print(f"    {tier:8s}: {n_tier:4d}  ({n_tier/len(individual_risks):.1%})")

    if not group_summary.empty:
        print(f"\n  High-risk group summary ({len(group_summary)} groups):")
        print(group_summary.to_string(index=False))

    # Bridge analysis
    bridge_analysis = analyzer.get_bridge_analysis()
    print(f"\n  Bridge analysis:")
    print(bridge_analysis.to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 7 — STRESS TESTING
    # ------------------------------------------------------------------
    _section("[7/13] Stress testing")
    stress = analyzer.stress_test(pd_multiplier=2.0, correlation_boost=0.2)
    print(f"  {'Metric':<20} {'Base':>10}  {'Stressed':>10}  {'Change':>8}")
    print(f"  {'-'*52}")
    for metric in ['expected_loss', 'var_95', 'es_95']:
        base = stress['base'][metric]
        stressed = stress['stressed'][metric]
        change = stress['change'][metric]
        print(f"  {metric:<20} {base:>10.4f}  {stressed:>10.4f}  {change:>+8.1%}")

    # Contagion cascade — find the 5 most systemically important nodes
    print("\n  Identifying critical cascade nodes...")
    cascade_tester = ContagionStressTester(copula, graph)
    critical_nodes = cascade_tester.identify_critical_nodes(top_k=5)
    print(f"  {'Node':>6}  {'Cascade multiplier':>20}  {'City':>8}  {'PD':>6}")
    for node_id, mult in critical_nodes:
        row = persons.iloc[node_id]
        print(f"  {node_id:>6}  {mult:>20.2f}  {row['city_name']:>8}  {row['base_pd']:>6.3f}")

    # Fraud ring detection
    print("\n  Detecting suspicious clusters...")
    fraud_detector = FraudRingDetector(graph, copula, persons)
    suspicious = fraud_detector.detect_suspicious_clusters(
        min_cluster_size=3, joint_pd_threshold=0.10, density_threshold=0.3
    )
    if suspicious:
        print(f"  Found {len(suspicious)} suspicious clusters. Top 3:")
        for clust in suspicious[:3]:
            print(f"    cluster {clust['cluster_id']:>2}: size={clust['size']:>3}"
                  f"  avg_pd={clust['avg_pd']:.3f}"
                  f"  density={clust['internal_density']:.3f}"
                  f"  score={clust['suspicion_score']:.3f}")
    else:
        print("  No suspicious clusters above thresholds.")

    # ------------------------------------------------------------------
    # STEP 8 — CLIENT VALUE METRICS
    # ------------------------------------------------------------------
    _section("[8/13] Client value metrics")
    client_calc = ClientValueCalculator(copula, persons, transactions, lgd=0.45)
    client_metrics = client_calc.compute_contagion_adjusted_sharpe()

    print(f"  {'Segment':<16}  {'Count':>6}  {'Avg Sharpe':>10}  {'Avg RAROC':>10}")
    print(f"  {'-'*46}")
    segments = client_calc.segment_clients()
    for seg_name in ['Stars', 'Cash Cows', 'Question Marks', 'Dogs']:
        seg = segments[segments['segment'] == seg_name]
        if len(seg):
            avg_sharpe = seg['client_sharpe'].mean()
            avg_raroc = seg['raroc'].mean()
            print(f"  {seg_name:<16}  {len(seg):>6}  {avg_sharpe:>10.3f}  {avg_raroc:>10.3f}")

    top_clients = client_calc.client_ranking(method='contagion').head(10)
    print(f"\n  Top 10 clients (contagion-adjusted Sharpe):")
    print(top_clients[['person_id', 'expected_revenue', 'expected_loss',
                        'client_sharpe', 'contagion_adjusted_sharpe', 'raroc']
                       ].to_string(index=False))

    # ------------------------------------------------------------------
    # 9. RATING ENGINE
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STEP 9 — RATING ENGINE
    # ------------------------------------------------------------------
    _section("[9/13] Rating engine — PD → discrete ratings + migration outlook")
    rating_engine = RatingEngine()
    rating_engine.fit(persons, pd_col="model_pd")
    dist = rating_engine.portfolio_distribution()
    print(f"  {'Rating':<8}  {'Count':>6}  {'Fraction':>9}")
    print(f"  {'-'*28}")
    for lbl in RATING_LABELS:
        bar = "█" * int(dist.fractions[lbl] * 30)
        print(f"  {lbl:<8}  {dist.counts[lbl]:>6}  {dist.fractions[lbl]:>8.1%}  {bar}")
    print(f"\n  Weighted avg PD:    {dist.weighted_avg_pd:.3%}")
    print(f"  Migration risk:     {dist.migration_risk_score:.1%}  "
          f"(fraction expected to change rating next year)")

    # Sample migration table for top-risk borrower
    top_pid = individual_risks.head(1)["person_id"].iloc[0]
    mig = rating_engine.migration_table(top_pid)
    print(f"\n  1-year migration for borrower {top_pid} "
          f"(current: {rating_engine.get_rating_profile(top_pid).current_rating_label}):")
    for _, row in mig.iterrows():
        if row["probability"] > 0.001:
            bar = "█" * int(row["probability"] * 40)
            print(f"    → {row['to_rating']:8s}  {row['probability']:.3%}  {bar}")

    rating_summary = rating_engine.summary_df()

    # ------------------------------------------------------------------
    # 10. STRUCTURAL PD (MERTON)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STEP 10 — STRUCTURAL PD (MERTON)
    # ------------------------------------------------------------------
    _section("[10/13] Structural PD — Merton model as second signal")
    struct_model = StructuralPDModel(alpha=0.35, T=1.0, r=0.02)
    persons_enriched = struct_model.fit_transform(persons, statistical_pd_col="model_pd")

    print("  PD signal summary statistics:")
    stats_df = struct_model.summary_stats()
    print(stats_df[["mean", "50%", "90%", "max"]].to_string())

    early_warnings = struct_model.get_early_warnings(divergence_threshold=0.05)
    print(f"\n  Early warnings (Merton divergence > 5%): {len(early_warnings)} borrowers")
    if len(early_warnings) > 0:
        print(early_warnings.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # 11. FLEXIBLE PROBABILITIES / REGIME-AWARE COPULA
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STEP 11 — FLEXIBLE PROBABILITIES / REGIME-AWARE COPULA
    # ------------------------------------------------------------------
    _section("[11/13] Flexible probabilities — regime-aware copula calibration")

    # Simulate 36-month portfolio average PD history
    np.random.seed(0)
    t_hist = 36
    avg_pd_history = (persons["model_pd"].mean() *
                      (0.5 + np.linspace(0, 1, t_hist)) +
                      0.01 * np.random.randn(t_hist))
    calib = build_calibrator_from_portfolio(avg_pd_history, half_life_periods=12.0)

    scenario_table = calib.calibrate_for_scenarios(
        np.array([0.10, 0.25, 0.50, 0.75, 0.90]), corr_matrix
    )
    print("  Copula theta across stress scenarios:")
    print(scenario_table.to_string(index=False))

    current_stress = float(np.clip(
        (persons["model_pd"].mean() - 0.05) / 0.20, 0.0, 1.0
    ))
    regime_result = calib.calibrate(current_stress, corr_matrix)
    print(f"\n  Current portfolio stress:   {current_stress:.2f}  "
          f"({regime_result.regime.label})")
    print(f"  Baseline copula theta:      {copula.params.theta:.4f}")
    print(f"  Regime-adjusted theta:      {regime_result.theta:.4f}  "
          f"({'↑ tighter' if regime_result.theta > copula.params.theta else '↓ looser'})")
    if regime_result.loadings is not None:
        print(f"  Systematic factor loadings: {regime_result.loadings.shape}")

    # ------------------------------------------------------------------
    # 12. CUSTOMER PROFILES
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STEP 12 — CUSTOMER PROFILES
    # ------------------------------------------------------------------
    _section("[12/13] Customer profiles — per-borrower risk reports")
    profiler = CustomerProfiler(early_warning_threshold=0.05)
    profiler.fit(
        persons=persons_enriched,
        transactions=transactions,
        graph=graph,
        copula=copula,
        pd_model=pd_model,
        rating_engine=rating_engine,
        structural_model=persons_enriched,
        client_value_calc=client_calc,
        individual_risks=individual_risks,
    )

    # Print profiles for top 3 riskiest borrowers
    top3_pids = individual_risks.head(3)["person_id"].tolist()
    for pid in top3_pids:
        print(profiler.profile_report(pid))

    # Build full watchlist
    watchlist_df = profiler.watchlist(tiers=["critical", "high"], top_n=30)
    print(f"\n  Watchlist: {len(watchlist_df)} borrowers flagged (critical + high tier)")

    # ------------------------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STEP 13 — OUTPUTS
    # ------------------------------------------------------------------
    _section("[13/13] Saving outputs")

    # Network visualisation coloured by model PD
    persons_aug = persons.copy()
    persons_aug['model_pd_plot'] = persons['model_pd']
    graph.persons = persons_aug   # let plot_network use model_pd
    fig1 = graph.plot_network(
        color_by='base_pd',
        size_by='degree',
        layout='city',
        title='Transaction Network (colour = base PD, size = degree)',
        figsize=(13, 9),
        edge_alpha=0.04,
    )
    fig1.savefig("output/network_by_pd.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved output/network_by_pd.png")

    # Loss distribution
    losses = analyzer.get_loss_distribution(n_simulations=10000)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(losses, bins=60, density=True, alpha=0.75, color='steelblue')
    ax2.axvline(portfolio.var_95,  color='orange', linestyle='--',
                label=f"VaR 95% = {portfolio.var_95:.3f}")
    ax2.axvline(portfolio.var_99,  color='darkorange', linestyle=':',
                label=f"VaR 99% = {portfolio.var_99:.3f}")
    ax2.axvline(portfolio.es_95,   color='red',    linestyle='--',
                label=f"ES 95%  = {portfolio.es_95:.3f}")
    ax2.set_title("Portfolio Loss Distribution (Clayton copula, 10 000 simulations)")
    ax2.set_xlabel("Portfolio Loss")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("output/loss_distribution.png", dpi=150)
    plt.close(fig2)
    print("  Saved output/loss_distribution.png")

    # Risk heatmap — top 50 by composite score
    top50 = individual_risks.head(50)
    heat_cols = ['marginal_pd', 'contagion_vulnerability', 'systemic_importance', 'network_exposure']
    heat_data = top50[heat_cols].values
    heat_data = (heat_data - heat_data.min(0)) / (heat_data.max(0) - heat_data.min(0) + 1e-10)
    fig3, ax3 = plt.subplots(figsize=(8, 14))
    im = ax3.imshow(heat_data, aspect='auto', cmap='YlOrRd')
    ax3.set_xticks(range(len(heat_cols)))
    ax3.set_xticklabels([c.replace('_', '\n') for c in heat_cols], fontsize=9)
    ax3.set_yticks(range(len(top50)))
    ax3.set_yticklabels(
        [f"{r['person_id']} ({r['city_name'][:3]})" for _, r in top50.iterrows()],
        fontsize=7
    )
    ax3.set_title("Top 50 Individuals — Normalised Risk Dimensions")
    plt.colorbar(im, ax=ax3, shrink=0.5)
    fig3.tight_layout()
    fig3.savefig("output/risk_heatmap.png", dpi=150)
    plt.close(fig3)
    print("  Saved output/risk_heatmap.png")

    # Feature importance bar chart
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    fi = pd_model.feature_importance_.sort_values()
    ax4.barh(fi.index, fi.values, color='steelblue')
    ax4.set_title("PD Model — Feature Importance (Gradient Boosting)")
    ax4.set_xlabel("Importance")
    fig4.tight_layout()
    fig4.savefig("output/feature_importance.png", dpi=150)
    plt.close(fig4)
    print("  Saved output/feature_importance.png")

    # Copula comparison bar chart
    fig5, axes5 = plt.subplots(1, 3, figsize=(12, 4))
    copula_names = [k for k, v in comparison.items() if 'error' not in v]
    thetas = [comparison[k]['theta'] for k in copula_names]
    ltd    = [comparison[k]['tail_dependence'] for k in copula_names]
    utd    = [comparison[k]['tail_dependence_upper'] for k in copula_names]
    for ax_, vals, title_ in zip(axes5, [thetas, ltd, utd],
                                  ['Theta (dependence)', 'Lower tail dep.', 'Upper tail dep.']):
        ax_.bar(copula_names, vals, color='steelblue')
        ax_.set_title(title_)
        ax_.tick_params(axis='x', rotation=30)
    fig5.suptitle("Copula Comparison")
    fig5.tight_layout()
    fig5.savefig("output/copula_comparison.png", dpi=150)
    plt.close(fig5)
    print("  Saved output/copula_comparison.png")

    # CSV exports
    top_risks_out = individual_risks.head(20)[[
        'person_id', 'city_name', 'risk_archetype',
        'marginal_pd', 'contagion_vulnerability',
        'systemic_importance', 'composite_risk_score', 'risk_tier',
    ]]
    top_risks_out.to_csv("output/top_risks.csv", index=False)
    print("  Saved output/top_risks.csv")

    client_metrics.sort_values('contagion_adjusted_sharpe', ascending=False).head(50).to_csv(
        "output/client_value.csv", index=False
    )
    print("  Saved output/client_value.csv")

    if not group_summary.empty:
        group_summary.to_csv("output/group_risks.csv", index=False)
        print("  Saved output/group_risks.csv")

    # Stress summary CSV
    stress_df = pd.DataFrame([
        {'metric': m,
         'base': stress['base'][m],
         'stressed': stress['stressed'][m],
         'change_pct': stress['change'][m] * 100}
        for m in ['expected_loss', 'var_95', 'es_95']
    ])
    stress_df.to_csv("output/stress_test.csv", index=False)
    print("  Saved output/stress_test.csv")

    # Rating summary
    rating_summary.to_csv("output/rating_summary.csv", index=False)
    print("  Saved output/rating_summary.csv")

    # Structural / Merton PD enriched persons
    persons_enriched[[
        "person_id", "city_name", "risk_archetype",
        "model_pd", "merton_pd", "blended_pd",
        "distance_to_default", "pd_signal_divergence",
    ]].to_csv("output/structural_pd.csv", index=False)
    print("  Saved output/structural_pd.csv")

    # Watchlist
    watchlist_df.to_csv("output/watchlist.csv", index=False)
    print("  Saved output/watchlist.csv")

    # Regime stress table
    scenario_table.to_csv("output/regime_stress_table.csv", index=False)
    print("  Saved output/regime_stress_table.csv")

    # Rating distribution bar chart
    fig_rating, ax_rating = plt.subplots(figsize=(9, 4))
    labels  = [l for l in RATING_LABELS]
    counts  = [dist.counts[l] for l in labels]
    colors  = ["#16a34a","#22c55e","#84cc16","#eab308",
               "#f97316","#ef4444","#991b1b","#1e293b"]
    bars = ax_rating.bar(labels, counts, color=colors)
    ax_rating.set_title("Portfolio Rating Distribution")
    ax_rating.set_ylabel("Number of Borrowers")
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax_rating.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           str(cnt), ha='center', va='bottom', fontsize=9)
    fig_rating.tight_layout()
    fig_rating.savefig("output/rating_distribution.png", dpi=150)
    plt.close(fig_rating)
    print("  Saved output/rating_distribution.png")

    # Merton vs Statistical PD scatter
    fig_merton, ax_merton = plt.subplots(figsize=(7, 6))
    ax_merton.scatter(
        persons_enriched["model_pd"], persons_enriched["merton_pd"],
        alpha=0.3, s=15, color="steelblue"
    )
    lim = max(persons_enriched["model_pd"].max(), persons_enriched["merton_pd"].max()) * 1.05
    ax_merton.plot([0, lim], [0, lim], "k--", linewidth=1, label="y=x (agree)")
    ax_merton.set_xlabel("Statistical PD (model)")
    ax_merton.set_ylabel("Merton Structural PD")
    ax_merton.set_title("PD Signal Comparison: Statistical vs Structural")
    ax_merton.legend()
    fig_merton.tight_layout()
    fig_merton.savefig("output/merton_vs_statistical_pd.png", dpi=150)
    plt.close(fig_merton)
    print("  Saved output/merton_vs_statistical_pd.png")

    _section("Pipeline complete")
    print(f"  PD model Val AUC:     {pd_metrics['val_auc']:.4f}")
    print(f"  Expected loss:        {portfolio.expected_loss:.4f}")
    print(f"  VaR 95%:              {portfolio.var_95:.4f}")
    print(f"  ES 95%:               {portfolio.es_95:.4f}")
    print(f"  Stress EL change:     {stress['change']['expected_loss']:+.1%}")
    print(f"  Current regime:       {regime_result.regime.label}  "
          f"(stress={current_stress:.2f}, θ={regime_result.theta:.4f})")
    print(f"  Early warnings:       {len(early_warnings)} borrowers (Merton divergence)")
    print(f"  Watchlist size:       {len(watchlist_df)} borrowers")
    print(f"  Outputs in:           output/")


if __name__ == "__main__":
    main()
