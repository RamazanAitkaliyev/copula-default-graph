#!/usr/bin/env python3
"""
Quick diagnostic helpers for development and debugging.

Usage:
    python debug.py smoke              # fast pipeline smoke test (no plots)
    python debug.py copula             # print copula state summary
    python debug.py profile <pid>      # print full risk profile for a borrower
    python debug.py stress             # run stress test and show delta table
    python debug.py ratings            # show rating distribution
    python debug.py graph              # print graph statistics
    python debug.py test <name>        # run one named test from test_copula_framework.py
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


def _build_pipeline(n_persons: int = 100, seed: int = 7):
    """Build a small but fully wired pipeline for fast debugging."""
    from src.data_generator import generate_network
    from src.graph_features import TransactionGraph, get_neighbor_risk_features
    from src.copula_model import CopulaDefaultModel
    from src.risk_metrics import RiskAnalyzer
    from src.pd_model import IndividualPDModel

    persons, transactions = generate_network(seed=seed)
    persons = persons.head(n_persons).copy()
    persons['person_id'] = range(n_persons)
    transactions = transactions[
        transactions['sender_id'].isin(persons['person_id']) &
        transactions['receiver_id'].isin(persons['person_id'])
    ]

    graph = TransactionGraph(transactions, persons)
    nb = get_neighbor_risk_features(graph, persons)
    persons = persons.merge(
        nb[['person_id', 'neighbor_pd_avg', 'neighbor_pd_max', 'n_high_risk_neighbors']],
        on='person_id', how='left'
    ).fillna(0)

    pd_model = IndividualPDModel(model_type='gradient_boosting')
    pd_model.fit(persons, target_col='default')
    persons['model_pd'] = pd_model.predict_proba(persons)

    corr = graph.get_correlation_matrix()
    copula = CopulaDefaultModel('clayton')
    copula.fit(persons['model_pd'].values, corr)

    exposures = persons['income'].values / persons['income'].mean()
    analyzer = RiskAnalyzer(copula, graph, persons, exposures=exposures, lgd=0.45)

    return persons, transactions, graph, copula, analyzer


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def cmd_smoke():
    print("Building small pipeline (n=100)...")
    persons, transactions, graph, copula, analyzer = _build_pipeline(n_persons=100)

    individual = analyzer.compute_individual_risks()
    portfolio = analyzer.compute_portfolio_risks(n_simulations=2000)
    stress = analyzer.stress_test(n_simulations=2000)

    print(f"  Persons:       {len(persons)}")
    print(f"  Copula theta:  {copula.params.theta:.4f}")
    print(f"  E[Loss]:       {portfolio.expected_loss:.4f}")
    print(f"  VaR 95%:       {portfolio.var_95:.4f}")
    print(f"  ES 95%:        {portfolio.es_95:.4f}")
    print(f"  Stress EL Δ:   {stress['change']['expected_loss']:+.1%}")
    print(f"  Risk tiers:    {individual['risk_tier'].value_counts().to_dict()}")
    print("\nSmoke test PASSED")


# ---------------------------------------------------------------------------
# copula
# ---------------------------------------------------------------------------

def cmd_copula():
    print("Building copula state summary...")
    persons, _, graph, copula, _ = _build_pipeline(n_persons=100)

    print(f"\n  Type:          {copula.params.copula_type}")
    print(f"  Theta:         {copula.params.theta:.4f}")
    print(f"  Lower tail ρ:  {copula.tail_dependence():.4f}")
    print(f"  Upper tail ρ:  {copula.tail_dependence_upper():.4f}")
    print(f"  Is fitted:     {copula.is_fitted}")
    print(f"  n marginals:   {len(copula.marginal_pds)}")
    print(f"  PD range:      [{copula.marginal_pds.min():.4f}, {copula.marginal_pds.max():.4f}]")

    corr = copula.correlation_matrix
    off = corr[~np.eye(len(corr), dtype=bool)]
    print(f"  Corr shape:    {corr.shape}")
    print(f"  Corr range:    [{off.min():.4f}, {off.max():.4f}]")
    print(f"  Corr mean:     {off.mean():.4f}")

    # Verify PSD
    eigvals = np.linalg.eigvalsh(corr)
    print(f"  Min eigenval:  {eigvals.min():.6f}  ({'PSD ✓' if eigvals.min() >= -1e-8 else 'NOT PSD ✗'})")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

def cmd_profile(pid: int):
    from src.structural_pd import StructuralPDModel
    from src.rating_engine import RatingEngine
    from src.customer_profile import CustomerProfiler
    from src.client_value_metrics import ClientValueCalculator

    print(f"Building profile for person_id={pid}...")
    persons, transactions, graph, copula, analyzer = _build_pipeline(n_persons=100)

    if pid not in persons['person_id'].values:
        print(f"  person_id {pid} not in range [0, {len(persons)-1}]. Using 0.")
        pid = 0

    struct_model = StructuralPDModel(alpha=0.35)
    persons_enriched = struct_model.fit_transform(persons, statistical_pd_col='model_pd')

    rating_engine = RatingEngine()
    rating_engine.fit(persons, pd_col='model_pd')

    client_calc = ClientValueCalculator(copula, persons, transactions, lgd=0.45)
    individual_risks = analyzer.compute_individual_risks()

    profiler = CustomerProfiler()
    profiler.fit(
        persons=persons_enriched,
        transactions=transactions,
        graph=graph,
        copula=copula,
        pd_model=None,
        rating_engine=rating_engine,
        structural_model=persons_enriched,
        client_value_calc=client_calc,
        individual_risks=individual_risks,
    )
    print(profiler.profile_report(pid))


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------

def cmd_stress():
    print("Running stress test (n=100)...")
    _, _, _, _, analyzer = _build_pipeline(n_persons=100)
    stress = analyzer.stress_test(pd_multiplier=2.0, correlation_boost=0.2, n_simulations=3000)

    print(f"\n  {'Metric':<22} {'Base':>10}  {'Stressed':>10}  {'Δ':>8}")
    print(f"  {'-'*56}")
    for m in ['expected_loss', 'var_95', 'es_95']:
        base = stress['base'][m]
        stressed = stress['stressed'][m]
        change = stress['change'][m]
        print(f"  {m:<22} {base:>10.4f}  {stressed:>10.4f}  {change:>+8.1%}")


# ---------------------------------------------------------------------------
# ratings
# ---------------------------------------------------------------------------

def cmd_ratings():
    from src.rating_engine import RatingEngine, RATING_LABELS
    print("Building rating distribution (n=100)...")
    persons, _, _, _, _ = _build_pipeline(n_persons=100)

    re = RatingEngine()
    re.fit(persons, pd_col='model_pd')
    dist = re.portfolio_distribution()

    print(f"\n  {'Rating':<8}  {'Count':>6}  {'Frac':>7}  Bar")
    print(f"  {'-'*45}")
    for lbl in RATING_LABELS:
        bar = '█' * int(dist.fractions[lbl] * 30)
        print(f"  {lbl:<8}  {dist.counts[lbl]:>6}  {dist.fractions[lbl]:>7.1%}  {bar}")
    print(f"\n  Weighted avg PD: {dist.weighted_avg_pd:.3%}")
    print(f"  Migration risk:  {dist.migration_risk_score:.1%}")


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------

def cmd_graph():
    print("Building graph statistics (n=100)...")
    _, _, graph, _, _ = _build_pipeline(n_persons=100)
    s = graph.get_network_stats()
    print(f"\n  Nodes:          {s.n_nodes}")
    print(f"  Edges:          {s.n_edges}")
    print(f"  Density:        {s.density:.5f}")
    print(f"  Avg degree:     {s.avg_degree:.2f}")
    print(f"  Avg clustering: {s.avg_clustering:.4f}")
    print(f"  Components:     {s.n_components}")


# ---------------------------------------------------------------------------
# test (single)
# ---------------------------------------------------------------------------

def cmd_test(name: str):
    import test_copula_framework as tf
    fn = getattr(tf, f"test_{name}", None)
    if fn is None:
        fn = getattr(tf, name, None)
    if fn is None:
        print(f"No test named 'test_{name}' or '{name}' found.")
        print("Available tests:")
        for attr in dir(tf):
            if attr.startswith("test_"):
                print(f"  {attr[5:]}")
        return
    try:
        fn()
    except TypeError:
        # Some tests accept fixtures — run after generating them
        persons, transactions = tf.test_data_generation()
        graph, corr = tf.test_graph(persons, transactions)
        fn(persons, transactions, graph, corr)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

COMMANDS = {
    'smoke':   (cmd_smoke,   []),
    'copula':  (cmd_copula,  []),
    'stress':  (cmd_stress,  []),
    'ratings': (cmd_ratings, []),
    'graph':   (cmd_graph,   []),
    'profile': (cmd_profile, ['pid:int']),
    'test':    (cmd_test,    ['name:str']),
}

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        print("Available commands:", ', '.join(COMMANDS))
        sys.exit(0)

    cmd = args[0]
    fn, params = COMMANDS[cmd]
    extra = args[1:]

    if cmd == 'profile':
        pid = int(extra[0]) if extra else 0
        fn(pid)
    elif cmd == 'test':
        name = extra[0] if extra else ''
        if not name:
            print("Usage: python debug.py test <test_name>")
        else:
            fn(name)
    else:
        fn()
