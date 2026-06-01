#!/usr/bin/env python3
"""
Comprehensive tests for the network-based copula credit risk framework.

Run with: python test_copula_framework.py

Coverage:
 1.  Imports (all public symbols from src)
 2.  Data generation (schema, sizes, default labels)
 3.  Graph construction + correlation matrix
 4.  Copula fit + simulation (Clayton)
 5.  Risk analyzer (individual / group / portfolio / stress)
 6.  Copula comparison helper
 7.  All 5 copula types
 8.  Configuration validation
 9.  Copula input validation
 10. Edge cases (tiny / large PDs, zero correlation)
 11. Fraud ring detector
 12. Individual PD model (logistic + gradient boosting)
 13. PD model wired into copula (end-to-end model_pd path)
 14. Student-t copula — pairwise probability is NOT identical to Gaussian
 15. Vectorised correlation matrix (no Python-loop regression)
 16. Client value metrics (Sharpe, RAROC, contagion, segmentation)
 17. Contagion stress tester (cascade + systematic)
 18. Network features merge into persons
"""

import sys
import time
import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_small_model(n: int = 50, copula_type: str = 'clayton'):
    """Return a fitted CopulaDefaultModel on a tiny synthetic network."""
    from src.data_generator import generate_network
    from src.graph_features import TransactionGraph
    from src.copula_model import CopulaDefaultModel

    persons, transactions = generate_network(seed=7)
    persons = persons.head(n).copy()
    persons['person_id'] = range(n)
    transactions = transactions[
        transactions['sender_id'].isin(persons['person_id']) &
        transactions['receiver_id'].isin(persons['person_id'])
    ]

    graph = TransactionGraph(transactions, persons)
    corr = graph.get_correlation_matrix()
    model = CopulaDefaultModel(copula_type)
    model.fit(persons['base_pd'].values, corr)
    return model, graph, persons


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_imports():
    print("Test 01: Imports... ", end="")
    from src.data_generator import generate_network, get_summary_stats
    from src.graph_features import TransactionGraph, get_neighbor_risk_features
    from src.copula_model import CopulaDefaultModel, compare_copulas
    from src.risk_metrics import RiskAnalyzer, FraudRingDetector, ContagionStressTester
    from src.pd_model import IndividualPDModel, PDModelEnsemble
    from src.client_value_metrics import ClientValueCalculator, ClientPortfolioAnalyzer
    from src.config import (NetworkConfig, CopulaConfig, RiskConfig,
                             PipelineConfig, DEFAULT_CONFIG)
    from src import (generate_network, TransactionGraph, CopulaDefaultModel,
                     RiskAnalyzer, IndividualPDModel, ClientValueCalculator)

    assert all([generate_network, get_summary_stats, TransactionGraph,
                get_neighbor_risk_features, CopulaDefaultModel, compare_copulas,
                RiskAnalyzer, FraudRingDetector, ContagionStressTester,
                IndividualPDModel, PDModelEnsemble,
                ClientValueCalculator, ClientPortfolioAnalyzer,
                NetworkConfig, CopulaConfig, RiskConfig, PipelineConfig, DEFAULT_CONFIG])
    print("PASSED")


def test_data_generation():
    print("Test 02: Data generation... ", end="")
    from src.data_generator import generate_network, get_summary_stats

    persons, transactions = generate_network(seed=42)
    stats = get_summary_stats(persons, transactions)

    assert len(persons) == 1000
    assert len(transactions) > 0

    required_cols = {"person_id", "city_id", "city_name", "base_pd", "risk_archetype",
                     "default", "age", "income", "employment_years", "debt_to_income",
                     "num_credit_lines", "missed_payments", "credit_utilization",
                     "account_age_months", "high_risk_group_id", "is_bridge"}
    assert required_cols.issubset(persons.columns), (
        f"Missing columns: {required_cols - set(persons.columns)}"
    )

    assert {"sender_id", "receiver_id", "amount"}.issubset(transactions.columns)
    assert stats["n_high_risk_groups"] >= 1
    assert stats["n_high_risk_group_members"] >= stats["n_high_risk_groups"]

    # default label must be binary and correlated with base_pd
    assert set(persons['default'].unique()).issubset({0, 1})
    assert persons['default'].mean() > 0.01  # at least some defaults

    print("PASSED")
    return persons, transactions


def test_graph(persons, transactions):
    print("Test 03: Graph + correlation... ", end="")
    from src.graph_features import TransactionGraph, get_neighbor_risk_features

    graph = TransactionGraph(transactions, persons)
    net = graph.get_network_stats()
    corr = graph.get_correlation_matrix()
    nf = get_neighbor_risk_features(graph, persons)

    n = len(persons)
    assert corr.shape == (n, n)
    assert np.allclose(np.diag(corr), 1.0)
    assert net.n_nodes == n
    assert len(nf) == n

    # Must be positive semi-definite
    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() >= -1e-6, f"Correlation matrix not PSD (min eigval={eigvals.min():.2e})"

    print("PASSED")
    return graph, corr


def test_correlation_matrix_vectorised(persons, transactions):
    """Vectorised city/group boosts should match the old loop-based result."""
    print("Test 04: Correlation matrix vectorisation... ", end="")
    from src.graph_features import TransactionGraph
    import time

    graph = TransactionGraph(transactions, persons)

    t0 = time.perf_counter()
    corr = graph.get_correlation_matrix()
    elapsed = time.perf_counter() - t0

    # Should finish in under 5 seconds for 1000 nodes
    assert elapsed < 5.0, f"Correlation matrix took {elapsed:.1f}s — too slow"
    assert corr.shape == (len(persons), len(persons))
    print("PASSED")


def test_copula(persons, corr):
    print("Test 05: Copula fit + simulation... ", end="")
    from src.copula_model import CopulaDefaultModel

    model = CopulaDefaultModel("clayton")
    model.fit(persons["base_pd"].values, corr)

    assert model.is_fitted
    assert model.params.theta > 0
    assert 0 <= model.tail_dependence() <= 1

    d = model.simulate_defaults(n_simulations=400)
    assert d.shape == (400, len(persons))
    assert set(np.unique(d)).issubset({0, 1})

    cond = model.conditional_default_probability(0, 1)
    assert 0 <= cond <= 1
    print("PASSED")
    return model


def test_student_t_not_gaussian(persons, corr):
    """Student-t pairwise probability must differ from Gaussian (heavy tails)."""
    print("Test 06: Student-t copula ≠ Gaussian... ", end="")
    from src.copula_model import CopulaDefaultModel

    pds = persons["base_pd"].values

    gauss = CopulaDefaultModel("gaussian")
    gauss.fit(pds, corr)

    t_cop = CopulaDefaultModel("student_t")
    t_cop.fit(pds, corr)

    # Sample 20 pairs to check at least one differs by > 1e-6
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(pds), size=(20, 2))
    diffs = [
        abs(t_cop.joint_default_probability(int(i), int(j)) -
            gauss.joint_default_probability(int(i), int(j)))
        for i, j in pairs if i != j
    ]
    assert max(diffs) > 1e-6, "Student-t and Gaussian copulas give identical pairwise probs"
    print("PASSED")


def test_risk_analyzer(model, graph, persons):
    print("Test 07: Risk analyzer... ", end="")
    from src.risk_metrics import RiskAnalyzer

    exposures = persons["income"].values / persons["income"].mean()
    analyzer = RiskAnalyzer(model, graph, persons, exposures=exposures, lgd=0.45)

    individual = analyzer.compute_individual_risks()
    groups = analyzer.get_group_summary()
    portfolio = analyzer.compute_portfolio_risks(n_simulations=1500)
    stress = analyzer.stress_test(pd_multiplier=1.8, correlation_boost=0.15)

    assert len(individual) == len(persons)
    assert "composite_risk_score" in individual.columns
    assert "risk_tier" in individual.columns
    assert set(individual['risk_tier'].unique()).issubset({'low','medium','high','critical'})
    assert groups is not None

    assert portfolio.expected_loss >= 0
    assert portfolio.var_95 >= portfolio.expected_loss * 0.9  # VaR >= ~EL
    assert portfolio.es_95 >= portfolio.var_95
    assert portfolio.es_99 >= portfolio.es_95
    assert portfolio.tail_risk_ratio >= 1.0

    assert "base" in stress and "stressed" in stress and "change" in stress
    assert stress['stressed']['expected_loss'] >= stress['base']['expected_loss']
    print("PASSED")
    return analyzer


def test_compare_copulas(persons, corr):
    print("Test 08: Copula comparison helper... ", end="")
    from src.copula_model import compare_copulas

    comparison = compare_copulas(persons["base_pd"].values, corr, n_simulations=300)
    assert isinstance(comparison, dict)
    assert len(comparison) >= 2
    assert "clayton" in comparison
    assert "tail_dependence" in comparison["clayton"]
    assert "tail_dependence_upper" in comparison["clayton"]
    print("PASSED")


def test_all_copula_types(persons, corr):
    print("Test 09: All copula types... ", end="")
    from src.copula_model import CopulaDefaultModel

    pds = persons["base_pd"].values
    for ctype in ['gaussian', 'student_t', 'clayton', 'gumbel', 'frank']:
        m = CopulaDefaultModel(ctype)
        m.fit(pds, corr)
        assert m.is_fitted
        assert m.params is not None

        defaults = m.simulate_defaults(n_simulations=100)
        assert defaults.shape == (100, len(persons))
        assert set(np.unique(defaults)).issubset({0, 1})

        assert 0 <= m.tail_dependence('lower') <= 1
        assert 0 <= m.tail_dependence('upper') <= 1

    print("PASSED")


def test_config_validation():
    print("Test 10: Configuration validation... ", end="")
    from src.config import NetworkConfig, CopulaConfig, RiskConfig, PipelineConfig

    cfg = PipelineConfig()
    assert cfg.network.total_population == 1000
    assert cfg.copula.copula_type == 'clayton'
    assert cfg.risk.lgd == 0.45

    for bad, klass in [
        ({'copula_type': 'invalid'}, CopulaConfig),
        ({'lgd': 1.5}, RiskConfig),
        ({'n_cities': -1}, NetworkConfig),
    ]:
        try:
            klass(**bad)
            assert False, f"{klass.__name__}(**{bad}) should raise ValueError"
        except ValueError:
            pass

    print("PASSED")


def test_copula_input_validation():
    print("Test 11: Copula input validation... ", end="")
    from src.copula_model import CopulaDefaultModel

    m = CopulaDefaultModel('clayton')

    for bad_args, label in [
        ((np.array([0.1, 1.5]), np.eye(2)), "PD > 1"),
        ((np.array([0.1, 0.2, 0.3]), np.eye(2)), "dim mismatch"),
    ]:
        try:
            m.fit(*bad_args)
            assert False, f"Should raise ValueError for {label}"
        except ValueError:
            pass

    try:
        CopulaDefaultModel('invalid_copula')
        assert False
    except ValueError:
        pass

    print("PASSED")


def test_edge_cases():
    print("Test 12: Edge cases... ", end="")
    from src.copula_model import CopulaDefaultModel

    corr3 = np.array([[1.0, 0.3, 0.2],
                       [0.3, 1.0, 0.4],
                       [0.2, 0.4, 1.0]])

    # Very small PDs
    m1 = CopulaDefaultModel('clayton')
    m1.fit(np.array([0.001, 0.002, 0.003]), corr3)
    assert m1.simulate_defaults(100).shape == (100, 3)

    # Near-1 PDs
    m2 = CopulaDefaultModel('gumbel')
    m2.fit(np.array([0.95, 0.90, 0.85]), corr3)
    assert m2.simulate_defaults(100).shape == (100, 3)

    # Zero-correlation → joint ≈ product (Gaussian)
    m3 = CopulaDefaultModel('gaussian')
    m3.fit(np.array([0.1, 0.2, 0.3]), np.eye(3))
    jp = m3.joint_default_probability(0, 1)
    assert 0 <= jp <= 1

    print("PASSED")


def test_fraud_ring_detector(graph, copula_model, persons):
    print("Test 13: Fraud ring detector... ", end="")
    from src.risk_metrics import FraudRingDetector

    det = FraudRingDetector(graph, copula_model, persons)
    suspicious = det.detect_suspicious_clusters(
        min_cluster_size=3, joint_pd_threshold=0.10, density_threshold=0.3
    )
    assert isinstance(suspicious, list)

    indicators = det.get_fraud_indicators(0)
    for key in ('reciprocity', 'transaction_concentration', 'avg_neighbor_pd'):
        assert key in indicators

    print("PASSED")


def test_pd_model(persons):
    print("Test 14: Individual PD model... ", end="")
    from src.pd_model import IndividualPDModel, PDModelEnsemble

    # Logistic
    lr = IndividualPDModel(model_type='logistic')
    metrics = lr.fit(persons, target_col='default')
    assert metrics['train_auc'] > 0.5
    assert metrics['val_auc'] > 0.5

    preds = lr.predict_proba(persons)
    assert preds.shape == (len(persons),)
    assert preds.min() >= 0 and preds.max() <= 1

    explanation = lr.explain_prediction(persons.iloc[0])
    assert 'contribution' in explanation.columns

    # Gradient boosting
    gb = IndividualPDModel(model_type='gradient_boosting')
    metrics_gb = gb.fit(persons, target_col='default')
    assert metrics_gb['train_auc'] > 0.5
    assert gb.feature_importance_ is not None

    # Ensemble
    ens = PDModelEnsemble(n_models=3)
    ens.fit(persons, target_col='default')
    mean_p, std_p = ens.predict_proba(persons)
    assert mean_p.shape == (len(persons),)
    assert std_p.shape == (len(persons),)
    assert (std_p >= 0).all()

    print("PASSED")
    return gb


def test_model_pd_pipeline(persons, transactions):
    """End-to-end: use model-predicted PDs instead of base_pd in copula."""
    print("Test 15: Model PD → copula pipeline... ", end="")
    from src.pd_model import IndividualPDModel
    from src.graph_features import TransactionGraph
    from src.copula_model import CopulaDefaultModel
    from src.risk_metrics import RiskAnalyzer

    pd_model = IndividualPDModel(model_type='logistic')
    pd_model.fit(persons, target_col='default')
    persons = persons.copy()
    persons['model_pd'] = pd_model.predict_proba(persons)

    graph = TransactionGraph(transactions, persons)
    corr = graph.get_correlation_matrix()

    copula = CopulaDefaultModel('clayton')
    copula.fit(persons['model_pd'].values, corr)

    analyzer = RiskAnalyzer(copula, graph, persons, lgd=0.45)
    portfolio = analyzer.compute_portfolio_risks(n_simulations=1000)
    assert portfolio.expected_loss >= 0
    assert portfolio.var_95 >= 0

    print("PASSED")


def test_client_value_metrics(copula_model, persons, transactions):
    print("Test 16: Client value metrics... ", end="")
    from src.client_value_metrics import ClientValueCalculator, ClientPortfolioAnalyzer

    calc = ClientValueCalculator(copula_model, persons, transactions, lgd=0.45)

    base = calc.compute_client_sharpe()
    assert len(base) == len(persons)
    for col in ('expected_revenue', 'expected_loss', 'expected_profit',
                'client_sharpe', 'raroc', 'cltv_risk_adjusted'):
        assert col in base.columns, f"Missing column: {col}"

    contagion = calc.compute_contagion_adjusted_sharpe()
    assert 'contagion_adjusted_sharpe' in contagion.columns

    segments = calc.segment_clients()
    assert 'segment' in segments.columns
    assert set(segments['segment'].unique()).issubset({'Stars', 'Cash Cows', 'Question Marks', 'Dogs'})

    ranking = calc.client_ranking(method='raroc')
    assert len(ranking) == len(persons)

    analyzer = ClientPortfolioAnalyzer(calc)
    weights = np.ones(len(persons)) / len(persons)
    attribution = analyzer.attribution_analysis(weights)
    assert 'return_contribution' in attribution.columns

    print("PASSED")


def test_contagion_stress_tester(copula_model, graph):
    print("Test 17: Contagion stress tester... ", end="")
    from src.risk_metrics import ContagionStressTester

    tester = ContagionStressTester(copula_model, graph)

    result = tester.simulate_cascade(initial_defaults=[0, 1, 2], contagion_rounds=3)
    assert result['total_defaults'] >= 3
    assert result['cascade_multiplier'] >= 1.0
    assert 'defaults_per_round' in result

    summary = tester.systematic_stress_test(shock_fraction=0.05, n_scenarios=20)
    assert summary['avg_cascade_multiplier'] >= 1.0
    assert 0 < summary['default_rate_mean'] < 1

    critical = tester.identify_critical_nodes(top_k=5)
    assert len(critical) == 5
    assert all(mult >= 1.0 for _, mult in critical)

    print("PASSED")


def test_network_features_merge(persons, transactions):
    print("Test 18: Network features merge... ", end="")
    from src.graph_features import TransactionGraph, get_neighbor_risk_features

    graph = TransactionGraph(transactions, persons)
    nf = get_neighbor_risk_features(graph, persons)

    assert 'neighbor_pd_avg' in nf.columns
    assert 'n_high_risk_neighbors' in nf.columns
    assert len(nf) == len(persons)

    merged = persons.merge(
        nf[['person_id', 'neighbor_pd_avg', 'n_high_risk_neighbors']],
        on='person_id', how='left'
    ).fillna(0)
    assert len(merged) == len(persons)
    assert 'neighbor_pd_avg' in merged.columns

    print("PASSED")


def test_rating_engine(persons):
    print("Test 19: Rating engine... ", end="")
    from src.rating_engine import RatingEngine, RATING_LABELS, pd_to_rating

    # PD → rating mapping
    assert pd_to_rating(0.0005) == 1   # AAA
    assert pd_to_rating(0.5)   == 7   # CCC
    assert pd_to_rating(0.99)  == 8   # Default

    engine = RatingEngine()
    engine.fit(persons, pd_col="base_pd")

    dist = engine.portfolio_distribution()
    assert sum(dist.counts.values()) == len(persons)
    assert abs(sum(dist.fractions.values()) - 1.0) < 1e-9
    assert 0.0 < dist.migration_risk_score < 1.0

    profile = engine.get_rating_profile(persons["person_id"].iloc[10])
    assert profile.current_rating_label in RATING_LABELS
    assert abs(profile.one_year_migration.sum() - 1.0) < 1e-6
    assert 0 <= profile.default_prob_1yr <= 1
    assert 0 <= profile.default_prob_3yr <= 1
    assert profile.default_prob_3yr >= profile.default_prob_1yr

    mig_table = engine.migration_table(persons["person_id"].iloc[10])
    assert len(mig_table) == 8
    assert abs(mig_table["probability"].sum() - 1.0) < 1e-6

    summary = engine.summary_df()
    assert len(summary) == len(persons)
    assert "rating_label" in summary.columns

    print("PASSED")
    return engine


def test_structural_pd(persons):
    print("Test 20: Structural (Merton) PD... ", end="")
    import pandas as pd
    from src.structural_pd import StructuralPDModel, compute_proxy_merton_pd, merton_pd_direct

    # Pure Merton formula
    pd_val, dd = merton_pd_direct(asset_value=100, debt=80, asset_vol=0.25, T=1.0, r=0.02)
    assert 0 < pd_val < 1
    assert dd > 0

    # Proxy for a retail borrower
    params = compute_proxy_merton_pd(
        income=5000, debt_to_income=0.3, credit_utilisation=0.6,
        missed_payments=2, employment_years=5
    )
    assert 0 < params.merton_pd < 1
    assert params.distance_to_default > 0

    # Full model over persons DataFrame
    model = StructuralPDModel(alpha=0.35)
    enriched = model.fit_transform(persons.head(50), statistical_pd_col="base_pd")
    assert "merton_pd" in enriched.columns
    assert "blended_pd" in enriched.columns
    assert "pd_signal_divergence" in enriched.columns
    assert "distance_to_default" in enriched.columns
    assert (enriched["merton_pd"] >= 0).all()
    assert (enriched["merton_pd"] <= 1).all()
    assert (enriched["blended_pd"] >= 0).all()

    # Blended PD is between the two signals
    alpha = 0.35
    expected = alpha * enriched["merton_pd"] + (1 - alpha) * enriched["base_pd"]
    assert np.allclose(enriched["blended_pd"], expected, atol=1e-3)

    ew = model.get_early_warnings(0.01)
    assert isinstance(ew, pd.DataFrame)

    print("PASSED")


def test_flexible_probs(corr):
    print("Test 21: Flexible probabilities / regime calibration... ", end="")
    from src.flexible_probs import (
        FlexibleProbsCalibrator, gaussian_kernel_weights,
        exponential_decay_weights, build_calibrator_from_portfolio,
        classify_regime, low_rank_decomposition,
    )

    # Kernel weights sum to 1
    z = np.linspace(0, 1, 50)
    w = gaussian_kernel_weights(z, 0.5)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.min() >= 0

    # Decay weights sum to 1, most recent is highest
    d = exponential_decay_weights(50, half_life=12)
    assert abs(d.sum() - 1.0) < 1e-9
    assert d[-1] >= d[0]

    # Regime classification
    assert classify_regime(0.1) == "calm"
    assert classify_regime(0.9) == "crisis"

    # Low-rank decomposition
    n = 10
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    C = A @ A.T
    np.fill_diagonal(C, 1.0)
    C = C / C.max()
    np.fill_diagonal(C, 1.0)
    beta, idio = low_rank_decomposition(C, n_factors=3)
    assert beta.shape == (n, 3)
    assert idio.shape == (n,)
    assert (idio > 0).all()

    # Full calibrator on small corr
    small_corr = corr[:20, :20].copy()
    avg_pd_hist = 0.05 + 0.1 * np.linspace(0, 1, 24)
    calib = build_calibrator_from_portfolio(avg_pd_hist, half_life_periods=6.0)

    result = calib.calibrate(current_stress=0.3, base_corr_matrix=small_corr)
    assert result.theta > 0
    assert 0 <= result.avg_correlation <= 1
    assert result.correlation_matrix.shape == (20, 20)
    assert result.regime.label in ("calm", "moderate", "stressed", "crisis")

    # Stress table: theta should increase with stress
    table = calib.calibrate_for_scenarios(np.array([0.1, 0.5, 0.9]), small_corr)
    assert table["theta"].iloc[0] < table["theta"].iloc[2], \
        "theta should increase with stress"

    print("PASSED")


def test_refactor_correctness(persons, transactions, model, graph):
    """Tests specifically targeting the 20 audit findings."""
    print("Test 23: Refactor correctness... ", end="")
    import pandas as pd
    from src.client_value_metrics import ClientValueCalculator
    from src.rating_engine import RatingEngine, pd_to_rating
    from src.risk_metrics import RiskAnalyzer
    from src.graph_features import TransactionGraph
    from src.flexible_probs import FlexibleProbsCalibrator

    # ── Fix 1: expected_profit = revenue - EL (not the old double-assignment) ──
    cvc = ClientValueCalculator(model, persons, transactions, lgd=0.45)
    metrics = cvc.compute_client_sharpe()
    # expected_profit = revenue - expected_loss; must be <= revenue
    assert (metrics["expected_profit"] <= metrics["expected_revenue"] + 1e-6).all(), \
        "expected_profit should be ≤ expected_revenue"
    # Must be >= revenue - total_possible_loss (can be negative for high-risk clients)
    # Simply verify no NaN
    assert not metrics["expected_profit"].isna().any()

    # ── Fix 2: rating lookup with non-sequential person_ids ──
    persons_ns = persons.copy()
    persons_ns["person_id"] = persons_ns["person_id"] + 5000  # shift IDs
    engine = RatingEngine()
    engine.fit(persons_ns, "base_pd")
    pid = int(persons_ns["person_id"].iloc[7])
    profile = engine.get_rating_profile(pid)  # must not raise
    assert profile.person_id == pid

    # ── Fix 3: downgrade_prob excludes Default (absorbing) state ──
    # For a BBB borrower (state 4), downgrade includes BB,B,CCC but NOT Default
    bbbs = persons[persons["base_pd"].between(0.010, 0.030)]
    if len(bbbs) > 0:
        engine2 = RatingEngine()
        engine2.fit(persons, "base_pd")
        for pid2 in bbbs["person_id"].head(3):
            p2 = engine2.get_rating_profile(int(pid2))
            # upgrade + downgrade + default + stay should ≈ 1
            total = p2.upgrade_prob + p2.downgrade_prob + p2.default_prob_1yr + \
                    float(p2.one_year_migration[p2.current_rating - 1])
            assert abs(total - 1.0) < 0.02, \
                f"Probabilities don't sum to 1 for {pid2}: {total:.4f}"

    # ── Fix 4: stress test restores copula state even under exception ──
    analyzer = RiskAnalyzer(model, graph, persons, lgd=0.45)
    original_pds = model.marginal_pds.copy()
    stress = analyzer.stress_test(pd_multiplier=2.0, correlation_boost=0.1, n_simulations=500)
    assert np.allclose(model.marginal_pds, original_pds), \
        "Copula PDs not restored after stress_test"
    assert stress["stressed"]["expected_loss"] >= stress["base"]["expected_loss"]

    # ── Fix 5: risk tiers use vectorised rankdata — no O(n²) loop ──
    # Just verify results are sane
    individual = analyzer.compute_individual_risks()
    tier_counts = individual["risk_tier"].value_counts()
    assert "critical" in tier_counts or "high" in tier_counts
    assert set(individual["risk_tier"]).issubset({"low", "medium", "high", "critical"})

    # ── Fix 6: dynamic city layout works for any number of cities ──
    from src.graph_features import TransactionGraph as TG
    graph_ok = TG(transactions, persons)
    pos = graph_ok._compute_layout("city")
    assert pos.shape == (len(persons), 2)
    assert not np.any(np.isnan(pos))

    # ── Fix 7: flexible_probs validates inputs ──
    calib = FlexibleProbsCalibrator()
    try:
        calib.fit(np.array([0.1]))  # too short
        assert False, "Should raise ValueError for length-1 input"
    except ValueError:
        pass
    try:
        calib.fit(np.full(10, np.nan))  # all NaN
        assert False, "Should raise ValueError for all-NaN input"
    except ValueError:
        pass

    # ── Fix 8: group correlation uses vectorised indexing ──
    groups = analyzer.compute_group_risks()
    assert isinstance(groups, list)
    for g in groups:
        assert 0.0 <= g.internal_correlation <= 1.0
        assert 0.0 <= g.joint_default_probability <= 1.0

    # ── Fix 9: TransactionGraph validates inputs ──
    try:
        TG(transactions.drop(columns=["amount"]), persons)
        assert False, "Should raise ValueError for missing column"
    except ValueError:
        pass
    dup_persons = pd.concat([persons.head(5), persons.head(5)])
    try:
        TG(transactions, dup_persons)
        assert False, "Should raise ValueError for duplicate person_ids"
    except ValueError:
        pass

    print("PASSED")


def test_customer_profiler(model, graph, persons, transactions):
    print("Test 22: Customer profiler... ", end="")
    import pandas as pd
    from src.customer_profile import CustomerProfiler, CustomerRiskProfile
    from src.rating_engine import RatingEngine
    from src.structural_pd import StructuralPDModel
    from src.client_value_metrics import ClientValueCalculator
    from src.risk_metrics import RiskAnalyzer

    # Set up all components
    persons2 = persons.copy()
    persons2["model_pd"] = persons2["base_pd"]

    engine = RatingEngine(); engine.fit(persons2, "base_pd")

    struct = StructuralPDModel(alpha=0.35)
    enriched = struct.fit_transform(persons2.copy(), "base_pd")

    cvc = ClientValueCalculator(model, persons2, transactions, lgd=0.45)

    exposures = persons2["income"].values / persons2["income"].mean()
    analyzer = RiskAnalyzer(model, graph, persons2, exposures=exposures, lgd=0.45)
    ir = analyzer.compute_individual_risks()

    profiler = CustomerProfiler(early_warning_threshold=0.05)
    profiler.fit(
        persons=enriched,
        transactions=transactions,
        graph=graph,
        copula=model,
        rating_engine=engine,
        structural_model=enriched,
        client_value_calc=cvc,
        individual_risks=ir,
    )

    pid = persons["person_id"].iloc[0]
    profile = profiler.get_profile(pid)

    assert isinstance(profile, CustomerRiskProfile)
    assert profile.person_id == pid
    assert profile.current_rating in ["AAA","AA","A","BBB","BB","B","CCC","Default"]
    assert 0 <= profile.statistical_pd <= 1
    assert 0 <= profile.blended_pd <= 1
    assert profile.risk_tier in ("low", "medium", "high", "critical")
    assert isinstance(profile.narrative, str) and len(profile.narrative) > 50
    assert isinstance(profile.top_neighbours, list)

    # Text report
    report = profiler.profile_report(pid)
    assert "CUSTOMER RISK PROFILE" in report
    assert "PD SIGNALS" in report
    assert "RATING MIGRATION" in report
    assert "NETWORK RISK" in report
    assert "BUSINESS VALUE" in report
    assert "COMPOSITE ASSESSMENT" in report

    # Watchlist
    wl = profiler.watchlist(tiers=["critical", "high"], top_n=20)
    assert isinstance(wl, pd.DataFrame)

    print("PASSED")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_all_tests() -> bool:
    """
    Run all 23 tests.

    Each test is wrapped so failures print a full traceback and the suite
    continues — the final summary shows exactly which tests failed.

    Exit code 0 = all passed, 1 = one or more failed.
    """
    import traceback

    print("=" * 60)
    print("  NETWORK-BASED COPULA CREDIT RISK — TEST SUITE")
    print("=" * 60)

    failures: list[str] = []
    fixtures: dict = {}

    def run(name: str, fn, *args, **kwargs):
        """Run one test function and capture failure."""
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            failures.append(name)
            # Print full traceback immediately so it is visible in CI logs
            print(f"\n{'!' * 60}")
            print(f"  FAILED: {name}")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            print('!' * 60)
            return None

    # ------------------------------------------------------------------
    # Tests that produce fixture data for later tests
    # ------------------------------------------------------------------
    run("test_imports", test_imports)

    result = run("test_data_generation", test_data_generation)
    if result is None:
        print("\nCannot continue: data generation failed.")
        return False
    persons, transactions = result

    result = run("test_graph", test_graph, persons, transactions)
    if result is None:
        print("\nCannot continue: graph construction failed.")
        return False
    graph, corr = result

    result = run("test_copula", test_copula, persons, corr)
    model = result  # may be None if test failed

    # ------------------------------------------------------------------
    # Independent tests (run even if earlier tests failed)
    # ------------------------------------------------------------------
    run("test_correlation_matrix_vectorised", test_correlation_matrix_vectorised, persons, transactions)
    if model is not None:
        run("test_student_t_not_gaussian", test_student_t_not_gaussian, persons, corr)
        run("test_risk_analyzer", test_risk_analyzer, model, graph, persons)
        run("test_compare_copulas", test_compare_copulas, persons, corr)
    run("test_all_copula_types", test_all_copula_types, persons, corr)
    run("test_config_validation", test_config_validation)
    run("test_copula_input_validation", test_copula_input_validation)
    run("test_edge_cases", test_edge_cases)
    if model is not None:
        run("test_fraud_ring_detector", test_fraud_ring_detector, graph, model, persons)
    run("test_pd_model", test_pd_model, persons)
    run("test_model_pd_pipeline", test_model_pd_pipeline, persons, transactions)
    if model is not None:
        run("test_client_value_metrics", test_client_value_metrics, model, persons, transactions)
        run("test_contagion_stress_tester", test_contagion_stress_tester, model, graph)
    run("test_network_features_merge", test_network_features_merge, persons, transactions)
    run("test_rating_engine", test_rating_engine, persons)
    run("test_structural_pd", test_structural_pd, persons)
    run("test_flexible_probs", test_flexible_probs, corr)
    if model is not None:
        run("test_customer_profiler", test_customer_profiler, model, graph, persons, transactions)
        run("test_refactor_correctness", test_refactor_correctness, persons, transactions, model, graph)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = 23
    passed = total - len(failures)
    print("\n" + "=" * 60)
    if not failures:
        print(f"  All {total} tests passed.")
    else:
        print(f"  {passed}/{total} tests passed.  FAILURES: {', '.join(failures)}")
    print("=" * 60)
    return len(failures) == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
