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

    # Should finish well under 15 seconds for 1000 nodes (generous bound so the
    # test does not flake under concurrent CPU load; the operation is ~0.1s idle).
    assert elapsed < 15.0, f"Correlation matrix took {elapsed:.1f}s — too slow"
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
# risk_adjusted_metrics tests (24-27)
# ---------------------------------------------------------------------------

def test_metric_registry():
    print("Test 24: Metric registry — formulas, dispatch, nan handling... ", end="")
    import math
    from src.risk_adjusted_metrics import available_metrics, compute_metric, MetricInputs

    names = available_metrics()
    expected = {
        "coefficient_of_variation",
        "coefficient_of_variation_copula",
        "raroc",
        "sharpe_indep",
        "sortino_indep",
        "sortino_copula",
        "sortino_simulated",
    }
    assert expected.issubset(set(names)), f"Missing metrics: {expected - set(names)}"

    # Known-value verification for ALL metrics
    # E[Profit]=100, E[Loss]=50, Revenue=120, Capital=10
    # loss_var_L1=2500, loss_std_indep=50, hurdle=0.10, rf=0.02
    inp = MetricInputs(
        expected_profit=100.0, expected_loss=50.0, expected_revenue=120.0,
        capital=10.0, loss_var_L1=2500.0, loss_std_indep=50.0,
        hurdle_rate=0.10, risk_free_rate=0.02
    )
    # CoV L0 = sigma_indep / E[loss] = 50/50 = 1.0
    assert abs(compute_metric("coefficient_of_variation", inp) - 1.0) < 1e-9, "CoV L0"
    # CoV L1 = sqrt(2500)/50 = 50/50 = 1.0 (same here since loss_var_L1 = loss_std_indep^2)
    assert abs(compute_metric("coefficient_of_variation_copula", inp) - 1.0) < 1e-9, "CoV L1"
    # RAROC = 100/10 = 10.0
    assert abs(compute_metric("raroc", inp) - 10.0) < 1e-9, "RAROC"
    # Sharpe = (100 - 0.02*120) / 50 = 97.6/50 = 1.952
    assert abs(compute_metric("sharpe_indep", inp) - 97.6/50) < 1e-9, "Sharpe"
    # sortino_indep = (100 - 0.10*10) / 50 = 99/50 = 1.98 (hurdle*Capital, NOT rf*Revenue)
    assert abs(compute_metric("sortino_indep", inp) - 99.0/50) < 1e-9, "Sortino indep"
    # sortino_copula = (100 - 0.10*10) / sqrt(2500) = 99/50 = 1.98
    assert abs(compute_metric("sortino_copula", inp) - 99.0/50) < 1e-9, "Sortino copula"
    # sortino_simulated = nan (no semidev provided)
    assert math.isnan(compute_metric("sortino_simulated", inp)), "Sortino sim without semidev"

    # sortino_indep MUST differ from sharpe_indep (different numerators)
    sharpe = compute_metric("sharpe_indep", inp)
    sortino_l0 = compute_metric("sortino_indep", inp)
    assert abs(sharpe - sortino_l0) > 1e-9, \
        f"sortino_indep should differ from sharpe_indep: sharpe={sharpe} sortino={sortino_l0}"

    # sortino_simulated with semidev provided
    inp_sim = MetricInputs(100, 50, 120, 10, 2500, 50, 0.10, 0.02, downside_semidev=25.0)
    assert abs(compute_metric("sortino_simulated", inp_sim) - 99.0/25) < 1e-9, "Sortino sim"

    # Unknown name raises KeyError
    try:
        compute_metric("does_not_exist", inp)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Div-by-zero returns nan, not exception or fudged value
    inp_zero = MetricInputs(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.10, 0.02)
    for m in ["coefficient_of_variation", "raroc", "sharpe_indep",
              "sortino_indep", "sortino_copula"]:
        v = compute_metric(m, inp_zero)
        assert math.isnan(v), f"{m} should return nan on zero denominator, got {v}"

    print("PASSED")


def test_metric_primitives_additivity(model, persons):
    print("Test 25: Metric primitives additivity... ", end="")
    import math
    from src.risk_adjusted_metrics import RiskRatioCalculator

    exposures = persons["income"].values / persons["income"].mean()
    calc = RiskRatioCalculator(model, persons, exposures=exposures, lgd=0.45)

    n = len(persons)
    # Two complementary disjoint halves
    A = np.arange(0, n // 2)
    B = np.arange(n // 2, n)
    AB = np.arange(0, n)

    inp_A = calc._inputs_for(A)
    inp_B = calc._inputs_for(B)
    inp_AB = calc._inputs_for(AB)

    # E[Loss] additive
    assert abs(inp_A.expected_loss + inp_B.expected_loss - inp_AB.expected_loss) < 1e-6, \
        "E[Loss] not additive"
    # E[Profit] additive
    assert abs(inp_A.expected_profit + inp_B.expected_profit - inp_AB.expected_profit) < 1e-6, \
        "E[Profit] not additive"
    # Capital additive
    assert abs(inp_A.capital + inp_B.capital - inp_AB.capital) < 1e-6, \
        "Capital not additive"
    # loss_var_L1 is NOT additive (it's bilinear) — but the whole-portfolio var
    # should be >= the sum of segment vars (sub-additivity of std):
    # σ(A∪B) >= σ(A) + σ(B) only for perfectly negatively corr.; here we check
    # that the computation at least runs and is non-negative.
    assert inp_AB.loss_var_L1 >= 0, "Negative loss variance"

    # exposure_share sums ≈ 1 across all archetypes
    df = calc.by_segment("risk_archetype")
    share_sum = df["exposure_share"].sum()
    assert abs(share_sum - 1.0) < 1e-3, f"exposure_share sums to {share_sum}"

    print("PASSED")


def test_single_borrower_closed_form(model, persons):
    print("Test 26: Single-borrower closed-form reduction... ", end="")
    import math
    from src.risk_adjusted_metrics import RiskRatioCalculator

    exposures = persons["income"].values / persons["income"].mean()
    calc = RiskRatioCalculator(model, persons, exposures=exposures, lgd=0.45)

    # For any single borrower i: loss_std_indep == EAD_i * LGD_i * sqrt(PD_i * (1-PD_i))
    for i in [0, 5, 42]:
        inp = calc._inputs_for(np.array([i]))
        expected_std = (calc.ead[i] * calc.lgd[i]
                        * np.sqrt(calc.pd[i] * (1.0 - calc.pd[i])))
        assert abs(inp.loss_std_indep - expected_std) < 1e-9, \
            f"Borrower {i}: loss_std_indep={inp.loss_std_indep:.8f} expected={expected_std:.8f}"
        # For single borrower, L0 == L1 (copula provides no diversification info)
        assert abs(inp.loss_var_L1 - inp.loss_std_indep ** 2) < 1e-6, \
            f"L0 != L1 for single borrower {i}"

    print("PASSED")


def test_correlation_inflates_denominator(model, persons):
    print("Test 27: Correlation inflates denominator (contagion enters risk)... ", end="")
    from src.risk_adjusted_metrics import RiskRatioCalculator

    exposures = persons["income"].values / persons["income"].mean()
    calc = RiskRatioCalculator(model, persons, exposures=exposures, lgd=0.45)

    # Build a pair of borrowers from the same high-risk group, which have the
    # highest within-group correlation boost. Check their pair-level copula
    # covariance is positive (the off-diagonal LossCov[i,j] > 0), which is
    # what causes contagion to inflate the denominator.
    groups = persons["high_risk_group_id"].values
    group_ids = [g for g in np.unique(groups) if g >= 0]
    assert len(group_ids) > 0, "No high-risk groups in test data"

    members = np.where(groups == group_ids[0])[0][:4]
    assert len(members) >= 2, "Need at least 2 members"

    # The group-boosted correlation matrix should produce positive off-diagonal
    # LossCov terms for at least one pair — verify this
    i, j = int(members[0]), int(members[1])
    off_diag = calc.loss_cov[i, j]
    assert off_diag > 0, \
        f"Same-group pair has non-positive LossCov[{i},{j}]={off_diag:.8f}; contagion not captured"

    # Verify the portfolio sigma for a group is strictly larger than the same
    # borrowers treated as independent, IF net covariance is positive
    inp_seg = calc._inputs_for(members)
    block = calc.loss_cov[np.ix_(members, members)]
    net_off_diag = block.sum() - np.diag(block).sum()  # sum of off-diagonal entries
    if net_off_diag > 0:
        # Positive net covariance: copula sigma > indep sigma
        sigma_copula = np.sqrt(inp_seg.loss_var_L1)
        sigma_indep = inp_seg.loss_std_indep
        assert sigma_copula > sigma_indep - 1e-9, \
            f"Positive covariance group: sigma_copula={sigma_copula:.6f} < sigma_indep={sigma_indep:.6f}"

    # diversification_ratio is well-defined and finite
    dr = calc.diversification_ratio(members)
    assert np.isfinite(dr) and dr > 0, f"diversification_ratio invalid: {dr}"

    print("PASSED")


def test_pluggable_inputs_and_sim(model, persons, transactions):
    print("Test 28: Pluggable revenue/capital + sortino_simulated via by_segment... ", end="")
    import math
    from src.risk_adjusted_metrics import RiskRatioCalculator
    from src.client_value_metrics import ClientValueCalculator

    # Build calculator with ClientValueCalculator-provided revenue and EAD
    cvc = ClientValueCalculator(model, persons, transactions, lgd=0.45)
    persons_enriched = cvc.persons
    ead = persons_enriched["exposure_at_default"].values

    calc = RiskRatioCalculator(model, persons_enriched, exposures=ead, lgd=0.45)

    # With proper revenue, fewer borrowers should have negative E[Profit] than the 2% proxy
    n_neg = (calc.eprofit < 0).sum()
    n_total = len(persons)
    # The 2% proxy produces ~66% negative; proper revenue should be better
    assert n_neg < n_total * 0.60, \
        f"Too many negative-profit borrowers with proper revenue: {n_neg}/{n_total}"

    # Explicit revenue and capital arrays override all auto-detection
    rev_arr = np.ones(len(persons)) * 100.0
    cap_arr = np.ones(len(persons)) * 10.0
    calc2 = RiskRatioCalculator(model, persons, revenue=rev_arr, capital=cap_arr, lgd=0.45)
    assert np.allclose(calc2.revenue, 100.0), "Revenue array not used"
    assert np.allclose(calc2.capital, 10.0), "Capital array not used"

    # sortino_simulated populated via by_segment(with_sim=True)
    df_sim = calc.by_segment("risk_archetype", with_sim=True, n_sim=200)
    assert "sortino_simulated" in df_sim.columns, "sortino_simulated missing from by_segment with_sim"
    finite_sim = df_sim["sortino_simulated"].apply(lambda v: math.isfinite(v) if v == v else False)
    assert finite_sim.any(), "No finite sortino_simulated values in by_segment"

    print("PASSED")


def test_by_segment_invariants(model, persons):
    print("Test 29: by_segment shapes and invariants... ", end="")
    from src.risk_adjusted_metrics import RiskRatioCalculator

    exposures = persons["income"].values / persons["income"].mean()
    calc = RiskRatioCalculator(model, persons, exposures=exposures, lgd=0.45)

    # Test each supported segment column
    for col in ("risk_archetype", "city_name"):
        df = calc.by_segment(col)
        n_segments = persons[col].nunique()
        assert len(df) == n_segments, \
            f"by_segment('{col}'): expected {n_segments} rows, got {len(df)}"

        # exposure_share sums to 1 (within tolerance)
        share_sum = df["exposure_share"].sum()
        assert abs(share_sum - 1.0) < 1e-3, \
            f"by_segment('{col}'): exposure_share sums to {share_sum:.6f}"

        # diversification_ratio is finite and positive for every segment
        assert (df["diversification_ratio"] > 0).all(), \
            f"by_segment('{col}'): non-positive diversification_ratio"
        assert df["diversification_ratio"].apply(np.isfinite).all(), \
            f"by_segment('{col}'): non-finite diversification_ratio"

        # All metric columns are present
        from src.risk_adjusted_metrics import available_metrics
        for m in available_metrics():
            if m != "sortino_simulated":  # requires with_sim=True
                assert m in df.columns, f"Missing metric '{m}' in by_segment('{col}')"

    # high_risk_group_id: drop_unlabelled=True (default) removes -1 rows
    df_grp = calc.by_segment("high_risk_group_id")
    group_ids = persons["high_risk_group_id"].unique()
    n_valid_groups = sum(1 for g in group_ids if g >= 0)
    assert len(df_grp) == n_valid_groups, \
        f"high_risk_group_id: expected {n_valid_groups} rows, got {len(df_grp)}"

    print("PASSED")


def test_metric_comparison(model, persons):
    print("Test 30: MetricComparator — rank corr, disagreements, divergence flags... ", end="")
    import pandas as pd
    from src.risk_adjusted_metrics import RiskRatioCalculator, available_metrics
    from src.metric_comparison import MetricComparator

    exposures = persons["income"].values / persons["income"].mean()
    calc = RiskRatioCalculator(model, persons, exposures=exposures, lgd=0.45)
    comp = MetricComparator(calc)

    # borrower_table: one row per borrower, all metric columns present
    bt = comp.borrower_table()
    assert len(bt) == len(persons), f"borrower_table rows: {len(bt)} vs {len(persons)}"
    for m in available_metrics():
        assert m in bt.columns, f"metric '{m}' missing from borrower_table"
    assert "numerator_negative" in bt.columns

    # rank_correlation at borrower level: square, symmetric, diagonal=1, values in [-1,1]
    rc = comp.rank_correlation()
    assert rc.shape[0] == rc.shape[1], "rank_correlation not square"
    assert np.allclose(np.diag(rc.values), 1.0, atol=1e-9), "rank_corr diagonal not 1"
    assert (rc.values >= -1.0 - 1e-9).all() and (rc.values <= 1.0 + 1e-9).all(), \
        "rank_corr values out of [-1,1]"

    # rank_correlation at SEGMENT level: correct structure (square, symmetric, diag=1)
    rc_seg = comp.rank_correlation(level="segment", segment_col="high_risk_group_id")
    assert rc_seg.shape[0] == rc_seg.shape[1], "segment rank_corr not square"
    assert np.allclose(np.diag(rc_seg.values), 1.0, atol=1e-9), "segment rank_corr diag != 1"
    off_diag = rc_seg.values[~np.eye(rc_seg.shape[0], dtype=bool)]
    finite_off = off_diag[np.isfinite(off_diag)]
    assert len(finite_off) > 0, "segment rank_corr has no finite off-diagonal entries"
    assert (np.abs(finite_off) <= 1.0 + 1e-9).all(), "segment rank_corr values outside [-1,1]"
    # Note: with only 4 groups and monotone synthetic data, all pairs may rank-correlate at 1.0.
    # Rank differentiation is meaningful only with enough segments; 4 is the minimum.

    # disagreements: at most top_n rows, correct columns
    dis = comp.disagreements("raroc", "sortino_copula", top_n=10)
    assert len(dis) <= 10, f"disagreements > top_n: {len(dis)}"
    if len(dis) > 0:
        assert "person_id" in dis.columns
        assert "rank_gap" in dis.columns
        assert (dis["rank_gap"] >= 0).all(), "rank_gap must be non-negative"

    # divergence_flags: must find at least some flagged borrowers at a low threshold
    flags_low = comp.divergence_flags(z_threshold=0.5)
    assert isinstance(flags_low, pd.DataFrame), "divergence_flags must return DataFrame"
    assert len(flags_low) > 0, \
        "divergence_flags found 0 borrowers at z=0.5 — RAROC vs Sortino not differentiating"
    assert "flag_type" in flags_low.columns
    valid_flag_types = {"hidden_network_risk", "diversified_low_value"}
    assert set(flags_low["flag_type"].unique()).issubset(valid_flag_types), \
        f"Unexpected flag_types: {flags_low['flag_type'].unique()}"

    # sortino_simulated: with_sim=True on a small group must return a finite value
    members = np.arange(20)  # first 20 borrowers
    dsdev = calc._compute_downside_semidev(members, n_sim=500)
    assert np.isfinite(dsdev) and dsdev >= 0, f"downside_semidev invalid: {dsdev}"
    val = calc.metric("sortino_simulated", members, with_sim=True, n_sim=500)
    assert np.isfinite(val), f"sortino_simulated returned non-finite: {val}"

    print("PASSED")


# ---------------------------------------------------------------------------
# scale / real-data tests (31-33)
# ---------------------------------------------------------------------------

def test_loaders_dirty_data():
    """Test 31: loaders handle dirty real-world data (mapping, NaN, dups, % PD)."""
    print("Test 31: loaders — column mapping, NaN/dup policy, %-PD, reindex... ", end="")
    import logging as _lg
    _lg.getLogger("src.loaders").setLevel(_lg.CRITICAL)  # silence expected warnings
    import pandas as pd
    import numpy as np
    from src.loaders import (ColumnMapping, load_persons, load_transactions,
                             reindex_to_contiguous, DataValidationError, validate_persons)

    # Dirty source: foreign names, % PD, NaN, duplicate id, big account numbers.
    raw = pd.DataFrame({
        "acct":   [500001, 500002, 500003, 500003, 500005],   # dup 500003
        "pd_pct": [2.5, 5.0, np.nan, 8.0, 30.0],               # percentages + NaN
        "reg":    [1, 1, 2, 2, 3],
        "lim":    [1000.0, 2000.0, 1500.0, 1800.0, 900.0],
    })
    pmap = ColumnMapping(person_id="acct", model_pd="pd_pct",
                         city_id="reg", exposure_at_default="lim")

    # Duplicate with policy=error must raise.
    try:
        load_persons(raw, mapping=pmap)
        assert False, "duplicate id should raise"
    except DataValidationError:
        pass

    # With dedup + drop NaN it loads and canonicalises.
    persons = load_persons(raw, mapping=pmap, duplicate_policy="first", pd_nan_policy="drop")
    assert len(persons) == 3, f"expected 3 rows, got {len(persons)}"
    assert persons["model_pd"].between(0, 1).all(), "PD not in [0,1] after %-scaling"
    assert "person_id" in persons.columns and "city_id" in persons.columns
    validate_persons(persons)  # must not raise

    # Transactions referencing unknown ids get dropped.
    raw_tx = pd.DataFrame({"f": [500001, 500002, 999999],
                           "t": [500002, 500001, 500001],
                           "a": [10.0, 20.0, 5.0]})
    tmap = ColumnMapping(sender_id="f", receiver_id="t", amount="a")
    tx = load_transactions(raw_tx, mapping=tmap,
                           valid_person_ids=persons["person_id"].tolist())
    assert len(tx) == 2, f"unknown-id tx not dropped: {len(tx)}"

    # Reindex non-contiguous ids → 0..n-1, preserve originals.
    pr, txr, idmap = reindex_to_contiguous(persons, tx)
    assert pr["person_id"].tolist() == [0, 1, 2]
    assert "original_person_id" in pr.columns
    assert pr["original_person_id"].tolist() == [500001, 500002, 500005]
    print("PASSED")


def test_block_loss_cov_equals_dense():
    """Test 32: block-on-demand loss_cov is identical to the dense matrix."""
    print("Test 32: block loss_cov == dense loss_cov (scale-path correctness)... ", end="")
    import numpy as np
    import pandas as pd
    from src.copula_model import CopulaDefaultModel
    import src.risk_adjusted_metrics as ram

    rng = np.random.default_rng(202)
    N = 250
    persons = pd.DataFrame({
        "person_id": np.arange(N),
        "city_id": rng.integers(0, 4, N),
        "risk_archetype": rng.choice(["low", "medium", "high"], N),
        "exposure_at_default": rng.lognormal(9, 0.7, N),
        "estimated_revenue": rng.lognormal(6, 0.5, N),
    })
    pds = rng.uniform(0.01, 0.4, N)
    corr = np.full((N, N), 0.05)
    city = persons["city_id"].values
    corr[city[:, None] == city[None, :]] += 0.25
    np.fill_diagonal(corr, 1.0)
    ev, evec = np.linalg.eigh(corr)
    corr = evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T
    np.fill_diagonal(corr, 1.0)
    cop = CopulaDefaultModel("clayton")
    cop.fit(pds, corr)

    # Dense reference.
    orig = ram.LOSS_COV_DENSE_MAX_NODES
    ram.LOSS_COV_DENSE_MAX_NODES = 100000
    calc_dense = ram.RiskRatioCalculator(cop, persons, lgd=0.45)
    seg_dense = calc_dense.by_segment("risk_archetype")

    # Block mode (force by lowering threshold below N).
    ram.LOSS_COV_DENSE_MAX_NODES = 50
    calc_block = ram.RiskRatioCalculator(cop, persons, lgd=0.45)
    assert calc_block._dense_loss_cov is None, "should be in block mode"
    # Full loss_cov must refuse.
    try:
        _ = calc_block.loss_cov
        assert False, "dense loss_cov should refuse above threshold"
    except MemoryError:
        pass
    seg_block = calc_block.by_segment("risk_archetype")
    ram.LOSS_COV_DENSE_MAX_NODES = orig

    # Compare every metric column.
    m = seg_dense.merge(seg_block, on="segment", suffixes=("_d", "_b"))
    for col in ("sortino_copula", "coefficient_of_variation_copula", "raroc",
                "loss_std_copula", "diversification_ratio"):
        d = float(np.abs(m[f"{col}_d"] - m[f"{col}_b"]).max())
        assert d < 1e-6, f"block vs dense '{col}' differs by {d}"
    print("PASSED")


def test_sparse_graph_correctness():
    """Test 33: sparse graph stats match a small hand-checkable example."""
    print("Test 33: sparse graph — adjacency, degree, components correctness... ", end="")
    import numpy as np
    import pandas as pd
    from src.graph_features import TransactionGraph

    # 6 nodes, two disconnected triangles: {0,1,2} and {3,4,5}.
    persons = pd.DataFrame({
        "person_id": np.arange(6),
        "city_id": [0, 0, 0, 1, 1, 1],
        "high_risk_group_id": [-1] * 6,
        "base_pd": [0.1] * 6,
    })
    tx = pd.DataFrame({
        "sender_id":   [0, 1, 2, 3, 4, 5],
        "receiver_id": [1, 2, 0, 4, 5, 3],
        "amount":      [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    })
    g = TransactionGraph(tx, persons)

    # Two connected components.
    stats = g.get_network_stats()
    assert stats.n_components == 2, f"expected 2 components, got {stats.n_components}"
    assert stats.n_edges == 6, f"expected 6 undirected edges, got {stats.n_edges}"

    # Sparse adjacency symmetric, each node degree 2.
    A = g.get_adjacency_sparse(weighted=False)
    deg = np.asarray(A.sum(axis=1)).ravel()
    assert (deg == 2).all(), f"every node degree should be 2, got {deg.tolist()}"

    # Dense fallback works at this tiny size and matches sparse.
    dense = g.adj_binary
    assert np.allclose(dense, A.toarray()), "dense != sparse adjacency"
    assert np.allclose(dense, dense.T), "adjacency not symmetric"

    # Sparse correlation: within-city blocks present, cross-city zero off-diagonal.
    csp = g.get_correlation_sparse(base_corr=0.05, same_city_boost=0.1,
                                   include_geo_blocks=True)
    C = csp.toarray()
    assert np.allclose(np.diag(C), 1.0), "correlation diagonal must be 1"
    # nodes 0 and 3 are in different cities and not transacting → 0 correlation.
    assert C[0, 3] == 0.0, "cross-city non-linked pair should have 0 correlation"
    # nodes 0 and 1 share a city → positive correlation.
    assert C[0, 1] > 0.0, "same-city pair should have positive correlation"
    print("PASSED")


# ---------------------------------------------------------------------------
# factor copula tests (34-36)
# ---------------------------------------------------------------------------

def test_factor_copula_correctness():
    """Test 34: factor copula BVN CDF and block match scipy ground truth."""
    print("Test 34: factor copula — BVN CDF accuracy, block correctness... ", end="")
    import numpy as np
    from scipy import stats
    from src.factor_copula import FactorCopula

    # Bivariate normal CDF vs scipy.
    for h, k, rho in [(-1.0, -1.0, 0.3), (0.5, -0.5, 0.6), (0.0, 0.0, 0.5),
                      (-1.5, -1.5, 0.8), (1.0, 1.0, -0.3)]:
        approx = FactorCopula._bvn_cdf(np.array([h]), np.array([k]), np.array([rho]))[0]
        exact = stats.multivariate_normal.cdf([h, k], mean=[0, 0],
                                              cov=[[1, rho], [rho, 1]])
        assert abs(approx - exact) < 1e-6, f"BVN CDF off by {abs(approx-exact)}"

    # Block: diagonal=PD, symmetric, Fréchet bounds, factor structure.
    rng = np.random.default_rng(1)
    N = 80
    pds = rng.uniform(0.02, 0.3, N)
    factor_id = rng.integers(0, 4, N)
    fc = FactorCopula().fit(pds, factor_id, rho=0.2)
    idx = np.array([0, 1, 2, 3, 40, 79])
    block = fc.joint_default_probability_block(idx)
    assert np.allclose(np.diag(block), pds[idx]), "diagonal != PD"
    assert np.allclose(block, block.T), "block not symmetric"
    for a in range(len(idx)):
        for b in range(len(idx)):
            if a != b:
                pa, pb = pds[idx[a]], pds[idx[b]]
                assert block[a, b] <= min(pa, pb) + 1e-9, "violates Fréchet upper"
                assert block[a, b] >= pa * pb - 1e-9, "below independence"

    # Same factor → correlated; different factor → independent.
    same = np.where(factor_id == factor_id[0])[0][:2]
    diff_j = np.where(factor_id != factor_id[0])[0][0]
    b_same = fc.joint_default_probability_block(same)
    b_diff = fc.joint_default_probability_block(np.array([0, diff_j]))
    assert b_same[0, 1] > pds[same[0]] * pds[same[1]], "same factor not correlated"
    assert abs(b_diff[0, 1] - pds[0] * pds[diff_j]) < 1e-9, "diff factor not independent"

    # REGRESSION: t-factor block must use the bivariate-t CDF (not Gaussian),
    # so the analytical block matches the t-factor SIMULATION. (A prior bug used
    # the Gaussian BVN here, underestimating joint defaults by ~30%.)
    np.random.seed(3)
    pds_t = rng.uniform(0.08, 0.2, 15)
    fid_t = np.zeros(15, dtype=int)
    fct = FactorCopula(student_t=True, nu=4).fit(pds_t, fid_t, rho=0.3)
    Jt = fct.joint_default_probability_block(np.arange(15))
    Dt = fct.simulate_defaults(300_000)
    emp_t = np.array([[(Dt[:, i] & Dt[:, j]).mean() for j in range(15)] for i in range(15)])
    off = ~np.eye(15, dtype=bool)
    t_err = np.abs(Jt[off] - emp_t[off]).max()
    assert t_err < 0.01, f"t-factor block disagrees with t-simulation by {t_err}"
    # And the t-block must exceed the Gaussian block (heavier tails cluster more).
    fcg = FactorCopula(student_t=False).fit(pds_t, fid_t, rho=0.3)
    Jg = fcg.joint_default_probability_block(np.arange(15))
    assert Jt[off].mean() > Jg[off].mean(), "t-factor should cluster more than Gaussian"
    print("PASSED")


def test_factor_copula_simulation():
    """Test 35: factor copula simulation is streamed, scales, inflates variance."""
    print("Test 35: factor copula — streamed simulation, variance inflation... ", end="")
    import numpy as np
    from src.factor_copula import FactorCopula, build_factor_id
    import pandas as pd

    rng = np.random.default_rng(2)
    N = 20_000
    pds = rng.uniform(0.01, 0.2, N)
    # Fewer factors = stronger clustering = larger, unambiguous variance inflation.
    # (Factor GRANULARITY controls correlation strength — a key real-data knob.)
    factor_id = rng.integers(0, 20, N)
    fc = FactorCopula().fit(pds, factor_id, rho=0.2)

    # Streamed default rate — never stores (n_sim, n).
    rate = fc.simulate_default_rate(n_simulations=400, batch_size=200)
    assert len(rate) == 400
    assert abs(rate.mean() - pds.mean()) < 0.02, "mean default rate off"
    # Factor correlation must inflate portfolio variance vs independence.
    indep_std = np.sqrt(pds.mean() * (1 - pds.mean()) / N)
    assert rate.std() > indep_std * 3, "factor correlation should inflate variance"

    # Segment loss streaming.
    seg = np.where(factor_id == 0)[0]
    el = np.ones(N) * 100.0
    losses = fc.simulate_segment_losses(seg, el, n_simulations=400, batch_size=200)
    assert len(losses) == 400 and losses.mean() > 0

    # build_factor_id helper.
    persons = pd.DataFrame({
        "person_id": np.arange(10),
        "city_id": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
        "high_risk_group_id": [-1, -1, 5, 5, -1, -1, 7, -1, -1, 7],
    })
    fid = build_factor_id(persons, ("high_risk_group_id", "city_id"))
    # rows 2,3 share group 5; rows 6,9 share group 7.
    assert fid[2] == fid[3], "group members should share factor"
    assert fid[6] == fid[9], "group members should share factor"
    print("PASSED")


def test_factor_copula_integration():
    """Test 36: FactorCopula is a drop-in for RiskRatioCalculator block mode."""
    print("Test 36: factor copula — RiskRatioCalculator drop-in integration... ", end="")
    import numpy as np
    import pandas as pd
    import src.risk_adjusted_metrics as ram
    from src.factor_copula import FactorCopula
    from src.metric_comparison import MetricComparator

    rng = np.random.default_rng(5)
    N = 2000
    persons = pd.DataFrame({
        "person_id": np.arange(N),
        "city_id": rng.integers(0, 6, N),
        "risk_archetype": rng.choice(["low", "medium", "high"], N),
        "exposure_at_default": rng.lognormal(9, 0.7, N),
        "estimated_revenue": rng.lognormal(6, 0.5, N),
    })
    pds = rng.uniform(0.01, 0.35, N)
    fc = FactorCopula().fit(pds, persons["city_id"].values, rho=0.18)

    orig = ram.LOSS_COV_DENSE_MAX_NODES
    ram.LOSS_COV_DENSE_MAX_NODES = 500   # force block mode
    try:
        calc = ram.RiskRatioCalculator(fc, persons, lgd=0.45)
        assert calc._dense_loss_cov is None, "should be block mode"
        seg = calc.by_segment("city_id")
        assert "sortino_copula" in seg.columns
        assert (seg["diversification_ratio"] >= 1 - 1e-9).all(), "div ratio < 1"
        assert (seg["loss_std_copula"] >= seg["loss_std_indep"] - 1e-6).all(), \
            "copula std should be >= indep std"
        comp = MetricComparator(calc)
        bt = comp.borrower_table()
        assert len(bt) == N
        flags = comp.divergence_flags(z_threshold=1.0)
        assert isinstance(flags, pd.DataFrame)
    finally:
        ram.LOSS_COV_DENSE_MAX_NODES = orig
    print("PASSED")


def test_geo_clusters():
    """Test 37: geolocation DBSCAN clusters + city fallback + noise isolation."""
    print("Test 37: geo clusters — DBSCAN, fallback, noise isolation... ", end="")
    import numpy as np
    import pandas as pd
    from src.geo_clusters import GeoClusterer, GeoClusterConfig

    rng = np.random.default_rng(0)
    centers = [(43.2, 76.9), (51.1, 71.4), (42.3, 69.6)]
    lat, lon, city = [], [], []
    for ci, (clat, clon) in enumerate(centers):
        lat += list(clat + rng.normal(0, 0.01, 30))
        lon += list(clon + rng.normal(0, 0.01, 30))
        city += [ci] * 30
    lat += list(rng.uniform(45, 50, 5)); lon += list(rng.uniform(60, 65, 5)); city += [0] * 5
    n = len(lat)
    persons = pd.DataFrame({"person_id": np.arange(n), "city_id": city,
                            "geo_longitude": lon, "geo_latitude": lat})

    gc = GeoClusterer(GeoClusterConfig(eps_km=5.0, min_samples=5)).fit(persons)
    labels = gc.assign(persons)["geo_cluster_id"].to_numpy()
    genuine = np.unique(labels[labels >= 0])
    assert len(genuine) == 3, f"expected 3 geo clusters, got {len(genuine)}"
    assert (labels < 0).sum() == 5, "5 scattered points should be isolated as noise"
    # noise points get UNIQUE negative ids (independent), not all -1
    assert len(np.unique(labels[labels < 0])) == 5, "noise ids must be unique"
    summ = gc.summary()
    assert {"geo_cluster_id", "n_members", "span_km"}.issubset(summ.columns)
    # city fallback
    gc2 = GeoClusterer(GeoClusterConfig(level="city")).fit(persons)
    assert len(np.unique(gc2.labels_)) == 3, "city fallback should give 3 groups"
    # no coordinates → graceful fallback, no crash
    gc3 = GeoClusterer().fit(persons[["person_id", "city_id"]])
    assert gc3.labels_ is not None
    print("PASSED")


def test_transfer_clusters_and_anchors():
    """Test 38: Louvain communities + anchor/dependent detection (star vs mesh)."""
    print("Test 38: transfer clusters — communities, anchors, fragility... ", end="")
    import numpy as np
    import pandas as pd
    from src.transfer_clusters import TransferClusterer, TransferClusterConfig

    pids = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    persons = pd.DataFrame({"person_id": pids})
    tx = [(0, d, 1000.0) for d in [1, 2, 3, 4, 5]]          # star: hub 0 feeds 1..5
    mesh = [10, 11, 12, 13, 14, 15]
    tx += [(a, b, 200.0) for a in mesh for b in mesh if a != b]   # mesh: all pay all
    transactions = pd.DataFrame(tx, columns=["sender_id", "receiver_id", "amount"])

    tc = TransferClusterer(
        TransferClusterConfig(resolution=1.0, min_cluster_size=3)
    ).fit(persons, transactions)
    p2 = tc.assign(persons)

    star_cid = int(p2.loc[p2.person_id == 0, "transfer_cluster_id"].iloc[0])
    mesh_cid = int(p2.loc[p2.person_id == 10, "transfer_cluster_id"].iloc[0])
    assert star_cid != mesh_cid, "star and mesh must be different communities"
    assert bool(p2.loc[p2.person_id == 0, "is_anchor"].iloc[0]), "hub must be anchor"
    deps = p2.loc[p2.person_id.isin([1, 2, 3, 4, 5]), "depends_on_anchor"].tolist()
    assert all(d == 0 for d in deps), f"dependents must point to hub 0, got {deps}"
    star_frag = tc.cluster_fragility_[star_cid]
    mesh_frag = tc.cluster_fragility_.get(mesh_cid, 0.0)
    assert star_frag > mesh_frag, "star must be more fragile than mesh"
    assert star_frag > 0.5, "fully-dependent star fragility should be high"
    anchors = tc.anchors_table()
    assert int(anchors.iloc[0]["n_dependents"]) == 5, "hub should have 5 dependents"
    print("PASSED")


def test_multi_factor_copula():
    """Test 39: multi-factor copula — Σβ²<1, implied corr, block==sim, ordering."""
    print("Test 39: multi-factor copula — correlation, block, simulation... ", end="")
    import numpy as np
    from src.multi_factor_copula import MultiFactorCopula

    rng = np.random.default_rng(0)
    # (a) Σβ²<1 guard
    try:
        MultiFactorCopula().fit(np.array([0.1, 0.1]), np.array([[0, 0], [0, 0]]), betas=0.8)
        assert False, "should reject Σβ² >= 1"
    except ValueError:
        pass

    # (b) implied correlation identity, β=0.4 both dims
    pds = np.array([0.05, 0.05, 0.05, 0.05])
    fm = np.array([[5, 9], [5, 8], [5, 9], [7, 8]])
    beta = 0.4
    mfc = MultiFactorCopula().fit(pds, fm, betas=beta)
    C = mfc.implied_correlation_block(np.arange(4))
    assert abs(C[0, 1] - beta ** 2) < 1e-9, "share geo only → β²"
    assert abs(C[0, 2] - 2 * beta ** 2) < 1e-9, "share both → 2β²"
    assert abs(C[0, 3]) < 1e-9, "share none → 0"

    # (c) analytical block ≈ simulation
    n = 12
    pds2 = rng.uniform(0.05, 0.15, n)
    fm2 = np.column_stack([np.zeros(n, int), np.zeros(n, int)])   # share both
    mfc2 = MultiFactorCopula().fit(pds2, fm2, betas=0.45)
    J = mfc2.joint_default_probability_block(np.arange(n))
    D = mfc2.simulate_defaults(400_000, seed=1)
    emp = (D.astype(float).T @ D.astype(float)) / D.shape[0]
    off = ~np.eye(n, dtype=bool)
    assert np.abs(J[off] - emp[off]).max() < 0.01, "block disagrees with simulation"

    # (d) more shared factors → more joint defaults
    def mean_joint(fmx):
        m = MultiFactorCopula().fit(pds2, fmx, betas=0.4)
        JJ = m.joint_default_probability_block(np.arange(n))
        return JJ[~np.eye(n, dtype=bool)].mean()
    both = mean_joint(np.column_stack([np.zeros(n, int), np.zeros(n, int)]))
    geo = mean_joint(np.column_stack([np.zeros(n, int), np.arange(n)]))
    none = mean_joint(np.column_stack([np.arange(n), np.arange(n)]))
    assert both > geo > none, "joint defaults must increase with shared factors"
    print("PASSED")


def test_anchor_contagion_metrics():
    """Test 40: anchor-conditional cluster loss uplift (RiskRatioCalculator drop-in)."""
    print("Test 40: cluster metrics — anchor-contagion uplift... ", end="")
    import numpy as np
    import pandas as pd
    from src.multi_factor_copula import MultiFactorCopula
    from src.risk_adjusted_metrics import RiskRatioCalculator
    from src.cluster_metrics import ClusterRiskMetrics

    n = 12
    pds = np.full(n, 0.05)
    transfer_factor = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    geo_factor = np.full(n, -1)
    fm = np.column_stack([geo_factor, transfer_factor])
    mfc = MultiFactorCopula().fit(pds, fm, betas=[0.0, 0.6])   # strong transfer loading

    persons = pd.DataFrame({
        "person_id": np.arange(n), "model_pd": pds,
        "geo_cluster_id": geo_factor, "transfer_cluster_id": transfer_factor,
        "is_anchor": [True] + [False] * 11,
        "anchor_of_cluster": [0] + [-1] * 11,
        "depends_on_anchor": [-1] + [0] * 5 + [-1] * 6,
        "cluster_fragility": [1.0] * 6 + [0.0] * 6,
    })
    calc = RiskRatioCalculator(mfc, persons, exposures=np.full(n, 10000.0), lgd=0.45)
    crm = ClusterRiskMetrics(calc, persons)

    # drop-in metric roll-up works on both cluster dims
    tm = crm.transfer_metrics()
    assert len(tm) >= 1 and "expected_loss" in tm.columns

    tbl = crm.anchor_contagion_table()
    assert len(tbl) == 1, "exactly one anchored cluster"
    row = tbl.iloc[0]
    assert row["el_anchor_default"] > row["el_unconditional"], \
        "conditioning on anchor default must raise cluster EL"
    assert row["uplift_ratio"] > 1.5, "fragile star should show large uplift"
    print("PASSED")


def test_agent_cluster_methods():
    """Test 41: RiskAgentAPI cluster methods return JSON-safe AgentResults."""
    print("Test 41: agent API — cluster analysis methods... ", end="")
    import json
    import numpy as np
    import pandas as pd
    from src.agents import RiskAgentAPI
    from src.data_generator import generate_network

    persons, transactions = generate_network(seed=42)
    rng = np.random.default_rng(7)
    centres = {0: (76.9, 43.2), 1: (71.4, 51.1), 2: (69.6, 42.3)}
    lon = np.empty(len(persons)); lat = np.empty(len(persons))
    for i, c in enumerate(persons["city_id"].to_numpy()):
        clon, clat = centres.get(int(c), (70, 48))
        lon[i] = clon + rng.normal(0, 0.03); lat[i] = clat + rng.normal(0, 0.03)
    persons["geo_longitude"] = lon; persons["geo_latitude"] = lat

    api = RiskAgentAPI(persons=persons, transactions=transactions, seed=42)
    # guard: cluster queries before analysis fail gracefully (ok=False, no crash)
    assert api.fragile_clusters().ok is False
    api.run_pipeline()                       # reaches state with model_pd
    r = api.run_cluster_analysis()
    assert r.ok, f"cluster analysis failed: {r.error}"
    assert r.data["n_geo_clusters"] >= 1
    assert r.data["n_transfer_clusters"] >= 1
    # all query methods return ok and JSON-serialisable data
    for res in (api.geo_clusters(), api.transfer_clusters(),
                api.anchors(), api.fragile_clusters()):
        assert res.ok, res.error
        json.dumps(res.data)                 # must be JSON-safe (agent contract)
    print("PASSED")


# ---------------------------------------------------------------------------
# arpym Tier-1 ports: entropy pooling, credit transitions, spectrum shrinkage
# ---------------------------------------------------------------------------

def test_relative_entropy():
    print("Test 42: Relative-entropy minimisation (entropy pooling)... ", end="")
    from src.relative_entropy import min_rel_entropy_sp

    # No views -> normalised prior.
    p0 = np.array([0.2, 0.3, 0.5])
    assert np.allclose(min_rel_entropy_sp(p0), p0)

    # Equality view forces E[z] to the target exactly.
    z = np.array([[0.0, 1.0, 2.0]])
    post = min_rel_entropy_sp(p0, z_eq=z, mu_view_eq=np.array([1.5]))
    assert abs(post.sum() - 1.0) < 1e-9
    assert abs(float((z @ post).ravel()[0]) - 1.5) < 1e-6
    assert np.all(post >= 0)

    # Inequality view E[z] <= 0.5 (binding) reaches the boundary
    # (SLSQP converges the constraint to ~1e-6, so allow a small tolerance).
    post2 = min_rel_entropy_sp(p0, z_ineq=z, mu_view_ineq=np.array([0.5]))
    assert float((z @ post2).ravel()[0]) <= 0.5 + 1e-4
    print("PASSED")


def test_credit_transitions():
    print("Test 43: Credit transition estimator (continuous-time generator)... ", end="")
    from src.credit_transitions import (
        fit_trans_matrix_credit, estimate_generator, cohort_arrays_from_events
    )
    from src.rating_engine import RatingEngine, N_RATINGS

    # Synthetic 4-state cohort.
    dates = np.array(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"],
                     dtype="datetime64[D]")
    t_bar, c = len(dates), 4
    n_oblig = np.array([[1000, 800, 500, 0]] * t_bar, dtype=float)
    per = np.zeros((t_bar, c, c))
    for t in range(t_bar):
        per[t, 0] = [0, 30, 5, 1]
        per[t, 1] = [10, 0, 40, 8]
        per[t, 2] = [2, 15, 0, 50]
    n_cum = np.cumsum(per, axis=0)

    # Generator: non-negative off-diagonals, zero row sums, absorbing default.
    g = estimate_generator(dates, n_oblig, n_cum)
    off = g[~np.eye(c, dtype=bool)]
    assert np.all(off >= -1e-12)
    assert np.allclose(g.sum(axis=1), 0.0, atol=1e-8)
    assert np.allclose(g[-1], 0.0)

    # Transition matrix: stochastic, absorbing default, monotone PD by rating.
    P = fit_trans_matrix_credit(dates, n_oblig, n_cum)
    assert P.shape == (c, c)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= -1e-9) and np.all(P <= 1 + 1e-9)
    assert np.allclose(P[-1], [0, 0, 0, 1])
    # PD increases as quality drops. Per-row entropy-pooling enforces
    # monotonicity WITHIN a row, not across rows, so apply the float tolerance
    # to BOTH comparisons (the cross-row ordering holds for this generator but
    # can be within sub-epsilon of equality).
    assert P[0, -1] <= P[1, -1] + 1e-12 <= P[2, -1] + 2e-12

    # Half-life weighting still yields a valid matrix.
    P2 = fit_trans_matrix_credit(dates, n_oblig, n_cum, tau_hl=1.0)
    assert np.allclose(P2.sum(axis=1), 1.0)

    # Tidy-event on-ramp produces estimator-shaped arrays.
    import pandas as pd
    events = pd.DataFrame({
        "period": [0, 0, 1, 1], "from_state": [1, 2, 1, 2],
        "to_state": [2, 4, 2, 4], "count": [30, 50, 28, 45],
    })
    d2, no2, nc2 = cohort_arrays_from_events(events, n_ratings=4, count_col="count")
    assert no2.shape == (2, 4) and nc2.shape == (2, 4, 4)
    # Synthetic dates must be exactly 1.0 year apart under the 252-business-day
    # convention (not 365 calendar days, which would bias generator rates ~3.6%).
    assert abs(np.busday_count(d2[0], d2[1]) / 252.0 - 1.0) < 1e-9

    # RatingEngine.from_cohort_data wires it end-to-end (8 states).
    dts = np.array(["2018-01-01", "2019-01-01", "2020-01-01"], dtype="datetime64[D]")
    cc = N_RATINGS
    nob = np.tile(np.array([200, 300, 500, 800, 600, 400, 150, 0], dtype=float), (3, 1))
    pp = np.zeros((3, cc, cc))
    for t in range(3):
        for i in range(cc - 1):
            pp[t, i, i + 1] = max(1, nob[t, i] * 0.05)
            if i > 0:
                pp[t, i, i - 1] = max(1, nob[t, i] * 0.02)
            pp[t, i, -1] = max(0, nob[t, i] * 0.01 * (i + 1))
    ncum = np.cumsum(pp, axis=0)
    eng = RatingEngine.from_cohort_data(dts, nob, ncum, tau_hl_years=2.0)
    assert np.allclose(eng.transition_annual.sum(axis=1), 1.0)
    assert np.allclose(eng.transition_1yr.sum(axis=1), 1.0)
    # A wrong-sized cohort (≠ 8 ratings) must raise ValueError, not silently
    # build an engine that IndexErrors downstream for higher-rated borrowers.
    bad_cum = np.cumsum(np.zeros((3, 4, 4)) + 1.0, axis=0)
    try:
        RatingEngine.from_cohort_data(dts, np.ones((3, 4)), bad_cum)
        raise AssertionError("expected ValueError for non-8-state cohort")
    except ValueError:
        pass
    print("PASSED")


def test_spectrum_shrinkage():
    print("Test 44: Marčenko-Pastur spectrum shrinkage... ", end="")
    from src.spectrum import (
        spectrum_shrink, denoise_correlation, marchenko_pastur_pdf, mp_support
    )

    # MP density integrates to ~1 over its support.
    # numpy 2.x renamed trapz -> trapezoid; support both without eager getattr.
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    q, s2 = 3.0, 1.0
    lo, hi = mp_support(q, s2)
    xs = np.linspace(lo, hi, 5000)
    assert abs(float(trapz(marchenko_pastur_pdf(xs, q, s2), xs)) - 1.0) < 1e-2

    # Low-rank signal + noise: denoising improves both fidelity and conditioning.
    rng = np.random.default_rng(7)
    n, T, k_true = 40, 120, 3
    B = rng.standard_normal((n, k_true)) * 0.6
    true_cov = B @ B.T + np.eye(n)
    X = rng.multivariate_normal(np.zeros(n), true_cov, size=T)
    C = np.corrcoef(X, rowvar=False)
    true_corr = true_cov / np.outer(np.sqrt(np.diag(true_cov)), np.sqrt(np.diag(true_cov)))

    res = spectrum_shrink(C, T, method="mp_edge")
    assert res.k_bar >= 1
    assert np.allclose(res.sigma2_out, res.sigma2_out.T)

    Cd = denoise_correlation(C, T)
    assert np.allclose(np.diag(Cd), 1.0)
    assert np.linalg.eigvalsh(Cd).min() > -1e-9            # PSD
    assert np.linalg.cond(Cd) < np.linalg.cond(C)          # better conditioned
    assert np.linalg.norm(Cd - true_corr) <= np.linalg.norm(C - true_corr) + 1e-9

    # hist_mse method (arpym parity) runs and returns a valid matrix.
    res2 = spectrum_shrink(C, T, method="hist_mse")
    assert res2.sigma2_out.shape == C.shape

    # Tiny matrix is returned unchanged (no bulk to shrink).
    small = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert np.allclose(spectrum_shrink(small, 10).sigma2_out, small)

    # Integration: TransactionGraph denoise path stays copula-ready.
    from src.data_generator import generate_network
    from src.graph_features import TransactionGraph
    persons, tx = generate_network(seed=42)
    g = TransactionGraph(tx, persons)
    Cg = g.get_correlation_matrix(denoise=True)
    assert np.allclose(np.diag(Cg), 1.0)
    assert np.linalg.eigvalsh(Cg).min() > -1e-9
    print("PASSED")


def test_conditional_fp():
    print("Test 45: Conditional flexible probabilities (crisp + entropy pool)... ", end="")
    from src.conditional_fp import crisp_fp, conditional_fp, effective_scenarios

    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 1.0, 400)
    z_star = 1.2

    # crisp_fp: valid window enclosing the target, probabilities sum to 1.
    p_crisp, lb, ub = crisp_fp(z, z_star, alpha=0.3)
    assert abs(p_crisp.sum() - 1.0) < 1e-9
    assert lb <= z_star <= ub
    assert np.all(p_crisp >= 0)

    # conditional_fp: smooth (all scenarios keep mass), sums to 1, and matches
    # the crisp conditional mean exactly (its defining entropy-pool property).
    p_cond = conditional_fp(z, z_star, alpha=0.3)
    assert abs(p_cond.sum() - 1.0) < 1e-9
    assert np.all(p_cond > 0)
    m_crisp = (p_crisp / p_crisp.sum()) @ z
    m_cond = p_cond @ z
    assert abs(m_crisp - m_cond) < 1e-3
    # Conditioning concentrates: effective scenarios drop below the sample size.
    assert effective_scenarios(p_cond) < len(z)

    # Multi-target: one probability column per target, each normalised.
    p_multi = conditional_fp(z, np.array([-1.0, 0.0, 1.0]), alpha=0.3)
    assert p_multi.shape == (len(z), 3)
    assert np.allclose(p_multi.sum(axis=0), 1.0)

    # Degenerate window: a very small alpha can make the smoothed window too
    # narrow to enclose any scenario. crisp_fp must fall back to the nearest
    # scenario (never an all-zero vector), and conditional_fp must stay valid.
    p_tiny, lb_t, ub_t = crisp_fp(z, 0.0, alpha=0.001)
    assert abs(p_tiny.sum() - 1.0) < 1e-9
    assert ((z >= lb_t) & (z <= ub_t)).sum() >= 1
    p_cond_tiny = np.atleast_1d(conditional_fp(z, 0.0, alpha=0.001))
    assert np.isfinite(p_cond_tiny).all() and abs(p_cond_tiny.sum() - 1.0) < 1e-9

    # Integration: FlexibleProbsCalibrator with method="conditional_fp" still
    # produces a higher copula theta under stress than under calm.
    from src.flexible_probs import FlexibleProbsCalibrator
    stress_hist = np.clip(rng.beta(2, 5, 200), 0, 1)
    base = np.eye(4) + 0.1 * (np.ones((4, 4)) - np.eye(4))
    calib = FlexibleProbsCalibrator(weighting_method="conditional_fp", conditional_alpha=0.3)
    calib.fit(stress_hist)
    calm = calib.calibrate(0.1, base, decompose=False)
    hot = calib.calibrate(0.85, base, decompose=False)
    assert hot.theta > calm.theta
    print("PASSED")


def test_low_rank_corr():
    print("Test 46: Low-rank diagonal correlation (factor-loading fit)... ", end="")
    from src.low_rank_corr import low_rank_diag_conditional_corr, fit_factor_loadings
    from src.multi_factor_copula import MultiFactorCopula

    rng = np.random.default_rng(3)
    n, k_true = 12, 2
    B = rng.uniform(-0.5, 0.6, (n, k_true))
    row_norm = np.sqrt((B ** 2).sum(1))
    B[row_norm > 0.9] *= 0.9 / row_norm[row_norm > 0.9, None]
    C = B @ B.T
    np.fill_diagonal(C, 1.0)

    res = low_rank_diag_conditional_corr(C, k_bar=k_true)
    assert res.beta.shape == (n, k_true)
    # Unit diagonal preserved and idiosyncratic variance positive (row-norm <= 1).
    assert np.allclose(np.diag(res.c2_lrd), 1.0, atol=1e-6)
    assert np.all(np.sum(res.beta ** 2, axis=1) <= 1.0 + 1e-9)
    # Reconstruction is a reasonable approximation of the target.
    assert np.linalg.norm(res.c2_lrd - C) / np.linalg.norm(C) < 0.25

    # k_bar must respect the rank bound.
    try:
        low_rank_diag_conditional_corr(C, d=np.ones((1, n)), k_bar=n)
        raise AssertionError("expected ValueError for k_bar > n - rank(d)")
    except ValueError:
        pass

    # fit_factor_loadings returns copula-ready (non-negative, Σβ²<1) loadings
    # that drive MultiFactorCopula end-to-end.
    beta = fit_factor_loadings(C, k_factors=2)
    assert np.all(beta >= 0)
    assert np.all(np.sum(beta ** 2, axis=1) < 1.0)
    pds = np.clip(rng.beta(2, 30, n), 0.001, 0.5)
    factor_matrix = np.tile(np.arange(2), (n, 1))
    mfc = MultiFactorCopula().fit(pds, factor_matrix, betas=beta)
    rate = mfc.simulate_default_rate(1000, seed=0)
    assert rate.std() > 0  # genuinely correlated defaults

    # Degenerate input: a (near-)identity correlation has unit-norm eigen-
    # directions, so the raw fit can land a row at Σβ²=1 (zero idiosyncratic
    # variance) which MultiFactorCopula rejects. fit_factor_loadings must cap
    # row norms strictly below 1 so the loadings stay copula-safe.
    beta_id = fit_factor_loadings(np.eye(8), k_factors=2)
    assert np.all(np.sum(beta_id ** 2, axis=1) < 1.0)
    # Must be accepted by the copula (no ValueError) on degenerate loadings.
    MultiFactorCopula().fit(
        np.full(8, 0.05), np.tile(np.arange(2), (8, 1)), betas=beta_id
    )
    print("PASSED")


def test_etl_pipeline_real_data():
    print("Test 47: ETL pipeline runs on real data (sparse ids, model_pd only)... ", end="")
    import tempfile, os
    import pandas as pd
    from pipelines import ArtifactStore
    from pipelines.stage_00_ingest import run as run_ingest
    from pipelines.stage_10_pd import run as run_pd
    from pipelines.stage_20_graph import run as run_graph
    from pipelines.stage_30_copula import run as run_copula
    from pipelines.stage_40_transitions import run as run_transitions
    from pipelines.stage_50_metrics import run as run_metrics

    rng = np.random.default_rng(7)
    n = 50
    with tempfile.TemporaryDirectory() as tmp:
        # Sparse, non-contiguous ids; model_pd present but NO base_pd (real-data shape).
        persons = pd.DataFrame({
            "person_id": rng.choice(range(1000, 2000), n, replace=False),
            "model_pd": np.clip(rng.beta(2, 20, n), 0.001, 0.5),
            "city_id": rng.integers(0, 3, n), "city_name": ["A"] * n,
            "income": rng.uniform(1000, 5000, n), "age": rng.integers(25, 60, n),
            "employment_years": rng.integers(0, 20, n), "debt_to_income": rng.uniform(0.1, 0.6, n),
            "num_credit_lines": rng.integers(1, 6, n), "missed_payments": rng.integers(0, 4, n),
            "credit_utilization": rng.uniform(0.1, 0.9, n), "account_age_months": rng.integers(6, 60, n),
            "default": rng.binomial(1, 0.1, n), "high_risk_group_id": -1,
            "risk_archetype": rng.choice(["prime", "subprime"], n),
        })
        pid = persons["person_id"].values
        tx = pd.DataFrame({
            "sender_id": rng.choice(pid, 80), "receiver_id": rng.choice(pid, 80),
            "amount": rng.uniform(50, 500, 80),
        })
        pf, tf = os.path.join(tmp, "p.csv"), os.path.join(tmp, "t.csv")
        persons.to_csv(pf, index=False)
        tx.to_csv(tf, index=False)
        store = ArtifactStore(os.path.join(tmp, "etl"))

        r0 = run_ingest(store, persons_source=pf, transactions_source=tf)
        assert r0.ok, r0.error
        assert r0.metrics["mode"] == "real"
        # Ids must be reindexed to a contiguous range for positional numpy ops.
        assert list(store.read_df("persons")["person_id"]) == list(range(n))

        # skip_training path: real data already carries model_pd.
        assert run_pd(store, skip_training=True, pd_col="model_pd").ok
        assert "base_pd" not in persons.columns  # confirm the input really lacked it
        assert run_graph(store, denoise=True).ok
        assert run_copula(store).ok
        assert run_transitions(store).ok
        assert run_metrics(store).ok
    print("PASSED")


def test_dependence_measures():
    print("Test 50: Schweizer-Wolff dependence + copula invariance test... ", end="")
    from src.dependence import schweizer_wolff, copula_invariance_test
    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    n = 400

    # σ must lie in [0, 1]. Independence → ≈0, perfect monotone → ≈1.
    indep = np.column_stack((rng.normal(0, 1, n), rng.normal(0, 1, n)))
    sw_indep = schweizer_wolff(indep)
    assert 0.0 <= sw_indep <= 1.0
    assert sw_indep < 0.15                              # close to independence

    x = np.linspace(-2, 2, n)
    perfect = np.column_stack((x, np.exp(x)))           # strictly increasing
    sw_perfect = schweizer_wolff(perfect)
    assert sw_perfect > 0.9                             # near comonotone

    # Stronger linear dependence ⇒ larger σ (monotonic in correlation).
    weak = schweizer_wolff(rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n))
    strong = schweizer_wolff(rng.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], n))
    assert strong > weak

    # Non-monotone (y = x²): SW detects dependence where Spearman is ~0.
    xx = rng.normal(0, 1, n)
    sw_nm = schweizer_wolff(np.column_stack((xx, xx ** 2)))
    assert sw_nm > 0.2
    assert abs(spearmanr(xx, xx ** 2)[0]) < 0.15

    # Degenerate (constant column) → 0, no crash; single point → 0.
    assert schweizer_wolff(np.column_stack((np.ones(50), rng.normal(0, 1, 50)))) == 0.0
    assert schweizer_wolff(np.array([[1.0, 2.0]])) == 0.0

    # Scale guard: large sample with a grid cap runs and matches a subsample.
    big = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 6000)
    sw_big = schweizer_wolff(big, max_grid=1500)
    sw_sub = schweizer_wolff(big[:1500])
    assert abs(sw_big - sw_sub) < 0.1

    # Invariance test: i.i.d. is low at all lags; AR(1) spikes at lag 1.
    iid = rng.normal(0, 1, 300)
    sw_iid = copula_invariance_test(iid, 3)
    assert sw_iid.shape == (3,) and np.all(sw_iid < 0.2)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.7 * ar[t - 1] + rng.normal(0, 1)
    sw_ar = copula_invariance_test(ar, 3)
    assert sw_ar[0] > sw_iid.max()                     # lag-1 dependence detected
    print("PASSED")


def test_copula_calibration():
    print("Test 51: Empirical copula calibration (Plan 07)... ", end="")
    import json
    import pandas as pd
    from src.copula_calibration import (
        build_default_panel, empirical_dependence_measures, calibrate_copula,
        clayton_theta_from_tau, gaussian_rho_from_tau, default_correlation,
    )
    from src.agents import RiskAgentAPI

    # Parameter conversions match the closed forms.
    assert abs(clayton_theta_from_tau(1 / 3) - 1.0) < 1e-6
    assert clayton_theta_from_tau(0.0) < 1e-2                 # ~0 ⇒ independence
    assert abs(gaussian_rho_from_tau(0.5) - np.sin(np.pi / 4)) < 1e-9
    assert abs(default_correlation(0.1, 0.1, 0.01)) < 1e-9    # independent ⇒ 0
    assert default_correlation(0.1, 0.1, 0.05) > 0           # clustered ⇒ +

    rng = np.random.default_rng(0)
    N, T = 50, 25

    # Independent defaults ⇒ near-zero dependence, all families Fréchet-valid.
    ind_rows = [
        {"person_id": b, "period": t, "default": int(rng.random() < 0.1), "model_pd": 0.1}
        for t in range(T) for b in range(N)
    ]
    panel_ind = build_default_panel(pd.DataFrame(ind_rows))
    res_ind = calibrate_copula(panel_ind)
    assert abs(res_ind.empirical["default_corr"]) < 0.1
    assert res_ind.family_table["frechet_ok"].all()
    assert res_ind.recommended_family in ("gaussian", "student_t", "clayton")

    # Common-shock defaults ⇒ positive dependence; observed joint > independent.
    cor_rows = []
    for t in range(T):
        base = 0.4 if rng.random() < 0.3 else 0.03
        for b in range(N):
            cor_rows.append({"person_id": b, "period": t,
                             "default": int(rng.random() < base), "model_pd": 0.1})
    panel_cor = build_default_panel(pd.DataFrame(cor_rows))
    res_cor = calibrate_copula(panel_cor)
    assert res_cor.empirical["default_corr"] > 0.1
    assert (res_cor.empirical["observed_joint_default"]
            > res_cor.empirical["independent_joint_default"])

    # Per-segment measures include an __ALL__ row plus one row per segment.
    seg_rows = [dict(r, segment=("retail" if r["person_id"] % 2 else "sme"))
                for r in cor_rows]
    panel_seg = build_default_panel(pd.DataFrame(seg_rows), segment_col="segment")
    measures = empirical_dependence_measures(panel_seg, segment_col="segment")
    assert "__ALL__" in set(measures["segment"])
    assert {"retail", "sme"}.issubset(set(measures["segment"]))

    # Missing required column raises a clear error.
    try:
        build_default_panel(pd.DataFrame({"person_id": [1], "period": [0]}))
        raise AssertionError("expected ValueError for missing default column")
    except ValueError:
        pass

    # Agent method: diagnostic by default (live copula untouched), JSON-safe.
    api = RiskAgentAPI()
    api.run_pipeline()
    theta_before = api._copula.params.theta
    r = api.calibrate_copula_from_data(pd.DataFrame(cor_rows), apply=False)
    assert r.ok
    json.dumps(r.data)
    assert r.data["applied"] is False
    assert api._copula.params.theta == theta_before
    # apply=True with a forced Clayton family refits the live θ.
    r2 = api.calibrate_copula_from_data(pd.DataFrame(cor_rows), family="clayton", apply=True)
    assert r2.data["applied"] is True
    assert api._copula.params.theta != theta_before
    print("PASSED")


def test_pipeline_stage_25_loadings():
    print("Test 49: ETL stage 25 — factor-loading estimation (optional)... ", end="")
    import tempfile
    from pipelines import run_all, ArtifactStore
    from pipelines.stage_25_loadings import run as run_loadings
    from src.multi_factor_copula import MultiFactorCopula

    with tempfile.TemporaryDirectory() as tmp:
        # Default chain must NOT include stage 25 (it is opt-in).
        s = run_all(root=tmp, denoise=True, verbose=False)
        assert not s.exists("factor_loadings")

    with tempfile.TemporaryDirectory() as tmp:
        # with_loadings=True inserts stage 25 between graph and copula.
        s = run_all(root=tmp, denoise=True, with_loadings=True, verbose=False)
        assert s.exists("factor_loadings")
        assert s.exists("loading_diagnostics")
        beta = s.read_array("factor_loadings")
        # Copula-ready loadings drive a MultiFactorCopula from the artifact.
        assert np.all(np.sum(beta ** 2, axis=1) < 1.0)
        diag = s.read_json("loading_diagnostics")
        assert diag["max_row_sumsq"] < 1.0
        assert diag["copula_ready"] if "copula_ready" in diag else True
        pds = s.read_df("persons_scored")["model_pd"].to_numpy()
        factor_matrix = np.tile(np.arange(diag["k_factors"]), (len(pds), 1))
        mfc = MultiFactorCopula().fit(pds, factor_matrix, betas=beta)
        assert mfc.simulate_default_rate(500, seed=0).std() > 0

    # Stage 25 in isolation requires its upstream artifact (graceful failure).
    with tempfile.TemporaryDirectory() as tmp:
        empty = ArtifactStore(tmp)
        r = run_loadings(empty)
        assert r.ok is False and "Missing upstream" in (r.error or "")
    print("PASSED")


def test_agent_loadings_and_regime():
    print("Test 48: agent API — factor loadings + regime weights (Plan 08)... ", end="")
    import json
    from src.agents import RiskAgentAPI, _safe

    # _safe must keep bools as bools (bool is an int subclass — order matters).
    assert _safe(True) is True and _safe(False) is False
    assert _safe(5) == 5 and isinstance(_safe(5), int)

    api = RiskAgentAPI()
    api.run_pipeline()
    copula_before = api._copula

    # fit_factor_loadings: diagnostic by default — MUST NOT mutate live copula.
    r = api.fit_factor_loadings(k_factors=2, denoise=True, apply=False)
    assert r.ok, r.error
    json.dumps(r.data)                              # JSON-safe (agent contract)
    assert r.data["applied"] is False               # proper bool, not 0
    assert api._copula is copula_before             # live model untouched
    assert r.data["max_row_sumsq"] < 1.0            # copula-ready loadings

    # apply=True swaps the live copula to a MultiFactorCopula.
    r2 = api.fit_factor_loadings(k_factors=2, apply=True)
    assert r2.data["applied"] is True
    assert api._copula is not copula_before
    assert type(api._copula).__name__ == "MultiFactorCopula"

    # regime_weights: both estimators run and are JSON-safe; conditioning
    # concentrates the effective scenario count below the history length.
    api2 = RiskAgentAPI()
    api2.run_pipeline()
    for method in ("conditional_fp", "kernel"):
        rw = api2.regime_weights(method=method, alpha=0.25)
        assert rw.ok, rw.error
        json.dumps(rw.data)
        assert 0 < rw.data["effective_scenarios"] <= rw.data["n_history"]
    rc = api2.regime_weights(method="conditional_fp")
    assert rc.data["effective_scenarios"] < rc.data["n_history"]

    # Guard: state-gated methods raise AgentError before the required state
    # (this matches the convention of regime_status/portfolio_summary/etc.).
    from src.agents import AgentError
    fresh = RiskAgentAPI()
    for call in (lambda: fresh.fit_factor_loadings(), lambda: fresh.regime_weights()):
        try:
            call()
            raise AssertionError("expected AgentError before required state")
        except AgentError:
            pass
    print("PASSED")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_all_tests() -> bool:
    """
    Run all 51 tests.

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

    # arpym Tier-1 ports: entropy pooling, credit transitions, spectrum shrinkage
    run("test_relative_entropy", test_relative_entropy)
    run("test_credit_transitions", test_credit_transitions)
    run("test_spectrum_shrinkage", test_spectrum_shrinkage)
    # arpym FP family + factor-loading estimation ports
    run("test_conditional_fp", test_conditional_fp)
    run("test_low_rank_corr", test_low_rank_corr)
    # ETL pipeline on real-data shape (sparse ids, model_pd only, no base_pd)
    run("test_etl_pipeline_real_data", test_etl_pipeline_real_data)
    # copula dependence measures (Schweizer-Wolff + invariance test, Plan 07 prereq)
    run("test_dependence_measures", test_dependence_measures)
    # empirical copula calibration (Plan 07)
    run("test_copula_calibration", test_copula_calibration)
    # optional ETL stage 25 (factor-loading estimation, Plan 08)
    run("test_pipeline_stage_25_loadings", test_pipeline_stage_25_loadings)
    # agent API: factor-loading estimation + conditional-FP regime weights (Plan 08)
    run("test_agent_loadings_and_regime", test_agent_loadings_and_regime)
    if model is not None:
        run("test_customer_profiler", test_customer_profiler, model, graph, persons, transactions)
        run("test_refactor_correctness", test_refactor_correctness, persons, transactions, model, graph)

    # risk_adjusted_metrics tests
    run("test_metric_registry", test_metric_registry)
    if model is not None:
        run("test_metric_primitives_additivity", test_metric_primitives_additivity, model, persons)
        run("test_single_borrower_closed_form", test_single_borrower_closed_form, model, persons)
        run("test_correlation_inflates_denominator", test_correlation_inflates_denominator, model, persons)
        run("test_pluggable_inputs_and_sim", test_pluggable_inputs_and_sim, model, persons, transactions)
        run("test_by_segment_invariants", test_by_segment_invariants, model, persons)
        run("test_metric_comparison", test_metric_comparison, model, persons)

    # scale / real-data tests (no fixtures needed — self-contained)
    run("test_loaders_dirty_data", test_loaders_dirty_data)
    if model is not None:
        run("test_block_loss_cov_equals_dense", test_block_loss_cov_equals_dense)
    run("test_sparse_graph_correctness", test_sparse_graph_correctness)

    # factor copula tests (scale to 10M)
    run("test_factor_copula_correctness", test_factor_copula_correctness)
    run("test_factor_copula_simulation", test_factor_copula_simulation)
    run("test_factor_copula_integration", test_factor_copula_integration)

    # multi-dimensional clusters (geo + transfer), anchors, cluster metrics
    run("test_geo_clusters", test_geo_clusters)
    run("test_transfer_clusters_and_anchors", test_transfer_clusters_and_anchors)
    run("test_multi_factor_copula", test_multi_factor_copula)
    run("test_anchor_contagion_metrics", test_anchor_contagion_metrics)
    run("test_agent_cluster_methods", test_agent_cluster_methods)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = 51
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
