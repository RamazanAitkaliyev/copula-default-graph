# ARPM Integration Analysis for Copula Default Graph

Date: 2026-06-06

This report maps what is present in the workspace, what the Copula Default Graph
platform already implements, what ARPM statistical, mathematical, and financial
concepts are available elsewhere in the repository, and what should be integrated
next.

## 1. Workspace Map

### 1.1 Main repository

Root inspected:

`/Users/ramazan.aitkaliyev/myFiles/arpm-root/git-repos/arpm-python`

Top-level structure:

- `projects/copula_default_graph/` - the credit-risk platform.
- `Python/arpym/arpym/` - reusable ARPyM package modules.
- `Python/scripts/sources/` - 435 research/demo scripts.
- `databases/temporary-databases/` - 144 temporary CSV/MAT data artifacts.
- `old-databases/global-databases/` - 53 legacy market, credit, pricing, and risk datasets.
- `Python/arpym/doc/` - Sphinx documentation, partly stale relative to this checkout.

Important repository caveat:

- The parent repository is already dirty, with many modified, deleted, and
  untracked files outside this report.
- The ARPyM docs list some functions that are not physically present in
  `Python/arpym/arpym`. The filesystem should be treated as authoritative.

### 1.2 Copula Default Graph project

Path:

`projects/copula_default_graph`

The project is a bank credit-risk analytical prototype for correlated default
modeling. It combines borrower PDs, transaction graph features, copulas, cluster
analysis, loss covariance, portfolio risk metrics, and agent-facing APIs.

Observed size:

- Source modules: 28 Python files under `src/` (including the seven ARPM ports:
  relative_entropy, credit_transitions, spectrum, conditional_fp, low_rank_corr,
  dependence, copula_calibration).
- Project docs, demos, and tests: about 23k total lines across source, docs,
  demos, and tests.
- Test suite: 51 tests.
- Generated outputs already present in `output/`: charts, CSVs, presentations,
  clustered persons, ratings, risk metrics, anchor contagion, stress tables, and
  watchlists.

Key docs:

- `README.md` - problem, architecture, quick commands, formulas.
- `ARCHITECTURE.md` - L0-L5 layered architecture.
- `METHODOLOGY.md` - math and finance formulas.
- `CAPABILITIES.md` - machine-readable capability catalog.
- `AGENTS.md` - agent contract and invariants.
- `RECIPES.md` - copy-paste workflows.
- `ROADMAP.md` - production platform gaps and migration plan.

## 2. Copula Default Graph: Current Capabilities

### 2.1 Data and ingestion

Implemented modules:

- `loaders.py` - column mapping, validation, PD percentage normalization,
  duplicate/NaN policies, transaction validation, contiguous reindexing.
- `data_generator.py` - synthetic persons, archetypes, transactions, bridges,
  high-risk groups, base PDs, and fraud-ring generation.
- `config.py` - dataclass configuration objects.

Main data contracts:

- `persons`: one row per borrower, unique `person_id`, PD column
  `model_pd` or `base_pd`, optional geography, EAD, revenue, default label,
  cluster columns.
- `transactions`: `sender_id`, `receiver_id`, `amount`.

Production relevance:

- The project is ready to accept a bank-supplied PD column through `loaders.py`.
- The project assumes contiguous integer IDs for numpy positional operations.

### 2.2 Individual PD modeling

Implemented modules:

- `pd_model.py` - logistic regression and gradient boosting via scikit-learn.
- `structural_pd.py` - Merton structural PD and a retail/KMV-style proxy.

Current theory:

- Statistical PD from ML.
- Structural balance-sheet PD from Merton:
  `PD = Phi(-d2)`.
- Blended / divergence signals to flag disagreement between statistical and
  structural views.

Current gap:

- No XGBoost/CatBoost backend yet.
- No explicit PD calibration layer such as Platt, isotonic, reliability curves,
  Brier score, expected calibration error, or out-of-time drift diagnostics.

### 2.3 Graph, clusters, and anchors

Implemented modules:

- `graph_features.py` - sparse transaction graph, centrality, PageRank,
  graph-derived risk features, sparse/dense correlation machinery.
- `geo_clusters.py` - DBSCAN on latitude/longitude with city fallback.
- `transfer_clusters.py` - weighted Louvain communities and anchor/dependent
  detection.
- `cluster_metrics.py` - cluster rollups and anchor-contagion uplift.

Current theory:

- Transaction-network dependence.
- Geographic shared-shock dependence.
- Community detection.
- Anchor person / dependent structure, where a key node can cause cluster
  cascade risk.

Important invariant:

- Clusters are just segment columns. Risk metrics must aggregate primitives and
  block-sum loss covariance, not average borrower ratios.

### 2.4 Copula and dependence modeling

Implemented modules:

- `copula_model.py` - dense copula model with Gaussian, Student-t, Clayton,
  Gumbel, and Frank style behavior.
- `factor_copula.py` - Vasicek single-factor copula, scalable to millions.
- `multi_factor_copula.py` - multi-factor copula for geo plus transfer factors.
- `flexible_probs.py` - regime-aware copula calibration from stress indicators.

Current theory:

- Default indicator as a latent variable below threshold.
- Gaussian copula for normal dependence.
- Student-t copula for symmetric tail dependence.
- Clayton copula for lower-tail default clustering.
- Vasicek/Basel style factor copula:
  `A_i = sqrt(rho_i) Y_f(i) + sqrt(1-rho_i) eps_i`.
- Multi-factor copula:
  `A_i = sum_k beta_i,k Y_k,f_k(i) + sqrt(1-sum beta^2) eps_i`.

Scale design:

- Dense paths are capped around 20k names.
- Factor and multi-factor paths avoid full `n x n` matrices and compute blocks
  on demand.

### 2.5 Loss covariance and risk metrics

Implemented modules:

- `risk_adjusted_metrics.py` - loss covariance, metric registry, CoV, RAROC,
  Sharpe, Sortino, diversification ratio, segment aggregation.
- `risk_metrics.py` - VaR, ES, portfolio loss simulation, stress testing,
  fraud rings, contagion.
- `client_value_metrics.py` - client-level value metrics, EAD/revenue proxies.
- `metric_comparison.py` - metric rank correlations and divergence flags.

Current theory:

- Basel expected loss:
  `EL_i = PD_i * EAD_i * LGD_i`.
- Bernoulli default covariance:
  `Cov(D_i,D_j) = P(D_i and D_j) - PD_i PD_j`.
- Loss covariance:
  `LossCov[i,j] = EAD_i LGD_i Cov(D_i,D_j) EAD_j LGD_j`.
- Segment variance:
  `Var(Loss_S) = sum_{i in S} sum_{j in S} LossCov[i,j]`.
- VaR and ES from simulated loss distribution.
- RAROC and Sortino divergence as an early warning for hidden correlation risk.

### 2.6 Ratings and transition risk

Implemented modules:

- `rating_engine.py` - PD to rating buckets, migration matrix, transition
  projection, correlated rating paths.
- `credit_transitions.py` - generator-based credit transition matrix estimator
  ported from ARPyM.
- `relative_entropy.py` - entropy pooling / minimum relative entropy engine used
  by transition monotonicity constraints.

Current theory:

- Continuous-time Markov generator:
  `P(delta_t) = exp(G * delta_t)`.
- Absorbing default state.
- Generator estimation from cohort/duration data.
- Entropy-regularized monotonicity correction for credit-transition rows.

### 2.7 Random-matrix denoising

Implemented module:

- `spectrum.py` - Marchenko-Pastur spectrum shrinkage and correlation denoising.

Current theory:

- Separate signal eigenvalues from noisy sample covariance/correlation bulk.
- Flatten noise eigenvalues to stabilize the copula correlation matrix.

### 2.8 Agent API and orchestration

Implemented modules:

- `agents.py` - `RiskAgentAPI` facade and JSON-safe result schema.
- `main.py` - 13-step end-to-end pipeline.
- `demo_clusters.py` - geo plus transfer cluster pipeline.
- `debug.py` - targeted diagnostics.

Main agent workflows:

- Run full pipeline.
- Run cluster analysis.
- Query borrower / segment / cluster.
- Find top risks.
- Find RAROC versus Sortino divergence.
- Run stress tests.
- Inspect anchors and fragile clusters.

## 3. ARPyM Subproject Inventory

Path:

`Python/arpym/arpym`

Observed module count:

- 64 Python files.

Package groups:

- `statistics/`
- `estimation/`
- `views/`
- `pricing/`
- `portfolio/`
- `tools/`

The `__init__.py` files are empty, so imports may require direct module paths
unless package exports are added.

### 3.1 Statistical tests and statistics

Present reusable modules include:

- `invariance_test_ellipsoid.py` - autocorrelation test with confidence bands
  and lag scatter/ellipse diagnostics.
- `invariance_test_ks.py` - two-sample Kolmogorov-Smirnov stability test on
  random partitions.
- `invariance_test_copula.py` - lag-copula invariance using Schweizer-Wolff
  dependence.
- `schweizer_wolff.py` - Schweizer-Wolff dependence measure on copula grades.
- `kalman_filter.py` - state-space filtering.
- `project_trans_matrix.py` - transition-matrix projection using generator
  fitting and matrix exponential.
- `simulate_markov_chain_multiv.py` - multivariate Markov-chain simulation.
- `simulate_mvou.py`, `moments_mvou.py` - multivariate Ornstein-Uhlenbeck
  simulation and moments.
- `simulate_quadn.py`, `saddle_point_quadn.py` - quadratic normal simulation
  and saddlepoint approximation.
- `twist_prob_mom_match.py`, `twist_scenarios_mom_match.py` - probability and
  scenario twisting to match target moments.
- `ffgn.py` - fractional Gaussian noise.

Integration value:

- These are strongest as validation and scenario-generation tools for the risk
  platform.
- The invariance tests should become a model-validation report for PD residuals,
  loss invariants, regime indicators, and factor residuals.

### 3.2 Estimation

Present reusable modules include:

- Flexible probabilities:
  `crisp_fp.py`, `conditional_fp.py`, `blow_spin_fp.py`,
  `high_breakdown_fp.py`.
- Linear factor models:
  `fit_lfm_ols.py`, `fit_lfm_mlfp.py`, `fit_lfm_roblasso.py`, `enet.py`.
- Robust location/dispersion:
  `fit_locdisp_mlfp.py`, `fit_locdisp_mlfp_difflength.py`.
- Dynamic dependence:
  `fit_dcc_t.py`.
- Time-series/state models:
  `fit_var1.py`, `var2mvou.py`, `fit_state_space.py`,
  `fit_stochastic_volatility_model.py`.
- Credit:
  `fit_trans_matrix_credit.py`.
- Random-matrix/shrinkage:
  `spectrum_shrink.py`, `low_rank_diag_conditional_corr.py`.
- Derivative-surface calibration:
  `fit_svi.py`.
- Cointegration:
  `cointegration_fp.py`.

Integration value:

- `fit_dcc_t.py` can make correlation/factor loadings time-varying.
- `fit_lfm_*` can estimate systematic factor loadings from observed loss,
  default, or macro histories.
- `fit_locdisp_mlfp.py` can provide robust Student-t estimates for stress
  regimes and heavy-tailed residuals.
- `spectrum_shrink.py` has already been ported as `src/spectrum.py`.
- `fit_trans_matrix_credit.py` has already been ported as
  `src/credit_transitions.py`.

### 3.3 Views and entropy pooling

Present reusable modules:

- `min_rel_entropy_sp.py`
- `min_rel_entropy_normal.py`
- `rel_entropy_normal.py`

Already integrated:

- `src/relative_entropy.py` is a self-contained port of
  `views.min_rel_entropy_sp`.

Integration value:

- Use entropy pooling to express macro, sector, geography, rating, or stress
  views as probability constraints on scenarios.
- Use it to create risk-department stress overlays without hard-coding PD
  multipliers.

### 3.4 Pricing

Present reusable modules:

- `bond_value.py`
- `bootstrap_nelson_siegel.py`
- `call_option_value.py`
- `fit_heston.py`
- `fit_nelson_siegel_bonds.py`
- `implvol_delta2m_moneyness.py`
- `zcb_value.py`

Integration value:

- Replace crude EAD/revenue proxies with product-level valuation for bonds,
  options, and rate-sensitive exposures.
- Add yield-curve scenarios and mark-to-market effects to stress loss.
- Use Heston/SVI/implied-vol tools if derivative or collateral valuation enters
  credit exposure.

### 3.5 Portfolio

Present reusable modules:

- `char_portfolio.py`
- `obj_tracking_err.py`
- `spectral_index.py`

Research scripts add many more portfolio topics:

- mean-variance frontiers,
- Bayesian allocation,
- Black-Litterman,
- risk attribution,
- Euler decomposition,
- economic capital,
- satisfaction/utility functions.

Integration value:

- Add a portfolio-decision layer on top of risk metrics:
  limit setting, risk-budget attribution, capital allocation, and acceptance
  policies.

### 3.6 Tools

Present reusable modules:

- `aggregate_rating_migrations.py`
- `backward_selection.py`
- `forward_selection.py`
- `naive_selection.py`
- `trade_quote_processing.py`
- `trade_quote_spreading.py`

Integration value:

- Feature selection can support PD/EAD/LGD modeling.
- Rating migration aggregation can support transition-matrix estimation.
- Trade/quote tools are less directly relevant unless the platform expands into
  traded-market risk.

## 4. Research Scripts Inventory

Path:

`Python/scripts/sources`

Observed count:

- 435 Python scripts.

Important themes found by filename:

- Copulas and marginal modeling:
  `s_cop_marg_separation.py`, `s_cop_marg_combination.py`,
  `s_cop_marg_stresstest.py`, `s_copula_opinion_pooling.py`,
  `s_copula_returns.py`, `s_display_norm_copula.py`,
  `s_display_t_copula.py`, `s_t_copula_norm_marginals.py`.
- Credit/default/rating:
  `s_default_merton_model.py`, `s_default_probabilities.py`,
  `s_rating_migrations.py`, `s_projection_univ_rating.py`,
  `s_projection_multiv_ratings.py`, `s_pricing_default_coupon_bond.py`.
- Statistical tests:
  many `s_elltest_*` and `s_kolmsmirn_*` scripts for invariance and
  Kolmogorov-Smirnov testing.
- Flexible probabilities and entropy:
  `s_exp_decay_fp.py`, `s_flex_prob_bootstrap.py`,
  `s_flex_prob_dirichlet.py`, `s_ensemble_flex_probs.py`,
  `s_min_entropy_fp.py`, `s_entropy_view.py`,
  `s_min_rel_ent_*`.
- Dynamic dependence:
  `s_dcc_fit.py`, `s_fit_garch_stocks.py`, `s_garch_likelihood.py`,
  `s_volatility_clustering_stock.py`.
- Shrinkage and random matrix theory:
  `s_marchenko_pastur.py`, `s_integral_marchenko_pastur.py`,
  `s_ledoit_wolf_covariance_shrinkage.py`, `s_shrink_cov_glasso.py`,
  `s_shrink_corr_clusters.py`, `s_shrink_spectrum_filt.py`,
  `s_lfm_rmt.py`.
- Markov/state models:
  `s_fit_discrete_markov_chain.py`, `s_markov_chain_monte_carlo.py`,
  `s_hidden_markov_model_stocks.py`, `s_project_trans_matrix.py`.
- Pricing and valuation:
  option pricing, Heston, SVI, yield curves, Nelson-Siegel, zero-coupon bonds,
  coupon bonds, equity/FX repricing.
- Portfolio/risk:
  risk attribution, economic capital, risk aggregation, utility/satisfaction,
  mean-variance, Black-Litterman, efficient frontier.

Integration rule:

- Treat these scripts as research references, not production modules.
- Promote only selected scripts into clean, tested, dependency-light modules.
- Remove plotting side effects from anything used inside the platform.

## 5. Data Inventory

### 5.1 Temporary databases

Path:

`databases/temporary-databases`

Observed count:

- 144 files.

Relevant datasets by name:

- `db_estimation_copula.csv`
- `db_estimation_credit_copula.csv`
- `db_copula_ratings.csv`
- `db_invariants_p_credit.csv`
- `db_credit_rd.csv`
- `db_riskdrivers_credit.csv`
- `db_projection_ratings.csv`
- `db_trans_matrix.csv`
- `db_sp500_garch_dcc_inv.csv`
- `db_GARCH_residuals.csv`
- `db_estimation_flexprob.csv`
- `db_scenario_probs.csv`
- `db_scenario_probs_bootstrap.csv`
- `db_stress_error.csv`
- `db_aggregation_regcred.csv`
- pricing, yield, option, stock, and portfolio datasets.

Integration value:

- Use as examples/test fixtures for credit copula, invariance, transition,
  scenario-probability, and stress workflows.

### 5.2 Legacy global databases

Path:

`old-databases/global-databases`

Observed count:

- 53 files.

Relevant datasets by name:

- `db_Ratings.mat`
- `db_GeneratorCredMatrix.mat`
- `db_ProjCreditTransitions.mat`
- `db_p_creditEP.mat`
- `db_CorporateBonds.mat`
- `db_HighYieldIndices.mat`
- `db_OneDayPL.mat`
- `db_PricingScenarioBased.mat`
- `db_NelsonSiegel_GE_JPM.mat`
- `db_IVsurf.mat`
- `db_ImpliedVol_SPX.mat`
- `db_SPX_zcb_Invariants.mat`
- `db_GarchParStocks.mat`
- `db_GarchParSP.mat`
- `db_Stocks.mat`, `db_StocksHighFreq.mat`, `db_FX.mat`, `db_VIX.mat`.

Integration value:

- Use to build realistic examples for transition matrices, credit exposures,
  default-coupon bond pricing, scenario-based valuation, and macro/market stress.

## 6. What Is Already Integrated from ARPM

The platform now contains **seven** self-contained ARPM ports (numpy / pandas /
scipy / scikit-learn only — no `arpym` or `statsmodels` runtime dependency):

1. `src/relative_entropy.py`
   - Port of `arpym.views.min_rel_entropy_sp`.
   - Used for entropy pooling and transition-matrix monotonicity.

2. `src/credit_transitions.py`
   - Port of `arpym.estimation.fit_trans_matrix_credit`.
   - Adds generator-based transition estimation, half-life weighting, and
     entropy-regularized monotonicity.
   - Wired into `RatingEngine.from_cohort_data()`.

3. `src/spectrum.py`
   - Port of `arpym.estimation.spectrum_shrink`.
   - Reimplements Marchenko-Pastur density without `skrmt`.
   - Wired into `TransactionGraph.get_correlation_matrix(denoise=True)`.

4. `src/conditional_fp.py`
   - Port of `arpym.estimation.conditional_fp` / `crisp_fp` (rigorous flexible
     probabilities: crisp window + entropy-pooling moment match).
   - Wired into `FlexibleProbsCalibrator(weighting_method="conditional_fp")`.

5. `src/low_rank_corr.py`
   - Port of `arpym.estimation.low_rank_diag_conditional_corr` / `conditional_pc`.
   - `fit_factor_loadings` turns a correlation matrix into `(n, k)` loadings for
     `MultiFactorCopula`. Exposed via `RiskAgentAPI.fit_factor_loadings`.

6. `src/dependence.py`
   - Port of `arpym.estimation.schweizer_wolff` + copula invariance test.
   - **Fixes a normalisation bug** in arpym's `schweizer_wolff` (which can return
     values > 1 on independent data).

7. `src/copula_calibration.py`
   - Empirical copula parameter calibration (Plan 07): Kendall τ / Schweizer-Wolff
     / default-correlation → Gaussian / Student-t / Clayton parameters, with a
     goodness-of-fit family recommendation. Exposed via
     `RiskAgentAPI.calibrate_copula_from_data`.

These ports are also surfaced through a modular **ETL layer** (`pipelines/`, one
ownable stage per concern communicating via an on-disk `ArtifactStore`) and a
matching per-stage notebook set (`notebooks/`); spectrum denoising and the
optional factor-loading stage (`run_all(with_loadings=True)`) run there.

The platform also independently implements ARPM-aligned ideas:

- flexible probabilities,
- regime-aware copula calibration,
- Merton structural PD,
- rating migration,
- factor copula / Vasicek logic,
- block loss covariance,
- risk attribution-style metric aggregation.

## 7. Highest-Value Integrations from ARPM Theory

### P0. Add a validation layer for PD, invariants, and copula assumptions

Add:

- `src/validation/invariance.py`
- `src/validation/backtesting.py`
- `src/validation/report.py`

Use:

- `invariance_test_ellipsoid`
- `invariance_test_ks`
- `invariance_test_copula`
- `schweizer_wolff`

Outputs:

- PD residual autocorrelation.
- Distribution stability across time windows or random partitions.
- Lag-copula dependence.
- Schweizer-Wolff dependence score by lag.
- Validation report per city, risk archetype, transfer cluster, and geo cluster.

Why:

- The copula platform assumes stable calibrated marginals and a credible
  dependence structure. These tests tell model validation where that assumption
  breaks.

### P0. Add PD calibration and backtesting

Add:

- calibration curves,
- Brier score,
- log loss,
- expected calibration error,
- KS/Gini/AUC,
- population stability index,
- out-of-time validation,
- segment-level calibration.

Use ARPM concepts:

- estimation assessment scripts,
- KS scripts,
- scenario-probability evaluation scripts.

Why:

- PD quality is upstream of every EL, VaR, ES, Sortino, RAROC, and transition
  metric.

### P0. Replace crude EAD and LGD with model components

Add:

- `src/exposure/ead_model.py`
- `src/exposure/lgd_model.py`
- per-borrower LGD array support everywhere.

Use ARPM concepts:

- credit aggregation scripts,
- pricing modules,
- scenario-based valuation datasets,
- default coupon bond pricing,
- product valuation logic.

Why:

- The current strongest model gap is not the copula. It is loss magnitude:
  EAD and LGD are too approximate for production-grade portfolio loss.

### P1. Build ARPM-style entropy stress views

Add:

- stress views expressed as constraints rather than hard-coded multipliers:
  "default rate in Almaty cluster rises to x",
  "loss in transfer cluster y exceeds threshold",
  "macro stress indicator equals current regime".

Use:

- `src/relative_entropy.py`
- `flexible_probs.py`
- ARPM entropy/min-relative-entropy scripts.

Why:

- This turns stress testing into a transparent scenario-view engine.

### P1. Estimate dynamic dependence with DCC-t or flexible probabilities

Add:

- dynamic factor loading calibration,
- time-varying correlation monitor,
- stress-regime loading override.

Use:

- `fit_dcc_t.py`
- `fit_locdisp_mlfp.py`
- `fit_lfm_ols.py`
- `fit_lfm_mlfp.py`
- `fit_lfm_roblasso.py`

Why:

- Static betas are good for a prototype. Banks need correlation to move by
  regime, time, segment, and stress.

### P1. Extend Markov chains to lifetime PD and IFRS9/CECL

Add:

- delinquency states:
  Current -> DPD30 -> DPD60 -> DPD90 -> Default.
- lifetime default probability.
- expected time to default.
- absorbing-chain analytics.
- ECL stage logic.

Use:

- `credit_transitions.py`
- `project_trans_matrix.py`
- rating migration scripts and datasets.

Why:

- Copula handles cross-sectional joint default.
- Markov chains handle temporal movement toward default.
- Together they produce multi-period correlated default simulation.

### P1. Add copula calibration from data

Add:

- estimate Gaussian/t/Clayton parameters from observed default, delinquency,
  or migration history.
- calibrate Kendall tau / Spearman / Schweizer-Wolff to copula parameters.
- tail-dependence diagnostics.
- goodness-of-fit comparisons across copula families.

Use:

- copula/marginal scripts,
- Schweizer-Wolff,
- invariance copula test,
- existing `compare_copulas`.

Why:

- Current copula parameters are mostly modeled/configured. Production risk
  needs an estimation and validation loop.

### P2. Add portfolio decision and risk-attribution layer

Add:

- risk-budget attribution by borrower, cluster, city, segment.
- capital allocation by Euler contribution.
- portfolio optimization subject to Sortino/RAROC/contagion constraints.
- policy engine for approve/review/reprice/reject.

Use:

- portfolio scripts,
- risk attribution scripts,
- characteristic portfolio,
- tracking error,
- Black-Litterman references.

Why:

- The platform already computes powerful metrics. A decision layer makes them
  operational.

### P2. Add market/pricing stress for collateral and traded exposures

Add:

- yield-curve repricing,
- option/derivative valuation,
- collateral haircut stress,
- mark-to-market linked to default stress.

Use:

- pricing modules,
- Nelson-Siegel,
- Heston,
- SVI,
- zero-coupon and bond pricing datasets.

Why:

- Useful if credit exposure depends on market value, collateral value, or
  derivative exposure.

## 8. Proposed Target Architecture

Recommended package split:

```text
risk_platform/
  contracts/
    person.py
    exposure.py
    graph.py
    scenarios.py
    results.py
  data/
    loaders.py
    arpm_datasets.py
    validation.py
  modeling/
    pd/
    lgd/
    ead/
    copula/
    transitions/
    structural/
    calibration/
  analytics/
    graph/
    geo/
    transfer/
    factors/
  scenarios/
    flexible_probabilities.py
    entropy_pooling.py
    stress_views.py
  validation/
    invariance.py
    calibration.py
    backtesting.py
    model_report.py
  riskmetrics/
    loss_covariance.py
    ratios.py
    attribution.py
    portfolio_policy.py
  orchestration/
    agents.py
    pipelines.py
```

Migration approach:

1. Keep current `src/` working.
2. Create new packages as thin wrappers.
3. Move one bounded context at a time.
4. Keep all existing tests green.
5. Add tests for every promoted ARPM module.

## 9. Suggested 30/60/90 Day Plan

### First 30 days

- Stabilize packaging: make `Python/arpym/arpym` importable or vendor selected
  functions into the platform.
- Add `validation/` with invariance ellipsoid, KS, copula invariance, and
  Schweizer-Wolff.
- Add PD calibration/backtesting report.
- Add ARPM dataset loader for selected credit/counterparty datasets.
- Add tests around validation outputs.

### Days 31-60

- Add LGD/EAD model interfaces.
- Add per-borrower LGD array support across risk metrics.
- Add entropy stress-view API.
- Add transition-matrix workflows from real/cohort data to lifetime PD.
- Add copula calibration diagnostics using observed defaults/migrations.

### Days 61-90

- Add DCC-t or flexible-probability dynamic factor calibration.
- Add portfolio risk attribution and policy decisions.
- Add production package split with contracts.
- Add generated model-validation report for risk committees.
- Add real-data example pipeline using ARPM credit and market datasets.

## 10. Verification Performed

Commands run from `projects/copula_default_graph`:

```bash
python test_copula_framework.py
MPLCONFIGDIR=/private/tmp/copula-mpl-cache python test_copula_framework.py
```

Result:

- First full run produced a transient `test_relative_entropy` assertion failure.
- The same test passed in isolation immediately afterward.
- Clean full rerun passed all 51 tests.
- Fontconfig warnings appeared because user font-cache directories were not
  writable in the sandbox; they did not prevent the clean test pass.

Current verified status:

- All 51 tests passed on the clean rerun.

## 11. Bottom Line

This workspace contains more than a toy project:

- The Copula Default Graph project is already a coherent credit-risk prototype
  with data ingestion, PD modeling, graph dependence, copulas, scalable factor
  copulas, cluster/anchor contagion, risk-adjusted metrics, stress testing,
  ratings, structural PD, flexible probabilities, customer reports, and an
  agent facade.
- The ARPyM subproject is a library and research corpus containing exactly the
  theory needed to professionalize it: invariance tests, flexible probabilities,
  entropy pooling, dynamic correlations, robust estimation, Markov transition
  models, random-matrix denoising, pricing, portfolio optimization, and risk
  attribution.
- The highest-value next move is not more charts. It is a model-validation and
  calibration layer: PD calibration, invariant testing, copula calibration, EAD
  and LGD modeling, and entropy-based stress views.

