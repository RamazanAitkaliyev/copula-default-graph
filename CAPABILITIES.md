# CAPABILITIES — machine-readable capability catalog

A structured index of what this framework can do, for AI agents to plan with.
Each capability lists: **entry point**, **inputs**, **outputs**, **when to use**.
For runnable snippets see `RECIPES.md`; for the safety contract see `AGENTS.md`.

Preferred entry point for agents: `src/agents.py:RiskAgentAPI` (every method
returns `AgentResult(ok, data, summary, warnings, error)` with JSON-safe data).

---

## Domain map (flat modules grouped by concern)

| Domain | Modules | Purpose |
|---|---|---|
| **io / ingestion** | `loaders.py` | Read persons/transactions, map columns, validate, reindex |
| **modeling: PD** | `pd_model.py`, `structural_pd.py` | Individual PD (ML) + Merton structural PD |
| **analytics: graph** | `graph_features.py`, `transfer_clusters.py` | Transaction graph, centrality, communities, anchors |
| **analytics: geo** | `geo_clusters.py` | Geolocation clustering (DBSCAN) |
| **dependence: copula** | `copula_model.py`, `factor_copula.py`, `multi_factor_copula.py` | Joint defaults via copulas; factor models for scale |
| **metrics** | `risk_metrics.py`, `risk_adjusted_metrics.py`, `client_value_metrics.py`, `cluster_metrics.py`, `metric_comparison.py` | Portfolio/segment/cluster/borrower risk metrics |
| **analytics: extensions** | `rating_engine.py`, `flexible_probs.py`, `customer_profile.py` | Ratings + migration, regime-aware copula, per-borrower reports |
| **orchestration** | `agents.py`, `main.py`, `demo_clusters.py` | Agent façade + pipelines |

---

## Capabilities

### C1. Ingest real data
- **entry**: `loaders.load_persons(src, mapping)`, `loaders.load_transactions(src, mapping)`, `loaders.ColumnMapping`
- **inputs**: CSV/parquet/DataFrame; a `ColumnMapping` from your column names → canonical (person_id, model_pd, geo_longitude, geo_latitude, city_id, exposure_at_default, estimated_revenue, sender_id, receiver_id, amount)
- **outputs**: validated, canonical DataFrames; PD auto-normalized (%→fraction)
- **when**: first step with any non-synthetic data
- **notes**: `validate_persons` enforces PD∈[0,1], geo ranges; `reindex_to_contiguous` makes person_id 0..n-1 (required for positional indexing)

### C2. Train / score individual PD
- **entry**: `pd_model.IndividualPDModel.fit/predict_proba`; agent: `RiskAgentAPI.run_pipeline()`
- **inputs**: persons with features + a `default` 0/1 label (to train)
- **outputs**: `model_pd ∈ [0,1]` per borrower
- **when**: you need PDs and don't already have them; if you HAVE PDs, set `model_pd` and skip
- **notes**: backend is sklearn (logistic / gradient boosting) today; ROADMAP §2 adds XGBoost/CatBoost + GPU

### C3. Geolocation clusters
- **entry**: `geo_clusters.GeoClusterer(GeoClusterConfig(eps_km, min_samples)).fit(persons).assign(persons)`; agent: part of `run_cluster_analysis()`
- **inputs**: persons with `geo_longitude`, `geo_latitude` (else falls back to `city_id`)
- **outputs**: `geo_cluster_id` (>=0 genuine cluster; <0 isolated/independent); `summary()` per-cluster table (centroid, span_km)
- **when**: correlation has a geographic component (shared local economy/shocks)
- **notes**: DBSCAN, arbitrary-size clusters, no fixed k; `eps_km` is the radius knob

### C4. Transfer communities (money-flow graph)
- **entry**: `transfer_clusters.TransferClusterer(TransferClusterConfig(resolution, min_cluster_size)).fit(persons, transactions).assign(persons)`; agent: part of `run_cluster_analysis()`
- **inputs**: persons + transactions (sender_id, receiver_id, amount)
- **outputs**: `transfer_cluster_id`; `summary()` (internal/external weight, conductance); `subgraph(cid)` for one community
- **when**: correlation has a money-flow component (families, salary circles, rings)
- **notes**: weighted Louvain; `resolution` controls granularity

### C5. Anchor / dependent detection (якорный человек)
- **entry**: same `TransferClusterer` — columns after `assign`; `anchors_table()`
- **inputs**: as C4
- **outputs**: `is_anchor`, `anchor_of_cluster`, `depends_on_anchor`, `anchor_score`, `cluster_fragility`; agent: `RiskAgentAPI.anchors()`
- **when**: find clusters that would cascade if one key person defaults
- **notes**: anchor score = money-source dominance + articulation-point + star-shape

### C6. Correlated defaults at scale (multi-factor copula)
- **entry**: `multi_factor_copula.MultiFactorCopula().fit(pds, factor_matrix, betas)`; single-factor: `factor_copula.FactorCopula`; legacy small-n: `copula_model.CopulaDefaultModel`
- **inputs**: PDs (n,), `factor_matrix` (n, K) of cluster ids per dimension (e.g. [geo, transfer]), `betas` loadings (equal → "equally important")
- **outputs**: `simulate_defaults`, `simulate_default_rate`, `joint_default_probability_block(idx)`; implied corr = Σ_k β_ik·β_jk over shared factors
- **when**: geo AND transfer (or more) should both drive default correlation; n up to 10M+
- **notes**: O(n·K) storage, never n×n; Σβ²<1 enforced; drop-in for RiskRatioCalculator

### C7. Risk metrics at any level (single person → cluster → portfolio)
- **entry**: `risk_adjusted_metrics.RiskRatioCalculator(copula, persons, exposures, lgd)` → `per_borrower()`, `by_segment(col)`, portfolio aggregate
- **inputs**: a fitted copula (any of C6), persons, EAD, LGD (scalar or per-borrower array)
- **outputs**: coefficient_of_variation, coefficient_of_variation_copula, raroc, sharpe_indep, sortino_indep, sortino_copula (primary), sortino_simulated, diversification_ratio
- **when**: you need risk-adjusted performance per borrower/cluster/segment/portfolio
- **notes**: segment metrics use block-sum loss covariance (INV-6), never an average of per-borrower ratios; clusters are just a segment column

### C8. Anchor-contagion uplift (cluster cascade risk)
- **entry**: `cluster_metrics.ClusterRiskMetrics(calc, persons).anchor_contagion_table()`; agent: `RiskAgentAPI.fragile_clusters(top_n)`
- **inputs**: a RiskRatioCalculator over a (multi-)factor copula + persons with anchor columns
- **outputs**: per anchored cluster: `el_unconditional`, `el_anchor_default`, `uplift_ratio`, `uplift_abs`
- **when**: quantify "if this anchor defaults, how much does the cluster's loss rise?"
- **notes**: conditional PD = P(D_j ∩ D_anchor)/PD_anchor read off the copula block

### C9. Metric agreement / divergence (early warning)
- **entry**: `metric_comparison.MetricComparator(calc)`; agent: `RiskAgentAPI.flag_divergences()`, `rank_metrics()`
- **outputs**: rank correlations, disagreements, RAROC-vs-Sortino divergence flags
- **when**: find borrowers where a risk-blind metric (RAROC) and a contagion-aware one (Sortino-copula) disagree → hidden correlated risk

### C10. Portfolio risk, stress, regimes, ratings, profiles (existing)
- **entry**: `risk_metrics.RiskAnalyzer` (VaR/ES/stress), `rating_engine.RatingEngine` (PD→rating + migration), `flexible_probs.FlexibleProbsCalibrator` (regime-aware), `customer_profile.CustomerProfiler`; agent: `run_stress()`, `regime_status()`, `portfolio_summary()`, `query_borrower()`, `top_risks()`
- **when**: standard portfolio analytics, stress testing, watchlists

---

## Agent method index (RiskAgentAPI)

| Method | State required | Returns (data) |
|---|---|---|
| `run_pipeline()` | empty+ | full pipeline summary |
| `run_cluster_analysis(...)` | pd_model | n_geo/n_transfer clusters, n_anchors, variance inflation, top_anchor |
| `geo_clusters()` / `transfer_clusters()` | after cluster analysis | per-cluster metric rows |
| `anchors()` | after cluster analysis | anchor list (id, dependents, fragility) |
| `fragile_clusters(top_n)` | after cluster analysis | clusters ranked by anchor-contagion uplift |
| `cluster_report(id, dimension)` | after cluster analysis | one cluster's members + contagion |
| `query_borrower(id)` | pipeline | per-borrower profile |
| `segment_metrics(col)` / `query_segment(col, val)` | pipeline | segment metrics |
| `flag_divergences(z)` / `rank_metrics(level)` | pipeline | divergence flags / metric ranking |
| `run_stress(...)` / `regime_status()` / `portfolio_summary()` | pipeline | stress / regime / portfolio |
| `top_risks(n)` / `available_metrics()` / `state()` / `persons()` | varies | top risks / metric list / state / data |

---

## Invariants an agent must respect (see AGENTS.md for full list)
- person_id unique integers, reindexed 0..n-1 for positional indexing
- marginal PDs ∈ [0,1]; Σβ² < 1 for the multi-factor copula
- segment/cluster metrics = block-sum of loss covariance, never an average of ratios
- never build a dense n×n correlation/loss matrix above ~20k nodes — use factor copulas + block-on-demand
