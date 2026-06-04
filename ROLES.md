# ROLES — who owns what

This platform is built so a bank risk department can split ownership across
teams. Each module below is tagged with the **primary role** that maintains it
and the **consumers** that depend on its output. Find your role, work in your
modules, and rely on the documented contracts at the boundaries.

> Layer order (data flows top → bottom):
> **Data Engineer → ML Engineer → Data Scientist → Risk Analyst → (Quant Analyst cross-cuts)**

---

## Role → modules map

### 🛠️ Data Engineer — ingestion, validation, data contracts
*Owns getting clean, correctly-shaped data into the platform.*

| Module | Responsibility |
|---|---|
| `src/loaders.py` | `ColumnMapping`, `load_persons`/`load_transactions`, validation, PD %→fraction, `reindex_to_contiguous`. The single entry for real data. |
| `src/data_generator.py` | Synthetic `generate_network()` (persons + transactions) for tests/demos. |
| `src/config.py` | All dataclass configs (Network/Copula/Risk/Pipeline). Tunable knobs live here. |

**Contract you provide downstream:** a `persons` DataFrame with unique
contiguous `person_id` (0..n-1), a PD column (`model_pd`/`base_pd`) in [0,1],
optional `geo_longitude`/`geo_latitude`, `exposure_at_default`, and a
`transactions` frame (sender_id, receiver_id, amount). See `validate_persons`/
`validate_transactions` — every violation is a loud `DataValidationError`.

---

### 🤖 ML Engineer — probability-of-default models
*Owns the borrower-level PD: training, scoring, calibration, explainability.*

| Module | Responsibility |
|---|---|
| `src/pd_model.py` | `IndividualPDModel` (logistic / gradient boosting), `PDModelEnsemble`, feature importance, optimal threshold. |
| `src/structural_pd.py` | `StructuralPDModel` — Merton structural PD as an independent second signal. |

**Contract you provide downstream:** `persons['model_pd'] ∈ [0,1]`, one row per
borrower. If the bank already has a PD model, the Data Engineer supplies
`model_pd` directly and this layer is bypassed. (Roadmap: XGBoost/CatBoost + GPU
backend — see `ROADMAP.md §2`.)

---

### 📊 Data Scientist — dependence structure (graphs, clusters, copulas)
*Owns how borrowers are CORRELATED: the transaction graph, geo/transfer clusters,
the anchor pattern, and the copula that turns PDs + correlation into joint
defaults.*

| Module | Responsibility |
|---|---|
| `src/graph_features.py` | `TransactionGraph`: sparse money-flow graph, centrality, correlation matrix, communities. |
| `src/geo_clusters.py` | `GeoClusterer`: DBSCAN on lat/lon → `geo_cluster_id`. |
| `src/transfer_clusters.py` | `TransferClusterer`: Louvain communities + **anchor/dependent** detection (якорный человек) + `cluster_fragility`. |
| `src/copula_model.py` | `CopulaDefaultModel` (5 copula types: Gaussian/t/Clayton/Gumbel/Frank) for ≤20k names. |
| `src/factor_copula.py` | `FactorCopula` (Vasicek single-factor, scales to 10M+). |
| `src/multi_factor_copula.py` | `MultiFactorCopula` (geo ⟂ transfer, equally weighted, O(n·K), 10M+). |
| `src/flexible_probs.py` | `FlexibleProbsCalibrator` — regime-aware copula reweighting. |

**Contract you provide downstream:** a fitted copula object exposing
`marginal_pds`, `is_fitted`, and `joint_default_probability_block(idx)` (or the
legacy full-matrix form). This is the input every risk metric needs.

---

### 🎯 Risk Analyst — risk metrics, portfolio, stress, ratings, monitoring
*Owns the numbers the bank acts on: expected loss, VaR/ES, risk-adjusted metrics
at every level, cluster contagion, stress tests, ratings, watchlists.*

| Module | Responsibility |
|---|---|
| `src/risk_adjusted_metrics.py` | `RiskRatioCalculator`: CoV/RAROC/Sharpe/Sortino family at borrower/segment/cluster/portfolio (block-sum loss cov). |
| `src/cluster_metrics.py` | `ClusterRiskMetrics`: per-cluster roll-ups + **anchor-contagion uplift**. |
| `src/risk_metrics.py` | `RiskAnalyzer` (VaR/ES/stress), `PortfolioRiskMetrics`, `FraudRingDetector`, `ContagionStressTester`. |
| `src/client_value_metrics.py` | `ClientValueCalculator` (Sharpe, RAROC, client segments, EAD/revenue proxies). |
| `src/metric_comparison.py` | `MetricComparator`: rank correlations + RAROC-vs-Sortino divergence early warning. |
| `src/rating_engine.py` | `RatingEngine`: PD → AAA…Default + migration matrix. |
| `src/customer_profile.py` | `CustomerProfiler`: per-borrower risk report + watchlist. |

**Contract you consume:** a fitted copula + `persons` + EAD/LGD. Everything rolls
up correctly because segment/cluster variance is the block-sum of the
loss-covariance matrix (never an average of per-borrower ratios — INV-6).

---

### 🧮 Quant Analyst (cross-cutting) — methodology & model validation
*Owns the mathematical/financial correctness across all layers. Not a single
module — reviews the formulas in `copula_model`, `factor_copula`,
`multi_factor_copula`, `risk_adjusted_metrics`, `structural_pd`, `rating_engine`.*

**Reference:** `METHODOLOGY.md` (every formula with derivation, assumptions, and
financial interpretation) and the 41-test suite (independent numerical checks:
loss variance vs simulation, Fréchet bounds, calibration, etc.).

---

### 🧭 Orchestration / Platform — pipelines & agent façade
*Wires the layers together; the entry point for everyone else and for AI agents.*

| Module | Responsibility |
|---|---|
| `src/agents.py` | `RiskAgentAPI` — safe, state-machine façade; cluster + anchor methods; JSON-safe results. |
| `main.py` | The base 13-step end-to-end pipeline. |
| `demo_clusters.py` | The geo+transfer cluster + anchor end-to-end pipeline (saves artifacts). |
| `src/__init__.py` | Public API surface — re-exports every public symbol. |

---

## How a new team member onboards

1. **Read** `ARCHITECTURE.md` (the big picture) and this file (your modules).
2. **Run** `python test_copula_framework.py` (should print "All 41 tests passed.")
   and `python demo_clusters.py` (see the cluster pipeline produce artifacts).
3. **Open** `tutorials/` for your role's step-by-step walkthrough.
4. **For the math**, read `METHODOLOGY.md`. **For the API catalog**, read
   `CAPABILITIES.md`. **For copy-paste snippets**, read `RECIPES.md`.
5. **Stay inside your contract** — touch your modules; consume/produce the
   documented DataFrame columns and copula interface at the boundaries.

## Boundaries that must not be crossed silently
- Data Engineer guarantees `person_id` is unique and contiguous; everyone else
  relies on positional indexing.
- ML guarantees `model_pd ∈ [0,1]`.
- Data Scientist guarantees the copula is fitted and exposes the block interface;
  for the multi-factor copula, `Σ_k β² < 1` per borrower.
- Risk Analyst computes segment/cluster metrics from block-sums of loss
  covariance, never from averaging per-borrower ratios.
- Nobody builds a dense n×n matrix above ~20k nodes — use factor copulas +
  block-on-demand.
