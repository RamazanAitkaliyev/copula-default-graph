# ROADMAP — From research framework to a bank risk-department platform

> Iterative gap analysis + build-out plan. Grounded in the current codebase
> (`src/` 17 modules, ~11.9k LOC, 36 tests). Each item states **what exists
> today**, **what's missing**, **the design**, and **effort**.
>
> Read order for the impatient: §0 (target architecture) → §1 table (priorities)
> → the sections you care about.

---

## 0. Target architecture — Domain-Driven Design, 4 bounded contexts

Today everything lives in a flat `src/`. To let a risk department actually *work*
on this (separate teams, separate release cadence, separate compute), split into
four **bounded contexts** with explicit contracts between them. Each is a Python
package with its own tests; they talk through typed dataclasses (the "published
language"), never by reaching into each other's internals.

```
risk_platform/
  contracts/                 # PUBLISHED LANGUAGE — dataclasses every layer agrees on
    person.py                #   PersonRecord, FeatureVector
    exposure.py              #   Exposure(ead, lgd, pd, segment_id, factor_id)
    graph.py                 #   ClusterRef, EdgeRecord
    results.py               #   MetricResult, PortfolioResult  (already ~ AgentResult)

  etl/                       # CONTEXT 1 — INGESTION  (today: src/loaders.py is the seed)
    sources/                 #   pluggable readers: parquet, JDBC, feature store, Kafka
    schema.py                #   ColumnMapping + validation (exists, expand)
    feature_pipeline.py      #   300–800 feature assembly, point-in-time correctness
    materialize.py           #   reindex_to_contiguous, partitioning, caching

  analytics/                 # CONTEXT 2 — GRAPH & CLUSTERS  (today: src/graph_features.py)
    graph_build.py           #   sparse transaction graph (exists)
    geo_clusters.py          #   NEW — geolocation clustering (H3 / DBSCAN)
    transfer_clusters.py     #   NEW — community detection on money flow (Louvain/Leiden)
    cluster_rollup.py        #   NEW — per-cluster EAD/LGD/PD aggregates of any size
    centrality.py            #   PageRank, betweenness (exists)

  modeling/                  # CONTEXT 3 — MODELS  (today: pd_model, structural_pd, factor_copula)
    pd/                       #   IndividualPDModel → add XGBoost/CatBoost + GPU
    lgd/                      #   NEW — LGD model (today: flat 0.45 constant)
    ead/                      #   NEW — EAD model (today: income proxy)
    markov/                  #   NEW — default-state Markov chains (see §4)
    copula/                   #   FactorCopula + CopulaDefaultModel (exists)
    structural/               #   Merton (exists)

  riskmetrics/               # CONTEXT 4 — METRICS & PROPAGATION  (today: risk_adjusted_metrics)
    registry.py               #   @register_metric (exists) — CoV/RAROC/Sortino/Sharpe
    propagation.py            #   NEW — individual → cluster → portfolio roll-up engine
    combinations.py           #   NEW — metric algebra (weighted blends, dominance, veto)
    comparison.py             #   MetricComparator (exists)

  orchestration/
    pipelines.py              #   declarative DAG wiring the 4 contexts (replaces main.py)
    agents.py                 #   RiskAgentAPI (exists) — becomes the façade over pipelines
```

**Why this matters for a bank:** the ETL team can swap a data source without
touching copula math; the modeling team can ship a new CatBoost PD without
breaking metric definitions; the risk team composes metrics declaratively. The
`contracts/` package is the single source of truth — change it deliberately, and
everything else is free to evolve behind it.

**Migration is non-breaking:** keep `src/` working, introduce `risk_platform/`
as thin re-export wrappers first, move logic module-by-module, delete `src/`
shims last. Tests stay green throughout.

---

## 1. Priority matrix

| # | Capability | Exists today? | Value | Effort | Priority |
|---|------------|---------------|-------|--------|----------|
| 2 | XGBoost/CatBoost PD + GPU | sklearn only | ★★★★★ | M | **P0** |
| 3 | Geo + transfer clusters as first-class entities | partial (local clustering coef only) | ★★★★★ | M | **P0** |
| 4 | Markov default-state chains | only rating-migration generator | ★★★★☆ | M | **P1** |
| 5 | LGD & EAD models | flat LGD=0.45, income-proxy EAD | ★★★★★ | M | **P0** |
| 6 | Metric propagation & combination engine | per-level metrics exist, no propagation algebra | ★★★★☆ | M | **P1** |
| 7 | Gauss-Markov / martingale diagnostics | none | ★★★☆☆ | S | P2 |
| 8 | DDD package split (§0) | flat src/ | ★★★★☆ | L | **P1** |
| 9 | Calibration & backtesting layer | none (AUC only) | ★★★★★ | M | **P0** |
| 10 | Serving / API / scheduling | RiskAgentAPI in-process only | ★★★☆☆ | L | P2 |
| 11 | Data quality & PIT correctness gates | basic validation in loaders | ★★★★☆ | M | P1 |

Legend: effort S<½wk, M≈1–2wk, L≈3–6wk per item with one engineer.

---

## 2. PD models on GPU (XGBoost / CatBoost)  — P0

**Today:** `src/pd_model.py:IndividualPDModel` uses `LogisticRegression` or
`GradientBoostingClassifier` (sklearn, CPU, single-threaded for GBM). Fine for
1000 synthetic rows; will not train on 10M × 300–800 features in acceptable time.

**Design:**
- Add a `backend=` arg to `IndividualPDModel`: `'logistic' | 'sklearn_gbm' |
  'xgboost' | 'catboost'`. Keep the same `.fit/.predict_proba/.feature_importance_`
  interface so nothing downstream changes (Liskov-substitutable).
- **GPU:** XGBoost `tree_method='hist', device='cuda'`; CatBoost `task_type='GPU'`.
  CatBoost is the better default for this domain — native categorical handling
  (city_id, group_id, archetype) without one-hot, strong defaults, robust to
  the monotonic-ish credit features.
- **Calibration is mandatory** for PD (a classifier's score ≠ a probability).
  Wrap with isotonic or Platt (`CalibratedClassifierCV`), or use CatBoost's
  `posterior` + a separate isotonic step. A miscalibrated PD poisons every
  downstream EL/VaR/metric. (See §9.)
- **Monotonic constraints:** credit risk wants monotonicity (more
  missed_payments ⇒ higher PD). Both XGBoost and CatBoost support per-feature
  monotone constraints — set them; regulators love it and it improves
  out-of-time stability.
- **Feature pipeline:** 300–800 features means feature-group governance:
  bureau, behavioural, transaction-graph (neighbor_pd_*, centrality), geo. Keep
  graph-derived features as a named block so you can ablate "does the graph add
  lift?" (it should — that's the project's whole thesis; measure it).

**Output contract:** unchanged `model_pd ∈ [0,1]` column. The copula/metric
layers already consume that — drop-in.

**GPU server note:** training is the GPU job. *Scoring* 10M rows is a batched
CPU/GPU predict; *simulation* (FactorCopula) is separately the CPU/GPU job. Keep
them as distinct pipeline stages so you can put each on the right hardware.

---

## 3. Geolocation & transfer clusters as first-class entities — P0

**Today:** `graph_features.py` computes a *local* clustering coefficient and has
a `# Community detection` placeholder, plus same-city correlation boosts. There
is **no object** that represents "cluster #7" with its own membership, size, and
rolled-up risk. Clusters are the natural unit a risk department reasons about
("the Almaty merchant ring", "the salary-circle in district 3").

**Design — two independent clusterings, then a join:**

1. **Geo clusters** (`analytics/geo_clusters.py`):
   - If you have lat/lon: **H3 hex binning** (Uber's library) at a chosen
     resolution gives nestable, variable-size geo cells for free — and H3 lets
     you roll up coarse↔fine trivially (your "differently sized clusters" ask).
   - If you only have city_id/region: hierarchical region tree (city → district
     → block) — same roll-up semantics, coarser.
   - Fallback: **DBSCAN** on coordinates for organic, density-based clusters of
     arbitrary size (no fixed k), which is exactly "differently sized clusters".

2. **Transfer clusters** (`analytics/transfer_clusters.py`):
   - Community detection on the money-flow graph: **Leiden** (preferred over
     Louvain — guarantees well-connected communities, faster on 10M-node sparse
     graphs via `igraph`/`leidenalg`). Resolution parameter = cluster
     granularity knob ⇒ "differently sized transfer clusters".
   - These are the "who pays whom" connection clusters — fraud rings, salary
     circles, supply chains.

3. **Cluster rollup** (`analytics/cluster_rollup.py`) — the payoff:
   - For every cluster (geo OR transfer, any size) compute:
     `EAD_cluster = Σ ead_i`, `EL_cluster = Σ pd_i·ead_i·lgd_i`,
     `PD_cluster` (exposure-weighted), and critically
     **`Var(Loss_cluster) = ΣΣ loss_cov[i,j]`** over members — this already
     exists as `RiskRatioCalculator.by_segment` / block-sum loss-cov (INV-6).
     So clusters become just another `segment_id`, and *every existing metric
     works on them unchanged.* That's the elegant part: you don't rebuild the
     metric engine, you feed it cluster ids.
   - Assign each cluster a **systematic factor id** for the FactorCopula
     (`build_factor_id` already does multi-column → factor). Geo cluster and
     transfer cluster can be **two factors** per borrower → multi-factor copula
     (see §3a). This directly answers your "correlation from BOTH transaction
     graph AND geo, equally important."

**3a. Multi-factor extension of FactorCopula (small, high-value):**
Today `FactorCopula` is single-factor (`A = √ρ·Y_{f(i)} + √(1-ρ)·ε`). Generalize
to *k* factors:
`A_i = Σ_k β_{i,k}·Y_k + √(1 - Σβ²)·ε_i`, with e.g. `Y_geo` and `Y_transfer`.
Implied corr becomes `Σ_k β_{i,k}β_{j,k}`. Still O(n·k) storage — scales to 10M.
This is the principled way to make geo and transfer "equally important": give
them equal loadings. ~1 week including tests.

---

## 4. Markov chains for default-state transitions — P1

You asked for "probability of transferring between different states of
defaulting and risk." Two distinct, both-useful Markov layers:

**4a. Rating/delinquency Markov chain (the classic).**
- States: `Current → DPD30 → DPD60 → DPD90 → Default(absorbing)` (or rating
  buckets AAA…D). You already have the machinery: `rating_engine.py` builds a
  generator `G` and does `P(Δt) = expm(G·Δt)`. Promote it to a full
  `modeling/markov/` module that:
  - Estimates the transition matrix from *historical* state sequences (cohort
    or duration/Aalen-Johansen estimator), not just a hand-set generator.
  - Exposes **n-step default probability** `P(default within m months | state)` —
    this is the "probability of transferring into default" you want, and it's a
    *term-structure of PD*, richer than a single PD number.
  - Absorbing-chain analytics: expected time-to-default `(I-Q)⁻¹`, absorption
    probabilities — directly feeds EAD (balance at the time of default) and
    provisioning (IFRS9 lifetime PD = Markov lifetime default prob).

**4b. Behavioural-state Markov (segment dynamics).**
- States = risk archetypes (`prime…deep_subprime`) or cluster membership.
- Transition matrix on these gives **migration of the *portfolio*** through
  regimes — couples beautifully with the existing `flexible_probs.py` regime
  classifier (regime = which transition matrix is in force).

**Why Markov + copula together:** copula gives the *cross-sectional* dependence
(who defaults *with* whom this period); Markov gives the *temporal* dependence
(how a borrower *moves toward* default over periods). Combined → a
through-the-cycle, multi-period correlated-default simulator. That's a genuinely
strong risk engine and not common in mid-tier banks.

**Regulatory bonus:** IFRS9 / CECL **require** lifetime PD term structures.
A calibrated delinquency Markov chain *is* the lifetime PD curve. This makes the
platform provisioning-grade, not just monitoring.

---

## 5. LGD & EAD models — P0 (currently the weakest link)

**Today:** LGD is a **flat constant 0.45**; EAD is an **income proxy**
(`income×… / 0.08`). Every loss number (EL, VaR, ES, every metric denominator)
is therefore only as good as these two crude inputs — and they're the crudest
part of the whole framework.

- **LGD model** (`modeling/lgd/`): beta-regression or gradient-boosted
  regressor on recovery features (collateral type/value, seniority, product,
  region, time-to-resolution). LGD ∈ [0,1] ⇒ beta or `clip`. At minimum,
  **segment-level LGD** (by product × collateral × region) is a huge step up
  from one global number and is easy. Downturn-LGD variant for stress.
- **EAD model** (`modeling/ead/`): for revolving products EAD =
  `current_balance + CCF × undrawn_limit` (credit-conversion-factor model);
  CCF estimated per segment. For term loans EAD ≈ outstanding balance
  (optionally Markov-timed, §4a). Replace the income proxy.
- **Contract change:** `Exposure(ead, lgd, pd, …)` dataclass in `contracts/`.
  These flow into the *same* loss-cov machinery — `RiskRatioCalculator` already
  takes per-borrower `exposures` and an `lgd`; generalize `lgd` from scalar to
  per-borrower array (one-line change, big correctness gain).

**Expected-loss identity becomes per-borrower:** `EL_i = PD_i · EAD_i · LGD_i`
with all three modelled. This is the Basel "PD×EAD×LGD" decomposition done
properly — auditable, each component independently backtestable.

---

## 6. Metric propagation & combination engine — P1

You asked to "propagate and combine different metrics on individual and
portfolio levels." Today metrics are computed *per level independently*; there's
no engine that *propagates* an individual metric up or *combines* metrics.

**Design (`riskmetrics/propagation.py` + `combinations.py`):**

- **Propagation (bottom-up roll-up):** define for each metric *how* it
  aggregates. The key insight is already in the codebase (INV-6): **additive
  primitives roll up, ratios do not.** So the engine rolls up the *primitives*
  (EL, EP, Capital, Var(Loss) via block-sum) to each level (individual →
  cluster → segment → portfolio), then *forms the ratio at that level*. One
  generic roll-up, all ratio metrics for free. Levels = the cluster hierarchy
  from §3 (geo tree + transfer communities), so propagation is literally
  "sum loss-cov over this subtree."

- **Combination (metric algebra):** a small DSL to compose metrics:
  - **Weighted blend:** `0.6·sortino_copula + 0.4·raroc` (after rank-normalizing
    — they're on different scales; use percentile or z-score).
  - **Dominance/veto:** "approve only if RAROC>h **and** sortino_copula>0 **and**
    not in a flagged contagion cluster" — boolean gates over metrics.
  - **Divergence signals:** RAROC-vs-Sortino gap already flagged in
    `metric_comparison.py` — generalize to any metric pair as a contagion/early-
    warning feature.
  - Output: a `CompositeMetric` registered just like base metrics, so it's
    usable everywhere (`by_segment`, agent API, presentation).

This turns the metric layer from "a menu of numbers" into "a programmable risk
policy," which is exactly what a risk department operationalizes.

---

## 7. Gauss-Markov / martingale diagnostics — P2

You mentioned "Gauss-Markov martingales." Concretely useful instances:

- **Gauss-Markov (BLUE) for the factor/loading estimation:** when you regress
  asset-return proxies on systematic factors to *estimate* the copula loadings
  `β`, GLS/Gauss-Markov gives the minimum-variance unbiased loadings under the
  stated covariance — a principled replacement for the currently hand-set `ρ`.
- **Martingale backtest of PD:** under a correct PD model, the cumulative
  `(default_i − PD_i)` sequence is a martingale ⇒ its scaled partial sums should
  look like a driftless random walk. A **martingale-residual** / CUSUM test is a
  clean, bank-friendly *calibration drift* monitor (ties into §9).
- **Martingale property of provisions:** discounted expected lifetime loss
  should be a martingale across reporting dates under a consistent model — a
  coherence check on the Markov lifetime-PD (§4a).

Small, high-signal, mostly diagnostics — schedule after the P0/P1 core.

---

## 8. Calibration & backtesting layer — P0 (non-negotiable for a bank)

Today: validation = a single **AUC** (discrimination). A risk department needs
**calibration** (are the probabilities right?) and **backtesting** (do they hold
out of time?). Without this, none of the above is trustworthy.

- **Calibration:** reliability diagrams, Brier score, Hosmer-Lemeshow,
  **Spiegelhalter Z**, expected-vs-observed by decile. Isotonic recalibration
  step in the PD pipeline (§2).
- **Discrimination (keep + extend):** AUC, **KS**, Gini, **CAP curve**.
- **Stability:** **PSI/CSI** (population & characteristic stability index)
  month-over-month — the standard "has the world drifted?" monitor.
- **Backtesting losses:** VaR/ES **Kupiec POF** + **Christoffersen** independence
  tests on realized vs predicted portfolio loss; traffic-light test.
- **Out-of-time / out-of-sample split** baked into the ETL (PIT correctness, §11).

Make this a `validation/` module that emits a one-page **model-monitoring
report** (the artifact regulators and model-risk committees ask for). The
existing presentation generator is the natural rendering target.

---

## 9. Serving, scheduling, lineage — P2 (productionization)

- **API:** wrap `RiskAgentAPI` in FastAPI; batch-score endpoint + on-demand
  borrower/cluster query. (Today it's in-process only.)
- **Scheduling:** nightly ETL → score → simulate → metrics → report DAG
  (Airflow/Prefect, or the repo's existing `/schedule` tooling for lighter use).
- **Lineage & reproducibility:** model registry (MLflow), data versioning,
  config-hash on every run so any number is reproducible — essential for audit.
- **Performance:** the 10M simulation is the cost center. Options, in order:
  (a) the analytical FactorCopula loss moments (no simulation needed for
  EL/Var — closed form via the block machinery), (b) GPU the Bernoulli draws
  (CuPy), (c) importance sampling for the tail (ES) instead of brute Monte-Carlo.

---

## 10. Cross-cutting correctness & data quality — P1

- **Point-in-time correctness** in ETL: no feature may use information from
  after the label date (the #1 silent killer of credit models). Enforce with a
  PIT join contract in `etl/feature_pipeline.py`.
- **Schema contracts** (`contracts/`): validate at every boundary (pandera /
  pydantic) so a bad upstream column fails loudly at ingestion, not deep in the
  copula.
- **Reconciliation:** Σ cluster EAD == portfolio EAD; Σ segment EL == portfolio
  EL — assert these as invariants (extends the existing INV-1..7 set).
- **Determinism:** seed every RNG; the test suite already pins seeds — extend to
  pipelines.

---

## 11. Suggested execution order (iterative, each ships value)

**Sprint 1 (foundations, P0):**
1. `contracts/` package (Exposure/PersonRecord/MetricResult) — unlocks everything.
2. Per-borrower LGD/EAD plumbing (scalar→array) + segment-LGD model. (§5)
3. CatBoost/XGBoost backend + calibration in `IndividualPDModel`. (§2, §8)

**Sprint 2 (the thesis — clusters & dependence, P0/P1):**
4. Geo (H3/DBSCAN) + transfer (Leiden) clustering → `cluster_id` columns. (§3)
5. Cluster rollups via existing `by_segment` block-sum; multi-factor copula. (§3, §3a)
6. Measure graph-feature lift on PD (ablation) — proves the project's value. (§2)

**Sprint 3 (dynamics & policy, P1):**
7. Delinquency Markov chain → lifetime PD term structure. (§4)
8. Metric propagation + combination engine. (§6)
9. Backtesting/monitoring report module. (§8)

**Sprint 4 (DDD split + productionize, P1/P2):**
10. Migrate `src/` → `risk_platform/` four contexts behind re-export shims. (§0)
11. FastAPI serving + scheduled DAG + lineage. (§9)
12. Gauss-Markov loading estimation + martingale drift monitors. (§7)

Each sprint keeps `test_copula_framework.py` green and adds tests for new code.

---

## 12. What is already strong (don't rebuild)

- **Scalable dependence core**: FactorCopula is O(n), proven to 2M (0.4s fit,
  64MB). The hard scalability problem is *solved*.
- **Correct aggregation**: block-sum loss-cov (INV-6) means clusters/segments/
  portfolio all reuse one correct engine — this is the load-bearing abstraction
  that makes §3 and §6 cheap.
- **Metric registry**: `@register_metric` + `by_segment` is already the right
  plug-in shape for §6.
- **Agent façade**: `RiskAgentAPI` is the right seam for §9 serving.
- **Two analytical CDFs** (Gaussian + Student-t bivariate) verified to ~1e-12 /
  ~3e-4 — reuse for any tail work.

The bones are good. The roadmap is mostly *adding the bank-grade flesh*
(real LGD/EAD, calibration, clusters-as-entities, Markov term structure, GPU
training) and *organizing* it (DDD) so a department can operate it.
