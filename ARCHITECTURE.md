# ARCHITECTURE — Copula Default Graph

How the platform is built, why, and where each piece lives. Pairs with
`ROLES.md` (who owns what), `METHODOLOGY.md` (the math), `CAPABILITIES.md`
(the API catalog), and `RECIPES.md` (snippets).

---

## 1. The problem in one paragraph

A bank has millions of borrowers, each with an individual probability of default
(PD). Treating those defaults as **independent** badly understates portfolio risk:
in reality borrowers are **correlated** — they share a local economy (geography),
they move money between each other (transaction graph), and whole clusters can
collapse together when a key person defaults (the *anchor*). This platform turns
raw data (persons + money transfers + geography) into **correlated** joint-default
risk, then computes risk metrics at every level — borrower, cluster, portfolio —
so a risk department can see and act on the hidden correlation.

---

## 2. Layered architecture

Data flows top → bottom. Each layer has a single responsibility and a typed
contract with the next. Roles (see ROLES.md) own one or two adjacent layers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ L0  INGESTION                                            [Data Engineer]  │
│     loaders.py · data_generator.py · config.py                            │
│     raw CSV/parquet ──► validated persons + transactions DataFrames       │
│     contract: person_id unique 0..n-1 · model_pd∈[0,1] · (lon,lat)? · tx  │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│ L1  PD MODELS                                              [ML Engineer]  │
│     pd_model.py · structural_pd.py                                        │
│     features ──► persons['model_pd'] ∈ [0,1]   (or supplied by the bank)  │
│     contract: one calibrated PD per borrower                              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│ L2  DEPENDENCE STRUCTURE                              [Data Scientist]    │
│     graph_features.py · geo_clusters.py · transfer_clusters.py            │
│     ─ transaction graph ──► centrality, communities (Louvain)            │
│     ─ geography ──► geo clusters (DBSCAN)                                 │
│     ─ money flow ──► transfer clusters + ANCHOR/dependent pattern        │
│     contract: geo_cluster_id, transfer_cluster_id, anchor columns        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│ L3  COPULA (joint defaults)                          [Data Scientist]    │
│     copula_model.py · factor_copula.py · multi_factor_copula.py          │
│     PDs + correlation ──► P(D_i ∩ D_j)   (joint default probabilities)    │
│     · CopulaDefaultModel  — 5 types, dense, ≤20k names                    │
│     · FactorCopula        — Vasicek 1-factor, 10M+                        │
│     · MultiFactorCopula   — geo ⟂ transfer, equal loadings, 10M+         │
│     contract: joint_default_probability_block(idx)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  RISK METRICS                                        [Risk Analyst]    │
│     risk_adjusted_metrics.py · cluster_metrics.py · risk_metrics.py       │
│     · loss-covariance matrix  LossCov[i,j] = (EAD·LGD)_i·Cov(D_i,D_j)·(…)_j│
│     · 7 metrics: CoV, RAROC, Sharpe, Sortino (indep/copula/sim)           │
│     · per borrower / cluster / segment / portfolio (block-sum, INV-6)     │
│     · anchor-contagion uplift · VaR/ES · stress · ratings · profiles      │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│ L5  ORCHESTRATION & SERVING                       [Platform / Agents]     │
│     agents.py (RiskAgentAPI) · main.py · demo_clusters.py                 │
│     state-machine façade · JSON-safe results · pipelines                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The load-bearing design decisions

**D1. Correlation from a sparse graph + low-rank factors, never a dense matrix.**
A dense n×n correlation/loss matrix at 10M borrowers is ~0.8 PB — physically
impossible. Instead:
- the transaction graph is `scipy.sparse` CSR (O(edges));
- dependence is expressed as **systematic factors** (geo cluster, transfer
  community), stored as O(n·K) loadings in the (multi-)factor copula;
- the loss-covariance matrix is **block-on-demand**: each segment's m×m block is
  computed only when needed, never the full n×n.
This is the single decision that makes 10M tractable. (Proven: 2M borrowers fit
in 0.30 s / 96 MB.)

**D2. Clusters are just a segment column → metrics for free.**
Geo and transfer clusters are written as `persons` columns. Every risk metric
already aggregates by an arbitrary column via `RiskRatioCalculator.by_segment`,
whose variance is the **block-sum of the loss-covariance matrix** (INV-6). So
adding a new clustering needs no new metric code — feed the column.

**D3. Additivity invariant (INV-6) is the correctness backbone.**
`E[Loss]`, `E[Profit]`, `Capital` are additive across borrowers. `Var(Loss_S)`
for a segment S is `ΣΣ LossCov[i,j]` over S — **never** an average of per-borrower
ratios (which is wrong under correlation). Every roll-up obeys this. The
block-on-demand path is verified mathematically identical to the dense path
(test 32, to 1e-9).

**D4. Two copula tracks, one metric interface.**
`CopulaDefaultModel` (dense, tail-dependent, ≤20k) and the factor copulas
(O(n), 10M) both expose `joint_default_probability_block(idx)`. The metric layer
duck-types on that, so either track plugs in unchanged.

**D5. The PD model is pluggable and often external.**
Most banks already have a PD model. The platform's value is the *dependency*
layer; it consumes `model_pd` and is agnostic to how it was produced (own GBM, or
the bank's XGBoost/CatBoost — see ROADMAP §2).

**D6. Anchors model temporal/structural contagion the copula alone misses.**
The copula captures cross-sectional co-movement. The anchor/dependent detector
adds the *structural* "if the breadwinner defaults, the family follows" pattern,
quantified as conditional-loss uplift `P(D_j∩D_anchor)/PD_anchor`.

---

## 4. Module responsibility map

| Layer | Module | Responsibility |
|---|---|---|
| L0 | `loaders.py` | column mapping, validation, PD %→fraction, reindex |
| L0 | `data_generator.py` | synthetic persons + transactions |
| L0 | `config.py` | all tunable dataclass configs |
| L1 | `pd_model.py` | `IndividualPDModel`, ensemble, importance |
| L1 | `structural_pd.py` | Merton structural PD (2nd signal) |
| L2 | `graph_features.py` | sparse graph, centrality, correlation, communities |
| L2 | `geo_clusters.py` | DBSCAN geo clusters |
| L2 | `transfer_clusters.py` | Louvain communities + anchor detection |
| L3 | `copula_model.py` | 5 copulas, dense |
| L3 | `factor_copula.py` | Vasicek single-factor, 10M |
| L3 | `multi_factor_copula.py` | K-factor (geo ⟂ transfer), 10M |
| L3 | `flexible_probs.py` | regime-aware copula reweighting |
| L4 | `risk_adjusted_metrics.py` | loss-cov + 7 metrics at any level |
| L4 | `cluster_metrics.py` | per-cluster roll-ups + anchor contagion |
| L4 | `risk_metrics.py` | VaR/ES, stress, fraud rings, cascades |
| L4 | `client_value_metrics.py` | client Sharpe/RAROC, EAD/revenue proxies |
| L4 | `metric_comparison.py` | rank-corr, RAROC-vs-Sortino divergence |
| L4 | `rating_engine.py` | PD→rating + migration matrix |
| L4 | `customer_profile.py` | per-borrower report + watchlist |
| L5 | `agents.py` | `RiskAgentAPI` façade |
| L5 | `main.py` | 13-step base pipeline |
| L5 | `demo_clusters.py` | geo+transfer cluster pipeline |

---

## 5. Data contracts (the boundaries)

**persons** (one row per borrower): `person_id` (unique int, 0..n-1), a PD column
`model_pd`|`base_pd` ∈ [0,1], optional `geo_longitude`/`geo_latitude`, `city_id`,
`exposure_at_default`, `estimated_revenue`, `default` (0/1, to train PD). After
L2: `geo_cluster_id`, `transfer_cluster_id`, `is_anchor`, `depends_on_anchor`,
`cluster_fragility`.

**transactions**: `sender_id`, `receiver_id`, `amount` (sender/receiver reference
person_id).

**copula** (object): `marginal_pds`, `is_fitted`,
`joint_default_probability_block(idx)` (or legacy no-arg full matrix).

**metric result**: `RiskRatioCalculator.by_segment(col)` →
DataFrame(segment, n, exposure, expected_loss, loss_std_copula,
coefficient_of_variation, raroc, sortino_copula, diversification_ratio, …).

---

## 6. Scale & performance posture

- Target: **one large server, 64–512 GB RAM**, Parquet inputs.
- Dense paths are **guarded**: `get_correlation_matrix` and the `loss_cov`
  property raise `MemoryError` above ~20k nodes — forcing the scalable path.
- Factor copulas: O(n·K) storage, streamed simulation (`simulate_default_rate`,
  `simulate_segment_losses`) never materialize (n_sim × n) at 10M.
- The cost center at 10M is simulation; analytical loss moments (mean/variance)
  need no simulation (closed form via the block machinery). See ROADMAP §9 for
  GPU/importance-sampling options.

---

## 7. Testing & correctness

`test_copula_framework.py` (41 tests) is the safety net. It includes independent
numerical references — not just "does it run":
- loss-covariance vs brute-force simulation (≈0 relative error);
- bivariate normal/t CDFs vs scipy (1e-12 / 3e-4);
- Fréchet bounds on joint probabilities; diversification ratio ≥ 1;
- block-on-demand == dense (1e-9);
- multi-factor implied correlation exact; anchor pattern on star-vs-mesh;
- agent methods return JSON-safe results.

Run: `python test_copula_framework.py` → `All 41 tests passed.`

---

## 8. Where to start reading (by role)

- **Everyone:** this file, then `ROLES.md`.
- **Data Engineer:** `loaders.py`, `tutorials/01_data_engineer.md`.
- **ML Engineer:** `pd_model.py`, `tutorials/02_ml_engineer.md`.
- **Data Scientist:** `graph_features.py` → `transfer_clusters.py` →
  `multi_factor_copula.py`, `tutorials/03_data_scientist.md`.
- **Risk Analyst:** `risk_adjusted_metrics.py` → `cluster_metrics.py`,
  `tutorials/04_risk_analyst.md`.
- **Quant / validation:** `METHODOLOGY.md` + the test suite.
- **AI agents:** `AGENTS.md` + `CAPABILITIES.md`.
