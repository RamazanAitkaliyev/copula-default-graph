# AGENTS.md — Copula Default Graph
*Read this file first. It is the authoritative contract between AI agents and this codebase.*

---

## What this system does (one paragraph)

A **bank credit-risk analytical pipeline** that models correlated defaults across a portfolio
of borrowers. It turns raw transaction data into a correlation matrix (via a graph), combines
that with individual PDs (you supply them, or train a GBM) through a Clayton copula to produce
joint default probabilities, then computes seven risk-adjusted metrics (CoV, RAROC, Sharpe,
Sortino variants) at any aggregation level — borrower, segment, city, group, or full portfolio.
The RAROC vs Sortino divergence signal is the primary early-warning output.

---

## Plugging in real data (start here for production)

The framework already has your PD model's role covered: **you supply `model_pd`** (or `base_pd`),
the framework supplies the *dependency* layer (graph + copula + metrics). Use `src/loaders.py`.

```python
from src.loaders import (ColumnMapping, load_persons, load_transactions,
                         reindex_to_contiguous)

# 1) Map YOUR column names to canonical ones:
pmap = ColumnMapping(person_id="acct_no", model_pd="pd_12m",
                     city_id="region", exposure_at_default="credit_limit",
                     high_risk_group_id="household_id")
persons = load_persons("clients.parquet", mapping=pmap,
                       duplicate_policy="first", pd_nan_policy="median")

tmap = ColumnMapping(sender_id="payer", receiver_id="payee", amount="rub")
tx = load_transactions("transfers.parquet", mapping=tmap,
                       valid_person_ids=persons["person_id"].tolist())

# 2) If your ids are account numbers (not 0..n-1), remap (REQUIRED):
persons, tx, id_map = reindex_to_contiguous(persons, tx)   # originals kept in 'original_person_id'

# 3) Now build the graph / copula / metrics as usual.
```

The loaders handle: foreign column names, PD stored as percentages (auto /100),
NaNs (policy: error/drop/zero/median), duplicate ids (error/first/last),
transactions referencing unknown persons (dropped + logged), non-contiguous ids.
Every problem becomes a loud `DataValidationError` or a logged warning — never silent corruption.

## Scale guarantees (10M+ borrowers)

This codebase was refactored to NOT build dense n×n matrices, which at 10M would be ~0.8 PB.

| Component | Scale behaviour |
|---|---|
| `TransactionGraph` | Sparse `scipy.sparse` CSR adjacency, O(n_edges). Vectorized build (no `iterrows`). |
| `get_correlation_matrix()` | Dense — **raises MemoryError above `DENSE_MATRIX_MAX_NODES` (20k)**. |
| `get_correlation_sparse()` | Sparse correlation — off-diagonal entries only where a link/city/group exists. Use at scale. |
| `RiskRatioCalculator` | Builds full dense `loss_cov` only for n ≤ `LOSS_COV_DENSE_MAX_NODES` (20k); **block-on-demand** above. |
| `loss_cov` property | Raises MemoryError above threshold — use `by_segment()` / `metric()` which compute blocks. |
| `per_borrower()` | **Fully vectorized**, O(n). Runs on 500k borrowers in ~0.02 s. |
| `_count_components()` | `scipy.sparse.csgraph` (iterative) — no recursion-limit crash. |
| `joint_default_probability_block(idx)` | Computes only the m×m segment block, never the full n×n. Vectorized for Clayton. |

**Correctness guarantee:** the block-on-demand path is mathematically identical to the dense
path (verified to 1e-9 in test 32). Scaling does not change the numbers.

**Still O(segment²):** a single segment's block is m×m dense. Keep segments / groups reasonably
sized. `get_correlation_sparse(max_block_size=...)` skips over-large same-city/group blocks.

**Environment:** designed for one large server (64–512 GB RAM) with Parquet inputs.

### The copula at 10M: `FactorCopula` (src/factor_copula.py)

`CopulaDefaultModel('clayton')` needs a dense n×n correlation and is capped at ~20k names.
For the full portfolio, use **`FactorCopula`** — a Vasicek single-factor model where dependence
comes from systematic factors (geography, household/group, industry), stored as O(n) loadings,
never an n×n matrix.

```python
from src.factor_copula import FactorCopula, build_factor_id

# One systematic factor per borrower (group wins over city; else city):
factor_id = build_factor_id(persons, ("high_risk_group_id", "city_id"))

fc = FactorCopula().fit(persons["model_pd"].values, factor_id, rho=0.15)
# rho = asset correlation / factor loading (Basel retail ~0.03–0.16).

# Drops straight into the metrics pipeline (block mode):
calc = RiskRatioCalculator(fc, persons, exposures=ead, lgd=0.45)
calc.by_segment("city_id")              # same API, scales to 10M

# Streamed simulation — never stores (n_sim, n):
losses = fc.simulate_segment_losses(members, el_vec, n_simulations=10_000)
rate   = fc.simulate_default_rate(n_simulations=1000)   # whole-portfolio rate
```

| Property | `CopulaDefaultModel('clayton')` | `FactorCopula` |
|---|---|---|
| Max n | ~20k (dense n×n) | 10M+ (O(n) loadings) |
| Dependence source | full pairwise (graph + city + group) | systematic factors |
| Tail dependence | yes (lower tail) | no (Gaussian) / yes if `student_t=True` |
| `joint_default_probability_block` | ✅ vectorized | ✅ bivariate-normal CDF (1e-8 accurate) |
| Drop-in for RiskRatioCalculator | ✅ | ✅ |

**Key knob — factor granularity controls correlation strength:** fewer, larger factors ⇒
stronger clustering ⇒ larger portfolio-loss variance. With 1000 factors over 2M borrowers the
factor model inflates portfolio default-rate variance ~11× vs independence (verified at scale).

**Tail dependence at scale:** pass `FactorCopula(student_t=True, nu=6)` for a t-factor copula
(defaults cluster harder in stress) — still O(n) per simulation via a shared chi-square draw.

**`simulate_defaults(n_sim)` returns a dense (n_sim, n) matrix** — fine for n up to ~100k, but
at 10M use the streamed `simulate_segment_losses` / `simulate_default_rate` instead.

---

## Multi-dimensional clusters & anchors (geo + transfer graph)

Correlation comes from TWO independent, equally-weighted sources, each a
systematic factor:

- **Geo clusters** — `geo_clusters.GeoClusterer` (DBSCAN on `geo_longitude`/
  `geo_latitude`, falls back to `city_id`) → `geo_cluster_id`.
- **Transfer communities** — `transfer_clusters.TransferClusterer` (weighted
  Louvain on the money-flow graph) → `transfer_cluster_id`, plus **anchor /
  dependent** detection (якорный человек): `is_anchor`, `depends_on_anchor`,
  `cluster_fragility`.

Feed BOTH into the **multi-factor copula** so geo and transfer drive default
correlation together:

```python
from src.multi_factor_copula import MultiFactorCopula
factors = persons[["geo_cluster_id", "transfer_cluster_id"]].to_numpy()
mfc = MultiFactorCopula().fit(persons["model_pd"].values, factors, betas=[0.3, 0.3])
# equal betas ⇒ equally important. implied corr = Σ_k β_ik·β_jk over shared factors.
# O(n·K) storage (never n×n) → scales to 10M. Drop-in for RiskRatioCalculator.
```

**Cluster metrics** are free: `RiskRatioCalculator.by_segment("geo_cluster_id")`
or `"transfer_cluster_id"` (block-sum loss covariance). The headline
anchor-contagion number — how much a cluster's expected loss rises if its anchor
defaults — is `cluster_metrics.ClusterRiskMetrics(calc, persons)
.anchor_contagion_table()` (conditional PD = P(D_j ∩ D_anchor)/PD_anchor).

Agent façade: `api.run_cluster_analysis()` then `api.geo_clusters()`,
`api.transfer_clusters()`, `api.anchors()`, `api.fragile_clusters()`,
`api.cluster_report(id)`. Runnable end-to-end: `python demo_clusters.py`.
See `CAPABILITIES.md` (catalog) and `RECIPES.md` (snippets).

**Invariant:** `Σ_k β² < 1` per borrower (positive idiosyncratic variance) —
`MultiFactorCopula.fit` raises if violated.

---

## Agent entry points

There are **three ways to interact** with this codebase, ordered from safest to most powerful:

### 1. High-level agent API (recommended for most tasks)
```python
from src.agents import RiskAgentAPI
api = RiskAgentAPI()               # uses default config, generates data automatically
api.run_pipeline()                 # executes all 13 steps
result = api.query_borrower(776)   # get full profile dict
result = api.query_segment("city_name", "Gamma")
result = api.flag_divergences()    # get RAROC vs Sortino flags
result = api.run_stress(pd_multiplier=2.0)
result = api.segment_metrics("risk_archetype")
```

### 2. Module-level classes (for custom pipelines)
```python
from src import CopulaDefaultModel, RiskRatioCalculator, MetricComparator
# See per-module docstrings for exact signatures.
# Always call fit() before any query method.
# Always validate copula.is_fitted == True before building RiskRatioCalculator.
```

### 3. main.py pipeline (for full end-to-end run)
```bash
python main.py          # writes all outputs to output/
python debug.py smoke   # fast smoke test, no plots
python test_copula_framework.py  # 30 unit tests
```

---

## Data contract

### Input: `persons` DataFrame
Every public method that accepts `persons` expects these columns to exist:

| Column | Type | Required | Notes |
|---|---|---|---|
| `person_id` | int | ✅ | Unique. All numpy positional ops index on this. |
| `city_id` | int | ✅ | 0-indexed city assignment. |
| `city_name` | str | ✅ | Human-readable city label. |
| `risk_archetype` | str | ✅ | One of: `low`, `medium`, `high`, `prime`, `near_prime`, `subprime`, `deep_subprime`, `bridge`. |
| `base_pd` | float | ✅ | Ground-truth PD ∈ [0,1]. Simulation label. |
| `model_pd` | float | ✅ (after step 3) | ML-predicted PD. Fed to copula. |
| `income` | float | ✅ | Used for EAD proxy if `exposure_at_default` absent. |
| `default` | int | ✅ (for training) | 0/1 binary default label. |
| `exposure_at_default` | float | preferred | Transaction-based EAD. Falls back to `income*3`. |
| `estimated_revenue` | float | preferred | Transaction-based revenue. Falls back to `EAD*0.02`. |
| `high_risk_group_id` | int | optional | -1 = not in a group. |
| `is_bridge` | bool | optional | Bridge node flag. |

**AGENT RULE:** Never assume `model_pd` exists before step 3 runs. Check `'model_pd' in persons.columns`.

### Input: `transactions` DataFrame

| Column | Type | Required |
|---|---|---|
| `sender_id` | int | ✅ |
| `receiver_id` | int | ✅ |
| `amount` | float | ✅ |

### Output: what each step produces

| Step | New column(s) on persons | Other outputs |
|---|---|---|
| 2 (graph) | `neighbor_pd_avg`, `neighbor_pd_max`, `n_high_risk_neighbors` | `TransactionGraph` object |
| 3 (PD model) | `model_pd` | `IndividualPDModel`, AUC metrics |
| 8 (client value) | `exposure_at_default`, `estimated_revenue` | `ClientValueCalculator` |
| 10 (Merton) | `merton_pd`, `blended_pd`, `distance_to_default`, `pd_signal_divergence` | `StructuralPDModel` |

---

## Invariants — never violate these

```
INV-1  copula.marginal_pds ∈ [0,1] at all times.
INV-2  copula.correlation_matrix must be PSD with diag=1.
        Always call _nearest_psd() after any modification.
INV-3  persons['person_id'] must be unique integers.
        All numpy positional indexing relies on this.
INV-4  risk_tier dtype must be object (not fixed-width str).
        Use np.full(n, 'low', dtype=object) — never np.array([...]) for string arrays.
INV-5  copula state is restored by _stressed_copula() context manager.
        Never mutate copula.marginal_pds or copula.correlation_matrix directly in stress paths.
INV-6  Segment metrics must use block-sum of loss_cov, never average per-borrower ratios.
        Var(Loss_S) = sum(loss_cov[S,S]) — this is the ONLY correct aggregation under correlation.
INV-7  RiskRatioCalculator requires copula.is_fitted == True.
        Raise ValueError if not; do not silently fall back.
```

---

## Common mistakes agents make (and how to avoid them)

### ❌ Wrong: computing segment metric as mean of per-borrower metrics
```python
# WRONG — loses all correlation structure
df.groupby('city_name')['sortino_copula'].mean()
```
```python
# CORRECT — mathematically exact under copula correlation
calc.by_segment('city_name')
```

### ❌ Wrong: using EAD = income/mean(income)
```python
# WRONG — produces normalised fractions (0.1–5), not dollar amounts
# revenue = EAD * 0.02 → ≈ 0.002–0.1, far below expected_loss → 66% negative profit
exposures = persons['income'].values / persons['income'].mean()
```
```python
# CORRECT — use ClientValueCalculator which produces realistic EAD
client_calc = ClientValueCalculator(copula, persons, transactions)
ead = client_calc.persons['exposure_at_default'].values  # range: 820–79815
```

### ❌ Wrong: building RiskRatioCalculator with unfitted copula
```python
calc = RiskRatioCalculator(copula, persons)  # raises ValueError if not fitted
```
```python
# CORRECT
assert copula.is_fitted, "Fit copula before building RiskRatioCalculator"
```

### ❌ Wrong: modifying copula state in stress test
```python
copula.marginal_pds *= 2.0  # corrupts global state
```
```python
# CORRECT — context manager restores original state
with analyzer._stressed_copula(pd_multiplier=2.0):
    result = analyzer.compute_portfolio_risks()
```

### ❌ Wrong: sorting divergence flags by person_id instead of z_score
```python
flags = comp.divergence_flags()
flags.sort_values('person_id')  # meaningless
flags.sort_values('z_score', key=abs, ascending=False)  # correct
```

### ❌ Wrong: interpreting negative Sortino as "bad" unconditionally
Sortino/RAROC/Sharpe sign flips when `E[Profit] < hurdle*Capital`.
Always check `numerator_negative` column first. Use `coefficient_of_variation`
for riskiness ranking when profit is negative.

---

## Pipeline step dependency graph

```
generate_network()
      │
      ├─→ TransactionGraph          [step 2]
      │         │
      │         └─→ get_correlation_matrix()
      │                   │
      ├─→ IndividualPDModel.fit()   [step 3] → persons['model_pd']
      │         │
      │         └─→ CopulaDefaultModel.fit(model_pd, corr_matrix)  [step 5]
      │                   │
      │         ┌─────────┴─────────────────┐
      │         ↓                           ↓
      │   RiskAnalyzer              ClientValueCalculator      [step 6, 8]
      │         │                           │
      │         │                   persons['exposure_at_default']
      │         │                   persons['estimated_revenue']
      │         │                           │
      │         └──────────────────→ RiskRatioCalculator      [step 8b]
      │                                     │
      │                             MetricComparator
      │
      ├─→ RatingEngine.fit()                [step 9]
      ├─→ StructuralPDModel.fit_transform() [step 10] → persons['merton_pd']
      ├─→ FlexibleProbsCalibrator           [step 11]
      └─→ CustomerProfiler.fit()            [step 12]
```

**Key dependency**: `RiskRatioCalculator` must come after both `CopulaDefaultModel.fit()`
and `ClientValueCalculator` (to get realistic EAD/revenue). See STEP 8b in `main.py`.

---

## Metric semantics — what each number means

| Metric | Formula | Range | Null when | Business meaning |
|---|---|---|---|---|
| `coefficient_of_variation` | σ_L0 / E[Loss] | [0, ∞) | E[Loss]=0 | Pure riskiness. Always positive. Safe for ranking. |
| `coefficient_of_variation_copula` | σ_L1 / E[Loss] | [0, ∞) | E[Loss]=0 | Copula-aware riskiness. Higher than L0 for correlated clusters. |
| `raroc` | E[Profit] / Capital | (−∞, ∞) | Capital=0 | Profitability per unit capital. Correlation-blind. |
| `sharpe_indep` | (E[Profit] − rf·Revenue) / σ_L0 | (−∞, ∞) | σ_L0=0 | Risk-free opportunity cost benchmark. |
| `sortino_indep` | (E[Profit] − h·Capital) / σ_L0 | (−∞, ∞) | σ_L0=0 | Hurdle-rate benchmark, independence assumption. |
| `sortino_copula` | (E[Profit] − h·Capital) / σ_L1 | (−∞, ∞) | σ_L1=0 | **Primary metric.** Copula-aware. Diverges from RAROC for contagious clusters. |
| `sortino_simulated` | (E[Profit] − h·Capital) / σ_L2 | (−∞, ∞) | sim not run | Full Monte Carlo tail. Expensive. Run with `with_sim=True`. |
| `diversification_ratio` | Σσ_i / σ_portfolio | [1, ∞) | — | 1=perfectly correlated; higher=more diversification. |

**Divergence flag logic:**
```
RAROC rank ≫ Sortino rank  →  "hidden_network_risk"    (looks fine individually, dangerous in network)
RAROC rank ≪ Sortino rank  →  "diversified_low_value"  (unprofitable but good for portfolio)
```
Flag threshold: |z_score| ≥ 1.5 (standard deviations of the rank gap distribution).

---

## Configuration knobs

All tunable parameters live in `src/config.py`. Override any field:

```python
from src.config import PipelineConfig, RiskConfig

cfg = PipelineConfig(
    risk=RiskConfig(
        hurdle_rate=0.12,          # increase required return threshold
        capital_ratio=0.10,        # Basel III advanced vs standard
        lgd=0.35,                  # lower LGD (more secured lending)
        metric_sim_paths=50_000,   # higher accuracy for sortino_simulated
    )
)
```

| Parameter | Default | Effect |
|---|---|---|
| `hurdle_rate` | 0.10 | Sortino/RAROC numerator threshold. Higher = fewer borrowers above hurdle. |
| `risk_free_rate` | 0.02 | Sharpe numerator benchmark. |
| `capital_ratio` | 0.08 | Regulatory capital per unit EAD (8% = Basel standard). |
| `lgd` | 0.45 | Loss given default. Higher = larger expected losses. |
| `metric_sim_paths` | 10_000 | Monte Carlo paths for `sortino_simulated`. 50k for production. |
| `copula_type` | `'clayton'` | `'gaussian'` has no tail dep; `'clayton'` has lower-tail dep (crisis clustering). |

---

## File map (what to touch for each task)

| Task | File(s) |
|---|---|
| Add a new risk-adjusted metric | `src/risk_adjusted_metrics.py` — decorate with `@register_metric("name")` |
| Add a new copula type | `src/copula_model.py:CopulaDefaultModel.SUPPORTED_COPULAS` + add `_<type>_copula()` |
| Add a new borrower feature | `src/data_generator.py:_generate_persons()` + add to PD model feature list |
| Change PD thresholds (ratings) | `src/rating_engine.py:PD_THRESHOLDS` |
| Change stress-test scenario | `src/risk_metrics.py:RiskAnalyzer._stressed_copula()` |
| Change Merton proxy calibration | `src/structural_pd.py:_proxy_asset_vol()` |
| Change regime classification | `src/flexible_probs.py:classify_regime()` |
| Change portfolio metrics (VaR/ES) | `src/risk_metrics.py # SECTION: PORTFOLIO METRICS` |
| Add a new output to the pipeline | `main.py` — add after the relevant `# STEP N` block |
| Add a customer profile field | `src/customer_profile.py:CustomerRiskProfile` dataclass + `_build_profile()` |
| Use the agent-facing API | `src/agents.py:RiskAgentAPI` |
| Change agent API response schema | `src/agents.py:_Result` dataclass |

---

## Testing and validation

```bash
# Must pass before any merge:
python test_copula_framework.py        # 30 unit tests

# Quick smoke (no plots, ~5s):
python debug.py smoke

# Specific test isolation:
python -c "from test_copula_framework import test_sortino_metrics; test_sortino_metrics()"

# Full pipeline with output:
python main.py                         # ~30-60s, writes to output/

# Regenerate presentation:
python generate_presentation.py        # reads output/, writes output/presentation.html
```

**Test coverage map:**
- Tests 1–10: data generation, graph, copula fitting, simulation
- Tests 11–20: risk metrics, stress test, rating engine, Merton, flexible probs
- Tests 21–23: customer profiles, client value, watchlist
- Tests 24–30: risk-adjusted metric family (CoV, RAROC, Sortino), additivity, aggregation, comparison

---

## Return value conventions

All public methods return one of:
1. **`pd.DataFrame`** — for multi-row results (by_segment, per_borrower, divergence_flags)
2. **`dict`** — for single-entity results (all_metrics, stress test dict)
3. **`float`** — for scalar results (diversification_ratio, single metric)
4. **`str`** — for narrative text (profile_report)
5. **`RiskAgentAPI` result objects** — structured dataclass with `.data`, `.summary`, `.warnings`

All metrics return `np.nan` (not 0, not None) on divide-by-zero. Check with `np.isfinite()`.
All DataFrames include a `numerator_negative` bool column when computing signed metrics.

---

## What agents should NOT do

```
✗  Do not call copula.fit() a second time — it overwrites params in-place.
✗  Do not mutate persons DataFrame rows directly — always work on copies.
✗  Do not sum per-borrower metric values to aggregate — use by_segment().
✗  Do not assume model_pd exists before step 3 — always check.
✗  Do not use loss_std_copula for a single borrower as contagion signal
   — L0 == L1 for n=1; diversification only visible at n≥2.
✗  Do not interpret np.nan metric as 0 — it means denominator was zero (safe division guard).
✗  Do not set copula_type='gaussian' for stress scenarios — it has no tail dependence.
✗  Do not call per_borrower() in a tight loop — it is O(n²) via loss_cov indexing.
   Cache the result from MetricComparator.borrower_table() instead.
```
