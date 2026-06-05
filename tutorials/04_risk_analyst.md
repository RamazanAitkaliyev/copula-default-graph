# Tutorial 04 — Risk Analyst

**Your job:** turn the fitted copula into the numbers the bank acts on — expected
loss, risk-adjusted metrics at every level, cluster contagion, stress tests,
ratings, watchlists.

**Your modules:** `src/risk_adjusted_metrics.py`, `src/cluster_metrics.py`,
`src/risk_metrics.py`, `src/metric_comparison.py`, `src/rating_engine.py`
(import via `from src.risk import ...`).

**The math:** `METHODOLOGY.md §5–9` (loss covariance, the 7 metrics,
diversification, anchor contagion, VaR/ES, early warning).

---

## Step 1 — build the calculator

```python
from src.risk import RiskRatioCalculator
ead = persons["exposure_at_default"].to_numpy()   # or income proxy
calc = RiskRatioCalculator(mfc, persons, exposures=ead, lgd=0.45)
#                          ^ the fitted copula from the Data Scientist
```
`lgd` can be a scalar or a per-borrower vector. The calculator builds the
loss-covariance object (dense for ≤20k, block-on-demand above).

## Step 2 — metrics at every level

```python
# per borrower
pb = calc.per_borrower()

# per ANY segment (block-sum loss covariance — correct under correlation)
geo_seg = calc.by_segment("geo_cluster_id")        # per geo cluster
calc.by_segment("transfer_cluster_id")             # per transfer community
calc.by_segment("city_id")                         # per city
# The grouping key is returned in a column literally named "segment"
# (NOT the input column name). So read it back as:
#     geo_seg["segment"]            # the geo_cluster_id values
# columns: segment, n, exposure, exposure_share, expected_profit, expected_loss,
#          loss_std_indep, loss_std_copula, diversification_ratio,
#          coefficient_of_variation, raroc, sortino_copula (primary), ...

# whole portfolio
print(calc.metric("sortino_copula"))            # one number, all borrowers
print(calc.diversification_ratio())             # >= 1
```

> ⚠️ **INV-6:** always use `by_segment(col)`. NEVER `pb.groupby(col).mean()` — a
> weighted average of per-borrower ratios is *wrong* under correlation. The
> segment variance must be the block-sum of the loss-covariance matrix.

## Step 3 — anchor contagion (the якорный-человек headline)

```python
from src.risk import ClusterRiskMetrics
crm = ClusterRiskMetrics(calc, persons)
contagion = crm.anchor_contagion_table()     # ranked by uplift_ratio
worst = contagion.iloc[0]
print(f"If anchor {int(worst['anchor_person_id'])} defaults, cluster "
      f"{int(worst['transfer_cluster_id'])} loss rises "
      f"{worst['uplift_ratio']:.2f}x "
      f"({worst['el_unconditional']:.0f} -> {worst['el_anchor_default']:.0f})")
```
`uplift_ratio ≈ 1` = diversified cluster; `≫ 1` = fragile, cascade risk.

## Step 4 — portfolio risk, VaR/ES, stress

```python
from src.risk import RiskAnalyzer
analyzer = RiskAnalyzer(copula, graph, persons, exposures=ead, lgd=0.45)
port = analyzer.compute_portfolio_risks(n_simulations=10000)
print(port.expected_loss, port.var_95, port.es_95)         # ES >= VaR >= ~EL
stress = analyzer.stress_test(pd_multiplier=2.0, correlation_boost=0.2)
print(stress["change"]["expected_loss"])                    # % EL increase
```

## Step 5 — the early-warning signal (RAROC vs Sortino divergence)

```python
from src.risk import MetricComparator
comp = MetricComparator(calc)
flags = comp.divergence_flags(z_threshold=1.5)
# Borrowers cheap on RAROC (correlation-blind) but expensive on sortino_copula
# (correlation-aware) are carrying HIDDEN contagion risk → investigate.
```

## Step 6 — ratings & watchlist

```python
from src.risk import RatingEngine, CustomerProfiler
ratings = RatingEngine().fit(persons, pd_col="model_pd")
# PD -> AAA..Default + 1-year migration outlook
profiler = CustomerProfiler(copula, graph, persons, ...)
watch = profiler.watchlist()    # critical + high tier borrowers
```

---

## How to read the metrics
- **`coefficient_of_variation_copula`** > **`coefficient_of_variation`**: the gap
  is the correlation penalty for that segment.
- **`sortino_copula`** is the primary risk-adjusted performance number (profit
  over hurdle, divided by the copula-aware downside).
- **`diversification_ratio`** near 1 means a concentrated, undiversified book.
- Ratio metrics with a profit numerator flip sign when profit < 0 — for pure
  riskiness ranking use the CoV metrics (always ≥ 0).

## Fastest path (agent façade does steps 1–5)
```python
from src.agents import RiskAgentAPI
api = RiskAgentAPI(persons=persons, transactions=transactions)
api.run_pipeline(); api.run_cluster_analysis()
print(api.fragile_clusters().summary)        # anchor-contagion ranking
print(api.flag_divergences().summary)        # early warning
```
