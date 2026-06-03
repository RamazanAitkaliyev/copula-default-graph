# Implementation Plan — Pluggable Risk-Adjusted Metric Family

**Audience:** the implementing agent (Sonnet 4.6). Read `CLAUDE.md` first for the file map,
data schema, and invariants. This plan is self-contained; follow it top to bottom.

**Goal.** Add a *registry of risk-adjusted metrics* (Coefficient of Variation, RAROC, Sharpe,
Sortino, and copula-driven Sortino at three correlation levels) computed from one shared set of
**absolute, additive primitives**, plus a **comparison harness** that runs every metric on the
same population and reports where they agree/disagree. The point is to let the user *test which
metric is most representative*, and to compute any metric at **borrower / segment / geo / group /
portfolio** level from the same engine so results aggregate correctly under correlation.

**Non-goals.** Do NOT modify the existing `client_value_metrics.py` formulas, the copula, the PD
model, or `data_generator.py`. The new module sits alongside them. The existing 23 tests must
still pass unchanged.

---

## 0. Background the implementer must internalize

Every metric here has the shape **reward per unit of "bad" variability** — the Sharpe/Sortino/CoV
family. They differ only in (a) what goes in the denominator and (b) whether the denominator sees
network correlation. We build them from shared primitives so they are mutually comparable:

```
E[Loss_i]        = EAD_i · LGD_i · PD_i                         (absolute, additive)
E[Revenue_i]     = pluggable; fallback = fee proxy (see §2.3)
E[Profit_i]      = E[Revenue_i] − E[Loss_i]                     (additive)
Capital_i        = pluggable; fallback = k · EAD_i  (k≈0.08)    (additive)
Cov(D_i, D_j)    = P(D_i ∩ D_j) − PD_i·PD_j   ← from copula.joint_default_probability()
LossCov[i,j]     = EAD_i·LGD_i · Cov(D_i,D_j) · EAD_j·LGD_j     (bilinear → block-summable)
Var(Loss_S)      = sum of LossCov over i,j ∈ S                  (segment S)
σ_down,S         = sqrt(Var(Loss_S))   [L1]  OR downside semideviation from sims [L2]
```

**Why this aggregates (the whole reason for the design):** numerator pieces are sums; the
denominator is a quadratic form on the loss-covariance matrix, so a segment's variance is just the
sum of that matrix's block. Therefore a segment metric is computed from segment-level aggregates,
**never** as an average of per-borrower metrics (that would be wrong under correlation). A
concentrated/contagious segment automatically gets a larger denominator via the off-diagonal terms.

**Single-borrower caveat (document in code):** for one borrower, L0=L1 and the "Sortino" reduces to
`E[profit] / (EAD·LGD·√(PD(1−PD)))` — a deterministic function of PD with no diversification info.
The metric is only *interesting* at segment/portfolio level. Per-borrower values are building blocks.

**Negative-numerator pathology (document + handle):** when `E[Profit] − hurdle < 0`, Sharpe/Sortino/
RAROC invert sign and "more risk looks better." Handle explicitly (see §2.5): emit the signed value
but also a boolean `numerator_negative` flag; for *riskiness ranking* the harness leads with CoV,
which is always ≥ 0 and well-defined.

---

## 1. Confirmed integration points (already verified — do not re-discover)

From `src/copula_model.py` (a fitted `CopulaDefaultModel`):
- `copula.marginal_pds` : np.ndarray (n,), PDs in [0,1].
- `copula.correlation_matrix` : np.ndarray (n,n), PSD, diag=1.
- `copula.n` : int.
- `copula.is_fitted` : bool.
- `copula.joint_default_probability(i, j) -> float` : P(D_i ∩ D_j) for a pair.
- `copula.joint_default_probability() -> np.ndarray (n,n)` : **full matrix, vectorized** — USE THIS
  to build `Cov(D_i,D_j)` for ALL pairs at once. Do **not** loop pairs in Python.
- `copula.simulate_defaults(n_simulations) -> np.ndarray (n_sim, n)` : binary draws (for L2).

Package wiring:
- `src/__init__.py` re-exports public symbols (lines ~43–52). Add the new ones there.
- Config dataclasses live in `src/config.py`. Add metric knobs to `RiskConfig` (see §3).

Test harness (`test_copula_framework.py`):
- Tests are functions `test_<name>(...)`; registered in `run_all_tests()` via `run(name, fn, *args)`
  which appends to a `failures` list. The summary uses a **hardcoded `total = 23`** near the bottom.
- **When adding N tests, bump `total` to `23 + N`** and add one `run(...)` line per test.
- Each test prints `"Test NN: <desc>... "` then `"PASSED"`. Match the existing style.

---

## 2. New module: `src/risk_adjusted_metrics.py`

### 2.1 Module-level metric registry

Implement each metric as a small pure function with a uniform signature, registered by name so the
user can list/select/extend them. Suggested contract:

```python
# A MetricFn maps a MetricInputs bundle -> float (the ratio for one unit/segment).
METRIC_REGISTRY: dict[str, MetricFn] = {}

def register_metric(name): ...        # decorator
def available_metrics() -> list[str]  # sorted registry keys
def compute_metric(name, inputs) -> float
```

`MetricInputs` (a dataclass) carries the already-aggregated primitives for a unit/segment so each
metric fn is trivial and correlation-aware:
```
expected_profit: float        # Σ E[profit_i] over the unit
expected_loss:   float        # Σ E[loss_i]
expected_revenue:float
capital:         float        # Σ capital_i
loss_var_L1:     float        # quadratic form on LossCov block (pairwise copula)
loss_std_indep:  float        # sqrt(Σ diag(LossCov))  → L0, ignores correlation
downside_semidev:Optional[float]  # from sims, may be None (L2)
hurdle_rate:     float
risk_free_rate:  float
```

**Metrics to register (all of them — this is the experiment surface):**

| name | formula | denominator sees correlation? |
|---|---|---|
| `coefficient_of_variation` | `loss_std_indep / expected_loss` | no (L0) — always ≥0, ranking-safe |
| `coefficient_of_variation_copula` | `sqrt(loss_var_L1) / expected_loss` | **yes (L1)** |
| `raroc` | `expected_profit / capital` | no (capital is k·EAD) |
| `sharpe_indep` | `(expected_profit − risk_free·expected_revenue) / loss_std_indep` | no (L0) |
| `sortino_indep` | same numerator / `loss_std_indep` (downside = all loss var) | no (L0) |
| `sortino_copula` | `(expected_profit − hurdle·capital) / sqrt(loss_var_L1)` | **yes (L1)** |
| `sortino_simulated` | `(expected_profit − hurdle·capital) / downside_semidev` | **yes (L2, tail)** |

Notes:
- In credit, "all loss variance is downside," so `sharpe_indep` and `sortino_indep` share the L0
  denominator; keep both names so the user can see they coincide here (a teaching point) and so the
  registry is explicit. `sortino_simulated` is the *true* downside-only one (semideviation of the
  simulated loss above its mean).
- Every metric returns a float; guard divide-by-zero by returning `np.nan` (NOT a fudge constant
  like `+0.01`/`+1` — those silently distort and are exactly what we're replacing).

### 2.2 The `RiskRatioCalculator` class

```python
class RiskRatioCalculator:
    def __init__(self, copula, persons, *,
                 exposures=None, lgd=0.45,
                 revenue=None, capital=None,
                 hurdle_rate=0.10, risk_free_rate=0.02,
                 capital_ratio=0.08):
        ...
```
Responsibilities:
1. **Resolve primitives once** (vectorized, no `iterrows`):
   - `self.ead`  = `exposures` if given else fallback (§2.3).
   - `self.lgd`  scalar or vector.
   - `self.revenue` = `revenue` arg / `persons['revenue']` if present else fee proxy (§2.3).
   - `self.capital` = `capital` arg / `persons['capital']` if present else `capital_ratio*ead`.
   - `self.pd` = `copula.marginal_pds`.
   - `self.el` = `ead*lgd*pd` ; `self.eprofit = revenue − el`.
2. **Build the loss-covariance matrix once**:
   ```
   J = copula.joint_default_probability()          # (n,n) full matrix
   D = pd_outer = np.outer(pd, pd)
   cov_def = J - D                                  # Cov(D_i,D_j); diag should ≈ pd*(1-pd)
   np.fill_diagonal(cov_def, pd*(1-pd))             # enforce exact Bernoulli variance on diag
   el_vec = ead*lgd                                 # (n,)
   self.loss_cov = (el_vec[:,None] * cov_def * el_vec[None,:])   # (n,n)
   ```
   Cache it. This is the single most important object; everything else is a block-sum of it.
3. **`_inputs_for(members) -> MetricInputs`**: given an index array (or None=all), aggregate:
   - sums of profit/loss/revenue/capital over `members`;
   - `loss_var_L1 = loss_cov[np.ix_(members,members)].sum()`;
   - `loss_std_indep = sqrt(diag(loss_cov)[members].sum())`;
   - `downside_semidev`: only if requested (lazy; see §2.4).
4. Public API:
   ```python
   def metric(self, name, members=None, *, with_sim=False) -> float
   def all_metrics(self, members=None, *, with_sim=False) -> dict[str,float]
   def by_segment(self, segment_col, *, metrics=None, with_sim=False) -> pd.DataFrame
   def per_borrower(self, *, metrics=None) -> pd.DataFrame
   def diversification_ratio(self, members=None) -> float   # Σσ_i / σ_portfolio  (≥1; =1 iff perfectly corr)
   ```

### 2.3 Pluggable revenue / capital (real-data readiness)
- If the caller passes `revenue`/`capital` arrays, or `persons` has `revenue`/`capital` columns,
  use them. Else reproduce the *existing* proxies so behavior matches today's pipeline:
  revenue ≈ fee proxy from transaction volume (mirror `client_value_metrics._compute_revenue_metrics`
  conceptually, but accept it precomputed to avoid re-reading transactions), capital = `0.08*EAD`.
- Document clearly that swapping in real interest-margin / RWA capital requires only these arrays —
  no formula changes. This is the hook that makes the metric usable on a real bank book.

### 2.4 `by_segment` — the headline feature
```python
calc.by_segment('city_name')        # geo
calc.by_segment('risk_archetype')   # segment
calc.by_segment('high_risk_group_id')  # groups (drop -1 = "no group")
```
Returns one row per segment with columns:
`segment, n, exposure, exposure_share, expected_profit, expected_loss,
 loss_std_indep, loss_std_copula(=sqrt L1), diversification_ratio,
 <one column per selected metric>`.
- `exposure_share` = segment EAD / total EAD.
- Aggregation is **always** from segment-level primitives (block-sum of `loss_cov`), never an
  average of per-borrower metrics. Add a one-line comment stating this invariant.
- `with_sim=True` adds the L2 `sortino_simulated` per segment by simulating once for the whole
  portfolio and slicing columns to segment members (re-use one `simulate_defaults` draw for all
  segments to keep cost down and noise consistent).

### 2.5 Negative-numerator handling
- Each metric fn that has profit in the numerator: if `expected_profit - hurdle*capital < 0`,
  still return the signed float, but `all_metrics`/`by_segment`/`per_borrower` must also carry a
  boolean `numerator_negative` column. Document that ranking by Sharpe/Sortino is only meaningful
  among numerator-positive units; for pure riskiness use `coefficient_of_variation*`.

---

## 3. Config additions (`src/config.py`, `RiskConfig`)
Add fields (with `__post_init__` validation in the existing style):
```
hurdle_rate: float = 0.10          # 0<h<1
risk_free_rate: float = 0.02       # 0<=rf<1
capital_ratio: float = 0.08        # 0<k<=1
metric_sim_paths: int = 10000      # >0  (L2 Monte-Carlo)
default_metrics: tuple[str,...] = ('coefficient_of_variation_copula','raroc','sortino_copula')
```
Do not change existing `RiskConfig` weights/fields. Keep the weights-sum-to-1 check intact.

---

## 4. Comparison harness: `src/metric_comparison.py`
A thin layer whose job is to let the user *test metrics against each other*:
```python
class MetricComparator:
    def __init__(self, calc: RiskRatioCalculator): ...
    def borrower_table(self, metrics=None) -> pd.DataFrame      # per-borrower, all metrics as columns
    def segment_table(self, segment_col, metrics=None) -> pd.DataFrame
    def rank_correlation(self, metrics=None, level='borrower', segment_col=None) -> pd.DataFrame
        # Spearman rank-corr matrix BETWEEN metrics → shows which metrics agree on ordering
    def disagreements(self, metric_a, metric_b, top_n=20) -> pd.DataFrame
        # units where rank(metric_a) and rank(metric_b) differ most → the "interesting" cases
    def divergence_flags(self) -> pd.DataFrame
        # e.g. raroc high but sortino_copula low → concentration/contagion early-warning
```
- `rank_correlation` is the core "which metric should I trust" tool: if two metrics are ~0.99
  Spearman-correlated they're redundant; low correlation means they encode different risk.
- `divergence_flags`: compute z-scores of `raroc` vs `sortino_copula` ranks; flag large gaps. This
  is the deliverable the user specifically wanted (RAROC blind to correlation, Sortino not).

---

## 5. Wiring into the existing pipeline (`main.py`)
Add **one new step** AFTER the current STEP 8 (client value), labelled `# STEP 8b — RISK-ADJUSTED
METRIC FAMILY` (do not renumber existing steps; keep the `# STEP` grep contract intact). It must:
1. Build `RiskRatioCalculator` from the already-fitted `copula`, `persons`, `exposures` (reuse the
   `exposures = income/mean(income)` already computed for `RiskAnalyzer`).
2. Print `available_metrics()`.
3. Print `calc.by_segment('city_name')` and `calc.by_segment('risk_archetype')`.
4. Build `MetricComparator`; print `rank_correlation()` and the top rows of `divergence_flags()`.
5. Save CSVs to `output/`: `metric_by_city.csv`, `metric_by_archetype.csv`,
   `metric_rank_correlation.csv`, `metric_divergence_flags.csv`.
6. Save one chart `output/metric_comparison.png`: a small-multiples bar chart of each metric across
   archetypes (re-use matplotlib only; no new deps).
Guard the whole step in try/except with a logged warning, matching the resilience style of other
optional steps, so a metric failure never aborts the pipeline.

Also: add the new public symbols to `src/__init__.py` exports
(`RiskRatioCalculator`, `MetricComparator`, `available_metrics`, `register_metric`,
`compute_metric`, `MetricInputs`).

## 6. Customer profile integration (`src/customer_profile.py`)
- Add three float fields to `CustomerRiskProfile`: `coefficient_of_variation`, `raroc`,
  `sortino_copula` (default 0.0 so existing construction can't break).
- In `CustomerProfiler.fit`, accept an optional `risk_ratio_calc=None`; if provided, look up the
  three per-borrower metrics in `_business_data` / a new `_metric_data(person_id)` and populate them.
- Add a "RISK-ADJUSTED METRICS" section to `profile_report`. Keep it optional — if no calc was
  passed, the section prints "n/a" rather than failing. Do not change the existing constructor call
  sites that omit the new arg.

---

## 7. Tests (`test_copula_framework.py`) — add 6, bump total 23 → 29
Add these and register each with a `run(...)` line; update `total = 29` and (if present) the
"All N tests passed" expectations:

1. `test_metric_registry()` — `available_metrics()` returns the 7 names; `compute_metric` dispatches;
   unknown name raises.
2. `test_metric_primitives_additivity()` — for two disjoint segments A,B: `EL(A∪B)==EL(A)+EL(B)` and
   `expected_profit(A∪B)==…` (exact, within 1e-9). This proves the aggregation contract.
3. `test_single_borrower_reduces_to_closed_form()` — for one borrower, `loss_std_indep` equals
   `EAD·LGD·sqrt(PD(1−PD))` within 1e-9, and `sortino_copula` denominator == that value.
4. `test_correlation_inflates_denominator()` — construct/select two strongly-correlated borrowers;
   assert their `sqrt(loss_var_L1)` (segment, with copula) > `loss_std_indep` (independent). Proves
   contagion enters the denominator.
5. `test_by_segment_shapes_and_invariants()` — `by_segment('risk_archetype')` has one row per
   archetype, `exposure_share` sums to ~1.0, `diversification_ratio >= 1 - 1e-9` for every segment.
6. `test_metric_comparison_rank_corr()` — `rank_correlation()` returns a square symmetric matrix with
   1.0 on the diagonal and all entries in [-1,1]; `disagreements('raroc','sortino_copula')` returns
   ≤ top_n rows.

Run `python test_copula_framework.py` → must print `All 29 tests passed.`
Run `python debug.py smoke` → must still pass.
Run `python main.py` → must complete and write the new outputs.

---

## 8. Acceptance checklist (the implementer must verify all)
- [ ] `src/risk_adjusted_metrics.py`, `src/metric_comparison.py` created; no `iterrows` in hot paths
      (loss-cov built from the full `joint_default_probability()` matrix).
- [ ] Existing 23 tests unchanged and passing; 6 new tests passing; summary reads `All 29 tests passed.`
- [ ] `main.py` STEP 8b runs, prints segment tables + rank-correlation, writes 4 CSVs + 1 PNG.
- [ ] `by_segment` works for `city_name`, `risk_archetype`, `high_risk_group_id` (−1 dropped).
- [ ] Divide-by-zero → `np.nan`, never a fudge constant. Negative-numerator → signed value + flag.
- [ ] Revenue/capital are pluggable (arrays or columns), with documented fallbacks.
- [ ] `CustomerRiskProfile` gains 3 metric fields; `profile_report` shows them or "n/a"; old call
      sites unaffected.
- [ ] New public symbols exported from `src/__init__.py`.
- [ ] `CLAUDE.md` updated: add the two new modules to the file map, add a "Risk-adjusted metrics"
      row to the "Where to look" table, and note the additivity invariant.

---

## 9. Sequencing for the implementer (suggested commits)
1. `risk_adjusted_metrics.py` (registry + calculator + primitives) + tests 1–4. Green.
2. `by_segment` + `diversification_ratio` + tests 5. Green.
3. `metric_comparison.py` + test 6. Green.
4. `config.py` knobs + `__init__.py` exports.
5. `main.py` STEP 8b + outputs.
6. `customer_profile.py` integration.
7. `CLAUDE.md` doc update. Final full-suite + `main.py` + `debug.py smoke` run.

Keep each step's tests green before moving on. Do not refactor unrelated code.
```
