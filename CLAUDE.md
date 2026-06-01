# CLAUDE.md — Copula Default Graph

AI agents (Claude Code, Copilot, Cursor, etc.) should read this file first.

## Quick orientation

This is a **credit-risk framework** that models correlated defaults using:

1. A gradient-boosting **PD model** (`src/pd_model.py`)
2. A **transaction-graph** layer that turns money flows into a correlation matrix (`src/graph_features.py`)
3. A **Clayton copula** that turns marginal PDs + correlations into joint defaults (`src/copula_model.py`)
4. **Risk metrics** at individual / group / portfolio level (`src/risk_metrics.py`)
5. Four analytical extensions: rating engine, Merton structural PD, flexible probabilities (regime-aware copula), customer profiles

**Run order:** `main.py` executes a 13-step pipeline end-to-end. Grep `# STEP` in `main.py` to jump to any step.

**Run tests:** `python test_copula_framework.py` — should print `All 23 tests passed.`

**Run pipeline:** `python main.py` — completes in ~30-60 s, writes PNGs + CSVs to `output/`

---

## File map

```
src/
  config.py               – Dataclass configs (NetworkConfig, CopulaConfig, …)
  data_generator.py       – generate_network() → (persons_df, transactions_df)
  graph_features.py       – TransactionGraph  → correlation matrix, network stats
  copula_model.py         – CopulaDefaultModel (5 copula types) + compare_copulas()
  risk_metrics.py         – RiskAnalyzer, PortfolioRiskMetrics, FraudRingDetector
  pd_model.py             – IndividualPDModel (logistic / gradient boosting)
  client_value_metrics.py – ClientValueCalculator (Sharpe, RAROC, client segments)
  rating_engine.py        – RatingEngine (PD → AAA…Default + migration matrix)
  structural_pd.py        – StructuralPDModel (Merton structural PD, proxy for retail)
  flexible_probs.py       – FlexibleProbsCalibrator (regime-aware copula via kernel)
  customer_profile.py     – CustomerProfiler (per-borrower risk report + watchlist)
  __init__.py             – Re-exports all public symbols; read this for the full API

main.py                   – 13-step end-to-end pipeline (entry point)
test_copula_framework.py  – 23 unit tests; run with: python test_copula_framework.py
debug.py                  – Quick one-off diagnostic helpers (see below)
requirements.txt          – pip dependencies
output/                   – Generated PNGs + CSVs (git-ignored in production)
```

---

## Data schema

### `persons` DataFrame (1000 rows in synthetic mode)

| Column | Type | Notes |
|---|---|---|
| `person_id` | int | Unique; used as array index in all numpy ops |
| `city_id` | int | 0, 1, 2 → same-city correlation boost |
| `city_name` | str | |
| `risk_archetype` | str | `prime`, `near_prime`, `subprime`, `deep_subprime`, `bridge` |
| `base_pd` | float | Ground-truth PD (simulation only) |
| `model_pd` | float | ML model prediction (added in step 3 of pipeline) |
| `default` | int | 0/1 binary label (used to train PD model) |
| `income`, `age`, `employment_years`, `debt_to_income` | float | Features |
| `missed_payments`, `credit_utilization`, `num_credit_lines` | float | Features |
| `account_age_months` | float | Feature |
| `high_risk_group_id` | int | -1 = not in a high-risk group |
| `is_bridge` | bool | Bridge node between communities |
| `neighbor_pd_avg`, `neighbor_pd_max`, `n_high_risk_neighbors` | float | Added in step 2 |
| `merton_pd`, `blended_pd`, `distance_to_default`, `pd_signal_divergence` | float | Added in step 10 |

### `transactions` DataFrame

| Column | Type | Notes |
|---|---|---|
| `sender_id` | int | References `person_id` |
| `receiver_id` | int | References `person_id` |
| `amount` | float | Transaction value |

---

## Invariants (do NOT break these)

1. `copula.marginal_pds` must remain in `[0, 1]` at all times.
2. `copula.correlation_matrix` must be PSD with `diag = 1`. Always call `_nearest_psd()` after modifying it.
3. `persons['person_id']` must be unique integers. All numpy positional indexing relies on this.
4. `risk_tier` arrays must use `dtype=object` (not fixed-width str) to avoid truncation of `'critical'`.
5. The copula's state is always restored by `_stressed_copula()` context manager — do not mutate `copula.marginal_pds` or `copula.correlation_matrix` directly in stress-test code paths.
6. `test_copula_framework.py` must pass completely before merging any change.

---

## Where to look for things

| I want to... | Go to |
|---|---|
| Change default PD thresholds | `src/rating_engine.py:PD_THRESHOLDS` |
| Add a new copula type | `src/copula_model.py:CopulaDefaultModel.SUPPORTED_COPULAS` → add `_<type>_copula()` and update `simulate()` dispatch |
| Change portfolio metrics (VaR, ES) | `src/risk_metrics.py # SECTION: PORTFOLIO METRICS` |
| Add a new borrower feature | `src/data_generator.py:_generate_persons()` → also add to `IndividualPDModel` feature list in `main.py:step 3` |
| Change the stress-test logic | `src/risk_metrics.py:RiskAnalyzer._stressed_copula()` |
| Tune Merton proxy calibration | `src/structural_pd.py:_proxy_asset_vol()` |
| Change regime classification | `src/flexible_probs.py:classify_regime()` |
| Add a new customer profile field | `src/customer_profile.py:CustomerRiskProfile` dataclass + `CustomerProfiler._build_profile()` |
| Change watchlist criteria | `src/customer_profile.py:CustomerProfiler.watchlist()` |
| Add a new output chart | `main.py` between `# STEP 13` blocks |

---

## How to add a test

Tests live in `test_copula_framework.py`. Each test is a standalone function `test_<name>(...)` that
prints `"Test NN: <description>... PASSED"`. Add your function, then call it from the `if __name__ == "__main__"` block at the bottom. The total count in the summary line must match.

Pattern:
```python
def test_my_new_thing():
    print("Test 24: My new thing... ", end="")
    # ... assertions ...
    print("PASSED")
```

---

## How to debug a failing test

```bash
# Run only one test by isolating its function:
python -c "
from test_copula_framework import test_my_new_thing
test_my_new_thing()
"

# Run all tests with full tracebacks:
python test_copula_framework.py 2>&1 | grep -E "(FAILED|Error|Traceback)"

# Quick pipeline smoke test (fast, no plots):
python debug.py smoke

# Inspect copula state:
python debug.py copula

# Check one borrower's full profile:
python debug.py profile 42
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `risk_tier` shows `'cri'` instead of `'critical'` | `np.array([...])` with fixed str dtype | Use `np.full(n, 'low', dtype=object)` |
| Stress-test leaves copula in wrong state | Modified `copula.marginal_pds` outside context manager | Use `with analyzer._stressed_copula(...)` |
| `KeyError: person_id` in rating engine | Used `df.index[0]` (DataFrame index) instead of positional | Use `np.where(ids == pid)[0][0]` |
| `expected_profit > expected_revenue` | Double-assignment bug — second line overwrites first | One line: `expected_profit = revenue - expected_loss` |
| Correlation matrix not PSD | Direct assignment after adding boosts | Call `_nearest_psd(corr)` before `copula.fit()` |
| `FlexibleProbsCalibrator` wrong weights | Passed length-1 or all-NaN history | Validate with `calib.fit(history, ...)` which raises `ValueError` |
| City layout crash with N≠3 cities | Used hardcoded city centers dict | `graph_features.py` uses dynamic polar layout — do not revert |

---

## Key math references

- **Clayton copula**: `C(u,v;θ) = (u⁻θ + v⁻θ - 1)^(-1/θ)`, lower tail dep. = `2^(-1/θ)`
- **Merton PD**: `PD = N(-d2)`, `d2 = [ln(V/D) + (r - σ²/2)T] / (σ√T)`
- **KMV proxy**: `V ≈ income×12 / 0.08` (capitalised monthly-income perpetuity)
- **Rating migration**: `P(Δt) = expm(G·Δt)` where G is generator matrix
- **Flexible probs kernel**: `w_t ∝ exp(-||z_hist_t - z_curr||² / (2h²))`
- **RAROC**: `E[Profit] / (8% × EAD)`
- **Client Sharpe**: `(E[Profit] - rf×Revenue) / σ(Profit)`
- **Contagion vulnerability**: weighted avg PD uplift from neighbour defaults
- **Systemic importance**: avg PD uplift caused in others by this borrower defaulting
- **HHI (portfolio concentration)**: `Σ (exposure_i / Σexposure)²`

---

## Dependencies

```
numpy, pandas, scipy, scikit-learn, matplotlib
```
All standard — no unusual installs. Install: `pip install -r requirements.txt`

No arpym library dependency in this project (the flexible_probs module is self-contained).
