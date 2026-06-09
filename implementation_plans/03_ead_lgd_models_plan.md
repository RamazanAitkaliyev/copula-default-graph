# Plan 03: EAD and LGD Models

## Objective

Replace crude exposure and loss-severity assumptions with proper borrower-level
EAD and LGD inputs while preserving backward compatibility with the current
flat-LGD workflow.

The current platform has strong dependence and metric logic, but loss magnitude
is only as good as `EAD` and `LGD`. This plan makes `EL = PD * EAD * LGD`
auditable at borrower, segment, cluster, and portfolio levels.

## Current Code To Read First

- `src/loaders.py`
- `src/client_value_metrics.py`
- `src/risk_adjusted_metrics.py`
- `src/risk_metrics.py`
- `src/cluster_metrics.py`
- `src/customer_profile.py`
- `src/agents.py`
- `METHODOLOGY.md`
- `AGENTS.md`

## New Modules

Create:

- `src/exposure/__init__.py`
- `src/exposure/contracts.py`
- `src/exposure/ead_model.py`
- `src/exposure/lgd_model.py`
- `src/exposure/summary.py`

## Data Contract

Canonical borrower columns:

- `exposure_at_default`
- `lgd`
- `outstanding_balance`
- `current_balance`
- `undrawn_limit`
- `credit_conversion_factor`
- `collateral_value`
- `collateral_type`
- `product_type`
- `seniority`
- `region`

Keep existing fallbacks:

- If `exposure_at_default` exists, use it.
- Else if `income` exists, use current income proxy.
- Else use safe flat fallback only for demos.
- If `lgd` exists, use it.
- Else use flat `0.45`.

## Public API

Implement:

```python
estimate_ead(
    persons,
    product_col="product_type",
    out_col="exposure_at_default",
    fallback="income_proxy",
    ccf_by_segment=None,
) -> pandas.DataFrame
```

```python
estimate_lgd(
    persons,
    product_col="product_type",
    collateral_col="collateral_type",
    out_col="lgd",
    fallback_lgd=0.45,
    lgd_by_segment=None,
) -> pandas.DataFrame
```

```python
get_ead_array(persons, fallback_value=10000.0) -> numpy.ndarray
```

```python
get_lgd_array(persons, fallback_lgd=0.45) -> numpy.ndarray
```

```python
exposure_summary(persons) -> dict
```

## EAD Logic

Term loan:

```text
EAD = outstanding_balance
```

Revolving credit:

```text
EAD = current_balance + CCF * undrawn_limit
```

Credit card:

```text
EAD = current_balance + CCF_card * undrawn_limit
```

Fallback:

```text
EAD = exposure_at_default if present
EAD = income proxy if no direct EAD fields
EAD = flat fallback only for synthetic/demo data
```

Validation:

- EAD must be finite.
- EAD must be nonnegative.
- Missing values should either be imputed with warnings or fail based on policy.

## LGD Logic

Direct LGD:

```text
LGD = persons["lgd"]
```

Collateral proxy:

```text
recovery_rate = min(collateral_value / outstanding_balance, max_recovery_cap)
LGD = 1 - recovery_rate
```

Segment table:

```text
LGD = lookup(product_type, collateral_type, region)
```

Fallback:

```text
LGD = 0.45
```

Validation:

- LGD must be finite.
- LGD must be in `[0,1]`.
- Clamp only if policy allows; otherwise raise.

## Changes To Existing Risk Code

Update `RiskRatioCalculator`:

- Current `lgd` scalar path must continue to work.
- Add support for `lgd` array with length equal to number of borrowers.
- Internally compute:

```python
loss_weight = exposures * lgd_array
```

Update modules that compute expected loss:

- `risk_adjusted_metrics.py`
- `risk_metrics.py`
- `cluster_metrics.py`
- `customer_profile.py`
- `client_value_metrics.py`

Rule:

```text
expected_loss_i = pd_i * ead_i * lgd_i
```

Do not average LGD after computing ratios. Aggregate primitives first.

## Loader Updates

Extend `ColumnMapping` in `src/loaders.py` to support:

- `lgd`
- `outstanding_balance`
- `current_balance`
- `undrawn_limit`
- `credit_conversion_factor`
- `collateral_value`
- `collateral_type`
- `product_type`
- `seniority`

Add validation:

- Nonnegative balances.
- Nonnegative collateral.
- LGD in `[0,1]`.
- CCF in sensible range, ideally `[0,1.5]` with warnings above `1`.

## Agent API Additions

In `src/agents.py`, add:

```python
RiskAgentAPI.estimate_exposures(...)
RiskAgentAPI.exposure_summary(...)
```

`estimate_exposures` should:

- add or refresh `exposure_at_default`.
- add or refresh `lgd`.
- return warnings for fallback usage.

`exposure_summary` should return:

- total EAD.
- average LGD.
- exposure by product.
- exposure by city/risk archetype if columns exist.
- count using fallback EAD.
- count using fallback LGD.

## Implementation Steps

1. Create `src/exposure/` package.
2. Implement EAD helpers and validation.
3. Implement LGD helpers and validation.
4. Update `RiskRatioCalculator` for LGD arrays.
5. Update expected-loss calculations across modules.
6. Extend `ColumnMapping`.
7. Update `load_persons` validation.
8. Add agent methods.
9. Add exports in `src/__init__.py`.
10. Add tests.
11. Run full test suite.

## Tests

Add tests for:

- Scalar LGD behavior is unchanged.
- Vector LGD changes expected loss correctly.
- EAD from `outstanding_balance` works for term loans.
- EAD from balance plus CCF times undrawn limit works for revolving products.
- LGD from direct column works.
- LGD from collateral proxy works.
- Invalid negative EAD raises.
- Invalid LGD above 1 raises or is clipped only under explicit clip policy.
- Segment EL equals sum of borrower EL.
- Agent exposure summary is JSON-safe.

## Acceptance Criteria

- Existing demos still run with flat LGD.
- Real borrower-level EAD/LGD can flow through all metrics.
- `EL`, `VaR`, `ES`, `Sortino`, `RAROC`, and cluster metrics use borrower-level
  loss weights.
- All tests pass.

## Non-Goals

- Do not build advanced recovery-time models yet.
- Do not build full product pricing in this task.
- Do not change copula dependence logic.

