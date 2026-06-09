# Plan 06: Lifetime PD via Markov Delinquency States

## Status

**Transition-estimation prerequisite implemented (2026-06-06).** The rigorous
estimator this plan relies on for "Transition Estimation" mode 2/3 now exists and
is tested:

- `src/credit_transitions.py` — `fit_trans_matrix_credit` (continuous-time
  generator + half-life weighting + entropy-regularised monotonicity),
  `estimate_generator`, `cohort_arrays_from_events` (tidy-event on-ramp).
- `RatingEngine.from_cohort_data(...)` builds an engine straight from cohort data.
- Covered by `test_credit_transitions`. Depends on `src/relative_entropy.py`
  (also implemented).

This plan remains open for the delinquency-state model, lifetime PD curves, and
IFRS9 staging (`src/markov/`) described below — which should reuse the functions
above rather than re-deriving them.

## Objective

Extend one-period PD into multi-period lifetime PD using Markov delinquency
states. This supports IFRS9/CECL-style expected credit loss workflows and
connects temporal default migration with the platform's cross-sectional copula
dependence.

## Current Code To Read First

- `src/rating_engine.py`
- `src/credit_transitions.py`
- `src/relative_entropy.py`
- `src/customer_profile.py`
- `src/risk_adjusted_metrics.py`
- `src/agents.py`
- `METHODOLOGY.md`
- `ROADMAP.md`

## New Modules

Create:

- `src/markov/__init__.py`
- `src/markov/delinquency.py`
- `src/markov/lifetime_pd.py`
- `src/markov/ifrs9.py`

## State Model

Default delinquency states:

```text
0 Current
1 DPD30
2 DPD60
3 DPD90
4 Default
```

Rules:

- `Default` is absorbing.
- Rows of transition matrices sum to 1.
- Worse states should generally have higher default absorption probability.
- Projection horizons should support months and years.

## Public API

Implement:

```python
DelinquencyMarkovModel(
    transition_matrix=None,
    state_names=None,
)
```

Methods:

```python
fit_from_cohorts(dates, n_oblig, n_cum, tau_hl=None)
```

```python
project(horizon_months)
```

```python
lifetime_pd_by_state(horizon_months)
```

```python
expected_time_to_default()
```

```python
borrower_lifetime_pd(persons, state_col="delinquency_state", horizon_months=36)
```

Implement IFRS9 helpers:

```python
assign_ifrs9_stage(
    persons,
    current_pd_col="model_pd",
    origination_pd_col=None,
    delinquency_state_col="delinquency_state",
    rating_downgrade_col=None,
) -> pandas.DataFrame
```

```python
compute_lifetime_ecl(
    persons,
    lifetime_pd_col="lifetime_pd",
    ead_col="exposure_at_default",
    lgd_col="lgd",
) -> pandas.DataFrame
```

## Transition Estimation

Support three input modes:

1. Supplied transition matrix.
2. Cohort arrays:
   - `dates`
   - `n_oblig`
   - `n_cum`
   - use `src.credit_transitions.fit_trans_matrix_credit`.
3. Tidy event table:
   - period.
   - from state.
   - to state.
   - count.
   - convert with `cohort_arrays_from_events`.

Projection:

```text
P(t) = exp(G * t)
```

Use existing generator logic where possible.

## Lifetime PD

For a borrower currently in state `s`:

```text
lifetime_pd(horizon) = P_horizon[s, Default]
```

For curves:

```text
horizon_months = [1, 3, 6, 12, 24, 36, 60]
```

Output:

- one row per state.
- one column per horizon.
- default absorption probability.

Expected time to default:

- Use absorbing Markov chain fundamentals where applicable.
- If absorption is impossible or matrix is degenerate, return `nan` plus warning.

## IFRS9 Staging Logic

Simple first version:

- Stage 1:
  - not defaulted.
  - no significant increase in credit risk.
  - use 12-month ECL.
- Stage 2:
  - significant increase in credit risk.
  - examples: large PD increase, rating downgrade, DPD30/DPD60.
  - use lifetime ECL.
- Stage 3:
  - defaulted or DPD90.
  - impaired/default.

Columns to add:

- `ifrs9_stage`
- `lifetime_pd`
- `pd_12m_markov`
- `lifetime_ecl`

## Copula Integration

Do not rewrite copula simulation in this task.

Current division:

- Markov chain: temporal migration toward default.
- Copula: cross-sectional dependence in a period.

Initial integration:

- Provide lifetime PD columns to reports and profiles.
- Later task can simulate correlated Markov paths.

## Agent API Additions

In `src/agents.py`, add:

```python
RiskAgentAPI.fit_delinquency_markov(...)
RiskAgentAPI.lifetime_pd(horizon_months=36)
RiskAgentAPI.ifrs9_staging(...)
```

Outputs should include:

- transition matrix.
- projected transition matrix.
- lifetime PD by state.
- borrower-level lifetime PD summary.
- stage distribution.
- ECL summary.

## Implementation Steps

1. Create `src/markov/` package.
2. Implement state constants and validation.
3. Implement `DelinquencyMarkovModel`.
4. Implement transition-matrix fitting from cohorts.
5. Implement projection by horizon.
6. Implement lifetime PD by state.
7. Implement borrower-level lifetime PD assignment.
8. Implement IFRS9 staging helper.
9. Implement lifetime ECL helper.
10. Add agent methods.
11. Export from `src/__init__.py`.
12. Add tests.
13. Run full test suite.

## Tests

Add tests for:

- Transition rows sum to 1.
- Default row is absorbing.
- Projection preserves row sums.
- Lifetime PD increases with horizon.
- DPD90 has higher lifetime PD than Current.
- Borrower-level mapping handles unknown/missing states with warnings.
- IFRS9 staging assigns Stage 3 to default/DPD90.
- Lifetime ECL equals lifetime PD times EAD times LGD.
- Agent outputs are JSON-safe.

## Acceptance Criteria

- Platform can produce 12-month and lifetime PD curves.
- Borrowers receive `lifetime_pd` and `ifrs9_stage`.
- Lifetime ECL can be computed from PD, EAD, and LGD.
- Existing rating engine remains compatible.
- All tests pass.

## Non-Goals

- Do not implement fully correlated multi-period Markov simulation yet.
- Do not replace `rating_engine.py`.
- Do not require IFRS9 accounting completeness in the first version.

