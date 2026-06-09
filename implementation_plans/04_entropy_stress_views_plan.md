# Plan 04: Entropy-Based Stress Views

## Status

**Prerequisite implemented (2026-06-06).** The minimum-relative-entropy primitive
this plan builds on now exists and is tested:

- `src/relative_entropy.py` — `min_rel_entropy_sp` (arpym `views.min_rel_entropy_sp`
  port, numpy/scipy only). Covered by `test_relative_entropy` in
  `test_copula_framework.py`.

It is also already used by `src/conditional_fp.py` (conditional FP) and
`src/credit_transitions.py` (transition-matrix monotonicity). This plan remains
open for the stress-view engine layer (`src/scenarios/`, weighted risk, agent API)
described below.

## Objective

Implement ARPM-style entropy stress testing, where stress scenarios are expressed
as probability views and solved through minimum relative entropy rather than
hard-coded PD or correlation multipliers.

The project already has `src/relative_entropy.py`, a local implementation of
minimum-relative-entropy scenario reweighting. This plan turns it into a
risk-department stress-view engine.

## Current Code To Read First

- `src/relative_entropy.py`
- `src/flexible_probs.py`
- `src/risk_metrics.py`
- `src/risk_adjusted_metrics.py`
- `src/agents.py`
- `METHODOLOGY.md`
- `CAPABILITIES.md`
- `AGENTS.md`

## New Modules

Create:

- `src/scenarios/__init__.py`
- `src/scenarios/stress_views.py`
- `src/scenarios/entropy_stress.py`
- `src/scenarios/weighted_risk.py`

## Stress View Concepts

A stress view is a constraint on expected scenario quantities under posterior
probabilities.

Examples:

```text
E[portfolio_default_rate] = 0.08
E[portfolio_loss] >= 2 * base_expected_loss
E[loss_city_Almaty] >= X
E[default_rate_transfer_cluster_12] >= 0.15
E[macro_stress_score] = current_stress_score
```

The solver finds posterior scenario probabilities closest to the prior while
satisfying the views.

## Public API

Implement:

```python
StressView(
    name: str,
    variable: str,
    operator: str,
    target: float,
)
```

Operators:

- `"=="`.
- `"<="`.
- `">="`.

Implement:

```python
build_scenario_matrix(
    scenario_df,
    variables,
) -> tuple[numpy.ndarray, list[str]]
```

```python
apply_entropy_views(
    prior_probs,
    scenario_values,
    views,
    variable_names,
) -> EntropyStressResult
```

```python
run_entropy_stress(
    losses,
    defaults=None,
    prior_probs=None,
    views=None,
    alpha=0.95,
) -> dict
```

Result fields:

- posterior probabilities.
- prior probabilities.
- KL divergence.
- effective number of scenarios.
- view satisfaction table.
- expected loss under prior.
- expected loss under posterior.
- VaR under prior.
- VaR under posterior.
- ES under prior.
- ES under posterior.
- warnings.

## Weighted Risk Functions

Implement weighted versions of:

- expected loss.
- VaR.
- ES.
- default rate.
- segment loss.

Weighted VaR:

- sort scenarios by loss.
- cumulative sum posterior probabilities.
- first loss where cumulative probability exceeds confidence level.

Weighted ES:

- weighted average of tail losses above VaR.
- handle edge cases where tail probability is tiny.

## Scenario Inputs

Support two paths:

1. Use existing simulated loss/default scenarios from `RiskAnalyzer`.
2. Use an externally supplied `scenario_df`.

Scenario data should support columns such as:

- `portfolio_loss`
- `portfolio_default_rate`
- `loss_city_<id>`
- `loss_geo_cluster_<id>`
- `loss_transfer_cluster_<id>`
- `macro_stress_score`

## Integration With Flexible Probabilities

If the platform has flexible-probability prior weights:

```text
prior = flexible_probs_weights
posterior = entropy_pooling(prior, views)
```

If not:

```text
prior = uniform
posterior = entropy_pooling(uniform, views)
```

Do not replace `FlexibleProbsCalibrator`; compose on top of it.

## Agent API Addition

In `src/agents.py`, add:

```python
RiskAgentAPI.run_entropy_stress(views, alpha=0.95, n_scenarios=10000, ...)
```

Input should be JSON-safe:

```python
views = [
    {"name": "portfolio_default_rate_8pct", "variable": "portfolio_default_rate", "operator": "==", "target": 0.08},
    {"name": "loss_floor", "variable": "portfolio_loss", "operator": ">=", "target": 1000000.0},
]
```

Output should include:

- summary sentence.
- weighted risk metrics.
- view satisfaction table.
- top reweighted scenarios.
- warnings if constraints are infeasible or too concentrated.

## Implementation Steps

1. Create `src/scenarios/` package.
2. Implement `StressView` dataclass.
3. Implement view parsing and validation.
4. Implement scenario matrix construction.
5. Implement `apply_entropy_views` using `min_rel_entropy_sp`.
6. Implement weighted EL, VaR, ES.
7. Connect to existing loss simulations.
8. Add `RiskAgentAPI.run_entropy_stress`.
9. Export public functions in `src/__init__.py`.
10. Add tests.
11. Run full test suite.

## Tests

Add tests for:

- No views returns normalized prior.
- Equality view reaches target within tolerance.
- Inequality `<=` is respected.
- Inequality `>=` is respected.
- Posterior sums to 1.
- Posterior is nonnegative.
- Effective scenario count decreases under strong views.
- Weighted VaR/ES are monotonic under harsher loss view.
- Infeasible view returns clear warning or exception.
- Agent API returns JSON-safe result.

## Acceptance Criteria

- Risk team can define stress as a list of views.
- Stress output includes prior vs posterior risk metrics.
- Existing hard-coded stress methods continue to work.
- All tests pass.

## Non-Goals

- Do not remove current `run_stress`.
- Do not create a GUI in this task.
- Do not require external ARPyM dependency.

