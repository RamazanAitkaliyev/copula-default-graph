# Plan 07: Copula Parameter Calibration From Data

## Status

**First version implemented (2026-06-06).**

- `src/dependence.py` — `schweizer_wolff` (rank dependence measure) +
  `copula_invariance_test`. Self-contained ports; the port **fixes a
  normalisation bug** in arpym's `schweizer_wolff` (the original returns values
  > 1 on independent data; ours correctly returns ≈0). Tested by
  `test_dependence_measures`.
- `src/copula_calibration.py` — `build_default_panel`,
  `empirical_dependence_measures` (Kendall τ, Spearman ρ, Schweizer-Wolff,
  default correlation, observed vs independent joint-default, lower-tail
  co-default), `calibrate_copula` (τ → Gaussian/Student-t/Clayton parameters +
  family comparison with Fréchet-bound check + recommendation). Tested by
  `test_copula_calibration`.
- Agent API: `RiskAgentAPI.calibrate_copula_from_data(events, family, apply)` —
  diagnostic by default; refits the live Clayton θ only when `apply=True`.

Still open (deeper second version): per-family maximum-likelihood fitters,
multi-factor `beta`-by-segment calibration, and a richer goodness-of-fit /
tail-dependence comparison.

## Objective

Estimate copula dependence parameters from observed default, delinquency,
rating, migration, or loss data instead of relying only on configured
correlation/loadings.

This gives the risk team a defensible answer to: "Where did the copula
parameters come from?"

## Current Code To Read First

- `src/copula_model.py`
- `src/factor_copula.py`
- `src/multi_factor_copula.py`
- `src/graph_features.py`
- `src/transfer_clusters.py`
- `src/geo_clusters.py`
- `src/risk_adjusted_metrics.py`
- `src/agents.py`
- `Python/arpym/arpym/statistics/schweizer_wolff.py`
- `Python/arpym/arpym/statistics/invariance_test_copula.py`
- `METHODOLOGY.md`

## New Modules

Create:

- `src/copula_calibration/__init__.py`
- `src/copula_calibration/empirical.py`
- `src/copula_calibration/fitters.py`
- `src/copula_calibration/diagnostics.py`
- `src/copula_calibration/report.py`

## Calibration Inputs

Minimum input:

- borrower ID.
- time period.
- observed default indicator or delinquency indicator.
- PD estimate.
- optional segment/cluster ID.

Useful optional inputs:

- `geo_cluster_id`
- `transfer_cluster_id`
- `city_id`
- `risk_archetype`
- rating migration state.
- loss amount.
- macro regime.

## Public API

Implement:

```python
build_default_panel(
    events,
    borrower_col="person_id",
    time_col="period",
    default_col="default",
    pd_col="model_pd",
) -> pandas.DataFrame
```

```python
empirical_dependence_measures(
    panel,
    default_col="default",
    pd_col="model_pd",
    segment_col=None,
) -> pandas.DataFrame
```

```python
fit_gaussian_copula_params(panel, ...)
```

```python
fit_student_t_copula_params(panel, ...)
```

```python
fit_clayton_copula_params(panel, ...)
```

```python
calibrate_copula(
    panel,
    family="auto",
    segment_col=None,
) -> dict
```

```python
copula_diagnostics(
    fitted_params,
    panel,
    family,
) -> dict
```

## Empirical Dependence Measures

Implement:

- Pairwise default co-occurrence by segment.
- Observed joint default rate.
- Expected independent joint default rate.
- Default correlation.
- Kendall tau.
- Spearman rho.
- Schweizer-Wolff dependence.
- Lower-tail co-default frequency.
- Upper-tail survival frequency if useful.

Segment-level outputs:

- `segment`
- `n_borrowers`
- `n_periods`
- `n_defaults`
- `observed_joint_default`
- `independent_joint_default`
- `default_corr`
- `kendall_tau`
- `spearman_rho`
- `schweizer_wolff`
- `tail_codefault_rate`
- `warnings`

## Fitting Logic

Gaussian copula:

- Estimate latent asset correlation from default correlation where possible.
- Alternative: fit by minimizing error between model joint default probability
  and observed co-default rate.

Student-t copula:

- Estimate correlation and degrees of freedom.
- Start with grid over `nu = [3, 4, 5, 6, 8, 10, 15, 30]`.
- Select by likelihood or joint-default error.

Clayton copula:

- Estimate theta from Kendall tau:

```text
theta = 2 * tau / (1 - tau)
```

- Validate positive theta.
- Use lower-tail co-default diagnostics.

Multi-factor copula:

- Estimate beta by segment/cluster using observed within-factor co-default
  inflation.
- Enforce:

```text
sum_k beta_i,k^2 < 1
```

## Goodness-of-Fit Diagnostics

Implement:

- Empirical vs model joint default probability.
- Frechet bounds check.
- Tail-dependence comparison.
- Segment-level fit error.
- Family comparison table:
  - Gaussian.
  - Student-t.
  - Clayton.
- Warnings for insufficient observations.
- Warnings for unstable/default-sparse segments.

Output:

- recommended family.
- recommended parameters.
- diagnostics table.
- warnings.

## Integration With Existing Models

Add methods carefully:

```python
CopulaDefaultModel.fit_calibrated(...)
```

or keep separate first:

```python
params = calibrate_copula(...)
model = CopulaDefaultModel(copula_type=params["family"])
model.fit(pds, corr_matrix, params=params)
```

For factor copulas:

```python
MultiFactorCopula.fit_calibrated(pds, factors, observed_panel)
```

This can be a second phase after dense copula calibration works.

## Agent API Additions

In `src/agents.py`, add:

```python
RiskAgentAPI.calibrate_copula(...)
RiskAgentAPI.copula_diagnostics(...)
```

Output should include:

- recommended family.
- parameters.
- segment diagnostics.
- warnings.
- whether calibrated parameters were applied to the live pipeline.

Default should be diagnostic-only:

```text
apply=False
```

Only overwrite live copula settings when `apply=True`.

## Implementation Steps

1. Create `src/copula_calibration/` package.
2. Implement panel validation.
3. Implement empirical dependence measures.
4. Implement Gaussian fitter.
5. Implement Clayton fitter.
6. Implement Student-t grid fitter.
7. Implement diagnostics and family comparison.
8. Implement segment-level calibration.
9. Add agent methods.
10. Export selected functions in `src/__init__.py`.
11. Add tests.
12. Run full test suite.

## Tests

Add tests for:

- Synthetic independent data estimates near-zero dependence.
- Synthetic Gaussian copula data recovers approximate positive correlation.
- Synthetic Clayton-like lower-tail data estimates positive theta.
- Kendall tau to Clayton theta formula behaves correctly.
- Insufficient defaults return warnings, not crashes.
- Model joint probabilities obey Frechet bounds.
- Diagnostics table is JSON-safe.
- Agent API does not mutate live model unless `apply=True`.

## Acceptance Criteria

- Risk team can run copula calibration from observed data.
- Output explains parameter source and fit quality.
- Calibrated parameters can be compared against configured parameters.
- Existing copula and risk metric APIs continue to work.
- All tests pass.

## Non-Goals

- Do not build a full academic maximum-likelihood suite in the first version.
- Do not require external ARPyM as a dependency.
- Do not replace graph-derived correlation immediately; calibrate and compare
  first.

