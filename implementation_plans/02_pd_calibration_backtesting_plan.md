# Plan 02: PD Calibration and Backtesting

## Objective

Make `model_pd` trustworthy as a probability before it feeds expected loss,
copula joint defaults, VaR, ES, ratings, customer profiles, and stress testing.

The platform already trains or accepts individual PDs. This plan adds the model
validation layer around those PDs: calibration metrics, calibration transforms,
segment diagnostics, drift checks, and an agent API.

## Current Code To Read First

- `src/pd_model.py`
- `src/loaders.py`
- `src/rating_engine.py`
- `src/risk_adjusted_metrics.py`
- `src/agents.py`
- `test_copula_framework.py`
- `METHODOLOGY.md`
- `AGENTS.md`

## New Modules

Create:

- `src/validation/__init__.py`
- `src/validation/pd_calibration.py`
- `src/validation/backtesting.py`
- `src/validation/report.py`

Optional later:

- `src/validation/plots.py`

## Public API

Implement:

```python
calibration_table(
    persons,
    pd_col="model_pd",
    default_col="default",
    n_bins=10,
    segment_col=None,
) -> pandas.DataFrame
```

```python
pd_calibration_metrics(
    persons,
    pd_col="model_pd",
    default_col="default",
) -> dict
```

```python
fit_pd_calibrator(
    train_df,
    pd_col="model_pd",
    default_col="default",
    method="isotonic",
)
```

```python
apply_pd_calibrator(
    persons,
    calibrator,
    pd_col="model_pd",
    out_col="calibrated_pd",
) -> pandas.DataFrame
```

```python
segment_calibration_report(
    persons,
    pd_col="model_pd",
    default_col="default",
    segment_cols=("city_name", "risk_archetype", "geo_cluster_id", "transfer_cluster_id"),
) -> dict
```

```python
pd_validation_report(
    persons,
    pd_col="model_pd",
    default_col="default",
    date_col=None,
    segment_cols=None,
) -> dict
```

## Metrics To Implement

Global metrics:

- Brier score.
- Log loss.
- AUC.
- Gini.
- KS statistic.
- Expected calibration error.
- Maximum calibration error.
- Mean PD.
- Observed default rate.
- PD/default rate gap.

Calibration table columns:

- `bucket`
- `pd_min`
- `pd_max`
- `count`
- `avg_pd`
- `observed_default_rate`
- `calibration_error`
- `abs_calibration_error`
- `defaults`

Segment metrics:

- Same global metrics per segment.
- Warning if segment is too small.
- Warning if one-class labels make AUC/KS undefined.

Drift metrics if `date_col` exists:

- Train period default rate.
- Validation period default rate.
- Train period mean PD.
- Validation period mean PD.
- PD drift.
- Default-rate drift.
- Optional population stability index by PD bucket.

## Calibration Methods

Support:

- `method="none"`: identity transform.
- `method="platt"`: logistic calibration over raw PD or logit(PD).
- `method="isotonic"`: `sklearn.isotonic.IsotonicRegression`.

Rules:

- Inputs must be clipped to `[eps, 1 - eps]` for logit transforms.
- Outputs must be clipped to `[0, 1]`.
- Calibrator must be serializable enough to store in memory during pipeline.
- Do not overwrite `model_pd` by default. Write to `calibrated_pd`.

## Agent API Additions

In `src/agents.py`, add:

```python
RiskAgentAPI.validate_pd(...)
RiskAgentAPI.calibrate_pd(...)
```

`validate_pd` should return `AgentResult` with:

- `ok`
- `summary`
- `data`
- `warnings`

Suggested summary examples:

- "PD validation complete: Brier=..., AUC=..., ECE=..."
- "PD validation found 3 weakly calibrated segments."

`calibrate_pd` should:

- require `default` labels.
- fit requested calibrator.
- add `calibrated_pd`.
- optionally set `model_pd = calibrated_pd` only if `replace=True`.

## Implementation Steps

1. Create `src/validation/` package.
2. Implement input validation helpers:
   - required columns exist.
   - PDs in `[0,1]`.
   - default labels are binary.
   - empty frames fail loudly.
3. Implement `calibration_table`.
4. Implement Brier, log loss, AUC, Gini, KS, ECE, MCE.
5. Implement segment-level report.
6. Implement optional date/out-of-sample report.
7. Implement calibrator fit/apply.
8. Wire exports in `src/__init__.py`.
9. Add `RiskAgentAPI.validate_pd`.
10. Add `RiskAgentAPI.calibrate_pd`.
11. Add tests.
12. Run full test suite.

## Tests

Add tests for:

- Perfectly calibrated synthetic data gives small calibration error.
- Calibration table counts sum to total rows.
- Brier score and log loss are finite.
- AUC and Gini are consistent.
- KS statistic is in `[0,1]`.
- Isotonic calibration outputs values in `[0,1]`.
- Platt calibration outputs values in `[0,1]`.
- Segment report handles small segments.
- Missing required columns raise a clear exception.
- Agent methods return JSON-safe data.

## Acceptance Criteria

- `api.validate_pd()` works after `api.run_pipeline()`.
- `api.calibrate_pd(method="isotonic")` adds `calibrated_pd`.
- Existing pipeline still works with `model_pd`.
- All tests pass.
- No downstream risk metric consumes uncalibrated PD unless explicitly chosen.

## Non-Goals

- Do not add GPU PD models in this task.
- Do not change the copula math.
- Do not replace `IndividualPDModel`; wrap and validate its output.

