# Implementation Plans Index

Date: 2026-06-06

This folder contains agent-ready implementation plans for the selected ARPM
theory integrations into the Copula Default Graph risk platform.

## Already implemented (ARPM ports landed in `src/`)

These self-contained numpy/scipy ports are done and tested (suite at 51 tests);
several plans below depend on them:

| Module | ARPM source | Tested by |
|---|---|---|
| `src/relative_entropy.py` (`min_rel_entropy_sp`) | `views.min_rel_entropy_sp` | `test_relative_entropy` |
| `src/credit_transitions.py` (`fit_trans_matrix_credit`, `cohort_arrays_from_events`) | `estimation.fit_trans_matrix_credit` | `test_credit_transitions` |
| `src/spectrum.py` (`spectrum_shrink`, `denoise_correlation`) | `estimation.spectrum_shrink` | `test_spectrum_shrinkage` |
| `src/conditional_fp.py` (`conditional_fp`, `crisp_fp`) | `estimation.conditional_fp` / `crisp_fp` | `test_conditional_fp` |
| `src/low_rank_corr.py` (`low_rank_diag_conditional_corr`, `fit_factor_loadings`) | `estimation.low_rank_diag_conditional_corr` | `test_low_rank_corr` |
| `src/dependence.py` (`schweizer_wolff`, `copula_invariance_test`) | `statistics.schweizer_wolff` (bug fixed) | `test_dependence_measures` |
| `src/copula_calibration.py` (`calibrate_copula`, `empirical_dependence_measures`) | Plan 07 (uses `schweizer_wolff`) | `test_copula_calibration` |

A modular ETL layer (`pipelines/`) and per-stage notebooks (`notebooks/`) wrap
these into an ownable, re-runnable workflow. See `CLAUDE.md`.

## Files

1. `02_pd_calibration_backtesting_plan.md`
   - Implement PD calibration, backtesting, segment validation, and agent API.

2. `03_ead_lgd_models_plan.md`
   - Implement proper EAD and LGD models and wire borrower-level loss inputs
     through all risk metrics.

3. `04_entropy_stress_views_plan.md` — *primitive done; engine layer open*
   - Implement ARPM-style entropy stress views using minimum relative entropy.
   - Builds on `src/relative_entropy.py` (implemented).

4. `06_lifetime_pd_markov_plan.md` — *transition estimator done; Markov/IFRS9 open*
   - Implement Markov delinquency states, lifetime PD curves, and IFRS9-style
     staging helpers.
   - Builds on `src/credit_transitions.py` (implemented).

5. `07_copula_parameter_calibration_plan.md` — *first version done; deeper fitters open*
   - Empirical copula calibration from observed default data.
   - `src/dependence.py` + `src/copula_calibration.py` implemented (empirical
     measures, τ→parameter mapping, family comparison, agent method). Remaining:
     per-family MLE fitters and multi-factor beta-by-segment calibration.

6. `08_factor_loading_fp_family_plan.md` — *done; deeper diagnostic open*
   - Factor-copula loading estimation + conditional-FP family.
   - Ports, agent API, and optional ETL stage 25 implemented; remaining work is
     a fuller fitted-vs-configured loading diagnostic.

## Recommended Order

1. PD calibration and backtesting.
2. EAD and LGD models.
3. Lifetime PD via Markov delinquency states (transition estimator already done).
4. Entropy stress views (primitive already done).
5. Factor-loading estimation & FP family follow-up (core already done — Plan 08).
6. Copula parameter calibration.

## Agent Instructions

Each plan is designed to be passed to an implementation agent as a standalone
task brief. Agents should:

- Read `AGENTS.md`, `ARCHITECTURE.md`, `METHODOLOGY.md`, and `CAPABILITIES.md`
  before editing code.
- Preserve existing public APIs unless the plan explicitly says otherwise.
- Add tests with every new module.
- Keep all existing tests passing.
- Avoid dense `n x n` matrices for scalable paths.
- Preserve the invariant that segment metrics aggregate primitives and block-sum
  loss covariance, never average borrower-level ratios.

