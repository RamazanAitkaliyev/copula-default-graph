# Plan 08: Factor-Loading Estimation & Conditional FP Family

## Status

**Implemented (2026-06-06).** Two ARPM estimators ported into `src/`, plus the
agent API and the optional ETL stage. The only remaining follow-up is a fuller
standalone loading-calibration diagnostic (see "Remaining Work" §2).

Already done:

- `src/low_rank_corr.py` — `low_rank_diag_conditional_corr`, `conditional_pc`,
  `fit_factor_loadings`, `LowRankResult` (arpym
  `estimation.low_rank_diag_conditional_corr` port).
- `src/conditional_fp.py` — `crisp_fp`, `conditional_fp`, `quantile_smooth`,
  `effective_scenarios` (arpym `estimation.conditional_fp` / `crisp_fp` ports).
- `FlexibleProbsCalibrator(weighting_method="conditional_fp")` path wired in
  `src/flexible_probs.py`.
- Exports in `src/__init__.py`; tests `test_low_rank_corr`, `test_conditional_fp`
  in `test_copula_framework.py`.
- **Agent API (done):** `RiskAgentAPI.fit_factor_loadings(k_factors, denoise,
  apply)` and `RiskAgentAPI.regime_weights(current_stress, method, alpha)` in
  `src/agents.py`. `fit_factor_loadings` is diagnostic by default and only
  refits the live copula as a `MultiFactorCopula` when `apply=True`.
  Covered by `test_agent_loadings_and_regime` (suite at 51 tests, incl. a
  real-data ETL pipeline regression test and the `_safe` bool-coercion fix).

Still open (the follow-up work this plan briefs):

- A standalone copula-calibration diagnostic module that compares fitted vs
  hand-set loadings in more depth (the agent method reports the headline stats;
  a fuller per-factor / per-segment diagnostic table is still useful).
- An optional ETL pipeline stage that fits loadings from the graph correlation
  (`pipelines/stage_25_loadings.py`).

## Objective

Turn two assumptions into data-driven estimates:

1. **Factor-copula loadings.** `MultiFactorCopula.fit(pds, factor_matrix, betas)`
   currently takes `betas` that are SET by hand (equal loadings = "equally
   important"). `low_rank_diag_conditional_corr` fits an `(n, k)` loading matrix
   from a target correlation matrix (e.g. the transaction-graph correlation), so
   the factor structure is estimated rather than assumed.

2. **Regime conditioning.** `flexible_probs.py` weighted history with a Gaussian
   kernel (a smooth approximation). `conditional_fp` instead matches the
   conditional mean AND variance exactly via entropy pooling — the rigorous
   ARPM definition — and `crisp_fp` gives the hard-window variant.

## Current Code To Read First

- `src/low_rank_corr.py`
- `src/conditional_fp.py`
- `src/relative_entropy.py`
- `src/flexible_probs.py`
- `src/multi_factor_copula.py`
- `src/factor_copula.py`
- `src/graph_features.py`
- `src/agents.py`
- `Python/arpym/arpym/estimation/low_rank_diag_conditional_corr.py`
- `Python/arpym/arpym/estimation/conditional_fp.py`
- `METHODOLOGY.md`

## Math Summary

Low-rank + diagonal correlation:

```text
C ≈ beta @ beta.T + diag(1 - rowSumSq(beta)),   beta shape (n, k)
sum_k beta_i,k^2 < 1        (positive idiosyncratic variance — copula-ready)
```

Alternating projection: eigendecompose the systematic part, keep top k
directions, scale over-length rows below unit norm, rebuild with unit diagonal,
repeat to convergence. Optional linear constraint `D @ beta = 0` removes a known
direction (e.g. a market-wide mode) from the systematic factors.

Conditional FP:

```text
p_crisp  = crisp window of mass alpha around z_star
m, s2    = conditional mean & variance under p_crisp
p_cond   = argmin KL(p || p_prior)  s.t.  E[z]=m,  E[z^2]=m^2+s2
```

## Public API (already implemented)

```python
low_rank_diag_conditional_corr(c2, d=None, k_bar=1, eta=0.01, gamma=0.1, max_iter=1000) -> LowRankResult
fit_factor_loadings(corr, k_factors=1, constraint=None, non_negative=True) -> numpy.ndarray   # (n, k)
conditional_pc(sigma2, d) -> (lam2_d, e_d)

crisp_fp(z, z_star, alpha) -> (p, z_lb, z_ub)
conditional_fp(z, z_star, alpha, p_prior=None) -> p
quantile_smooth(c_bar, x, p=None, h=None) -> q
effective_scenarios(p) -> float
```

Loading-sign note: `fit_factor_loadings(non_negative=True)` returns `|beta|`
because `MultiFactorCopula` (Vasicek) requires non-negative loadings; the within-
factor correlation `beta_i·beta_j` is preserved for same-factor borrowers. Set
`non_negative=False` to keep signed loadings for the full `beta beta.T`
reconstruction.

## Integration Pattern (already verified end-to-end)

```python
from src.low_rank_corr import fit_factor_loadings
from src.multi_factor_copula import MultiFactorCopula

corr  = graph.get_correlation_matrix(denoise=True)     # transaction-graph corr
beta  = fit_factor_loadings(corr, k_factors=2)          # FITTED loadings (n, 2)
mfc   = MultiFactorCopula().fit(pds, factor_matrix, betas=beta)
rate  = mfc.simulate_default_rate(2000, seed=0)
```

```python
from src.flexible_probs import FlexibleProbsCalibrator

calib = FlexibleProbsCalibrator(weighting_method="conditional_fp", conditional_alpha=0.25)
calib.fit(stress_history)
regime_copula = calib.calibrate(current_stress=0.8, base_corr_matrix=corr)
```

## Remaining Work

### 1. Agent API additions (`src/agents.py`) — DONE

```python
RiskAgentAPI.fit_factor_loadings(k_factors=2, denoise=True, apply=False)
RiskAgentAPI.regime_weights(current_stress=None, method="conditional_fp", alpha=0.25)
```

`fit_factor_loadings` builds the graph correlation (optionally denoised), fits
`(n, k)` loadings, reports summary stats + implied within-factor correlation,
only refits the live copula (as a `MultiFactorCopula`) when `apply=True`, and
returns a JSON-safe `AgentResult`. `regime_weights` reports the effective number
of scenarios and regime θ under either the conditional-FP or kernel estimator.
Covered by `test_agent_loadings_and_regime`.

### 3. Optional ETL stage — DONE

`pipelines/stage_25_loadings.py` reads `corr_matrix`, writes `factor_loadings`
(npy) + `loading_diagnostics` (json), following the stage contract
(`ArtifactStore`, `StageResult`, `timed_stage`). It is off the default chain;
enable with `run_all(with_loadings=True)`. A `MultiFactorCopula` can consume the
artifact via `betas=store.read_array("factor_loadings")`. Covered by
`test_pipeline_stage_25_loadings`.

### 2. Loading-calibration diagnostic — STILL OPEN

The agent method and stage 25 report the headline stats (avg loading, max Σβ²,
off-diagonal Frobenius vs the input correlation). A fuller standalone diagnostic
is still worthwhile:

- Frobenius distance between `beta_fit beta_fit.T` and the *configured* hand-set
  implied correlation (compare fitted vs assumed, not just vs the input).
- Per-factor average loading and per-segment breakdown.
- Effective number of factors actually used (rows with non-trivial loading).
- Warnings for unstable fits (`distance` not converged, `constraint_ok=False`).

## Tests

Already present:

- `test_low_rank_corr`: loading shape, unit diagonal preserved, row-norm
  constraint, reconstruction error bound, `k_bar` rank-bound error, end-to-end
  `MultiFactorCopula` simulation from fitted loadings.
- `test_conditional_fp`: crisp window validity, conditional mean match (entropy
  pool), smoothness, effective-scenario drop, multi-target shape, and the
  `FlexibleProbsCalibrator(method="conditional_fp")` stress > calm theta check.

Add with the follow-up work:

- Agent `fit_factor_loadings` returns JSON-safe data and does not mutate the live
  model unless `apply=True`.
- Loading diagnostic distance is finite and decreases when `k_factors` rises.
- Pipeline stage 25 round-trips `factor_loadings` through the artifact store.

## Acceptance Criteria

- Risk team can fit factor-copula loadings from observed correlation instead of
  configuring them by hand.
- Regime conditioning can use the rigorous conditional-FP estimator, not only the
  kernel approximation.
- Existing `MultiFactorCopula`, `FactorCopula`, and `FlexibleProbsCalibrator`
  APIs continue to work unchanged (both paths remain available).
- All tests pass.

## Non-Goals

- Do not remove the hand-set-loadings or kernel-weighting paths; both stay as
  defaults.
- Do not require an external ARPyM dependency (everything is a self-contained
  numpy/scipy port).
- Do not rebuild the copula simulation engine.
```
