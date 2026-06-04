"""
src.copula — Data Scientist role view: dependence (copula) models.

A NAVIGATION facade re-exporting the copula models (`from src.copula import
MultiFactorCopula`). Implementation stays in `src/*.py`. See ROLES.md.

Owns: turning marginal PDs + correlation structure into JOINT default
probabilities.
  - CopulaDefaultModel  — 5 copula types, dense, for <= ~20k names.
  - FactorCopula        — Vasicek single-factor, scales to 10M+.
  - MultiFactorCopula   — K systematic factors (geo ⟂ transfer), 10M+.
  - FlexibleProbsCalibrator — regime-aware copula reweighting.

Contract provided downstream: a fitted object exposing marginal_pds, is_fitted,
and joint_default_probability_block(idx) (or the legacy full-matrix form).
"""
from ..copula_model import CopulaDefaultModel, CopulaParams, compare_copulas
from ..factor_copula import FactorCopula, FactorCopulaParams, build_factor_id
from ..multi_factor_copula import MultiFactorCopula, MultiFactorCopulaParams
from ..flexible_probs import (
    FlexibleProbsCalibrator,
    RegimeAdjustedCopula,
    RegimeState,
    build_calibrator_from_portfolio,
)

__all__ = [
    "CopulaDefaultModel", "CopulaParams", "compare_copulas",
    "FactorCopula", "FactorCopulaParams", "build_factor_id",
    "MultiFactorCopula", "MultiFactorCopulaParams",
    "FlexibleProbsCalibrator", "RegimeAdjustedCopula", "RegimeState",
    "build_calibrator_from_portfolio",
]
