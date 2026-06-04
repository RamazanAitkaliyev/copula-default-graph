"""
src.ml — ML Engineer role view (probability-of-default models).

A NAVIGATION facade re-exporting the flat modules an ML Engineer owns
(`from src.ml import IndividualPDModel`). Implementation stays in `src/*.py`;
this only groups by role. See ROLES.md.

Owns: borrower-level PD — training, scoring, calibration, explainability.
Contract provided downstream: persons['model_pd'] in [0, 1].
"""
from ..pd_model import IndividualPDModel, PDModelEnsemble
from ..structural_pd import StructuralPDModel, MertonParams, compute_proxy_merton_pd

__all__ = [
    "IndividualPDModel", "PDModelEnsemble",
    "StructuralPDModel", "MertonParams", "compute_proxy_merton_pd",
]
