"""
Stage 30 — Copula fitting  (owner: Data Scientist · copula)
===========================================================

Fit the joint-default copula from the marginal PDs (stage 10) and the
correlation matrix (stage 20), then persist the joint-default probability
matrix and the fitted parameter(s). This is the dependence-structure heart of
the framework.

Inputs (artifacts):  persons_scored, corr_matrix
Outputs (artifacts): joint_default_matrix (npy), copula_params (json)
"""

from __future__ import annotations

import numpy as np

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "30_copula"


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    copula_type: str = "clayton",
    nu: float = 4.0,
    n_simulations: int = 500,
) -> StageResult:
    """
    Fit the copula and store the joint-default probability matrix.

    Parameters
    ----------
    copula_type : {"clayton", "gaussian", "student_t", "gumbel", "frank"}
        Copula family (Clayton is the default — lower-tail default clustering).
    nu : float
        Degrees of freedom for the Student-t copula (ignored otherwise).
    n_simulations : int
        Monte Carlo paths used when estimating the joint-default matrix.
    """
    store.require("persons_scored", "corr_matrix")
    persons = store.read_df("persons_scored")
    corr = store.read_array("corr_matrix")
    res = StageResult(stage=STAGE, ok=True)

    pds = persons["model_pd"].to_numpy()

    from src.copula_model import CopulaDefaultModel
    copula = CopulaDefaultModel(copula_type)
    copula.fit(pds, corr, nu=nu)

    # Joint default probability matrix P(D_i ∩ D_j).
    joint = copula.joint_default_probability_matrix(sample_size=n_simulations)
    res.outputs.append(store.write_array("joint_default_matrix", joint))
    res.metrics["avg_joint_pd"] = round(
        float(joint[~np.eye(len(joint), dtype=bool)].mean()), 6
    )

    params = {"copula_type": copula_type, "nu": nu, "n_obligors": int(len(pds))}
    # The fitted dependence parameter lives on copula.params.theta (the
    # CopulaParams record), not as a top-level copula.theta attribute.
    cop_params = getattr(copula, "params", None)
    theta = getattr(cop_params, "theta", None) if cop_params is not None else None
    if theta is not None:
        params["theta"] = float(theta)
        res.metrics["theta"] = round(params["theta"], 4)
    res.outputs.append(store.write_json("copula_params", params))
    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s).summary())
