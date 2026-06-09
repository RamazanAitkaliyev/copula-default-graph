"""
Stage 40 — Ratings & credit transitions  (owner: Risk Analyst / Credit Risk)
============================================================================

Assign credit ratings to borrowers and attach a transition matrix. This stage
hosts arpym Tier-1 concept #1: the **rigorous continuous-time transition-matrix
estimator** (``credit_transitions.fit_trans_matrix_credit``), used when cohort
migration data is supplied; otherwise it uses the engine's baseline matrix.

Two ways to provide cohort data for estimation:
  * a migration-event artifact ``migration_events`` (a tidy table —
    period / from_state / to_state / count), OR
  * pre-built cohort arrays passed directly to ``RatingEngine.from_cohort_data``
    in your own driver code.

Inputs (artifacts):  persons_scored, [migration_events]
Outputs (artifacts): transition_matrix (npy), ratings (persons + rating cols),
                     rating_distribution (json)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "40_transitions"


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    estimate_from_events: bool = True,
    tau_hl_years: Optional[float] = None,
    pd_col: str = "model_pd",
) -> StageResult:
    """
    Build ratings and the transition matrix.

    Parameters
    ----------
    estimate_from_events : bool, default True
        If a ``migration_events`` artifact exists, estimate the annual transition
        matrix from it with the rigorous generator-based estimator (arpym #1).
        If absent, fall back to the ``RatingEngine`` baseline matrix.
    tau_hl_years : float, optional
        Exponential half-life (years) for time-weighting older migration periods.
    pd_col : str
        PD column used to bucket borrowers into ratings.
    """
    store.require("persons_scored")
    persons = store.read_df("persons_scored")
    res = StageResult(stage=STAGE, ok=True)

    from src.rating_engine import RatingEngine, N_RATINGS

    estimated = False
    has_events = store.exists("migration_events.parquet") or store.exists("migration_events.csv")
    if estimate_from_events and has_events:
        from src.credit_transitions import cohort_arrays_from_events
        events = store.read_df("migration_events")
        count_col = "count" if "count" in events.columns else None
        dates, n_oblig, n_cum = cohort_arrays_from_events(
            events, n_ratings=N_RATINGS, count_col=count_col,
        )
        engine = RatingEngine.from_cohort_data(
            dates, n_oblig, n_cum, tau_hl_years=tau_hl_years,
        )
        estimated = True
    else:
        engine = RatingEngine()  # industry-standard baseline transition matrix

    engine.fit(persons, pd_col=pd_col)

    res.outputs.append(store.write_array("transition_matrix", engine.transition_annual))
    res.outputs.append(store.write_df("ratings", engine.summary_df()))

    dist = engine.portfolio_distribution()
    res.outputs.append(store.write_json("rating_distribution", {
        "counts": dist.counts, "fractions": dist.fractions,
        "weighted_avg_pd": dist.weighted_avg_pd,
        "migration_risk_score": dist.migration_risk_score,
    }))

    res.metrics["transition_estimated_from_data"] = estimated
    res.metrics["weighted_avg_pd"] = round(float(dist.weighted_avg_pd), 5)
    res.metrics["migration_risk"] = round(float(dist.migration_risk_score), 4)
    # Monotonicity sanity: default-column PD should be non-decreasing by rating.
    pd_col_vals = engine.transition_annual[:N_RATINGS - 1, -1]
    res.metrics["pd_monotone"] = bool(np.all(np.diff(pd_col_vals) >= -1e-9))
    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s).summary())
