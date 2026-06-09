"""
Stage 50 — Risk-adjusted metrics  (owner: Risk Analyst)
=======================================================

Compute per-borrower and segment-level risk-adjusted metrics (CoV, RAROC,
Sortino family, diversification ratio) from a fitted copula, plus the
RAROC-vs-Sortino divergence early-warning flags. This is the analytics payoff
stage that the whole chain feeds.

Inputs (artifacts):  persons_scored, corr_matrix
Outputs (artifacts): per_borrower_metrics, segment_metrics_<col>,
                     divergence_flags, portfolio_metrics (json)
"""

from __future__ import annotations

from typing import List, Optional

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "50_metrics"


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    copula_type: str = "clayton",
    lgd: float = 0.45,
    segment_cols: Optional[List[str]] = None,
    divergence_z: float = 1.5,
) -> StageResult:
    """
    Compute risk-adjusted metrics at borrower / segment / portfolio level.

    Parameters
    ----------
    copula_type : str
        Copula family to fit for the loss-covariance (defaults to Clayton).
    lgd : float
        Loss given default (scalar or, in code, a per-borrower array).
    segment_cols : list of str, optional
        Columns to aggregate metrics over (e.g. ["city_name", "risk_archetype"]).
        Aggregation uses block-sum loss covariance, never an average of ratios.
    divergence_z : float
        Z-threshold for flagging RAROC-vs-Sortino divergences (early warning).
    """
    store.require("persons_scored", "corr_matrix")
    persons = store.read_df("persons_scored")
    corr = store.read_array("corr_matrix")
    res = StageResult(stage=STAGE, ok=True)

    # Fit the copula used by the metric calculator.
    from src.copula_model import CopulaDefaultModel
    copula = CopulaDefaultModel(copula_type)
    copula.fit(persons["model_pd"].to_numpy(), corr)

    from src.risk_adjusted_metrics import RiskRatioCalculator
    calc = RiskRatioCalculator(copula, persons, lgd=lgd)

    per_borrower = calc.per_borrower()
    res.outputs.append(store.write_df("per_borrower_metrics", per_borrower))

    # Segment roll-ups (block-sum loss covariance — INV-6).
    seg_cols = segment_cols or [c for c in ("city_name", "risk_archetype")
                                if c in persons.columns]
    for col in seg_cols:
        seg = calc.by_segment(col)
        res.outputs.append(store.write_df(f"segment_metrics_{col}", seg))

    # RAROC vs Sortino-copula divergence — the primary early-warning output.
    from src.metric_comparison import MetricComparator
    comparator = MetricComparator(calc)
    flags = comparator.divergence_flags(z_threshold=divergence_z)
    res.outputs.append(store.write_df("divergence_flags", flags))
    res.metrics["n_divergences"] = int(len(flags))

    # Portfolio-level summary: every registered metric over all borrowers.
    port = calc.all_metrics(members=None)
    res.outputs.append(store.write_json("portfolio_metrics", dict(port)))

    res.metrics["n_borrowers"] = int(len(per_borrower))
    res.metrics["segments"] = ",".join(seg_cols) if seg_cols else "(none)"
    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s).summary())
