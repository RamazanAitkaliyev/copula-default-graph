"""
ETL Pipelines  (pipelines/)
===========================

A thin orchestration layer that turns the flat ``src/`` modules into an
**ETL-style sequence of independent stages**. Each stage:

  * reads its inputs from a shared :class:`ArtifactStore` (on disk),
  * runs ONE well-defined piece of the credit-risk computation, and
  * writes its outputs back to the store.

Because stages communicate only through persisted artifacts (CSV / parquet /
npy / json), they are **decoupled**: different people / teams can own, run, and
re-run different stages without touching each other's code, and a stage can be
re-executed in isolation as long as its upstream artifacts exist.

STAGE MAP (who owns what — see ROLES.md)
----------------------------------------
    Stage                         Owner             Module(s) used
    ---------------------------   ---------------   ----------------------------
    00  ingest                    Data Engineer     loaders / data_generator
    10  pd_scoring                ML Engineer       pd_model
    20  graph_features            Data Scientist    graph_features (+ spectrum
                                  (graph)             denoising — arpym #2)
    30  copula                    Data Scientist    copula / factor copula
                                  (copula)
    40  rating_transitions        Risk Analyst /    rating_engine +
                                  Credit Risk         credit_transitions (arpym #1)
    50  risk_metrics              Risk Analyst      risk_adjusted_metrics,
                                                      risk_metrics, metric_comparison

DATA FLOW (artifacts on disk)
-----------------------------
    00 ─persons,transactions─▶ 10 ─persons(+model_pd)─▶ 20 ─corr_matrix─▶ 30
                                                          │                 │
                                            (denoised)    ▼                 ▼
                                                       40 rating_transitions │
                                                          │                 │
                                                          └──────▶ 50 ◀──────┘
                                                                 risk_metrics

USAGE
-----
Run the whole chain:

    from pipelines import run_all
    store = run_all(seed=42, denoise=True)        # writes output/etl/*

Run a single stage (others' artifacts must already exist):

    from pipelines import ArtifactStore
    from pipelines.stage_20_graph import run as run_graph
    store = ArtifactStore("output/etl")
    run_graph(store, denoise=True)                # re-do ONLY the graph stage

Each ``stage_*.run(store, **opts)`` is also runnable from its matching notebook
in ``notebooks/``.
"""

from .artifacts import ArtifactStore, StageResult
from .runner import run_all, STAGES

__all__ = ["ArtifactStore", "StageResult", "run_all", "STAGES"]
