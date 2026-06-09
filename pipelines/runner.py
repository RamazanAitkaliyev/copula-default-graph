"""
Pipeline runner  (pipelines/runner.py)
======================================

Chains the ETL stages in order and reports a per-stage summary. Each stage is
independent (communicates only via the :class:`ArtifactStore`), so the runner is
deliberately thin: it just calls them in sequence and stops early if a stage
fails (so a downstream stage never reads a half-written artifact).

The same stages can be run individually from their modules or notebooks; this
module only exists for the "run the whole thing" convenience path.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

from .artifacts import ArtifactStore, StageResult
from .stage_00_ingest import run as run_ingest
from .stage_10_pd import run as run_pd
from .stage_20_graph import run as run_graph
from .stage_30_copula import run as run_copula
from .stage_40_transitions import run as run_transitions
from .stage_50_metrics import run as run_metrics
from .stage_25_loadings import run as run_loadings

logger = logging.getLogger(__name__)

# Ordered registry: (stage name, callable). Owners pick their stage by name.
STAGES: List[tuple[str, Callable[..., StageResult]]] = [
    ("00_ingest", run_ingest),
    ("10_pd", run_pd),
    ("20_graph", run_graph),
    ("30_copula", run_copula),
    ("40_transitions", run_transitions),
    ("50_metrics", run_metrics),
]

# Optional stage 25 (factor-loading estimation) sits between 20 and 30. It is
# off the default chain because it writes artifacts the default copula path does
# not consume; enable it via run_all(with_loadings=True).
OPTIONAL_LOADINGS_STAGE = ("25_loadings", run_loadings)


def run_all(
    root: str = "output/etl",
    seed: int = 42,
    denoise: bool = False,
    with_loadings: bool = False,
    stage_opts: Dict[str, dict] | None = None,
    stop_on_error: bool = True,
    verbose: bool = True,
) -> ArtifactStore:
    """
    Run the full ETL chain and return the populated artifact store.

    Parameters
    ----------
    root : str
        Artifact store directory.
    seed : int
        Seed forwarded to the ingest stage.
    denoise : bool
        Forwarded to the graph stage — enable Marčenko-Pastur denoising (#2).
    with_loadings : bool
        Insert the optional stage 25 (factor-loading estimation) after the graph
        stage. It writes a ``factor_loadings`` artifact a MultiFactorCopula can
        consume; the default copula path does not use it.
    stage_opts : dict, optional
        Per-stage keyword overrides, keyed by stage name, e.g.
        ``{"30_copula": {"copula_type": "student_t", "nu": 6},
           "40_transitions": {"tau_hl_years": 2.0}}``.
    stop_on_error : bool
        Stop the chain if a stage fails (default True). The store still holds
        whatever earlier stages produced.
    verbose : bool
        Print each stage's summary as it completes.

    Returns
    -------
    ArtifactStore with all stage artifacts written.
    """
    store = ArtifactStore(root)
    opts = stage_opts or {}

    # Inject the top-level convenience flags into the stages that own them.
    opts.setdefault("00_ingest", {}).setdefault("seed", seed)
    opts.setdefault("20_graph", {}).setdefault("denoise", denoise)

    # Build the effective stage list, inserting stage 25 after 20 if requested.
    stages = list(STAGES)
    if with_loadings:
        idx = next(i for i, (n, _) in enumerate(stages) if n == "20_graph") + 1
        stages.insert(idx, OPTIONAL_LOADINGS_STAGE)

    results: List[StageResult] = []
    for name, fn in stages:
        result = fn(store, **opts.get(name, {}))
        results.append(result)
        if verbose:
            print(result.summary())
        if not result.ok and stop_on_error:
            print(f"\n✗ pipeline halted at stage {name}")
            break
    else:
        if verbose:
            print("\n✓ pipeline complete — artifacts in", store.root)

    return store


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_all(denoise=True)
