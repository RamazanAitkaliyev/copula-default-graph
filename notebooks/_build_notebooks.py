"""
Generate the modular per-stage notebooks (notebooks/*.ipynb).

Run from the project root:
    python notebooks/_build_notebooks.py

Each ETL stage gets its own notebook so different owners can work independently.
Notebooks are intentionally thin: they import the corresponding
``pipelines.stage_*`` function, run it against the shared ArtifactStore, and
inspect the artifacts it produced. Regenerate any time the stage API changes.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# Shared setup cell — every stage notebook starts the same way: make the project
# root importable and open the shared artifact store.
SETUP = """\
# --- setup: make the project root importable + open the shared artifact store ---
import sys, os
ROOT = os.path.abspath("..")          # notebooks/ lives one level below the project root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipelines import ArtifactStore
store = ArtifactStore("output/etl")   # the SAME store every stage reads/writes
print("artifact store:", store.root)
print("existing artifacts:", store.list())
"""


def stage_nb(num, name, owner, title, blurb, inputs, outputs, run_code, inspect_code):
    cells = [
        md(f"# Stage {num} — {title}\n\n"
           f"**Owner:** {owner}  \n"
           f"**Pipeline stage:** `pipelines.stage_{name}`\n\n"
           f"{blurb}\n\n"
           f"**Reads:** {inputs}  \n"
           f"**Writes:** {outputs}\n\n"
           "> This notebook is *modular*: it only needs its upstream artifacts to exist in the\n"
           "> shared store. You can re-run just this stage without touching the others.\n"),
        md("## 1. Setup"),
        code(SETUP),
        md("## 2. Run this stage"),
        code(run_code),
        md("## 3. Inspect what this stage produced"),
        code(inspect_code),
    ]
    path = NB_DIR / f"{name}.ipynb"
    path.write_text(json.dumps(notebook(cells), indent=1))
    print("wrote", path.name)


# ── 00 ingest ─────────────────────────────────────────────────────────────────
stage_nb(
    "00", "00_ingest", "Data Engineer", "Ingest persons & transactions",
    "Produce the two canonical tables every later stage consumes. Synthetic by "
    "default; point at real files via `persons_source` / `transactions_source` "
    "+ a `ColumnMapping` to ingest production data instead.",
    "— (source stage)", "`persons`, `transactions`",
    """\
from pipelines.stage_00_ingest import run as run_ingest

# Synthetic mode (default). For real data:
#   from src.loaders import ColumnMapping
#   run_ingest(store, persons_source="data/persons.csv",
#              transactions_source="data/tx.csv", mapping=ColumnMapping(...))
result = run_ingest(store, seed=42)
print(result.summary())
""",
    """\
persons = store.read_df("persons")
transactions = store.read_df("transactions")
print(persons.shape, transactions.shape)
display(persons.head())
display(transactions.head())
""",
)

# ── 10 pd ─────────────────────────────────────────────────────────────────────
stage_nb(
    "10", "10_pd", "ML Engineer", "Individual PD scoring",
    "Merge neighbour-risk features, then train the gradient-boosting PD model and "
    "write `model_pd` for every borrower. If you already have PDs, pass "
    "`skip_training=True, pd_col=...`.",
    "`persons`, `transactions`",
    "`persons_scored`, `pd_feature_importance`, `pd_metrics`",
    """\
from pipelines.stage_10_pd import run as run_pd

result = run_pd(store, model_type="gradient_boosting")
print(result.summary())
""",
    """\
scored = store.read_df("persons_scored")
print("PD range:", scored["model_pd"].min(), "→", scored["model_pd"].max())
display(store.read_df("pd_feature_importance"))
print("metrics:", store.read_json("pd_metrics"))
""",
)

# ── 20 graph ──────────────────────────────────────────────────────────────────
stage_nb(
    "20", "20_graph", "Data Scientist · graph", "Graph features & correlation (with MP denoising)",
    "Build the transaction graph and derive the borrower-borrower correlation "
    "matrix the copula consumes. **arpym concept #2 lives here**: set "
    "`denoise=True` to apply Marčenko-Pastur spectrum shrinkage before the PSD "
    "projection — better conditioning and out-of-sample stability.",
    "`persons_scored`, `transactions`",
    "`corr_matrix`, `network_stats`, `spectrum_diagnostics` (when denoising)",
    """\
from pipelines.stage_20_graph import run as run_graph

# Toggle denoise to compare. With denoising, spectrum_diagnostics records the
# conditioning improvement vs the raw matrix.
result = run_graph(store, denoise=True, denoise_method="mp_edge")
print(result.summary())
""",
    """\
import numpy as np
corr = store.read_array("corr_matrix")
print("corr shape:", corr.shape, "| PSD:", np.linalg.eigvalsh(corr).min() > -1e-9)
print("network_stats:", store.read_json("network_stats"))
if store.exists("spectrum_diagnostics"):
    print("spectrum_diagnostics:", store.read_json("spectrum_diagnostics"))
""",
)

# ── 30 copula ─────────────────────────────────────────────────────────────────
stage_nb(
    "30", "30_copula", "Data Scientist · copula", "Copula fitting (joint defaults)",
    "Fit the joint-default copula from the marginal PDs and the correlation "
    "matrix, then persist the joint-default probability matrix and parameters. "
    "Clayton (lower-tail clustering) is the default; try `student_t` for "
    "symmetric tail dependence.",
    "`persons_scored`, `corr_matrix`",
    "`joint_default_matrix`, `copula_params`",
    """\
from pipelines.stage_30_copula import run as run_copula

result = run_copula(store, copula_type="clayton", n_simulations=500)
print(result.summary())
""",
    """\
import numpy as np
J = store.read_array("joint_default_matrix")
off = J[~np.eye(len(J), dtype=bool)]
print("joint-default matrix shape:", J.shape, "| avg off-diag P(D_i∩D_j):", off.mean())
print("copula_params:", store.read_json("copula_params"))
""",
)

# ── 40 transitions ────────────────────────────────────────────────────────────
stage_nb(
    "40", "40_transitions", "Risk Analyst / Credit Risk", "Ratings & credit transitions",
    "Assign credit ratings and attach a transition matrix. **arpym concept #1 "
    "lives here**: if a `migration_events` artifact exists, the annual transition "
    "matrix is estimated with the rigorous continuous-time generator + "
    "entropy-regularised monotonicity estimator (`fit_trans_matrix_credit`); "
    "otherwise the engine's baseline matrix is used.",
    "`persons_scored`, optional `migration_events`",
    "`transition_matrix`, `ratings`, `rating_distribution`",
    """\
from pipelines.stage_40_transitions import run as run_transitions

# OPTIONAL — supply real cohort migration events to trigger the arpym #1 estimator:
#   import pandas as pd
#   events = pd.DataFrame({"period":[...], "from_state":[...],
#                          "to_state":[...], "count":[...]})
#   store.write_df("migration_events", events)
result = run_transitions(store, estimate_from_events=True, tau_hl_years=2.0)
print(result.summary())
""",
    """\
import numpy as np
P = store.read_array("transition_matrix")
print("transition matrix rows sum to 1:", np.allclose(P.sum(axis=1), 1.0))
print("default-column PD by rating:", np.round(P[:7, -1], 4))
display(store.read_df("ratings").head())
print("rating_distribution:", store.read_json("rating_distribution"))
""",
)

# ── 50 metrics ────────────────────────────────────────────────────────────────
stage_nb(
    "50", "50_metrics", "Risk Analyst", "Risk-adjusted metrics & divergence flags",
    "Compute per-borrower / segment / portfolio risk-adjusted metrics (CoV, "
    "RAROC, Sortino family, diversification ratio) and the RAROC-vs-Sortino "
    "divergence early-warning flags. Segment roll-ups use block-sum loss "
    "covariance, never an average of per-borrower ratios.",
    "`persons_scored`, `corr_matrix`",
    "`per_borrower_metrics`, `segment_metrics_*`, `divergence_flags`, `portfolio_metrics`",
    """\
from pipelines.stage_50_metrics import run as run_metrics

result = run_metrics(store, copula_type="clayton", lgd=0.45,
                     segment_cols=["city_name", "risk_archetype"])
print(result.summary())
""",
    """\
display(store.read_df("per_borrower_metrics").head())
display(store.read_df("segment_metrics_city_name"))
print("portfolio_metrics:", store.read_json("portfolio_metrics"))
flags = store.read_df("divergence_flags")
print("RAROC-vs-Sortino divergence flags:", len(flags))
display(flags.head())
""",
)


# ── orchestration notebook ────────────────────────────────────────────────────
orch = [
    md("# ETL Orchestration — run the whole chain\n\n"
       "Runs all six stages in order via `pipelines.run_all`. Each stage is still "
       "independent (they communicate only through the shared artifact store); this "
       "notebook is just the convenience path. For working on a single stage, open "
       "its own notebook (`00_ingest.ipynb` … `50_metrics.ipynb`).\n\n"
       "**Stage map**\n\n"
       "| Stage | Owner | Concept added |\n"
       "|---|---|---|\n"
       "| 00 ingest | Data Engineer | — |\n"
       "| 10 pd | ML Engineer | — |\n"
       "| 20 graph | Data Scientist · graph | **arpym #2: MP spectrum denoising** |\n"
       "| 30 copula | Data Scientist · copula | — |\n"
       "| 40 transitions | Risk Analyst / Credit Risk | **arpym #1: generator-based estimator** |\n"
       "| 50 metrics | Risk Analyst | — |\n"),
    md("## 1. Setup"),
    code(SETUP),
    md("## 2. Run the full pipeline\n\n"
       "`denoise=True` turns on the Marčenko-Pastur denoising in stage 20. Use "
       "`stage_opts` to pass per-stage overrides."),
    code("""\
from pipelines import run_all

store = run_all(
    root="output/etl",
    seed=42,
    denoise=True,
    stage_opts={
        "30_copula": {"copula_type": "clayton"},
        "40_transitions": {"tau_hl_years": 2.0},
        "50_metrics": {"segment_cols": ["city_name", "risk_archetype"]},
    },
    verbose=True,
)
"""),
    md("## 3. All artifacts produced"),
    code("""\
for a in store.list():
    print(a)
"""),
]
(NB_DIR / "99_run_all.ipynb").write_text(json.dumps(notebook(orch), indent=1))
print("wrote 99_run_all.ipynb")

print("\nAll notebooks generated in", NB_DIR)
