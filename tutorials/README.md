# Tutorials — by role

Step-by-step, runnable walkthroughs. Each follows the data as it flows down the
layered architecture (see `../ARCHITECTURE.md`). Do them in order, or jump to
yours.

| # | Tutorial | Role | You produce | You consume |
|---|---|---|---|---|
| 01 | [Data Engineer](01_data_engineer.md) | 🛠️ ingestion | clean persons + transactions | raw CSV/parquet |
| 02 | [ML Engineer](02_ml_engineer.md) | 🤖 PD models | `model_pd ∈ [0,1]` | persons + `default` label |
| 03 | [Data Scientist](03_data_scientist.md) | 📊 graphs, clusters, copula | a fitted copula + cluster columns | persons + transactions + PD |
| 04 | [Risk Analyst](04_risk_analyst.md) | 🎯 metrics, stress, ratings | risk numbers the bank acts on | the fitted copula + EAD/LGD |

**Cross-cutting reading:**
- `../ARCHITECTURE.md` — the big picture (layers, design decisions).
- `../METHODOLOGY.md` — every formula with derivation (for quants / validation).
- `../ROLES.md` — who owns which module + boundary contracts.
- `../CAPABILITIES.md` — machine-readable API catalog (for AI agents).
- `../RECIPES.md` — copy-paste snippets.
- `../docs/architecture.html` — the visual, shareable architecture diagram.
- `../docs/risk_department.html` — the full descriptive guide for the risk
  department: every pipeline stage, every module + the logic behind it, the
  seven metrics decoded, and the standing risk-team workflows.

**Verify the whole thing:**
```bash
python test_copula_framework.py     # All 41 tests passed.
python demo_clusters.py             # end-to-end cluster pipeline → output/
python main.py                      # the base 13-step pipeline
```
