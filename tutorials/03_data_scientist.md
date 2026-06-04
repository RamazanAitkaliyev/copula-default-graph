# Tutorial 03 — Data Scientist

**Your job:** model how borrowers are **correlated** — build the transaction
graph, the geo and transfer clusters, detect the anchor/dependent pattern, and
fit the copula that turns PDs + correlation into joint defaults.

**Your modules:** `src/graph_features.py`, `src/geo_clusters.py`,
`src/transfer_clusters.py`, the copulas in `src/copula/`
(import via `from src.analytics import ...` and `from src.copula import ...`).

**The math:** see `METHODOLOGY.md §2–4` (copulas, Vasicek, multi-factor).

---

## Step 1 — the transaction graph

```python
from src.analytics import TransactionGraph
graph = TransactionGraph(transactions, persons)     # sparse CSR, scales
print(graph.get_network_stats())                     # nodes, edges, components, clustering
```

## Step 2 — geo clusters (DBSCAN on lat/lon)

```python
from src.analytics import GeoClusterer, GeoClusterConfig
gc = GeoClusterer(GeoClusterConfig(eps_km=8.0, min_samples=5)).fit(persons)
persons = gc.assign(persons)                 # adds geo_cluster_id
gc.summary().to_csv("output/geo_summary.csv", index=False)
```
`eps_km` is your granularity knob; clusters are arbitrary-size (no fixed k).
Geographic outliers get unique negative ids → independent in the copula.

## Step 3 — transfer communities + the anchor pattern (якорный человек)

```python
from src.analytics import TransferClusterer, TransferClusterConfig
tc = TransferClusterer(TransferClusterConfig(resolution=1.0, min_cluster_size=4))
tc.fit(persons, transactions)
persons = tc.assign(persons)
# new columns: transfer_cluster_id, is_anchor, depends_on_anchor,
#              anchor_score, cluster_fragility
tc.anchors_table().to_csv("output/anchors.csv", index=False)

# inspect one community's money flow (for a specific case):
g = tc.subgraph(cluster_id=0)
```
`resolution` ↑ ⇒ more, smaller communities. An **anchor** is a person whose
default would cascade to dependents; **cluster_fragility** ∈ [0,1] measures how
concentrated the cluster's survival is on that one node.

## Step 4 — the copula (choose the right one)

```python
from src.copula import MultiFactorCopula     # geo AND transfer, equally weighted
factors = persons[["geo_cluster_id", "transfer_cluster_id"]].to_numpy()
mfc = MultiFactorCopula().fit(persons["model_pd"].to_numpy(), factors,
                              betas=[0.30, 0.30])   # equal betas ⇒ equal weight
```

Which copula?
| Use | When | Module |
|---|---|---|
| `MultiFactorCopula` | geo + transfer drive correlation; up to 10M | `src.copula` |
| `FactorCopula` | one systematic factor; up to 10M; t-tails via `student_t=True` | `src.copula` |
| `CopulaDefaultModel('clayton')` | ≤20k names; want explicit lower-tail dependence | `src.copula` |

Constraint for the multi-factor copula: `Σ_k β² < 1` per borrower (it raises if
violated). Implied correlation = `Σ_k β_ik·β_jk` over shared factors — share geo
only ⇒ β_geo², share both ⇒ β_geo²+β_transfer², share none ⇒ 0.

## Step 5 — sanity-check the dependence you built

```python
import numpy as np
dr = mfc.simulate_default_rate(3000)
indep_std = np.sqrt((persons["model_pd"]*(1-persons["model_pd"])).sum())/len(persons)
print("variance inflation vs independence:", round((dr.std()/indep_std)**2, 1), "x")
# >1 means your factors actually introduce correlation (they should).
```

---

## What you hand to Risk
A fitted copula object (`mfc`) + the enriched `persons` (with cluster + anchor
columns). The Risk Analyst plugs `mfc` straight into `RiskRatioCalculator`.

## Pitfalls
- Don't build a dense n×n correlation above ~20k names — use the factor copulas.
- Fewer/larger factors ⇒ stronger clustering ⇒ fatter loss tail. Choose factor
  granularity deliberately; validate with the variance-inflation check.
- The Gaussian factor copula has **no tail dependence**; for deep-stress
  clustering use `FactorCopula(student_t=True, nu=6)`.

## Verify
```bash
python demo_clusters.py    # runs steps 1–5 end-to-end and saves artifacts
```
