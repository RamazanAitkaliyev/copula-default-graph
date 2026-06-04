# RECIPES — copy-paste end-to-end snippets

Practical, runnable recipes for humans and AI agents. Each is self-contained.
For the authoritative API contract see `AGENTS.md`; for the capability catalog
see `CAPABILITIES.md`.

---

## 1. Run everything with the agent façade (easiest)

```python
from src.agents import RiskAgentAPI

api = RiskAgentAPI()          # synthetic data
api.run_pipeline()            # base 13-step pipeline (PD model, copula, metrics)
api.run_cluster_analysis()    # geo + transfer clusters + anchors + multi-factor copula

print(api.fragile_clusters(top_n=5).summary)   # анкор-contagion ranking
print(api.anchors().data)                       # якорный человек list
```

---

## 2. Plug in YOUR data (persons + transfers with arbitrary column names)

```python
from src.loaders import load_persons, load_transactions, ColumnMapping

persons = load_persons("persons.csv", mapping=ColumnMapping(
    person_id="client_id",
    model_pd="pd_1y",               # your existing PD model output, in [0,1] or %
    geo_longitude="lon",
    geo_latitude="lat",
    city_id="region_code",
    exposure_at_default="ead",
    estimated_revenue="revenue",
))
transactions = load_transactions("transfers.csv", mapping=ColumnMapping(
    sender_id="from_client",
    receiver_id="to_client",
    amount="amt",
))

# Feed straight into the agent API:
from src.agents import RiskAgentAPI
api = RiskAgentAPI(persons=persons, transactions=transactions)
# If persons already has model_pd you can skip training; otherwise run_pipeline()
# trains a PD model from a 'default' label. To use cluster analysis you need PDs:
api.run_pipeline()
api.run_cluster_analysis()
print(api.fragile_clusters().summary)
```

PD normalization: `load_persons` auto-detects percentage PDs (e.g. `2.5` → `0.025`)
via a median heuristic. Verify with `describe_persons(persons)`.

---

## 3. Multi-dimensional clusters by hand (full control)

```python
import numpy as np
from src.geo_clusters import GeoClusterer, GeoClusterConfig
from src.transfer_clusters import TransferClusterer, TransferClusterConfig
from src.multi_factor_copula import MultiFactorCopula
from src.risk_adjusted_metrics import RiskRatioCalculator
from src.cluster_metrics import ClusterRiskMetrics

# --- geo clusters (DBSCAN on lat/lon; eps in km) ---
gc = GeoClusterer(GeoClusterConfig(eps_km=8.0, min_samples=5)).fit(persons)
persons = gc.assign(persons)            # adds geo_cluster_id
gc.summary().to_csv("output/geo_summary.csv", index=False)

# --- transfer communities + anchors (Louvain) ---
tc = TransferClusterer(TransferClusterConfig(resolution=1.0, min_cluster_size=4))
tc.fit(persons, transactions)
persons = tc.assign(persons)            # adds transfer_cluster_id + anchor cols
tc.anchors_table().to_csv("output/anchors.csv", index=False)

# --- multi-factor copula: geo AND transfer, EQUALLY weighted ---
pds = persons["model_pd"].to_numpy()
factors = persons[["geo_cluster_id", "transfer_cluster_id"]].to_numpy()
mfc = MultiFactorCopula().fit(pds, factors, betas=[0.30, 0.30])   # equal loadings

# --- cluster risk metrics + anchor contagion ---
# EAD: prefer a real exposure column; fall back to income, then a flat value.
if "exposure_at_default" in persons.columns:
    ead = persons["exposure_at_default"].to_numpy()
elif "income" in persons.columns:
    ead = persons["income"].to_numpy()
else:
    ead = np.full(len(persons), 10000.0)
calc = RiskRatioCalculator(mfc, persons, exposures=ead, lgd=0.45)
crm = ClusterRiskMetrics(calc, persons)
crm.geo_metrics().to_csv("output/geo_metrics.csv", index=False)
crm.transfer_metrics().to_csv("output/transfer_metrics.csv", index=False)
crm.anchor_contagion_table().to_csv("output/anchor_contagion.csv", index=False)
```

---

## 4. The anchor / dependents pattern (якорный человек) explained in code

```python
# After tc.assign(persons), each person row carries:
#   is_anchor          — True if this person's default would cascade to a cluster
#   anchor_of_cluster  — the transfer cluster this person anchors (else -1)
#   depends_on_anchor  — the anchor person_id this person depends on (else -1)
#   cluster_fragility  — 0..1; how concentrated the cluster's survival is on one node

# Find the families most at risk of joint default:
fragile = persons[persons["cluster_fragility"] > 0.5]
anchors = persons[persons["is_anchor"]]

# Quantify it: how much does a cluster's expected loss rise if the anchor defaults?
contagion = crm.anchor_contagion_table()   # uplift_ratio column; >1 = cascade risk
worst = contagion.iloc[0]
print(f"If anchor {int(worst['anchor_person_id'])} defaults, cluster "
      f"{int(worst['transfer_cluster_id'])} loss goes "
      f"{worst['el_unconditional']:.0f} -> {worst['el_anchor_default']:.0f} "
      f"({worst['uplift_ratio']:.2f}x)")
```

---

## 5. Scale to millions (factor / multi-factor copula)

```python
import numpy as np
from src.multi_factor_copula import MultiFactorCopula

n = 10_000_000
pds = np.random.uniform(0.005, 0.08, n)
geo = np.random.randint(0, 200, n)         # geo cluster id per person
transfer = np.random.randint(0, 500, n)    # transfer community id per person
factors = np.column_stack([geo, transfer])

mfc = MultiFactorCopula().fit(pds, factors, betas=[0.3, 0.3])  # O(n*K), ~seconds
dr = mfc.simulate_default_rate(2000)        # streamed, memory-bounded
print(dr.mean(), dr.std())                  # portfolio default-rate distribution
```

Never build a dense n×n correlation matrix at scale — the (multi-)factor copula
stores only O(n·K) loadings and computes joint-default blocks on demand.

---

## 6. Per-cluster metrics for any segment dimension

```python
# Clusters are just a column → every existing metric rolls up correctly
# (block-sum loss covariance, never an average of per-borrower ratios).
calc.by_segment("geo_cluster_id")        # per geo cluster
calc.by_segment("transfer_cluster_id")   # per transfer community
calc.by_segment("city_id")               # per city
# columns: exposure, expected_loss, loss_std_copula, coefficient_of_variation,
#          raroc, sortino_copula, diversification_ratio, ...
```

---

## 7. Inspect ONE cluster's money-flow graph (specific-case viewing)

```python
g = tc.subgraph(cluster_id=0)            # NetworkX DiGraph of that community
# plot it (red = anchor), or inspect edges:
for u, v, d in g.edges(data=True):
    print(u, "->", v, d["weight"])
```

`demo_clusters.py` saves example subgraph PNGs automatically.

---

## 8. Run the whole reference pipeline as a script

```bash
python demo_clusters.py        # geo + transfer + anchors + metrics → output/*.csv + PNGs
python main.py                 # the base 13-step pipeline
python test_copula_framework.py   # 41 tests — should print "All 41 tests passed."
```
