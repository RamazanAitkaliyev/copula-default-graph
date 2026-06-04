# PLAN — Multi-dimensional risk clusters (geo + transfer), anchor patterns, cluster metrics

> Builds on the existing framework. Each phase is a coherent unit that ships with
> tests and keeps `test_copula_framework.py` green. Decisions locked by the user:
> **multi-factor copula** (geo & transfer equally weighted), **networkx +
> python-louvain** (no new installs; analysis is **saved** to artifacts; viewing
> specific cases is supported, no million-node rendering needed),
> **anchor detection + fragility scoring now**.

## Goal (user's words, made concrete)
1. Read a person properly → analytics + PD. *(loaders already do this; add geo.)*
2. Multi-dimensional clusters:
   - **Geo clusters** from `(city, geo_longitude, geo_latitude)`.
   - **Transfer clusters** (communities) from inner-person money transfers.
3. Understand / analyse / classify **risk clusters** and their probability of
   "going down together" — esp. **anchor person → dependents** (якорный человек
   и зависимые): if a family/cluster depends on one person and that person
   defaults, the dependents likely default too. Detect this and similar patterns.
4. Many **risk metrics** for both single persons and clusters.
5. Everything else already in the module keeps working.

## What already exists (reuse, don't rebuild)
- `loaders.py`: `ColumnMapping`, `load_persons/transactions`, validation, reindex.
  → **Add geo fields** (`geo_longitude`, `geo_latitude`) to `ColumnMapping`.
- `graph_features.py`: sparse CSR transaction graph, centrality, components.
  → **Add Louvain communities + anchor/dependency features.**
- `factor_copula.py`: single-factor Vasicek, O(n), proven to 2M.
  → **Generalize to multi-factor** (`MultiFactorCopula`), keep single as-is.
- `risk_adjusted_metrics.py`: `RiskRatioCalculator`, `by_segment(col)`, block-sum
  loss-cov (INV-6), **already accepts per-borrower lgd array**.
  → Clusters are just new `*_cluster_id` columns fed to `by_segment`. Add
  **cluster-fragility & anchor metrics**.
- `agents.py`: `RiskAgentAPI` façade. → Add cluster query methods at the end.

---

## Phase 1 — Geo ingestion + geo clusters  *(foundation)*
**Files:** `src/loaders.py` (extend), **new** `src/geo_clusters.py`, tests.

1.1 `ColumnMapping`: add `geo_longitude: str = "geo_longitude"`,
    `geo_latitude: str = "geo_latitude"`. Thread through `persons_rename_map`
    and `load_persons` (optional — absent ⇒ geo features skipped, no crash).
    Validate ranges (lon∈[-180,180], lat∈[-90,90]) in `validate_persons`.

1.2 `geo_clusters.py` — `GeoClusterer`:
    - `fit(persons)` → integer `geo_cluster_id` per person.
    - Method = **DBSCAN on (lat,lon)** (haversine metric) → density-based,
      arbitrary-size, no fixed k → "differently sized clusters". `-1` = noise =
      own cluster (independent). Fallback to **city_id** grouping if no coords.
    - Multi-resolution: `eps` parameter exposed; also a coarse `city` level and a
      fine `dbscan` level so clusters nest (user's "differently sized" ask).
    - **Save**: `assign(persons)` returns persons + `geo_cluster_id`; a
      `summary()` DataFrame (cluster_id, n_members, centroid_lat/lon, span_km).

**Test:** synthetic 3 tight geo blobs + scatter → DBSCAN recovers 3 clusters +
noise; city fallback works when coords absent.

---

## Phase 2 — Transfer communities + anchor/dependency patterns  *(the core idea)*
**Files:** **new** `src/transfer_clusters.py`, `src/graph_features.py` (small add).

2.1 `transfer_clusters.py` — `TransferClusterer`:
    - Build undirected weighted graph from transactions (reuse the sparse CSR;
      convert to networkx only for the community call — fine at the sizes we
      view; for very large graphs we operate on the sparse matrix and only
      materialize subgraphs for case viewing).
    - **Communities = python-louvain** (`community.best_partition`, weighted) →
      `transfer_cluster_id` per person. Resolution parameter = granularity knob.
    - `assign(persons)` → persons + `transfer_cluster_id`; `summary()` per
      community (n_members, internal_weight, external_weight, conductance).
    - **Save** community assignment + summary to CSV.

2.2 **Anchor / dependents detection** (`AnchorAnalyzer` in same module):
    The pattern "everyone in the cluster depends on one person":
    - **Anchor score** per node, combining:
      - *Money-source dominance*: fraction of the cluster's inbound money that
        originates from this node (out-strength to cluster members / total
        cluster inbound). High ⇒ cluster is financially fed by this node.
      - *Articulation point*: removing the node disconnects the cluster
        (networkx `articulation_points` on the community subgraph). True ⇒
        structural single-point-of-failure.
      - *Star/hub asymmetry*: high out-degree to many members who have low
        out-degree among themselves (ego-net is star-shaped).
    - **Dependents**: members whose inbound is dominated by the anchor (≥ θ of
      their inflow comes from the anchor).
    - Output columns on persons: `anchor_score`, `is_anchor` (bool),
      `anchor_of_cluster` (the cluster it anchors, else -1),
      `depends_on_anchor` (anchor's person_id this node depends on, else -1).

2.3 **Cluster fragility score** (cluster-level):
    - `fragility = anchor_inbound_share × (1 - redundancy)` where redundancy =
      how many alternative money sources members have. High fragility ⇒ if the
      anchor defaults, the cluster likely cascades.
    - This is the quantified "family depends on a single person" risk.

**Test:** build a star cluster (1 hub pays 5 dependents, dependents don't pay
each other) + a mesh cluster (everyone pays everyone). Assert: hub flagged
`is_anchor`, the 5 flagged `depends_on_anchor=hub`, star fragility ≫ mesh
fragility, hub is an articulation point.

---

## Phase 3 — Multi-factor copula (geo ⟂ transfer, equally weighted)  *(dependence)*
**Files:** **new** `src/multi_factor_copula.py`, tests.

3.1 `MultiFactorCopula`:
    `A_i = Σ_k β_{i,k}·Y_k + √(1 − Σ_k β_{i,k}²)·ε_i`, Y_k ⟂ standard normals.
    - `fit(pds, factor_matrix, betas)` where `factor_matrix` is shape (n, K_used)
      of factor ids per dimension (col 0 = geo_cluster_id, col 1 =
      transfer_cluster_id), `betas` shape (n, K) loadings (equal by default ⇒
      "equally important").
    - **Constraint:** `Σ β² < 1` (else negative idiosyncratic variance) — clip /
      validate, raise on violation.
    - Implied correlation: `corr(A_i,A_j) = Σ_k β_{i,k}β_{j,k}·[same factor k]`.
    - `simulate_defaults`, `simulate_default_rate` — same streamed, O(n·K)
      memory design as FactorCopula (draw K systematic normals per scenario).
    - `joint_default_probability_block(idx)` — reuse the fast `_bvn_cdf` with
      the summed pairwise correlation. (t-variant deferred; Gaussian first.)
    - Storage O(n·K) — K=2 here ⇒ scales to 10M exactly like the single-factor.

3.2 **Drop-in for metrics:** `RiskRatioCalculator` only calls
    `copula.simulate_*` / `joint_default_probability_block` — implement the same
    duck-typed interface so `MultiFactorCopula` plugs in unchanged.

**Test:** two factors, equal betas. Assert: (a) Σβ²<1 enforced; (b) two borrowers
sharing the geo factor but not transfer have corr = β²; sharing both = 2β²;
sharing none = 0; (c) analytical block matches simulation (~MC noise);
(d) variance inflation > independence; (e) more shared factors ⇒ more joint
defaults.

---

## Phase 4 — Cluster & anchor RISK METRICS  *(single-person + cluster)*
**Files:** `src/risk_adjusted_metrics.py` (extend), **new** thin
`src/cluster_metrics.py` orchestration.

4.1 Cluster-level (reuse `by_segment` block-sum — INV-6, already correct):
    - For `geo_cluster_id` AND `transfer_cluster_id`: EAD, EL, σ(Loss),
      CoV, RAROC, Sortino-copula, diversification ratio — **all existing metrics
      "for free" per cluster** by feeding the cluster-id column.
    - **Cluster default-correlation / contagion**: avg pairwise implied corr,
      and **expected cluster co-default rate** from the copula block.
    - **Anchor-conditional loss**: cluster EL *given the anchor defaults*
      (condition the copula on anchor=default, re-sum) − unconditional EL =
      "anchor contagion uplift". This is the headline number for the якорный-
      человек pattern: *how much does this cluster's risk hinge on one person.*

4.2 Single-person (extend `per_borrower`):
    - `anchor_score`, `is_anchor`, `depends_on_anchor`, `cluster_fragility`
      (joined from Phase 2), plus existing per-borrower metrics.
    - **Systemic-importance for anchors**: PD uplift this borrower causes in its
      dependents if it defaults (already have a contagion primitive in
      `risk_metrics.py` — wire it to the anchor set).

4.3 **Save everything**: `output/cluster_geo_metrics.csv`,
    `output/cluster_transfer_metrics.csv`, `output/anchors.csv`,
    `output/cluster_fragility.csv`, and per-case subgraph PNGs on demand
    (`plot_cluster(cluster_id)` for viewing specific cases).

**Test:** anchor-conditional cluster EL > unconditional EL for a star cluster;
metrics roll up consistently (Σ cluster EL == portfolio EL — reconciliation).

---

## Phase 5 — Pipeline + agent + presentation wiring  *(make it usable)*
**Files:** `main.py` (new steps), `src/agents.py` (new methods),
`src/__init__.py` (exports), `CLAUDE.md`/`AGENTS.md` (docs), tests.

5.1 `main.py`: add steps — geo clusters → transfer communities + anchors →
    multi-factor copula → cluster metrics → save artifacts + a few case plots.
5.2 `RiskAgentAPI`: `geo_clusters()`, `transfer_clusters()`, `anchors()`,
    `cluster_report(cluster_id)`, `fragile_clusters(top_n)` → all return
    `AgentResult` with saved-artifact paths.
5.3 Exports in `src/__init__.py`; update `CLAUDE.md` file map + test count;
    add a short section to `AGENTS.md` ("Clusters & anchors") and the
    presentation generators (EN/RU) — optional, last.

---

## Cross-cutting rules
- **Non-breaking:** new modules + additive columns; existing tests stay green.
- **Optional geo/graph:** absent coords or transactions ⇒ skip gracefully.
- **Saved analysis:** every cluster analysis writes a CSV/artifact (user wants
  results saved); per-case subgraph plotting for inspecting specific connections.
- **Scale-aware:** community detection materializes networkx only for sizes we
  view; copula/metrics use the O(n·K) sparse/block paths for 10M.
- **Tests first per phase**, then wire into `main.py`. Update the test count in
  the suite summary each phase.
- **Determinism:** seed DBSCAN/Louvain where randomness exists.

## Execution order (steady, accurate)
P1 geo → P2 transfer+anchors → P3 multi-factor copula → P4 metrics → P5 wiring.
Commit after each green phase (only when you ask me to push).
