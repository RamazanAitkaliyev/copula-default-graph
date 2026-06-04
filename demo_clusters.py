"""
demo_clusters.py — End-to-end multi-dimensional cluster + anchor risk pipeline.

Runs the full NEW workflow on synthetic data and writes saved artifacts:

    1. Read persons (+ geo coords) and money transfers.
    2. GEO clusters      — DBSCAN on (lat, lon)          → geo_cluster_id
    3. TRANSFER clusters — Louvain on money-flow graph   → transfer_cluster_id
       + ANCHOR / dependents detection (якорный человек) → is_anchor, ...
    4. MULTI-FACTOR copula (geo ⟂ transfer, equally weighted) for correlated PDs.
    5. CLUSTER risk metrics (by_segment for each dimension) + ANCHOR-CONTAGION
       uplift (how much a cluster's loss rises if its anchor defaults).
    6. Save everything to output/ and plot a few example cluster subgraphs.

Point this at YOUR data by replacing step 1 with loaders.load_persons /
load_transactions using a ColumnMapping (see src/loaders.py). Everything
downstream is unchanged.

Run:  python demo_clusters.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.data_generator import generate_network
from src.geo_clusters import GeoClusterer, GeoClusterConfig
from src.transfer_clusters import TransferClusterer, TransferClusterConfig
from src.multi_factor_copula import MultiFactorCopula
from src.risk_adjusted_metrics import RiskRatioCalculator
from src.cluster_metrics import ClusterRiskMetrics

OUT = "output"
os.makedirs(OUT, exist_ok=True)


# City centres (lon, lat) — Kazakhstan-ish, illustrative only.
CITY_CENTRES = {
    0: (76.9, 43.2),   # Almaty
    1: (71.4, 51.1),   # Astana
    2: (69.6, 42.3),   # Shymkent
}


def add_geo_coordinates(persons: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Attach synthetic lat/lon clustered around each person's city centre.

    Mimics the user's dataset shape (person_id, city, geo_longitude, geo_latitude).
    People in the same city scatter within ~a few km; a few are placed far out as
    geographic outliers (they will fall out of geo clusters → independent)."""
    rng = np.random.default_rng(seed)
    persons = persons.copy()
    lon = np.empty(len(persons)); lat = np.empty(len(persons))
    for i, cid in enumerate(persons["city_id"].to_numpy()):
        clon, clat = CITY_CENTRES.get(int(cid), (70.0, 48.0))
        lon[i] = clon + rng.normal(0, 0.03)   # ~3 km spread
        lat[i] = clat + rng.normal(0, 0.03)
    # 3% geographic outliers
    n_out = max(1, int(0.03 * len(persons)))
    out_idx = rng.choice(len(persons), n_out, replace=False)
    lon[out_idx] = rng.uniform(60, 80, n_out)
    lat[out_idx] = rng.uniform(40, 53, n_out)
    persons["geo_longitude"] = lon
    persons["geo_latitude"] = lat
    return persons


def main() -> None:
    print("=" * 64)
    print("  MULTI-DIMENSIONAL CLUSTER + ANCHOR RISK PIPELINE")
    print("=" * 64)

    # ── 1. data ───────────────────────────────────────────────────────────────
    print("\n[1/6] Generating synthetic persons (+geo) and transfers ...")
    persons, transactions = generate_network(seed=42)
    persons = add_geo_coordinates(persons)
    # contiguous person_id 0..n-1 (the framework relies on positional indexing)
    persons = persons.sort_values("person_id").reset_index(drop=True)
    n = len(persons)
    pd_col = "model_pd" if "model_pd" in persons.columns else "base_pd"
    print(f"      persons={n}  transactions={len(transactions)}  PD column='{pd_col}'")

    # ── 2. geo clusters ───────────────────────────────────────────────────────
    print("\n[2/6] Geo clusters (DBSCAN on lat/lon) ...")
    gc = GeoClusterer(GeoClusterConfig(eps_km=8.0, min_samples=5)).fit(persons)
    persons = gc.assign(persons)
    geo_summary = gc.summary()
    geo_summary.to_csv(f"{OUT}/cluster_geo_summary.csv", index=False)
    n_geo = int((persons["geo_cluster_id"] >= 0).any()) and persons.loc[
        persons["geo_cluster_id"] >= 0, "geo_cluster_id"].nunique()
    print(f"      method={gc.method_}  genuine geo clusters={n_geo}  "
          f"isolated={int((persons['geo_cluster_id'] < 0).sum())}")

    # ── 3. transfer communities + anchors ─────────────────────────────────────
    print("\n[3/6] Transfer communities (Louvain) + anchor detection ...")
    tc = TransferClusterer(
        TransferClusterConfig(resolution=1.0, min_cluster_size=4)
    ).fit(persons, transactions)
    persons = tc.assign(persons)
    tc.summary().to_csv(f"{OUT}/cluster_transfer_summary.csv", index=False)
    anchors = tc.anchors_table()
    anchors.to_csv(f"{OUT}/anchors.csv", index=False)
    n_tx_clusters = persons.loc[
        persons["transfer_cluster_id"] >= 0, "transfer_cluster_id"].nunique()
    print(f"      transfer communities={n_tx_clusters}  anchors found={len(anchors)}")
    if len(anchors):
        top = anchors.iloc[0]
        print(f"      most fragile anchored cluster: id={int(top['transfer_cluster_id'])}"
              f"  anchor=person {int(top['anchor_person_id'])}"
              f"  dependents={int(top['n_dependents'])}"
              f"  fragility={top['cluster_fragility']:.2f}")

    # ── 4. multi-factor copula (geo ⟂ transfer, equally weighted) ─────────────
    print("\n[4/6] Multi-factor copula (geo + transfer, equal loadings) ...")
    pds = persons[pd_col].to_numpy()
    factor_matrix = persons[["geo_cluster_id", "transfer_cluster_id"]].to_numpy()
    # equal Basel-ish loadings; Σβ² = 0.18 < 1
    mfc = MultiFactorCopula().fit(pds, factor_matrix, betas=[0.30, 0.30])
    dr = mfc.simulate_default_rate(3000, seed=1)
    indep_std = np.sqrt((pds * (1 - pds)).sum()) / n
    print(f"      portfolio default rate: mean={dr.mean():.4f}  std={dr.std():.4f}"
          f"  (×{(dr.std()/indep_std)**2:.0f} variance vs independence)")

    # ── 5. cluster metrics + anchor contagion ─────────────────────────────────
    print("\n[5/6] Cluster risk metrics + anchor-contagion uplift ...")
    ead = persons["exposure_at_default"].to_numpy() if \
        "exposure_at_default" in persons.columns else \
        (persons["income"].to_numpy() if "income" in persons.columns else np.full(n, 10000.0))
    calc = RiskRatioCalculator(mfc, persons, exposures=ead, lgd=0.45)
    crm = ClusterRiskMetrics(calc, persons)
    res = crm.compute()
    res.geo_metrics.to_csv(f"{OUT}/cluster_geo_metrics.csv", index=False)
    res.transfer_metrics.to_csv(f"{OUT}/cluster_transfer_metrics.csv", index=False)
    res.anchor_contagion.to_csv(f"{OUT}/anchor_contagion.csv", index=False)
    if len(res.anchor_contagion):
        worst = res.anchor_contagion.iloc[0]
        print(f"      worst anchor-contagion: cluster {int(worst['transfer_cluster_id'])}"
              f"  EL {worst['el_unconditional']:.0f} → {worst['el_anchor_default']:.0f}"
              f"  ({worst['uplift_ratio']:.2f}× if anchor defaults)")
    else:
        print("      (no anchored clusters large enough for contagion analysis)")

    # save the enriched persons table (single source of truth)
    keep = ["person_id", "city_id", pd_col, "geo_cluster_id", "transfer_cluster_id",
            "anchor_score", "is_anchor", "anchor_of_cluster", "depends_on_anchor",
            "cluster_fragility"]
    keep = [c for c in keep if c in persons.columns]
    persons[keep].to_csv(f"{OUT}/persons_clustered.csv", index=False)

    # ── 6. plot a few example cluster subgraphs (specific-case viewing) ───────
    print("\n[6/6] Plotting example cluster subgraphs ...")
    try:
        _plot_examples(tc, anchors)
        print(f"      saved example subgraph PNGs to {OUT}/")
    except Exception as e:  # plotting is optional / best-effort
        print(f"      (plot skipped: {e})")

    print("\n" + "=" * 64)
    print("  DONE — artifacts written to output/")
    print("    cluster_geo_summary.csv / cluster_geo_metrics.csv")
    print("    cluster_transfer_summary.csv / cluster_transfer_metrics.csv")
    print("    anchors.csv / anchor_contagion.csv / persons_clustered.csv")
    print("=" * 64)


def _plot_examples(tc: TransferClusterer, anchors: pd.DataFrame, k: int = 2) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    if not len(anchors):
        return
    for rank in range(min(k, len(anchors))):
        cid = int(anchors.iloc[rank]["transfer_cluster_id"])
        anchor_pid = int(anchors.iloc[rank]["anchor_person_id"])
        g = tc.subgraph(cid)
        if g.number_of_nodes() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = nx.spring_layout(g, seed=1)
        node_colors = ["#d62728" if node == anchor_pid else "#1f77b4" for node in g.nodes()]
        node_sizes = [600 if node == anchor_pid else 250 for node in g.nodes()]
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
        nx.draw_networkx_edges(g, pos, alpha=0.4, arrows=True, ax=ax)
        nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
        ax.set_title(f"Transfer cluster {cid} — anchor (red) = person {anchor_pid}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(f"{OUT}/cluster_{cid}_subgraph.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
