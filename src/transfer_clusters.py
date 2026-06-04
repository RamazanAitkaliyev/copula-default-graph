"""
transfer_clusters.py — Money-transfer communities + anchor/dependent patterns.

This module turns the inner-person money-transfer graph into:

  1. TRANSFER COMMUNITIES (`transfer_cluster_id`) — groups of people who move
     money among themselves (families, salary circles, supply chains, rings).
     Detected with weighted **Louvain** community detection. The resolution knob
     controls granularity → "differently sized transfer clusters".

  2. ANCHOR / DEPENDENT structure (якорный человек и зависимые) — within each
     community, identify a person on whom the others financially depend, such
     that if the anchor defaults the dependents are likely to default too:
        * anchor_score   — how much the community's money is fed by this node,
                           combined with whether the node is a structural
                           single-point-of-failure (articulation point) and how
                           star-shaped its ego-network is;
        * is_anchor      — boolean flag;
        * depends_on_anchor — for a dependent, the person_id of its anchor;
        * cluster_fragility — community-level: how concentrated survival is on
                           the anchor (high ⇒ anchor default likely cascades).

These columns become (a) systematic-factor input for the copula and
(b) risk-metric features at the single-person and cluster level.

Design / dependencies
---------------------
- Graph is built from the transactions DataFrame directly (sender→receiver,
  weighted by amount). We materialize a NetworkX graph for community detection
  and per-case inspection — appropriate for the sizes you actually VIEW. The
  framework's large-scale paths (sparse matrix, factor copula) are unaffected;
  for very large graphs you operate on communities, not on rendered graphs.
- Communities: `python-louvain` (the `community` package) — already installed.
- Anchor structure: directed money flows + NetworkX `articulation_points`.
- No new dependencies.

Everything is SAVED-friendly: `assign()` augments persons; `summary()` and
`anchors_table()` return CSV-ready frames; `subgraph(cluster_id)` returns a
NetworkX graph for plotting a specific case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False

try:
    import community as community_louvain  # python-louvain
    _HAS_LOUVAIN = True
except Exception:  # pragma: no cover
    _HAS_LOUVAIN = False


@dataclass
class TransferClusterConfig:
    """Configuration for transfer-community detection and anchor analysis."""
    resolution: float = 1.0          # Louvain resolution (↑ = more, smaller communities)
    random_state: int = 42           # determinism
    min_cluster_size: int = 3        # communities smaller than this are not anchor-analysed
    anchor_inflow_share: float = 0.5 # a dependent draws ≥ this share of its inflow from anchor
    anchor_score_threshold: float = 0.5  # is_anchor if anchor_score ≥ this
    out_col: str = "transfer_cluster_id"


class TransferClusterer:
    """
    Detect money-transfer communities and the anchor/dependent structure.

    Usage
    -----
        tc = TransferClusterer(TransferClusterConfig(resolution=1.0))
        tc.fit(persons, transactions)
        persons = tc.assign(persons)        # adds transfer_cluster_id + anchor cols
        comm    = tc.summary()              # per-community table → CSV
        anchors = tc.anchors_table()        # anchors + dependents → CSV
        g       = tc.subgraph(cluster_id=7) # NetworkX subgraph for plotting one case
    """

    def __init__(self, config: Optional[TransferClusterConfig] = None) -> None:
        self.config = config or TransferClusterConfig()
        self._person_ids: Optional[np.ndarray] = None
        self._pid_to_pos: Optional[Dict[int, int]] = None
        self.labels_: Optional[np.ndarray] = None            # transfer_cluster_id per person (pos order)
        self.anchor_score_: Optional[np.ndarray] = None
        self.is_anchor_: Optional[np.ndarray] = None
        self.anchor_of_cluster_: Optional[np.ndarray] = None  # cluster this node anchors, else -1
        self.depends_on_anchor_: Optional[np.ndarray] = None  # anchor person_id, else -1
        self.cluster_fragility_: Dict[int, float] = {}
        self._G: Optional["nx.DiGraph"] = None
        self._n: int = 0

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, persons: pd.DataFrame, transactions: pd.DataFrame) -> "TransferClusterer":
        if not (_HAS_NX and _HAS_LOUVAIN):
            raise ImportError(
                "transfer_clusters requires networkx and python-louvain "
                "(`pip install networkx python-louvain`)."
            )
        cfg = self.config
        self._person_ids = persons["person_id"].to_numpy()
        self._n = len(self._person_ids)
        self._pid_to_pos = {int(pid): i for i, pid in enumerate(self._person_ids)}

        # Build a DIRECTED, weighted money-flow graph (amount summed per pair).
        G = nx.DiGraph()
        G.add_nodes_from(self._person_ids.tolist())
        if len(transactions):
            grouped = (
                transactions.groupby(["sender_id", "receiver_id"])["amount"]
                .sum().reset_index()
            )
            edges = [
                (int(s), int(r), float(w))
                for s, r, w in grouped.itertuples(index=False, name=None)
                if int(s) in self._pid_to_pos and int(r) in self._pid_to_pos
            ]
            G.add_weighted_edges_from(edges)
        self._G = G

        # 1) Communities via weighted Louvain on the UNDIRECTED projection.
        UG = G.to_undirected()
        # collapse reciprocal edges by summing weights
        for u, v, data in G.edges(data=True):
            if UG.has_edge(u, v):
                # ensure weight is the sum of both directions
                pass
        partition = community_louvain.best_partition(
            UG, weight="weight",
            resolution=cfg.resolution, random_state=cfg.random_state,
        )
        labels = np.array(
            [partition.get(int(pid), -1) for pid in self._person_ids], dtype=np.int64
        )
        # Singletons (community of size 1) → unique negative ids = independent.
        labels = self._explode_singletons(labels)
        self.labels_ = labels

        # 2) Anchor / dependent structure per community.
        self._analyse_anchors(G)
        return self

    # ── assignment / tables ──────────────────────────────────────────────────
    def assign(self, persons: pd.DataFrame) -> pd.DataFrame:
        if self.labels_ is None:
            raise RuntimeError("fit() must be called before assign().")
        out = persons.copy()
        out[self.config.out_col] = self.labels_
        out["anchor_score"] = self.anchor_score_
        out["is_anchor"] = self.is_anchor_
        out["anchor_of_cluster"] = self.anchor_of_cluster_
        out["depends_on_anchor"] = self.depends_on_anchor_
        # join cluster fragility onto each member
        frag = np.array([
            self.cluster_fragility_.get(int(c), 0.0) for c in self.labels_
        ])
        out["cluster_fragility"] = frag
        return out

    def summary(self) -> pd.DataFrame:
        """Per-community table: id, n_members, internal/external weight, conductance,
        anchor person_id, fragility."""
        if self.labels_ is None:
            raise RuntimeError("fit() must be called before summary().")
        G = self._G
        rows = []
        for cid in np.unique(self.labels_[self.labels_ >= 0]):
            members_pos = np.flatnonzero(self.labels_ == cid)
            member_pids = set(int(self._person_ids[p]) for p in members_pos)
            internal = external = 0.0
            for u in member_pids:
                for _, v, data in G.out_edges(u, data=True):
                    w = data.get("weight", 1.0)
                    if v in member_pids:
                        internal += w
                    else:
                        external += w
            total = internal + external
            conductance = (external / total) if total > 0 else 0.0
            anchor_pid = self._anchor_pid_for_cluster(int(cid))
            rows.append({
                "transfer_cluster_id": int(cid),
                "n_members": int(len(members_pos)),
                "internal_weight": float(internal),
                "external_weight": float(external),
                "conductance": float(conductance),
                "anchor_person_id": (int(anchor_pid) if anchor_pid is not None else -1),
                "cluster_fragility": float(self.cluster_fragility_.get(int(cid), 0.0)),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("cluster_fragility", ascending=False).reset_index(drop=True)
        return df

    def anchors_table(self) -> pd.DataFrame:
        """One row per anchor: person_id, cluster, score, #dependents, fragility."""
        if self.is_anchor_ is None:
            raise RuntimeError("fit() must be called before anchors_table().")
        rows = []
        for pos in np.flatnonzero(self.is_anchor_):
            pid = int(self._person_ids[pos])
            cid = int(self.anchor_of_cluster_[pos])
            n_dep = int(np.sum(self.depends_on_anchor_ == pid))
            rows.append({
                "anchor_person_id": pid,
                "transfer_cluster_id": cid,
                "anchor_score": float(self.anchor_score_[pos]),
                "n_dependents": n_dep,
                "cluster_fragility": float(self.cluster_fragility_.get(cid, 0.0)),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["cluster_fragility", "anchor_score"], ascending=False).reset_index(drop=True)
        return df

    def subgraph(self, cluster_id: int) -> "nx.DiGraph":
        """Return the directed money-flow subgraph of one community (for plotting a case)."""
        if self.labels_ is None:
            raise RuntimeError("fit() must be called before subgraph().")
        members = [
            int(self._person_ids[p]) for p in np.flatnonzero(self.labels_ == cluster_id)
        ]
        return self._G.subgraph(members).copy()

    # ── anchor analysis internals ────────────────────────────────────────────
    def _analyse_anchors(self, G: "nx.DiGraph") -> None:
        n = self._n
        cfg = self.config
        anchor_score = np.zeros(n)
        is_anchor = np.zeros(n, dtype=bool)
        anchor_of_cluster = np.full(n, -1, dtype=np.int64)
        depends_on = np.full(n, -1, dtype=np.int64)
        self.cluster_fragility_ = {}

        for cid in np.unique(self.labels_[self.labels_ >= 0]):
            members_pos = np.flatnonzero(self.labels_ == cid)
            if len(members_pos) < cfg.min_cluster_size:
                continue
            member_pids = [int(self._person_ids[p]) for p in members_pos]
            member_set = set(member_pids)
            sub = G.subgraph(member_pids)

            # Per-member inbound from WITHIN the community (who is fed, and by whom).
            inflow_total: Dict[int, float] = {p: 0.0 for p in member_pids}
            inflow_from: Dict[int, Dict[int, float]] = {p: {} for p in member_pids}
            for u, v, data in sub.edges(data=True):
                w = data.get("weight", 1.0)
                if v in member_set:
                    inflow_total[v] += w
                    inflow_from[v][u] = inflow_from[v].get(u, 0.0) + w

            # Outbound money each node SENDS to community members (feeding power).
            out_to_members: Dict[int, float] = {p: 0.0 for p in member_pids}
            for u, v, data in sub.edges(data=True):
                if u in member_set and v in member_set:
                    out_to_members[u] += data.get("weight", 1.0)

            community_inflow = sum(inflow_total.values())

            # Articulation points = structural single points of failure.
            try:
                undirected = sub.to_undirected()
                artic = set(nx.articulation_points(undirected))
            except Exception:
                artic = set()

            # Score each candidate: combine money-source dominance, articulation,
            # and star-shape (this node feeds many who don't feed each other).
            best_pid, best_score = None, -1.0
            for p in member_pids:
                source_dominance = (
                    out_to_members[p] / community_inflow if community_inflow > 0 else 0.0
                )
                # how many members draw a MAJORITY of their inflow from p
                dependents = [
                    q for q in member_pids
                    if q != p and inflow_total[q] > 0
                    and inflow_from[q].get(p, 0.0) / inflow_total[q] >= cfg.anchor_inflow_share
                ]
                dep_frac = len(dependents) / max(len(member_pids) - 1, 1)
                is_artic = 1.0 if p in artic else 0.0
                # weighted blend (money dominance + dependent fraction + structural)
                score = 0.45 * source_dominance + 0.35 * dep_frac + 0.20 * is_artic
                pos = self._pid_to_pos[p]
                anchor_score[pos] = score
                if score > best_score:
                    best_pid, best_score = p, score

            # Promote the best candidate to anchor if it clears the threshold.
            if best_pid is not None and best_score >= cfg.anchor_score_threshold:
                a_pos = self._pid_to_pos[best_pid]
                is_anchor[a_pos] = True
                anchor_of_cluster[a_pos] = int(cid)
                # mark dependents
                for q in member_pids:
                    if q == best_pid:
                        continue
                    if inflow_total[q] > 0 and \
                       inflow_from[q].get(best_pid, 0.0) / inflow_total[q] >= cfg.anchor_inflow_share:
                        depends_on[self._pid_to_pos[q]] = best_pid

                # Cluster fragility: anchor's share of community inflow ×
                # (1 - redundancy), redundancy = avg #alternative sources of dependents.
                dep_positions = [self._pid_to_pos[q] for q in member_pids
                                 if depends_on[self._pid_to_pos[q]] == best_pid]
                if dep_positions:
                    redundancies = []
                    for q in member_pids:
                        if depends_on[self._pid_to_pos[q]] == best_pid:
                            n_sources = len([s for s, w in inflow_from[q].items() if w > 0])
                            redundancies.append(max(n_sources - 1, 0))
                    avg_alt = np.mean(redundancies) if redundancies else 0.0
                    redundancy = avg_alt / (avg_alt + 1.0)  # ∈ [0,1)
                else:
                    redundancy = 0.0
                anchor_inflow_share = (
                    out_to_members[best_pid] / community_inflow if community_inflow > 0 else 0.0
                )
                self.cluster_fragility_[int(cid)] = float(
                    anchor_inflow_share * (1.0 - redundancy)
                )
            else:
                self.cluster_fragility_[int(cid)] = 0.0

        self.anchor_score_ = anchor_score
        self.is_anchor_ = is_anchor
        self.anchor_of_cluster_ = anchor_of_cluster
        self.depends_on_anchor_ = depends_on

    def _anchor_pid_for_cluster(self, cid: int) -> Optional[int]:
        if self.is_anchor_ is None:
            return None
        positions = np.flatnonzero((self.anchor_of_cluster_ == cid) & self.is_anchor_)
        if len(positions) == 0:
            return None
        return int(self._person_ids[positions[0]])

    @staticmethod
    def _explode_singletons(labels: np.ndarray) -> np.ndarray:
        """Communities of size 1 → unique negative ids (independent in the copula)."""
        out = labels.astype(np.int64).copy()
        unique, counts = np.unique(out, return_counts=True)
        singletons = unique[counts == 1]
        neg = -1
        for s in singletons:
            pos = np.flatnonzero(out == s)
            out[pos] = neg
            neg -= 1
        return out


__all__ = ["TransferClusterer", "TransferClusterConfig"]
