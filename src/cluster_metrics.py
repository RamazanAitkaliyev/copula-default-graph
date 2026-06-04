"""
cluster_metrics.py — Risk metrics for multi-dimensional clusters + anchors.

Ties together the cluster layers (geo + transfer), the multi-factor copula, and
the risk-metric engine to answer the questions the user actually cares about:

  * For every GEO cluster and TRANSFER community: EAD, EL, σ(Loss), CoV, RAROC,
    Sortino — produced by the existing `RiskRatioCalculator.by_segment` (block-sum
    loss covariance, INV-6), so no metric math is re-implemented here.

  * For the ANCHOR / dependents pattern (якорный человек): the headline number is
    **anchor-conditional cluster loss** — how much a cluster's expected loss rises
    if its anchor defaults. If the cluster is fragile (everyone depends on one
    person), conditioning on the anchor's default sharply raises the others' PDs,
    so the conditional EL ≫ unconditional EL. That gap is the quantified
    "if the breadwinner defaults, the family defaults too" risk.

Conditional PD under the Gaussian factor model
----------------------------------------------
For borrower j sharing systematic factor(s) with anchor a, conditioning on the
anchor defaulting (A_a ≤ z_a) shifts j's default probability:

    PD_j | (a defaults) = P(D_a ∩ D_j) / PD_a

which is read straight off the copula's joint-default block — no new model. The
cluster's conditional EL is Σ_j EAD_j·LGD_j·(PD_j | a defaults), with the anchor
itself at PD=1.

Everything is SAVED-friendly: `cluster_report()` returns CSV-ready frames and
`anchor_contagion_table()` ranks clusters by conditional-loss uplift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ClusterMetricsResult:
    geo_metrics: pd.DataFrame
    transfer_metrics: pd.DataFrame
    anchor_contagion: pd.DataFrame


class ClusterRiskMetrics:
    """
    Compute cluster-level metrics and anchor-contagion uplift.

    Parameters
    ----------
    calc : RiskRatioCalculator
        Already constructed with the (multi-factor) copula + persons + EAD/LGD.
    persons : pd.DataFrame
        Must contain the cluster id columns and (if available) anchor columns
        produced by TransferClusterer.assign.
    geo_col, transfer_col : str
        Cluster id columns to roll up.
    """

    def __init__(
        self,
        calc,
        persons: pd.DataFrame,
        geo_col: str = "geo_cluster_id",
        transfer_col: str = "transfer_cluster_id",
    ) -> None:
        self.calc = calc
        self.persons = persons.reset_index(drop=True)
        self.geo_col = geo_col
        self.transfer_col = transfer_col
        # positional index of each person_id (calc uses 0..n-1 contiguous)
        self._pid_to_pos = {
            int(p): i for i, p in enumerate(self.persons["person_id"].to_numpy())
        }

    # ── cluster roll-ups (reuse by_segment) ──────────────────────────────────
    def geo_metrics(self) -> pd.DataFrame:
        if self.geo_col not in self.persons.columns:
            return pd.DataFrame()
        return self.calc.by_segment(self.geo_col)

    def transfer_metrics(self) -> pd.DataFrame:
        if self.transfer_col not in self.persons.columns:
            return pd.DataFrame()
        return self.calc.by_segment(self.transfer_col)

    # ── anchor-conditional contagion ─────────────────────────────────────────
    def anchor_contagion_table(self) -> pd.DataFrame:
        """
        For each transfer cluster that HAS an anchor, compute:
          - el_unconditional  : Σ EAD·LGD·PD over members
          - el_anchor_default : Σ EAD·LGD·(PD_j | anchor defaults), anchor at PD=1
          - uplift_ratio      : el_anchor_default / el_unconditional
          - uplift_abs        : el_anchor_default − el_unconditional
        Ranked by uplift_ratio (most anchor-dependent clusters first).
        """
        persons = self.persons
        needed = {"is_anchor", "anchor_of_cluster", self.transfer_col}
        if not needed.issubset(persons.columns):
            return pd.DataFrame()

        calc = self.calc
        el_weight = calc.ead * calc.lgd          # (n,) loss given default
        pd_vec = calc.pd

        rows: List[Dict] = []
        anchors = persons[persons["is_anchor"]]
        for _, arow in anchors.iterrows():
            cid = int(arow["anchor_of_cluster"])
            if cid < 0:
                continue
            anchor_pid = int(arow["person_id"])
            members = persons[persons[self.transfer_col] == cid]
            member_pos = np.array(
                [self._pid_to_pos[int(p)] for p in members["person_id"]]
            )
            if len(member_pos) < 2:
                continue
            a_pos = self._pid_to_pos[anchor_pid]

            # unconditional EL over the cluster
            el_uncond = float((el_weight[member_pos] * pd_vec[member_pos]).sum())

            # conditional PDs: P(D_j ∩ D_anchor) / PD_anchor, anchor itself = 1.
            block_idx = member_pos
            J = calc._copula.joint_default_probability_block(block_idx)  # (m,m)
            a_local = int(np.flatnonzero(block_idx == a_pos)[0])
            pd_anchor = max(pd_vec[a_pos], 1e-12)
            cond_pd = J[a_local, :] / pd_anchor          # (m,)
            cond_pd = np.clip(cond_pd, 0.0, 1.0)
            cond_pd[a_local] = 1.0                        # anchor defaults for sure
            el_cond = float((el_weight[member_pos] * cond_pd).sum())

            rows.append({
                "transfer_cluster_id": cid,
                "anchor_person_id": anchor_pid,
                "n_members": int(len(member_pos)),
                "el_unconditional": el_uncond,
                "el_anchor_default": el_cond,
                "uplift_abs": el_cond - el_uncond,
                "uplift_ratio": (el_cond / el_uncond) if el_uncond > 0 else np.nan,
                "cluster_fragility": float(arow.get("cluster_fragility", np.nan)),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("uplift_ratio", ascending=False).reset_index(drop=True)
        return df

    # ── one-shot bundle ──────────────────────────────────────────────────────
    def compute(self) -> ClusterMetricsResult:
        return ClusterMetricsResult(
            geo_metrics=self.geo_metrics(),
            transfer_metrics=self.transfer_metrics(),
            anchor_contagion=self.anchor_contagion_table(),
        )


__all__ = ["ClusterRiskMetrics", "ClusterMetricsResult"]
