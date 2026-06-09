"""
Stage 10 — Individual PD scoring  (owner: ML Engineer)
======================================================

Train (or apply) the individual probability-of-default model and write a
``model_pd`` column for every borrower. Mirrors ``main.py`` STEP 2-3: neighbour
risk features are merged first, then the gradient-boosting PD model is fit.

If your data already carries a PD, set ``skip_training=True`` and provide
``pd_col`` — the stage just copies it to ``model_pd`` and the chain proceeds.

Inputs (artifacts):  persons, transactions
Outputs (artifacts): persons_scored (persons + neighbour features + model_pd),
                     pd_feature_importance, pd_metrics (json)
"""

from __future__ import annotations

from typing import List, Optional

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "10_pd"

DEFAULT_FEATURES: List[str] = [
    "age", "income", "employment_years", "debt_to_income",
    "num_credit_lines", "missed_payments", "credit_utilization",
    "account_age_months",
]


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    model_type: str = "gradient_boosting",
    feature_columns: Optional[List[str]] = None,
    skip_training: bool = False,
    pd_col: str = "model_pd",
) -> StageResult:
    """
    Score borrowers with the individual PD model.

    Parameters
    ----------
    model_type : {"gradient_boosting", "logistic"}
        Backend estimator.
    feature_columns : list of str, optional
        Features to train on (defaults to the same set as ``main.py``).
    skip_training : bool
        If True, do not train; copy an existing ``pd_col`` to ``model_pd``.
    pd_col : str
        Source PD column used when ``skip_training`` is True.
    """
    store.require("persons", "transactions")
    persons = store.read_df("persons")
    transactions = store.read_df("transactions")
    res = StageResult(stage=STAGE, ok=True)

    # Merge neighbour risk features (contagion channel at the feature level).
    # get_neighbor_risk_features needs a `base_pd` column. Synthetic data has it;
    # real data may only carry `model_pd` / `pd_col`. Provide a base_pd fallback
    # (without overwriting an existing one) so this works on real portfolios too.
    from src.graph_features import TransactionGraph, get_neighbor_risk_features
    persons_for_graph = persons
    if "base_pd" not in persons.columns:
        fallback_pd = pd_col if pd_col in persons.columns else (
            "model_pd" if "model_pd" in persons.columns else None
        )
        if fallback_pd is not None:
            persons_for_graph = persons.copy()
            persons_for_graph["base_pd"] = persons_for_graph[fallback_pd].clip(0.0, 1.0)
        else:
            persons_for_graph = persons.copy()
            persons_for_graph["base_pd"] = 0.0  # last resort: no PD info yet
    graph = TransactionGraph(transactions, persons_for_graph)
    neighbor = get_neighbor_risk_features(graph, persons_for_graph)
    persons = persons.merge(
        neighbor[["person_id", "neighbor_pd_avg", "neighbor_pd_max",
                  "n_high_risk_neighbors"]],
        on="person_id", how="left",
    ).fillna(0)

    if skip_training:
        if pd_col not in persons.columns:
            raise ValueError(
                f"skip_training=True but column {pd_col!r} not present to copy into model_pd"
            )
        persons["model_pd"] = persons[pd_col].clip(0.0, 1.0)
        res.metrics["trained"] = False
    else:
        from src.pd_model import IndividualPDModel
        model = IndividualPDModel(
            model_type=model_type,
            feature_columns=feature_columns or DEFAULT_FEATURES,
        )
        metrics = model.fit(persons, target_col="default", validation_split=0.2)
        persons["model_pd"] = model.predict_proba(persons)
        res.metrics["trained"] = True
        res.metrics["train_auc"] = round(float(metrics.get("train_auc", float("nan"))), 4)
        res.metrics["val_auc"] = round(float(metrics.get("val_auc", float("nan"))), 4)
        # Persist feature importance for the ML owner's review.
        imp = model.feature_importance_.reset_index()
        imp.columns = ["feature", "importance"]
        res.outputs.append(store.write_df("pd_feature_importance", imp))
        res.outputs.append(store.write_json("pd_metrics", {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in metrics.items()
        }))

    res.metrics["model_pd_min"] = round(float(persons["model_pd"].min()), 4)
    res.metrics["model_pd_max"] = round(float(persons["model_pd"].max()), 4)
    res.outputs.append(store.write_df("persons_scored", persons))
    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s).summary())
