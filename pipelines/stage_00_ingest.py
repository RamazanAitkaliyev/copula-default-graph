"""
Stage 00 — Ingest  (owner: Data Engineer)
==========================================

Produce the two canonical tables every downstream stage relies on:
``persons`` and ``transactions``.

In synthetic mode it calls ``data_generator.generate_network``. To ingest real
data instead, point this stage at your files via the ``loaders`` module (the
``source`` option) — the rest of the pipeline is unchanged.

Inputs (artifacts):  none (this is the source stage)
Outputs (artifacts): persons, transactions
"""

from __future__ import annotations

from typing import Optional

from .artifacts import ArtifactStore, StageResult, timed_stage

STAGE = "00_ingest"


@timed_stage(STAGE)
def run(
    store: ArtifactStore,
    seed: int = 42,
    persons_source: Optional[str] = None,
    transactions_source: Optional[str] = None,
    mapping: Optional[object] = None,
) -> StageResult:
    """
    Generate (or load) the persons + transactions tables.

    Parameters
    ----------
    seed : int
        RNG seed for synthetic generation.
    persons_source, transactions_source : str, optional
        If both are given, ingest REAL data from these paths via
        ``src.loaders`` instead of generating synthetic data.
    mapping : src.loaders.ColumnMapping, optional
        Column mapping for real-data ingestion (your columns → canonical names).
    """
    res = StageResult(stage=STAGE, ok=True)

    if persons_source and transactions_source:
        from src.loaders import (
            load_persons, load_transactions, ColumnMapping,
            reindex_to_contiguous,
        )
        m = mapping or ColumnMapping()
        persons = load_persons(persons_source, mapping=m)
        transactions = load_transactions(transactions_source, mapping=m)
        # reindex_to_contiguous returns (persons, transactions, id_map); the map
        # records original-id -> contiguous-id for traceability.
        persons, transactions, id_map = reindex_to_contiguous(persons, transactions)
        res.outputs.append(store.write_json("person_id_map", {str(k): int(v) for k, v in id_map.items()}))
        res.metrics["mode"] = "real"
    else:
        from src.data_generator import generate_network
        persons, transactions = generate_network(seed=seed)
        res.metrics["mode"] = "synthetic"

    res.outputs.append(store.write_df("persons", persons))
    res.outputs.append(store.write_df("transactions", transactions))
    res.metrics["n_persons"] = len(persons)
    res.metrics["n_transactions"] = len(transactions)
    return res


if __name__ == "__main__":
    s = ArtifactStore()
    print(run(s).summary())
