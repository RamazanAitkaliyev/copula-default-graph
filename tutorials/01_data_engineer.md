# Tutorial 01 — Data Engineer

**Your job:** get clean, correctly-shaped data into the platform. Everything
downstream trusts your contract, so validation is your product.

**Your modules:** `src/loaders.py`, `src/data_generator.py`, `src/config.py`
(import via `from src.data_eng import ...`).

---

## What you must deliver

1. A **persons** DataFrame: unique integer `person_id` (reindexed to 0..n-1), a
   PD column (`model_pd` or `base_pd`) in [0,1], optionally
   `geo_longitude`/`geo_latitude`, `city_id`, `exposure_at_default`,
   `estimated_revenue`, and `default` (0/1) if a PD model must be trained.
2. A **transactions** DataFrame: `sender_id`, `receiver_id`, `amount`.

If `person_id` is not 0..n-1, you MUST reindex — all downstream code uses
positional indexing.

---

## Step 1 — map your columns

```python
from src.data_eng import ColumnMapping, load_persons, load_transactions

pmap = ColumnMapping(
    person_id="client_id",
    model_pd="pd_12m",            # your bank's PD, in [0,1] or as a percentage
    geo_longitude="lon", geo_latitude="lat",
    city_id="region_code",
    exposure_at_default="ead",
    estimated_revenue="revenue",
)
persons = load_persons("clients.parquet", mapping=pmap,
                       duplicate_policy="first", pd_nan_policy="median")

tmap = ColumnMapping(sender_id="from_client", receiver_id="to_client", amount="amt")
tx = load_transactions("transfers.parquet", mapping=tmap,
                       valid_person_ids=persons["person_id"].tolist())
```

The loaders auto-detect percentage PDs (e.g. `2.5` → `0.025`) via a median
heuristic, apply your NaN/duplicate policies, drop transactions referencing
unknown persons (logged), and never silently corrupt — every problem is a loud
`DataValidationError` or a logged warning.

## Step 2 — make ids contiguous (REQUIRED if not already 0..n-1)

```python
from src.loaders import reindex_to_contiguous
persons, tx, id_map = reindex_to_contiguous(persons, tx)
# originals preserved in 'original_person_id'; id_map maps new → original
```

## Step 3 — validate and profile

```python
from src.loaders import validate_persons, validate_transactions, describe_persons
validate_persons(persons)        # raises DataValidationError on any violation
validate_transactions(tx)
print(describe_persons(persons))  # PD range, missingness, %-detection result
```

## Step 4 — hand off

Downstream code just needs `persons` and `tx`. To smoke-test the whole platform
on your data:

```python
from src.agents import RiskAgentAPI
api = RiskAgentAPI(persons=persons, transactions=tx)
api.run_pipeline()          # or supply model_pd and skip training
print(api.portfolio_summary().summary)
```

---

## Checklist before you ship data
- [ ] `person_id` unique, integer, **0..n-1** (use `reindex_to_contiguous`)
- [ ] PD column present, in [0,1] (check `describe_persons` didn't mis-detect %)
- [ ] geo columns within [-180,180]/[-90,90] if present
- [ ] transactions reference only known person_ids
- [ ] `validate_persons` / `validate_transactions` pass with no exception

**Pitfall:** if PDs come in as percentages but the median is ≤ 1 (e.g. mostly
tiny PDs with a few large), auto-detection may misfire — always eyeball
`describe_persons(persons)["pd_summary"]`.
