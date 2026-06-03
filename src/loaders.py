"""
Data Loading & Validation Layer  (src/loaders.py)
=================================================

PURPOSE
-------
The boundary between YOUR real data and the framework. This is the module you
configure first when plugging in a real portfolio. It:

  1. Maps your column names to the framework's canonical names (ColumnMapping).
  2. Validates the schema, dtypes, ranges, and key uniqueness (validate_persons,
     validate_transactions) — raising clear DataValidationError instead of
     letting bad data silently corrupt downstream metrics.
  3. Handles missing values, duplicates, and out-of-range PDs with explicit,
     logged policies (not silent coercion).
  4. Loads from CSV / Parquet / existing DataFrame with optional chunked reads
     for large files (load_persons, load_transactions).

WHY THIS MATTERS FOR REAL DATA
------------------------------
The synthetic generator produces perfectly clean data with fixed column names.
Real portfolios have: different column names, NaNs, duplicate IDs, PDs stored
as percentages (0–100 not 0–1), transactions referencing unknown persons, etc.
Every one of these silently breaks the copula or the loss-covariance math.
This module turns silent corruption into loud, early, actionable errors.

QUICK START
-----------
    from src.loaders import ColumnMapping, load_persons, load_transactions

    # Describe how YOUR columns map to the canonical schema:
    mapping = ColumnMapping(
        person_id="client_id",
        model_pd="pd_12m",              # your PD column
        city_id="region_code",
        income="monthly_income",
        # ... anything not specified uses the canonical default name
    )

    persons = load_persons("clients.parquet", mapping=mapping)
    transactions = load_transactions(
        "transfers.parquet",
        mapping=ColumnMapping(sender_id="from_acct",
                              receiver_id="to_acct",
                              amount="rub_amount"),
    )

    # persons now has canonical columns: person_id, model_pd, city_id, ...
    # and is validated, deduplicated, NaN-policed.

CANONICAL SCHEMA (what the framework expects after loading)
-----------------------------------------------------------
  persons:  person_id (int, unique), model_pd OR base_pd (float in [0,1]),
            optional: city_id, city_name, income, exposure_at_default,
            estimated_revenue, high_risk_group_id, risk_archetype, plus any
            feature columns for the PD model.
  transactions: sender_id, receiver_id, amount — all referencing person_id.

SCALE NOTE (10M+ borrowers)
---------------------------
  - Use Parquet, not CSV, for 10M rows (10–50x faster, typed).
  - load_persons(..., columns=[...]) reads only needed columns.
  - load_transactions(..., chunksize=...) streams large transaction files.
  - The framework NEVER builds a dense n×n matrix from these (see graph_features
    and risk_adjusted_metrics — they use sparse / block-local computation).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── exceptions ───────────────────────────────────────────────────────────────

class DataValidationError(ValueError):
    """
    Raised when input data fails schema, dtype, range, or key-integrity checks.

    Always carries an actionable message naming the column and the problem,
    so an operator (or agent) can fix the source data or the ColumnMapping.
    """


# ─── NaN / dedup policy ───────────────────────────────────────────────────────

NanPolicy = str  # one of: "error", "drop", "zero", "median"
VALID_NAN_POLICIES = ("error", "drop", "zero", "median")

DuplicatePolicy = str  # one of: "error", "first", "last"
VALID_DUP_POLICIES = ("error", "first", "last")


# ─── column mapping ───────────────────────────────────────────────────────────

@dataclass
class ColumnMapping:
    """
    Maps YOUR source column names to the framework's canonical names.

    Each attribute is the canonical name; the value you assign is the column
    name in your source data. Leave an attribute as its default (the canonical
    name itself) when your data already uses that name.

    Only the keys you actually have in your data matter — unspecified canonical
    columns are simply absent (the framework treats most as optional, with
    documented fallbacks).

    PERSONS canonical columns
    -------------------------
      person_id            (REQUIRED) unique borrower identifier
      model_pd             predicted PD in [0,1] (preferred PD source)
      base_pd              alternative PD column (used if model_pd absent)
      city_id              integer geography code
      city_name            geography label
      income               used for EAD proxy if exposure_at_default absent
      exposure_at_default  EAD (preferred for loss math)
      estimated_revenue    revenue (preferred for profit math)
      high_risk_group_id   group/cluster id, -1 = no group
      risk_archetype       segment label
      default              0/1 label (only needed to TRAIN a PD model)

    TRANSACTIONS canonical columns
    ------------------------------
      sender_id, receiver_id, amount
    """
    # persons
    person_id: str = "person_id"
    model_pd: str = "model_pd"
    base_pd: str = "base_pd"
    city_id: str = "city_id"
    city_name: str = "city_name"
    income: str = "income"
    exposure_at_default: str = "exposure_at_default"
    estimated_revenue: str = "estimated_revenue"
    high_risk_group_id: str = "high_risk_group_id"
    risk_archetype: str = "risk_archetype"
    default: str = "default"
    # transactions
    sender_id: str = "sender_id"
    receiver_id: str = "receiver_id"
    amount: str = "amount"

    def persons_rename_map(self, available: Sequence[str]) -> Dict[str, str]:
        """Return {source_name: canonical_name} for persons columns present in `available`."""
        person_keys = [
            "person_id", "model_pd", "base_pd", "city_id", "city_name",
            "income", "exposure_at_default", "estimated_revenue",
            "high_risk_group_id", "risk_archetype", "default",
        ]
        out = {}
        avail = set(available)
        for canonical in person_keys:
            source = getattr(self, canonical)
            if source in avail and source != canonical:
                out[source] = canonical
        return out

    def transactions_rename_map(self, available: Sequence[str]) -> Dict[str, str]:
        """Return {source_name: canonical_name} for transaction columns present in `available`."""
        out = {}
        avail = set(available)
        for canonical in ("sender_id", "receiver_id", "amount"):
            source = getattr(self, canonical)
            if source in avail and source != canonical:
                out[source] = canonical
        return out


# default mapping (identity) for when data already uses canonical names
DEFAULT_MAPPING = ColumnMapping()


# ─── low-level readers ────────────────────────────────────────────────────────

def _read_any(
    source: Union[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read a CSV/Parquet file or pass through a DataFrame.

    Parameters
    ----------
    source : str | pd.DataFrame
        Path ending in .csv/.parquet/.pq, or an in-memory DataFrame.
    columns : list[str], optional
        Restrict to these columns (huge speed-up for wide tables).
        NOTE: these are SOURCE column names (pre-mapping).
    chunksize : int, optional
        If set and source is a CSV, read and concatenate in chunks
        (lower peak memory for very large files). Ignored for Parquet/DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        df = source
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return df.copy()

    if not isinstance(source, str):
        raise DataValidationError(
            f"source must be a file path or DataFrame, got {type(source).__name__}."
        )
    if not os.path.exists(source):
        raise DataValidationError(f"File not found: {source}")

    ext = os.path.splitext(source)[1].lower()
    if ext in (".parquet", ".pq"):
        # Parquet supports efficient column projection natively.
        return pd.read_parquet(source, columns=columns)
    if ext in (".csv", ".txt", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        if chunksize:
            parts = []
            for chunk in pd.read_csv(source, sep=sep, usecols=columns,
                                     chunksize=chunksize):
                parts.append(chunk)
            return pd.concat(parts, ignore_index=True)
        return pd.read_csv(source, sep=sep, usecols=columns)
    raise DataValidationError(
        f"Unsupported file extension '{ext}'. Use .parquet, .csv, or .tsv."
    )


# ─── normalisation helpers ────────────────────────────────────────────────────

def _coerce_pd_range(series: pd.Series, col: str) -> pd.Series:
    """
    Ensure a PD column is in [0,1]. If values look like percentages (max > 1.5
    and <= 100), divide by 100 with a warning. Otherwise clip with a warning.
    """
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        raise DataValidationError(f"PD column '{col}' has no finite values.")
    mx = float(finite.max())

    # Percentage-scale detection: decide on the MEDIAN, not the max, so a single
    # large outlier (e.g. a 110% data-entry error) does not suppress the rescale,
    # and a column already in [0,1] with one bad value is not wrongly divided.
    median = float(finite.median())
    looks_like_percentage = (median > 1.0) and (mx <= 100.0 + 1e-9)
    # Also treat slightly-over-100 as percentages if the bulk of values are >1
    # (i.e. clearly on a 0–100 scale with a few out-of-range outliers).
    frac_above_one = float((finite > 1.0).mean())
    if not looks_like_percentage and frac_above_one >= 0.5 and mx <= 150.0:
        looks_like_percentage = True

    if looks_like_percentage:
        logger.warning(
            "PD column '%s' (median=%.2f, max=%.2f) looks like a percentage; "
            "dividing by 100.", col, median, mx,
        )
        s = s / 100.0

    if (s < 0).any() or (s > 1).any():
        n_bad = int(((s < 0) | (s > 1)).sum())
        logger.warning(
            "PD column '%s' has %d values outside [0,1] after scaling; clipping.",
            col, n_bad,
        )
        s = s.clip(0.0, 1.0)
    return s


def _apply_nan_policy(
    df: pd.DataFrame,
    col: str,
    policy: NanPolicy,
) -> pd.DataFrame:
    """Apply a missing-value policy to one column. Returns possibly-filtered df."""
    if policy not in VALID_NAN_POLICIES:
        raise DataValidationError(
            f"nan_policy must be one of {VALID_NAN_POLICIES}, got '{policy}'."
        )
    n_nan = int(df[col].isna().sum())
    if n_nan == 0:
        return df
    if policy == "error":
        raise DataValidationError(
            f"Column '{col}' has {n_nan} missing values and nan_policy='error'. "
            f"Set nan_policy to 'drop', 'zero', or 'median', or clean the source."
        )
    if policy == "drop":
        logger.warning("Dropping %d rows with NaN in '%s'.", n_nan, col)
        return df[df[col].notna()].copy()
    if policy == "zero":
        logger.warning("Filling %d NaN in '%s' with 0.", n_nan, col)
        df = df.copy()
        df[col] = df[col].fillna(0)
        return df
    if policy == "median":
        med = df[col].median()
        logger.warning("Filling %d NaN in '%s' with median=%.4g.", n_nan, col, med)
        df = df.copy()
        df[col] = df[col].fillna(med)
        return df
    return df


# ─── public: persons ──────────────────────────────────────────────────────────

def load_persons(
    source: Union[str, pd.DataFrame],
    *,
    mapping: ColumnMapping = DEFAULT_MAPPING,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    duplicate_policy: DuplicatePolicy = "error",
    pd_nan_policy: NanPolicy = "error",
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load and validate a persons table from file or DataFrame.

    Parameters
    ----------
    source : str | pd.DataFrame
        Path to .parquet/.csv, or an in-memory DataFrame.
    mapping : ColumnMapping
        How your column names map to canonical names.
    columns : list[str], optional
        Source column names to read (speed-up). If given, MUST include the
        person_id source column and at least one PD source column.
    chunksize : int, optional
        Stream large CSVs in chunks.
    duplicate_policy : {"error","first","last"}
        What to do with duplicate person_id. Default "error" (safest).
    pd_nan_policy : {"error","drop","zero","median"}
        What to do with missing PD values. Default "error".
    validate : bool
        Run validate_persons() after loading (default True).

    Returns
    -------
    pd.DataFrame with canonical column names, unique integer person_id,
    PD column in [0,1], ready for the pipeline.

    Raises
    ------
    DataValidationError on any schema/integrity problem.
    """
    raw = _read_any(source, columns=columns, chunksize=chunksize)
    rename = mapping.persons_rename_map(raw.columns)
    df = raw.rename(columns=rename)

    # person_id is mandatory
    if "person_id" not in df.columns:
        raise DataValidationError(
            f"No person_id column found. Expected source column "
            f"'{mapping.person_id}' (configure ColumnMapping.person_id)."
        )

    # person_id → int, deduplicate
    df = _normalise_person_id(df, duplicate_policy)

    # Resolve a usable PD column into 'model_pd'
    df = _resolve_pd_column(df, pd_nan_policy)

    if validate:
        validate_persons(df)

    logger.info("Loaded persons: %d rows, %d columns.", len(df), df.shape[1])
    return df


def _normalise_person_id(df: pd.DataFrame, duplicate_policy: DuplicatePolicy) -> pd.DataFrame:
    """Coerce person_id to int and handle duplicates per policy."""
    if duplicate_policy not in VALID_DUP_POLICIES:
        raise DataValidationError(
            f"duplicate_policy must be one of {VALID_DUP_POLICIES}, got '{duplicate_policy}'."
        )
    df = df.copy()
    # Coerce to integer where possible
    try:
        df["person_id"] = pd.to_numeric(df["person_id"], errors="raise").astype("int64")
    except Exception as e:
        raise DataValidationError(
            f"person_id could not be converted to integers: {e}. "
            f"The framework uses person_id for positional numpy indexing; "
            f"it must be integer-valued."
        )
    dup = df["person_id"].duplicated(keep=False)
    n_dup = int(dup.sum())
    if n_dup > 0:
        if duplicate_policy == "error":
            sample = df.loc[dup, "person_id"].unique()[:5].tolist()
            raise DataValidationError(
                f"person_id has {n_dup} duplicate rows (e.g. {sample}). "
                f"Set duplicate_policy='first' or 'last' to deduplicate, "
                f"or fix the source."
            )
        keep = "first" if duplicate_policy == "first" else "last"
        before = len(df)
        df = df.drop_duplicates(subset="person_id", keep=keep).copy()
        logger.warning("Dropped %d duplicate person_id rows (kept %s).",
                       before - len(df), keep)
    return df.reset_index(drop=True)


def _resolve_pd_column(df: pd.DataFrame, pd_nan_policy: NanPolicy) -> pd.DataFrame:
    """
    Ensure df has a valid 'model_pd' column in [0,1].
    Priority: model_pd → base_pd. Applies NaN policy and range coercion.
    """
    df = df.copy()
    pd_col = None
    if "model_pd" in df.columns:
        pd_col = "model_pd"
    elif "base_pd" in df.columns:
        pd_col = "base_pd"
        logger.info("No 'model_pd' column; using 'base_pd' as the PD source.")

    if pd_col is None:
        # Not necessarily fatal: a PD model may be trained later if 'default' exists.
        if "default" in df.columns:
            logger.warning(
                "No PD column found, but 'default' label is present. "
                "Train a PD model (IndividualPDModel) to create 'model_pd' "
                "before fitting the copula."
            )
            return df
        raise DataValidationError(
            "No PD column found (looked for 'model_pd' and 'base_pd'), and no "
            "'default' label to train one. Provide a PD column via ColumnMapping."
        )

    df = _apply_nan_policy(df, pd_col, pd_nan_policy)
    df[pd_col] = _coerce_pd_range(df[pd_col], pd_col)

    # Canonicalise to model_pd so the rest of the pipeline finds it.
    if pd_col == "base_pd" and "model_pd" not in df.columns:
        df["model_pd"] = df["base_pd"].values
    return df


def validate_persons(df: pd.DataFrame) -> None:
    """
    Validate a persons DataFrame against the canonical schema.

    Checks:
      - person_id present, integer, unique
      - a PD column (model_pd or base_pd) present and in [0,1]
      - no NaN in the PD column
      - exposure_at_default / income, if present, are non-negative

    Raises DataValidationError on the first violation found.
    """
    if "person_id" not in df.columns:
        raise DataValidationError("persons missing required column 'person_id'.")
    if df["person_id"].duplicated().any():
        raise DataValidationError("persons has duplicate person_id values.")
    if not pd.api.types.is_integer_dtype(df["person_id"]):
        raise DataValidationError("person_id must be integer dtype.")

    pd_col = "model_pd" if "model_pd" in df.columns else (
        "base_pd" if "base_pd" in df.columns else None
    )
    if pd_col is None:
        raise DataValidationError(
            "persons has no PD column ('model_pd' or 'base_pd')."
        )
    if df[pd_col].isna().any():
        raise DataValidationError(f"PD column '{pd_col}' contains NaN.")
    if not df[pd_col].between(0, 1).all():
        raise DataValidationError(f"PD column '{pd_col}' has values outside [0,1].")

    for money_col in ("exposure_at_default", "income", "estimated_revenue"):
        if money_col in df.columns:
            vals = pd.to_numeric(df[money_col], errors="coerce")
            if (vals < 0).any():
                raise DataValidationError(
                    f"Column '{money_col}' has negative values; "
                    f"exposures/revenue must be ≥ 0."
                )


# ─── public: transactions ─────────────────────────────────────────────────────

def load_transactions(
    source: Union[str, pd.DataFrame],
    *,
    mapping: ColumnMapping = DEFAULT_MAPPING,
    chunksize: Optional[int] = None,
    valid_person_ids: Optional[Sequence[int]] = None,
    drop_unknown: bool = True,
    drop_self_loops: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load and validate a transactions table from file or DataFrame.

    Parameters
    ----------
    source : str | pd.DataFrame
        Path to .parquet/.csv, or an in-memory DataFrame.
    mapping : ColumnMapping
        How your sender/receiver/amount columns map to canonical names.
    chunksize : int, optional
        Stream large CSVs in chunks.
    valid_person_ids : sequence[int], optional
        If provided, transactions referencing ids NOT in this set are handled
        per `drop_unknown`. Strongly recommended: pass persons['person_id'].
    drop_unknown : bool
        If True (default), drop transactions referencing unknown person_ids
        (with a warning). If False, raise DataValidationError instead.
    drop_self_loops : bool
        If True (default), drop transactions where sender == receiver.
    validate : bool
        Run validate_transactions() after loading.

    Returns
    -------
    pd.DataFrame with canonical columns sender_id, receiver_id, amount.

    SCALE NOTE: for tens of millions of transactions, pass a Parquet path and
    a chunksize; this function concatenates lazily and only keeps the three
    canonical columns.
    """
    raw = _read_any(source, chunksize=chunksize)
    rename = mapping.transactions_rename_map(raw.columns)
    df = raw.rename(columns=rename)

    required = {"sender_id", "receiver_id", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"transactions missing columns {sorted(missing)}. "
            f"Configure ColumnMapping.sender_id/receiver_id/amount. "
            f"Available source columns: {list(raw.columns)[:20]}."
        )

    df = df[["sender_id", "receiver_id", "amount"]].copy()

    # Coerce types
    for col in ("sender_id", "receiver_id"):
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
        except Exception as e:
            raise DataValidationError(f"transactions['{col}'] not integer-coercible: {e}.")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop NaN / non-positive amounts
    n_bad_amt = int((~np.isfinite(df["amount"]) | (df["amount"] <= 0)).sum())
    if n_bad_amt > 0:
        logger.warning("Dropping %d transactions with NaN/non-positive amount.", n_bad_amt)
        df = df[np.isfinite(df["amount"]) & (df["amount"] > 0)].copy()

    # Self-loops
    if drop_self_loops:
        n_self = int((df["sender_id"] == df["receiver_id"]).sum())
        if n_self > 0:
            logger.warning("Dropping %d self-loop transactions (sender==receiver).", n_self)
            df = df[df["sender_id"] != df["receiver_id"]].copy()

    # Unknown person references
    if valid_person_ids is not None:
        valid = set(int(x) for x in valid_person_ids)
        mask_known = df["sender_id"].isin(valid) & df["receiver_id"].isin(valid)
        n_unknown = int((~mask_known).sum())
        if n_unknown > 0:
            if drop_unknown:
                logger.warning(
                    "Dropping %d transactions referencing unknown person_ids.", n_unknown
                )
                df = df[mask_known].copy()
            else:
                raise DataValidationError(
                    f"{n_unknown} transactions reference person_ids not present in "
                    f"persons. Set drop_unknown=True or reconcile the data."
                )

    df = df.reset_index(drop=True)
    if validate:
        validate_transactions(df, valid_person_ids=valid_person_ids)

    logger.info("Loaded transactions: %d edges.", len(df))
    return df


def validate_transactions(
    df: pd.DataFrame,
    valid_person_ids: Optional[Sequence[int]] = None,
) -> None:
    """
    Validate a transactions DataFrame.

    Checks:
      - sender_id, receiver_id, amount columns present
      - sender_id, receiver_id integer dtype
      - amount finite and positive
      - (optional) all ids present in valid_person_ids

    Raises DataValidationError on the first violation.
    """
    required = {"sender_id", "receiver_id", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"transactions missing columns {sorted(missing)}.")
    for col in ("sender_id", "receiver_id"):
        if not pd.api.types.is_integer_dtype(df[col]):
            raise DataValidationError(f"transactions['{col}'] must be integer dtype.")
    if not np.isfinite(df["amount"]).all():
        raise DataValidationError("transactions['amount'] contains NaN/inf.")
    if (df["amount"] <= 0).any():
        raise DataValidationError("transactions['amount'] has non-positive values.")
    if valid_person_ids is not None:
        valid = set(int(x) for x in valid_person_ids)
        unknown = (~df["sender_id"].isin(valid)) | (~df["receiver_id"].isin(valid))
        if unknown.any():
            raise DataValidationError(
                f"{int(unknown.sum())} transactions reference unknown person_ids."
            )


# ─── public: reindex to contiguous range ──────────────────────────────────────

def reindex_to_contiguous(
    persons: pd.DataFrame,
    transactions: Optional[pd.DataFrame] = None,
) -> tuple:
    """
    Remap arbitrary integer person_ids to a contiguous 0..n-1 range.

    The framework uses person_id for positional numpy indexing, which REQUIRES
    ids to be 0..n-1. Real data often has sparse/large ids (account numbers,
    hashes). This function remaps them and rewrites transaction references.

    Parameters
    ----------
    persons : pd.DataFrame  (must have person_id)
    transactions : pd.DataFrame, optional  (sender_id, receiver_id)

    Returns
    -------
    (persons_reindexed, transactions_reindexed, id_map)
      id_map : dict {original_id: new_contiguous_id}
      The original ids are preserved in a new column 'original_person_id'.

    AGENT NOTE: Always call this if your person_id is not already 0..n-1.
    The pipeline's positional indexing will produce wrong results otherwise.
    """
    persons = persons.copy()
    original = persons["person_id"].values
    id_map = {int(orig): i for i, orig in enumerate(original)}

    persons["original_person_id"] = original
    persons["person_id"] = np.arange(len(persons), dtype="int64")

    tx_out = None
    if transactions is not None:
        tx_out = transactions.copy()
        tx_out["sender_id"] = tx_out["sender_id"].map(id_map)
        tx_out["receiver_id"] = tx_out["receiver_id"].map(id_map)
        # Drop any rows that referenced ids not in persons (became NaN)
        before = len(tx_out)
        tx_out = tx_out[tx_out["sender_id"].notna() & tx_out["receiver_id"].notna()].copy()
        if len(tx_out) < before:
            logger.warning(
                "reindex: dropped %d transactions referencing ids not in persons.",
                before - len(tx_out),
            )
        tx_out["sender_id"] = tx_out["sender_id"].astype("int64")
        tx_out["receiver_id"] = tx_out["receiver_id"].astype("int64")
        tx_out = tx_out.reset_index(drop=True)

    logger.info("Reindexed %d persons to contiguous 0..%d.", len(persons), len(persons) - 1)
    return persons, tx_out, id_map


# ─── summary / profiling ──────────────────────────────────────────────────────

def describe_persons(df: pd.DataFrame) -> Dict[str, object]:
    """
    Return a compact, JSON-safe summary of a loaded persons table.

    Useful for an agent or operator to sanity-check the data before running the
    pipeline: row count, PD distribution, missing-value counts, id range.
    """
    pd_col = "model_pd" if "model_pd" in df.columns else (
        "base_pd" if "base_pd" in df.columns else None
    )
    out: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "person_id_min": int(df["person_id"].min()) if "person_id" in df else None,
        "person_id_max": int(df["person_id"].max()) if "person_id" in df else None,
        "person_id_contiguous": bool(
            "person_id" in df
            and df["person_id"].min() == 0
            and df["person_id"].max() == len(df) - 1
        ),
        "pd_column": pd_col,
    }
    if pd_col:
        s = df[pd_col]
        out["pd_min"] = float(s.min())
        out["pd_mean"] = float(s.mean())
        out["pd_max"] = float(s.max())
    # Missing-value counts for key columns
    missing = {}
    for c in ("model_pd", "base_pd", "income", "exposure_at_default",
              "estimated_revenue", "city_id", "high_risk_group_id"):
        if c in df.columns:
            missing[c] = int(df[c].isna().sum())
    out["missing_counts"] = missing
    return out
