"""
Artifact store + stage contract  (pipelines/artifacts.py)
=========================================================

The shared, on-disk data-passing layer between ETL stages. A stage never
imports another stage; it only reads/writes named artifacts through an
:class:`ArtifactStore`. This is what makes the stages independently ownable and
independently re-runnable.

Supported artifact formats (chosen by file extension):
    .parquet  → pandas DataFrame (preferred for tables; falls back to .csv)
    .csv      → pandas DataFrame
    .npy      → numpy array
    .json     → JSON-serialisable dict / list

Artifacts live under a single root directory (default ``output/etl``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Outcome of running one ETL stage."""
    stage: str
    ok: bool
    outputs: List[str] = field(default_factory=list)   # artifact names written
    metrics: Dict[str, Any] = field(default_factory=dict)  # small scalars to log
    seconds: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        status = "OK " if self.ok else "FAIL"
        head = f"[{status}] stage {self.stage} ({self.seconds:.2f}s)"
        if self.error:
            return f"{head}\n   error: {self.error}"
        out = ", ".join(self.outputs) if self.outputs else "(none)"
        met = "  ".join(f"{k}={v}" for k, v in self.metrics.items())
        return f"{head}\n   wrote: {out}" + (f"\n   {met}" if met else "")


class ArtifactStore:
    """
    A directory of named artifacts that stages read from and write to.

    Parameters
    ----------
    root : str or Path
        Directory under which artifacts are stored (created if missing).
    prefer_parquet : bool, default True
        Save DataFrames as parquet when a parquet engine is available; otherwise
        transparently fall back to CSV (so the project's minimal dependency set
        still works out of the box).
    """

    def __init__(self, root: str | Path = "output/etl", prefer_parquet: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefer_parquet = prefer_parquet and _parquet_available()

    # ── path helpers ──────────────────────────────────────────────────────────

    # Extensions tried (in order) when an artifact name is given without one.
    _RESOLVE_EXTS = (".parquet", ".csv", ".npy", ".json")

    def path(self, name: str) -> Path:
        """Absolute path for an artifact name (extension included)."""
        return self.root / name

    def resolve(self, name: str) -> Optional[Path]:
        """
        Resolve an artifact name to an existing file path, or None.

        If ``name`` already has a known extension, only that exact file is
        checked. Otherwise each candidate extension is tried in turn — so a
        stage can ``require("persons")`` regardless of whether it was persisted
        as parquet or csv.
        """
        if name.endswith(self._RESOLVE_EXTS):
            p = self.path(name)
            return p if p.exists() else None
        for ext in self._RESOLVE_EXTS:
            p = self.path(f"{name}{ext}")
            if p.exists():
                return p
        return None

    def exists(self, name: str) -> bool:
        return self.resolve(name) is not None

    def require(self, *names: str) -> None:
        """Raise a clear error if any required upstream artifact is missing."""
        missing = [n for n in names if not self.exists(n)]
        if missing:
            raise FileNotFoundError(
                f"Missing upstream artifact(s): {missing}. "
                f"Run the stage(s) that produce them first (store root: {self.root})."
            )

    def list(self) -> List[str]:
        """All artifact file names currently in the store."""
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    # ── DataFrame IO ──────────────────────────────────────────────────────────

    def write_df(self, name: str, df: pd.DataFrame) -> str:
        """
        Write a DataFrame. ``name`` may omit the extension; the store picks
        parquet or csv. Returns the actual artifact file name written.
        """
        stem = name.rsplit(".", 1)[0]
        if self.prefer_parquet:
            fname = f"{stem}.parquet"
            df.to_parquet(self.path(fname), index=False)
        else:
            fname = f"{stem}.csv"
            df.to_csv(self.path(fname), index=False)
        logger.info("wrote artifact %s (%d rows)", fname, len(df))
        return fname

    @staticmethod
    def _read_csv(p: Path) -> pd.DataFrame:
        """Read a CSV, tolerating a genuinely empty file (0 cols / 0 rows)."""
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            # A stage legitimately produced an empty result (e.g. no divergence
            # flags). Return an empty frame instead of raising.
            return pd.DataFrame()

    def read_df(self, name: str) -> pd.DataFrame:
        """
        Read a DataFrame by name. If ``name`` has no extension, try parquet then
        csv. Raises FileNotFoundError with guidance if neither exists. An empty
        CSV yields an empty DataFrame rather than an error.
        """
        if name.endswith((".parquet", ".csv")):
            p = self.path(name)
            if not p.exists():
                self.require(name)
            return pd.read_parquet(p) if name.endswith(".parquet") else self._read_csv(p)

        for ext, reader in ((".parquet", pd.read_parquet), (".csv", self._read_csv)):
            p = self.path(f"{name}{ext}")
            if p.exists():
                return reader(p)
        self.require(f"{name}.parquet")  # raises with guidance

    # ── numpy IO ──────────────────────────────────────────────────────────────

    def write_array(self, name: str, arr: np.ndarray) -> str:
        stem = name.rsplit(".", 1)[0]
        fname = f"{stem}.npy"
        np.save(self.path(fname), np.asarray(arr))
        logger.info("wrote artifact %s (shape %s)", fname, np.asarray(arr).shape)
        return fname

    def read_array(self, name: str) -> np.ndarray:
        stem = name if name.endswith(".npy") else f"{name}.npy"
        if not self.exists(stem):
            self.require(stem)
        return np.load(self.path(stem))

    # ── JSON IO ───────────────────────────────────────────────────────────────

    def write_json(self, name: str, obj: Any) -> str:
        stem = name.rsplit(".", 1)[0]
        fname = f"{stem}.json"
        with open(self.path(fname), "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        return fname

    def read_json(self, name: str) -> Any:
        stem = name if name.endswith(".json") else f"{name}.json"
        if not self.exists(stem):
            self.require(stem)
        with open(self.path(stem)) as f:
            return json.load(f)


def timed_stage(stage_name: str):
    """
    Decorator: wrap a ``run(store, **opts) -> StageResult`` stage function so it
    is timed and so uncaught exceptions become a ``StageResult(ok=False)``
    instead of crashing the whole pipeline. The wrapped function should populate
    ``result.outputs`` / ``result.metrics``; timing and error handling are added
    here.
    """
    def decorator(fn):
        def wrapper(store: ArtifactStore, **opts) -> StageResult:
            t0 = time.time()
            try:
                result = fn(store, **opts)
                if result is None:
                    result = StageResult(stage=stage_name, ok=True)
                result.seconds = time.time() - t0
                return result
            except Exception as exc:  # noqa: BLE001 - stages must not crash the chain
                logger.exception("stage %s failed", stage_name)
                return StageResult(
                    stage=stage_name, ok=False, seconds=time.time() - t0,
                    error=f"{type(exc).__name__}: {exc}",
                )
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
