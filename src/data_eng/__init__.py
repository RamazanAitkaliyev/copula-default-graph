"""
src.data_eng — Data Engineering role view (ingestion, validation, config).

A NAVIGATION facade: it re-exports the flat modules a Data Engineer owns so you
can do `from src.data_eng import load_persons`. The implementation still lives in
the flat `src/*.py` modules (no files moved); this package only groups them by
role. See ROLES.md.

Owns: getting clean, correctly-shaped data into the platform.
"""
from ..loaders import (
    ColumnMapping,
    DataValidationError,
    load_persons,
    load_transactions,
    validate_persons,
    validate_transactions,
    reindex_to_contiguous,
    describe_persons,
)
from ..data_generator import generate_network, CityConfig, get_summary_stats
from ..config import (
    NetworkConfig,
    CopulaConfig,
    RiskConfig,
    VisualizationConfig,
    PipelineConfig,
    DEFAULT_CONFIG,
)

__all__ = [
    "ColumnMapping", "DataValidationError", "load_persons", "load_transactions",
    "validate_persons", "validate_transactions", "reindex_to_contiguous",
    "describe_persons", "generate_network", "CityConfig", "get_summary_stats",
    "NetworkConfig", "CopulaConfig", "RiskConfig", "VisualizationConfig",
    "PipelineConfig", "DEFAULT_CONFIG",
]
