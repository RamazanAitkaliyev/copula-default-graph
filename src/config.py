"""
Configuration classes for the Copula Default Graph framework.

Provides centralized configuration for:
- Network generation parameters
- Copula model settings
- Risk analysis parameters
- Visualization options
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List


@dataclass
class NetworkConfig:
    """Configuration for synthetic network generation."""

    # Population settings
    n_cities: int = 3
    total_population: int = 1000

    # Risk structure
    n_high_risk_groups: int = 4
    group_size_range: Tuple[int, int] = (10, 20)
    n_bridges: int = 15

    # Risk archetype distribution
    low_risk_pct: float = 0.70
    medium_risk_pct: float = 0.20
    high_risk_pct: float = 0.10

    # Transaction patterns
    transactions_per_person: int = 8
    within_city_prob: float = 0.80
    within_group_prob: float = 0.60

    # Random seed
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_cities <= 0:
            raise ValueError("n_cities must be positive")
        if self.total_population <= 0:
            raise ValueError("total_population must be positive")
        if self.n_high_risk_groups < 0:
            raise ValueError("n_high_risk_groups cannot be negative")
        if not (0 <= self.within_city_prob <= 1):
            raise ValueError("within_city_prob must be in [0, 1]")

        total_pct = self.low_risk_pct + self.medium_risk_pct + self.high_risk_pct
        if abs(total_pct - 1.0) > 0.01:
            raise ValueError(
                f"Risk percentages must sum to 1.0, got {total_pct:.2f}"
            )


@dataclass
class CopulaConfig:
    """Configuration for copula model."""

    # Copula type
    copula_type: Literal['gaussian', 'student_t', 'clayton', 'gumbel', 'frank'] = 'clayton'

    # Student-t degrees of freedom (only used for student_t copula)
    nu: float = 4.0

    # Correlation matrix parameters
    base_correlation: float = 0.05
    max_correlation: float = 0.60
    same_city_boost: float = 0.10
    same_group_boost: float = 0.20

    # Simulation settings
    default_n_simulations: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_copulas = ('gaussian', 'student_t', 'clayton', 'gumbel', 'frank')
        if self.copula_type not in valid_copulas:
            raise ValueError(
                f"copula_type must be one of {valid_copulas}, got {self.copula_type}"
            )
        if self.nu <= 0:
            raise ValueError("nu must be positive")
        if not (0 <= self.base_correlation <= 1):
            raise ValueError("base_correlation must be in [0, 1]")
        if not (0 <= self.max_correlation <= 1):
            raise ValueError("max_correlation must be in [0, 1]")
        if self.default_n_simulations <= 0:
            raise ValueError("default_n_simulations must be positive")


@dataclass
class RiskConfig:
    """Configuration for risk analysis."""

    # Loss given default
    lgd: float = 0.45

    # Risk tier thresholds (percentiles)
    medium_threshold: float = 0.60
    high_threshold: float = 0.85
    critical_threshold: float = 0.95

    # Composite score weights
    marginal_pd_weight: float = 0.40
    network_exposure_weight: float = 0.25
    vulnerability_weight: float = 0.20
    importance_weight: float = 0.15

    # Stress test defaults
    default_pd_multiplier: float = 2.0
    default_correlation_boost: float = 0.20

    # Contagion settings
    contagion_threshold: float = 0.50
    max_contagion_rounds: int = 5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0 <= self.lgd <= 1):
            raise ValueError("lgd must be in [0, 1]")

        weights_sum = (
            self.marginal_pd_weight +
            self.network_exposure_weight +
            self.vulnerability_weight +
            self.importance_weight
        )
        if abs(weights_sum - 1.0) > 0.01:
            raise ValueError(
                f"Composite score weights must sum to 1.0, got {weights_sum:.2f}"
            )


@dataclass
class VisualizationConfig:
    """Configuration for network visualization."""

    # Figure settings
    figsize: Tuple[int, int] = (14, 10)
    dpi: int = 150

    # Node settings
    node_alpha: float = 0.7
    min_node_size: int = 50
    max_node_size: int = 250
    node_edge_color: str = 'white'
    node_edge_width: float = 0.5

    # Edge settings
    edge_alpha: float = 0.1
    edge_width: float = 0.3
    edge_color: str = 'black'

    # Colormap
    default_cmap: str = 'RdYlGn_r'

    # Layout
    default_layout: Literal['spring', 'city', 'circular'] = 'city'
    spring_iterations: int = 50


@dataclass
class PipelineConfig:
    """Combined configuration for the full pipeline."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    copula: CopulaConfig = field(default_factory=CopulaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Output settings
    output_dir: str = 'output'
    save_figures: bool = True
    save_csv: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict) -> PipelineConfig:
        """Create configuration from dictionary."""
        network = NetworkConfig(**config_dict.get('network', {}))
        copula = CopulaConfig(**config_dict.get('copula', {}))
        risk = RiskConfig(**config_dict.get('risk', {}))
        visualization = VisualizationConfig(**config_dict.get('visualization', {}))

        return cls(
            network=network,
            copula=copula,
            risk=risk,
            visualization=visualization,
            output_dir=config_dict.get('output_dir', 'output'),
            save_figures=config_dict.get('save_figures', True),
            save_csv=config_dict.get('save_csv', True),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)


# Default configurations for common use cases
DEFAULT_CONFIG = PipelineConfig()

STRESS_TEST_CONFIG = PipelineConfig(
    copula=CopulaConfig(copula_type='clayton', base_correlation=0.10),
    risk=RiskConfig(lgd=0.60, default_pd_multiplier=3.0),
)

LOW_CORRELATION_CONFIG = PipelineConfig(
    copula=CopulaConfig(
        copula_type='gaussian',
        base_correlation=0.02,
        max_correlation=0.30,
    ),
)
