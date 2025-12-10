"""Data balancing configuration."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BalancingConfig:
    """Configuration for data balancing strategies."""
    strategy: str = "adaptive"

    target_threshold: int = 3000
    target_threshold_small: int = 500
    increase_factor: float = 0.1
    decrease_factor: float = 0.4
    decrease_factor_small: float = 0.5

    tier_limits: Dict[str, int] = field(default_factory=lambda: {
        "small": 50,
        "medium": 1000,
        "large": 10000,
        "xlarge": 20000,
        "xxlarge": 50000,
        "max": 125000
    })
