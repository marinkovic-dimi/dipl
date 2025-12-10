"""Weights & Biases configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project: str = "klasifikator"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: Optional[str] = None
    log_model: bool = True
    log_gradients: bool = False
    log_frequency: int = 100
    save_code: bool = True
