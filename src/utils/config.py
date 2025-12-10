import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data processing."""
    raw_data_path: str = "data/ads-ai-20250109.json"
    processed_data_dir: str = "data/processed"

    text_column: str = "text"
    class_column: str = "group_id"
    clean_text_column: str = "clean_text"

    max_samples: int = 30000
    min_samples_per_class: int = 10
    max_samples_per_class: int = 125000
    remove_ostalo_groups: bool = True
    ostalo_groups_file: str = "data/ostalo-grupe.csv"

    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""
    vocab_size: int = 15000
    max_length: int = 100
    min_frequency: int = 2
    special_tokens: list = field(default_factory=lambda: ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"])
    use_cached_tokenizer: bool = True
    tokenizer_cache_dir: str = "cache/tokenizers"


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 512
    dropout_rate: float = 0.2
    pooling_strategy: str = "cls"
    label_smoothing: float = 0.0

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 10

    top_k: int = 3


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


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    save_dir: str = "experiments"
    model_name: str = "klasifikator"
    save_best_only: bool = True
    patience: int = 3
    monitor: str = "val_loss"
    mode: str = "min"

    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    min_lr: float = 1e-6

    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"

    checkpoint_freq: str = "epoch"
    save_frequency: int = 1


@dataclass
class Config:
    """Main configuration class."""
    project_name: str = "klasifikator"
    version: str = "1.0.0"
    experiment_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    data: DataConfig = field(default_factory=DataConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        """Set experiment name if not provided."""
        if self.experiment_name is None:
            self.experiment_name = f"{self.project_name}_{self.timestamp}"


class ConfigManager:
    """Configuration manager for loading and saving configs."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    @classmethod
    def from_yaml(cls, config_path: str) -> Config:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls._dict_to_config(config_dict)

    @classmethod
    def _dict_to_config(cls, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        data_config = DataConfig(**config_dict.get('data', {}))
        tokenization_config = TokenizationConfig(**config_dict.get('tokenization', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        balancing_config = BalancingConfig(**config_dict.get('balancing', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        wandb_config = WandbConfig(**config_dict.get('wandb', {}))

        main_config_dict = {k: v for k, v in config_dict.items()
                           if k not in ['data', 'tokenization', 'model', 'balancing', 'training', 'wandb']}

        return Config(
            data=data_config,
            tokenization=tokenization_config,
            model=model_config,
            balancing=balancing_config,
            training=training_config,
            wandb=wandb_config,
            **main_config_dict
        )

    def save_config(self, config: Config, filename: str) -> Path:
        """Save configuration to YAML file."""
        config_path = self.config_dir / filename
        config_dict = self._config_to_dict(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        return config_path

    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'project_name': config.project_name,
            'version': config.version,
            'experiment_name': config.experiment_name,
            'timestamp': config.timestamp,
            'data': config.data.__dict__,
            'tokenization': config.tokenization.__dict__,
            'model': config.model.__dict__,
            'balancing': config.balancing.__dict__,
            'training': config.training.__dict__,
            'wandb': config.wandb.__dict__
        }

    def create_default_config(self) -> Config:
        """Create default configuration."""
        return Config()


SERBIAN_STOP_WORDS = [
    'baš', 'bez', 'biće', 'bio', 'biti', 'blizu', 'broj', 'dana', 'danas', 'doći',
    'dobar', 'dobiti', 'dok', 'dole', 'došao', 'drugi', 'duž', 'dva', 'često', 'čiji',
    'gde', 'gore', 'hvala', 'ići', 'iako', 'ide', 'ima', 'imam', 'imao', 'ispod',
    'između', 'iznad', 'izvan', 'izvoli', 'jedan', 'jedini', 'jednom', 'jeste', 'još',
    'juče', 'kad', 'kako', 'kao', 'koga', 'koja', 'koje', 'koji', 'kroz', 'mali',
    'manji', 'misli', 'mnogo', 'moći', 'mogu', 'mora', 'morao', 'naći', 'naš', 'negde',
    'nego', 'nekad', 'neki', 'nemam', 'nešto', 'nije', 'nijedan', 'nikada', 'nismo',
    'ništa', 'njega', 'njegov', 'njen', 'njih', 'njihov', 'oko', 'okolo', 'ona',
    'onaj', 'oni', 'ono', 'osim', 'ostali', 'otišao', 'ovako', 'ovamo', 'ovde',
    'ove', 'ovo', 'pitati', 'početak', 'pojedini', 'posle', 'povodom', 'praviti',
    'pre', 'preko', 'prema', 'prvi', 'put', 'radije', 'sada', 'smeti', 'šta',
    'stvar', 'stvarno', 'sutra', 'svaki', 'sve', 'svim', 'svugde', 'tačno', 'tada',
    'taj', 'takođe', 'tamo', 'tim', 'učinio', 'učiniti', 'umalo', 'unutra',
    'upotrebiti', 'uzeti', 'vaš', 'većina', 'veoma', 'video', 'više', 'zahvaliti',
    'zašto', 'zbog', 'želeo', 'želi', 'znati', 'novo', 'nova', 'nove', 'novi',
    'nov', 'za', 'na', 'sa', 'od', 'do', 'iz', 'i'
]