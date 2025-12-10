from dataclasses import dataclass


@dataclass
class DataConfig:
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
