import argparse
import hashlib
from pathlib import Path

from ..utils.config.data_config import DataConfig
from ..utils.config import ConfigManager
from ..utils.logging import setup_logging, get_logger
from .loaders import create_data_loader
from .preprocessors import SerbianTextPreprocessor
from .filters import create_default_filters
import pandas as pd


def get_preprocessing_hash(data_config: DataConfig) -> str:
    hash_params = {
        'raw_data_path': str(data_config.raw_data_path),
        'min_samples_per_class': data_config.min_samples_per_class,
        'max_samples_per_class': data_config.max_samples_per_class,
        'remove_ostalo_groups': data_config.remove_ostalo_groups,
        'ostalo_groups_file': str(data_config.ostalo_groups_file) if data_config.remove_ostalo_groups else None,
    }
    hash_str = str(sorted(hash_params.items()))
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]


def get_processed_data_path(data_config: DataConfig) -> Path:
    base_dir = Path(data_config.processed_data_dir)
    config_hash = get_preprocessing_hash(data_config)
    return base_dir / config_hash


def preprocess_data(config_path: str = "configs/default.yaml"):
    setup_logging()
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Starting data preprocessing pipeline")
    logger.info("=" * 60)

    config = ConfigManager.from_yaml(config_path)
    data_config = config.data

    raw_data_path = data_config.raw_data_path
    processed_data_dir = get_processed_data_path(data_config)
    config_hash = get_preprocessing_hash(data_config)

    logger.info(f"Raw data path: {raw_data_path}")
    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Processed data directory: {processed_data_dir}")

    processed_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Loading raw data")
    logger.info("=" * 60)

    data_loader = create_data_loader(
        source=raw_data_path,
        data_config=data_config,
        loader_type="auto",
        verbose=True,
    )
    data = data_loader.load(raw_data_path)
    data = transform_columns(data, data_config)

    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Columns: {list(data.columns)}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Applying text preprocessing")
    logger.info("=" * 60)

    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=True,
        verbose=True
    )

    data = preprocessor.preprocess_dataframe(
        data=data,
        text_column=data_config.text_column,
        output_column=data_config.clean_text_column,
        inplace=False
    )

    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Applying data filters")
    logger.info("=" * 60)

    filters = create_default_filters(
        class_column=data_config.class_column,
        text_column=data_config.clean_text_column,
        min_samples_per_class=data_config.min_samples_per_class,
        max_samples_per_class=data_config.max_samples_per_class,
        ostalo_groups_file=data_config.ostalo_groups_file if data_config.remove_ostalo_groups else None,
        remove_ostalo=data_config.remove_ostalo_groups
    )

    data = filters.filter(data)

    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Saving processed data")
    logger.info("=" * 60)

    output_path = processed_data_dir / "processed_data.csv"
    data.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Processed data saved to: {output_path}")
    logger.info(f"Final dataset size: {len(data)} samples")

    metadata = {
        'total_samples': len(data),
        'num_classes': len(data[data_config.class_column].unique()),
        'columns': list(data.columns),
        'text_column': data_config.text_column,
        'class_column': data_config.class_column,
        'clean_text_column': data_config.clean_text_column
    }

    metadata_path = processed_data_dir / "metadata.json"
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to: {metadata_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples: {metadata['total_samples']}")
    logger.info(f"Number of classes: {metadata['num_classes']}")
    logger.info(f"Text column: {metadata['text_column']}")
    logger.info(f"Class column: {metadata['class_column']}")
    logger.info(f"Clean text column: {metadata['clean_text_column']}")

    class_distribution = data[data_config.class_column].value_counts()
    logger.info(f"\nTop 10 classes by sample count:")
    for class_id, count in class_distribution.head(10).items():
        logger.info(f"  Class {class_id}: {count} samples")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing completed successfully!")
    logger.info("=" * 60)

def transform_columns(data: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    ads = data['data']

    df = pd.DataFrame({
        data_config.text_column: [ad['name'] for ad in ads],
        data_config.class_column: [ad['group_id'] for ad in ads]
    })

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw data for advertisement classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)"
    )

    args = parser.parse_args()
    preprocess_data(config_path=args.config)


if __name__ == "__main__":
    main()
