import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / '.env')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager
from src.utils.callbacks import (
    TrainingPlotCallback,
    plot_confusion_matrix,
    plot_classification_report,
    plot_cumulative_accuracy
)
from src.data import StratifiedDataSplitter, DataBalancer
from src.data.preprocess import preprocess_data, get_processed_data_path
from src.tokenization import WordPieceTokenizer, TokenizedDatasetCache
from src.models import AdClassifier, calculate_class_weights


def load_or_preprocess_data(config):
    processed_dir = get_processed_data_path(config.data)
    processed_path = processed_dir / "processed_data.csv"
    metadata_path = processed_dir / "metadata.json"

    if processed_path.exists() and metadata_path.exists():
        logger = setup_logging(log_level=config.training.log_level)
        logger.info(f"Loading preprocessed data from {processed_path}")

        data = pd.read_csv(processed_path, encoding='utf-8')

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        logger.info(f"Loaded {metadata['total_samples']} samples with {metadata['num_classes']} classes")
        return data, metadata
    else:
        logger = setup_logging(log_level=config.training.log_level)
        logger.info("Preprocessed data not found. Running preprocessing pipeline...")

        preprocess_data(config_path=config.config_path)

        data = pd.read_csv(processed_path, encoding='utf-8')

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return data, metadata


def main(config_path: str = "configs/default.yaml"):
    config = ConfigManager.from_yaml(config_path, project_root=_project_root)
    config.config_path = config_path

    logger = setup_logging(
        log_level=config.training.log_level,
        experiment_name=f'train_wandb_{config.timestamp}'
    )

    logger.info("=" * 60)
    logger.info("TRAINING: Klasifikator")
    logger.info("=" * 60)
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Experiment: {config.experiment_name}")

    if config.wandb.name is None:
        config.wandb.name = f"train_{config.timestamp}"

    logger.info("\n[1/5] LOADING DATA")
    logger.info("-" * 60)

    data, metadata = load_or_preprocess_data(config)

    logger.info(f"Dataset: {len(data)} samples, {metadata['num_classes']} classes")
    logger.info(f"Columns: {metadata['columns']}")
    logger.info(f"Class distribution (top 10):\n{data[metadata['class_column']].value_counts().head(10)}")

    logger.info("\n[2/5] SPLITTING DATA")
    logger.info("-" * 60)

    class_col = metadata['class_column']
    clean_text_col = metadata['clean_text_column']

    splitter = StratifiedDataSplitter(
        class_col,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        verbose=True
    )
    train_data, val_data, test_data = splitter.split(data)

    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    balancer = None
    if config.balancing.strategy != "none":
        logger.info("\n[2.5/5] BALANCING TRAINING DATA")
        logger.info("-" * 60)

        # Get unique classes for statistics
        unique_classes = sorted(data[class_col].unique())
        num_classes = len(unique_classes)

        # Log BEFORE balancing
        original_train_size = len(train_data)
        logger.info(f"Data sizes BEFORE balancing:")
        logger.info(f"  Total train samples: {original_train_size:,}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Avg per class: {original_train_size / num_classes:.0f}")

        balancer = DataBalancer(
            config=config.balancing,
            class_column=class_col,
            random_state=config.data.random_state
        )
        train_data = balancer.balance(train_data)

        # Log AFTER balancing
        balanced_train_size = len(train_data)
        data_retained_pct = 100 * balanced_train_size / original_train_size
        logger.info(f"\nData sizes AFTER balancing:")
        logger.info(f"  Total train samples: {balanced_train_size:,}")
        logger.info(f"  Avg per class: {balanced_train_size / num_classes:.0f}")
        logger.info(f"  Data retained: {data_retained_pct:.1f}%")
        logger.info(f"  Data lost: {100 - data_retained_pct:.1f}%")

    logger.info("\n[3/5] TOKENIZATION")
    logger.info("-" * 60)

    # Initialize tokenized dataset cache manager
    dataset_cache = TokenizedDatasetCache(
        cache_dir=config.tokenization.tokenized_cache_dir,
        verbose=True
    )

    # Compute cache key based on all relevant configuration
    split_config = {
        'val_size': config.data.val_size,
        'test_size': config.data.test_size,
        'random_state': config.data.random_state,
    }
    cache_key = dataset_cache.compute_cache_key(
        data_config=config.data,
        tokenization_config=config.tokenization,
        split_config=split_config,
        balancing_config=config.balancing if balancer else None
    )

    # Try to load from cache
    use_cache = config.tokenization.use_tokenized_cache
    cache_loaded = False

    if use_cache and dataset_cache.exists(cache_key):
        try:
            logger.info(f"Loading tokenized datasets from cache: {cache_key}")
            X_train, X_val, X_test, y_train, y_val, y_test, cache_metadata = dataset_cache.load(cache_key)

            # Load class_map from metadata (convert string keys back to int)
            class_map = {int(k): v for k, v in cache_metadata['class_map'].items()}
            num_classes = cache_metadata['num_classes']

            # Still need to train/load tokenizer for model vocab_size
            tokenizer = WordPieceTokenizer(
                vocab_size=config.tokenization.vocab_size,
                max_length=config.tokenization.max_length,
                verbose=True
            )
            tokenizer.train(
                train_data[clean_text_col],
                use_cache=config.tokenization.use_cached_tokenizer
            )

            logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
            logger.info(f"Max sequence length: {tokenizer.max_length}")
            cache_loaded = True

        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load cache: {e}")
            logger.info("Will perform tokenization from scratch")
            cache_loaded = False

    if not cache_loaded:
        # Cache miss or disabled - perform tokenization
        logger.info("Performing tokenization...")

        tokenizer = WordPieceTokenizer(
            vocab_size=config.tokenization.vocab_size,
            max_length=config.tokenization.max_length,
            verbose=True
        )
        tokenizer.train(
            train_data[clean_text_col],
            use_cache=config.tokenization.use_cached_tokenizer
        )

        logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        logger.info(f"Max sequence length: {tokenizer.max_length}")

        # Tokenize datasets
        logger.info("Encoding training data...")
        X_train = tokenizer.encode_batch(train_data[clean_text_col].tolist())
        logger.info("Encoding validation data...")
        X_val = tokenizer.encode_batch(val_data[clean_text_col].tolist())
        logger.info("Encoding test data...")
        X_test = tokenizer.encode_batch(test_data[clean_text_col].tolist())

        # Encode labels
        unique_classes = sorted(data[class_col].unique())
        class_map = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}

        y_train = np.array([class_map[x] for x in train_data[class_col].values])
        y_val = np.array([class_map[x] for x in val_data[class_col].values])
        y_test = np.array([class_map[x] for x in test_data[class_col].values])

        num_classes = len(unique_classes)

        # Save to cache
        if use_cache:
            logger.info(f"Saving tokenized datasets to cache: {cache_key}")
            metadata = {
                'num_classes': num_classes,
                'class_map': {k: v for k, v in class_map.items()},
                'vocab_size': tokenizer.get_vocab_size(),
                'max_length': tokenizer.max_length,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
            }
            dataset_cache.save(
                cache_key=cache_key,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                metadata=metadata
            )

    logger.info(f"Tokenized data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"Number of classes: {num_classes}")

    logger.info("\n[4/5] MODEL TRAINING")
    logger.info("-" * 60)

    classifier = AdClassifier(
        vocab_size=tokenizer.get_vocab_size(),
        num_classes=num_classes,
        max_length=config.tokenization.max_length,
        embed_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        ff_dim=config.model.ff_dim,
        dropout_rate=config.model.dropout_rate,
        pooling_strategy=config.model.pooling_strategy,
        label_smoothing=config.model.label_smoothing
    )

    classifier.build_model()
    classifier.compile_model(
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
        optimizer='adamw'
    )

    logger.info("Model architecture:")
    logger.info(classifier.get_model_summary())

    class_counts = {i: np.sum(y_train == i) for i in range(num_classes)}
    class_weights = calculate_class_weights(class_counts, strategy='balanced')
    classifier.set_class_weights(class_weights)

    logger.info(f"Training with {len(class_weights)} class weights")

    model_dir = Path(config.training.save_dir) / f'model_wandb_{config.timestamp}'
    model_dir.mkdir(parents=True, exist_ok=True)

    config_manager = ConfigManager(config_dir=str(model_dir))
    config_manager.save_config(config, 'config.yaml')
    logger.info(f"Config saved to: {model_dir / 'config.yaml'}")

    if balancer is not None and balancer.stats is not None:
        balancer.plot_balancing_stats(output_path=str(model_dir / 'balancing_stats.png'))
        balancer.save_stats_json(output_path=str(model_dir / 'balancing_stats.json'))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.training.monitor,
        patience=config.training.patience,
        mode=config.training.mode,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_dir / 'checkpoint.keras'),
        monitor=config.training.monitor,
        save_best_only=config.training.save_best_only,
        mode=config.training.mode,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=config.training.monitor,
        factor=config.training.reduce_lr_factor,
        patience=config.training.reduce_lr_patience,
        min_lr=config.training.min_lr,
        verbose=1
    )

    training_plot = TrainingPlotCallback(
        output_dir=str(model_dir),
        plot_name='training_progress.png',
        csv_name='training_log.csv'
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr, training_plot]

    logger.info("\nTraining model with W&B logging...")
    history = classifier.train(
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        callbacks=callbacks,
        wandb_config=config.wandb,
        verbose=1
    )

    logger.info("\n[5/5] EVALUATION & SAVING")
    logger.info("-" * 60)

    test_results = classifier.model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"Test Loss: {test_results[0]:.4f}")
    logger.info(f"Test Accuracy: {test_results[1]:.4f}")
    logger.info(f"Test Top-3 Accuracy: {test_results[2]:.4f}")

    logger.info("\nGenerating evaluation plots...")
    y_pred_proba = classifier.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=str(model_dir / 'confusion_matrix.png'),
        title='Test Set Confusion Matrix'
    )

    report = plot_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        output_path=str(model_dir / 'classification_report.png'),
        top_n=min(20, num_classes)
    )

    report_path = model_dir / 'classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Classification report saved to: {report_path}")

    cumulative_acc = plot_cumulative_accuracy(
        y_true=y_test,
        y_pred_proba=y_pred_proba,
        output_path=str(model_dir / 'cumulative_accuracy.png'),
        max_k=5
    )
    logger.info(f"Cumulative accuracy: {cumulative_acc}")

    model_path = model_dir / 'classifier.keras'
    classifier.save_model(str(model_path), save_format='keras')
    logger.info(f"Model saved to: {model_path}")

    tokenizer_path = model_dir / 'tokenizer.json'
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer saved to: {tokenizer_path}")

    class_map_path = model_dir / 'class_map.json'
    class_map_serializable = {int(k): v for k, v in class_map.items()}
    with open(class_map_path, 'w') as f:
        json.dump(class_map_serializable, f, indent=2)
    logger.info(f"Class mapping saved to: {class_map_path}")

    metadata = {
        'timestamp': config.timestamp,
        'experiment_name': config.experiment_name,
        'num_classes': num_classes,
        'vocab_size': tokenizer.get_vocab_size(),
        'max_length': config.tokenization.max_length,
        'embed_dim': config.model.embedding_dim,
        'num_heads': config.model.num_heads,
        'num_layers': config.model.num_layers,
        'ff_dim': config.model.ff_dim,
        'dropout_rate': config.model.dropout_rate,
        'pooling_strategy': config.model.pooling_strategy,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'test_accuracy': float(test_results[1]),
        'test_loss': float(test_results[0]),
        'test_top3_accuracy': float(test_results[2]),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'cumulative_accuracy': cumulative_acc
    }

    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"\nModel artifacts saved to: {model_dir}")
    logger.info(f"Test Accuracy: {test_results[1]:.2%}")
    logger.info(f"Test Top-3 Accuracy: {test_results[2]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Klasifikator')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to YAML config file (relative to project root or absolute)'
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root / config_path

        tf.config.run_functions_eagerly(False)

    physical_devices = tf.config.list_physical_devices('GPU')

    try:

        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
    except:
        pass
    
    os.environ['PYTHONUNBUFFERED'] = '1'

    main(config_path=str(config_path))
