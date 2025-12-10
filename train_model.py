import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from src.models import classifier
from src.utils import setup_logging
from src.data import (SerbianTextPreprocessor, create_default_filters, StratifiedDataSplitter)
from src.tokenization import WordPieceTokenizer
from src.models import create_classifier_model, calculate_class_weights
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_ad_data(json_path: str, max_samples: int = 10000):
    """Load ad data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract data array
    ads = data['data'][:max_samples]

    # Convert to DataFrame
    df = pd.DataFrame({
        'text': [ad['name'] for ad in ads],
        'group_id': [ad['group_id'] for ad in ads]
    })

    return df


def main():
    """Run training on real data."""

    logger = setup_logging(log_level='INFO', experiment_name='train_model')
    logger.info("="*60)
    logger.info("TRAINING: Serbian Ad Classifier on Real Data")
    logger.info("="*60)
    logger.info("\n[1/6] LOADING DATA")
    logger.info("-" * 60)

    data_path = 'data/telefoni_json.json'
    data = load_ad_data(data_path, max_samples=30000)

    logger.info(f"Loaded {len(data)} ads from {data_path}")
    logger.info(f"Unique categories: {data['group_id'].nunique()}")
    logger.info(f"Class distribution (top 10):\n{data['group_id'].value_counts().head(10)}")
    logger.info("\n[2/6] TEXT PREPROCESSING")
    logger.info("-" * 60)

    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=False,
        verbose=False
    )

    data = preprocessor.preprocess_dataframe(data, 'text', 'clean_text')
    
    logger.info(f"Example:\n  Original: {data['text'].iloc[0]}\n  Cleaned:  {data['clean_text'].iloc[0]}")
    logger.info("\n[3/6] FILTERING DATA")
    logger.info("-" * 60)

    filters = create_default_filters(
        'group_id',
        'clean_text',
        min_samples_per_class=10,
        remove_ostalo=True
    )
    data = filters.filter(data)

    logger.info(f"After filtering: {len(data)} samples, {data['group_id'].nunique()} classes")

    splitter = StratifiedDataSplitter('group_id', val_size=0.15, test_size=0.15, verbose=True)
    train_data, val_data, test_data = splitter.split(data)

    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    logger.info("\n[4/6] TOKENIZATION")
    logger.info("-" * 60)

    tokenizer = WordPieceTokenizer(vocab_size=5000, max_length=64, verbose=True)
    tokenizer.train(train_data['clean_text'], use_cache=True)

    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info(f"Max sequence length: {tokenizer.max_length}")

    X_train = tokenizer.encode_batch(train_data['clean_text'].tolist())
    X_val = tokenizer.encode_batch(val_data['clean_text'].tolist())
    X_test = tokenizer.encode_batch(test_data['clean_text'].tolist())

    unique_classes = sorted(data['group_id'].unique())
    class_map = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}

    y_train = np.array([class_map[x] for x in train_data['group_id'].values])
    y_val = np.array([class_map[x] for x in val_data['group_id'].values])
    y_test = np.array([class_map[x] for x in test_data['group_id'].values])

    num_classes = len(unique_classes)
    logger.info(f"Tokenized data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info("\n[5/6] MODEL TRAINING")
    logger.info("-" * 60)

    classifier = create_classifier_model(
        vocab_size=tokenizer.get_vocab_size(),
        num_classes=num_classes,
        config={
            'max_length': 64,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'ff_dim': 512,
            'dropout_rate': 0.2,
            'pooling_strategy': 'cls',
            'label_smoothing': 0.0
        }
    )

    classifier.build_model()
    classifier.compile_model(learning_rate=1e-4, optimizer='adamw')

    logger.info("Model architecture:")
    logger.info(classifier.get_model_summary())

    class_counts = {i: np.sum(y_train == i) for i in range(num_classes)}
    class_weights = calculate_class_weights(class_counts, strategy='balanced')
    classifier.set_class_weights(class_weights)

    logger.info(f"Training with {len(class_weights)} class weights")

    logger.info("\nTraining model...")
    history = classifier.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )

    logger.info("\n[6/6] EVALUATION & SAVING")
    logger.info("-" * 60)

    test_results = classifier.model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"Test Loss: {test_results[0]:.4f}")
    logger.info(f"Test Accuracy: {test_results[1]:.4f}")
    logger.info(f"Test Top-3 Accuracy: {test_results[2]:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path('experiments') / f'model_{timestamp}'
    model_dir.mkdir(parents=True, exist_ok=True)

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
        'timestamp': timestamp,
        'num_classes': num_classes,
        'vocab_size': tokenizer.get_vocab_size(),
        'max_length': tokenizer.max_length,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'test_accuracy': float(test_results[1]),
        'test_loss': float(test_results[0])
    }

    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"\nModel artifacts saved to: {model_dir}")
    logger.info(f"Test Accuracy: {test_results[1]:.2%}")


if __name__ == "__main__":
    main()
