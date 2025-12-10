"""Example script demonstrating training with Weights & Biases logging."""

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils import Config, setup_logging, get_logger
from src.models import AdClassifier

setup_logging(log_level="INFO")
logger = get_logger(__name__)


def create_synthetic_data(num_samples=1000, vocab_size=5000, max_length=100, num_classes=50):
    """Create synthetic data for demonstration purposes."""
    logger.info(f"Creating synthetic dataset: {num_samples} samples")

    x = np.random.randint(0, vocab_size, size=(num_samples, max_length))
    y = np.random.randint(0, num_classes, size=(num_samples,))

    return x, y


def main():
    """Main training function with W&B logging."""

    logger.info("Loading configuration...")
    config = Config()

    config.wandb.enabled = True
    config.wandb.project = "klasifikator"
    config.wandb.name = f"training_run_{config.timestamp}"
    config.wandb.tags = ["demo", "synthetic_data", "transformer"]
    config.wandb.notes = "Example training run with synthetic data"
    config.wandb.log_model = True
    config.wandb.log_frequency = 50

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"W&B Project: {config.wandb.project}")
    logger.info(f"W&B Run: {config.wandb.name}")

    vocab_size = config.tokenization.vocab_size
    max_length = config.tokenization.max_length
    num_classes = 50

    logger.info("Creating synthetic training and validation data...")
    x_train, y_train = create_synthetic_data(
        num_samples=10000,
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=num_classes
    )

    x_val, y_val = create_synthetic_data(
        num_samples=2000,
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=num_classes
    )

    logger.info(f"Training set: {x_train.shape}, {y_train.shape}")
    logger.info(f"Validation set: {x_val.shape}, {y_val.shape}")

    logger.info("Building classifier model...")
    classifier = AdClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_length=max_length,
        embed_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        ff_dim=config.model.ff_dim,
        dropout_rate=config.model.dropout_rate,
        pooling_strategy='cls',
        activation='relu'
    )

    classifier.build_model()
    logger.info(f"Model built successfully")
    logger.info(f"\n{classifier.get_model_summary()}")

    logger.info("Compiling model...")
    classifier.compile_model(
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
        optimizer='adamw',
        top_k=config.model.top_k
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.training.monitor,
        patience=config.training.patience,
        mode=config.training.mode,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{config.training.save_dir}/{config.experiment_name}/checkpoint.keras",
        monitor=config.training.monitor,
        save_best_only=config.training.save_best_only,
        mode=config.training.mode,
        verbose=1
    )

    callbacks = [early_stopping, model_checkpoint]

    logger.info("Starting training with W&B logging...")
    history = classifier.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        callbacks=callbacks,
        wandb_config=config.wandb,
        verbose=1
    )

    logger.info("Training completed!")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")

    save_path = Path(config.training.save_dir) / config.experiment_name / "final_model.keras"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    classifier.save_model(str(save_path))
    logger.info(f"Model saved to {save_path}")

    logger.info("Making predictions on validation set...")
    predicted_classes = classifier.predict_classes(x_val[:10])

    logger.info(f"Sample predictions (first 10):")
    for i in range(10):
        logger.info(f"  Sample {i}: True={y_val[i]}, Predicted={predicted_classes[i]}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
