"""Batch progress callback for logging batch-level metrics."""

import keras
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from ..logging import get_logger

logger = get_logger(__name__)


class BatchProgressCallback(keras.callbacks.Callback):
    """Callback to log and visualize batch-level metrics."""

    def __init__(
        self,
        log_interval: int = 100,
        validation_data: Optional[Tuple] = None,
        output_dir: Optional[str] = None,
        wandb_enabled: bool = False
    ):
        """
        Initialize batch progress callback.

        Args:
            log_interval: Log every N batches
            validation_data: Tuple of (X_val, y_val) for validation metrics
            output_dir: Directory to save batch logs
            wandb_enabled: Whether to log to W&B
        """
        super().__init__()
        self.log_interval = log_interval
        self.validation_data = validation_data
        self.output_dir = Path(output_dir) if output_dir else None
        self.wandb_enabled = wandb_enabled

        self.batch_count = 0
        self.epoch_count = 0
        self.batch_history = []

        self.wandb = None
        if wandb_enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed for batch logging")

    def on_epoch_begin(self, epoch, logs=None):
        """Track epoch count."""
        self.epoch_count = epoch + 1

    def on_batch_end(self, batch, logs=None):
        """Log metrics at specified intervals."""
        self.batch_count += 1

        if self.batch_count % self.log_interval != 0:
            return

        logs = logs or {}

        # Get current metrics
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('sparse_categorical_accuracy', 0)
        train_top_k = logs.get('top_k_acc', 0)

        batch_data = {
            'global_batch': self.batch_count,
            'epoch': self.epoch_count,
            'batch': batch + 1,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'train_top_k_acc': float(train_top_k)
        }

        # Evaluate on validation data if provided
        if self.validation_data is not None:
            val_results = self.model.evaluate(
                self.validation_data[0],
                self.validation_data[1],
                verbose=0
            )
            batch_data['val_loss'] = float(val_results[0])
            batch_data['val_accuracy'] = float(val_results[1])
            if len(val_results) > 2:
                batch_data['val_top_k_acc'] = float(val_results[2])

        self.batch_history.append(batch_data)

        # Log to W&B
        if self.wandb and self.wandb.run:
            wandb_metrics = {f'batch/{k}': v for k, v in batch_data.items()}
            self.wandb.log(wandb_metrics)

        # Save to CSV
        if self.output_dir:
            df = pd.DataFrame(self.batch_history)
            df.to_csv(self.output_dir / 'batch_log.csv', index=False)
