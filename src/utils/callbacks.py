"""Training callbacks for the AI classifier project."""

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from .logging import get_logger

logger = get_logger(__name__)


class TrainingPlotCallback(keras.callbacks.Callback):
    """Callback to plot training progress after each epoch."""

    def __init__(
        self,
        output_dir: str,
        plot_name: str = "training_progress.png",
        csv_name: str = "training_log.csv"
    ):
        """
        Initialize training plot callback.

        Args:
            output_dir: Directory to save plots and logs
            plot_name: Filename for the plot
            csv_name: Filename for the CSV log
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_path = self.output_dir / plot_name
        self.csv_path = self.output_dir / csv_name
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        """Save metrics and update plot after each epoch."""
        logs = logs or {}

        # Get learning rate
        try:
            lr = float(self.model.optimizer.learning_rate)
        except TypeError:
            # If learning rate is a schedule, get current value
            lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))

        # Store metrics
        epoch_data = {
            'epoch': epoch + 1,
            'learning_rate': lr,
            **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.floating))}
        }
        self.history.append(epoch_data)

        # Save to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)

        # Update plot
        self._plot_training_progress(df)

    def _plot_training_progress(self, df: pd.DataFrame):
        """Create training progress plots."""
        epochs = df['epoch']

        # Determine which metrics are available
        has_top_k = 'top_k_acc' in df.columns or 'val_top_k_acc' in df.columns

        if has_top_k:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = axes.reshape(1, -1)

        # Plot 1: Accuracy
        ax1 = axes[0, 0] if has_top_k else axes[0, 0]
        if 'sparse_categorical_accuracy' in df.columns:
            ax1.plot(epochs, df['sparse_categorical_accuracy'], 'b-', label='Train Accuracy')
        if 'val_sparse_categorical_accuracy' in df.columns:
            ax1.plot(epochs, df['val_sparse_categorical_accuracy'], 'r-', label='Val Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss
        ax2 = axes[0, 1] if has_top_k else axes[0, 1]
        if 'loss' in df.columns:
            ax2.plot(epochs, df['loss'], 'b-', label='Train Loss')
        if 'val_loss' in df.columns:
            ax2.plot(epochs, df['val_loss'], 'r-', label='Val Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if has_top_k:
            # Plot 3: Top-K Accuracy
            ax3 = axes[1, 0]
            if 'top_k_acc' in df.columns:
                ax3.plot(epochs, df['top_k_acc'], 'b-', label='Train Top-3 Acc')
            if 'val_top_k_acc' in df.columns:
                ax3.plot(epochs, df['val_top_k_acc'], 'r-', label='Val Top-3 Acc')
            ax3.set_title('Training and Validation Top-3 Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Top-3 Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Learning Rate
            ax4 = axes[1, 1]
            if 'learning_rate' in df.columns:
                ax4.plot(epochs, df['learning_rate'], 'g-', label='Learning Rate')
                ax4.set_title('Learning Rate Schedule')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Learning Rate')
                ax4.set_yscale('log')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()


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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    normalize: bool = True
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the matrix
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)

    # If too many classes, don't show annotations
    annot = len(cm) <= 30

    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names and len(class_names) <= 30 else False,
        yticklabels=class_names if class_names and len(class_names) <= 30 else False
    )

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")

    plt.close()


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    top_n: int = 20
):
    """
    Plot classification report metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
        top_n: Number of classes to show (sorted by F1 score)
    """
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Convert to DataFrame
    df = pd.DataFrame(report).T

    # Filter out summary rows
    class_metrics = df[~df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]

    # Sort by F1 score and take top N
    class_metrics = class_metrics.sort_values('f1-score', ascending=True).tail(top_n)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, top_n * 0.3)))

    metrics = ['precision', 'recall', 'f1-score']
    colors = ['steelblue', 'darkorange', 'forestgreen']

    for ax, metric, color in zip(axes, metrics, colors):
        ax.barh(class_metrics.index, class_metrics[metric], color=color, alpha=0.7)
        ax.set_xlabel(metric.capitalize())
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Classification Metrics (Top {top_n} classes by F1)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Classification report plot saved to {output_path}")

    plt.close()

    return report


def plot_cumulative_accuracy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    max_k: int = 5
):
    """
    Plot cumulative top-k accuracy.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (num_samples, num_classes)
        output_path: Path to save the plot
        max_k: Maximum k for top-k accuracy
    """
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :max_k]

    cumulative_acc = {}
    for k in range(1, max_k + 1):
        correct = np.any(top_k_preds[:, :k] == y_true[:, np.newaxis], axis=1)
        cumulative_acc[f'Top-{k}'] = np.mean(correct) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(list(cumulative_acc.keys()), list(cumulative_acc.values()), 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Top N Predictions')
    plt.ylabel('Cumulative Accuracy (%)')
    plt.title('Cumulative Top-K Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)

    # Add value labels
    for i, (k, v) in enumerate(cumulative_acc.items()):
        plt.annotate(f'{v:.1f}%', (i, v), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Cumulative accuracy plot saved to {output_path}")

    plt.close()

    return cumulative_acc


class WandbCallback(keras.callbacks.Callback):
    """Weights & Biases logging callback for Keras training."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        project: str = "ad-classifier",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        log_model: bool = True,
        log_gradients: bool = False,
        log_frequency: int = 100,
        save_code: bool = True
    ):
        """
        Initialize Weights & Biases callback.

        Args:
            config: Configuration dictionary to log
            project: W&B project name
            entity: W&B entity (username or team)
            name: Run name
            tags: List of tags
            notes: Run notes
            log_model: Whether to log model checkpoints
            log_gradients: Whether to log gradients
            log_frequency: Frequency of logging (batches)
            save_code: Whether to save code
        """
        super().__init__()
        self.config = config or {}
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.notes = notes
        self.log_model = log_model
        self.log_gradients = log_gradients
        self.log_frequency = log_frequency
        self.save_code = save_code

        self.wandb = None
        self.batch_count = 0

        try:
            import wandb
            self.wandb = wandb
            logger.info("Weights & Biases imported successfully")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.wandb = None

    def on_train_begin(self, logs=None):
        """Initialize W&B run at the start of training."""
        if self.wandb is None:
            logger.warning("W&B callback disabled - wandb not installed")
            return

        try:
            self.wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                save_code=self.save_code,
                reinit=True
            )

            if self.model and hasattr(self.model, 'summary'):
                logger.info("Logging model summary to W&B")
                import io
                summary_io = io.StringIO()
                self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
                model_summary = summary_io.getvalue()

                self.wandb.config.update({
                    'model_summary': model_summary,
                    'total_params': self.model.count_params()
                })

            logger.info(f"W&B run initialized: {self.wandb.run.name}")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.wandb = None

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if self.wandb is None or not self.wandb.run:
            return

        logs = logs or {}

        # Log epoch metrics with 'epoch/' prefix
        metrics = {'epoch': epoch + 1}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics[f'epoch/{key}'] = value

        try:
            self.wandb.log(metrics)
            logger.debug(f"Logged epoch {epoch + 1} metrics to W&B")
        except Exception as e:
            logger.error(f"Failed to log epoch metrics: {e}")

    def on_batch_end(self, batch, logs=None):
        """Log batch-level metrics."""
        if self.wandb is None or not self.wandb.run:
            return

        self.batch_count += 1

        if self.batch_count % self.log_frequency == 0:
            logs = logs or {}

            # Log batch metrics with 'batch/' prefix
            batch_metrics = {'batch': self.batch_count}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    batch_metrics[f'batch/{key}'] = value

            try:
                self.wandb.log(batch_metrics)
            except Exception as e:
                logger.error(f"Failed to log batch metrics: {e}")

    def on_train_end(self, logs=None):
        """Finish W&B run at the end of training."""
        if self.wandb is None or not self.wandb.run:
            return

        try:
            if self.log_model and self.model:
                model_path = Path("wandb_model_checkpoint.keras")
                self.model.save(model_path)

                self.wandb.save(str(model_path))
                logger.info(f"Saved model to W&B: {model_path}")

                if model_path.exists():
                    model_path.unlink()

            self.wandb.finish()
            logger.info("W&B run finished")

        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}")


def create_wandb_callback(
    config: Optional[Dict[str, Any]] = None,
    wandb_config: Optional[Any] = None
) -> Optional[WandbCallback]:
    """
    Factory function to create W&B callback from configuration.

    Args:
        config: Full configuration dictionary
        wandb_config: WandbConfig object or dict

    Returns:
        WandbCallback if enabled, None otherwise
    """
    if wandb_config is None:
        return None

    if hasattr(wandb_config, '__dict__'):
        wandb_dict = wandb_config.__dict__
    else:
        wandb_dict = wandb_config

    if not wandb_dict.get('enabled', False):
        logger.info("W&B logging disabled in configuration")
        return None

    return WandbCallback(
        config=config,
        project=wandb_dict.get('project', 'ad-classifier'),
        entity=wandb_dict.get('entity'),
        name=wandb_dict.get('name'),
        tags=wandb_dict.get('tags', []),
        notes=wandb_dict.get('notes'),
        log_model=wandb_dict.get('log_model', True),
        log_gradients=wandb_dict.get('log_gradients', False),
        log_frequency=wandb_dict.get('log_frequency', 100),
        save_code=wandb_dict.get('save_code', True)
    )
