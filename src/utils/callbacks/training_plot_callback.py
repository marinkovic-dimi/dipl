import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingPlotCallback(keras.callbacks.Callback):

    def __init__(
        self,
        output_dir: str,
        plot_name: str = "training_progress.png",
        csv_name: str = "training_log.csv"
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_path = self.output_dir / plot_name
        self.csv_path = self.output_dir / csv_name
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        try:
            lr = float(self.model.optimizer.learning_rate)
        except TypeError:
            lr = float(self.model.optimizer.learning_rate(self.model.optimizer.iterations))

        epoch_data = {
            'epoch': epoch + 1,
            'learning_rate': lr,
            **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.floating))}
        }
        self.history.append(epoch_data)

        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)

        self._plot_training_progress(df)

    def _plot_training_progress(self, df: pd.DataFrame):
        epochs = df['epoch']

        has_top_k = 'top_k_acc' in df.columns or 'val_top_k_acc' in df.columns

        if has_top_k:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = axes.reshape(1, -1)

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
