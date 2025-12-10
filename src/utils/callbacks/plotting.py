import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..logging import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    normalize: bool = True
):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)

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
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    df = pd.DataFrame(report).T

    class_metrics = df[~df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]

    class_metrics = class_metrics.sort_values('f1-score', ascending=True).tail(top_n)

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

    for i, (k, v) in enumerate(cumulative_acc.items()):
        plt.annotate(f'{v:.1f}%', (i, v), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Cumulative accuracy plot saved to {output_path}")

    plt.close()

    return cumulative_acc
