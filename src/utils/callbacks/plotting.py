import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from collections import Counter

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


def plot_top_confused_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 10)
):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)

    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = confused_pairs[:top_n]

    involved_classes = sorted(set([p[0] for p in top_pairs] + [p[1] for p in top_pairs]))

    if len(involved_classes) == 0:
        logger.warning("No confused pairs found")
        return

    idx_map = {c: i for i, c in enumerate(involved_classes)}
    sub_cm = np.zeros((len(involved_classes), len(involved_classes)))

    for i, j, count in top_pairs:
        if i in idx_map and j in idx_map:
            sub_cm[idx_map[i], idx_map[j]] = count

    for c in involved_classes:
        sub_cm[idx_map[c], idx_map[c]] = cm[c, c]

    row_sums = sub_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sub_cm_norm = sub_cm / row_sums

    if class_names:
        labels = [str(class_names[c])[:15] for c in involved_classes]
    else:
        labels = [str(c) for c in involved_classes]

    plt.figure(figsize=figsize)
    sns.heatmap(
        sub_cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f'Confusion Matrix - Top {top_n} Most Confused Class Pairs')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Top confused classes matrix saved to {output_path}")

    plt.close()

    return top_pairs


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    show_worst_n: int = 50,
    min_support: int = 10,
    figsize: Tuple[int, int] = (14, 12)
):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    class_stats = []
    for i in range(len(cm)):
        support = cm[i].sum()
        if support >= min_support:
            accuracy = cm[i, i] / support if support > 0 else 0
            class_stats.append({
                'class_id': i,
                'accuracy': accuracy,
                'support': support,
                'correct': cm[i, i],
                'name': class_names[i] if class_names else str(i)
            })

    class_stats.sort(key=lambda x: x['accuracy'])
    worst_classes = class_stats[:show_worst_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    names = [f"{c['name'][:20]}" for c in worst_classes]
    accuracies = [c['accuracy'] * 100 for c in worst_classes]
    supports = [c['support'] for c in worst_classes]

    colors = plt.cm.RdYlGn([a/100 for a in accuracies])
    bars = ax1.barh(range(len(names)), accuracies, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title(f'Worst {show_worst_n} Classes by Accuracy\n(min support={min_support})')
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')

    for i, (acc, sup) in enumerate(zip(accuracies, supports)):
        ax1.annotate(f'n={sup}', xy=(acc + 1, i), va='center', fontsize=7)

    all_accuracies = [c['accuracy'] * 100 for c in class_stats]
    ax2.hist(all_accuracies, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=np.mean(all_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(all_accuracies):.1f}%')
    ax2.axvline(x=np.median(all_accuracies), color='green', linestyle='--', label=f'Median: {np.median(all_accuracies):.1f}%')
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_ylabel('Number of Classes')
    ax2.set_title('Distribution of Per-Class Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Per-class accuracy plot saved to {output_path}")

    plt.close()

    return class_stats


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax1 = axes[0, 0]
    correct_mask = y_true == y_pred
    correct_conf = y_pred_proba[np.arange(len(y_pred)), y_pred][correct_mask]
    incorrect_conf = y_pred_proba[np.arange(len(y_pred)), y_pred][~correct_mask]

    ax1.hist(correct_conf, bins=50, alpha=0.7, label=f'Correct (n={len(correct_conf)})', color='green')
    ax1.hist(incorrect_conf, bins=50, alpha=0.7, label=f'Incorrect (n={len(incorrect_conf)})', color='red')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution: Correct vs Incorrect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    confidences = y_pred_proba[np.arange(len(y_pred)), y_pred]
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    bin_accuracies = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracies.append((y_true[mask] == y_pred[mask]).mean() * 100)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)

    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    bars = ax2.bar(bin_labels, bin_accuracies, color='steelblue', alpha=0.7)
    ax2.plot(range(len(bin_labels)), [i*10 + 5 for i in range(10)], 'r--', label='Perfect calibration')
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Calibration: Accuracy by Confidence')
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, bin_counts):
        ax2.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

    ax3 = axes[1, 0]
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        true_labels_incorrect = y_true[incorrect_indices]
        proba_incorrect = y_pred_proba[incorrect_indices]

        ranks = []
        for i, true_label in enumerate(true_labels_incorrect):
            sorted_indices = np.argsort(proba_incorrect[i])[::-1]
            rank = np.where(sorted_indices == true_label)[0][0] + 1
            ranks.append(min(rank, 11))

        rank_counts = Counter(ranks)
        rank_labels = [str(i) if i <= 10 else '11+' for i in range(1, 12)]
        rank_values = [rank_counts.get(i, 0) for i in range(1, 12)]

        ax3.bar(rank_labels, rank_values, color='coral', alpha=0.7)
        ax3.set_xlabel('True Label Rank in Predictions')
        ax3.set_ylabel('Count')
        ax3.set_title('Where Does True Label Rank? (Incorrect Predictions Only)')
        ax3.grid(True, alpha=0.3, axis='y')

        cumsum = np.cumsum(rank_values) / len(incorrect_indices) * 100
        ax3_twin = ax3.twinx()
        ax3_twin.plot(range(len(rank_labels)), cumsum, 'b-o', markersize=4)
        ax3_twin.set_ylabel('Cumulative %', color='blue')
        ax3_twin.set_ylim(0, 105)

    ax4 = axes[1, 1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    supports = []
    accuracies_per_class = []
    for i in range(len(cm)):
        support = cm[i].sum()
        if support > 0:
            supports.append(support)
            accuracies_per_class.append(cm[i, i] / support * 100)

    ax4.scatter(supports, accuracies_per_class, alpha=0.5, s=20)
    ax4.set_xscale('log')
    ax4.set_xlabel('Class Support (log scale)')
    ax4.set_ylabel('Class Accuracy (%)')
    ax4.set_title('Class Size vs Accuracy')
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    if len(supports) > 2:
        z = np.polyfit(np.log10(supports), accuracies_per_class, 1)
        p = np.poly1d(z)
        x_trend = np.logspace(np.log10(min(supports)), np.log10(max(supports)), 100)
        ax4.plot(x_trend, p(np.log10(x_trend)), 'g--', alpha=0.7, label='Trend')
        ax4.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error analysis plot saved to {output_path}")

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
