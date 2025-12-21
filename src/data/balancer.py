import pandas as pd
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from src.utils import LoggerMixin, BalancingConfig


@dataclass
class BalancingStats:
    original_samples: int
    balanced_samples: int
    original_distribution: Dict[int, int]
    balanced_distribution: Dict[int, int]
    oversampled_classes: int
    undersampled_classes: int
    unchanged_classes: int


class DataBalancer(LoggerMixin):

    def __init__(
        self,
        config: BalancingConfig,
        class_column: str = "group_id",
        random_state: int = 42
    ):
        self.config = config
        self.class_column = class_column
        self.random_state = random_state
        self.stats: Optional[BalancingStats] = None

    def _get_tier(self, count: int) -> str:
        limits = self.config.tier_limits
        if count <= limits.get("small", 50):
            return "small"
        elif count <= limits.get("medium", 1000):
            return "medium"
        elif count <= limits.get("large", 10000):
            return "large"
        elif count <= limits.get("xlarge", 20000):
            return "xlarge"
        elif count <= limits.get("xxlarge", 50000):
            return "xxlarge"
        else:
            return "max"

    def _calculate_target_size(self, current_size: int, tier: str) -> int:
        target_threshold = self.config.target_threshold
        target_threshold_small = self.config.target_threshold_small

        if current_size < target_threshold_small:
            increase = int(current_size * self.config.increase_factor)
            target = min(current_size + increase, target_threshold_small)
            return max(target, current_size)

        elif current_size > target_threshold:
            if tier in ["xxlarge", "max"]:
                factor = self.config.decrease_factor
            else:
                factor = self.config.decrease_factor_small

            target = int(current_size * (1 - factor))
            return max(target, target_threshold)

        return current_size

    def balance(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.strategy != "adaptive":
            self.logger.warning(f"Strategy '{self.config.strategy}' not implemented, returning original data")
            return data

        self.logger.info("Starting adaptive data balancing...")

        original_distribution = data[self.class_column].value_counts().to_dict()
        original_samples = len(data)

        np.random.seed(self.random_state)

        balanced_dfs = []
        oversampled = 0
        undersampled = 0
        unchanged = 0

        class_counts = data[self.class_column].value_counts()

        for class_id, current_size in class_counts.items():
            class_data = data[data[self.class_column] == class_id]
            tier = self._get_tier(current_size)
            target_size = self._calculate_target_size(current_size, tier)

            if target_size > current_size:
                n_oversample = target_size - current_size
                oversampled_indices = np.random.choice(
                    class_data.index,
                    size=n_oversample,
                    replace=True
                )
                oversampled_data = data.loc[oversampled_indices]
                balanced_dfs.append(class_data)
                balanced_dfs.append(oversampled_data)
                oversampled += 1

            elif target_size < current_size:
                sampled_indices = np.random.choice(
                    class_data.index,
                    size=target_size,
                    replace=False
                )
                balanced_dfs.append(data.loc[sampled_indices])
                undersampled += 1

            else:
                balanced_dfs.append(class_data)
                unchanged += 1

        balanced_data = pd.concat(balanced_dfs, ignore_index=True)
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        balanced_distribution = balanced_data[self.class_column].value_counts().to_dict()

        self.stats = BalancingStats(
            original_samples=original_samples,
            balanced_samples=len(balanced_data),
            original_distribution=original_distribution,
            balanced_distribution=balanced_distribution,
            oversampled_classes=oversampled,
            undersampled_classes=undersampled,
            unchanged_classes=unchanged
        )

        self._log_stats()

        return balanced_data

    def _log_stats(self):
        if not self.stats:
            return

        self.logger.info(f"Balancing complete:")
        self.logger.info(f"  Original samples: {self.stats.original_samples:,}")
        self.logger.info(f"  Balanced samples: {self.stats.balanced_samples:,}")
        diff = self.stats.balanced_samples - self.stats.original_samples
        diff_pct = diff / self.stats.original_samples * 100
        self.logger.info(f"  Difference: {diff:+,} ({diff_pct:+.1f}%)")
        self.logger.info(f"  Oversampled classes: {self.stats.oversampled_classes}")
        self.logger.info(f"  Undersampled classes: {self.stats.undersampled_classes}")
        self.logger.info(f"  Unchanged classes: {self.stats.unchanged_classes}")

    def get_distribution_comparison(self, top_n: int = 20) -> pd.DataFrame:
        if not self.stats:
            return pd.DataFrame()

        orig = self.stats.original_distribution
        bal = self.stats.balanced_distribution

        all_classes = set(orig.keys()) | set(bal.keys())

        rows = []
        for class_id in all_classes:
            orig_count = orig.get(class_id, 0)
            bal_count = bal.get(class_id, 0)
            diff = bal_count - orig_count
            diff_pct = (diff / orig_count * 100) if orig_count > 0 else 0
            rows.append({
                'class_id': class_id,
                'original': orig_count,
                'balanced': bal_count,
                'difference': diff,
                'diff_pct': diff_pct
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('original', ascending=False)

        return df.head(top_n)

    def plot_balancing_stats(self, output_path: Optional[str] = None, top_n: int = 30):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        if not self.stats:
            self.logger.warning("No balancing stats available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        categories = ['Uvećane', 'Smanjene', 'Nepromenjene']
        values = [
            self.stats.oversampled_classes,
            self.stats.undersampled_classes,
            self.stats.unchanged_classes
        ]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Broj klasa')
        ax1.set_title('Klase po akciji balansiranja')
        for bar, val in zip(bars, values):
            ax1.annotate(str(val), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2 = axes[0, 1]
        labels = ['Pre', 'Posle']
        sizes = [self.stats.original_samples, self.stats.balanced_samples]
        diff = self.stats.balanced_samples - self.stats.original_samples
        diff_pct = diff / self.stats.original_samples * 100
        bars = ax2.bar(labels, sizes, color=['#9b59b6', '#1abc9c'], alpha=0.8)
        ax2.set_ylabel('Broj uzoraka')
        ax2.set_title(f'Ukupno uzoraka (Promena: {diff:+,} / {diff_pct:+.1f}%)')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        for bar, val in zip(bars, sizes):
            ax2.annotate(f'{val:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax3 = axes[1, 0]
        comparison = self.get_distribution_comparison(top_n=top_n)
        x = np.arange(len(comparison))
        width = 0.35
        ax3.bar(x - width/2, comparison['original'], width, label='Pre', alpha=0.8)
        ax3.bar(x + width/2, comparison['balanced'], width, label='Posle', alpha=0.8)
        ax3.set_xlabel('Klasa (sortirano po originalnoj veličini)')
        ax3.set_ylabel('Broj uzoraka')
        ax3.set_title(f'Top {top_n} klasa: pre i posle balansiranja')
        ax3.set_xticks(x[::5])
        ax3.set_xticklabels([str(i) for i in range(0, len(comparison), 5)])
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = axes[1, 1]
        orig_values = np.array(list(self.stats.original_distribution.values()))
        bal_values = np.array(list(self.stats.balanced_distribution.values()))
        ax4.hist(orig_values, bins=50, alpha=0.6, label='Pre', color='#9b59b6')
        ax4.hist(bal_values, bins=50, alpha=0.6, label='Posle', color='#1abc9c')
        ax4.set_xlabel('Veličina klase')
        ax4.set_ylabel('Broj klasa')
        ax4.set_title('Distribucija veličina klasa')
        ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Balancing plot saved to {output_path}")

        plt.close()

    def save_stats_json(self, output_path: str):
        import json

        if not self.stats:
            return

        stats_dict = {
            'original_samples': self.stats.original_samples,
            'balanced_samples': self.stats.balanced_samples,
            'difference': self.stats.balanced_samples - self.stats.original_samples,
            'difference_pct': (self.stats.balanced_samples - self.stats.original_samples) / self.stats.original_samples * 100,
            'oversampled_classes': self.stats.oversampled_classes,
            'undersampled_classes': self.stats.undersampled_classes,
            'unchanged_classes': self.stats.unchanged_classes,
            'total_classes': len(self.stats.original_distribution),
            'config': {
                'strategy': self.config.strategy,
                'target_threshold': self.config.target_threshold,
                'target_threshold_small': self.config.target_threshold_small,
                'increase_factor': self.config.increase_factor,
                'decrease_factor': self.config.decrease_factor,
                'decrease_factor_small': self.config.decrease_factor_small
            }
        }

        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        self.logger.info(f"Balancing stats saved to {output_path}")
