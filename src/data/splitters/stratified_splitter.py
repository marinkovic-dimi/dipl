import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from .base import DataSplitter


class StratifiedDataSplitter(DataSplitter):

    def __init__(
        self,
        class_column: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True
    ):
        super().__init__(random_state, verbose)
        self.class_column = class_column
        self.val_size = val_size
        self.test_size = test_size

        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be between 0.0 and 1.0")
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")
        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1.0")

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.class_column not in data.columns:
            raise ValueError(f"Class column '{self.class_column}' not found in data")

        if self.verbose:
            self.logger.info(f"Stratified splitting with val_size={self.val_size}, test_size={self.test_size}")

        temp_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            stratify=data[self.class_column],
            random_state=self.random_state
        )

        adjusted_val_size = self.val_size / (1 - self.test_size)

        train_data, val_data = train_test_split(
            temp_data,
            test_size=adjusted_val_size,
            stratify=temp_data[self.class_column],
            random_state=self.random_state
        )

        self._log_split_results(train_data, val_data, test_data, self.class_column)
        return train_data, val_data, test_data

    def validate_stratification(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Dict[str, Any]:
        train_dist = train[self.class_column].value_counts(normalize=True).sort_index()
        val_dist = val[self.class_column].value_counts(normalize=True).sort_index()
        test_dist = test[self.class_column].value_counts(normalize=True).sort_index()

        common_classes = set(train_dist.index) & set(val_dist.index) & set(test_dist.index)

        if not common_classes:
            return {
                'stratification_valid': False,
                'error': 'No common classes across all splits'
            }

        max_deviation = 0.0
        class_deviations = {}

        for class_id in common_classes:
            train_prop = train_dist.get(class_id, 0)
            val_prop = val_dist.get(class_id, 0)
            test_prop = test_dist.get(class_id, 0)

            deviation = max(
                abs(train_prop - val_prop),
                abs(train_prop - test_prop),
                abs(val_prop - test_prop)
            )

            class_deviations[class_id] = deviation
            max_deviation = max(max_deviation, deviation)

        stratification_valid = max_deviation < 0.05

        return {
            'stratification_valid': stratification_valid,
            'max_deviation': max_deviation,
            'class_deviations': class_deviations,
            'common_classes': len(common_classes),
            'total_classes': {
                'train': len(train_dist),
                'val': len(val_dist),
                'test': len(test_dist)
            }
        }
