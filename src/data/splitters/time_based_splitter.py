import pandas as pd
from typing import Tuple
from .base import DataSplitter


class TimeBasedDataSplitter(DataSplitter):

    def __init__(
        self,
        time_column: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        verbose: bool = True
    ):
        super().__init__(random_state=None, verbose=verbose)
        self.time_column = time_column
        self.val_size = val_size
        self.test_size = test_size

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")

        if self.verbose:
            self.logger.info(f"Time-based splitting with val_size={self.val_size}, test_size={self.test_size}")

        sorted_data = data.sort_values(self.time_column)

        total_size = len(sorted_data)
        test_start = int(total_size * (1 - self.test_size))
        val_start = int(total_size * (1 - self.test_size - self.val_size))

        train_data = sorted_data.iloc[:val_start].copy()
        val_data = sorted_data.iloc[val_start:test_start].copy()
        test_data = sorted_data.iloc[test_start:].copy()

        self._log_split_results(train_data, val_data, test_data)
        return train_data, val_data, test_data
