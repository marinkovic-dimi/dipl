import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from .base import DataSplitter


class RandomDataSplitter(DataSplitter):

    def __init__(
        self,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True
    ):
        super().__init__(random_state, verbose)
        self.val_size = val_size
        self.test_size = test_size

        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be between 0.0 and 1.0")
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")
        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1.0")

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.verbose:
            self.logger.info(f"Random splitting with val_size={self.val_size}, test_size={self.test_size}")

        temp_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state
        )

        adjusted_val_size = self.val_size / (1 - self.test_size)

        train_data, val_data = train_test_split(
            temp_data,
            test_size=adjusted_val_size,
            random_state=self.random_state
        )

        self._log_split_results(train_data, val_data, test_data)
        return train_data, val_data, test_data
