import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from ...utils import LoggerMixin


class DataSplitter(ABC, LoggerMixin):

    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    def _log_split_results(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        class_column: Optional[str] = None
    ) -> None:
        if self.verbose:
            total_size = len(train) + len(val) + len(test)
            train_pct = len(train) / total_size * 100
            val_pct = len(val) / total_size * 100
            test_pct = len(test) / total_size * 100

            self.logger.info("Data split results:")
            self.logger.info(f"  Train: {len(train)} samples ({train_pct:.1f}%)")
            self.logger.info(f"  Validation: {len(val)} samples ({val_pct:.1f}%)")
            self.logger.info(f"  Test: {len(test)} samples ({test_pct:.1f}%)")

            if class_column and class_column in train.columns:
                train_classes = len(train[class_column].unique())
                val_classes = len(val[class_column].unique())
                test_classes = len(test[class_column].unique())

                self.logger.info("Class distribution:")
                self.logger.info(f"  Train classes: {train_classes}")
                self.logger.info(f"  Validation classes: {val_classes}")
                self.logger.info(f"  Test classes: {test_classes}")
