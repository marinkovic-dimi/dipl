import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from ...utils import LoggerMixin


class DataLoader(ABC, LoggerMixin):

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        pass

    def _log_data_info(self, data: pd.DataFrame, source: str) -> None:
        if self.verbose:
            self.logger.info(f"Loaded data from {source}")
            self.logger.info(f"Shape: {data.shape}")
            self.logger.info(f"Columns: {list(data.columns)}")

            missing = data.isnull().sum()
            if missing.any():
                self.logger.warning("Missing values found:")
                for col, count in missing[missing > 0].items():
                    self.logger.warning(f"  {col}: {count} missing values")
