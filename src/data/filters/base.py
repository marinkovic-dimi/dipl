import pandas as pd
from abc import ABC, abstractmethod
from ...utils import LoggerMixin


class DataFilter(ABC, LoggerMixin):

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def _log_filter_results(
        self,
        original_data: pd.DataFrame,
        filtered_data: pd.DataFrame,
        filter_name: str
    ) -> None:
        if self.verbose:
            original_size = len(original_data)
            filtered_size = len(filtered_data)
            removed_count = original_size - filtered_size
            removal_rate = removed_count / original_size if original_size > 0 else 0

            self.logger.info(f"{filter_name} filter:")
            self.logger.info(f"  Removed: {removed_count} samples ({removal_rate:.1%})")
            self.logger.info(f"  Remaining: {filtered_size} samples")
