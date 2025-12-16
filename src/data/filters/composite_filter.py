import pandas as pd
from typing import List
from .base import DataFilter


class CompositeFilter(DataFilter):

    def __init__(self, filters: List[DataFilter], verbose: bool = True):
        super().__init__(verbose)
        self.filters = filters

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        original_data = data.copy()
        current_data = data.copy()

        if self.verbose:
            self.logger.info(f"Applying {len(self.filters)} filters in sequence")

        for i, filter_instance in enumerate(self.filters):
            if self.verbose:
                self.logger.info(f"Applying filter {i+1}/{len(self.filters)}: {type(filter_instance).__name__}")

            current_data = filter_instance.filter(current_data)

        if self.verbose:
            original_size = len(original_data)
            final_size = len(current_data)
            total_removed = original_size - final_size
            removal_rate = total_removed / original_size if original_size > 0 else 0

            self.logger.info("Composite filtering complete:")
            self.logger.info(f"  Total removed: {total_removed} samples ({removal_rate:.1%})")
            self.logger.info(f"  Final size: {final_size} samples")

        return current_data
