import pandas as pd
from typing import List, Optional
from .base import DataFilter


class GroupSizeFilter(DataFilter):

    def __init__(
        self,
        class_column: str,
        min_samples: int = 50,
        max_samples: Optional[int] = None,
        exception_groups: Optional[List] = None,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.class_column = class_column
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.exception_groups = set(exception_groups or [])

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        original_data = data.copy()

        group_counts = data[self.class_column].value_counts()

        if self.verbose:
            self.logger.info(f"Group size filtering: min={self.min_samples}, max={self.max_samples}")
            self.logger.info(f"Original groups: {len(group_counts)}")

        valid_groups = set()

        for group_id, count in group_counts.items():
            if group_id in self.exception_groups:
                valid_groups.add(group_id)
                if self.verbose:
                    self.logger.debug(f"Group {group_id}: {count} samples (exception, kept)")
                continue

            if count < self.min_samples:
                if self.verbose:
                    self.logger.debug(f"Group {group_id}: {count} samples (too small, removed)")
                continue

            if self.max_samples and count > self.max_samples:
                if self.verbose:
                    self.logger.debug(f"Group {group_id}: {count} samples (too large, removed)")
                continue

            valid_groups.add(group_id)

        filtered_data = data[data[self.class_column].isin(valid_groups)].copy()

        if self.verbose:
            remaining_groups = len(filtered_data[self.class_column].unique())
            removed_groups = len(group_counts) - remaining_groups
            self.logger.info(f"Removed {removed_groups} groups, kept {remaining_groups} groups")

        self._log_filter_results(original_data, filtered_data, "Group size")
        return filtered_data
