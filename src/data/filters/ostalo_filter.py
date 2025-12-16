import pandas as pd
from pathlib import Path
from typing import Union, Set
from .base import DataFilter


class OstaloGroupFilter(DataFilter):

    def __init__(
        self,
        class_column: str,
        ostalo_groups_file: Union[str, Path],
        separator: str = ',',
        group_id_column: str = 'category_id',
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.class_column = class_column
        self.ostalo_groups_file = Path(ostalo_groups_file)
        self.separator = separator
        self.group_id_column = group_id_column

    def _load_ostalo_groups(self) -> Set:
        if not self.ostalo_groups_file.exists():
            self.logger.warning(f"Ostalo groups file not found: {self.ostalo_groups_file}")
            return set()

        try:
            ostalo_df = pd.read_csv(
                self.ostalo_groups_file,
                sep=self.separator,
                quotechar='"'
            )

            if self.group_id_column not in ostalo_df.columns:
                self.logger.error(f"Column '{self.group_id_column}' not found in ostalo file")
                return set()

            ostalo_ids = set(ostalo_df[self.group_id_column].unique())

            if self.verbose:
                self.logger.info(f"Loaded {len(ostalo_ids)} ostalo group IDs")

            return ostalo_ids

        except Exception as e:
            self.logger.error(f"Error loading ostalo groups file: {e}")
            return set()

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        original_data = data.copy()

        ostalo_ids = self._load_ostalo_groups()

        if not ostalo_ids:
            self.logger.warning("No ostalo groups to filter")
            return data

        data_copy = data.copy()
        data_copy[self.class_column] = data_copy[self.class_column].astype(
            type(next(iter(ostalo_ids)))
        )

        filtered_data = data_copy[~data_copy[self.class_column].isin(ostalo_ids)].copy()

        if self.verbose:
            original_groups = len(original_data[self.class_column].unique())
            filtered_groups = len(filtered_data[self.class_column].unique())
            removed_groups = original_groups - filtered_groups
            self.logger.info(f"Removed {removed_groups} ostalo groups")

        self._log_filter_results(original_data, filtered_data, "Ostalo groups")
        return filtered_data
