import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union, Set, Optional
from pathlib import Path
from ..utils import LoggerMixin


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


class TextQualityFilter(DataFilter):

    def __init__(
        self,
        text_column: str,
        min_text_length: int = 1,
        max_text_length: Optional[int] = None,
        min_word_count: int = 1,
        max_word_count: Optional[int] = None,
        remove_duplicates: bool = True,
        remove_empty: bool = True,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.text_column = text_column
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.remove_duplicates = remove_duplicates
        self.remove_empty = remove_empty

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        original_data = data.copy()
        filtered_data = data.copy()

        if self.remove_empty:
            before_count = len(filtered_data)
            empty_mask = (
                filtered_data[self.text_column].isnull() |
                (filtered_data[self.text_column].str.strip() == '')
            )
            filtered_data = filtered_data[~empty_mask]
            removed_empty = before_count - len(filtered_data)
            if self.verbose and removed_empty > 0:
                self.logger.info(f"Removed {removed_empty} empty texts")

        if self.min_text_length > 0 or self.max_text_length:
            before_count = len(filtered_data)
            text_lengths = filtered_data[self.text_column].str.len()

            length_mask = text_lengths >= self.min_text_length
            if self.max_text_length:
                length_mask &= text_lengths <= self.max_text_length

            filtered_data = filtered_data[length_mask]
            removed_length = before_count - len(filtered_data)
            if self.verbose and removed_length > 0:
                self.logger.info(f"Removed {removed_length} texts by length criteria")

        if self.min_word_count > 0 or self.max_word_count:
            before_count = len(filtered_data)
            word_counts = filtered_data[self.text_column].str.split().str.len()

            word_mask = word_counts >= self.min_word_count
            if self.max_word_count:
                word_mask &= word_counts <= self.max_word_count

            filtered_data = filtered_data[word_mask]
            removed_words = before_count - len(filtered_data)
            if self.verbose and removed_words > 0:
                self.logger.info(f"Removed {removed_words} texts by word count criteria")

        if self.remove_duplicates:
            before_count = len(filtered_data)
            filtered_data = filtered_data.drop_duplicates(subset=[self.text_column])
            removed_duplicates = before_count - len(filtered_data)
            if self.verbose and removed_duplicates > 0:
                self.logger.info(f"Removed {removed_duplicates} duplicate texts")

        self._log_filter_results(original_data, filtered_data, "Text quality")
        return filtered_data


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


def create_default_filters(
    class_column: str,
    text_column: str,
    min_samples_per_class: int = 50,
    ostalo_groups_file: Optional[str] = None,
    remove_ostalo: bool = True,
    **kwargs
) -> CompositeFilter:
    filters = []

    text_filter = TextQualityFilter(
        text_column=text_column,
        min_text_length=1,
        min_word_count=1,
        remove_duplicates=True,
        remove_empty=True,
        **kwargs.get('text_filter', {})
    )
    filters.append(text_filter)

    if remove_ostalo and ostalo_groups_file:
        ostalo_filter = OstaloGroupFilter(
            class_column=class_column,
            ostalo_groups_file=ostalo_groups_file,
            **kwargs.get('ostalo_filter', {})
        )
        filters.append(ostalo_filter)

    group_filter = GroupSizeFilter(
        class_column=class_column,
        min_samples=min_samples_per_class,
        **kwargs.get('group_filter', {})
    )
    filters.append(group_filter)

    return CompositeFilter(filters)