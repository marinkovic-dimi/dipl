import pandas as pd
from typing import Optional
from .base import DataFilter


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
