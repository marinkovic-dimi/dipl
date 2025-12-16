import pandas as pd
from abc import ABC, abstractmethod
from ...utils import LoggerMixin


class TextPreprocessor(ABC, LoggerMixin):

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        pass

    def preprocess_dataframe(
        self,
        data: pd.DataFrame,
        text_column: str,
        output_column: str,
        inplace: bool = False
    ) -> pd.DataFrame:
        if not inplace:
            data = data.copy()

        if self.verbose:
            self.logger.info(f"Preprocessing text column '{text_column}' -> '{output_column}'")

        data[output_column] = data[text_column].apply(self.preprocess_text)

        empty_mask = data[output_column].str.strip() == ''
        if empty_mask.any():
            empty_count = empty_mask.sum()
            data = data[~empty_mask]
            if self.verbose:
                self.logger.warning(f"Removed {empty_count} rows with empty preprocessed text")

        if self.verbose:
            self.logger.info(f"Preprocessing complete. Final dataset size: {len(data)} rows")

        return data
