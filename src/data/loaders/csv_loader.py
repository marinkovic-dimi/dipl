import pandas as pd
from pathlib import Path
from typing import Union
from ...utils.config.data_config import DataConfig
from .base import DataLoader


class CSVDataLoader(DataLoader):

    def __init__(self, data_config: DataConfig, verbose: bool = True, **kwargs):
        super().__init__(verbose)
        self.data_config = data_config
        self.csv_kwargs = kwargs

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")

        try:
            default_kwargs = {'encoding': 'utf-8'}
            default_kwargs.update(self.csv_kwargs)

            data = pd.read_csv(source_path, **default_kwargs)

            if data.empty:
                self.logger.warning(f"Loaded empty DataFrame from {source_path}")

            self._log_data_info(data, str(source_path))
            return data

        except Exception as e:
            raise ValueError(f"Error reading CSV file {source_path}: {e}")
