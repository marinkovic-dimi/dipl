import json
import pandas as pd
from pathlib import Path
from typing import Union
from ...utils.config.data_config import DataConfig
from .base import DataLoader


class JSONDataLoader(DataLoader):

    def __init__(self, data_config: DataConfig, verbose: bool = True, **kwargs):
        super().__init__(verbose)
        self.data_config = data_config
        self.json_kwargs = kwargs

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"JSON file not found: {source_path}")

        try:
            nrows = getattr(self.data_config, 'max_samples', None)
            data = pd.read_json(source_path)
            if isinstance(data, pd.DataFrame) and nrows is not None:
                data = data.iloc[:nrows]

            if data.empty:
                self.logger.warning(f"Loaded empty DataFrame from {source_path}")

            self._log_data_info(data, str(source_path))
            return data

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid JSON format in {source_path}: {e}")
