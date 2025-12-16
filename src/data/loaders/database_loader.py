import json
import pandas as pd
from pathlib import Path
from typing import Union, Optional

from ...utils.config.data_config import DataConfig
from .base import DataLoader


class DatabaseDataLoader(DataLoader):

    def __init__(self,data_config: DataConfig, data_key: str = "data", metadata_key: Optional[str] = None, verbose: bool = True):
        super().__init__(verbose)
        self.data_config = data_config
        self.data_key = data_key
        self.metadata_key = metadata_key

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"JSON file not found: {source_path}")

        try:
            with open(source_path, 'rb') as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list) and len(raw_data) > 0:
                for item in raw_data:
                    if isinstance(item, dict) and self.data_key in item:
                        json_data = item[self.data_key]
                        break
                else:
                    raise ValueError(f"Data key '{self.data_key}' not found in any list item")
            elif isinstance(raw_data, dict):
                if self.data_key in raw_data:
                    json_data = raw_data[self.data_key]
                else:
                    raise ValueError(f"Data key '{self.data_key}' not found in JSON")
            else:
                raise ValueError("Unsupported JSON structure")

            data = pd.json_normalize(json_data)

            if data.empty:
                self.logger.warning(f"Loaded empty DataFrame from {source_path}")

            self._log_data_info(data, str(source_path))
            return data

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid JSON format in {source_path}: {e}")
