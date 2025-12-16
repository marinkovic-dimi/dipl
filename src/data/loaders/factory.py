from pathlib import Path
from typing import Union

from ...utils.config.data_config import DataConfig
from .base import DataLoader
from .json_loader import JSONDataLoader
from .database_loader import DatabaseDataLoader
from .csv_loader import CSVDataLoader


def create_data_loader(source: Union[str, Path], data_config: DataConfig, loader_type: str = "auto", **kwargs) -> DataLoader:
    source_path = Path(source)

    if loader_type == "auto":
        if source_path.suffix.lower() == '.json':
            try:
                with open(source_path, 'r') as f:
                    first_char = f.read(1)
                    if first_char == '[':
                        return DatabaseDataLoader(data_config, **kwargs)
                    else:
                        return JSONDataLoader(data_config, **kwargs)
            except:
                return JSONDataLoader(data_config, **kwargs)
        elif source_path.suffix.lower() == '.csv':
            return CSVDataLoader(data_config, **kwargs)
        else:
            raise ValueError(f"Cannot auto-detect loader for file: {source_path}")

    elif loader_type == "json":
        return JSONDataLoader(data_config, **kwargs)
    elif loader_type == "database":
        return DatabaseDataLoader(data_config, **kwargs)
    elif loader_type == "csv":
        return CSVDataLoader(data_config, **kwargs)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")
