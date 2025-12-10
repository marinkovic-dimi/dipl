import json
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
from ..utils import LoggerMixin


class DataLoader(ABC, LoggerMixin):
    """Abstract base class for data loaders."""

    def __init__(self, verbose: bool = True):
        """
        Initialize data loader.

        Args:
            verbose: Whether to log information about loaded data
        """
        self.verbose = verbose

    @abstractmethod
    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from source.

        Args:
            source: Data source (file path, URL, etc.)

        Returns:
            Loaded DataFrame
        """
        pass

    def _log_data_info(self, data: pd.DataFrame, source: str) -> None:
        """Log information about loaded data."""
        if self.verbose:
            self.logger.info(f"Loaded data from {source}")
            self.logger.info(f"Shape: {data.shape}")
            self.logger.info(f"Columns: {list(data.columns)}")

            missing = data.isnull().sum()
            if missing.any():
                self.logger.warning("Missing values found:")
                for col, count in missing[missing > 0].items():
                    self.logger.warning(f"  {col}: {count} missing values")


class JSONDataLoader(DataLoader):
    """Data loader for JSON files."""

    def __init__(self, verbose: bool = True, **kwargs):
        """
        Initialize JSON data loader.

        Args:
            verbose: Whether to log information about loaded data
            **kwargs: Additional arguments for pandas.read_json
        """
        super().__init__(verbose)
        self.json_kwargs = kwargs

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from JSON file.

        Args:
            source: Path to JSON file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If JSON format is invalid
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"JSON file not found: {source_path}")

        try:
            default_kwargs = {'orient': 'records', 'lines': True}
            default_kwargs.update(self.json_kwargs)

            data = pd.read_json(source_path, **default_kwargs)

            if data.empty:
                self.logger.warning(f"Loaded empty DataFrame from {source_path}")

            self._log_data_info(data, str(source_path))
            return data

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid JSON format in {source_path}: {e}")


class DatabaseDataLoader(DataLoader):
    """Data loader for database-style JSON files (with nested structure)."""

    def __init__(self, data_key: str = "data", metadata_key: Optional[str] = None, verbose: bool = True):
        """
        Initialize database data loader.

        Args:
            data_key: Key in JSON structure containing the data
            metadata_key: Optional key for metadata
            verbose: Whether to log information about loaded data
        """
        super().__init__(verbose)
        self.data_key = data_key
        self.metadata_key = metadata_key

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from database-style JSON file.

        Expected format:
        {
            "metadata": {...},
            "data": [{"id": 1, "text": "..."}, ...]
        }

        Args:
            source: Path to JSON file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If JSON format is invalid or data key not found
        """
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


class CSVDataLoader(DataLoader):
    """Data loader for CSV files."""

    def __init__(self, verbose: bool = True, **kwargs):
        """
        Initialize CSV data loader.

        Args:
            verbose: Whether to log information about loaded data
            **kwargs: Additional arguments for pandas.read_csv
        """
        super().__init__(verbose)
        self.csv_kwargs = kwargs

    def load(self, source: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            source: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If CSV format is invalid
        """
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


def create_data_loader(source: Union[str, Path], loader_type: str = "auto", **kwargs) -> DataLoader:
    """
    Factory function to create appropriate data loader.

    Args:
        source: Path to data file
        loader_type: Type of loader ('auto', 'json', 'database', 'csv')
        **kwargs: Additional arguments for the loader

    Returns:
        Configured data loader

    Raises:
        ValueError: If loader_type is not supported
    """
    source_path = Path(source)

    if loader_type == "auto":
        if source_path.suffix.lower() == '.json':
            try:
                with open(source_path, 'r') as f:
                    first_char = f.read(1)
                    if first_char == '[':
                        return DatabaseDataLoader(**kwargs)
                    else:
                        return JSONDataLoader(**kwargs)
            except:
                return JSONDataLoader(**kwargs)
        elif source_path.suffix.lower() == '.csv':
            return CSVDataLoader(**kwargs)
        else:
            raise ValueError(f"Cannot auto-detect loader for file: {source_path}")

    elif loader_type == "json":
        return JSONDataLoader(**kwargs)
    elif loader_type == "database":
        return DatabaseDataLoader(**kwargs)
    elif loader_type == "csv":
        return CSVDataLoader(**kwargs)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")


def validate_dataframe(
    data: pd.DataFrame,
    required_columns: list,
    text_column: str,
    class_column: str
) -> None:
    """
    Validate loaded DataFrame.

    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        text_column: Name of text column
        class_column: Name of class column

    Raises:
        ValueError: If validation fails
    """
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("DataFrame is empty")

    if text_column not in data.columns:
        raise ValueError(f"Text column '{text_column}' not found")

    empty_text = data[text_column].isnull() | (data[text_column].str.strip() == '')
    if empty_text.any():
        empty_count = empty_text.sum()
        logger = LoggerMixin().logger
        logger.warning(f"Found {empty_count} rows with empty text")

    if class_column not in data.columns:
        raise ValueError(f"Class column '{class_column}' not found")

    missing_classes = data[class_column].isnull()
    if missing_classes.any():
        missing_count = missing_classes.sum()
        logger = LoggerMixin().logger
        logger.warning(f"Found {missing_count} rows with missing class labels")