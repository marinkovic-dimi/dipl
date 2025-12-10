import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
from .logging import get_logger

logger = get_logger(__name__)


class SerializationManager:
    """Manager for various file serialization operations."""

    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path], **kwargs) -> None:
        """
        Save data to JSON file.

        Args:
            data: Data to save
            file_path: Path to save file
            **kwargs: Additional arguments for json.dump
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        default_kwargs = {'ensure_ascii': False, 'indent': 2}
        default_kwargs.update(kwargs)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **default_kwargs)

        logger.info(f"Saved JSON data to {file_path}")

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Any:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Loaded data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded JSON data from {file_path}")
        return data

    @staticmethod
    def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
        """
        Save data to pickle file.

        Args:
            data: Data to save
            file_path: Path to save file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved pickle data to {file_path}")

    @staticmethod
    def load_pickle(file_path: Union[str, Path]) -> Any:
        """
        Load data from pickle file.

        Args:
            file_path: Path to pickle file

        Returns:
            Loaded data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        logger.info(f"Loaded pickle data from {file_path}")
        return data

    @staticmethod
    def save_dataframe_json(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        orient: str = 'records',
        lines: bool = True
    ) -> None:
        """
        Save DataFrame to JSON file.

        Args:
            df: DataFrame to save
            file_path: Path to save file
            orient: JSON orientation
            lines: Whether to use line-delimited JSON
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_json(file_path, orient=orient, lines=lines, force_ascii=False)
        logger.info(f"Saved DataFrame to {file_path} ({len(df)} rows)")

    @staticmethod
    def load_dataframe_json(
        file_path: Union[str, Path],
        orient: str = 'records',
        lines: bool = True
    ) -> pd.DataFrame:
        """
        Load DataFrame from JSON file.

        Args:
            file_path: Path to JSON file
            orient: JSON orientation
            lines: Whether to expect line-delimited JSON

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        df = pd.read_json(file_path, orient=orient, lines=lines)
        logger.info(f"Loaded DataFrame from {file_path} ({len(df)} rows)")
        return df

    @staticmethod
    def save_train_val_test(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        base_path: Union[str, Path],
        suffix: str = ""
    ) -> Tuple[Path, Path, Path]:
        """
        Save train/validation/test DataFrames.

        Args:
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            base_path: Base path for files
            suffix: Optional suffix for filenames

        Returns:
            Tuple of saved file paths
        """
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = f"_{suffix}" if suffix else ""

        train_path = base_path.parent / f"{base_path.stem}_train{suffix}.json"
        val_path = base_path.parent / f"{base_path.stem}_val{suffix}.json"
        test_path = base_path.parent / f"{base_path.stem}_test{suffix}.json"

        SerializationManager.save_dataframe_json(train, train_path)
        SerializationManager.save_dataframe_json(val, val_path)
        SerializationManager.save_dataframe_json(test, test_path)

        logger.info(f"Saved train/val/test split: {len(train)}/{len(val)}/{len(test)} samples")
        return train_path, val_path, test_path

    @staticmethod
    def load_train_val_test(
        base_path: Union[str, Path],
        suffix: str = ""
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train/validation/test DataFrames.

        Args:
            base_path: Base path for files
            suffix: Optional suffix for filenames

        Returns:
            Tuple of loaded DataFrames
        """
        base_path = Path(base_path)
        suffix = f"_{suffix}" if suffix else ""

        train_path = base_path.parent / f"{base_path.stem}_train{suffix}.json"
        val_path = base_path.parent / f"{base_path.stem}_val{suffix}.json"
        test_path = base_path.parent / f"{base_path.stem}_test{suffix}.json"

        train = SerializationManager.load_dataframe_json(train_path)
        val = SerializationManager.load_dataframe_json(val_path)
        test = SerializationManager.load_dataframe_json(test_path)

        logger.info(f"Loaded train/val/test split: {len(train)}/{len(val)}/{len(test)} samples")
        return train, val, test


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        File size string
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return "File not found"

    size_bytes = file_path.stat().st_size

    if size_bytes == 0:
        return "0 B"

    size_units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0

    while size_bytes >= 1024 and unit_index < len(size_units) - 1:
        size_bytes /= 1024
        unit_index += 1

    return f"{size_bytes:.1f} {size_units[unit_index]}"