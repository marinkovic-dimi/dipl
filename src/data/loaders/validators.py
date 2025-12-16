import pandas as pd
from ...utils import LoggerMixin


def validate_dataframe(
    data: pd.DataFrame,
    required_columns: list,
    text_column: str,
    class_column: str
) -> None:
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
