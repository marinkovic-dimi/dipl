from typing import Optional
from .base import DataSplitter
from .stratified_splitter import StratifiedDataSplitter
from .random_splitter import RandomDataSplitter
from .time_based_splitter import TimeBasedDataSplitter


def create_data_splitter(
    splitter_type: str = "stratified",
    class_column: Optional[str] = None,
    **kwargs
) -> DataSplitter:
    if splitter_type == "stratified":
        if class_column is None:
            raise ValueError("class_column is required for stratified splitting")
        return StratifiedDataSplitter(class_column=class_column, **kwargs)
    elif splitter_type == "random":
        return RandomDataSplitter(**kwargs)
    elif splitter_type == "time":
        if 'time_column' not in kwargs:
            raise ValueError("time_column is required for time-based splitting")
        return TimeBasedDataSplitter(**kwargs)
    else:
        raise ValueError(f"Unsupported splitter type: {splitter_type}")
