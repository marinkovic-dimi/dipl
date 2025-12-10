import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict, Any
from sklearn.model_selection import train_test_split
from ..utils import LoggerMixin


class DataSplitter(ABC, LoggerMixin):
    """Abstract base class for data splitters."""

    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Initialize data splitter.

        Args:
            random_state: Random state for reproducibility
            verbose: Whether to log splitting information
        """
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        pass

    def _log_split_results(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        class_column: Optional[str] = None
    ) -> None:
        """Log splitting results."""
        if self.verbose:
            total_size = len(train) + len(val) + len(test)
            train_pct = len(train) / total_size * 100
            val_pct = len(val) / total_size * 100
            test_pct = len(test) / total_size * 100

            self.logger.info("Data split results:")
            self.logger.info(f"  Train: {len(train)} samples ({train_pct:.1f}%)")
            self.logger.info(f"  Validation: {len(val)} samples ({val_pct:.1f}%)")
            self.logger.info(f"  Test: {len(test)} samples ({test_pct:.1f}%)")

            if class_column and class_column in train.columns:
                train_classes = len(train[class_column].unique())
                val_classes = len(val[class_column].unique())
                test_classes = len(test[class_column].unique())

                self.logger.info("Class distribution:")
                self.logger.info(f"  Train classes: {train_classes}")
                self.logger.info(f"  Validation classes: {val_classes}")
                self.logger.info(f"  Test classes: {test_classes}")


class StratifiedDataSplitter(DataSplitter):
    """Stratified splitter that maintains class distribution across splits."""

    def __init__(
        self,
        class_column: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize stratified data splitter.

        Args:
            class_column: Name of the class column for stratification
            val_size: Proportion of data for validation (0.0 to 1.0)
            test_size: Proportion of data for test (0.0 to 1.0)
            random_state: Random state for reproducibility
            verbose: Whether to log splitting information
        """
        super().__init__(random_state, verbose)
        self.class_column = class_column
        self.val_size = val_size
        self.test_size = test_size

        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be between 0.0 and 1.0")
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")
        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1.0")

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data using stratified sampling.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if self.class_column not in data.columns:
            raise ValueError(f"Class column '{self.class_column}' not found in data")

        if self.verbose:
            self.logger.info(f"Stratified splitting with val_size={self.val_size}, test_size={self.test_size}")

        temp_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            stratify=data[self.class_column],
            random_state=self.random_state
        )

        adjusted_val_size = self.val_size / (1 - self.test_size)

        train_data, val_data = train_test_split(
            temp_data,
            test_size=adjusted_val_size,
            stratify=temp_data[self.class_column],
            random_state=self.random_state
        )

        self._log_split_results(train_data, val_data, test_data, self.class_column)
        return train_data, val_data, test_data

    def validate_stratification(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate that stratification worked correctly.

        Args:
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame

        Returns:
            Validation results dictionary
        """
        train_dist = train[self.class_column].value_counts(normalize=True).sort_index()
        val_dist = val[self.class_column].value_counts(normalize=True).sort_index()
        test_dist = test[self.class_column].value_counts(normalize=True).sort_index()

        common_classes = set(train_dist.index) & set(val_dist.index) & set(test_dist.index)

        if not common_classes:
            return {
                'stratification_valid': False,
                'error': 'No common classes across all splits'
            }

        max_deviation = 0.0
        class_deviations = {}

        for class_id in common_classes:
            train_prop = train_dist.get(class_id, 0)
            val_prop = val_dist.get(class_id, 0)
            test_prop = test_dist.get(class_id, 0)

            deviation = max(
                abs(train_prop - val_prop),
                abs(train_prop - test_prop),
                abs(val_prop - test_prop)
            )

            class_deviations[class_id] = deviation
            max_deviation = max(max_deviation, deviation)

        stratification_valid = max_deviation < 0.05

        return {
            'stratification_valid': stratification_valid,
            'max_deviation': max_deviation,
            'class_deviations': class_deviations,
            'common_classes': len(common_classes),
            'total_classes': {
                'train': len(train_dist),
                'val': len(val_dist),
                'test': len(test_dist)
            }
        }


class RandomDataSplitter(DataSplitter):
    """Random splitter without stratification."""

    def __init__(
        self,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize random data splitter.

        Args:
            val_size: Proportion of data for validation (0.0 to 1.0)
            test_size: Proportion of data for test (0.0 to 1.0)
            random_state: Random state for reproducibility
            verbose: Whether to log splitting information
        """
        super().__init__(random_state, verbose)
        self.val_size = val_size
        self.test_size = test_size

        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be between 0.0 and 1.0")
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be between 0.0 and 1.0")
        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1.0")

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data using random sampling.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if self.verbose:
            self.logger.info(f"Random splitting with val_size={self.val_size}, test_size={self.test_size}")

        temp_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state
        )

        adjusted_val_size = self.val_size / (1 - self.test_size)

        train_data, val_data = train_test_split(
            temp_data,
            test_size=adjusted_val_size,
            random_state=self.random_state
        )

        self._log_split_results(train_data, val_data, test_data)
        return train_data, val_data, test_data


class TimeBasedDataSplitter(DataSplitter):
    """Time-based splitter for temporal data."""

    def __init__(
        self,
        time_column: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize time-based data splitter.

        Args:
            time_column: Name of the time column
            val_size: Proportion of data for validation (0.0 to 1.0)
            test_size: Proportion of data for test (0.0 to 1.0)
            verbose: Whether to log splitting information
        """
        super().__init__(random_state=None, verbose=verbose)
        self.time_column = time_column
        self.val_size = val_size
        self.test_size = test_size

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data based on time ordering.

        Args:
            data: Input DataFrame with time column

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")

        if self.verbose:
            self.logger.info(f"Time-based splitting with val_size={self.val_size}, test_size={self.test_size}")

        sorted_data = data.sort_values(self.time_column)

        total_size = len(sorted_data)
        test_start = int(total_size * (1 - self.test_size))
        val_start = int(total_size * (1 - self.test_size - self.val_size))

        train_data = sorted_data.iloc[:val_start].copy()
        val_data = sorted_data.iloc[val_start:test_start].copy()
        test_data = sorted_data.iloc[test_start:].copy()

        self._log_split_results(train_data, val_data, test_data)
        return train_data, val_data, test_data


def create_data_splitter(
    splitter_type: str = "stratified",
    class_column: Optional[str] = None,
    **kwargs
) -> DataSplitter:
    """
    Factory function to create data splitter.

    Args:
        splitter_type: Type of splitter ('stratified', 'random', 'time')
        class_column: Name of class column (required for stratified)
        **kwargs: Additional arguments for the splitter

    Returns:
        Configured data splitter

    Raises:
        ValueError: If splitter_type is not supported or required args missing
    """
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