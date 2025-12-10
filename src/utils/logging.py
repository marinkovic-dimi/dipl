import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']

        record.levelname = f"{log_color}{record.levelname}{reset_color}"

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    experiment_name: Optional[str] = None,
    colored_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        experiment_name: Name for log file
        colored_output: Whether to use colored console output

    Returns:
        Configured logger instance
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)

    logger = logging.getLogger("ad_classifier")
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if colored_output:
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    else:
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt="%H:%M:%S")

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{experiment_name or 'experiment'}_{timestamp}.log"
        file_handler = logging.FileHandler(log_dir_path / log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        file_format = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_dir_path / log_filename}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')

    return logging.getLogger(f"ad_classifier.{name}")


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)