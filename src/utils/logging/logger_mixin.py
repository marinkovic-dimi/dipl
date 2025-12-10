import logging

from .setup import get_logger


class LoggerMixin:

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)
