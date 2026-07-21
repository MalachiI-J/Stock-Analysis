"""Application errors and stable command-line exit codes."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Documented process outcomes used by the command-line interface."""

    SUCCESS = 0
    INVALID_ARGUMENTS = 2
    INVALID_DATE = 3
    INVALID_CONFIGURATION = 4
    MISSING_DATA = 5
    PARTIAL_FAILURE = 6
    DATABASE_FAILURE = 7
    OPERATION_FAILED = 8


class StockScrapperError(Exception):
    """Base class for expected operational failures."""


class InvalidDateError(StockScrapperError):
    """A supplied date is malformed or logically invalid."""


class InvalidConfigurationError(StockScrapperError):
    """A configuration file failed deterministic validation."""


class MissingDataError(StockScrapperError):
    """The requested operation has no eligible stored data."""


class PartialFailureError(StockScrapperError):
    """Some, but not all, requested work failed."""


class OperationFailedError(StockScrapperError):
    """The complete requested operation failed."""
