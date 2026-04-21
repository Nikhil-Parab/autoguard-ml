"""Custom exceptions for AutoGuard."""
from __future__ import annotations


class AutoGuardError(Exception):
    """Base exception for all AutoGuard errors."""


class DataError(AutoGuardError):
    """Raised when dataset validation or preprocessing fails."""


class NotFittedError(AutoGuardError):
    """Raised when attempting inference before .fit() is called."""


class AutoMLError(AutoGuardError):
    """Raised when the AutoML engine encounters an unrecoverable error."""


class DriftError(AutoGuardError):
    """Raised when drift detection cannot proceed."""


class ExplainError(AutoGuardError):
    """Raised when SHAP computation fails."""


class ConfigError(AutoGuardError):
    """Raised when configuration is invalid."""


class PluginError(AutoGuardError):
    """Raised when a model plugin fails to register."""
