"""Utility modules and helper functions for Koïos."""

from utils.exceptions import (
    KoiosError,
    InvalidDimensionError,
    InvalidInputError,
    NumericalInstabilityError,
    SingularMatrixError,
    ConvergenceError,
    DomainError,
    ExpressionParseError,
    UndefinedOperationError,
    ConfigurationError,
)
from utils.math_helpers import clamp, is_close, safe_divide
from utils.result_builder import ResultBuilder, create_result
from utils.validators import validate_number, validate_dimension, validate_domain

__all__ = [
    'KoiosError',
    'InvalidDimensionError',
    'InvalidInputError',
    'NumericalInstabilityError',
    'SingularMatrixError',
    'ConvergenceError',
    'DomainError',
    'ExpressionParseError',
    'UndefinedOperationError',
    'ConfigurationError',
    'clamp',
    'is_close',
    'safe_divide',
    'ResultBuilder',
    'create_result',
    'validate_number',
    'validate_dimension',
    'validate_domain',
]
