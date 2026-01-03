"""Custom exception classes for Koïos calculator with user-friendly messages.

Usage Guide:
- InvalidDimensionError: For matrix shape mismatches (e.g., adding incompatible matrices)
- InvalidInputError: For malformed user inputs or parsing failures
- NumericalInstabilityError: For numerical issues (e.g., ill-conditioned systems)
- SingularMatrixError: For singular matrix operations (determinant = 0)
- ConvergenceError: For iterative algorithm failures
- DomainError: For out-of-domain function inputs
- ExpressionParseError: For expression parsing errors
- UndefinedOperationError: For mathematically undefined operations
- ConfigurationError: For invalid configuration parameters

All module code should import and use these exceptions instead of generic ValueError/TypeError.
"""


class KoiosError(Exception):
    """Base exception class for all Koïos errors."""
    
    def __init__(self, message: str, hint: str = ""):
        self.message = message
        self.hint = hint
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format the error message with optional hint."""
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class InvalidDimensionError(KoiosError):
    """Raised when matrix dimensions are incompatible for an operation."""
    
    def __init__(self, operation: str, shape1: tuple, shape2: tuple = None):
        if shape2:
            message = f"Cannot perform {operation}: incompatible dimensions {shape1} and {shape2}"
            hint = f"For {operation}, ensure dimension requirements are met"
        else:
            message = f"Invalid dimension {shape1} for operation {operation}"
            hint = "Check that your matrix has the correct shape"
        super().__init__(message, hint)


class InvalidInputError(KoiosError):
    """Raised when user input is invalid or cannot be parsed."""
    
    def __init__(self, input_name: str, value: str, expected: str = ""):
        message = f"Invalid input for {input_name}: '{value}'"
        hint = f"Expected {expected}" if expected else "Check the format and try again"
        super().__init__(message, hint)


class NumericalInstabilityError(KoiosError):
    """Raised when numerical computation becomes unstable."""
    
    def __init__(self, operation: str, reason: str = ""):
        message = f"Numerical instability in {operation}"
        hint = reason or "Try different parameters or increase precision"
        super().__init__(message, hint)


class SingularMatrixError(KoiosError):
    """Raised when attempting operations on singular (non-invertible) matrices."""
    
    def __init__(self, operation: str = "matrix operation"):
        message = f"Cannot perform {operation}: matrix is singular (determinant = 0)"
        hint = "Singular matrices cannot be inverted or used in certain operations"
        super().__init__(message, hint)


class ConvergenceError(KoiosError):
    """Raised when iterative algorithms fail to converge."""
    
    def __init__(self, algorithm: str, iterations: int = 0):
        message = f"{algorithm} failed to converge"
        if iterations:
            hint = f"Reached maximum {iterations} iterations. Try different initial values or increase tolerance"
        else:
            hint = "Try different initial values, increase max iterations, or adjust tolerance"
        super().__init__(message, hint)


class DomainError(KoiosError):
    """Raised when input value is outside the valid domain for a function."""
    
    def __init__(self, function: str, value: str, valid_domain: str):
        message = f"Domain error in {function}: {value} is not in valid domain"
        hint = f"Valid domain: {valid_domain}"
        super().__init__(message, hint)


class ExpressionParseError(KoiosError):
    """Raised when mathematical expression cannot be parsed."""
    
    def __init__(self, expression: str, details: str = ""):
        message = f"Cannot parse expression: '{expression}'"
        hint = details or "Check syntax, parentheses matching, and function names"
        super().__init__(message, hint)


class UndefinedOperationError(KoiosError):
    """Raised when operation is mathematically undefined."""
    
    def __init__(self, operation: str, reason: str):
        message = f"Operation {operation} is undefined: {reason}"
        hint = "This operation cannot be performed with the given inputs"
        super().__init__(message, hint)


class ConfigurationError(KoiosError):
    """Raised when configuration or parameters are invalid."""
    
    def __init__(self, parameter: str, value: str, requirement: str):
        message = f"Invalid configuration for {parameter}: {value}"
        hint = f"Requirement: {requirement}"
        super().__init__(message, hint)
