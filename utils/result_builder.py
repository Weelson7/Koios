"""Standard result structure builder for Koïos engine modules.

This module provides utilities for building consistent result dictionaries across all engine modules.

Standard Result Structure:
{
    'success': bool,           # Always required - whether computation succeeded
    'result': Any,             # Main computational result
    'result_type': str,        # Description of result type (e.g., 'matrix', 'scalar', 'function')
    'iterations': int,         # For iterative methods (optional)
    'converged': bool,         # For iterative methods (optional)
    'error': str,              # Error message if failed (optional)
    'warnings': List[str],     # Non-fatal warnings (optional)
    'computation_time': float, # Time in seconds (optional)
    'metadata': dict           # Additional engine-specific data (optional)
}
"""

from typing import Any, Dict, List, Optional, TypedDict
import time


class StandardResult(TypedDict, total=False):
    """TypedDict for standard result structure."""
    success: bool
    result: Any
    result_type: str
    iterations: int
    converged: bool
    error: str
    warnings: List[str]
    computation_time: float
    metadata: Dict[str, Any]


class ResultBuilder:
    """Builder class for creating standard result dictionaries."""
    
    def __init__(self):
        self._start_time = time.time()
        self._result: Dict[str, Any] = {
            'success': True,
            'warnings': [],
            'metadata': {}
        }
    
    def success(self, value: bool = True) -> 'ResultBuilder':
        """Set success status."""
        self._result['success'] = value
        return self
    
    def result(self, value: Any, result_type: str = 'unknown') -> 'ResultBuilder':
        """Set the main result and its type."""
        self._result['result'] = value
        self._result['result_type'] = result_type
        return self
    
    def error(self, error_message: str) -> 'ResultBuilder':
        """Set error message and mark as failed."""
        self._result['error'] = error_message
        self._result['success'] = False
        return self
    
    def iterations(self, count: int, converged: bool = True) -> 'ResultBuilder':
        """Set iteration count and convergence status."""
        self._result['iterations'] = count
        self._result['converged'] = converged
        return self
    
    def add_warning(self, warning: str) -> 'ResultBuilder':
        """Add a non-fatal warning message."""
        if 'warnings' not in self._result:
            self._result['warnings'] = []
        self._result['warnings'].append(warning)
        return self
    
    def metadata(self, key: str, value: Any) -> 'ResultBuilder':
        """Add metadata entry."""
        if 'metadata' not in self._result:
            self._result['metadata'] = {}
        self._result['metadata'][key] = value
        return self
    
    def merge_metadata(self, metadata: Dict[str, Any]) -> 'ResultBuilder':
        """Merge a dictionary into metadata."""
        if 'metadata' not in self._result:
            self._result['metadata'] = {}
        self._result['metadata'].update(metadata)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return the final result dictionary."""
        # Add computation time if not already set
        if 'computation_time' not in self._result:
            self._result['computation_time'] = time.time() - self._start_time
        
        # Clean up empty lists and dicts
        if not self._result.get('warnings'):
            self._result.pop('warnings', None)
        if not self._result.get('metadata'):
            self._result.pop('metadata', None)
        
        return self._result


def create_result(success: bool = True, result: Any = None, 
                 result_type: str = 'unknown', error: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
    """Convenience function to create a result dictionary.
    
    Args:
        success: Whether the operation succeeded
        result: The main result value
        result_type: Type description of the result
        error: Error message if failed
        **kwargs: Additional fields like iterations, warnings, metadata, etc.
    
    Returns:
        Standard result dictionary
    """
    builder = ResultBuilder().success(success)
    
    if result is not None:
        builder.result(result, result_type)
    
    if error:
        builder.error(error)
    
    for key, value in kwargs.items():
        if key == 'iterations':
            converged = kwargs.get('converged', True)
            builder.iterations(value, converged)
        elif key == 'warnings':
            if isinstance(value, list):
                for warning in value:
                    builder.add_warning(warning)
        elif key == 'metadata':
            if isinstance(value, dict):
                builder.merge_metadata(value)
    
    return builder.build()
