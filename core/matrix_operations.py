import numpy as np
import sympy as sp
from sympy import Matrix as SpMatrix
from typing import List, Union, Dict, Any, Optional
import pandas as pd
from functools import lru_cache
from utils.exceptions import (
    SingularMatrixError, InvalidDimensionError, InvalidInputError,
    NumericalInstabilityError
)

class MatrixOperations:
    """
    Comprehensive matrix operations using both NumPy and SymPy
    """
    
    def __init__(self):
        self.precision = 15
    
    def create_matrix(self, data: List[List[Union[int, float, str]]], symbolic: bool = False) -> Dict[str, Any]:
        """
        Create a matrix from input data
        
        Args:
            data: 2D list representing matrix elements
            symbolic: Whether to create symbolic matrix
            
        Returns:
            Dictionary with matrix creation results
        """
        result = {
            'success': False,
            'matrix': None,
            'shape': None,
            'type': None,
            'error': None
        }
        
        try:
            if symbolic:
                # Create SymPy matrix for symbolic computation
                matrix = SpMatrix(data)
                result['type'] = 'symbolic'
            else:
                # Create NumPy matrix for numerical computation
                matrix = np.array(data, dtype=float)
                result['type'] = 'numeric'
            
            result['success'] = True
            result['matrix'] = matrix
            result['shape'] = matrix.shape
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_addition(self, matrix1: Union[np.ndarray, SpMatrix], matrix2: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Add two matrices
        """
        result = {
            'success': False,
            'result_matrix': None,
            'error': None
        }
        
        try:
            if matrix1.shape != matrix2.shape:
                raise InvalidDimensionError("addition", matrix1.shape, matrix2.shape)
            
            result_matrix = matrix1 + matrix2
            
            result['success'] = True
            result['result_matrix'] = result_matrix
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_multiplication(self, matrix1: Union[np.ndarray, SpMatrix], matrix2: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Multiply two matrices
        """
        result = {
            'success': False,
            'result_matrix': None,
            'error': None
        }
        
        try:
            if isinstance(matrix1, np.ndarray) != isinstance(matrix2, np.ndarray):
                raise InvalidInputError("matrix types", 
                    f"matrix1={type(matrix1).__name__}, matrix2={type(matrix2).__name__}",
                    "both must be NumPy arrays or both must be SymPy matrices")
            
            if isinstance(matrix1, np.ndarray):
                result_matrix = np.dot(matrix1, matrix2)
            else:
                result_matrix = matrix1 * matrix2
            
            result['success'] = True
            result['result_matrix'] = result_matrix
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_determinant(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Calculate matrix determinant
        """
        result = {
            'success': False,
            'determinant': None,
            'error': None
        }
        
        try:
            if matrix.shape[0] != matrix.shape[1]:
                raise InvalidDimensionError("determinant", matrix.shape)
            
            if isinstance(matrix, np.ndarray):
                det = np.linalg.det(matrix)
            else:
                det = matrix.det()
            
            result['success'] = True
            result['determinant'] = det
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_inverse(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Calculate matrix inverse
        """
        result = {
            'success': False,
            'inverse_matrix': None,
            'error': None
        }
        
        try:
            if matrix.shape[0] != matrix.shape[1]:
                raise InvalidDimensionError("inverse", matrix.shape)
            
            if isinstance(matrix, np.ndarray):
                det = np.linalg.det(matrix)
                if abs(det) < np.finfo(float).eps * matrix.shape[0]:
                    raise SingularMatrixError("inverse")
                inverse_matrix = np.linalg.inv(matrix)
            else:
                try:
                    inverse_matrix = matrix.inv()
                except:
                    raise SingularMatrixError("inverse")
            
            result['success'] = True
            result['inverse_matrix'] = inverse_matrix
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_eigenvalues(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Calculate eigenvalues and eigenvectors
        """
        result = {
            'success': False,
            'eigenvalues': None,
            'eigenvectors': None,
            'error': None
        }
        
        try:
            if matrix.shape[0] != matrix.shape[1]:
                raise InvalidDimensionError("eigenvalues", matrix.shape)
            
            if isinstance(matrix, np.ndarray):
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                result['eigenvalues'] = eigenvalues.tolist()
                result['eigenvectors'] = eigenvectors.tolist()
            else:
                eigenvals = matrix.eigenvals()
                eigenvects = matrix.eigenvects()
                result['eigenvalues'] = list(eigenvals.keys())
                result['eigenvectors'] = [vect[2] for vect in eigenvects]
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_rank(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Calculate matrix rank
        """
        result = {
            'success': False,
            'rank': None,
            'error': None
        }
        
        try:
            if isinstance(matrix, np.ndarray):
                rank = np.linalg.matrix_rank(matrix)
            else:
                rank = matrix.rank()
            
            result['success'] = True
            result['rank'] = rank
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_transpose(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Calculate matrix transpose
        """
        result = {
            'success': False,
            'transpose_matrix': None,
            'error': None
        }
        
        try:
            if isinstance(matrix, np.ndarray):
                transpose_matrix = matrix.T
            else:
                transpose_matrix = matrix.T
            
            result['success'] = True
            result['transpose_matrix'] = transpose_matrix
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_linear_system(self, coefficient_matrix: Union[np.ndarray, SpMatrix], 
                          constants_vector: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """
        Solve system of linear equations Ax = b
        """
        result = {
            'success': False,
            'solution': None,
            'error': None
        }
        
        try:
            if isinstance(coefficient_matrix, np.ndarray):
                solution = np.linalg.solve(coefficient_matrix, constants_vector)
                result['solution'] = solution.tolist()
            else:
                solution = coefficient_matrix.LUsolve(constants_vector)
                result['solution'] = solution
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def matrix_norm(self, matrix: Union[np.ndarray, SpMatrix], norm_type: str = 'frobenius') -> Dict[str, Any]:
        """
        Calculate matrix norm
        """
        result = {
            'success': False,
            'norm': None,
            'error': None
        }
        
        try:
            if isinstance(matrix, np.ndarray):
                if norm_type == 'frobenius':
                    norm = np.linalg.norm(matrix, 'fro')
                elif norm_type == '1':
                    norm = np.linalg.norm(matrix, 1)
                elif norm_type == '2':
                    norm = np.linalg.norm(matrix, 2)
                elif norm_type == 'inf':
                    norm = np.linalg.norm(matrix, np.inf)
                else:
                    raise InvalidInputError("norm type", norm_type, "frobenius, 1, 2, or inf")
            else:
                # SymPy doesn't have built-in matrix norms, calculate manually
                # Flatten the matrix to iterate over all elements
                norm = sp.sqrt(sum(element**2 for row in matrix.tolist() for element in row))
            
            result['success'] = True
            result['norm'] = float(norm) if isinstance(norm, (np.number, sp.Basic)) else norm
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def determinant(self, matrix_data: List[List[Union[int, float]]]) -> Union[float, None]:
        """
        Simple determinant method for testing compatibility
        """
        # Validate input
        if not matrix_data or len(matrix_data) == 0:
            raise InvalidInputError("matrix data", "empty", "non-empty 2D list")
            
        # Check for non-numeric values
        for row in matrix_data:
            if not row:
                raise InvalidInputError("matrix row", "empty", "row with at least one element")
            for val in row:
                if isinstance(val, str):
                    raise InvalidInputError("matrix element", str(val), "numeric value")
                    
        try:
            matrix_result = self.create_matrix(matrix_data, symbolic=False)
            if matrix_result['success']:
                det_result = self.matrix_determinant(matrix_result['matrix'])
                if det_result['success']:
                    return det_result['determinant']
            if matrix_result.get('error'):
                raise InvalidInputError("matrix creation", matrix_result['error'], "valid numeric matrix")
            return None
        except InvalidInputError:
            raise
        except Exception as e:
            raise NumericalInstabilityError("determinant calculation", str(e))
    
    def transpose(self, matrix_data: List[List[Union[int, float]]]) -> Union[List[List[float]], None]:
        """
        Simple transpose method for testing compatibility
        """
        try:
            matrix_result = self.create_matrix(matrix_data, symbolic=False)
            if matrix_result['success']:
                transpose_result = self.matrix_transpose(matrix_result['matrix'])
                if transpose_result['success']:
                    return transpose_result['transpose_matrix'].tolist()
            if matrix_result.get('error'):
                raise InvalidInputError("matrix creation", matrix_result['error'], "valid numeric matrix")
            return None
        except InvalidInputError:
            raise
        except Exception as e:
            raise NumericalInstabilityError("transpose calculation", str(e))
    
    def rank(self, matrix_data: List[List[Union[int, float]]]) -> Union[int, None]:
        """
        Simple rank method for testing compatibility
        """
        try:
            matrix_result = self.create_matrix(matrix_data, symbolic=False)
            if matrix_result['success']:
                rank_result = self.matrix_rank(matrix_result['matrix'])
                if rank_result['success']:
                    return rank_result['rank']
            if matrix_result.get('error'):
                raise InvalidInputError("matrix creation", matrix_result['error'], "valid numeric matrix")
            return None
        except InvalidInputError:
            raise
        except Exception as e:
            raise NumericalInstabilityError("rank calculation", str(e))
    
    def generate_random_matrix(self, rows: int, cols: int, min_val: float = -10, max_val: float = 10) -> Dict[str, Any]:
        """
        Generate a random matrix with specified dimensions
        
        Args:
            rows: Number of rows
            cols: Number of columns  
            min_val: Minimum value for random numbers
            max_val: Maximum value for random numbers
            
        Returns:
            Dictionary with success status and generated matrix
        """
        result = {
            'success': False,
            'matrix': None,
            'error': None
        }
        
        try:
            if rows <= 0 or cols <= 0:
                raise InvalidInputError("matrix dimensions", f"({rows}x{cols})", "positive dimensions")
            if min_val >= max_val:
                raise InvalidInputError("value range", f"min={min_val}, max={max_val}", "min_val < max_val")
                
            # Generate random matrix
            random_matrix = np.random.uniform(min_val, max_val, size=(rows, cols))
            
            result['success'] = True
            result['matrix'] = random_matrix
            result['shape'] = (rows, cols)
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def create_predefined_matrix(self, matrix_type: str, size: int) -> Dict[str, Any]:
        """
        Create a predefined matrix (identity, zeros, ones)
        
        Args:
            matrix_type: Type of matrix ('identity', 'zeros', 'ones')
            size: Size of the square matrix
            
        Returns:
            Dictionary with success status and created matrix
        """
        result = {
            'success': False,
            'matrix': None,
            'error': None
        }
        
        try:
            if size <= 0:
                raise InvalidInputError("matrix size", str(size), "positive integer")
                
            if matrix_type == 'identity':
                matrix = np.eye(size)
            elif matrix_type == 'zeros':
                matrix = np.zeros((size, size))
            elif matrix_type == 'ones':
                matrix = np.ones((size, size))
            else:
                raise InvalidInputError("matrix type", matrix_type, "identity, zero, or ones")
                
            result['success'] = True
            result['matrix'] = matrix
            result['type'] = matrix_type
            result['size'] = size
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    # Wrapper methods for compatibility
    def add(self, matrix1: Union[np.ndarray, SpMatrix], matrix2: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """Wrapper for matrix_addition for compatibility"""
        return self.matrix_addition(matrix1, matrix2)
    
    def multiply(self, matrix1: Union[np.ndarray, SpMatrix], matrix2: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """Wrapper for matrix_multiplication for compatibility"""
        return self.matrix_multiplication(matrix1, matrix2)
    
    def inverse(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """Wrapper for matrix_inverse for compatibility"""
        return self.matrix_inverse(matrix)
    
    def eigenvalues(self, matrix: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """Wrapper for matrix_eigenvalues for compatibility"""
        return self.matrix_eigenvalues(matrix)
    
    def norm(self, matrix: Union[np.ndarray, SpMatrix], norm_type: str = 'frobenius') -> Dict[str, Any]:
        """Wrapper for matrix_norm for compatibility"""
        return self.matrix_norm(matrix, norm_type)
    
    def create_identity_matrix(self, size: int) -> Dict[str, Any]:
        """Create an identity matrix of given size"""
        return self.create_predefined_matrix('identity', size)
    
    def solve(self, coefficient_matrix: Union[np.ndarray, SpMatrix], 
             constant_vector: Union[np.ndarray, SpMatrix]) -> Dict[str, Any]:
        """Wrapper for solve_linear_system for compatibility"""
        return self.solve_linear_system(coefficient_matrix, constant_vector)

# Global matrix operations instance
matrix_operations = MatrixOperations()
