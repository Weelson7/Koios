import re
import sympy as sp
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
import math

class MathValidators:
    """
    Input validation utilities for mathematical expressions and parameters
    """
    
    @staticmethod
    def validate_expression(expression: str) -> Dict[str, Any]:
        """
        Validate mathematical expression
        
        Returns:
            Dictionary with validation results including:
            - is_valid: boolean
            - error_message: string if invalid
            - variables: list of variables found
            - functions: list of functions found
            - complexity_score: integer rating complexity
        """
        result = {
            'is_valid': False,
            'error_message': None,
            'variables': [],
            'functions': [],
            'complexity_score': 0,
            'suggestions': []
        }
        
        # Basic checks
        if not expression or not isinstance(expression, str):
            result['error_message'] = "Expression cannot be empty"
            return result
        
        # Remove whitespace
        expr_clean = expression.replace(' ', '')
        
        # Check for balanced parentheses
        if not MathValidators._check_balanced_parentheses(expr_clean):
            result['error_message'] = "Unbalanced parentheses"
            result['suggestions'].append("Check that all opening parentheses have matching closing parentheses")
            return result
        
        # Check for valid characters (^ will be auto-converted to ** by expression_parser)
        valid_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^().,_= ')
        invalid_chars = set(expr_clean) - valid_chars
        if invalid_chars:
            result['error_message'] = f"Invalid characters found: {', '.join(invalid_chars)}"
            result['suggestions'].append("Use only letters, numbers, and mathematical operators")
            return result
        
        # Note: ^ is allowed but will be auto-converted to ** for exponentiation
        
        # Check for consecutive operators
        if MathValidators._has_consecutive_operators(expr_clean):
            result['error_message'] = "Consecutive operators found"
            result['suggestions'].append("Avoid consecutive operators like '++' or '**'")
            return result
        
        # Try to parse with SymPy
        try:
            from core.expression_parser import expression_parser
            parsed_expr = expression_parser.parse(expression)
            
            # Extract variables and functions
            result['variables'] = [str(var) for var in parsed_expr.free_symbols]
            result['functions'] = [str(func) for func in parsed_expr.atoms(sp.Function)]
            
            # Calculate complexity score
            result['complexity_score'] = MathValidators._calculate_complexity(parsed_expr)
            
            result['is_valid'] = True
            
        except Exception as e:
            result['error_message'] = f"Parse error: {str(e)}"
            result['suggestions'].extend(MathValidators._get_parse_suggestions(expression))
        
        return result
    
    @staticmethod
    def validate_matrix_input(matrix_data: List[List[Any]]) -> Dict[str, Any]:
        """
        Validate matrix input data
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': False,
            'error_message': None,
            'rows': 0,
            'cols': 0,
            'is_square': False,
            'is_numeric': True,
            'suggestions': []
        }
        
        if not matrix_data:
            result['error_message'] = "Matrix cannot be empty"
            return result
        
        # Check if all rows have same length
        row_lengths = [len(row) for row in matrix_data]
        if len(set(row_lengths)) > 1:
            result['error_message'] = "All rows must have the same number of columns"
            result['suggestions'].append("Ensure each row has the same number of elements")
            return result
        
        result['rows'] = len(matrix_data)
        result['cols'] = len(matrix_data[0]) if matrix_data else 0
        result['is_square'] = result['rows'] == result['cols']
        
        # Check if all elements are numeric
        for i, row in enumerate(matrix_data):
            for j, element in enumerate(row):
                if not MathValidators._is_numeric(element):
                    result['is_numeric'] = False
                    result['error_message'] = f"Non-numeric element at position ({i+1}, {j+1}): {element}"
                    result['suggestions'].append("All matrix elements must be numeric")
                    return result
        
        result['is_valid'] = True
        return result
    
    @staticmethod
    def validate_function_domain(expression: str, variable: str, 
                                x_min: float, x_max: float) -> Dict[str, Any]:
        """
        Validate function domain and identify potential issues
        
        Returns:
            Dictionary with domain validation results
        """
        result = {
            'is_valid': True,
            'warnings': [],
            'undefined_points': [],
            'discontinuities': [],
            'suggestions': []
        }
        
        try:
            from core.expression_parser import expression_parser
            expr = expression_parser.parse(expression)
            var_symbol = sp.Symbol(variable)
            
            # Check for division by zero
            denominators = MathValidators._find_denominators(expr)
            for denom in denominators:
                zeros = sp.solve(denom, var_symbol)
                for zero in zeros:
                    try:
                        zero_val = float(zero)
                        if x_min <= zero_val <= x_max:
                            result['discontinuities'].append(zero_val)
                            result['warnings'].append(f"Division by zero at x = {zero_val}")
                    except:
                        result['warnings'].append(f"Potential division by zero at x = {zero}")
            
            # Check for square roots of negative numbers
            sqrt_args = MathValidators._find_sqrt_arguments(expr)
            for arg in sqrt_args:
                # Check if argument can be negative in domain
                try:
                    # Test at endpoints and find critical points
                    critical_points = sp.solve(arg, var_symbol)
                    test_points = [x_min, x_max] + [float(pt) for pt in critical_points if x_min <= float(pt) <= x_max]
                    if any(arg.subs(var_symbol, pt) < 0 for pt in test_points):
                        result['warnings'].append("Square root may have negative argument in domain")
                except Exception:
                    result['warnings'].append("Could not verify square root domain")
            
            # Check for logarithms of non-positive numbers
            log_args = MathValidators._find_log_arguments(expr)
            for arg in log_args:
                # Check if argument can be non-positive in domain
                try:
                    critical_points = sp.solve(arg, var_symbol)
                    test_points = [x_min, x_max] + [float(pt) for pt in critical_points if x_min <= float(pt) <= x_max]
                    if any(arg.subs(var_symbol, pt) <= 0 for pt in test_points):
                        result['warnings'].append("Logarithm may have non-positive argument in domain")
                except Exception:
                    result['warnings'].append("Could not verify logarithm domain")
            
            # Add suggestions based on warnings
            if result['warnings']:
                result['suggestions'].append("Consider restricting the domain to avoid undefined regions")
                if result['discontinuities']:
                    result['suggestions'].append("Plot may have gaps at discontinuities")
        
        except Exception as e:
            result['warnings'].append(f"Could not analyze domain: {str(e)}")
        
        return result
    
    @staticmethod
    def validate_numerical_range(value: Union[float, int], 
                                min_val: Optional[float] = None,
                                max_val: Optional[float] = None,
                                name: str = "value") -> Dict[str, Any]:
        """
        Validate numerical value is within acceptable range
        """
        result = {
            'is_valid': True,
            'error_message': None,
            'warnings': [],
            'suggestions': []
        }
        
        if not isinstance(value, (int, float)):
            result['is_valid'] = False
            result['error_message'] = f"{name} must be a number"
            return result
        
        if math.isnan(value) or math.isinf(value):
            result['is_valid'] = False
            result['error_message'] = f"{name} cannot be NaN or infinite"
            return result
        
        if min_val is not None and value < min_val:
            result['is_valid'] = False
            result['error_message'] = f"{name} must be >= {min_val}"
            result['suggestions'].append(f"Try a value >= {min_val}")
            return result
        
        if max_val is not None and value > max_val:
            result['is_valid'] = False
            result['error_message'] = f"{name} must be <= {max_val}"
            result['suggestions'].append(f"Try a value <= {max_val}")
            return result
        
        if abs(value) > 1e10:
            result['warnings'].append(f"{name} is very large, may cause numerical issues")
        
        if abs(value) < 1e-10 and value != 0:
            result['warnings'].append(f"{name} is very small, may cause numerical issues")
        
        return result
    
    @staticmethod
    def validate_ode_input(ode_expression: str) -> Dict[str, Any]:
        """
        Validate ODE expression format
        """
        result = {
            'is_valid': False,
            'error_message': None,
            'order': None,
            'variables': [],
            'suggestions': []
        }
        
        if not ode_expression:
            result['error_message'] = "ODE expression cannot be empty"
            return result
        
        # Check for derivative notation
        has_derivative = any(notation in ode_expression for notation in ["'", "diff", "Derivative"])
        
        if not has_derivative:
            result['error_message'] = "ODE must contain derivative notation (', diff, or Derivative)"
            result['suggestions'].append("Use y' for dy/dx or y'' for d²y/dx²")
            return result
        
        # Try to determine order using proper parsing
        order = None
        try:
            from core.expression_parser import expression_parser
            expr = expression_parser.parse(ode_expression)
            derivatives = list(expr.atoms(sp.Derivative))
            if derivatives:
                order = max(deriv.derivative_count for deriv in derivatives)
            else:
                # Fallback: count consecutive apostrophes
                max_consecutive = 0
                current_consecutive = 0
                for char in ode_expression:
                    if char == "'":
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 0
                order = max_consecutive if max_consecutive > 0 else 1
        except Exception:
            # Fallback to counting apostrophes
            order = max(ode_expression.count("'"), 1)
        
        result['order'] = order
        
        # Basic validation passed
        result['is_valid'] = True
        
        return result
    
    @staticmethod
    def validate_physics_parameters(simulation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate physics simulation parameters
        
        Note: Constraints represent reasonable defaults for typical scenarios.
        For specialized cases outside these ranges, validation warnings may be
        issued but simulation will still proceed.
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Define parameter constraints for each simulation type (reasonable defaults)
        constraints = {
            'projectile_motion': {
                'v0': {'min': 0, 'max': 1000, 'name': 'Initial velocity'},
                'angle': {'min': 0, 'max': 90, 'name': 'Launch angle'},
                'g': {'min': 0.1, 'max': 50, 'name': 'Gravity'}
            },
            'simple_harmonic_motion': {
                'amplitude': {'min': 0.001, 'max': 100, 'name': 'Amplitude'},
                'frequency': {'min': 0.001, 'max': 100, 'name': 'Frequency'},
                'time_max': {'min': 0.1, 'max': 1000, 'name': 'Simulation time'}
            },
            'circuit_rc': {
                'R': {'min': 1, 'max': 1e9, 'name': 'Resistance'},
                'C': {'min': 1e-12, 'max': 1, 'name': 'Capacitance'},
                'time_max': {'min': 1e-6, 'max': 1000, 'name': 'Simulation time'}
            }
        }
        
        if simulation_type in constraints:
            param_constraints = constraints[simulation_type]
            
            for param_name, constraint in param_constraints.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    validation = MathValidators.validate_numerical_range(
                        value, 
                        constraint.get('min'), 
                        constraint.get('max'),
                        constraint['name']
                    )
                    
                    if not validation['is_valid']:
                        result['is_valid'] = False
                        result['errors'].append(validation['error_message'])
                        result['suggestions'].extend(validation['suggestions'])
                    
                    result['warnings'].extend(validation['warnings'])
        
        return result
    
    @staticmethod
    def _check_balanced_parentheses(expression: str) -> bool:
        """Check if parentheses are balanced"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in expression:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    @staticmethod
    def _has_consecutive_operators(expression: str) -> bool:
        """Check for consecutive operators. Allows valid patterns: ** (power), unary signs, and exponentiation patterns."""
        for i in range(len(expression) - 1):
            current = expression[i]
            next_char = expression[i+1]
            
            # Allow ** for exponentiation
            if expression[i:i+2] == '**':
                continue
            
            # Allow unary minus/plus after operators (*, /, ^, or opening parenthesis)
            if current in '*/^(' and next_char in '+-':
                continue
            
            # Allow unary minus/plus after comparison/assignment operators or commas
            if current in ',=<>!' and next_char in '+-':
                continue
            
            # Check for invalid consecutive binary operators (not unary signs or exponentiation)
            # Only flag if both are binary operators like +*, +/, -*,  -/, *+, *-, /+, /-
            binary_operators = '+-*/'
            if current in binary_operators and next_char in binary_operators:
                # But allow consecutive +/- (unary sign patterns like -+ or +-)
                if current in '+-' and next_char in '+-':
                    continue
                return True
                
        return False
    
    @staticmethod
    def _calculate_complexity(expr) -> int:
        """Calculate complexity score of expression"""
        complexity = 0
        
        # Count operations
        complexity += len(list(expr.atoms(sp.Add, sp.Mul, sp.Pow)))
        
        # Count functions
        complexity += len(list(expr.atoms(sp.Function)))
        
        # Count variables
        complexity += len(expr.free_symbols)
        
        # Add bonus for transcendental functions
        transcendental = [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]
        for func_type in transcendental:
            complexity += 2 * len(list(expr.atoms(func_type)))
        
        return complexity
    
    @staticmethod
    def _get_parse_suggestions(expression: str) -> List[str]:
        """Get suggestions for fixing parse errors"""
        suggestions = []
        
        if '*' not in expression and any(c.isdigit() for c in expression) and any(c.isalpha() for c in expression):
            suggestions.append("Use * for multiplication (e.g., 2*x instead of 2x)")
        
        if '^' in expression:
            suggestions.append("Use ** for exponentiation instead of ^")
        
        if expression.count('(') != expression.count(')'):
            suggestions.append("Check parentheses balance")
        
        if any(op in expression for op in ['++', '--', '**', '//']):
            suggestions.append("Check for double operators")
        
        return suggestions
    
    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Check if value can be converted to float"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def _find_denominators(expr) -> List:
        """Find all denominators in expression using SymPy's denom() function"""
        denominators = []
        
        # Use SymPy's built-in denom() to get the main denominator
        main_denom = sp.denom(expr)
        if main_denom != 1:
            denominators.append(main_denom)
        
        # Also check subexpressions for additional denominators
        for atom in sp.preorder_traversal(expr):
            if atom.is_Pow and atom.exp.is_negative:
                denominators.append(atom.base)
            elif atom.is_Mul:
                for arg in atom.args:
                    if arg.is_Pow and arg.exp.is_negative:
                        denominators.append(arg.base)
        
        return list(set(denominators))  # Remove duplicates
    
    @staticmethod
    def _find_sqrt_arguments(expr) -> List:
        """Find arguments of square root functions"""
        sqrt_args = []
        
        for atom in expr.atoms(sp.Pow):
            if atom.exp == sp.Rational(1, 2):
                sqrt_args.append(atom.base)
        
        for atom in expr.atoms(sp.sqrt):
            sqrt_args.append(atom.args[0])
        
        return sqrt_args
    
    @staticmethod
    def _find_log_arguments(expr) -> List:
        """Find arguments of logarithm functions"""
        log_args = []
        
        for atom in expr.atoms(sp.log):
            log_args.append(atom.args[0])
        
        return log_args


# Create a simple alias for backward compatibility
class ExpressionValidator:
    """Alias for MathValidators for backward compatibility"""
    
    @staticmethod
    def validate_expression(expression: str) -> Dict[str, Any]:
        return MathValidators.validate_expression(expression)
    
    @staticmethod
    def validate_matrix_input(matrix_data: List[List[Any]]) -> Dict[str, Any]:
        return MathValidators.validate_matrix_input(matrix_data)

# Create global instance
expression_validator = ExpressionValidator()

class InputSanitizer:
    """
    Sanitize and clean mathematical input
    """
    
    @staticmethod
    def sanitize_expression(expression: str) -> str:
        """Clean and standardize mathematical expression"""
        if not expression:
            return ""
        
        # Remove extra whitespace
        expr = ' '.join(expression.split())
        
        # Replace common alternative notations
        replacements = {
            '^': '**',      # Power notation
            'ln(': 'log(',  # Natural logarithm
            '×': '*',       # Multiplication
            '÷': '/',       # Division
            '√': 'sqrt',    # Square root
            'π': 'pi',      # Pi
            '∞': 'oo',      # Infinity
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # Add implicit multiplication
        expr = InputSanitizer._add_implicit_multiplication(expr)
        
        return expr
    
    @staticmethod
    def sanitize_matrix_input(matrix_str: str) -> List[List[float]]:
        """Parse and sanitize matrix input from string"""
        try:
            # Remove brackets and split by rows
            matrix_str = matrix_str.strip('[]{}()')
            rows = matrix_str.split(';') if ';' in matrix_str else matrix_str.split('\n')
            
            matrix = []
            for row in rows:
                row = row.strip()
                if row:
                    # Split by comma or space
                    elements = row.replace(',', ' ').split()
                    row_values = [float(elem) for elem in elements if elem]
                    if row_values:
                        matrix.append(row_values)
            
            return matrix
        
        except (ValueError, IndexError):
            raise ValueError("Invalid matrix format")
    
    @staticmethod
    def _add_implicit_multiplication(expr: str) -> str:
        """Add implicit multiplication operators
        
        Note: This function treats patterns like '2x' as '2*x' (multiplication).
        If you have variables with numbers in their names (e.g., 'x2' as a single variable),
        use explicit naming or underscores (e.g., 'x_2') to avoid ambiguity.
        
        Supported patterns:
        - number + letter: '2x' -> '2*x'
        - letter + number: 'x2' -> 'x*2' (treats as multiplication)
        - ')' + alphanumeric or '(': '(x)(y)' -> '(x)*(y)'
        - alphanumeric + '(': 'x(y)' -> 'x*(y)'
        """
        result = ""
        for i, char in enumerate(expr):
            result += char
            
            # Add * between number and letter
            if i < len(expr) - 1:
                current = char
                next_char = expr[i + 1]
                
                # Cases where we need implicit multiplication
                if ((current.isdigit() and next_char.isalpha()) or
                    (current.isalpha() and next_char.isdigit()) or
                    (current == ')' and (next_char.isalnum() or next_char == '(')) or
                    (current.isalnum() and next_char == '(')):
                    result += '*'
        
        return result

# Global instances
validators = MathValidators()
sanitizer = InputSanitizer()
