import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import re
from typing import Any, Dict, List, Union
from utils.exceptions import ExpressionParseError

class ExpressionParser:
    """
    Advanced expression parser for mathematical expressions using SymPy
    """
    
    def __init__(self):
        self.transformations = (standard_transformations + 
                              (implicit_multiplication_application,))
        self.local_dict = self._setup_symbols()

        # Maximum expression length to reduce injection risk
        self.max_expression_length = 500

    def _validate_expression(self, expression: str) -> None:
        """Basic validation to reduce injection surface before parsing."""
        if len(expression) > self.max_expression_length:
            raise ExpressionParseError(expression, "Expression exceeds maximum length of 500 characters")

        # Block common code-injection markers and dangerous tokens
        blacklist = [
            "__", "import", "sys.", "os.", "subprocess", "eval(", "exec(",
            "open(", "globals()", "locals()", "getattr", "setattr", "lambda"
        ]
        lowered = expression.lower()
        for token in blacklist:
            if token in lowered:
                raise ExpressionParseError(expression, f"Disallowed pattern '{token}' detected in expression")
    
    def _setup_symbols(self) -> Dict[str, Any]:
        """Setup common mathematical symbols and functions"""
        # Import SymPy classes explicitly for the local dict
        from sympy import Symbol, pi, E, I, oo, sin, cos, tan, asin, acos, atan
        from sympy import sinh, cosh, tanh, log, exp, sqrt, Abs, factorial, gamma
        from sympy import floor, ceiling, Integer, Float, Rational
        
        local_dict = {
            # SymPy types for number parsing
            'Integer': Integer,
            'Float': Float,
            'Rational': Rational,
            
            # Common variables
            'x': Symbol('x'),
            'y': Symbol('y'),
            'z': Symbol('z'),
            't': Symbol('t'),
            's': Symbol('s'),
            'n': Symbol('n'),
            'u': Symbol('u'),
            'v': Symbol('v'),
            'w': Symbol('w'),
            'a': Symbol('a'),
            'b': Symbol('b'),
            'c': Symbol('c'),
            'r': Symbol('r'),
            'theta': Symbol('theta'),
            
            # Mathematical constants
            'pi': pi,
            'e': E,
            'E': E,
            'i': I,
            'I': I,
            'oo': oo,  # infinity
            
            # Common functions
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'asin': asin,
            'acos': acos,
            'atan': atan,
            'sinh': sinh,
            'cosh': cosh,
            'tanh': tanh,
            'ln': log,
            'log': log,
            'log10': lambda x: log(x, 10),
            'exp': exp,
            'sqrt': sqrt,
            'abs': Abs,
            'factorial': factorial,
            'gamma': gamma,
            'floor': floor,
            'ceil': ceiling,
            'diff': sp.diff,
            'integrate': sp.integrate,
            'limit': sp.limit,
        }
        return local_dict
    
    def parse_expression(self, expression: str) -> sp.Expr:
        """
        Alias for parse method for testing compatibility
        """
        return self.parse(expression)
    
    def parse(self, expression: str) -> sp.Expr:
        """
        Parse a mathematical expression string into a SymPy expression
        
        Args:
            expression: String representation of mathematical expression
            
        Returns:
            SymPy expression object
            
        Raises:
            ExpressionParseError: If expression cannot be parsed
        """
        try:
            self._validate_expression(expression)

            # Clean the expression
            expression = self._preprocess_expression(expression)
            
            # Parse using SymPy with proper global dict
            expr = parse_expr(
                expression,
                transformations=self.transformations,
                local_dict=self.local_dict,
                global_dict={"__builtins__": None}
            )
            
            return expr
            
        except ExpressionParseError:
            raise
        except Exception as e:
            raise ExpressionParseError(expression, str(e))
    
    def _preprocess_expression(self, expression: str) -> str:
        """
        Preprocess expression string to handle common notation
        """
        # Remove whitespace
        expression = expression.replace(' ', '')

        # Normalize scientific notation: 1e-3 -> 1*10**(-3) so SymPy parses it correctly
        expression = re.sub(r'(?<!\w)(\d*\.?\d+)[eE]([+-]?\d+)', r"\1*10**(\2)", expression)
        
        # Define known functions to avoid breaking them
        functions = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                'exp', 'log', 'log10', 'ln', 'sqrt', 'abs', 'factorial', 'gamma', 'floor', 'ceil',
                'diff', 'integrate', 'limit']
        
        expression = expression.replace('ln(', 'log(')
        
        # Handle implicit multiplication carefully
        # Numbers followed by variables: 2x -> 2*x (but not in scientific notation)
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        
        # Note: We intentionally keep variable names with numeric suffixes (x2, y1) intact
        # so they are treated as distinct symbols rather than implicit multiplication.
        
        # Closing parenthesis followed by variables/numbers/opening parenthesis: (x)y -> (x)*y
        expression = re.sub(r'(\))([a-zA-Z\d\(])', r'\1*\2', expression)
        
        # Variables followed by opening parenthesis, but NOT if it's a function call
        # We need to be more careful here - use a simpler approach
        # First, protect function calls by temporarily replacing them
        protected_funcs = {}
        for i, func in enumerate(functions):
            placeholder = f"__FUNC_{i}__"
            if f"{func}(" in expression:
                expression = expression.replace(f"{func}(", f"{placeholder}(")
                protected_funcs[placeholder] = func
        
        # Now apply the multiplication rule safely
        expression = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expression)
        
        # Restore function calls
        for placeholder, func in protected_funcs.items():
            expression = expression.replace(f"{placeholder}(", f"{func}(")

        # Convert power notation after other preprocessing to avoid touching protected names
        expression = expression.replace('^', '**')
        
        return expression
    
    def validate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Validate an expression and return validation results
        
        Args:
            expression: String representation of mathematical expression
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'expression': None,
            'error': None,
            'variables': [],
            'functions': []
        }
        
        try:
            expr = self.parse(expression)
            result['valid'] = True
            result['expression'] = expr
            result['variables'] = [str(var) for var in expr.free_symbols]
            result['functions'] = [str(func) for func in expr.atoms(sp.Function)]
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_expression_info(self, expr: sp.Expr) -> Dict[str, Any]:
        """
        Get detailed information about a SymPy expression
        """
        return {
            'variables': [str(var) for var in expr.free_symbols],
            'functions': [str(func) for func in expr.atoms(sp.Function)],
            'constants': [str(const) for const in expr.atoms(sp.Number)],
            'complexity': len(expr.args),
            'is_polynomial': expr.is_polynomial(*expr.free_symbols) if expr.free_symbols else False,
            'is_rational': expr.is_rational_function(),
        }

# Global parser instance
expression_parser = ExpressionParser()
