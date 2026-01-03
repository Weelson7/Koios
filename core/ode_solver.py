import numpy as np
import sympy as sp
from scipy.integrate import odeint, solve_ivp
from typing import Dict, Any, List, Callable, Optional, Union
from core.expression_parser import expression_parser
from utils.exceptions import ExpressionParseError, ConfigurationError, InvalidInputError

class ODESolver:
    """
    Differential equation solver supporting various numerical methods
    """
    
    def __init__(self):
        self.parser = expression_parser
    
    def parse_ode(self, ode_string: str, dependent_var: str = 'y', 
                  independent_var: str = 'x') -> Dict[str, Any]:
        """
        Parse ODE string into SymPy expression
        
        Args:
            ode_string: String representation of ODE (e.g., "y' + 2*y = 0")
            dependent_var: Dependent variable (usually y)
            independent_var: Independent variable (usually x or t)
            
        Returns:
            Dictionary with parsed ODE information
            
        Raises:
            ValueError: If variable names conflict or are invalid
        """
        result = {
            'success': False,
            'ode_expr': None,
            'order': None,
            'error': None
        }
        
        try:
            # Validate variable names - ensure they don't conflict
            if dependent_var == independent_var:
                raise InvalidInputError(
                    "ODE variables",
                    f"dependent='{dependent_var}', independent='{independent_var}'",
                    "different variable names"
                )
            if not dependent_var.isidentifier():
                raise InvalidInputError(
                    "dependent variable",
                    f"'{dependent_var}'",
                    "valid Python identifier"
                )
            if not independent_var.isidentifier():
                raise InvalidInputError(
                    "independent variable",
                    f"'{independent_var}'",
                    "valid Python identifier"
                )
            
            # Define symbols
            x = sp.Symbol(independent_var)
            y = sp.Function(dependent_var)
            
            # Replace derivative notation and prepare for SymPy parsing
            # Use regex to avoid double replacement issues
            import re
            
            # Check if string already contains "Derivative" to avoid double replacement
            if "Derivative" not in ode_string:
                # Replace higher-order derivatives first (more specific patterns first)
                ode_string = re.sub(r"\by''\b", f"Derivative({dependent_var}({independent_var}), {independent_var}, 2)", ode_string)
                ode_string = re.sub(r"\by'\b", f"Derivative({dependent_var}({independent_var}), {independent_var})", ode_string)
            # Replace standalone y with function notation (but not y inside Derivative or other functions)
            ode_string = re.sub(r"\by\b(?!\s*\(|\s*,|\s*\))", f"{dependent_var}({independent_var})", ode_string)
            
            # Parse the ODE using SymPy directly
            if '=' in ode_string:
                parts = ode_string.split('=')
                if len(parts) != 2:
                    raise InvalidInputError("ode_string", ode_string, 
                        "Single equation with format 'expr1=expr2' (must contain exactly one '=' sign)")
                left, right = parts
                # Use SymPy's sympify for parsing
                left_expr = sp.sympify(left, locals={'x': x, 'y': y})
                right_expr = sp.sympify(right, locals={'x': x, 'y': y})
                ode_expr = left_expr - right_expr
            else:
                ode_expr = sp.sympify(ode_string, locals={'x': x, 'y': y})
            
            # Determine order by inspecting Derivative atoms
            order = 1
            derivatives = list(ode_expr.atoms(sp.Derivative))
            if derivatives:
                # Use SymPy's built-in derivative_count property
                orders = [deriv.derivative_count for deriv in derivatives]
                order = max(orders) if orders else 1
            
            result['success'] = True
            result['ode_expr'] = ode_expr
            result['order'] = order
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_symbolic_ode(self, ode_string: str, dependent_var: str = 'y', 
                          independent_var: str = 'x') -> Dict[str, Any]:
        """
        Solve ODE symbolically using SymPy
        """
        result = {
            'success': False,
            'ode_expr': None,
            'solution': None,
            'error': None
        }
        
        try:
            # Define symbols and function
            x = sp.Symbol(independent_var)
            y = sp.Function(dependent_var)
            
            # Parse and solve
            parsed_result = self.parse_ode(ode_string, dependent_var, independent_var)
            if not parsed_result['success']:
                result['error'] = parsed_result['error']
                return result
            
            # Try to solve symbolically
            ode_expr = parsed_result['ode_expr']
            solutions = sp.dsolve(ode_expr, y(x))
            
            result['success'] = True
            result['ode_expr'] = ode_expr
            result['solution'] = solutions
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def euler_method(self, func: Callable, y0: float, x_span: tuple, 
                    num_points: int = 100) -> Dict[str, Any]:
        """
        Solve ODE using Euler's method
        
        Args:
            func: Function representing dy/dx = f(x, y)
            y0: Initial condition
            x_span: Tuple (x_start, x_end)
            num_points: Number of points to compute
            
        Returns:
            Dictionary with solution results
        """
        result = {
            'success': False,
            'x_values': None,
            'y_values': None,
            'method': 'Euler',
            'error': None
        }
        
        try:
            x_start, x_end = x_span
            h = (x_end - x_start) / (num_points - 1)
            
            x_values = np.linspace(x_start, x_end, num_points)
            y_values = np.zeros(num_points)
            y_values[0] = y0
            
            # Euler's method iteration
            for i in range(1, num_points):
                y_values[i] = y_values[i-1] + h * func(x_values[i-1], y_values[i-1])
            
            result['success'] = True
            result['x_values'] = x_values.tolist()
            result['y_values'] = y_values.tolist()
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def runge_kutta_4(self, func: Callable, y0: float, x_span: tuple, 
                     num_points: int = 100) -> Dict[str, Any]:
        """
        Solve ODE using 4th order Runge-Kutta method
        """
        result = {
            'success': False,
            'x_values': None,
            'y_values': None,
            'method': 'Runge-Kutta 4',
            'error': None
        }
        
        try:
            x_start, x_end = x_span
            h = (x_end - x_start) / (num_points - 1)
            
            x_values = np.linspace(x_start, x_end, num_points)
            y_values = np.zeros(num_points)
            y_values[0] = y0
            
            # RK4 method iteration
            for i in range(1, num_points):
                x_i = x_values[i-1]
                y_i = y_values[i-1]
                
                k1 = h * func(x_i, y_i)
                k2 = h * func(x_i + h/2, y_i + k1/2)
                k3 = h * func(x_i + h/2, y_i + k2/2)
                k4 = h * func(x_i + h, y_i + k3)
                
                y_values[i] = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            result['success'] = True
            result['x_values'] = x_values.tolist()
            result['y_values'] = y_values.tolist()
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_ivp_wrapper(self, func: Callable, y0: Union[float, List[float]], 
                         x_span: tuple, method: str = 'RK45', 
                         num_points: int = 100) -> Dict[str, Any]:
        """
        Wrapper for scipy.integrate.solve_ivp for advanced ODE solving
        
        Returns:
            Dictionary with:
            - x_values: Time points (1D array)
            - y_values: Solution values (2D array for multi-dimensional, 1D for single)
            Note: For consistency, single-variable results are stored as 1D arrays
        """
        result = {
            'success': False,
            'x_values': None,
            'y_values': None,
            'method': method,
            'error': None
        }
        
        try:
            # Ensure y0 is array-like and track original dimensionality
            is_scalar = isinstance(y0, (int, float))
            if is_scalar:
                y0 = [y0]
            
            # Create evaluation points
            x_eval = np.linspace(x_span[0], x_span[1], num_points)
            
            # Solve using scipy
            sol = solve_ivp(func, x_span, y0, t_eval=x_eval, method=method)
            
            if sol.success:
                result['success'] = True
                result['x_values'] = sol.t.tolist()
                # For consistency: return 1D array for single variable, 2D for multiple
                if is_scalar:
                    result['y_values'] = sol.y[0].tolist()
                else:
                    result['y_values'] = sol.y.tolist()
            else:
                result['error'] = "scipy solve_ivp failed"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def create_ode_function(self, expression: str, dependent_var: str = 'y', 
                           independent_var: str = 'x') -> Dict[str, Any]:
        """
        Create a callable function from ODE expression string
        """
        result = {
            'success': False,
            'function': None,
            'error': None
        }
        
        try:
            # Parse expression
            expr = self.parser.parse(expression)
            
            # Create symbols
            x_sym = sp.Symbol(independent_var)
            y_sym = sp.Symbol(dependent_var)
            
            # Convert to lambda function
            func = sp.lambdify((x_sym, y_sym), expr, 'numpy')
            
            result['success'] = True
            result['function'] = func
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_system_odes(self, expressions: List[str], initial_conditions: List[float],
                         x_span: tuple, variables: List[str], 
                         method: str = 'RK45') -> Dict[str, Any]:
        """
        Solve system of ODEs
        
        Args:
            expressions: List of ODE expressions for each variable
            initial_conditions: Initial values for each variable
            x_span: Tuple (x_start, x_end) for integration domain
            variables: List of variable names corresponding to expressions
            method: Integration method (e.g., 'RK45')
            
        Returns:
            Dictionary with solution arrays
            
        Raises:
            ValueError: If lengths don't match or inputs are invalid
        """
        result = {
            'success': False,
            'x_values': None,
            'y_values': None,
            'method': method,
            'error': None
        }
        
        try:
            # Validate input consistency
            if len(expressions) != len(variables):
                raise ConfigurationError(
                    "ODE system",
                    f"{len(expressions)} expressions vs {len(variables)} variables",
                    "expressions count must equal variables count"
                )
            if len(initial_conditions) != len(variables):
                raise ConfigurationError(
                    "ODE system",
                    f"{len(initial_conditions)} initial conditions vs {len(variables)} variables",
                    "initial conditions count must equal variables count"
                )
            if len(expressions) == 0:
                raise ConfigurationError(
                    "ODE system",
                    "empty system",
                    "at least one equation required"
                )
            
            # Parse all expressions once before creating system function (optimization)
            parsed_exprs = []
            for expr_str in expressions:
                try:
                    expr = self.parser.parse(expr_str)
                    parsed_exprs.append(expr)
                except Exception as e:
                    raise ExpressionParseError(expr_str, str(e))
            
            # Create system function using pre-parsed expressions
            def system_func(x, y):
                dydt = []
                for i, expr in enumerate(parsed_exprs):
                    # Create substitution dictionary
                    subs_dict = {'x': x}
                    for j, var in enumerate(variables):
                        subs_dict[var] = y[j]
                    
                    # Evaluate pre-parsed expression
                    try:
                        value = float(expr.subs(subs_dict))
                        dydt.append(value)
                    except Exception as e:
                        raise ValueError(f"Failed to evaluate expression {i} at x={x}: {str(e)}")
                
                return dydt
            
            # Solve system
            sol = solve_ivp(system_func, x_span, initial_conditions, method=method)
            
            if sol.success:
                result['success'] = True
                result['x_values'] = sol.t.tolist()
                result['y_values'] = sol.y.tolist()
            else:
                result['error'] = "Failed to solve ODE system"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

# Global ODE solver instance
ode_solver = ODESolver()
