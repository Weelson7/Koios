import sympy as sp
import numpy as np
from functools import lru_cache
from typing import Any, Dict, List, Union, Optional
from core.expression_parser import expression_parser
from utils.exceptions import InvalidInputError, DomainError, UndefinedOperationError
import math


@lru_cache(maxsize=256)
def _cached_parse(expression: str):
    """Cache SymPy parsing for repeated expressions."""
    return expression_parser.parse(expression)

class CalculationEngine:
    """
    Core calculation engine for mathematical operations
    """
    
    def __init__(self):
        self.parser = expression_parser
        self.precision = 15  # Default precision for numerical calculations
    
    def evaluate(self, expression: str, variables: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluate an expression and return a consistent result dictionary.
        """
        if expression is None:
            raise InvalidInputError("expression", "None", "non-empty string")
        if expression == "":
            raise InvalidInputError("expression", "", "non-empty string")

        result = self.evaluate_expression(expression, variables)
        value = result.get('numeric_result') if result.get('numeric_result') is not None else result.get('symbolic_result')
        result['value'] = value
        return result
    
    def evaluate_expression(self, expression: str, variables: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression numerically or symbolically
        
        Args:
            expression: String representation of mathematical expression
            variables: Dictionary of variable values for substitution
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'success': False,
            'symbolic_result': None,
            'numeric_result': None,
            'error': None
        }
        
        try:
            def _to_numeric(val: sp.Expr):
                """Convert SymPy numeric expressions to float or complex."""
                try:
                    if val.is_real:
                        return float(val.evalf(self.precision))
                    if val.is_number:
                        real_part, imag_part = val.as_real_imag()
                        return complex(float(real_part.evalf(self.precision)), float(imag_part.evalf(self.precision)))
                except Exception:
                    return None
                return None

            # Parse the expression
            expr = _cached_parse(expression)
            result['symbolic_result'] = expr
            
            # If variables provided, substitute and evaluate numerically
            if variables:
                substituted_expr = expr.subs(variables)
                if substituted_expr.is_number:
                    numeric_val = _to_numeric(substituted_expr)
                    if numeric_val is not None:
                        result['numeric_result'] = numeric_val
                else:
                    result['symbolic_result'] = substituted_expr
            else:
                # Try to evaluate if no free symbols
                if not expr.free_symbols:
                    numeric_val = _to_numeric(expr)
                    if numeric_val is not None:
                        result['numeric_result'] = numeric_val
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def simplify_expression(self, expression: str) -> Dict[str, Any]:
        """
        Simplify a mathematical expression.
        
        Args:
            expression: String representation of mathematical expression
            
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - original: Original SymPy expression
                - simplified: Simplified SymPy expression
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'original': None,
            'simplified': None,
            'error': None
        }
        
        try:
            expr = _cached_parse(expression)
            simplified = sp.simplify(expr)
            
            result['success'] = True
            result['original'] = expr
            result['simplified'] = simplified
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def expand_expression(self, expression: str) -> Dict[str, Any]:
        """
        Expand a mathematical expression.
        
        Args:
            expression: String representation of mathematical expression
            
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - original: Original SymPy expression
                - expanded: Expanded SymPy expression
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'original': None,
            'expanded': None,
            'error': None
        }
        
        try:
            expr = _cached_parse(expression)
            expanded = sp.expand(expr)
            
            result['success'] = True
            result['original'] = expr
            result['expanded'] = expanded
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def factor_expression(self, expression: str) -> Dict[str, Any]:
        """
        Factor a mathematical expression.
        
        Args:
            expression: String representation of mathematical expression
            
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - original: Original SymPy expression
                - factored: Factored SymPy expression
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'original': None,
            'factored': None,
            'error': None
        }
        
        try:
            expr = _cached_parse(expression)
            factored = sp.factor(expr)
            
            result['success'] = True
            result['original'] = expr
            result['factored'] = factored
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_equation(self, equation: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve an equation for a given variable.
        
        Args:
            equation: Equation string (with or without = sign)
            variable: Variable to solve for (default: 'x')
            
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - equation: Parsed equation expression
                - solutions: List of solution strings
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'equation': None,
            'solutions': [],
            'error': None
        }
        
        try:
            # Handle equation format (with = sign) or expression format
            if '=' in equation:
                left, right = equation.split('=')
                expr = _cached_parse(left) - _cached_parse(right)
            else:
                expr = _cached_parse(equation)
            
            var_symbol = sp.Symbol(variable)
            solutions = sp.solve(expr, var_symbol)
            
            result['success'] = True
            result['equation'] = expr
            result['solutions'] = [str(sol) for sol in solutions]
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def evaluate_at_points(self, expression: str, variable: str, points: List[float]) -> Dict[str, Any]:
        """
        Evaluate expression at multiple points.
        
        Args:
            expression: String representation of mathematical expression
            variable: Variable name to substitute
            points: List of numeric points to evaluate at
            
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - expression: Parsed SymPy expression
                - points: Input points list
                - values: Evaluated values at each point (None if undefined)
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'expression': None,
            'points': [],
            'values': [],
            'error': None
        }
        
        try:
            expr = _cached_parse(expression)
            var_symbol = sp.Symbol(variable)
            
            values = []
            for point in points:
                try:
                    value = float(expr.subs(var_symbol, point).evalf(self.precision))
                    values.append(value)
                except (ValueError, TypeError, ZeroDivisionError, OverflowError, AttributeError) as e:
                    # For undefined points (division by zero, domain errors, etc.)
                    values.append(None)
            
            result['success'] = True
            result['expression'] = expr
            result['points'] = points
            result['values'] = values
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_function_domain(self, expression: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Analyze the domain of a function.
        
        Args:
            expression: String representation of mathematical expression
            variable: Variable to analyze domain for (default: 'x')
            
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - expression: Parsed SymPy expression
                - domain: String description of the domain
                - discontinuities: List of discontinuity points
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'expression': None,
            'domain': None,
            'discontinuities': [],
            'error': None
        }
        
        try:
            expr = _cached_parse(expression)
            var_symbol = sp.Symbol(variable, real=True)
            
            # Find discontinuities
            discontinuities = sp.solve(sp.denom(expr), var_symbol)
            
            result['success'] = True
            result['expression'] = expr
            result['discontinuities'] = [str(d) for d in discontinuities]
            
            # Basic domain analysis (can be extended)
            if not discontinuities:
                result['domain'] = "All real numbers"
            else:
                result['domain'] = f"All real numbers except {discontinuities}"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def signal_processing(self, signal_data: List[float], operation: str, **kwargs) -> Dict[str, Any]:
        """
        Engineering signal processing operations.
        
        Args:
            signal_data: List of signal samples
            operation: Operation type ('fft', 'filter', 'correlation')
            **kwargs: Additional operation-specific parameters:
                - sampling_rate: For FFT operations
                - window_size: For filter operations
                
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - processed_signal: Processed signal data
                - frequency_domain: Frequency domain analysis results
                - filter_response: Filter response data
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'processed_signal': None,
            'frequency_domain': None,
            'filter_response': None,
            'error': None
        }
        
        try:
            signal = np.array(signal_data)
            
            if operation == 'fft':
                # Fast Fourier Transform
                fft_result = np.fft.fft(signal)
                frequencies = np.fft.fftfreq(len(signal), kwargs.get('sampling_rate', 1.0))
                result['frequency_domain'] = {
                    'frequencies': frequencies.tolist(),
                    'magnitude': np.abs(fft_result).tolist(),
                    'phase': np.angle(fft_result).tolist()
                }
                
            elif operation == 'filter':
                # Simple moving average filter
                window_size = kwargs.get('window_size', 5)
                filtered = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
                result['processed_signal'] = filtered.tolist()
                
            elif operation == 'correlation':
                # Auto-correlation
                correlation = np.correlate(signal, signal, mode='full')
                result['processed_signal'] = correlation.tolist()
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def control_systems(self, transfer_function: str, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Control systems engineering calculations.
        
        Args:
            transfer_function: Transfer function string in Laplace domain
            operation: Operation type ('step_response', 'stability')
            **kwargs: Additional operation-specific parameters:
                - time_max: Maximum simulation time
                - points: Number of time points
                
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - step_response: Step response time-domain data
                - impulse_response: Impulse response data
                - bode_plot: Bode plot data
                - stability: Stability analysis results
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'step_response': None,
            'impulse_response': None,
            'bode_plot': None,
            'stability': None,
            'error': None
        }
        
        try:
            # Parse transfer function (simplified)
            s = sp.Symbol('s')
            tf = _cached_parse(transfer_function)
            
            if operation == 'step_response':
                time_points = np.linspace(0, kwargs.get('time_max', 10), kwargs.get('points', 100))
                t = sp.Symbol('t', real=True, positive=True)

                try:
                    step_tf = tf / s  # Multiply by 1/s for unit step input
                    time_response = sp.inverse_laplace_transform(step_tf, s, t)
                    step_func = sp.lambdify(t, time_response, modules=['numpy'])
                    amplitudes = np.real(step_func(time_points))

                    result['step_response'] = {
                        'time': time_points.tolist(),
                        'amplitude': amplitudes.tolist(),
                        'response_expression': str(time_response)
                    }
                except Exception as transform_err:
                    result['error'] = f"Step response calculation failed: {transform_err}"
                    return result
                
            elif operation == 'stability':
                # Check poles (denominator roots)
                denominator = sp.denom(tf)
                poles = sp.solve(denominator, s)
                stable = all(sp.re(pole) < 0 for pole in poles if pole.is_number)
                result['stability'] = {'stable': stable, 'poles': [str(p) for p in poles]}
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

    def clear_cached_parses(self) -> None:
        """Clear cached expression parses when parser configuration changes."""
        _cached_parse.cache_clear()
    
    def structural_analysis(self, beam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic structural engineering calculations.
        
        Args:
            beam_data: Dictionary containing beam parameters:
                - length: Beam length in meters
                - load: Applied load in Newtons
                - elastic_modulus: Young's modulus in Pa
                - moment_of_inertia: Second moment of area in m^4
                
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - deflection: Deflection profile data
                - stress: Stress distribution
                - moment: Bending moment data
                - shear: Shear force data
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'deflection': None,
            'stress': None,
            'moment': None,
            'shear': None,
            'error': None
        }
        
        try:
            # Extract beam parameters
            length = beam_data.get('length', 1.0)  # meters
            load = beam_data.get('load', 1000)  # Newtons
            E = beam_data.get('elastic_modulus', 200e9)  # Pa (steel)
            I = beam_data.get('moment_of_inertia', 1e-6)  # m^4
            
            # Validate parameters
            if length <= 0 or E <= 0 or I <= 0:
                result['error'] = "Invalid parameters: length, E, and I must be positive"
                return result
            
            # Simply supported beam with point load at center
            x = np.linspace(0, length, 100)
            
            # Maximum deflection at center: δ = PL³/(48EI)
            max_deflection = (load * length**3) / (48 * E * I)
            
            # Deflection along beam
            deflection = []
            for xi in x:
                if xi <= length/2:
                    delta = (load * xi) / (48 * E * I) * (3 * length**2 - 4 * xi**2)
                else:
                    delta = (load * (length - xi)) / (48 * E * I) * (3 * length**2 - 4 * (length - xi)**2)
                deflection.append(delta)
            
            # Maximum moment at center: M = PL/4
            max_moment = load * length / 4
            
            result['success'] = True
            result['deflection'] = {
                'position': x.tolist(),
                'deflection': deflection,
                'max_deflection': max_deflection
            }
            result['moment'] = {'max_moment': max_moment}
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def thermodynamics(self, process_type: str, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thermodynamic process calculations.
        
        Args:
            process_type: Type of process ('isothermal', 'adiabatic', 'isobaric', 'isochoric')
            state_data: Dictionary containing initial state parameters:
                - pressure1: Initial pressure in Pa
                - volume1: Initial volume in m^3
                - temperature1: Initial temperature in K
                - volume2 or temperature2: Final state parameter
                
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - final_state: Final thermodynamic state
                - work_done: Work done by/on system in Joules
                - heat_transfer: Heat transfer amount
                - efficiency: Process efficiency
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'final_state': None,
            'work_done': None,
            'heat_transfer': None,
            'efficiency': None,
            'error': None
        }
        
        try:
            # Extract initial state
            P1 = state_data.get('pressure1', 101325)  # Pa
            V1 = state_data.get('volume1', 0.001)  # m³
            T1 = state_data.get('temperature1', 300)  # K
            
            # Gas constant (air)
            R = 287  # J/(kg·K)
            gamma = 1.4  # Heat capacity ratio
            
            if process_type == 'isothermal':
                # PV = constant
                V2 = state_data.get('volume2', 0.002)
                P2 = P1 * V1 / V2
                T2 = T1
                work_done = P1 * V1 * np.log(V2 / V1)
                
            elif process_type == 'adiabatic':
                # PV^γ = constant
                if gamma == 1:
                    raise InvalidInputError("gamma", str(gamma), "value not equal to 1")
                V2 = state_data.get('volume2', 0.002)
                P2 = P1 * (V1 / V2) ** gamma
                T2 = T1 * (V1 / V2) ** (gamma - 1)
                work_done = (P1 * V1 - P2 * V2) / (gamma - 1)
                
            elif process_type == 'isobaric':
                # P = constant
                V2 = state_data.get('volume2', 0.002)
                P2 = P1
                T2 = T1 * V2 / V1
                work_done = P1 * (V2 - V1)
                
            elif process_type == 'isochoric':
                # V = constant
                T2 = state_data.get('temperature2', 400)
                V2 = V1
                P2 = P1 * T2 / T1
                work_done = 0
                
            result['success'] = True
            result['final_state'] = {'pressure': P2, 'volume': V2, 'temperature': T2}
            result['work_done'] = work_done
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def electrical_circuits(self, circuit_type: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Electrical circuit analysis.
        
        Args:
            circuit_type: Type of circuit ('rlc_series', etc.)
            components: Dictionary containing circuit parameters:
                - resistance: Resistance in Ohms
                - inductance: Inductance in Henry
                - capacitance: Capacitance in Farad
                - frequency: Operating frequency in Hz
                - voltage: Applied voltage in Volts
                
        Returns:
            Dictionary with keys:
                - success: Boolean indicating operation success
                - impedance: Complex impedance data
                - current: Current magnitude and phase
                - power: Power dissipation
                - frequency_response: Frequency response data
                - error: Error message if operation failed
        """
        result = {
            'success': False,
            'impedance': None,
            'current': None,
            'power': None,
            'frequency_response': None,
            'error': None
        }
        
        try:
            if circuit_type == 'rlc_series':
                R = components.get('resistance', 100)  # Ohms
                L = components.get('inductance', 0.001)  # Henry
                C = components.get('capacitance', 1e-6)  # Farad
                frequency = components.get('frequency', 1000)  # Hz
                voltage = components.get('voltage', 10)  # Volts
                
                omega = 2 * np.pi * frequency
                
                # Complex impedance
                Z_R = R
                Z_L = 1j * omega * L
                Z_C = 1 / (1j * omega * C)
                Z_total = Z_R + Z_L + Z_C
                
                # Current
                current = voltage / Z_total
                
                # Power
                power = np.real(voltage * np.conj(current))
                
                # Resonant frequency
                f_resonant = 1 / (2 * np.pi * np.sqrt(L * C))
                
                result['impedance'] = {
                    'magnitude': abs(Z_total),
                    'phase': np.angle(Z_total),
                    'real': np.real(Z_total),
                    'imaginary': np.imag(Z_total)
                }
                result['current'] = {
                    'magnitude': abs(current),
                    'phase': np.angle(current)
                }
                result['power'] = power
                result['resonant_frequency'] = f_resonant
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # Wrapper methods for compatibility with UI/tests
    def simplify(self, expression: str) -> Any:
        """
        Wrapper for simplify_expression for compatibility.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Simplified SymPy expression or None
            
        Raises:
            InvalidInputError: If expression cannot be simplified
        """
        result = self.simplify_expression(expression)
        if result['success']:
            return result['simplified']
        if result.get('error'):
            raise InvalidInputError("expression", expression, "valid mathematical expression")
        return None
    
    def expand(self, expression: str) -> Any:
        """
        Wrapper for expand_expression for compatibility.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Expanded SymPy expression or None
            
        Raises:
            InvalidInputError: If expression cannot be expanded
        """
        result = self.expand_expression(expression)
        if result['success']:
            return result['expanded']
        if result.get('error'):
            raise InvalidInputError("expression", expression, "valid mathematical expression")
        return None
    
    def factor(self, expression: str) -> Any:
        """
        Wrapper for factor_expression for compatibility.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Factored SymPy expression or None
            
        Raises:
            InvalidInputError: If expression cannot be factored
        """
        result = self.factor_expression(expression)
        if result['success']:
            return result['factored']
        if result.get('error'):
            raise InvalidInputError("expression", expression, "valid mathematical expression")
        return None
    
    def evaluate_with_variables(self, expression: str, variables: Dict[str, float]) -> Any:
        """
        Wrapper for evaluate with variables for compatibility.
        
        Args:
            expression: Mathematical expression string
            variables: Dictionary mapping variable names to numeric values
            
        Returns:
            Evaluation result dictionary
        """
        return self.evaluate(expression, variables)

# Global calculation engine instance
calculation_engine = CalculationEngine()
