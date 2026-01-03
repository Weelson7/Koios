import sympy as sp
import numpy as np
from functools import lru_cache
from typing import Any, Dict, List, Union, Optional
from core.expression_parser import expression_parser
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
    
    def evaluate(self, expression: str, variables: Optional[Dict[str, float]] = None) -> Any:
        """
        Simple evaluate method for testing compatibility
        """
        # Validate input
        if expression is None:
            raise ValueError("Expression cannot be None")
        if expression == "":
            raise ValueError("Expression cannot be empty")
            
        try:
            result = self.evaluate_expression(expression, variables)
            if result['success']:
                # Prefer numeric result when available, even if it is 0.0
                if result.get('numeric_result') is not None:
                    return result['numeric_result']
                return result.get('symbolic_result')
            if result.get('error'):
                raise ValueError(result['error'])
            return None
        except Exception as e:
            raise e
    
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
        Simplify a mathematical expression
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
        Expand a mathematical expression
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
        Factor a mathematical expression
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
        Solve an equation for a given variable
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
        Evaluate expression at multiple points
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
                except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
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
        Analyze the domain of a function
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
        Engineering signal processing operations
        """
        result = {
            'success': False,
            'processed_signal': None,
            'frequency_domain': None,
            'filter_response': None,
            'error': None
        }
        
        try:
            import numpy as np
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
        Control systems engineering calculations
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
            tf = _cached_parse(transfer_function.replace('s', str(s)))
            
            if operation == 'step_response':
                # Simplified step response calculation
                time_points = np.linspace(0, kwargs.get('time_max', 10), kwargs.get('points', 100))
                # This is a simplified implementation
                result['step_response'] = {
                    'time': time_points.tolist(),
                    'amplitude': np.exp(-time_points).tolist()  # Example exponential response
                }
                
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
    
    def structural_analysis(self, beam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic structural engineering calculations
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
        Thermodynamic process calculations
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
        Electrical circuit analysis
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
        """Wrapper for simplify_expression for compatibility"""
        result = self.simplify_expression(expression)
        if result['success']:
            return result['simplified']
        if result.get('error'):
            raise ValueError(result['error'])
        return None
    
    def expand(self, expression: str) -> Any:
        """Wrapper for expand_expression for compatibility"""
        result = self.expand_expression(expression)
        if result['success']:
            return result['expanded']
        if result.get('error'):
            raise ValueError(result['error'])
        return None
    
    def factor(self, expression: str) -> Any:
        """Wrapper for factor_expression for compatibility"""
        result = self.factor_expression(expression)
        if result['success']:
            return result['factored']
        if result.get('error'):
            raise ValueError(result['error'])
        return None
    
    def evaluate_with_variables(self, expression: str, variables: Dict[str, float]) -> Any:
        """Wrapper for evaluate with variables for compatibility"""
        return self.evaluate(expression, variables)

# Global calculation engine instance
calculation_engine = CalculationEngine()
