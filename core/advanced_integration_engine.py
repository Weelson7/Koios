import sympy as sp
from sympy import symbols, integrate, simplify, expand, factor, apart, trigsimp
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

class IntegrationMethod(Enum):
    """Available integration methods"""
    DIRECT = "DIRECT"
    SUBSTITUTION = "SUBSTITUTION"
    BY_PARTS = "BY_PARTS"
    PARTIAL_FRACTIONS = "PARTIAL_FRACTIONS"
    TRIG_SUBSTITUTION = "TRIG_SUBSTITUTION"
    SPECIAL_FUNCTIONS = "SPECIAL_FUNCTIONS"

@dataclass
class IntegrationResult:
    """Result of integration operation"""
    success: bool
    integral: Optional[sp.Expr]
    method_used: IntegrationMethod
    steps: List[str]
    error: Optional[str] = None
    
class AdvancedIntegrationEngine:
    """Advanced integration calculation engine"""
    
    def __init__(self):
        self.x = symbols('x')
        self.integration_rules = self._build_integration_rules()
        
    def _build_integration_rules(self) -> Dict[str, Any]:
        """Build database of integration patterns and methods"""
        return {
            'substitution_patterns': [
                # u-substitution patterns
                {'pattern': 'f(g(x))*g\'(x)', 'method': 'u_substitution'},
                {'pattern': 'sqrt(a²-x²)', 'substitution': 'x = a*sin(θ)'},
                {'pattern': 'sqrt(a²+x²)', 'substitution': 'x = a*tan(θ)'},
                {'pattern': 'sqrt(x²-a²)', 'substitution': 'x = a*sec(θ)'},
            ],
            'by_parts_patterns': [
                # Integration by parts patterns (LIATE rule)
                {'pattern': 'ln(x)*p(x)', 'u': 'ln(x)', 'dv': 'p(x)dx'},
                {'pattern': 'arctan(x)*p(x)', 'u': 'arctan(x)', 'dv': 'p(x)dx'},
                {'pattern': 'x^n*e^x', 'u': 'x^n', 'dv': 'e^x dx'},
                {'pattern': 'x^n*sin(x)', 'u': 'x^n', 'dv': 'sin(x)dx'},
            ]
        }
    
    def integrate_expression(self, expression: str, variable: str = 'x',
                           method: Optional[IntegrationMethod] = None) -> IntegrationResult:
        """
        Integrate an expression using advanced techniques
        
        Args:
            expression: Mathematical expression to integrate
            variable: Integration variable
            method: Specific method to use (optional)
            
        Returns:
            IntegrationResult with integral and steps
        """
        try:
            # Parse expression
            var = symbols(variable)
            expr = sp.sympify(expression)
            steps = [f"Original expression: ∫ {expr} d{variable}"]
            
            # Try direct integration first
            if method is None or method == IntegrationMethod.DIRECT:
                try:
                    result = integrate(expr, var)
                    if result.has(sp.Integral):
                        # Integration failed, try other methods
                        pass
                    else:
                        steps.append(f"Direct integration: {result} + C")
                        return IntegrationResult(
                            success=True,
                            integral=result,
                            method_used=IntegrationMethod.DIRECT,
                            steps=steps
                        )
                except:
                    pass
            
            # Try substitution method
            if method is None or method == IntegrationMethod.SUBSTITUTION:
                sub_result = self._try_substitution(expr, var, steps)
                if sub_result:
                    return sub_result
            
            # Try integration by parts
            if method is None or method == IntegrationMethod.BY_PARTS:
                parts_result = self._try_by_parts(expr, var, steps)
                if parts_result:
                    return parts_result
            
            # Try partial fractions
            if method is None or method == IntegrationMethod.PARTIAL_FRACTIONS:
                pf_result = self._try_partial_fractions(expr, var, steps)
                if pf_result:
                    return pf_result
            
            # Try trigonometric substitution
            if method is None or method == IntegrationMethod.TRIG_SUBSTITUTION:
                trig_result = self._try_trig_substitution(expr, var, steps)
                if trig_result:
                    return trig_result
            
            # If all methods fail, return symbolic integral
            result = sp.Integral(expr, var)
            steps.append("No elementary antiderivative found")
            return IntegrationResult(
                success=False,
                integral=result,
                method_used=IntegrationMethod.DIRECT,
                steps=steps,
                error="Could not find elementary antiderivative"
            )
            
        except Exception as e:
            return IntegrationResult(
                success=False,
                integral=None,
                method_used=IntegrationMethod.DIRECT,
                steps=[],
                error=str(e)
            )
    
    def _try_substitution(self, expr: sp.Expr, var: sp.Symbol, 
                         steps: List[str]) -> Optional[IntegrationResult]:
        """Try u-substitution method"""
        # Check for composite functions
        if expr.has(sp.exp):
            # Try substitution for exponential
            u = expr.as_independent(sp.exp)[1]
            if u != expr:
                steps.append(f"Trying substitution: u = {u}")
                du = sp.diff(u, var)
                if expr.has(du):
                    # Valid substitution found
                    new_expr = expr.subs(u, sp.Symbol('u')) / du
                    result = integrate(new_expr, sp.Symbol('u'))
                    final = result.subs(sp.Symbol('u'), u)
                    steps.append(f"After substitution: ∫ {new_expr} du = {result}")
                    steps.append(f"Back-substitution: {final} + C")
                    return IntegrationResult(
                        success=True,
                        integral=final,
                        method_used=IntegrationMethod.SUBSTITUTION,
                        steps=steps
                    )
        
        # Check for sqrt patterns
        if expr.has(sp.sqrt):
            # Implement trigonometric substitutions
            pass
            
        return None
    
    def _try_by_parts(self, expr: sp.Expr, var: sp.Symbol,
                     steps: List[str]) -> Optional[IntegrationResult]:
        """Try integration by parts: ∫u dv = uv - ∫v du"""
        # LIATE rule: Log, Inverse trig, Algebraic, Trig, Exponential
        
        if expr.is_Mul:
            factors = expr.as_ordered_factors()
            
            # Check for logarithmic * polynomial
            for i, factor in enumerate(factors):
                if factor.has(sp.log):
                    u = factor
                    dv = expr / u
                    du = sp.diff(u, var)
                    v = integrate(dv, var)
                    
                    if not v.has(sp.Integral):
                        result = u * v - integrate(v * du, var)
                        steps.append(f"Integration by parts: u = {u}, dv = {dv}dx")
                        steps.append(f"du = {du}dx, v = {v}")
                        steps.append(f"Result: {u}·{v} - ∫{v}·{du}dx = {result} + C")
                        return IntegrationResult(
                            success=True,
                            integral=result,
                            method_used=IntegrationMethod.BY_PARTS,
                            steps=steps
                        )
        
        return None
    
    def _try_partial_fractions(self, expr: sp.Expr, var: sp.Symbol,
                              steps: List[str]) -> Optional[IntegrationResult]:
        """Try partial fractions decomposition"""
        if expr.is_rational_function(var):
            pf = apart(expr, var)
            if pf != expr:
                steps.append(f"Partial fractions: {expr} = {pf}")
                result = integrate(pf, var)
                steps.append(f"Integration: {result} + C")
                return IntegrationResult(
                    success=True,
                    integral=result,
                    method_used=IntegrationMethod.PARTIAL_FRACTIONS,
                    steps=steps
                )
        return None
    
    def _try_trig_substitution(self, expr: sp.Expr, var: sp.Symbol,
                              steps: List[str]) -> Optional[IntegrationResult]:
        """Try trigonometric substitution"""
        # Patterns: sqrt(a²-x²), sqrt(a²+x²), sqrt(x²-a²)
        
        if expr.has(sp.sqrt):
            arg = None
            for term in expr.atoms(sp.sqrt):
                arg = term.args[0]
                
                # Check for a²-x² pattern
                if arg.is_Add and len(arg.args) == 2:
                    const_term = None
                    var_term = None
                    
                    for term in arg.args:
                        if term.has(var):
                            var_term = term
                        else:
                            const_term = term
                    
                    if const_term and var_term and var_term.is_Pow and var_term.exp == 2:
                        # Found pattern, apply substitution
                        a = sp.sqrt(abs(const_term))
                        if const_term > 0 and var_term.coeff(var**2) < 0:
                            # sqrt(a²-x²): x = a*sin(θ)
                            theta = sp.Symbol('theta')
                            sub = a * sp.sin(theta)
                            steps.append(f"Trigonometric substitution: {var} = {sub}")
                            # Continue with substitution...
                            
        return None
    
    def evaluate_improper_integral(self, expression: str, variable: str,
                                  lower_limit: Union[str, float], 
                                  upper_limit: Union[str, float]) -> Dict[str, Any]:
        """
        Evaluate improper integrals (infinite limits or discontinuities)
        
        Args:
            expression: Expression to integrate
            variable: Integration variable
            lower_limit: Lower limit (can be -oo)
            upper_limit: Upper limit (can be oo)
            
        Returns:
            Dictionary with result and convergence information
        """
        try:
            var = symbols(variable)
            expr = sp.sympify(expression)
            
            # Convert limits
            a = sp.sympify(lower_limit)
            b = sp.sympify(upper_limit)
            
            # Evaluate the integral
            result = integrate(expr, (var, a, b))
            
            # Check convergence
            converges = not (result.has(sp.oo) or result.has(sp.nan) or 
                           result.has(sp.zoo) or result == sp.oo)
            
            return {
                'success': True,
                'result': str(result),
                'converges': converges,
                'numeric_value': float(result) if converges and result.is_number else None,
                'steps': [
                    f"Improper integral: ∫[{a},{b}] {expr} d{variable}",
                    f"Result: {result}",
                    f"Convergence: {'Yes' if converges else 'No'}"
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'converges': False
            }
    
    def integration_techniques_catalog(self) -> Dict[str, List[str]]:
        """Return catalog of integration techniques with examples"""
        return {
            'basic_integrals': [
                "∫ x^n dx = x^(n+1)/(n+1) + C, n ≠ -1",
                "∫ 1/x dx = ln|x| + C",
                "∫ e^x dx = e^x + C",
                "∫ sin(x) dx = -cos(x) + C",
                "∫ cos(x) dx = sin(x) + C"
            ],
            'substitution': [
                "∫ f(g(x))g'(x) dx: Let u = g(x)",
                "∫ sin(3x) dx: u = 3x, du = 3dx",
                "∫ x·e^(x²) dx: u = x², du = 2x dx"
            ],
            'by_parts': [
                "∫ u dv = uv - ∫ v du",
                "∫ x·ln(x) dx: u = ln(x), dv = x dx",
                "∫ x·e^x dx: u = x, dv = e^x dx"
            ],
            'trig_substitution': [
                "√(a²-x²): x = a·sin(θ)",
                "√(a²+x²): x = a·tan(θ)",
                "√(x²-a²): x = a·sec(θ)"
            ],
            'partial_fractions': [
                "∫ 1/((x-a)(x-b)) dx",
                "∫ (2x+3)/((x²+1)(x-1)) dx"
            ]
        }