import sympy as sp
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from functools import lru_cache
from core.expression_parser import expression_parser
from utils.exceptions import InvalidInputError, UndefinedOperationError


class TransformsSeriesEngine:
    """
    Comprehensive transforms and series engine for Laplace, Fourier transforms,
    transfer functions, and series analysis
    """
    
    def __init__(self):
        self.parser = expression_parser
        
    def laplace_transform(self, expression: str, t_var: str = 't', s_var: str = 's') -> Dict[str, Any]:
        """
        Compute Laplace transform of an expression
        
        Args:
            expression: Mathematical expression as string (function of time)
            t_var: Time variable (default: 't')
            s_var: Laplace variable (default: 's')
            
        Returns:
            Dictionary with Laplace transform results
        """
        result = {
            'success': False,
            'original_expression': None,
            'transform': None,
            'error': None
        }
        
        try:
            expr = self.parser.parse(expression)
            t = sp.Symbol(t_var, real=True, positive=True)
            s = sp.Symbol(s_var)
            
            # Substitute the correct time variable if parsing introduced different variable
            # Only substitute if we have a single free symbol OR if the target variable isn't present
            expr_vars = expr.free_symbols
            if len(expr_vars) == 1:
                old_var = list(expr_vars)[0]
                if str(old_var) != t_var:
                    expr = expr.subs(old_var, t)
            elif t not in expr_vars and expr_vars:
                # Multiple symbols but t is not one of them
                # Try to find a likely time variable (single letter, not a known constant)
                likely_t_vars = [v for v in expr_vars if len(str(v)) == 1 and str(v) not in ['I', 'E', 'a', 'b', 'c', 'k', 'n', 'm']]
                if len(likely_t_vars) == 1:
                    expr = expr.subs(likely_t_vars[0], t)
            
            # Compute Laplace transform
            transform = sp.laplace_transform(expr, t, s, noconds=True)
            
            result['success'] = True
            result['original_expression'] = expr
            result['transform'] = sp.simplify(transform)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def inverse_laplace_transform(self, expression: str, s_var: str = 's', t_var: str = 't') -> Dict[str, Any]:
        """
        Compute inverse Laplace transform
        
        Args:
            expression: Laplace domain expression as string
            s_var: Laplace variable (default: 's')
            t_var: Time variable (default: 't')
            
        Returns:
            Dictionary with inverse Laplace transform results
        """
        result = {
            'success': False,
            'original_expression': None,
            'transform': None,
            'error': None
        }
        
        try:
            expr = self.parser.parse(expression)
            s = sp.Symbol(s_var)
            t = sp.Symbol(t_var, real=True, positive=True)
            
            # Substitute the correct Laplace variable if parsing introduced different variable
            expr_vars = expr.free_symbols
            if len(expr_vars) == 1:
                old_var = list(expr_vars)[0]
                if str(old_var) not in ['I', 'E', 'pi'] and str(old_var) != s_var:
                    expr = expr.subs(old_var, s)
            elif s not in expr_vars and expr_vars:
                # Multiple symbols but s is not one of them
                # Try to find a likely Laplace variable
                likely_s_vars = [v for v in expr_vars if len(str(v)) == 1 and str(v) not in ['I', 'E', 'a', 'b', 'c', 'k', 'n', 'm', 't']]
                if len(likely_s_vars) == 1:
                    expr = expr.subs(likely_s_vars[0], s)
            
            # Compute inverse Laplace transform
            transform = sp.inverse_laplace_transform(expr, s, t)
            
            result['success'] = True
            result['original_expression'] = expr
            result['transform'] = sp.simplify(transform)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def fourier_transform(self, expression: str, t_var: str = 't', omega_var: str = 'omega') -> Dict[str, Any]:
        """
        Compute Fourier transform of an expression
        
        Args:
            expression: Mathematical expression as string (function of time)
            t_var: Time variable (default: 't')
            omega_var: Frequency variable (default: 'omega')
            
        Returns:
            Dictionary with Fourier transform results
        """
        result = {
            'success': False,
            'original_expression': None,
            'transform': None,
            'error': None
        }
        
        try:
            expr = self.parser.parse(expression)
            t = sp.Symbol(t_var, real=True)
            omega = sp.Symbol(omega_var, real=True)
            
            # Substitute the correct time variable if parsing introduced different variable
            expr_vars = expr.free_symbols
            if len(expr_vars) == 1:
                old_var = list(expr_vars)[0]
                if str(old_var) != t_var:
                    expr = expr.subs(old_var, t)
            elif t not in expr_vars and expr_vars:
                # Multiple symbols but t is not one of them
                likely_t_vars = [v for v in expr_vars if len(str(v)) == 1 and str(v) not in ['I', 'E', 'a', 'b', 'c', 'k', 'n', 'm']]
                if len(likely_t_vars) == 1:
                    expr = expr.subs(likely_t_vars[0], t)
            
            # Compute Fourier transform
            transform = sp.fourier_transform(expr, t, omega)
            
            result['success'] = True
            result['original_expression'] = expr
            result['transform'] = sp.simplify(transform)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def inverse_fourier_transform(self, expression: str, omega_var: str = 'omega', t_var: str = 't') -> Dict[str, Any]:
        """
        Compute inverse Fourier transform
        
        Args:
            expression: Frequency domain expression as string
            omega_var: Frequency variable (default: 'omega')
            t_var: Time variable (default: 't')
            
        Returns:
            Dictionary with inverse Fourier transform results
        """
        result = {
            'success': False,
            'original_expression': None,
            'transform': None,
            'error': None
        }
        
        try:
            expr = self.parser.parse(expression)
            omega = sp.Symbol(omega_var, real=True)
            t = sp.Symbol(t_var, real=True)
            
            # Substitute the correct frequency variable if parsing introduced different variable
            expr_vars = expr.free_symbols
            if len(expr_vars) == 1:
                old_var = list(expr_vars)[0]
                if str(old_var) not in ['I', 'E', 'pi'] and str(old_var) != omega_var:
                    expr = expr.subs(old_var, omega)
            elif omega not in expr_vars and expr_vars:
                # Multiple symbols but omega is not one of them
                likely_omega_vars = [v for v in expr_vars if len(str(v)) == 1 and str(v) not in ['I', 'E', 'a', 'b', 'c', 'k', 'n', 'm', 't', 's']]
                if len(likely_omega_vars) == 1:
                    expr = expr.subs(likely_omega_vars[0], omega)
            
            # Compute inverse Fourier transform
            transform = sp.inverse_fourier_transform(expr, omega, t)
            
            result['success'] = True
            result['original_expression'] = expr
            result['transform'] = sp.simplify(transform)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def transfer_function_from_de(self, differential_equation: str, input_var: str = 'u', 
                                 output_var: str = 'y', t_var: str = 't', s_var: str = 's') -> Dict[str, Any]:
        """
        Find transfer function from differential equation using Laplace transform
        
        Args:
            differential_equation: Differential equation as string
            input_var: Input variable name
            output_var: Output variable name
            t_var: Time variable
            s_var: Laplace variable
            
        Returns:
            Dictionary with transfer function results
        """
        result = {
            'success': False,
            'differential_equation': None,
            'transfer_function': None,
            'poles': [],
            'zeros': [],
            'error': None
        }
        
        try:
            # Parse the differential equation and normalize to lhs - rhs = 0
            eq_parsed = self.parser.parse(differential_equation)
            t = sp.Symbol(t_var, real=True, positive=True)
            s = sp.Symbol(s_var)
            u = sp.Function(input_var)
            y = sp.Function(output_var)

            if isinstance(eq_parsed, sp.Equality):
                eq_expr = sp.simplify(eq_parsed.lhs - eq_parsed.rhs)
            else:
                eq_expr = sp.simplify(eq_parsed)

            # Expand to collect coefficients
            expr = sp.expand(eq_expr)

            y_func = y(t)
            u_func = u(t)

            def derivative_order(der: sp.Derivative) -> int:
                return sum(count for _, count in der.variable_count)

            # Collect coefficient dictionaries
            y_coeffs: Dict[int, Any] = {}
            u_coeffs: Dict[int, Any] = {}

            # First, collect all derivative terms
            y_derivs = {}
            u_derivs = {}
            
            for der in expr.atoms(sp.Derivative):
                if der.expr == y_func:
                    order = derivative_order(der)
                    y_derivs[order] = der
                elif der.expr == u_func:
                    order = derivative_order(der)
                    u_derivs[order] = der
            
            # Extract coefficients for y terms (including y itself)
            # Start with the full expression and subtract derivative terms
            remaining_expr = expr
            
            # Get coefficients for derivatives first
            for order in sorted(y_derivs.keys(), reverse=True):
                der = y_derivs[order]
                coeff = sp.expand(remaining_expr).coeff(der)
                if coeff is not None and coeff != 0:
                    y_coeffs[order] = coeff
                    remaining_expr = sp.expand(remaining_expr - coeff * der)
            
            # Order 0 coefficient (y itself)
            if remaining_expr.has(y_func):
                coeff = sp.expand(remaining_expr).coeff(y_func)
                if coeff is not None and coeff != 0:
                    y_coeffs[0] = coeff
            
            # Extract coefficients for u terms
            remaining_expr = expr
            
            for order in sorted(u_derivs.keys(), reverse=True):
                der = u_derivs[order]
                coeff = sp.expand(remaining_expr).coeff(der)
                if coeff is not None and coeff != 0:
                    u_coeffs[order] = coeff
                    remaining_expr = sp.expand(remaining_expr - coeff * der)
            
            # Order 0 coefficient (u itself)
            if remaining_expr.has(u_func):
                coeff = sp.expand(remaining_expr).coeff(u_func)
                if coeff is not None and coeff != 0:
                    u_coeffs[0] = coeff

            y_orders = sorted(y_coeffs.keys())
            u_orders = sorted(u_coeffs.keys())

            if not y_orders and not u_orders:
                raise InvalidInputError("differential equation", "no y(t) or u(t) terms", "equation must contain output or input terms")

            max_y_order = max(y_orders) if y_orders else 0
            max_u_order = max(u_orders) if u_orders else 0

            # Build denominator A(s) and numerator B(s)
            # Standard form: A(s)*Y(s) = B(s)*U(s)
            # For equation: a_n*y^(n) + ... + a_0*y = b_m*u^(m) + ... + b_0*u
            # Transfer function: H(s) = Y(s)/U(s) = B(s)/A(s)
            A_s = sp.S(0)
            for k in range(max_y_order + 1):
                coeff = y_coeffs.get(k, 0)
                if coeff != 0:
                    A_s += coeff * s**k

            B_s = sp.S(0)
            for k in range(max_u_order + 1):
                coeff = u_coeffs.get(k, 0)
                if coeff != 0:
                    # Note: coefficients from equation are already with correct sign
                    # If equation is: y'' + 2y' + y - 5u = 0, then u_coeff = -5
                    # We need B(s) = 5, so negate the coefficient
                    B_s += (-coeff) * s**k

            if A_s == 0:
                raise UndefinedOperationError("transfer function computation", "zero denominator coefficients")

            transfer_func = sp.simplify(B_s / A_s)

            numer, denom = sp.fraction(transfer_func)
            poles = sp.solve(denom, s)
            zeros = sp.solve(numer, s)

            result['success'] = True
            result['differential_equation'] = eq_parsed
            result['transfer_function'] = transfer_func
            result['poles'] = poles
            result['zeros'] = zeros
            result['denominator_poly'] = sp.simplify(A_s)
            result['numerator_poly'] = sp.simplify(B_s)

        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_de_with_laplace(self, differential_equation: str, initial_conditions: Dict[str, float],
                             input_function: str, t_var: str = 't', s_var: str = 's') -> Dict[str, Any]:
        """
        Solve differential equation using Laplace transform
        
        Args:
            differential_equation: Differential equation as string
            initial_conditions: Dictionary of initial conditions
            input_function: Input function (right-hand side)
            t_var: Time variable
            s_var: Laplace variable
            
        Returns:
            Dictionary with solution and analysis
        """
        result = {
            'success': False,
            'differential_equation': None,
            'solution': None,
            'transient': None,
            'steady_state': None,
            'zero_input': None,
            'zero_state': None,
            'error': None
        }
        
        try:
            # Parse differential equation
            eq = self.parser.parse(differential_equation)
            input_expr = self.parser.parse(input_function)
            
            t = sp.Symbol(t_var, real=True, positive=True)
            s = sp.Symbol(s_var)
            
            # Use dsolve with Laplace method if available
            # For now, use standard dsolve
            y = sp.Function('y')
            
            # Try to solve the ODE
            if isinstance(eq, sp.Equality):
                solution = sp.dsolve(eq, y(t))
            else:
                # Create equation
                solution = sp.dsolve(sp.Eq(eq, input_expr), y(t))
            
            # Helper function to apply ICs
            def apply_initial_conditions(sol, ics):
                if not ics or not hasattr(sol, 'rhs'):
                    return sol
                
                constants = sol.rhs.free_symbols - {t}
                if not constants:
                    return sol
                
                ic_equations = []
                const_list = sorted(list(constants), key=str)
                
                # Build IC equations up to the number of constants available
                # Always use available ICs - don't gate on length
                if 'y(0)' in ics:
                    ic_equations.append(sol.rhs.subs(t, 0) - ics['y(0)'])
                
                if "y'(0)" in ics and len(ic_equations) < len(constants):
                    dy_dt = sp.diff(sol.rhs, t)
                    ic_equations.append(dy_dt.subs(t, 0) - ics["y'(0)"])
                
                if "y''(0)" in ics and len(ic_equations) < len(constants):
                    d2y_dt2 = sp.diff(sol.rhs, t, 2)
                    ic_equations.append(d2y_dt2.subs(t, 0) - ics["y''(0)"])
                
                # Solve for constants if we have enough equations
                if ic_equations and len(ic_equations) >= len(constants):
                    try:
                        const_sol = sp.solve(ic_equations[:len(constants)], const_list)
                        if const_sol:
                            return sol.subs(const_sol)
                    except:
                        pass  # Keep original solution if IC application fails
                
                return sol
            
            # Apply initial conditions to full solution
            solution = apply_initial_conditions(solution, initial_conditions)
            
            # Zero-input response: solve with input = 0, use ICs
            try:
                if isinstance(eq, sp.Equality):
                    # For equation like: y'' + 2*y' + y = u(t)
                    # Zero-input: y'' + 2*y' + y = 0
                    # Replace RHS with 0
                    eq_zero_input = sp.Eq(eq.lhs, 0)
                else:
                    # For expression like: y'' + 2*y' + y - u(t)
                    # Need to remove u(t) terms and set to 0
                    eq_expr_zi = eq
                    # Remove all u derivative terms
                    for atom in eq.atoms(sp.Derivative):
                        if hasattr(atom, 'expr') and atom.expr == u(t):
                            eq_expr_zi = eq_expr_zi.subs(atom, 0)
                    # Remove u(t) itself
                    eq_expr_zi = eq_expr_zi.subs(u(t), 0)
                    eq_zero_input = sp.Eq(eq_expr_zi, 0)
                
                sol_zero_input = sp.dsolve(eq_zero_input, y(t))
                sol_zero_input = apply_initial_conditions(sol_zero_input, initial_conditions)
                
                if hasattr(sol_zero_input, 'rhs'):
                    result['zero_input'] = sp.simplify(sol_zero_input.rhs)
            except:
                result['zero_input'] = None
            
            # Zero-state response: solve with ICs = 0, use input
            try:
                # Always create zero ICs - check what order ICs were provided
                zero_ics = {'y(0)': 0}
                # If higher order ICs were provided, also zero them
                if initial_conditions:
                    if "y'(0)" in initial_conditions:
                        zero_ics["y'(0)"] = 0
                    if "y''(0)" in initial_conditions:
                        zero_ics["y''(0)"] = 0
                
                if isinstance(eq, sp.Equality):
                    sol_zero_state = sp.dsolve(eq, y(t))
                else:
                    sol_zero_state = sp.dsolve(sp.Eq(eq, input_expr), y(t))
                
                sol_zero_state = apply_initial_conditions(sol_zero_state, zero_ics)
                
                if hasattr(sol_zero_state, 'rhs'):
                    result['zero_state'] = sp.simplify(sol_zero_state.rhs)
            except:
                result['zero_state'] = None
            
            # Decompose solution using pole analysis
            if hasattr(solution, 'rhs'):
                full_solution = solution.rhs
                
                # Try to identify transient and steady-state using pole analysis
                transient = sp.S(0)
                steady = sp.S(0)
                
                # Extract poles from the characteristic equation if possible
                try:
                    # For each exponential term, check if it decays (Re{λ} < 0)
                    if full_solution.is_Add:
                        for term in full_solution.as_ordered_terms():
                            is_transient = False
                            is_steady = False
                            
                            # Check for exponential terms
                            has_exp_with_t = False
                            for exp_term in sp.preorder_traversal(term):
                                if isinstance(exp_term, sp.exp):
                                    arg = exp_term.args[0]
                                    if arg.has(t):
                                        has_exp_with_t = True
                                        # Extract coefficient of t
                                        coeff = arg.coeff(t)
                                        if coeff is not None:
                                            # Check if real part is negative (decaying)
                                            try:
                                                re_part = sp.re(coeff)
                                                if re_part.is_negative or (re_part.is_number and complex(re_part).real < 0):
                                                    is_transient = True
                                                    break
                                                elif re_part == 0:
                                                    # Pure imaginary (bounded oscillation) - could be steady
                                                    is_steady = True
                                                else:
                                                    # Positive real part - growing (unstable transient)
                                                    is_transient = True
                                                    break
                                            except:
                                                # Can't determine - assume transient
                                                is_transient = True
                                                break
                            
                            if is_transient:
                                transient += term
                            elif is_steady:
                                steady += term
                            elif has_exp_with_t:
                                # Has exponential but couldn't determine - assume transient
                                transient += term
                            else:
                                # No time-dependent exponential - check limit
                                try:
                                    limit_val = sp.limit(term, t, sp.oo)
                                    if limit_val.is_finite and not limit_val.has(t):
                                        steady += term
                                    else:
                                        # Unbounded or indeterminate
                                        transient += term
                                except:
                                    # Can't compute limit - default to steady if no exp
                                    steady += term
                    else:
                        # Single term - check if it has decaying exponential
                        is_transient = False
                        is_steady = False
                        has_exp_with_t = False
                        
                        for exp_term in sp.preorder_traversal(full_solution):
                            if isinstance(exp_term, sp.exp):
                                arg = exp_term.args[0]
                                if arg.has(t):
                                    has_exp_with_t = True
                                    coeff = arg.coeff(t)
                                    if coeff is not None:
                                        try:
                                            re_part = sp.re(coeff)
                                            if re_part.is_negative or (re_part.is_number and complex(re_part).real < 0):
                                                is_transient = True
                                                break
                                            elif re_part == 0:
                                                # Pure imaginary - bounded oscillation
                                                is_steady = True
                                            else:
                                                # Growing exponential
                                                is_transient = True
                                                break
                                        except:
                                            is_transient = True
                                            break
                        
                        if is_transient:
                            transient = full_solution
                        elif is_steady:
                            steady = full_solution
                        elif has_exp_with_t:
                            transient = full_solution
                        else:
                            try:
                                limit_val = sp.limit(full_solution, t, sp.oo)
                                if limit_val.is_finite:
                                    steady = full_solution
                                else:
                                    transient = full_solution
                            except:
                                steady = full_solution
                
                except:
                    # Fallback to simple heuristic
                    if full_solution.has(sp.exp):
                        transient = full_solution
                    else:
                        steady = full_solution
                
                result['transient'] = sp.simplify(transient) if transient != 0 else None
                result['steady_state'] = sp.simplify(steady) if steady != 0 else None
                result['solution'] = full_solution
            else:
                result['solution'] = solution
            
            result['success'] = True
            result['differential_equation'] = eq
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def solve_de_with_fourier(self, differential_equation: str, boundary_conditions: Dict[str, float],
                              x_var: str = 'x', omega_var: str = 'omega') -> Dict[str, Any]:
        """
        Solve differential equation using Fourier transform
        
        Args:
            differential_equation: Differential equation as string
            boundary_conditions: Dictionary of boundary conditions
            x_var: Spatial variable
            omega_var: Frequency variable
            
        Returns:
            Dictionary with solution
        """
        result = {
            'success': False,
            'differential_equation': None,
            'solution': None,
            'error': None
        }
        
        try:
            eq = self.parser.parse(differential_equation)
            x = sp.Symbol(x_var, real=True)
            omega = sp.Symbol(omega_var, real=True)
            
            # Apply Fourier transform to the equation
            # Solve in frequency domain
            # Apply inverse Fourier transform
            
            # This is a placeholder for more sophisticated implementation
            result['success'] = False
            result['error'] = "Fourier transform DE solving not yet fully implemented"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def compute_fourier_series(self, expression: str, variable: str = 'x', 
                               period: float = 2*np.pi, n_terms: int = 10) -> Dict[str, Any]:
        """
        Compute Fourier series expansion of a periodic function
        
        Args:
            expression: Mathematical expression as string
            variable: Variable name
            period: Period of the function (default: 2π)
            n_terms: Number of terms in the series
            
        Returns:
            Dictionary with Fourier series results
        """
        result = {
            'success': False,
            'original_expression': None,
            'fourier_series': None,
            'a0': None,
            'an_coefficients': {},
            'bn_coefficients': {},
            'complex_form': None,
            'error': None
        }
        
        try:
            expr = self.parser.parse(expression)
            var = sp.Symbol(variable, real=True)
            
            # Compute Fourier series
            L = period / 2  # Half period
            
            # a0 coefficient (average value)
            a0 = (1 / L) * sp.integrate(expr, (var, -L, L))
            
            # an and bn coefficients
            an_coeffs = {}
            bn_coeffs = {}
            
            for n in range(1, n_terms + 1):
                # an = (1/L) * integral of f(x)*cos(n*pi*x/L) from -L to L
                an = (1 / L) * sp.integrate(expr * sp.cos(n * sp.pi * var / L), (var, -L, L))
                an_coeffs[n] = sp.simplify(an)
                
                # bn = (1/L) * integral of f(x)*sin(n*pi*x/L) from -L to L
                bn = (1 / L) * sp.integrate(expr * sp.sin(n * sp.pi * var / L), (var, -L, L))
                bn_coeffs[n] = sp.simplify(bn)
            
            # Build Fourier series
            fourier_series = a0 / 2
            for n in range(1, n_terms + 1):
                fourier_series += an_coeffs[n] * sp.cos(n * sp.pi * var / L)
                fourier_series += bn_coeffs[n] * sp.sin(n * sp.pi * var / L)
            
            fourier_series = sp.simplify(fourier_series)
            
            # Complex exponential form
            complex_coeffs = {}
            for n in range(-n_terms, n_terms + 1):
                cn = (1 / period) * sp.integrate(expr * sp.exp(-sp.I * n * 2 * sp.pi * var / period), 
                                                  (var, -L, L))
                complex_coeffs[n] = sp.simplify(cn)
            
            complex_form = sum(complex_coeffs[n] * sp.exp(sp.I * n * 2 * sp.pi * var / period) 
                             for n in range(-n_terms, n_terms + 1))
            
            result['success'] = True
            result['original_expression'] = expr
            result['fourier_series'] = fourier_series
            result['a0'] = sp.simplify(a0)
            result['an_coefficients'] = an_coeffs
            result['bn_coefficients'] = bn_coeffs
            result['complex_form'] = sp.simplify(complex_form)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def decompose_solution(self, solution: str, t_var: str = 't') -> Dict[str, Any]:
        """
        Decompose solution into transient and steady-state components
        
        Args:
            solution: Solution expression as string
            t_var: Time variable
            
        Returns:
            Dictionary with decomposed components
        """
        result = {
            'success': False,
            'original_solution': None,
            'transient': None,
            'steady_state': None,
            'error': None
        }
        
        try:
            sol = self.parser.parse(solution)
            t = sp.Symbol(t_var, real=True, positive=True)
            
            transient_part = sp.S(0)
            steady_part = sp.S(0)
            
            if sol.is_Add:
                for term in sol.as_ordered_terms():
                    # Check if term decays to zero as t -> infinity
                    limit_at_inf = sp.limit(term, t, sp.oo)
                    if limit_at_inf == 0 or (hasattr(limit_at_inf, 'is_zero') and limit_at_inf.is_zero):
                        transient_part += term
                    else:
                        steady_part += term
            else:
                # Single term
                limit_at_inf = sp.limit(sol, t, sp.oo)
                if limit_at_inf == 0 or (hasattr(limit_at_inf, 'is_zero') and limit_at_inf.is_zero):
                    transient_part = sol
                else:
                    steady_part = sol
            
            result['success'] = True
            result['original_solution'] = sol
            result['transient'] = sp.simplify(transient_part) if transient_part != 0 else None
            result['steady_state'] = sp.simplify(steady_part) if steady_part != 0 else None
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def compute_zero_input_zero_state(self, transfer_function: str, input_function: str, 
                                     initial_conditions: Dict[str, float], 
                                     s_var: str = 's', t_var: str = 't') -> Dict[str, Any]:
        """
        Compute zero-input and zero-state responses
        
        Args:
            transfer_function: Transfer function H(s) as string
            input_function: Input u(t) as string
            initial_conditions: Dictionary of initial conditions
            s_var: Laplace variable
            t_var: Time variable
            
        Returns:
            Dictionary with zero-input and zero-state responses
        """
        result = {
            'success': False,
            'zero_input': None,
            'zero_state': None,
            'total_response': None,
            'error': None
        }
        
        try:
            H_s = self.parser.parse(transfer_function)
            u_t = self.parser.parse(input_function)
            
            s = sp.Symbol(s_var)
            t = sp.Symbol(t_var, real=True, positive=True)
            
            # Zero-state response: Y_zs(s) = H(s) * U(s)
            U_s = sp.laplace_transform(u_t, t, s, noconds=True)
            Y_zs_s = H_s * U_s
            y_zs = sp.inverse_laplace_transform(Y_zs_s, s, t)
            
            # Zero-input response: depends on initial conditions and system poles
            # Y_zi(s) = sum of IC terms / denominator of H(s)
            # This requires knowledge of the original differential equation
            # For now, we'll provide a placeholder
            
            # Extract denominator from H(s) for characteristic equation
            numer, denom = sp.fraction(H_s)
            
            # Zero-input response calculation would require initial conditions
            # and the characteristic equation solution
            y_zi = sp.S(0)  # Placeholder
            
            if initial_conditions:
                # Build zero-input response from initial conditions
                # This is simplified - actual implementation needs characteristic equation
                pass
            
            result['success'] = True
            result['zero_state'] = sp.simplify(y_zs)
            result['zero_input'] = y_zi if y_zi != 0 else None
            result['total_response'] = sp.simplify(y_zs + y_zi) if y_zi != 0 else sp.simplify(y_zs)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def analyze_frequency_response(self, transfer_function: str, s_var: str = 's') -> Dict[str, Any]:
        """
        Analyze frequency response of a transfer function
        
        Args:
            transfer_function: Transfer function H(s) as string
            s_var: Laplace variable
            
        Returns:
            Dictionary with frequency response analysis
        """
        result = {
            'success': False,
            'transfer_function': None,
            'magnitude': None,
            'phase': None,
            'poles': [],
            'zeros': [],
            'stable': None,
            'error': None
        }
        
        try:
            H_s = self.parser.parse(transfer_function)
            s = sp.Symbol(s_var)
            omega = sp.Symbol('omega', real=True)
            
            # Substitute s = j*omega for frequency response
            H_jw = H_s.subs(s, sp.I * omega)
            
            # Magnitude and phase
            magnitude = sp.simplify(sp.Abs(H_jw))
            phase = sp.simplify(sp.arg(H_jw))
            
            # Find poles and zeros
            numer, denom = sp.fraction(H_s)
            poles = sp.solve(denom, s)
            zeros = sp.solve(numer, s)
            
            # Stability check: all poles must have negative real parts
            stable = True
            for pole in poles:
                try:
                    # Evaluate the real part
                    if pole.is_real:
                        pole_val = complex(pole)
                        if pole_val.real >= 0:
                            stable = False
                            break
                    else:
                        real_part = sp.re(pole)
                        real_val = complex(real_part.evalf())
                        if real_val.real >= 0:
                            stable = False
                            break
                except:
                    # If we can't evaluate, assume potentially unstable
                    stable = None
                    break
            
            result['success'] = True
            result['transfer_function'] = H_s
            result['magnitude'] = magnitude
            result['phase'] = phase
            result['poles'] = poles
            result['zeros'] = zeros
            result['stable'] = stable
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


# Global transforms and series engine instance
transforms_series_engine = TransformsSeriesEngine()
