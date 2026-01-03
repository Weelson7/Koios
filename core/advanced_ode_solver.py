import numpy as np
import sympy as sp
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum

class ODEType(Enum):
    """Types of differential equations"""
    FIRST_ORDER = "FIRST_ORDER"
    SECOND_ORDER = "SECOND_ORDER"
    HIGHER_ORDER = "HIGHER_ORDER"
    SYSTEM = "SYSTEM"
    PARTIAL = "PARTIAL"
    DELAY = "DELAY"
    STOCHASTIC = "STOCHASTIC"

class SolverMethod(Enum):
    """Available numerical methods"""
    EULER = "EULER"
    IMPROVED_EULER = "IMPROVED_EULER"
    RK4 = "RK4"
    RK45 = "RK45"
    ADAMS_BASHFORTH = "ADAMS_BASHFORTH"
    ADAMS_MOULTON = "ADAMS_MOULTON"
    BDF = "BDF"  # Backward Differentiation Formula
    RADAU = "RADAU"
    DOP853 = "DOP853"

@dataclass
class ODEProblem:
    """Definition of an ODE problem"""
    equation: Union[Callable, str]
    initial_conditions: Dict[str, float]
    domain: Tuple[float, float]
    parameters: Optional[Dict[str, float]] = None
    boundary_conditions: Optional[Dict[str, float]] = None
    ode_type: ODEType = ODEType.FIRST_ORDER

@dataclass
class ODESolution:
    """Solution of an ODE problem"""
    t: np.ndarray
    y: np.ndarray
    method: str
    success: bool
    message: str
    error_estimate: Optional[np.ndarray] = None
    stability_info: Optional[Dict[str, Any]] = None

class AdvancedODESolver:
    """Advanced differential equation solver"""
    
    def __init__(self):
        self.methods = {
            SolverMethod.EULER: self._euler_method,
            SolverMethod.IMPROVED_EULER: self._improved_euler_method,
            SolverMethod.RK4: self._rk4_method,
            SolverMethod.RK45: self._rk45_scipy,
            SolverMethod.ADAMS_BASHFORTH: self._adams_bashforth,
            SolverMethod.BDF: self._bdf_scipy,
            SolverMethod.RADAU: self._radau_scipy,
            SolverMethod.DOP853: self._dop853_scipy
        }
    
    def solve_ode(self, problem: ODEProblem, method: SolverMethod = SolverMethod.RK45,
                  num_points: int = 1000, adaptive: bool = True) -> ODESolution:
        """Solve an ODE problem using specified method"""
        if method not in self.methods:
            return ODESolution(
                t=np.array([]), y=np.array([]),
                method=method.value, success=False,
                message=f"Method {method.value} not implemented"
            )
        
        try:
            return self.methods[method](problem, num_points, adaptive)
        except Exception as e:
            return ODESolution(
                t=np.array([]), y=np.array([]),
                method=method.value, success=False,
                message=f"Solver failed: {str(e)}"
            )
    
    def solve_system(self, equations: List[Callable], initial_conditions: np.ndarray,
                    t_span: Tuple[float, float], method: str = 'RK45') -> ODESolution:
        """Solve a system of ODEs"""
        def system(t, y):
            return [eq(t, y) for eq in equations]
        
        sol = solve_ivp(system, t_span, initial_conditions, method=method, 
                       dense_output=True, rtol=1e-8, atol=1e-10)
        
        return ODESolution(
            t=sol.t, y=sol.y.T,
            method=method, success=sol.success,
            message=sol.message if hasattr(sol, 'message') else "Solved successfully"
        )
    
    def solve_bvp(self, equation: Callable, boundary_conditions: Callable,
                 x_span: Tuple[float, float], y_guess: np.ndarray) -> ODESolution:
        """Solve boundary value problem"""
        x = np.linspace(x_span[0], x_span[1], 100)
        y_init = np.ones((len(y_guess), x.size))
        
        for i, guess in enumerate(y_guess):
            y_init[i] = guess
        
        sol = solve_bvp(equation, boundary_conditions, x, y_init)
        
        return ODESolution(
            t=sol.x, y=sol.y.T,
            method="BVP", success=sol.success,
            message=sol.message if hasattr(sol, 'message') else "BVP solved"
        )
    
    def _euler_method(self, problem: ODEProblem, num_points: int, 
                     adaptive: bool) -> ODESolution:
        """Basic Euler method"""
        t0, tf = problem.domain
        t = np.linspace(t0, tf, num_points)
        h = (tf - t0) / (num_points - 1)
        
        # Convert equation if string
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        # Initialize solution
        y = np.zeros(num_points)
        y[0] = list(problem.initial_conditions.values())[0]
        
        # Euler iteration
        for i in range(1, num_points):
            y[i] = y[i-1] + h * f(t[i-1], y[i-1])
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.EULER.value,
            success=True,
            message="Solved using Euler method"
        )
    
    def _improved_euler_method(self, problem: ODEProblem, num_points: int,
                             adaptive: bool) -> ODESolution:
        """Improved Euler (Heun's) method"""
        t0, tf = problem.domain
        t = np.linspace(t0, tf, num_points)
        h = (tf - t0) / (num_points - 1)
        
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y = np.zeros(num_points)
        y[0] = list(problem.initial_conditions.values())[0]
        
        for i in range(1, num_points):
            # Predictor
            y_pred = y[i-1] + h * f(t[i-1], y[i-1])
            # Corrector
            y[i] = y[i-1] + h/2 * (f(t[i-1], y[i-1]) + f(t[i], y_pred))
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.IMPROVED_EULER.value,
            success=True,
            message="Solved using Improved Euler method"
        )
    
    def _rk4_method(self, problem: ODEProblem, num_points: int,
                   adaptive: bool) -> ODESolution:
        """4th order Runge-Kutta method"""
        t0, tf = problem.domain
        t = np.linspace(t0, tf, num_points)
        h = (tf - t0) / (num_points - 1)
        
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y = np.zeros(num_points)
        y[0] = list(problem.initial_conditions.values())[0]
        
        for i in range(1, num_points):
            k1 = h * f(t[i-1], y[i-1])
            k2 = h * f(t[i-1] + h/2, y[i-1] + k1/2)
            k3 = h * f(t[i-1] + h/2, y[i-1] + k2/2)
            k4 = h * f(t[i], y[i-1] + k3)
            
            y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.RK4.value,
            success=True,
            message="Solved using RK4 method"
        )
    
    def _rk45_scipy(self, problem: ODEProblem, num_points: int,
                   adaptive: bool) -> ODESolution:
        """Adaptive RK45 using SciPy"""
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y0 = list(problem.initial_conditions.values())[0]
        
        sol = solve_ivp(
            lambda t, y: f(t, y),
            problem.domain,
            [y0],
            method='RK45',
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        # Evaluate at requested points
        t = np.linspace(problem.domain[0], problem.domain[1], num_points)
        y = sol.sol(t)[0]
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.RK45.value,
            success=sol.success,
            message=sol.message if hasattr(sol, 'message') else "Solved using RK45"
        )
    
    def _adams_bashforth(self, problem: ODEProblem, num_points: int,
                        adaptive: bool) -> ODESolution:
        """4-step Adams-Bashforth method"""
        # Use RK4 for initial steps
        rk4_sol = self._rk4_method(problem, 4, False)
        
        t0, tf = problem.domain
        t = np.linspace(t0, tf, num_points)
        h = (tf - t0) / (num_points - 1)
        
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y = np.zeros(num_points)
        y[:4] = rk4_sol.y[:4]
        
        # Store function evaluations
        f_vals = [f(t[i], y[i]) for i in range(4)]
        
        # Adams-Bashforth 4-step formula
        for i in range(4, num_points):
            y[i] = y[i-1] + h/24 * (55*f_vals[-1] - 59*f_vals[-2] + 
                                    37*f_vals[-3] - 9*f_vals[-4])
            
            # Update function values
            f_vals.pop(0)
            f_vals.append(f(t[i], y[i]))
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.ADAMS_BASHFORTH.value,
            success=True,
            message="Solved using Adams-Bashforth method"
        )
    
    def _bdf_scipy(self, problem: ODEProblem, num_points: int,
                  adaptive: bool) -> ODESolution:
        """Backward Differentiation Formula for stiff equations"""
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y0 = list(problem.initial_conditions.values())[0]
        
        sol = solve_ivp(
            lambda t, y: f(t, y),
            problem.domain,
            [y0],
            method='BDF',
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        t = np.linspace(problem.domain[0], problem.domain[1], num_points)
        y = sol.sol(t)[0]
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.BDF.value,
            success=sol.success,
            message="Solved using BDF (stiff solver)"
        )
    
    def _radau_scipy(self, problem: ODEProblem, num_points: int,
                    adaptive: bool) -> ODESolution:
        """Radau method for stiff equations"""
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y0 = list(problem.initial_conditions.values())[0]
        
        sol = solve_ivp(
            lambda t, y: f(t, y),
            problem.domain,
            [y0],
            method='Radau',
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        t = np.linspace(problem.domain[0], problem.domain[1], num_points)
        y = sol.sol(t)[0]
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.RADAU.value,
            success=sol.success,
            message="Solved using Radau method"
        )
    
    def _dop853_scipy(self, problem: ODEProblem, num_points: int,
                     adaptive: bool) -> ODESolution:
        """8th order Dormand-Prince method"""
        if isinstance(problem.equation, str):
            f = self._parse_equation(problem.equation)
        else:
            f = problem.equation
        
        y0 = list(problem.initial_conditions.values())[0]
        
        sol = solve_ivp(
            lambda t, y: f(t, y),
            problem.domain,
            [y0],
            method='DOP853',
            dense_output=True,
            rtol=1e-10,
            atol=1e-12
        )
        
        t = np.linspace(problem.domain[0], problem.domain[1], num_points)
        y = sol.sol(t)[0]
        
        return ODESolution(
            t=t, y=y,
            method=SolverMethod.DOP853.value,
            success=sol.success,
            message="Solved using DOP853 (high precision)"
        )
    
    def _parse_equation(self, equation_str: str) -> Callable:
        """Parse string equation to callable function"""
        # Use SymPy for parsing
        t = sp.Symbol('t')
        y = sp.Symbol('y')
        
        expr = sp.sympify(equation_str)
        f_lambdified = sp.lambdify((t, y), expr, 'numpy')
        
        return f_lambdified
    
    def analyze_stability(self, equation: Callable, equilibrium_point: float,
                         linearize: bool = True) -> Dict[str, Any]:
        """Analyze stability of ODE at equilibrium point"""
        if linearize:
            # Numerical derivative at equilibrium
            eps = 1e-8
            df_dy = (equation(0, equilibrium_point + eps) - 
                    equation(0, equilibrium_point - eps)) / (2 * eps)
            
            stability = {
                'equilibrium': equilibrium_point,
                'derivative': df_dy,
                'stable': df_dy < 0,
                'type': 'stable' if df_dy < 0 else 'unstable'
            }
        else:
            # Nonlinear stability analysis would go here
            stability = {
                'equilibrium': equilibrium_point,
                'message': 'Nonlinear stability analysis not implemented'
            }
        
        return stability
    
    def estimate_error(self, coarse_solution: np.ndarray, 
                      fine_solution: np.ndarray) -> np.ndarray:
        """Estimate error using Richardson extrapolation"""
        # Interpolate fine solution to coarse grid
        n_coarse = len(coarse_solution)
        n_fine = len(fine_solution)
        
        indices = np.linspace(0, n_fine-1, n_coarse).astype(int)
        fine_on_coarse = fine_solution[indices]
        
        # Error estimate
        error = np.abs(fine_on_coarse - coarse_solution)
        
        return error
    
    def solve_pde_heat_equation(self, initial_condition: Callable,
                               boundary_conditions: Tuple[float, float],
                               x_domain: Tuple[float, float],
                               t_domain: Tuple[float, float],
                               nx: int = 100, nt: int = 1000,
                               alpha: float = 1.0) -> Dict[str, np.ndarray]:
        """Solve 1D heat equation using finite differences"""
        x = np.linspace(x_domain[0], x_domain[1], nx)
        t = np.linspace(t_domain[0], t_domain[1], nt)
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        
        # Stability check
        r = alpha * dt / dx**2
        if r > 0.5:
            print(f"Warning: Unstable parameters. r = {r} > 0.5")
        
        # Initialize solution
        u = np.zeros((nt, nx))
        u[0, :] = initial_condition(x)
        
        # Boundary conditions
        u[:, 0] = boundary_conditions[0]
        u[:, -1] = boundary_conditions[1]
        
        # Time stepping
        for n in range(0, nt-1):
            for i in range(1, nx-1):
                u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        
        return {
            'x': x,
            't': t,
            'u': u,
            'stability_parameter': r
        }
    
    def solve_pde_wave_equation(self, initial_position: Callable,
                               initial_velocity: Callable,
                               boundary_conditions: Tuple[float, float],
                               x_domain: Tuple[float, float],
                               t_domain: Tuple[float, float],
                               nx: int = 100, nt: int = 1000,
                               c: float = 1.0) -> Dict[str, np.ndarray]:
        """Solve 1D wave equation using finite differences"""
        x = np.linspace(x_domain[0], x_domain[1], nx)
        t = np.linspace(t_domain[0], t_domain[1], nt)
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        
        # Courant number
        C = c * dt / dx
        if C > 1:
            print(f"Warning: CFL condition violated. C = {C} > 1")
        
        # Initialize solution
        u = np.zeros((nt, nx))
        u[0, :] = initial_position(x)
        
        # First time step using initial velocity
        for i in range(1, nx-1):
            u[1, i] = u[0, i] + dt * initial_velocity(x[i]) + \
                     0.5 * C**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
        
        # Time stepping
        for n in range(1, nt-1):
            for i in range(1, nx-1):
                u[n+1, i] = 2*u[n, i] - u[n-1, i] + \
                           C**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            # Boundary conditions
            u[n+1, 0] = boundary_conditions[0]
            u[n+1, -1] = boundary_conditions[1]
        
        return {
            'x': x,
            't': t,
            'u': u,
            'courant_number': C
        }

# Example problems and usage
def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    """Lorenz attractor system"""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def van_der_pol(t, state, mu=1.0):
    """Van der Pol oscillator"""
    x, y = state
    return [y, mu * (1 - x**2) * y - x]

def predator_prey(t, state, a=1.0, b=0.1, c=1.5, d=0.075):
    """Lotka-Volterra predator-prey model"""
    prey, predator = state
    return [
        a * prey - b * prey * predator,
        -c * predator + d * prey * predator
    ]

# Example usage
if __name__ == "__main__":
    solver = AdvancedODESolver()
    
    # Example 1: Simple exponential decay
    problem1 = ODEProblem(
        equation=lambda t, y: -0.5 * y,
        initial_conditions={'y0': 1.0},
        domain=(0, 10)
    )
    
    solution1 = solver.solve_ode(problem1, SolverMethod.RK45)
    print(f"Exponential decay solved: {solution1.success}")
    
    # Example 2: Lorenz system
    lorenz_solution = solver.solve_system(
        [lambda t, s: lorenz_system(t, s)[i] for i in range(3)],
        initial_conditions=np.array([1.0, 1.0, 1.0]),
        t_span=(0, 50)
    )
    print(f"Lorenz system solved: {lorenz_solution.success}")
    
    # Example 3: Heat equation
    heat_solution = solver.solve_pde_heat_equation(
        initial_condition=lambda x: np.sin(np.pi * x),
        boundary_conditions=(0, 0),
        x_domain=(0, 1),
        t_domain=(0, 0.5)
    )
    print(f"Heat equation solved with stability parameter: {heat_solution['stability_parameter']}")