import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as sp_opt
from scipy.stats import norm
import warnings
from utils.exceptions import UndefinedOperationError

class OptimizationType(Enum):
    """Types of optimization problems"""
    UNCONSTRAINED = "UNCONSTRAINED"
    CONSTRAINED = "CONSTRAINED"
    MULTI_OBJECTIVE = "MULTI_OBJECTIVE"
    ROBUST = "ROBUST"
    STOCHASTIC = "STOCHASTIC"

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    x_opt: np.ndarray
    f_opt: float
    iterations: int
    converged: bool
    constraint_violation: Optional[float] = None
    pareto_front: Optional[np.ndarray] = None
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    history: Optional[Dict[str, List]] = None

class OptimizationAlgorithmsEngine:
    """Advanced optimization algorithms engine"""
    
    def __init__(self):
        self.tolerance = 1e-8
        self.max_iterations = 1000
        self.population_size = 50
        
    def lbfgs_b(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None,
                m: int = 10) -> OptimizationResult:
        """
        Limited-memory BFGS with bound constraints
        
        Args:
            f: Objective function
            grad_f: Gradient function
            x0: Initial point
            bounds: Variable bounds [(low, high), ...]
            m: Number of corrections to approximate Hessian
        """
        n = len(x0)
        x = x0.copy()
        
        # Initialize L-BFGS memory
        s_list = []  # x differences
        y_list = []  # gradient differences
        
        # History tracking
        history = {'f_vals': [f(x)], 'grad_norms': []}
        
        g = grad_f(x)
        history['grad_norms'].append(np.linalg.norm(g))
        
        for k in range(self.max_iterations):
            if np.linalg.norm(g) < self.tolerance:
                break
            
            # Compute search direction using L-BFGS two-loop recursion
            q = g.copy()
            alpha_list = []
            
            # First loop
            for i in range(len(s_list) - 1, -1, -1):
                rho_i = 1.0 / np.dot(y_list[i], s_list[i])
                alpha_i = rho_i * np.dot(s_list[i], q)
                q = q - alpha_i * y_list[i]
                alpha_list.append(alpha_i)
            
            # Scaling
            if len(s_list) > 0:
                denominator = np.dot(y_list[-1], y_list[-1])
                if abs(denominator) < 1e-10:
                    gamma = 1.0  # Default scaling
                else:
                    gamma = np.dot(s_list[-1], y_list[-1]) / denominator
                r = gamma * q
            else:
                r = q
            
            # Second loop
            alpha_list.reverse()
            for i in range(len(s_list)):
                rho_i = 1.0 / np.dot(y_list[i], s_list[i])
                beta = rho_i * np.dot(y_list[i], r)
                r = r + s_list[i] * (alpha_list[i] - beta)
            
            # Search direction
            d = -r
            
            # Line search with bound constraints
            alpha = self._bounded_line_search(f, grad_f, x, d, bounds)
            
            # Update
            x_new = x + alpha * d
            
            # Apply bounds
            if bounds:
                for i, (low, high) in enumerate(bounds):
                    x_new[i] = np.clip(x_new[i], low, high)
            
            g_new = grad_f(x_new)
            
            # Update L-BFGS memory
            s = x_new - x
            y = g_new - g
            
            if np.dot(s, y) > 1e-10:  # Curvature condition
                s_list.append(s)
                y_list.append(y)
                
                # Limit memory
                if len(s_list) > m:
                    s_list.pop(0)
                    y_list.pop(0)
            
            # Move to next iteration
            x = x_new
            g = g_new
            
            history['f_vals'].append(f(x))
            history['grad_norms'].append(np.linalg.norm(g))
        
        return OptimizationResult(
            x_opt=x,
            f_opt=f(x),
            iterations=k + 1,
            converged=np.linalg.norm(g) < self.tolerance,
            history=history
        )
    
    def _bounded_line_search(self, f: Callable, grad_f: Callable, 
                           x: np.ndarray, d: np.ndarray,
                           bounds: Optional[List[Tuple[float, float]]]) -> float:
        """Line search with bound constraints"""
        alpha_max = 1.0
        
        # Find maximum step size that satisfies bounds
        if bounds:
            for i, (low, high) in enumerate(bounds):
                if d[i] > 0:
                    alpha_max = min(alpha_max, (high - x[i]) / d[i])
                elif d[i] < 0:
                    alpha_max = min(alpha_max, (low - x[i]) / d[i])
        
        # Armijo line search
        alpha = min(1.0, alpha_max)
        c1 = 1e-4
        rho = 0.8
        
        f0 = f(x)
        g0 = np.dot(grad_f(x), d)
        
        while alpha > 1e-10:
            if f(x + alpha * d) <= f0 + c1 * alpha * g0:
                break
            alpha *= rho
        
        return alpha
    
    def optimize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        General optimization interface
        
        Args:
            params: Dictionary containing optimization parameters
        """
        try:
            objective_function = params['objective_function']
            variables = params['variables']
            method = params['method']
            initial_guess = params['initial_guess']
            tolerance = params.get('tolerance', self.tolerance)
            max_iterations = params.get('max_iterations', self.max_iterations)
            
            # Parse objective function
            import sympy as sp
            from core.expression_parser import expression_parser
            
            expr = expression_parser.parse(objective_function)
            var_symbols = [sp.Symbol(var) for var in variables]
            
            # Create numerical function
            func = sp.lambdify(var_symbols, expr, 'numpy')
            
            # Create gradient function
            grad_exprs = [sp.diff(expr, var) for var in var_symbols]
            grad_func = sp.lambdify(var_symbols, grad_exprs, 'numpy')
            
            def objective(x):
                return float(func(*x))
            
            def gradient(x):
                result = grad_func(*x)
                if isinstance(result, (int, float)):
                    return np.array([result])
                return np.array(result)
            
            # Set up bounds (default to reasonable range)
            bounds = [(-10, 10)] * len(variables)
            
            # Use appropriate method
            if method == 'gradient_descent':
                result = self.gradient_descent(objective, gradient, np.array(initial_guess), tolerance, max_iterations)
            elif method == 'newton_method':
                result = self.newton_method(objective, gradient, np.array(initial_guess), tolerance, max_iterations)
            elif method == 'nelder_mead':
                result = self.nelder_mead(objective, np.array(initial_guess), tolerance, max_iterations)
            else:
                # Default to genetic algorithm
                result = self.genetic_algorithm(objective, bounds, crossover_prob=0.8, mutation_prob=0.1)
            
            return {
                'success': True,
                'optimal_point': result.x_opt.tolist(),
                'optimal_value': result.f_opt,
                'iterations': result.iterations,
                'convergence_history': result.history.get('f_vals', []) if result.history else []
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def linear_programming(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Linear programming solver
        
        Args:
            params: Dictionary containing LP parameters
        """
        try:
            c = params['c']
            A = params['A']
            b = params['b']
            objective = params.get('objective', 'minimize')
            bounds = params.get('bounds', None)
            
            # Simple simplex-like solver (basic implementation)
            # This is a simplified version - full implementation would use scipy.optimize.linprog
            
            # Convert to standard form: minimize c^T x subject to Ax <= b, x >= 0
            c = np.array(c)
            A = np.array(A)
            b = np.array(b)
            
            # If maximizing, negate the objective
            if objective == 'maximize':
                c = -c
            
            # Use a simple iterative approach for demonstration
            n_vars = len(c)
            n_constraints = len(b)
            
            # Start with a feasible point (origin if feasible)
            x = np.zeros(n_vars)
            
            # Check if origin is feasible
            if np.all(A @ x <= b):
                feasible = True
            else:
                # Find a feasible point (simplified)
                x = np.linalg.lstsq(A, b * 0.9, rcond=None)[0]
                x = np.maximum(x, 0)  # Ensure non-negativity
                feasible = np.all(A @ x <= b)
            
            if not feasible:
                return {
                    'success': False,
                    'error': 'No feasible solution found'
                }
            
            # Simple gradient-based approach
            for iteration in range(100):
                # Compute gradient
                grad = c
                
                # Project gradient onto feasible direction
                # This is a simplified approach
                step_size = 0.01
                x_new = x - step_size * grad
                
                # Ensure non-negativity
                x_new = np.maximum(x_new, 0)
                
                # Check constraints
                if np.all(A @ x_new <= b):
                    x = x_new
                else:
                    # Project back to feasible region
                    for i in range(n_constraints):
                        if (A @ x_new)[i] > b[i]:
                            # Scale back
                            alpha = b[i] / (A @ x_new)[i] if (A @ x_new)[i] > 0 else 1
                            x_new = x + alpha * (x_new - x)
                    x = x_new
            
            optimal_value = c @ x
            if objective == 'maximize':
                optimal_value = -optimal_value
            
            # Compute slack variables
            slack = b - A @ x
            
            return {
                'success': True,
                'optimal_solution': x.tolist(),
                'optimal_value': optimal_value,
                'slack_variables': slack.tolist()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def gradient_descent(self, f: Callable, grad_f: Callable, x0: np.ndarray, 
                        tolerance: float = None, max_iterations: int = None) -> OptimizationResult:
        """Simple gradient descent implementation"""
        if tolerance is None:
            tolerance = self.tolerance
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        x = x0.copy()
        history = {'f_vals': [f(x)]}
        
        for iteration in range(max_iterations):
            grad = grad_f(x)
            if np.linalg.norm(grad) < tolerance:
                break
            
            # Simple line search
            alpha = 0.01
            x = x - alpha * grad
            history['f_vals'].append(f(x))
        
        return OptimizationResult(
            x_opt=x,
            f_opt=f(x),
            iterations=iteration + 1,
            converged=np.linalg.norm(grad) < tolerance,
            history=history
        )
    
    def newton_method(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                     tolerance: float = None, max_iterations: int = None) -> OptimizationResult:
        """Simple Newton's method implementation"""
        if tolerance is None:
            tolerance = self.tolerance
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        x = x0.copy()
        history = {'f_vals': [f(x)]}
        
        for iteration in range(max_iterations):
            grad = grad_f(x)
            if np.linalg.norm(grad) < tolerance:
                break
            
            # Approximate Hessian with identity (simplified)
            hess_inv = np.eye(len(x))
            x = x - hess_inv @ grad
            history['f_vals'].append(f(x))
        
        return OptimizationResult(
            x_opt=x,
            f_opt=f(x),
            iterations=iteration + 1,
            converged=np.linalg.norm(grad) < tolerance,
            history=history
        )
    
    def nelder_mead(self, f: Callable, x0: np.ndarray,
                   tolerance: float = None, max_iterations: int = None) -> OptimizationResult:
        """Simple Nelder-Mead implementation"""
        if tolerance is None:
            tolerance = self.tolerance
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        n = len(x0)
        # Create initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            x = x0.copy()
            x[i] += 0.1
            simplex.append(x)
        
        history = {'f_vals': [f(x0)]}
        
        for iteration in range(max_iterations):
            # Evaluate simplex
            values = [f(x) for x in simplex]
            
            # Sort by function value
            indices = np.argsort(values)
            simplex = [simplex[i] for i in indices]
            values = [values[i] for i in indices]
            
            # Check convergence
            if values[-1] - values[0] < tolerance:
                break
            
            # Centroid of best n points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            reflected = centroid + (centroid - simplex[-1])
            f_reflected = f(reflected)
            
            # Check bounds to prevent index errors
            if len(values) < 2:
                break
            
            if values[0] <= f_reflected < values[-2]:
                simplex[-1] = reflected
            elif f_reflected < values[0]:
                # Expansion
                expanded = centroid + 2 * (reflected - centroid)
                if f(expanded) < f_reflected:
                    simplex[-1] = expanded
                else:
                    simplex[-1] = reflected
            else:
                # Contraction
                contracted = centroid + 0.5 * (simplex[-1] - centroid)
                if f(contracted) < values[-1]:
                    simplex[-1] = contracted
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            
            history['f_vals'].append(min(f(x) for x in simplex))
        
        best_x = min(simplex, key=f)
        return OptimizationResult(
            x_opt=best_x,
            f_opt=f(best_x),
            iterations=iteration + 1,
            converged=True,
            history=history
        )

    def genetic_algorithm(self, f: Callable, bounds: List[Tuple[float, float]],
                         pop_size: Optional[int] = None,
                         crossover_prob: float = 0.8,
                         mutation_prob: float = 0.1) -> OptimizationResult:
        """
        Genetic algorithm for global optimization
        
        Args:
            f: Objective function
            bounds: Variable bounds
            pop_size: Population size
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
        """
        if pop_size is None:
            pop_size = self.population_size
        
        n_vars = len(bounds)
        
        # Initialize population
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(pop_size, n_vars)
        )
        
        # Evaluate initial population
        fitness = np.array([f(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = {'best_fitness': [best_fitness], 'avg_fitness': [np.mean(fitness)]}
        
        for generation in range(self.max_iterations):
            # Selection (tournament)
            new_population = []
            
            for _ in range(pop_size):
                # Tournament selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                if fitness[idx1] < fitness[idx2]:
                    parent1 = population[idx1]
                else:
                    parent1 = population[idx2]
                
                idx3, idx4 = np.random.choice(pop_size, 2, replace=False)
                if fitness[idx3] < fitness[idx4]:
                    parent2 = population[idx3]
                else:
                    parent2 = population[idx4]
                
                if np.random.random() < crossover_prob:
                    mask = np.random.random(n_vars) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    child = parent1.copy()
                
                for i in range(n_vars):
                    if np.random.random() < mutation_prob:
                        child[i] += np.random.normal(0, 0.1 * (bounds[i][1] - bounds[i][0]))
                        child[i] = np.clip(child[i], bounds[i][0], bounds[i][1])
                
                new_population.append(child)
            
            population = np.array(new_population)
            fitness = np.array([f(ind) for ind in population])
            
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_individual = population[min_idx].copy()
                best_fitness = fitness[min_idx]
            
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitness))
            
            # Convergence check
            if generation > 100:
                recent_improvement = abs(history['best_fitness'][-1] - 
                                      history['best_fitness'][-100])
                if recent_improvement < self.tolerance:
                    break
        
        return OptimizationResult(
            x_opt=best_individual,
            f_opt=best_fitness,
            iterations=generation + 1,
            converged=True,
            history=history
        )
    
    def particle_swarm_optimization(self, f: Callable, bounds: List[Tuple[float, float]],
                                  swarm_size: Optional[int] = None,
                                  w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> OptimizationResult:
        """
        Particle Swarm Optimization (PSO)
        
        Args:
            f: Objective function
            bounds: Variable bounds
            swarm_size: Number of particles
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        if swarm_size is None:
            swarm_size = self.population_size
        
        n_vars = len(bounds)
        
        # Initialize swarm
        positions = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(swarm_size, n_vars)
        )
        
        velocities = np.random.uniform(
            low=[-0.1 * (b[1] - b[0]) for b in bounds],
            high=[0.1 * (b[1] - b[0]) for b in bounds],
            size=(swarm_size, n_vars)
        )
        
        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([f(p) for p in positions])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        history = {'global_best': [global_best_score]}
        
        for iteration in range(self.max_iterations):
            # Update velocities and positions
            for i in range(swarm_size):
                # Random coefficients
                r1 = np.random.random(n_vars)
                r2 = np.random.random(n_vars)
                
                # Update velocity
                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (personal_best_positions[i] - positions[i]) +
                               c2 * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Apply bounds
                for j in range(n_vars):
                    positions[i, j] = np.clip(positions[i, j], bounds[j][0], bounds[j][1])
                
                # Update personal best
                score = f(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()
            
            history['global_best'].append(global_best_score)
            
            # Convergence check
            if iteration > 50:
                recent_improvement = abs(history['global_best'][-1] - 
                                      history['global_best'][-50])
                if recent_improvement < self.tolerance:
                    break
        
        return OptimizationResult(
            x_opt=global_best_position,
            f_opt=global_best_score,
            iterations=iteration + 1,
            converged=True,
            history=history
        )
    
    def sequential_quadratic_programming(self, f: Callable, grad_f: Callable,
                                       constraints: List[Dict[str, Any]],
                                       x0: np.ndarray) -> OptimizationResult:
        """
        Sequential Quadratic Programming (SQP) for constrained optimization
        
        Args:
            f: Objective function
            grad_f: Gradient of objective
            constraints: List of constraint dicts with 'type', 'fun', 'jac'
            x0: Initial point
        """
        x = x0.copy()
        n = len(x)
        
        # Separate equality and inequality constraints
        eq_constraints = [c for c in constraints if c['type'] == 'eq']
        ineq_constraints = [c for c in constraints if c['type'] == 'ineq']
        
        m_eq = len(eq_constraints)
        m_ineq = len(ineq_constraints)
        
        # Lagrange multipliers
        lambda_eq = np.zeros(m_eq)
        lambda_ineq = np.zeros(m_ineq)
        
        history = {'f_vals': [f(x)], 'constraint_violations': []}
        
        for iteration in range(self.max_iterations):
            # Evaluate constraints and gradients
            g_eq = np.array([c['fun'](x) for c in eq_constraints]) if m_eq > 0 else np.array([])
            g_ineq = np.array([c['fun'](x) for c in ineq_constraints]) if m_ineq > 0 else np.array([])
            
            # Constraint violation
            violation = 0
            if m_eq > 0:
                violation += np.sum(np.abs(g_eq))
            if m_ineq > 0:
                violation += np.sum(np.maximum(0, g_ineq))
            
            history['constraint_violations'].append(violation)
            
            if violation < self.tolerance and iteration > 0:
                # Check KKT conditions
                grad_L = grad_f(x)
                if m_eq > 0:
                    for i, c in enumerate(eq_constraints):
                        grad_L += lambda_eq[i] * c['jac'](x)
                if m_ineq > 0:
                    for i, c in enumerate(ineq_constraints):
                        grad_L += lambda_ineq[i] * c['jac'](x)
                
                if np.linalg.norm(grad_L) < self.tolerance:
                    break
            
            # Set up QP subproblem
            # Approximate Hessian (BFGS update would go here)
            B = np.eye(n)  # Simplified - using identity
            
            # Gradients
            grad_f_x = grad_f(x)
            
            # Linearized constraints
            A_eq = np.array([c['jac'](x) for c in eq_constraints]) if m_eq > 0 else None
            b_eq = -g_eq if m_eq > 0 else None
            
            A_ineq = np.array([c['jac'](x) for c in ineq_constraints]) if m_ineq > 0 else None
            b_ineq = -g_ineq if m_ineq > 0 else None
            
            # Solve QP subproblem (simplified - would use QP solver)
            # min 0.5 * p.T @ B @ p + grad_f.T @ p
            # s.t. A_eq @ p = b_eq, A_ineq @ p <= b_ineq
            
            # Simple gradient step (full SQP would solve QP)
            p = -grad_f_x / np.linalg.norm(grad_f_x)
            
            # Line search
            alpha = 1.0
            merit_param = 10.0  # Penalty parameter
            
            def merit_function(x):
                result = f(x)
                if m_eq > 0:
                    result += merit_param * np.sum(np.abs([c['fun'](x) for c in eq_constraints]))
                if m_ineq > 0:
                    result += merit_param * np.sum(np.maximum(0, [c['fun'](x) for c in ineq_constraints]))
                return result
            
            merit_0 = merit_function(x)
            
            while alpha > 1e-10:
                x_new = x + alpha * p
                if merit_function(x_new) < merit_0:
                    break
                alpha *= 0.5
            
            x = x_new
            history['f_vals'].append(f(x))
        
        return OptimizationResult(
            x_opt=x,
            f_opt=f(x),
            iterations=iteration + 1,
            converged=violation < self.tolerance,
            constraint_violation=violation,
            history=history
        )
    
    def nsga_ii(self, objectives: List[Callable], bounds: List[Tuple[float, float]],
                pop_size: Optional[int] = None) -> OptimizationResult:
        """
        NSGA-II for multi-objective optimization
        
        Args:
            objectives: List of objective functions
            bounds: Variable bounds
            pop_size: Population size
        """
        if pop_size is None:
            pop_size = self.population_size
        
        n_vars = len(bounds)
        n_obj = len(objectives)
        
        # Initialize population
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(pop_size, n_vars)
        )
        
        def evaluate_objectives(x):
            return np.array([obj(x) for obj in objectives])
        
        def non_dominated_sorting(pop, obj_vals):
            """Fast non-dominated sorting"""
            n = len(pop)
            domination_count = np.zeros(n)
            dominated_solutions = [[] for _ in range(n)]
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Check if i dominates j
                    if all(obj_vals[i] <= obj_vals[j]) and any(obj_vals[i] < obj_vals[j]):
                        dominated_solutions[i].append(j)
                    elif all(obj_vals[j] <= obj_vals[i]) and any(obj_vals[j] < obj_vals[i]):
                        domination_count[i] += 1
                
                if domination_count[i] == 0:
                    fronts[0].append(i)
            
            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for sol in fronts[i]:
                    for dominated in dominated_solutions[sol]:
                        domination_count[dominated] -= 1
                        if domination_count[dominated] == 0:
                            next_front.append(dominated)
                i += 1
                fronts.append(next_front)
            
            return fronts[:-1]  # Remove empty last front
        
        def crowding_distance(obj_vals):
            """Calculate crowding distance"""
            n = len(obj_vals)
            distance = np.zeros(n)
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(obj_vals[:, m])
                
                # Boundary points have infinite distance
                distance[sorted_idx[0]] = np.inf
                distance[sorted_idx[-1]] = np.inf
                
                # Calculate distance for others
                obj_range = obj_vals[sorted_idx[-1], m] - obj_vals[sorted_idx[0], m]
                if obj_range > 0:
                    for i in range(1, n-1):
                        distance[sorted_idx[i]] += (
                            obj_vals[sorted_idx[i+1], m] - 
                            obj_vals[sorted_idx[i-1], m]
                        ) / obj_range
            
            return distance
        
        # Main NSGA-II loop
        pareto_front_history = []
        
        for generation in range(self.max_iterations // 10):  # Fewer iterations for multi-obj
            # Evaluate objectives
            obj_vals = np.array([evaluate_objectives(ind) for ind in population])
            
            # Create offspring
            offspring = []
            for _ in range(pop_size):
                # Tournament selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                parent1 = population[idx1]
                parent2 = population[idx2]
                
                # Crossover (SBX)
                child = np.zeros(n_vars)
                for i in range(n_vars):
                    if np.random.random() < 0.5:
                        # SBX crossover
                        beta = np.random.random()
                        child[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    else:
                        child[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]
                
                # Mutation
                for i in range(n_vars):
                    if np.random.random() < 0.1:
                        child[i] += np.random.normal(0, 0.1 * (bounds[i][1] - bounds[i][0]))
                    child[i] = np.clip(child[i], bounds[i][0], bounds[i][1])
                
                offspring.append(child)
            
            offspring = np.array(offspring)
            
            # Combine population and offspring
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.array([evaluate_objectives(ind) for ind in combined_pop])
            
            # Non-dominated sorting
            fronts = non_dominated_sorting(combined_pop, combined_obj)
            
            # Select next population
            new_pop = []
            new_pop_idx = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(combined_pop[front])
                    new_pop_idx.extend(front)
                else:
                    # Use crowding distance for selection
                    remaining = pop_size - len(new_pop)
                    front_obj = combined_obj[front]
                    distances = crowding_distance(front_obj)
                    
                    # Sort by crowding distance
                    sorted_idx = np.argsort(distances)[::-1]
                    selected = [front[i] for i in sorted_idx[:remaining]]
                    
                    new_pop.extend(combined_pop[selected])
                    new_pop_idx.extend(selected)
                    break
            
            population = np.array(new_pop)
            
            # Store Pareto front
            pareto_front = combined_obj[fronts[0]]
            pareto_front_history.append(pareto_front)
        
        # Final Pareto front
        final_obj = np.array([evaluate_objectives(ind) for ind in population])
        fronts = non_dominated_sorting(population, final_obj)
        pareto_front = population[fronts[0]]
        pareto_obj = final_obj[fronts[0]]
        
        return OptimizationResult(
            x_opt=pareto_front[0],  # Return one solution
            f_opt=np.mean(pareto_obj[0]),  # Mean of objectives
            iterations=generation + 1,
            converged=True,
            pareto_front=pareto_front,
            history={'pareto_fronts': pareto_front_history}
        )
    
    def simulated_annealing(self, f: Callable, bounds: List[Tuple[float, float]],
                          x0: Optional[np.ndarray] = None,
                          T0: float = 100.0, cooling_rate: float = 0.95) -> OptimizationResult:
        """
        Simulated Annealing for global optimization
        
        Args:
            f: Objective function
            bounds: Variable bounds
            x0: Initial point (optional)
            T0: Initial temperature
            cooling_rate: Temperature cooling rate
        """
        n_vars = len(bounds)
        
        # Initialize
        if x0 is None:
            x = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds]
            )
        else:
            x = x0.copy()
        
        best_x = x.copy()
        best_f = f(x)
        current_f = best_f
        
        T = T0
        history = {'best_f': [best_f], 'current_f': [current_f], 'temperature': [T]}
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = x.copy()
            
            # Random walk with temperature-dependent step size
            for i in range(n_vars):
                step_size = 0.1 * (bounds[i][1] - bounds[i][0]) * T / T0
                neighbor[i] += np.random.uniform(-step_size, step_size)
                neighbor[i] = np.clip(neighbor[i], bounds[i][0], bounds[i][1])
            
            # Evaluate neighbor
            neighbor_f = f(neighbor)
            
            # Accept or reject
            delta = neighbor_f - current_f
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x = neighbor
                current_f = neighbor_f
                
                if current_f < best_f:
                    best_x = x.copy()
                    best_f = current_f
            
            # Cool down
            T *= cooling_rate
            
            # Record history
            history['best_f'].append(best_f)
            history['current_f'].append(current_f)
            history['temperature'].append(T)
            
            # Stop if frozen
            if T < 1e-10:
                break
        
        return OptimizationResult(
            x_opt=best_x,
            f_opt=best_f,
            iterations=iteration + 1,
            converged=True,
            history=history
        )
    
    def robust_optimization(self, f: Callable, bounds: List[Tuple[float, float]],
                          uncertainty_set: Dict[str, Any],
                          method: str = 'worst_case') -> OptimizationResult:
        """
        Robust optimization under uncertainty
        
        Args:
            f: Objective function f(x, xi) where xi is uncertain parameter
            bounds: Variable bounds
            uncertainty_set: Description of uncertainty
            method: 'worst_case' or 'expected_value'
        """
        n_vars = len(bounds)
        
        if method == 'worst_case':
            # Min-max robust optimization
            def robust_objective(x):
                # Sample uncertainty set
                n_samples = 100
                worst_val = -np.inf
                
                for _ in range(n_samples):
                    # Generate uncertain parameter
                    if uncertainty_set['type'] == 'box':
                        xi = np.random.uniform(
                            uncertainty_set['lower'],
                            uncertainty_set['upper']
                        )
                    elif uncertainty_set['type'] == 'ellipsoid':
                        # Sample from ellipsoid
                        xi = np.random.multivariate_normal(
                            uncertainty_set['center'],
                            uncertainty_set['shape']
                        )
                    
                    val = f(x, xi)
                    worst_val = max(worst_val, val)
                
                return worst_val
            
        else:  # expected_value
            # Stochastic optimization
            def robust_objective(x):
                n_samples = 100
                values = []
                
                for _ in range(n_samples):
                    if uncertainty_set['type'] == 'normal':
                        xi = np.random.normal(
                            uncertainty_set['mean'],
                            uncertainty_set['std']
                        )
                    else:
                        xi = np.random.uniform(
                            uncertainty_set['lower'],
                            uncertainty_set['upper']
                        )
                    
                    values.append(f(x, xi))
                
                return np.mean(values)
        
        # Use genetic algorithm for robust objective
        result = self.genetic_algorithm(robust_objective, bounds)
        
        # Compute uncertainty bounds
        x_opt = result.x_opt
        values = []
        for _ in range(1000):
            if uncertainty_set['type'] == 'normal':
                xi = np.random.normal(uncertainty_set['mean'], uncertainty_set['std'])
            else:
                xi = np.random.uniform(uncertainty_set['lower'], uncertainty_set['upper'])
            values.append(f(x_opt, xi))
        
        uncertainty_bounds = (np.percentile(values, 5), np.percentile(values, 95))
        
        result.uncertainty_bounds = uncertainty_bounds
        return result
    
    def lagrange_multiplier_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lagrange multiplier method for constrained optimization
        
        Args:
            params: Dictionary containing optimization parameters
        """
        try:
            objective_function = params['objective_function']
            variables = params['variables']
            constraints = params.get('constraints', [])
            initial_guess = params['initial_guess']
            tolerance = params.get('tolerance', self.tolerance)
            max_iterations = params.get('max_iterations', self.max_iterations)
            bounds = params.get('bounds', None)
            
            import sympy as sp
            from core.expression_parser import expression_parser
            
            # Parse objective function
            f_expr = expression_parser.parse(objective_function)
            var_symbols = [sp.Symbol(var) for var in variables]
            
            # Parse multiple constraints
            constraint_exprs = []
            for constraint in constraints:
                if constraint['type'] == 'equality':
                    # Handle multiple constraints separated by newlines or semicolons
                    expr_str = constraint['expression']
                    if '\n' in expr_str:
                        constraint_lines = [line.strip() for line in expr_str.split('\n') if line.strip()]
                    elif ';' in expr_str:
                        constraint_lines = [line.strip() for line in expr_str.split(';') if line.strip()]
                    else:
                        constraint_lines = [expr_str.strip()]
                    
                    for line in constraint_lines:
                        constraint_exprs.append(expression_parser.parse(line))
            
            # Create Lagrangian
            # L = f - sum(lambda_i * g_i)
            lambda_symbols = [sp.Symbol(f'lambda_{i}') for i in range(len(constraint_exprs))]
            all_symbols = var_symbols + lambda_symbols
            
            lagrangian = f_expr
            for i, constraint_expr in enumerate(constraint_exprs):
                lagrangian = lagrangian - lambda_symbols[i] * constraint_expr
            
            # Compute gradient of Lagrangian
            lagrangian_grad = [sp.diff(lagrangian, var) for var in all_symbols]
            
            # Add constraint equations
            equations = lagrangian_grad + constraint_exprs
            
            # Try to solve symbolically
            try:
                solutions = sp.solve(equations, all_symbols)
                
                if solutions:
                    if isinstance(solutions, dict):
                        # Single solution
                        var_values = [float(solutions[var]) for var in var_symbols]
                        lambda_values = [float(solutions[lam]) for lam in lambda_symbols]
                        
                        # Evaluate objective at solution
                        obj_value = float(f_expr.subs(dict(zip(var_symbols, var_values))))
                        
                        # Check if bounded constraints are satisfied
                        feasible = True
                        if bounds:
                            for i, (lower, upper) in enumerate(bounds):
                                if not (lower <= var_values[i] <= upper):
                                    feasible = False
                                    break
                        
                        return {
                            'success': True,
                            'optimal_point': var_values,
                            'optimal_value': obj_value,
                            'lagrange_multipliers': lambda_values,
                            'feasible': feasible,
                            'iterations': 1
                        }
                    
                    elif isinstance(solutions, list):
                        # Multiple solutions - return the best feasible one
                        best_solution = None
                        best_value = float('inf')
                        
                        for sol in solutions:
                            if isinstance(sol, dict):
                                var_values = [float(sol[var]) for var in var_symbols if var in sol]
                                if len(var_values) == len(variables):
                                    # Check bounds
                                    feasible = True
                                    if bounds:
                                        for i, (lower, upper) in enumerate(bounds):
                                            if not (lower <= var_values[i] <= upper):
                                                feasible = False
                                                break
                                    
                                    if feasible:
                                        obj_value = float(f_expr.subs(dict(zip(var_symbols, var_values))))
                                        if obj_value < best_value:
                                            best_value = obj_value
                                            best_solution = {
                                                'optimal_point': var_values,
                                                'optimal_value': obj_value,
                                                'lagrange_multipliers': [float(sol[lam]) for lam in lambda_symbols if lam in sol]
                                            }
                        
                        if best_solution:
                            return {
                                'success': True,
                                'feasible': True,
                                'iterations': 1,
                                **best_solution
                            }
                
                # If no symbolic solution found, try numerical approach
                raise UndefinedOperationError("symbolic solution", "No symbolic solution found")
                
            except UndefinedOperationError:
                # Numerical approach using penalty method
                return self._numerical_lagrange_optimization(
                    f_expr, var_symbols, constraint_exprs, initial_guess, bounds, tolerance, max_iterations
                )
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _numerical_lagrange_optimization(self, f_expr, var_symbols, constraint_exprs, 
                                       initial_guess, bounds, tolerance, max_iterations):
        """Numerical Lagrange optimization using penalty method"""
        try:
            import sympy as sp
            
            # Create numerical functions
            f_func = sp.lambdify(var_symbols, f_expr, 'numpy')
            constraint_funcs = [sp.lambdify(var_symbols, expr, 'numpy') for expr in constraint_exprs]
            
            # Penalty method
            def penalized_objective(x):
                obj = float(f_func(*x))
                penalty = 0
                
                # Add penalty for constraint violations
                for constraint_func in constraint_funcs:
                    violation = float(constraint_func(*x))
                    penalty += 1000 * violation**2  # Quadratic penalty
                
                # Add penalty for bound violations
                if bounds:
                    for i, (lower, upper) in enumerate(bounds):
                        if x[i] < lower:
                            penalty += 1000 * (lower - x[i])**2
                        elif x[i] > upper:
                            penalty += 1000 * (x[i] - upper)**2
                
                return obj + penalty
            
            # Use gradient descent
            x = np.array(initial_guess, dtype=float)
            history = []
            
            for iteration in range(max_iterations):
                # Numerical gradient
                grad = np.zeros_like(x)
                h = 1e-8
                f0 = penalized_objective(x)
                
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += h
                    grad[i] = (penalized_objective(x_plus) - f0) / h
                
                if np.linalg.norm(grad) < tolerance:
                    break
                
                # Line search
                alpha = 0.01
                x_new = x - alpha * grad
                
                # Ensure bounds are respected
                if bounds:
                    for i, (lower, upper) in enumerate(bounds):
                        x_new[i] = np.clip(x_new[i], lower, upper)
                
                x = x_new
                history.append(float(f_func(*x)))
            
            # Check constraint satisfaction
            constraint_violations = [abs(float(func(*x))) for func in constraint_funcs]
            max_violation = max(constraint_violations) if constraint_violations else 0
            
            return {
                'success': True,
                'optimal_point': x.tolist(),
                'optimal_value': float(f_func(*x)),
                'iterations': iteration + 1,
                'constraint_violation': max_violation,
                'feasible': max_violation < tolerance,
                'convergence_history': history
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }