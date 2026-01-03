import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.optimization_algorithms_engine import OptimizationAlgorithmsEngine, OptimizationType
from utils.ui_helpers import run_task, copy_button

def render_multivariable_optimization():
    """Render multivariable optimization section"""
    st.subheader("Multivariable Optimization")
    st.markdown("*Find critical points and analyze multivariable functions*")
    
    from core.optimization_algorithms_engine import OptimizationAlgorithmsEngine
    from core.expression_parser import expression_parser
    import sympy as sp
    
    engine = OptimizationAlgorithmsEngine()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2 + y**2 - 2*x*y + 3*x",
            key="multivar_optimization_expr"
        )
    
    with col2:
        variables = st.text_input(
            "Variables:",
            value="x,y",
            key="multivar_optimization_vars"
        )
    
    if expression and variables:
        var_list = [v.strip() for v in variables.split(',') if v.strip()]
        
        if st.button("Find Critical Points", key="find_critical_multivar_opt"):
            try:
                expr = expression_parser.parse(expression)
                
                # Compute gradient components
                gradient_eqs = []
                for var in var_list:
                    partial = sp.diff(expr, sp.Symbol(var))
                    gradient_eqs.append(partial)
                
                st.markdown("**Gradient Equations (set to zero):**")
                for i, eq in enumerate(gradient_eqs):
                    st.markdown(f"∂f/∂{var_list[i]} = `{eq}` = 0")
                
                # Solve the system of gradient equations
                symbols = [sp.Symbol(var) for var in var_list]
                critical_points = sp.solve(gradient_eqs, symbols)
                
                st.markdown("**Critical Points:**")
                if critical_points:
                    if isinstance(critical_points, dict):
                        # Single critical point
                        point_str = ", ".join([f"{var} = {critical_points[sp.Symbol(var)]}" for var in var_list])
                        st.markdown(f"• ({point_str})")
                    elif isinstance(critical_points, list):
                        # Multiple critical points
                        for i, point in enumerate(critical_points):
                            if isinstance(point, dict):
                                point_str = ", ".join([f"{var} = {point[sp.Symbol(var)]}" for var in var_list])
                                st.markdown(f"• Point {i+1}: ({point_str})")
                            else:
                                st.markdown(f"• Point {i+1}: {point}")
                    else:
                        st.markdown(f"• {critical_points}")
                else:
                    st.markdown("No critical points found or system is too complex to solve analytically")
                
                # Hessian matrix for classification
                if len(var_list) == 2:  # For 2D functions, compute Hessian determinant
                    st.markdown("---")
                    st.markdown("**Hessian Matrix Analysis (2D):**")
                    
                    x, y = sp.symbols('x y')
                    fxx = sp.diff(expr, x, 2)
                    fyy = sp.diff(expr, y, 2)
                    fxy = sp.diff(expr, x, y)
                    
                    st.markdown(f"f_xx = `{fxx}`")
                    st.markdown(f"f_yy = `{fyy}`")
                    st.markdown(f"f_xy = `{fxy}`")
                    
                    # Hessian determinant
                    hessian_det = fxx * fyy - fxy**2
                    st.markdown(f"Hessian determinant = `{hessian_det}`")
                    
                    st.markdown("**Classification:** Use the second derivative test:")
                    st.markdown("• If D > 0 and f_xx > 0: Local minimum")
                    st.markdown("• If D > 0 and f_xx < 0: Local maximum")
                    st.markdown("• If D < 0: Saddle point")
                    st.markdown("• If D = 0: Test is inconclusive")
                
            except Exception as e:
                st.error(f"Error in optimization analysis: {str(e)}")

def render_lagrange_optimization():
    """Render Lagrange multiplier optimization section"""
    st.subheader("Lagrange Multiplier Optimization")
    st.markdown("*Constrained optimization using Lagrange multipliers with optional bounds*")
    
    from core.optimization_algorithms_engine import OptimizationAlgorithmsEngine
    
    engine = OptimizationAlgorithmsEngine()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Objective Function**")
        objective = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2 + y**2",
            key="lagrange_objective"
        )
        
        variables = st.text_input(
            "Variables (comma-separated):",
            value="x,y",
            key="lagrange_vars"
        )
        
        initial_guess = st.text_input(
            "Initial guess (comma-separated):",
            value="1,1",
            key="lagrange_initial"
        )
    
    with col2:
        st.markdown("**Constraints**")
        constraint_type = st.selectbox(
            "Constraint type:",
            ["equality"],
            key="lagrange_constraint_type"
        )
        
        constraint_expr = st.text_input(
            "Constraint g(x,y,z,...) = 0:",
            placeholder="x**2 + y**2 - 1",
            key="lagrange_constraint"
        )
        
        # Bounds
        use_bounds = st.checkbox("Add variable bounds", key="lagrange_use_bounds")
        
        if use_bounds:
            bounds_input = st.text_input(
                "Bounds (lower,upper; lower,upper; ...):",
                placeholder="-10,10; -10,10",
                key="lagrange_bounds"
            )
    
    if objective and variables and constraint_expr and initial_guess:
        if st.button("Solve with Lagrange Multipliers", key="solve_lagrange"):
            try:
                # Parse inputs
                var_list = [v.strip() for v in variables.split(',') if v.strip()]
                initial_vals = [float(x.strip()) for x in initial_guess.split(',')]
                
                # Parse bounds if provided
                bounds = None
                if use_bounds and bounds_input:
                    bound_pairs = []
                    for bound_str in bounds_input.split(';'):
                        lower, upper = bound_str.split(',')
                        bound_pairs.append((float(lower.strip()), float(upper.strip())))
                    bounds = bound_pairs
                
                # Prepare parameters
                params = {
                    'objective_function': objective,
                    'variables': var_list,
                    'constraints': [{
                        'type': 'equality',
                        'expression': constraint_expr
                    }],
                    'initial_guess': initial_vals,
                    'bounds': bounds
                }
                
                # Solve
                result, elapsed, error = run_task(
                    "Optimizing with Lagrange multipliers...",
                    engine.lagrange_multiplier_optimization,
                    params,
                    success_message="Optimization complete",
                    error_message="Optimization failed"
                )
                if error:
                    return
                
                if result['success']:
                    st.success(f"Lagrange optimization completed in {elapsed:.2f}s")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Optimal Point:**")
                        for i, var in enumerate(var_list):
                            st.write(f"{var} = {result['optimal_point'][i]:.6f}")
                        
                        st.markdown(f"**Optimal Value:** {result['optimal_value']:.6f}")
                        copy_button(str(result['optimal_value']), key="lagrange-optimal-value")
                        
                        if 'constraint_violation' in result:
                            st.info(f"Constraint violation: {result['constraint_violation']:.2e}")
                    
                    with col2:
                        if 'lagrange_multipliers' in result:
                            st.markdown("**Lagrange Multipliers:**")
                            for i, lam in enumerate(result['lagrange_multipliers']):
                                st.write(f"λ_{i+1} = {lam:.6f}")
                        
                        st.info(f"Iterations: {result['iterations']}")
                        st.info(f"Feasible: {result.get('feasible', 'Unknown')}")
                    
                    # Show convergence if available
                    if 'convergence_history' in result and result['convergence_history']:
                        st.markdown("**Convergence History:**")
                        history = result['convergence_history']
                        for i, val in enumerate(history[-5:]):  # Show last 5
                            st.write(f"Iter {len(history)-5+i+1}: {val:.6f}")
                
                else:
                    st.error(f"Lagrange optimization failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Please fill in all required fields (objective, variables, constraint, initial guess)")

def render_optimization_panel():
    """Render the optimization algorithms interface"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Advanced Optimization Algorithms</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Powerful optimization techniques for complex problems</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "Multivariable Optimization",
        "Lagrange Optimization", 
        "Gradient-Based",
        "Evolutionary",
        "Multi-Objective",
        "Constrained",
        "Robust/Stochastic"
    ])
    
    with tabs[0]:
        render_multivariable_optimization()
    
    with tabs[1]:
        render_lagrange_optimization()
    
    with tabs[2]:
        render_gradient_based()
    
    with tabs[3]:
        render_evolutionary()
        
    with tabs[4]:
        render_multi_objective()
        
    with tabs[5]:
        render_constrained()
        
    with tabs[6]:
        render_robust_stochastic()

def render_gradient_based():
    """Gradient-based optimization methods"""
    st.subheader("Gradient-Based Optimization")
    
    method = st.selectbox(
        "Select method",
        ["L-BFGS-B", "Conjugate Gradient", "Newton's Method"]
    )
    
    # Test function selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Test Function**")
        func_type = st.selectbox(
            "Function",
            ["Rosenbrock", "Sphere", "Rastrigin", "Ackley", "Custom"]
        )
        
        dim = st.slider("Dimensions", 2, 20, 2)
        
        # Initial point
        st.write("**Initial Point**")
        init_type = st.selectbox("Initialization", ["Random", "Fixed", "Custom"])
        
        if init_type == "Fixed":
            x0 = np.ones(dim) * 2.0
        elif init_type == "Random":
            x0 = np.random.randn(dim) * 2
        else:
            x0 = []
            for i in range(min(dim, 5)):  # Show only first 5 for large dims
                x0.append(st.number_input(f"x₀[{i}]", value=1.0, key=f"grad_x0_{i}"))
            if dim > 5:
                x0.extend([1.0] * (dim - 5))
            x0 = np.array(x0)
    
    with col2:
        st.write("**Algorithm Settings**")
        
        if method == "L-BFGS-B":
            m = st.slider("Memory size (m)", 3, 20, 10)
            use_bounds = st.checkbox("Use bounds")
            if use_bounds:
                lower_bound = st.number_input("Lower bound", value=-5.0)
                upper_bound = st.number_input("Upper bound", value=5.0)
                bounds = [(lower_bound, upper_bound)] * dim
            else:
                bounds = None
        
        max_iter = st.number_input("Max iterations", 100, 5000, 1000)
        tol = st.number_input("Tolerance", value=1e-8, format="%.2e")
    
    # Define test functions
    if func_type == "Rosenbrock":
        def f(x):
            return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))
        def grad_f(x):
            g = np.zeros_like(x)
            for i in range(len(x)-1):
                g[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1-x[i])
                g[i+1] += 200*(x[i+1] - x[i]**2)
            return g
        optimal_val = 0.0
        
    elif func_type == "Sphere":
        def f(x):
            return np.sum(x**2)
        def grad_f(x):
            return 2*x
        optimal_val = 0.0
        
    elif func_type == "Rastrigin":
        def f(x):
            return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        def grad_f(x):
            return 2*x + 20*np.pi*np.sin(2*np.pi*x)
        optimal_val = 0.0
        
    elif func_type == "Ackley":
        def f(x):
            a, b, c = 20, 0.2, 2*np.pi
            n = len(x)
            return -a*np.exp(-b*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(c*x))/n) + a + np.e
        def grad_f(x):
            a, b, c = 20, 0.2, 2*np.pi
            n = len(x)
            term1 = np.sqrt(np.sum(x**2)/n)
            safe_term1 = term1 if term1 > 1e-12 else 1e-12  # avoid division by zero at optimum
            g1 = a*b*np.exp(-b*safe_term1) * x / (n*safe_term1)
            g2 = c*np.sin(c*x) * np.exp(np.sum(np.cos(c*x))/n) / n
            return g1 + g2
        optimal_val = 0.0
    else:
        st.warning("Please select a predefined function")
        return
    
    if st.button("Optimize"):
        try:
            engine = OptimizationAlgorithmsEngine()
            engine.max_iterations = max_iter
            engine.tolerance = tol
            
            if method == "L-BFGS-B":
                result, elapsed, error = run_task(
                    "Optimizing...",
                    engine.lbfgs_b,
                    f, grad_f, x0, bounds, m,
                    success_message="Optimization complete",
                    error_message="Optimization failed"
                )
                if error:
                    return
            else:
                st.error("Method not yet implemented")
                return
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Results (in {elapsed:.2f}s)**")
                st.write(f"Optimal value: {result.f_opt:.6f}")
                copy_button(str(result.f_opt), key="opt-value")
                st.write(f"True optimal: {optimal_val:.6f}")
                st.write(f"Error: {abs(result.f_opt - optimal_val):.2e}")
                st.write(f"Iterations: {result.iterations}")
                st.write(f"Converged: {result.converged}")
                
                if dim <= 10:
                    st.write("**Optimal point:**")
                    for i in range(min(dim, 5)):
                        st.write(f"x[{i}] = {result.x_opt[i]:.6f}")
                    if dim > 5:
                        st.write("...")
            
            with col2:
                # Convergence plot
                if result.history:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                    
                    # Function values
                    ax1.semilogy(result.history['f_vals'])
                    ax1.set_xlabel('Iteration')
                    ax1.set_ylabel('f(x)')
                    ax1.set_title('Function Value Convergence')
                    ax1.grid(True, alpha=0.3)
                    
                    # Gradient norms
                    ax2.semilogy(result.history['grad_norms'])
                    ax2.set_xlabel('Iteration')
                    ax2.set_ylabel('||∇f||')
                    ax2.set_title('Gradient Norm Convergence')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # 2D visualization for 2D problems
            if dim == 2 and st.checkbox("Show 2D landscape"):
                x_range = result.x_opt[0] + np.linspace(-2, 2, 100)
                y_range = result.x_opt[1] + np.linspace(-2, 2, 100)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)
                
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
                
                # Plot optimization path
                if result.history and len(result.history['f_vals']) > 1:
                    # Would need to store x history for this
                    pass
                
                ax.scatter([result.x_opt[0]], [result.x_opt[1]], [result.f_opt], 
                          color='red', s=100, label='Optimum')
                ax.set_xlabel('x₁')
                ax.set_ylabel('x₂')
                ax.set_zlabel('f(x)')
                ax.set_title(f'{func_type} Function Landscape')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")

def render_evolutionary():
    """Evolutionary optimization algorithms"""
    st.subheader("Evolutionary Algorithms")
    
    method = st.selectbox(
        "Select algorithm",
        ["Genetic Algorithm", "Particle Swarm", "Differential Evolution", "Simulated Annealing"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Problem Setup**")
        
        # Function selection
        func_type = st.selectbox(
            "Test function",
            ["Schwefel", "Griewank", "Levy", "Michalewicz", "Custom"],
            key="evo_func"
        )
        
        dim = st.slider("Dimensions", 2, 30, 5, key="evo_dim")
        
        # Bounds
        st.write("**Variable Bounds**")
        bound_type = st.selectbox("Bound type", ["Symmetric", "Custom"])
        if bound_type == "Symmetric":
            bound = st.number_input("Bound magnitude", value=10.0)
            bounds = [(-bound, bound)] * dim
        else:
            bounds = [(-10, 10)] * dim  # Simplified
    
    with col2:
        st.write("**Algorithm Parameters**")
        
        if method == "Genetic Algorithm":
            pop_size = st.slider("Population size", 20, 200, 50)
            crossover_prob = st.slider("Crossover probability", 0.5, 1.0, 0.8)
            mutation_prob = st.slider("Mutation probability", 0.01, 0.3, 0.1)
            
        elif method == "Particle Swarm":
            swarm_size = st.slider("Swarm size", 20, 200, 50)
            w = st.slider("Inertia weight", 0.4, 0.9, 0.7)
            c1 = st.slider("Cognitive parameter", 1.0, 2.5, 1.5)
            c2 = st.slider("Social parameter", 1.0, 2.5, 1.5)
            
        elif method == "Simulated Annealing":
            T0 = st.number_input("Initial temperature", value=100.0)
            cooling_rate = st.slider("Cooling rate", 0.8, 0.99, 0.95)
        
        max_iter = st.number_input("Max iterations", 100, 5000, 1000, key="evo_iter")
    
    # Define test functions
    if func_type == "Schwefel":
        def f(x):
            return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        optimal_val = 0.0
        
    elif func_type == "Griewank":
        def f(x):
            sum_sq = np.sum(x**2) / 4000
            prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
            return sum_sq - prod_cos + 1
        optimal_val = 0.0
        
    elif func_type == "Levy":
        def f(x):
            w = 1 + (x - 1) / 4
            term1 = np.sin(np.pi * w[0])**2
            term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
            wi = w[:-1]
            sum_term = np.sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2))
            return term1 + sum_term + term3
        optimal_val = 0.0
        
    elif func_type == "Michalewicz":
        def f(x):
            m = 10
            return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x**2) / np.pi)**(2*m))
        optimal_val = -1.8013 if dim == 2 else None  # Depends on dimension
    else:
        st.warning("Please select a predefined function")
        return
    
    if st.button("Run Optimization", key="evo_run"):
        try:
            engine = OptimizationAlgorithmsEngine()
            engine.max_iterations = max_iter
            
            if method == "Genetic Algorithm":
                result, elapsed, error = run_task(
                    f"Running {method}...",
                    engine.genetic_algorithm,
                    f, bounds, pop_size, crossover_prob, mutation_prob,
                    success_message="Optimization complete",
                    error_message="Optimization failed"
                )
            elif method == "Particle Swarm":
                result, elapsed, error = run_task(
                    f"Running {method}...",
                    engine.particle_swarm_optimization,
                    f, bounds, swarm_size, w, c1, c2,
                    success_message="Optimization complete",
                    error_message="Optimization failed"
                )
            elif method == "Simulated Annealing":
                result, elapsed, error = run_task(
                    f"Running {method}...",
                    engine.simulated_annealing,
                    f, bounds, None, T0, cooling_rate,
                    success_message="Optimization complete",
                    error_message="Optimization failed"
                )
            else:
                st.error("Method not implemented")
                return
                
            if error:
                return
            
            # Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Results (in {elapsed:.2f}s)**")
                st.write(f"Best value found: {result.f_opt:.6f}")
                copy_button(str(result.f_opt), key="evo-value")
                if optimal_val is not None:
                    st.write(f"Known optimal: {optimal_val:.6f}")
                    st.write(f"Error: {abs(result.f_opt - optimal_val):.2e}")
                st.write(f"Iterations: {result.iterations}")
                
                if dim <= 10:
                    st.write("**Best solution:**")
                    for i in range(min(dim, 5)):
                        st.write(f"x[{i}] = {result.x_opt[i]:.6f}")
                    if dim > 5:
                        st.write("...")
            
            with col2:
                # Convergence plot
                if result.history:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if 'best_fitness' in result.history:
                        ax.plot(result.history['best_fitness'], label='Best')
                    if 'avg_fitness' in result.history:
                        ax.plot(result.history['avg_fitness'], label='Average', alpha=0.5)
                    if 'global_best' in result.history:
                        ax.plot(result.history['global_best'], label='Global Best')
                    if 'temperature' in result.history and method == "Simulated Annealing":
                        ax2 = ax.twinx()
                        ax2.plot(result.history['temperature'], 'r--', alpha=0.5, label='Temperature')
                        ax2.set_ylabel('Temperature')
                        ax2.legend(loc='upper right')
                    
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Objective Value')
                    ax.set_title(f'{method} Convergence')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")

def render_multi_objective():
    """Multi-objective optimization"""
    st.subheader("Multi-Objective Optimization")
    
    st.write("**NSGA-II Algorithm**")
    
    # Problem selection
    problem = st.selectbox(
        "Test problem",
        ["ZDT1", "ZDT2", "ZDT3", "DTLZ1", "Custom"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if problem == "ZDT1":
            st.write("**ZDT1 Problem**")
            st.latex(r"f_1(x) = x_1")
            st.latex(r"f_2(x) = g(x)[1 - \sqrt{x_1/g(x)}]")
            st.latex(r"g(x) = 1 + 9\sum_{i=2}^n x_i/(n-1)")
            
            n_vars = st.slider("Number of variables", 2, 30, 10)
            bounds = [(0, 1)] * n_vars
            
            def f1(x):
                return x[0]
            def f2(x):
                g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
                return g * (1 - np.sqrt(x[0] / g))
            
            objectives = [f1, f2]
            
        elif problem == "ZDT2":
            st.write("**ZDT2 Problem**")
            st.latex(r"f_1(x) = x_1")
            st.latex(r"f_2(x) = g(x)[1 - (x_1/g(x))^2]")
            
            n_vars = st.slider("Number of variables", 2, 30, 10)
            bounds = [(0, 1)] * n_vars
            
            def f1(x):
                return x[0]
            def f2(x):
                g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
                return g * (1 - (x[0] / g)**2)
            
            objectives = [f1, f2]
        else:
            st.info("Other problems not yet implemented")
            return
    
    with col2:
        st.write("**Algorithm Settings**")
        pop_size = st.slider("Population size", 50, 300, 100)
        n_generations = st.slider("Generations", 50, 500, 200)
    
    if st.button("Run NSGA-II"):
        try:
            engine = OptimizationAlgorithmsEngine()
            engine.population_size = pop_size
            engine.max_iterations = n_generations
            
            with st.spinner("Running NSGA-II..."):
                result = engine.nsga_ii(objectives, bounds, pop_size)
            
            # Plot Pareto front
            if result.pareto_front is not None:
                # Evaluate objectives for Pareto front
                pareto_obj = np.array([
                    [obj(x) for obj in objectives] 
                    for x in result.pareto_front
                ])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(pareto_obj[:, 0], pareto_obj[:, 1], alpha=0.6)
                ax.set_xlabel('f₁(x)')
                ax.set_ylabel('f₂(x)')
                ax.set_title(f'Pareto Front - {problem}')
                ax.grid(True, alpha=0.3)
                
                # True Pareto front for ZDT problems
                if problem in ["ZDT1", "ZDT2"]:
                    x1_true = np.linspace(0, 1, 100)
                    if problem == "ZDT1":
                        f2_true = 1 - np.sqrt(x1_true)
                    else:
                        f2_true = 1 - x1_true**2
                    ax.plot(x1_true, f2_true, 'r--', label='True Pareto front')
                    ax.legend()
                
                st.pyplot(fig)
                
                st.write(f"**Results**")
                st.write(f"Pareto front size: {len(result.pareto_front)}")
                st.write(f"Iterations: {result.iterations}")
                
                # Download Pareto front
                if st.checkbox("Show Pareto solutions"):
                    st.dataframe({
                        'Solution': [f"x{i}" for i in range(len(result.pareto_front))],
                        'f₁': pareto_obj[:, 0],
                        'f₂': pareto_obj[:, 1]
                    })
                    
        except Exception as e:
            st.error(f"Multi-objective optimization error: {str(e)}")

def render_constrained():
    """Constrained optimization"""
    st.subheader("Constrained Optimization")
    
    st.write("**Sequential Quadratic Programming (SQP)**")
    
    # Problem type
    problem = st.selectbox(
        "Test problem",
        ["Simple constrained", "Engineering design", "Custom"]
    )
    
    if problem == "Simple constrained":
        st.write("**Problem:**")
        st.latex(r"\min_{x,y} \quad (x-2)^2 + (y-1)^2")
        st.latex(r"\text{s.t.} \quad x^2 + y^2 \leq 1")
        st.latex(r"\quad\quad\quad x + y = 1")
        
        def f(x):
            return (x[0] - 2)**2 + (x[1] - 1)**2
        
        def grad_f(x):
            return np.array([2*(x[0] - 2), 2*(x[1] - 1)])
        
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: 1 - x[0]**2 - x[1]**2,
                'jac': lambda x: np.array([-2*x[0], -2*x[1]])
            },
            {
                'type': 'eq',
                'fun': lambda x: x[0] + x[1] - 1,
                'jac': lambda x: np.array([1, 1])
            }
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            x0_1 = st.number_input("Initial x", value=0.5)
            x0_2 = st.number_input("Initial y", value=0.5)
            x0 = np.array([x0_1, x0_2])
        
        with col2:
            max_iter = st.number_input("Max iterations", 50, 500, 100)
        
        if st.button("Solve"):
            try:
                engine = OptimizationAlgorithmsEngine()
                engine.max_iterations = max_iter
                
                with st.spinner("Solving constrained problem..."):
                    result = engine.sequential_quadratic_programming(
                        f, grad_f, constraints, x0
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Results**")
                    st.write(f"Optimal point: ({result.x_opt[0]:.4f}, {result.x_opt[1]:.4f})")
                    st.write(f"Optimal value: {result.f_opt:.4f}")
                    st.write(f"Constraint violation: {result.constraint_violation:.2e}")
                    st.write(f"Iterations: {result.iterations}")
                    st.write(f"Converged: {result.converged}")
                
                with col2:
                    # Visualization
                    x = np.linspace(-1.5, 2.5, 100)
                    y = np.linspace(-1.5, 2.5, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = (X - 2)**2 + (Y - 1)**2
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Objective contours
                    contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
                    ax.clabel(contour, inline=True, fontsize=8)
                    
                    # Constraints
                    # Inequality constraint
                    circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle)
                    
                    # Equality constraint
                    x_eq = np.linspace(-1, 2, 100)
                    y_eq = 1 - x_eq
                    ax.plot(x_eq, y_eq, 'g-', linewidth=2, label='x + y = 1')
                    
                    # Solution
                    ax.plot(result.x_opt[0], result.x_opt[1], 'ro', markersize=10, 
                           label=f'Solution: ({result.x_opt[0]:.3f}, {result.x_opt[1]:.3f})')
                    
                    # Initial point
                    ax.plot(x0[0], x0[1], 'bo', markersize=8, label='Initial point')
                    
                    ax.set_xlim(-1.5, 2.5)
                    ax.set_ylim(-1.5, 2.5)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title('Constrained Optimization')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal')
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Constrained optimization error: {str(e)}")
    else:
        st.info("Other constrained problems coming soon!")

def render_robust_stochastic():
    """Robust and stochastic optimization"""
    st.subheader("Robust Optimization")
    
    st.write("**Optimization under uncertainty**")
    
    method = st.selectbox(
        "Method",
        ["Worst-case robust", "Expected value", "Chance-constrained"]
    )
    
    # Simple robust optimization example
    st.write("**Example: Portfolio optimization with uncertain returns**")
    st.latex(r"\min_x \quad f(x, \xi) = -\sum_i \xi_i x_i + \lambda \|x\|^2")
    st.latex(r"\text{where } \xi \text{ are uncertain returns}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_assets = st.slider("Number of assets", 2, 10, 4)
        
        # Uncertainty set
        uncertainty_type = st.selectbox(
            "Uncertainty type",
            ["Box uncertainty", "Ellipsoidal", "Normal distribution"]
        )
        
        if uncertainty_type == "Box uncertainty":
            uncertainty_level = st.slider("Uncertainty level (%)", 10, 50, 20)
            nominal_returns = np.random.randn(n_assets) * 0.1 + 0.05
            
            uncertainty_set = {
                'type': 'box',
                'lower': nominal_returns * (1 - uncertainty_level/100),
                'upper': nominal_returns * (1 + uncertainty_level/100)
            }
        elif uncertainty_type == "Normal distribution":
            std_dev = st.slider("Standard deviation", 0.01, 0.1, 0.03)
            nominal_returns = np.random.randn(n_assets) * 0.1 + 0.05
            
            uncertainty_set = {
                'type': 'normal',
                'mean': nominal_returns,
                'std': std_dev
            }
        
        lambda_reg = st.slider("Regularization λ", 0.0, 1.0, 0.1)
    
    with col2:
        st.write("**Nominal returns:**")
        for i in range(min(n_assets, 5)):
            st.write(f"Asset {i+1}: {nominal_returns[i]:.3f}")
        if n_assets > 5:
            st.write("...")
    
    # Bounds for portfolio weights
    bounds = [(0, 1)] * n_assets
    
    # Objective function
    def f(x, xi):
        return -np.dot(xi, x) + lambda_reg * np.dot(x, x)
    
    if st.button("Solve Robust Problem"):
        try:
            engine = OptimizationAlgorithmsEngine()
            
            with st.spinner("Solving robust optimization..."):
                result = engine.robust_optimization(
                    f, bounds, uncertainty_set, 
                    method='worst_case' if method == "Worst-case robust" else 'expected_value'
                )
            
            st.write("**Results**")
            st.write("Optimal portfolio weights:")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Portfolio weights
            ax1.bar(range(n_assets), result.x_opt)
            ax1.set_xlabel('Asset')
            ax1.set_ylabel('Weight')
            ax1.set_title('Robust Portfolio Allocation')
            ax1.set_xticks(range(n_assets))
            ax1.set_xticklabels([f'A{i+1}' for i in range(n_assets)])
            
            # Uncertainty bounds on objective
            if result.uncertainty_bounds:
                ax2.hist(np.random.normal(
                    result.f_opt, 
                    (result.uncertainty_bounds[1] - result.uncertainty_bounds[0])/4,
                    1000
                ), bins=50, alpha=0.7, density=True)
                ax2.axvline(result.uncertainty_bounds[0], color='r', linestyle='--', 
                           label=f'5% VaR: {result.uncertainty_bounds[0]:.3f}')
                ax2.axvline(result.uncertainty_bounds[1], color='r', linestyle='--',
                           label=f'95% VaR: {result.uncertainty_bounds[1]:.3f}')
                ax2.axvline(result.f_opt, color='g', linewidth=2,
                           label=f'Expected: {result.f_opt:.3f}')
                ax2.set_xlabel('Portfolio Return')
                ax2.set_ylabel('Probability Density')
                ax2.set_title('Return Distribution')
                ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Expected return: {-result.f_opt:.3f}")
                st.write(f"Sum of weights: {np.sum(result.x_opt):.3f}")
            with col2:
                if result.uncertainty_bounds:
                    st.write(f"5% VaR: {result.uncertainty_bounds[0]:.3f}")
                    st.write(f"95% VaR: {result.uncertainty_bounds[1]:.3f}")
                    
        except Exception as e:
            st.error(f"Robust optimization error: {str(e)}")