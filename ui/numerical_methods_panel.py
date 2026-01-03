import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent.parent
sys.path.extend([str(current_dir / "core"), str(current_dir / "utils")])

from core.numerical_methods_engine import NumericalMethodsEngine
from core.optimization_algorithms_engine import OptimizationAlgorithmsEngine
from utils.ui_helpers import run_task, copy_button


def render_numerical_methods_panel():
    """Main numerical methods panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Numerical Methods & Algorithms</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Advanced numerical computation and optimization techniques</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different categories
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Root Finding", "Integration", "ODE Solving", "Optimization"])

    with tab1:
        render_root_finding()

    with tab2:
        render_numerical_integration()

    with tab3:
        render_ode_solving()

    with tab4:
        render_optimization_methods()


def render_root_finding():
    """Root finding methods"""
    st.subheader("Root Finding Methods")

    engine = NumericalMethodsEngine()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Function Setup")

        # Function input
        function = st.text_input("Function f(x) = 0",
                                 value="x^3 - 2*x - 5",
                                 key="root_function")

        # Method selection
        method = st.selectbox(
            "Method", ["bisection", "newton_raphson", "secant", "fixed_point"],
            key="root_method")

        # Method-specific parameters
        if method == "bisection":
            a = st.number_input("Left endpoint (a)",
                                value=-10.0,
                                key="root_bisection_a")
            b = st.number_input("Right endpoint (b)",
                                value=10.0,
                                key="root_bisection_b")
            tolerance = st.number_input("Tolerance",
                                        value=1e-6,
                                        format="%.2e",
                                        key="root_bisection_tol")
            max_iter = st.number_input("Max iterations",
                                       value=100,
                                       min_value=1,
                                       key="root_bisection_iter")

        elif method == "newton_raphson":
            x0 = st.number_input("Initial guess",
                                 value=1.0,
                                 key="root_newton_x0")
            tolerance = st.number_input("Tolerance",
                                        value=1e-6,
                                        format="%.2e",
                                        key="root_newton_tol")
            max_iter = st.number_input("Max iterations",
                                       value=100,
                                       min_value=1,
                                       key="root_newton_iter")

        elif method == "secant":
            x0 = st.number_input("First guess",
                                 value=0.0,
                                 key="root_secant_x0")
            x1 = st.number_input("Second guess",
                                 value=1.0,
                                 key="root_secant_x1")
            tolerance = st.number_input("Tolerance",
                                        value=1e-6,
                                        format="%.2e",
                                        key="root_secant_tol")
            max_iter = st.number_input("Max iterations",
                                       value=100,
                                       min_value=1,
                                       key="root_secant_iter")

        elif method == "fixed_point":
            x0 = st.number_input("Initial guess",
                                 value=1.0,
                                 key="root_fixed_x0")
            tolerance = st.number_input("Tolerance",
                                        value=1e-6,
                                        format="%.2e",
                                        key="root_fixed_tol")
            max_iter = st.number_input("Max iterations",
                                       value=100,
                                       min_value=1,
                                       key="root_fixed_iter")

    with col2:
        st.markdown("### Results")

        if st.button("Find Root", type="primary", key="find_root_btn"):
            try:
                params = {'tolerance': tolerance, 'max_iterations': max_iter}

                if method == "bisection":
                    params.update({'a': a, 'b': b})
                elif method == "newton_raphson":
                    params.update({'x0': x0})
                elif method == "secant":
                    params.update({'x0': x0, 'x1': x1})
                elif method == "fixed_point":
                    params.update({'x0': x0})

                def _solve_root():
                    return engine.find_root(function, method, params)

                result, elapsed, error = run_task(
                    "Solving for root...",
                    _solve_root,
                    success_message="Root found",
                    error_message="Root finding failed"
                )
                if error:
                    return

                if result['success']:
                    st.success(f"Root: {result['root']:.8f}")
                    st.info(f"Iterations: {result['iterations']} (completed in {elapsed:.2f}s)")
                    st.info(
                        f"Function value: f({result['root']:.8f}) = {result['function_value']:.2e}"
                    )
                    copy_button(str(result['root']), key="root-result")

                    if 'convergence_data' in result:
                        st.markdown("### Convergence History")
                        convergence = result['convergence_data']
                        for i, x in enumerate(convergence):
                            st.write(f"Iteration {i+1}: x = {x:.8f}")
                else:
                    st.error(f"Failed to find root: {result['error']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_numerical_integration():
    """Numerical integration methods"""
    st.subheader("Numerical Integration")

    engine = NumericalMethodsEngine()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Integration Setup")

        # Function input
        function = st.text_input("Function f(x)",
                                 value="sin(x)",
                                 key="integration_function")

        # Integration bounds
        a = st.number_input("Lower bound (a)",
                            value=0.0,
                            key="integration_lower")
        b = st.number_input("Upper bound (b)",
                            value=3.14159,
                            key="integration_upper")

        # Method selection
        method = st.selectbox(
            "Method", ["trapezoidal", "simpson", "gaussian", "romberg"],
            key="integration_method")

        # Number of intervals/points
        if method in ["trapezoidal", "simpson"]:
            n = st.number_input("Number of intervals",
                                value=100,
                                min_value=2,
                                step=2,
                                key="integration_intervals")
        elif method == "gaussian":
            n = st.number_input("Number of points",
                                value=10,
                                min_value=2,
                                key="integration_gauss_points")
        elif method == "romberg":
            n = st.number_input("Number of levels",
                                value=6,
                                min_value=2,
                                key="integration_romberg_levels")

    with col2:
        st.markdown("### Results")

        if st.button("Integrate", type="primary", key="integrate_btn"):
            try:
                params = {'a': a, 'b': b, 'n': n}

                result, elapsed, error = run_task(
                    "Integrating...",
                    engine.numerical_integration,
                    function,
                    method,
                    params,
                    success_message="Integration complete",
                    error_message="Integration failed"
                )
                if error:
                    return

                if result['success']:
                    st.success(f"Integral value: {result['integral']:.8f} (in {elapsed:.2f}s)")
                    copy_button(str(result['integral']), key="integration-result")

                    if 'error_estimate' in result:
                        st.info(
                            f"Error estimate: {result['error_estimate']:.2e}")

                    if 'convergence_table' in result:
                        st.markdown("### Convergence Table")
                        table = result['convergence_table']
                        st.dataframe(table)
                else:
                    st.error(f"Integration failed: {result['error']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_ode_solving():
    """ODE solving methods"""
    st.subheader("Ordinary Differential Equations")

    engine = NumericalMethodsEngine()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ODE Setup")

        # ODE type
        ode_type = st.selectbox(
            "ODE Type", ["First-order IVP", "System of ODEs", "Second-order"],
            key="ode_type")

        if ode_type == "First-order IVP":
            # dy/dx = f(x, y)
            function = st.text_input("dy/dx = f(x,y)",
                                     value="x - y",
                                     key="ode_first_order")
            y0 = st.number_input("Initial condition y(x0)",
                                 value=1.0,
                                 key="ode_y0")

        elif ode_type == "System of ODEs":
            st.markdown("System: dy/dx = f(x,y), dz/dx = g(x,y,z)")
            f_expr = st.text_input("f(x,y,z)", value="z", key="ode_system_f")
            g_expr = st.text_input("g(x,y,z)", value="-y", key="ode_system_g")
            y0 = st.number_input("Initial y(x0)",
                                 value=1.0,
                                 key="ode_system_y0")
            z0 = st.number_input("Initial z(x0)",
                                 value=0.0,
                                 key="ode_system_z0")

        elif ode_type == "Second-order":
            st.markdown("d²y/dx² + p(x)dy/dx + q(x)y = r(x)")
            p_expr = st.text_input("p(x)", value="0", key="ode_second_p")
            q_expr = st.text_input("q(x)", value="1", key="ode_second_q")
            r_expr = st.text_input("r(x)", value="0", key="ode_second_r")
            y0 = st.number_input("Initial y(x0)",
                                 value=1.0,
                                 key="ode_second_y0")
            dy0 = st.number_input("Initial y'(x0)",
                                  value=0.0,
                                  key="ode_second_dy0")

        # Common parameters
        x0 = st.number_input("Initial x", value=0.0, key="ode_x0")
        xf = st.number_input("Final x", value=10.0, key="ode_xf")
        h = st.number_input("Step size", value=0.1, key="ode_step_size")

        # Method selection
        method = st.selectbox(
            "Method",
            ["euler", "runge_kutta_4", "adams_bashforth", "backward_euler"],
            key="ode_method")

    with col2:
        st.markdown("### Results")

        if st.button("Solve ODE", type="primary", key="solve_ode_btn"):
            try:
                params = {'x0': x0, 'xf': xf, 'h': h, 'method': method}

                def _solve_ode():
                    if ode_type == "First-order IVP":
                        params.update({'function': function, 'y0': y0})
                        return engine.solve_ode(
                            ode_type.lower().replace(' ', '_'), params)

                    if ode_type == "System of ODEs":
                        params.update({
                            'functions': [f_expr, g_expr],
                            'initial_conditions': [y0, z0]
                        })
                        return engine.solve_ode_system(params)

                    params.update({
                        'p': p_expr,
                        'q': q_expr,
                        'r': r_expr,
                        'y0': y0,
                        'dy0': dy0
                    })
                    return engine.solve_second_order_ode(params)

                result, elapsed, error = run_task(
                    "Solving ODE...",
                    _solve_ode,
                    success_message="ODE solved",
                    error_message="ODE solving failed"
                )
                if error:
                    return

                if result['success']:
                    st.success(f"ODE solved successfully (in {elapsed:.2f}s)")

                    x_vals = result['x_values']
                    y_vals = result['y_values']

                    st.markdown("### Solution")
                    if len(x_vals) <= 20:
                        for i in range(len(x_vals)):
                            if ode_type == "System of ODEs" and len(
                                    y_vals[0]) > 1:
                                st.write(
                                    f"x = {x_vals[i]:.3f}: y = {y_vals[i][0]:.6f}, z = {y_vals[i][1]:.6f}"
                                )
                            else:
                                st.write(
                                    f"x = {x_vals[i]:.3f}: y = {y_vals[i]:.6f}"
                                )
                    else:
                        st.info(f"Solution computed for {len(x_vals)} points")
                        st.write(
                            f"Final value: x = {x_vals[-1]:.3f}, y = {y_vals[-1]:.6f}"
                        )
                    copy_button(str(y_vals[-1]), key="ode-final-value")

                else:
                    st.error(f"ODE solving failed: {result['error']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_optimization_methods():
    """Optimization algorithms"""
    st.subheader("Optimization Algorithms")

    engine = OptimizationAlgorithmsEngine()

    # Optimization tabs
    opt_tab1, opt_tab2, opt_tab3 = st.tabs(
        ["Function Optimization", "Linear Programming", "Genetic Algorithm"])

    with opt_tab1:
        render_function_optimization(engine)

    with opt_tab2:
        render_linear_programming(engine)

    with opt_tab3:
        render_genetic_algorithm(engine)


def render_function_optimization(engine):
    """Function optimization methods"""
    st.markdown("### Function Optimization")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Setup")

        # Objective function
        objective = st.text_input("Objective function f(x,y)",
                                  value="x^2 + y^2",
                                  key="opt_objective")

        # Variables
        variables = st.text_input("Variables (comma-separated)",
                                  value="x,y",
                                  key="opt_variables")

        # Method
        method = st.selectbox("Method", [
            "gradient_descent", "newton_method", "conjugate_gradient",
            "nelder_mead"
        ],
                              key="opt_method")

        # Initial guess
        initial_guess = st.text_input("Initial guess",
                                      value="1,1",
                                      key="opt_initial")

        # Parameters
        tol = st.number_input("Tolerance",
                              value=1e-8,
                              format="%.2e",
                              key="opt_func_tolerance")
        max_iter = st.number_input("Max iterations",
                                   value=1000,
                                   min_value=1,
                                   key="opt_func_max_iter")

        # Constraints (optional)
        use_constraints = st.checkbox("Add constraints",
                                      key="opt_use_constraints")

        if use_constraints:
            constraint_type = st.selectbox("Constraint type",
                                           ["equality", "inequality"],
                                           key="opt_constraint_type")
            constraint_expr = st.text_input(
                "Constraint g(x,y) = 0 or g(x,y) ≤ 0",
                value="x + y - 1",
                key="opt_constraint_expr")

    with col2:
        st.markdown("#### Results")

        if st.button("Optimize", type="primary", key="optimize_btn"):
            try:
                vars_list = [v.strip() for v in variables.split(',')]
                initial_vals = [
                    float(x.strip()) for x in initial_guess.split(',')
                ]

                params = {
                    'objective_function': objective,
                    'variables': vars_list,
                    'method': method,
                    'initial_guess': initial_vals,
                    'tolerance': tol,
                    'max_iterations': max_iter
                }

                if use_constraints:
                    params['constraints'] = [{
                        'type': constraint_type,
                        'expression': constraint_expr
                    }]

                result, elapsed, error = run_task(
                    "Optimizing...",
                    engine.optimize,
                    params,
                    success_message="Optimization completed",
                    error_message="Optimization failed"
                )
                if error:
                    return

                if result['success']:
                    st.success(f"Optimization completed in {elapsed:.2f}s")

                    optimal_point = result['optimal_point']
                    optimal_value = result['optimal_value']

                    st.markdown("**Optimal Point:**")
                    for i, var in enumerate(vars_list):
                        st.write(f"{var} = {optimal_point[i]:.8f}")

                    st.markdown(f"**Optimal Value:** {optimal_value:.8f}")
                    st.info(f"Iterations: {result['iterations']}")
                    copy_button(str(optimal_value), key="optimization-value")

                    if 'convergence_history' in result:
                        st.markdown("**Convergence History:**")
                        history = result['convergence_history']
                        for i, val in enumerate(history[-5:]):  # Show last 5
                            st.write(f"Iter {len(history)-5+i+1}: {val:.8f}")

                else:
                    st.error(f"Optimization failed: {result['error']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_linear_programming(engine):
    """Linear programming solver"""
    st.markdown("### Linear Programming")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Problem Setup")
        st.markdown("Standard form: min/max c^T x subject to Ax ≤ b, x ≥ 0")

        # Objective
        objective_type = st.selectbox("Objective", ["minimize", "maximize"],
                                      key="lp_objective_type")

        c_coeffs = st.text_input("Objective coefficients c",
                                 value="1,2",
                                 key="lp_c_coeffs")

        # Constraints
        st.markdown("**Constraints (Ax ≤ b):**")

        A_matrix = st.text_area(
            "Constraint matrix A (rows separated by semicolons)",
            value="1,1; 2,1; 1,2",
            key="lp_A_matrix")

        b_vector = st.text_input("Constraint vector b",
                                 value="4,6,5",
                                 key="lp_b_vector")

        # Variable bounds
        use_bounds = st.checkbox("Custom variable bounds", key="lp_use_bounds")
        if use_bounds:
            bounds = st.text_input("Bounds (x1_min,x1_max; x2_min,x2_max)",
                                   value="0,inf; 0,inf",
                                   key="lp_bounds")

    with col2:
        st.markdown("#### Results")

        if st.button("Solve LP", type="primary", key="solve_lp_btn"):
            try:
                c = [float(x.strip()) for x in c_coeffs.split(',')]

                A = []
                for row in A_matrix.split(';'):
                    A.append([float(x.strip()) for x in row.split(',')])

                b = [float(x.strip()) for x in b_vector.split(',')]

                params = {'c': c, 'A': A, 'b': b, 'objective': objective_type}

                if use_bounds:
                    bound_pairs = []
                    for bound_str in bounds.split(';'):
                        lower, upper = bound_str.split(',')
                        lower = float(
                            lower.strip()) if lower.strip() != 'inf' else None
                        upper = float(
                            upper.strip()) if upper.strip() != 'inf' else None
                        bound_pairs.append((lower, upper))
                    params['bounds'] = bound_pairs

                result, elapsed, error = run_task(
                    "Solving linear program...",
                    engine.linear_programming,
                    params,
                    success_message="Linear program solved",
                    error_message="LP solving failed"
                )
                if error:
                    return

                if result['success']:
                    st.success(f"Linear program solved in {elapsed:.2f}s")

                    optimal_solution = result['optimal_solution']
                    optimal_value = result['optimal_value']

                    st.markdown("**Optimal Solution:**")
                    for i, val in enumerate(optimal_solution):
                        st.write(f"x{i+1} = {val:.6f}")

                    st.markdown(f"**Optimal Value:** {optimal_value:.6f}")
                    copy_button(str(optimal_value), key="lp-optimal-value")

                    if 'slack_variables' in result:
                        st.markdown("**Slack Variables:**")
                        slack = result['slack_variables']
                        for i, val in enumerate(slack):
                            st.write(f"s{i+1} = {val:.6f}")

                else:
                    st.error(f"LP solving failed: {result['error']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_genetic_algorithm(engine):
    """Genetic algorithm optimization"""
    st.markdown("### Genetic Algorithm")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### GA Setup")

        # Objective function
        objective = st.text_input("Objective function",
                                  value="x^2 + y^2",
                                  key="ga_objective")

        # Variables and bounds
        variables = st.text_input("Variables", value="x,y", key="ga_variables")

        bounds = st.text_input("Variable bounds (x_min,x_max; y_min,y_max)",
                               value="-5,5; -5,5",
                               key="ga_bounds")

        # GA parameters
        population_size = st.number_input("Population size",
                                          value=50,
                                          min_value=10,
                                          key="ga_pop_size")
        generations = st.number_input("Generations",
                                      value=100,
                                      min_value=10,
                                      key="ga_generations")
        mutation_rate = st.number_input("Mutation rate",
                                        value=0.1,
                                        min_value=0.0,
                                        max_value=1.0,
                                        key="ga_mutation_rate")
        crossover_rate = st.number_input("Crossover rate",
                                         value=0.8,
                                         min_value=0.0,
                                         max_value=1.0,
                                         key="ga_crossover_rate")

        # Optimization type
        optimization_type = st.selectbox("Optimization type",
                                         ["minimize", "maximize"],
                                         key="ga_opt_type")

    with col2:
        st.markdown("#### Results")

        if st.button("Run GA", type="primary", key="run_ga_btn"):
            try:
                # Parse inputs
                vars_list = [v.strip() for v in variables.split(',')]

                bound_pairs = []
                for bound_str in bounds.split(';'):
                    lower, upper = bound_str.split(',')
                    bound_pairs.append(
                        (float(lower.strip()), float(upper.strip())))

                params = {
                    'objective_function': objective,
                    'variables': vars_list,
                    'bounds': bound_pairs,
                    'population_size': population_size,
                    'generations': generations,
                    'mutation_rate': mutation_rate,
                    'crossover_rate': crossover_rate,
                    'optimization_type': optimization_type
                }

                # Parse inputs for genetic algorithm
                var_list = [v.strip() for v in variables.split(',') if v.strip()]

                # Parse bounds
                bound_pairs = []
                for bound_str in bounds.split(';'):
                    lower, upper = bound_str.split(',')
                    bound_pairs.append(
                        (float(lower.strip()), float(upper.strip())))

                bounds_list = bound_pairs

                # Create objective function
                import sympy as sp
                from core.expression_parser import expression_parser

                expr = expression_parser.parse(objective)
                var_symbols = [sp.Symbol(var) for var in var_list]
                func = sp.lambdify(var_symbols, expr, 'numpy')

                def objective_func(x):
                    return float(func(*x))

                result, elapsed, error = run_task(
                    "Running genetic algorithm...",
                    engine.genetic_algorithm,
                    objective_func, bounds_list,
                    pop_size=population_size,
                    crossover_prob=crossover_rate,
                    mutation_prob=mutation_rate,
                    success_message="GA completed",
                    error_message="GA failed"
                )
                if error:
                    return

                if result.converged:
                    st.success(f"Genetic algorithm completed in {elapsed:.2f}s")

                    best_individual = result.x_opt
                    best_fitness = result.f_opt

                    st.markdown("**Best Solution:**")
                    for i, var in enumerate(var_list):
                        st.write(f"{var} = {best_individual[i]:.6f}")

                    st.markdown(f"**Best Fitness:** {best_fitness:.6f}")
                    copy_button(str(best_fitness), key="ga-fitness")

                    if result.history and 'best_fitness' in result.history:
                        st.markdown("**Convergence:**")
                        history = result.history['best_fitness']
                        for i in range(0, len(history),
                                       max(1, len(history) // 10)):
                            st.write(f"Gen {i+1}: {history[i]:.6f}")

                else:
                    st.error("GA failed to converge")

            except Exception as e:
                st.error(f"Error: {str(e)}")