import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.calculation_engine import calculation_engine
from core.ode_solver import ode_solver
from core.matrix_operations import matrix_operations
from utils.ui_helpers import run_task, copy_button
from utils.exceptions import InvalidInputError, ExpressionParseError, ConvergenceError

# Example equations for quick loading
EQUATION_EXAMPLES = {
    "Quadratic": {"equation": "x**2 - 4", "type": "Single Equation"},
    "Trigonometric": {"equation": "sin(x) - 0.5", "type": "Single Equation"},
    "2×2 Linear System": {"equations": ["2*x + y - 5", "x - y - 1"], "type": "System"},
    "Cubic Polynomial": {"equation": "x**3 - 6*x**2 + 11*x - 6", "type": "Polynomial"},
}

def render_equation_solver_panel():
    """Render the equation solver panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Equation Solver</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Solve linear equations, systems of equations, and polynomial equations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example selector
    st.markdown("**Load Example:**")
    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        example_choice = st.selectbox(
            "Choose an example:",
            [""] + list(EQUATION_EXAMPLES.keys()),
            help="Select an example to auto-fill equation fields"
        )
    with col_ex2:
        if st.button("Load Example", disabled=not example_choice):
            if example_choice in EQUATION_EXAMPLES:
                ex = EQUATION_EXAMPLES[example_choice]
                if "equation" in ex:
                    st.session_state.equation_example = ex["equation"]
                st.rerun()
    
    st.markdown("---")
    
    # Solver type selection
    solver_type = st.selectbox(
        "Select Equation Type:",
        [
            "Single Equation",
            "System of Linear Equations",
            "Polynomial Equations"
        ]
    )
    
    if solver_type == "Single Equation":
        render_single_equation_solver()
    elif solver_type == "System of Linear Equations":
        render_linear_system_solver()
    elif solver_type == "Polynomial Equations":
        render_polynomial_solver()

def render_single_equation_solver():
    """Render single equation solver interface"""
    st.subheader("Single Equation Solver")
    st.markdown("*Solve equations of the form f(x) = 0 or f(x) = g(x)*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        equation = st.text_input(
            "Enter equation:",
            value=st.session_state.get("equation_example", ""),
            placeholder="x^2 - 4 = 0 or sin(x) = cos(x)",
            help="Use = for equations or enter expression that equals zero"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x")
    
    # Solution method selection
    method = st.radio(
        "Solution Method:",
        ["Symbolic", "Numerical"],
        help="Symbolic: Exact solutions, Numerical: Approximate solutions"
    )
    
    if equation and variable:
        if st.button("Solve Equation", key="solve_single"):
            solve_single_equation(equation, variable, method)
        
        # Additional options for numerical methods
        if method == "Numerical":
            st.markdown("---")
            st.subheader("Numerical Method Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_min = st.number_input("Search Range Min:", value=-10.0)
            with col2:
                x_max = st.number_input("Search Range Max:", value=10.0)
            with col3:
                num_points = st.number_input("Search Points:", min_value=100, max_value=10000, value=1000)
            
            if st.button("Find Numerical Solutions", key="solve_numerical"):
                find_numerical_solutions(equation, variable, x_min, x_max, num_points)

def render_linear_system_solver():
    """Render linear system solver interface"""
    st.subheader("System of Linear Equations")
    st.markdown("*Solve systems of the form Ax = b*")
    
    # Input method selection
    input_method = st.radio(
        "Input Method:",
        ["Manual Entry", "Matrix Form", "Equation Form"]
    )
    
    if input_method == "Manual Entry":
        render_manual_system_entry()
    elif input_method == "Matrix Form":
        render_matrix_system_entry()
    elif input_method == "Equation Form":
        render_equation_system_entry()

def render_manual_system_entry():
    """Manual entry for linear systems"""
    col1, col2 = st.columns(2)
    
    with col1:
        num_equations = st.number_input("Number of equations:", min_value=2, max_value=10, value=3)
    with col2:
        num_variables = st.number_input("Number of variables:", min_value=2, max_value=10, value=3)
    
    st.markdown("**Enter coefficient matrix A and constants vector b:**")
    
    # Create input grids
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Coefficient Matrix A:**")
        coefficient_matrix = []
        for i in range(num_equations):
            row = []
            cols = st.columns(num_variables)
            for j in range(num_variables):
                with cols[j]:
                    value = st.number_input(
                        f"a[{i+1},{j+1}]",
                        value=0.0,
                        key=f"coeff_{i}_{j}",
                        format="%.6f"
                    )
                    row.append(value)
            coefficient_matrix.append(row)
    
    with col2:
        st.markdown("**Constants Vector b:**")
        constants_vector = []
        for i in range(num_equations):
            value = st.number_input(
                f"b[{i+1}]",
                value=0.0,
                key=f"const_{i}",
                format="%.6f"
            )
            constants_vector.append(value)
    
    if st.button("Solve Linear System", key="solve_linear_manual"):
        solve_linear_system(coefficient_matrix, constants_vector)

def render_matrix_system_entry():
    """Matrix form entry for linear systems"""
    st.info("Use the Matrix Operations panel for detailed matrix input and linear system solving")

def render_equation_system_entry():
    """Equation form entry for linear systems"""
    st.markdown("**Enter equations in standard form:**")
    st.markdown("*Example: 2*x + 3*y - z = 5*")
    
    num_equations = st.number_input("Number of equations:", min_value=2, max_value=10, value=3, key="eq_count")
    
    equations = []
    for i in range(num_equations):
        eq = st.text_input(
            f"Equation {i+1}:",
            placeholder=f"2*x + 3*y - z = {i+1}",
            key=f"equation_{i}"
        )
        if eq:
            equations.append(eq)
    
    variables = st.text_input(
        "Variables (comma-separated):",
        placeholder="x, y, z",
        key="system_variables"
    )
    
    if len(equations) == num_equations and variables:
        if st.button("Solve Equation System", key="solve_eq_system"):
            solve_equation_system(equations, variables)

def render_polynomial_solver():
    """Render polynomial equation solver"""
    st.subheader("Polynomial Equation Solver")
    st.markdown("*Solve polynomial equations of any degree*")
    
    # Input method
    input_method = st.radio(
        "Input Method:",
        ["Polynomial Expression", "Coefficients"]
    )
    
    if input_method == "Polynomial Expression":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            polynomial = st.text_input(
                "Polynomial equation:",
                placeholder="x^4 - 5*x^3 + 6*x^2 + 4*x - 8 = 0",
                key="poly_expr"
            )
        
        with col2:
            variable = st.text_input("Variable:", value="x", key="poly_var")
        
        if polynomial and st.button("Solve Polynomial", key="solve_poly_expr"):
            solve_polynomial_expression(polynomial, variable)
    
    else:
        st.markdown("**Enter polynomial coefficients (highest degree first):**")
        degree = st.number_input("Polynomial degree:", min_value=1, max_value=10, value=3)
        
        coefficients = []
        cols = st.columns(min(degree + 1, 6))  # Limit columns for display
        
        for i in range(degree + 1):
            col_idx = i % 6
            with cols[col_idx]:
                power = degree - i
                coeff = st.number_input(
                    f"x^{power}" if power > 0 else "constant",
                    value=0.0,
                    key=f"coeff_{i}",
                    format="%.6f"
                )
                coefficients.append(coeff)
        
        if st.button("Solve by Coefficients", key="solve_poly_coeff"):
            solve_polynomial_coefficients(coefficients)

def render_differential_equation_solver():
    """Render differential equation solver"""
    st.subheader("Differential Equation Solver")
    st.markdown("*Solve ordinary differential equations (ODEs)*")
    
    # ODE type selection
    ode_type = st.selectbox(
        "ODE Type:",
        ["First Order ODE", "Second Order ODE", "System of ODEs"]
    )
    
    if ode_type == "First Order ODE":
        render_first_order_ode()
    elif ode_type == "Second Order ODE":
        render_second_order_ode()
    elif ode_type == "System of ODEs":
        render_ode_system()

def render_first_order_ode():
    """Render first order ODE solver"""
    st.markdown("**First Order ODE: dy/dx = f(x, y)**")
    
    # Solution method tabs
    tab1, tab2 = st.tabs(["Symbolic Solution", "Numerical Solution"])
    
    with tab1:
        ode_expr = st.text_input(
            "ODE Expression:",
            placeholder="y' + 2*y = x",
            help="Use y' for dy/dx",
            key="ode_symbolic"
        )
        
        if ode_expr and st.button("Solve Symbolically", key="solve_ode_symbolic"):
            solve_ode_symbolic(ode_expr)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            func_expr = st.text_input(
                "dy/dx = f(x, y):",
                placeholder="x - 2*y",
                key="ode_func"
            )
        
        with col2:
            method = st.selectbox(
                "Numerical Method:",
                ["Euler", "Runge-Kutta 4", "scipy solve_ivp"]
            )
        
        # Initial conditions and parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            y0 = st.number_input("Initial y(x0):", value=1.0)
        with col2:
            x0 = st.number_input("Initial x0:", value=0.0)
        with col3:
            x_end = st.number_input("End x:", value=5.0)
        
        num_points = st.number_input("Number of points:", min_value=10, max_value=10000, value=100)
        
        if func_expr and st.button("Solve Numerically", key="solve_ode_numerical"):
            solve_ode_numerical(func_expr, y0, (x0, x_end), method, num_points)

def render_second_order_ode():
    """Render second order ODE solver"""
    st.markdown("**Second Order ODE: y'' + p(x)y' + q(x)y = r(x)**")
    
    st.info("Second order ODE solving is a complex topic. This interface provides basic symbolic solving.")
    
    ode_expr = st.text_input(
        "ODE Expression:",
        placeholder="y'' + 2*y' + y = 0",
        help="Use y' for dy/dx and y'' for d²y/dx²",
        key="ode2_expr"
    )
    
    if ode_expr and st.button("Solve Second Order ODE", key="solve_ode2"):
        solve_second_order_ode(ode_expr)

def render_ode_system():
    """Render system of ODEs solver"""
    st.markdown("**System of ODEs**")
    
    num_equations = st.number_input("Number of equations:", min_value=2, max_value=10, value=2)
    
    st.markdown("**Enter system of equations (dy_i/dx = f_i(x, y1, y2, ...)):**")
    
    equations = []
    for i in range(num_equations):
        eq = st.text_input(
            f"dy{i+1}/dx =",
            placeholder=f"y{2 if i == 0 else 1} + x",
            key=f"ode_system_{i}"
        )
        equations.append(eq)
    
    # Initial conditions
    st.markdown("**Initial Conditions:**")
    initial_conditions = []
    cols = st.columns(min(num_equations, 4))
    
    for i in range(num_equations):
        col_idx = i % 4
        with cols[col_idx]:
            ic = st.number_input(f"y{i+1}(0):", value=1.0, key=f"ic_{i}")
            initial_conditions.append(ic)
    
    # Solution parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_start = st.number_input("Start x:", value=0.0)
    with col2:
        x_end = st.number_input("End x:", value=5.0)
    with col3:
        method = st.selectbox("Method:", ["RK45", "RK23", "DOP853"])
    
    variables = [f"y{i+1}" for i in range(num_equations)]
    
    if all(equations) and st.button("Solve ODE System", key="solve_ode_system"):
        solve_ode_system(equations, initial_conditions, (x_start, x_end), variables, method)

# Solution functions
def solve_single_equation(equation, variable, method):
    """Solve a single equation"""
    try:
        if method == "Symbolic":
            result, _, error = run_task(
                "Solving equation...",
                calculation_engine.solve_equation,
                equation,
                variable,
                success_message="Equation solved",
                error_message="Equation solve failed"
            )
            if error:
                return
            if result['success']:
                st.success("Equation solved successfully")
                
                st.markdown("**Solutions:**")
                if result['solutions']:
                    for i, solution in enumerate(result['solutions']):
                        st.write(f"Solution {i+1}: {variable} = {solution}")
                        copy_button(str(solution), key=f"solution-{i}")
                        
                        # Try to evaluate numerically
                        try:
                            numeric_val = float(sp.sympify(solution).evalf())
                            st.write(f"   Numeric value: {numeric_val:.10g}")
                        except Exception:
                            pass
                else:
                    st.warning("No solutions found")
                
                # Display equation
                st.markdown("**Equation:**")
                st.latex(sp.latex(result['equation']))
            else:
                st.error(f"Error: {result['error']}")
        
        else:  # Numerical method handled separately
            st.info("Use 'Find Numerical Solutions' button below for numerical methods")
    
    except Exception as e:
        st.error(f"Error solving equation: {str(e)}")

def find_numerical_solutions(equation, variable, x_min, x_max, num_points):
    """Find numerical solutions by searching for zeros"""
    try:
        from scipy.optimize import fsolve
        import numpy as np
        
        # Parse and create function
        if '=' in equation:
            left, right = equation.split('=')
            expr_str = f"({left}) - ({right})"
        else:
            expr_str = equation
        
        result = calculation_engine.evaluate_expression(expr_str)
        if not result['success']:
            st.error(f"Cannot parse equation: {result['error']}")
            return
        
        # Create numerical function
        var_symbol = sp.Symbol(variable)
        expr = result['symbolic_result']
        func = sp.lambdify(var_symbol, expr, 'numpy')
        
        # Search for sign changes
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = func(x_vals)
        
        # Find approximate roots where sign changes
        roots = []
        for i in range(len(y_vals) - 1):
            if y_vals[i] * y_vals[i+1] < 0:  # Sign change
                try:
                    root = fsolve(func, x_vals[i])[0]
                    # Verify it's actually a root
                    if abs(func(root)) < 1e-10:
                        roots.append(root)
                except:
                    pass
        
        # Remove duplicates
        unique_roots = []
        for root in roots:
            is_duplicate = False
            for existing in unique_roots:
                if abs(root - existing) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_roots.append(root)
        
        st.success(f"Found {len(unique_roots)} numerical solutions")
        
        if unique_roots:
            for i, root in enumerate(unique_roots):
                st.write(f"Solution {i+1}: {variable} = {root:.10g}")
                st.write(f"   Verification: f({root:.6g}) = {func(root):.2e}")
            copy_button(", ".join(f"{r:.10g}" for r in unique_roots), key="numeric-roots")
        else:
            st.warning("No numerical solutions found in the specified range")
        
        # Plot the function
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        if unique_roots:
            fig.add_trace(go.Scatter(
                x=unique_roots, 
                y=[0] * len(unique_roots), 
                mode='markers', 
                marker=dict(color='red', size=10), 
                name='Roots'
            ))
        
        fig.update_layout(
            title="Function Plot with Numerical Solutions",
            xaxis_title=variable,
            yaxis_title="f(x)"
        )
        
        st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Numerical solution failed: {str(e)}")

def solve_linear_system(coefficient_matrix, constants_vector):
    """Solve linear system Ax = b"""
    try:
        # Create matrices
        A_result = matrix_operations.create_matrix(coefficient_matrix, symbolic=False)
        b_result = matrix_operations.create_matrix([[c] for c in constants_vector], symbolic=False)
        
        if not A_result['success']:
            st.error(f"Error creating coefficient matrix: {A_result['error']}")
            return
        
        if not b_result['success']:
            st.error(f"Error creating constants vector: {b_result['error']}")
            return
        
        A = A_result['matrix']
        b = b_result['matrix'].flatten()
        
        # Solve system
        result, _, error = run_task(
            "Solving Ax = b...",
            matrix_operations.solve_linear_system,
            A,
            b,
            success_message="Linear system solved",
            error_message="Linear solve failed"
        )
        
        if not error and result['success']:
            st.success("Linear system solved successfully")
            
            st.markdown("**Solution:**")
            solution = result['solution']
            
            for i, val in enumerate(solution):
                st.write(f"x_{i+1} = {val:.10g}")
            copy_button(", ".join(f"{v:.10g}" for v in solution), key="lin-sys-solution")
            
            # Verification
            st.markdown("**Verification (Ax):**")
            verification = np.dot(A, solution)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Computed Ax:**")
                for val in verification:
                    st.write(f"{val:.10g}")
            
            with col2:
                st.write("**Expected b:**")
                for val in constants_vector:
                    st.write(f"{val:.10g}")
            
            # Error analysis
            error_val = np.linalg.norm(verification - np.array(constants_vector))
            st.metric("Solution Error (||Ax - b||)", f"{error_val:.2e}")
    
    except Exception as e:
        st.error(f"Error solving linear system: {str(e)}")

def solve_equation_system(equations, variables):
    """Solve system of equations in equation form"""
    try:
        # Parse variables
        var_list = [var.strip() for var in variables.split(',')]
        
        # Solve using SymPy
        parsed_equations = []
        for eq in equations:
            if '=' in eq:
                left, right = eq.split('=')
                expr = calculation_engine.parser.parse(left) - calculation_engine.parser.parse(right)
            else:
                expr = calculation_engine.parser.parse(eq)
            parsed_equations.append(expr)
        
        var_symbols = [sp.Symbol(var) for var in var_list]
        solutions = sp.solve(parsed_equations, var_symbols)
        
        st.success("Equation system solved successfully")
        
        if isinstance(solutions, dict):
            st.markdown("**Solution:**")
            for var, val in solutions.items():
                st.write(f"{var} = {val}")
                try:
                    numeric_val = float(val.evalf())
                    st.write(f"   Numeric: {numeric_val:.10g}")
                except:
                    pass
        
        elif isinstance(solutions, list) and solutions:
            st.markdown("**Multiple Solutions:**")
            for i, sol in enumerate(solutions):
                st.write(f"Solution {i+1}:")
                if isinstance(sol, dict):
                    for var, val in sol.items():
                        st.write(f"  {var} = {val}")
                else:
                    st.write(f"  {sol}")
        
        else:
            st.warning("No solutions found or infinite solutions exist")
    
    except Exception as e:
        st.error(f"Error solving equation system: {str(e)}")

def solve_polynomial_expression(polynomial, variable):
    """Solve polynomial equation from expression"""
    try:
        result, _, error = run_task(
            "Solving polynomial...",
            calculation_engine.solve_equation,
            polynomial,
            variable,
            success_message="Polynomial solved",
            error_message="Polynomial solve failed"
        )
        if error:
            return
        if result['success']:
            st.success("Polynomial solved successfully")
            
            st.markdown("**Roots:**")
            if result['solutions']:
                copy_button(", ".join(str(sol) for sol in result['solutions']), key="poly-roots")
                for i, solution in enumerate(result['solutions']):
                    st.write(f"Root {i+1}: {variable} = {solution}")
                    
                    # Classify root type
                    try:
                        val = sp.sympify(solution)
                        if val.is_real:
                            numeric_val = float(val.evalf())
                            st.write(f"   Real root: {numeric_val:.10g}")
                        else:
                            st.write(f"   Complex root: {val}")
                    except Exception:
                        st.write("   Symbolic root")
            else:
                st.warning("No roots found")
        else:
            st.error(f"Error: {result['error']}")
    
    except Exception as e:
        st.error(f"Error solving polynomial: {str(e)}")

def solve_polynomial_coefficients(coefficients):
    """Solve polynomial from coefficients"""
    try:
        # Remove leading zeros
        while len(coefficients) > 1 and coefficients[0] == 0:
            coefficients.pop(0)
        
        if len(coefficients) == 0:
            st.error("Invalid polynomial (all coefficients are zero)")
            return
        
        # Create polynomial
        roots = np.roots(coefficients)
        
        st.success("Polynomial solved successfully")
        
        # Build polynomial expression for display
        degree = len(coefficients) - 1
        poly_terms = []
        for i, coeff in enumerate(coefficients):
            power = degree - i
            if coeff != 0:
                if power == 0:
                    poly_terms.append(f"{coeff}")
                elif power == 1:
                    poly_terms.append(f"{coeff}x" if coeff != 1 else "x")
                else:
                    poly_terms.append(f"{coeff}x^{power}" if coeff != 1 else f"x^{power}")
        
        polynomial_str = " + ".join(poly_terms).replace("+ -", "- ")
        st.write(f"**Polynomial:** {polynomial_str} = 0")
        
        st.markdown("**Roots:**")
        for i, root in enumerate(roots):
            if np.isreal(root):
                st.write(f"Root {i+1}: x = {root.real:.10g} (real)")
            else:
                st.write(f"Root {i+1}: x = {root:.6g} (complex)")
    
    except Exception as e:
        st.error(f"Error solving polynomial: {str(e)}")

def solve_ode_symbolic(ode_expr):
    """Solve ODE symbolically"""
    try:
        result = ode_solver.solve_symbolic_ode(ode_expr)
        
        if result['success']:
            st.success("ODE solved symbolically")
            
            st.markdown("**Solution:**")
            st.latex(sp.latex(result['solution']))
            
            st.markdown("**ODE:**")
            st.latex(sp.latex(result['ode_expr']))
        
        else:
            st.error(f"Error: {result['error']}")
            st.info("Try numerical methods if symbolic solution fails")
    
    except Exception as e:
        st.error(f"Error solving ODE: {str(e)}")

def solve_ode_numerical(func_expr, y0, x_span, method, num_points):
    """Solve ODE numerically"""
    try:
        # Create function
        func_result = ode_solver.create_ode_function(func_expr)
        
        if not func_result['success']:
            st.error(f"Error creating function: {func_result['error']}")
            return
        
        func = func_result['function']
        
        # Solve based on method
        if method == "Euler":
            result = ode_solver.euler_method(func, y0, x_span, num_points)
        elif method == "Runge-Kutta 4":
            result = ode_solver.runge_kutta_4(func, y0, x_span, num_points)
        else:  # scipy solve_ivp
            result = ode_solver.solve_ivp_wrapper(
                lambda t, y: func(t, y[0]), [y0], x_span, method="RK45", num_points=num_points
            )
            if result['success']:
                result['y_values'] = result['y_values'][0]  # Extract first component
        
        if result['success']:
            st.success(f"ODE solved using {result['method']}")
            
            # Plot solution
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result['x_values'], 
                y=result['y_values'], 
                mode='lines',
                name='y(x)'
            ))
            
            fig.update_layout(
                title=f"ODE Solution using {result['method']}",
                xaxis_title="x",
                yaxis_title="y"
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Solution statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Value", f"{y0:.6g}")
            with col2:
                st.metric("Final Value", f"{result['y_values'][-1]:.6g}")
            with col3:
                st.metric("Points Computed", len(result['x_values']))
        
        else:
            st.error(f"Error: {result['error']}")
    
    except Exception as e:
        st.error(f"Error solving ODE numerically: {str(e)}")

def solve_second_order_ode(ode_expr):
    """Solve second order ODE"""
    try:
        # This is a simplified implementation
        result = ode_solver.solve_symbolic_ode(ode_expr)
        
        if result['success']:
            st.success("Second order ODE solved")
            
            st.markdown("**Solution:**")
            st.latex(sp.latex(result['solution']))
        else:
            st.error(f"Error: {result['error']}")
            st.info("Second order ODEs often require specific techniques. Consider simplifying or using numerical methods.")
    
    except Exception as e:
        st.error(f"Error solving second order ODE: {str(e)}")

def solve_ode_system(equations, initial_conditions, x_span, variables, method):
    """Solve system of ODEs"""
    try:
        result = ode_solver.solve_system_odes(
            equations, initial_conditions, x_span, variables, method
        )
        
        if result['success']:
            st.success(f"ODE system solved using {result['method']}")
            
            # Plot solutions
            fig = go.Figure()
            
            for i, var in enumerate(variables):
                fig.add_trace(go.Scatter(
                    x=result['x_values'], 
                    y=result['y_values'][i], 
                    mode='lines',
                    name=f'{var}(x)'
                ))
            
            fig.update_layout(
                title="ODE System Solution",
                xaxis_title="x",
                yaxis_title="y"
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Phase portrait for 2D systems
            if len(variables) == 2:
                st.markdown("**Phase Portrait:**")
                fig_phase = go.Figure()
                fig_phase.add_trace(go.Scatter(
                    x=result['y_values'][0], 
                    y=result['y_values'][1], 
                    mode='lines+markers',
                    marker=dict(size=4),
                    name='Trajectory'
                ))
                
                fig_phase.update_layout(
                    title="Phase Portrait",
                    xaxis_title=variables[0],
                    yaxis_title=variables[1]
                )
                
                st.plotly_chart(fig_phase, width='stretch')
        
        else:
            st.error(f"Error: {result['error']}")
    
    except Exception as e:
        st.error(f"Error solving ODE system: {str(e)}")

