import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from core.calculus_engine import calculus_engine
from core.expression_parser import expression_parser
from core.advanced_ode_solver import AdvancedODESolver, ODEProblem, ODEType, SolverMethod
from core.transforms_series_engine import transforms_series_engine
from utils.ui_helpers import run_task, copy_button
from utils.exceptions import InvalidInputError, ExpressionParseError, DomainError

# Example expressions for calculus operations
CALCULUS_EXAMPLES = {
    "Polynomial Derivative": {"expr": "x**3 + 2*x**2 - 3*x + 1", "var": "x", "type": "derivative"},
    "Trig Integration": {"expr": "sin(x)*cos(x)", "var": "x", "type": "integral"},
    "Limit at Infinity": {"expr": "(x**2 + 1)/(2*x**2 - 3)", "var": "x", "type": "limit"},
    "Taylor Series": {"expr": "exp(x)", "var": "x", "type": "series"},
}

def render_calculus_panel():
    """Render the calculus tools panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Calculus Tools</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Comprehensive calculus operations including derivatives, integrals, limits, and series</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example selector
    st.markdown("**Load Example:**")
    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        example_choice = st.selectbox(
            "Choose an example:",
            [""] + list(CALCULUS_EXAMPLES.keys()),
            help="Select an example to auto-fill expression fields"
        )
    with col_ex2:
        if st.button("Load Example", disabled=not example_choice):
            if example_choice in CALCULUS_EXAMPLES:
                ex = CALCULUS_EXAMPLES[example_choice]
                st.session_state.calculus_example_expr = ex["expr"]
                st.session_state.calculus_example_var = ex["var"]
                st.rerun()
    
    st.markdown("---")
    
    # Operation selection tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Derivatives", "Integrals", "Limits", "Series", "Function Analysis", "Multivariable Calculus", "Differential Equations", "Transforms & Series"
    ])
    
    with tab1:
        render_derivatives_section()
    
    with tab2:
        render_integrals_section()
    
    with tab3:
        render_limits_section()
    
    with tab4:
        render_series_section()
    
    with tab5:
        render_function_analysis_section()
    
    with tab6:
        render_multivariable_calculus_section()
    
    with tab7:
        render_differential_equations_section()
    
    with tab8:
        render_transforms_series_section()

def render_derivatives_section():
    """Render derivatives calculation section"""
    st.subheader("Derivative Calculator")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            value=st.session_state.get("calculus_example_expr", ""),
            placeholder="x^3 + 2*x^2 - 3*x + 1",
            key="derivative_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value=st.session_state.get("calculus_example_var", "x"), key="derivative_var")
    
    with col3:
        order = st.number_input("Order:", min_value=1, max_value=5, value=1, key="derivative_order")
    
    if expression:
        # Single variable derivative
        st.markdown("**Single Variable Derivative:**")
        result = calculus_engine.compute_derivative(expression, variable, order)
        display_derivative_result(result, order)
        
        # Note: Partial derivatives have been moved to the Multivariable Calculus tab

def render_integrals_section():
    """Render integrals calculation section"""
    st.subheader("Integral Calculator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            value=st.session_state.get("calculus_example_expr", ""),
            placeholder="x^2 + sin(x)",
            key="integral_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value=st.session_state.get("calculus_example_var", "x"), key="integral_var")
    
    # Integral type selection
    integral_type = st.radio(
        "Integral Type:",
        ["Indefinite", "Definite"],
        key="integral_type"
    )
    
    # Initialize limits with default values
    lower_limit = 0.0
    upper_limit = 1.0
    
    # Definite integral limits
    if integral_type == "Definite":
        col1, col2 = st.columns(2)
        with col1:
            lower_limit = st.number_input("Lower Limit:", value=0.0, key="lower_limit")
        with col2:
            upper_limit = st.number_input("Upper Limit:", value=1.0, key="upper_limit")
    
    if expression:
        result = None
        if integral_type == "Indefinite":
            result = calculus_engine.compute_integral(expression, variable, definite=False)
        elif integral_type == "Definite":
            result = calculus_engine.compute_integral(
                expression, variable, definite=True, 
                lower_limit=lower_limit, upper_limit=upper_limit
            )
        
        if result:
            display_integral_result(result, integral_type)
            
            # Numerical integration comparison for definite integrals
            if integral_type == "Definite" and result['success']:
                st.markdown("---")
                st.markdown("**Numerical Integration Verification:**")
                perform_numerical_integration(expression, variable, lower_limit, upper_limit)

def render_limits_section():
    """Render limits calculation section"""
    st.subheader("Limit Calculator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            value=st.session_state.get("calculus_example_expr", ""),
            placeholder="sin(x)/x",
            key="limit_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value=st.session_state.get("calculus_example_var", "x"), key="limit_var")
    
    col1, col2 = st.columns(2)
    
    with col1:
        limit_point = st.text_input(
            "Limit Point:",
            value="0",
            placeholder="Enter number or 'oo' for infinity",
            key="limit_point"
        )
    
    with col2:
        direction = st.selectbox(
            "Direction:",
            ["+-", "+", "-"],
            help="+-: both sides, +: from right, -: from left",
            key="limit_direction"
        )
    
    if expression and limit_point:
        # Convert limit point
        try:
            if limit_point.lower() in ['oo', 'inf', 'infinity']:
                limit_val = 'oo'
            elif limit_point == '-oo':
                limit_val = '-oo'
            else:
                limit_val = float(limit_point)
        except ValueError:
            limit_val = limit_point
        
        result = calculus_engine.compute_limit(expression, variable, limit_val, direction)
        display_limit_result(result, limit_point, direction)

def render_series_section():
    """Render series expansion section"""
    st.subheader("Series Expansion")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            value=st.session_state.get("calculus_example_expr", ""),
            placeholder="exp(x)",
            key="series_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value=st.session_state.get("calculus_example_var", "x"), key="series_var")
    
    col1, col2 = st.columns(2)
    
    with col1:
        point = st.number_input("Expansion Point:", value=0.0, key="series_point")
    
    with col2:
        order = st.number_input("Order:", min_value=1, max_value=20, value=6, key="series_order")
    
    if expression:
        result = calculus_engine.compute_series(expression, variable, point, order)
        display_series_result(result, point, order)
        
        # Series convergence visualization
        if result['success']:
            st.markdown("---")
            st.markdown("**Series Convergence Visualization:**")
            visualize_series_convergence(expression, variable, point, order)

def render_function_analysis_section():
    """Render comprehensive function analysis section"""
    st.subheader("Function Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            value=st.session_state.get("calculus_example_expr", ""),
            placeholder="x^3 - 3*x^2 + 2*x",
            key="analysis_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value=st.session_state.get("calculus_example_var", "x"), key="analysis_var")
    
    if expression:
        result = calculus_engine.analyze_function(expression, variable)
        display_function_analysis(result)
        
        # Critical points analysis
        if result['success'] and result['critical_points']:
            st.markdown("---")
            st.markdown("**Critical Points Analysis:**")
            analyze_critical_points(expression, variable, result['critical_points'])

def display_derivative_result(result, order):
    """Display derivative calculation results"""
    if result['success']:
        st.success(f"{order}{'st' if order == 1 else 'nd' if order == 2 else 'rd' if order == 3 else 'th'} derivative calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            st.markdown(f"**{order}{'st' if order == 1 else 'nd' if order == 2 else 'rd' if order == 3 else 'th'} Derivative:**")
            st.latex(sp.latex(result['simplified_derivative']))
        
        # Step-by-step if available
        if str(result['derivative']) != str(result['simplified_derivative']):
            st.markdown("**Step-by-step:**")
            st.write("1. Raw derivative:")
            st.latex(sp.latex(result['derivative']))
            st.write("2. Simplified:")
            st.latex(sp.latex(result['simplified_derivative']))
    
    else:
        st.error(f"Derivative calculation failed: {result['error']}")

def display_integral_result(result, integral_type):
    """Display integral calculation results"""
    if result['success']:
        st.success(f"{integral_type} integral calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            st.markdown(f"**{integral_type} Integral:**")
            st.latex(sp.latex(result['simplified_integral']))
            
            if result['numeric_value'] is not None:
                st.metric("Numeric Value", f"{result['numeric_value']:.10g}")
        
        # Step-by-step if available
        if str(result['integral']) != str(result['simplified_integral']):
            st.markdown("**Step-by-step:**")
            st.write("1. Raw integral:")
            st.latex(sp.latex(result['integral']))
            st.write("2. Simplified:")
            st.latex(sp.latex(result['simplified_integral']))
    
    else:
        st.error(f"Integral calculation failed: {result['error']}")

def display_limit_result(result, limit_point, direction):
    """Display limit calculation results"""
    if result['success']:
        st.success("Limit calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            st.markdown("**Limit Result:**")
            limit_latex = f"\\lim_{{x \\to {limit_point}{'⁺' if direction == '+' else '⁻' if direction == '-' else ''}}} f(x) = {sp.latex(result['limit'])}"
            st.latex(limit_latex)
        
        # Interpret result
        if result['limit'] == sp.oo:
            st.info("Limit approaches positive infinity")
        elif result['limit'] == -sp.oo:
            st.info("Limit approaches negative infinity")
        elif result['limit'].has(sp.oo):
            st.warning("Limit does not exist (involves infinity)")
        else:
            st.info(f"Limit exists and equals {result['limit']}")    
    else:
        st.error(f"Limit calculation failed: {result['error']}")

def display_series_result(result, point, order):
    """Display series expansion results"""
    if result['success']:
        st.success("Series expansion calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            series_type = "Maclaurin" if point == 0 else "Taylor"
            st.markdown(f"**{series_type} Series (order {order}):**")
            st.latex(sp.latex(result['series']))
        
        # Series information
        st.info(f"{series_type} series expansion around x = {point} up to order {order}")
    
    else:
        st.error(f"Series expansion failed: {result['error']}")

def display_function_analysis(result):
    """Display comprehensive function analysis"""
    if result['success']:
        st.success("Function analysis completed successfully")
        
        # Derivatives
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**First Derivative:**")
            st.latex(sp.latex(result['first_derivative']))
        
        with col2:
            st.markdown("**Second Derivative:**")
            st.latex(sp.latex(result['second_derivative']))
        
        # Critical points and inflection points
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Critical Points:**")
            if result['critical_points']:
                for i, point in enumerate(result['critical_points']):
                    st.write(f"x_{i+1} = {point}")
            else:
                st.write("No critical points found")
        
        with col2:
            st.markdown("**Inflection Points:**")
            if result['inflection_points']:
                for i, point in enumerate(result['inflection_points']):
                    st.write(f"x_{i+1} = {point}")
            else:
                st.write("No inflection points found")
    
    else:
        st.error(f"Function analysis failed: {result['error']}")

def compute_partial_derivatives(expression, variables):
    """Compute and display partial derivatives"""
    for var in variables:
        result = calculus_engine.compute_partial_derivative(expression, var)
        if result['success']:
            st.write(f"**∂f/∂{var}:**")
            st.latex(sp.latex(result['simplified_partial']))
        else:
            st.error(f"Error computing ∂f/∂{var}: {result['error']}")

def display_gradient_result(result):
    """Display gradient calculation results"""
    if result['success']:
        st.success("Gradient calculated successfully")
        
        st.markdown("**Gradient Vector:**")
        gradient_components = []
        for var, partial in result['gradient'].items():
            gradient_components.append(f"\\frac{{\\partial f}}{{\\partial {var}}} = {sp.latex(partial)}")
        
        gradient_latex = "\\nabla f = \\begin{pmatrix} " + " \\\\ ".join(gradient_components) + " \\end{pmatrix}"
        st.latex(gradient_latex)
    
    else:
        st.error(f"Gradient calculation failed: {result['error']}")

def perform_numerical_integration(expression, variable, lower_limit, upper_limit):
    """Perform numerical integration for comparison"""
    try:
        from scipy import integrate
        
        # Create numerical function
        expr = expression_parser.parse(expression)
        var_symbol = sp.Symbol(variable)
        func = sp.lambdify(var_symbol, expr, 'numpy')
        
        # Numerical integration
        numerical_result, error = integrate.quad(func, lower_limit, upper_limit)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Numerical Result", f"{numerical_result:.10g}")
        
        with col2:
            st.metric("Estimated Error", f"{error:.2e}")
    
    except Exception as e:
        st.error(f"Numerical integration failed: {str(e)}")

def visualize_series_convergence(expression, variable, point, max_order):
    """Visualize series convergence"""
    try:
        # Create plot data
        x_vals = np.linspace(point - 2, point + 2, 1000)
        
        # Original function
        expr = expression_parser.parse(expression)
        var_symbol = sp.Symbol(variable)
        original_func = sp.lambdify(var_symbol, expr, 'numpy')
        
        try:
            y_original = original_func(x_vals)
        except:
            st.warning("Cannot plot original function (may have singularities)")
            return
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Original function
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_original,
            mode='lines',
            name='Original Function',
            line=dict(color='black', width=3)
        ))
        
        # Series approximations
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for order in [1, 2, 3, 4, min(max_order, 6)]:
            series_result = calculus_engine.compute_series(expression, variable, point, order)
            if series_result['success']:
                series_func = sp.lambdify(var_symbol, series_result['series'], 'numpy')
                try:
                    y_series = series_func(x_vals)
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_series,
                        mode='lines',
                        name=f'Order {order}',
                        line=dict(color=colors[order-1], width=2, dash='dash')
                    ))
                except:
                    continue
        
        fig.update_layout(
            title="Series Convergence Visualization",
            xaxis_title=variable,
            yaxis_title="f(x)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Visualization failed: {str(e)}")

def analyze_critical_points(expression, variable, critical_points):
    """Analyze the nature of critical points"""
    try:
        # Second derivative test
        second_derivative_result = calculus_engine.compute_derivative(expression, variable, 2)
        
        if second_derivative_result['success']:
            second_deriv = second_derivative_result['simplified_derivative']
            var_symbol = sp.Symbol(variable)
            
            for i, point_str in enumerate(critical_points):
                try:
                    point = float(point_str)
                    second_deriv_value = float(second_deriv.subs(var_symbol, point))
                    
                    if second_deriv_value > 0:
                        nature = "Local Minimum"
                        icon = "🔻"
                    elif second_deriv_value < 0:
                        nature = "Local Maximum"
                        icon = "🔺"
                    else:
                        nature = "Inconclusive (may be inflection point)"
                        icon = "❓"
                    
                    st.write(f"{icon} **x = {point}**: {nature}")
                    st.write(f"   Second derivative value: {second_deriv_value:.6g}")
                
                except (ValueError, TypeError):
                    st.write(f"❓ **x = {point_str}**: Could not analyze (complex or symbolic)")
    
    except Exception as e:
        st.error(f"Critical point analysis failed: {str(e)}")

def render_multivariable_calculus_section():
    """Render multivariable calculus section"""
    st.subheader("Multivariable Calculus")
    st.markdown("*Partial derivatives, gradients, multiple integrals, and vector calculus*")
    
    # Sub-tabs for different multivariable operations
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Partial Derivatives", "Gradients & Directional", "Multiple Integrals", "Vector Fields"
    ])
    
    with subtab1:
        render_partial_derivatives()
    
    with subtab2:
        render_gradients_section()
    
    with subtab3:
        render_multiple_integrals()
    
    with subtab4:
        render_vector_fields()
    
    

def render_partial_derivatives():
    """Render partial derivatives section"""
    st.markdown("**Partial Derivatives**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2*y + y**2*z + sin(x*y)",
            key="partial_expr"
        )
    
    with col2:
        variables = st.text_input(
            "Variables (comma-separated):",
            value="x,y,z",
            key="partial_vars_unique"
        )
    
    if expression and variables:
        var_list = [v.strip() for v in variables.split(',') if v.strip()]
        
        if st.button("Compute Partial Derivatives", key="compute_partials"):
            try:
                expr = expression_parser.parse(expression)
                
                st.markdown("**Partial Derivatives:**")
                
                for var in var_list:
                    try:
                        result, _, error = run_task(
                            f"Computing ∂f/∂{var}...",
                            calculus_engine.compute_partial_derivative,
                            expression,
                            var,
                            success_message=f"∂f/∂{var} ready",
                            error_message=f"Failed ∂f/∂{var}"
                        )
                        if not error and result['success']:
                            st.markdown(f"∂f/∂{var} = `{result['partial_derivative']}`")
                            copy_button(str(result['partial_derivative']), key=f"partial-{var}")
                        elif not error:
                            st.error(f"Failed to compute ∂f/∂{var}: {result['error']}")
                    except Exception as e:
                        st.error(f"Error computing ∂f/∂{var}: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error parsing expression: {str(e)}")
    
    # Second order partial derivatives
    st.markdown("---")
    st.markdown("**Second-Order Partial Derivatives**")
    
    if st.checkbox("Compute second-order partials", key="second_order_check"):
        if expression and variables:
            var_list = [v.strip() for v in variables.split(',') if v.strip()]
            
            if st.button("Compute Second-Order Partials", key="compute_second_partials"):
                try:
                    expr = expression_parser.parse(expression)
                    
                    for i, var1 in enumerate(var_list):
                        for j, var2 in enumerate(var_list):
                            if i <= j:  # Avoid redundant calculations due to symmetry
                                try:
                                    # First compute ∂f/∂var1
                                    first_deriv = sp.diff(expr, sp.Symbol(var1))
                                    # Then compute ∂²f/∂var2∂var1
                                    second_deriv = sp.diff(first_deriv, sp.Symbol(var2))
                                    
                                    if var1 == var2:
                                        st.markdown(f"∂²f/∂{var1}² = `{second_deriv}`")
                                    else:
                                        st.markdown(f"∂²f/∂{var2}∂{var1} = `{second_deriv}`")
                                        
                                except Exception as e:
                                    st.error(f"Error computing ∂²f/∂{var2}∂{var1}: {str(e)}")
                                    
                except Exception as e:
                    st.error(f"Error in second-order computation: {str(e)}")

def render_gradients_section():
    """Render gradients and directional derivatives section"""
    st.markdown("**Gradient Vector**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2 + y**2 + z**2",
            key="gradient_expr"
        )
    
    with col2:
        variables = st.text_input(
            "Variables:",
            value="x,y,z",
            key="gradient_vars_unique"
        )
    
    if expression and variables:
        var_list = [v.strip() for v in variables.split(',') if v.strip()]
        
        if st.button("Compute Gradient", key="compute_gradient"):
            try:
                result, _, error = run_task(
                    "Computing gradient...",
                    calculus_engine.compute_gradient,
                    expression,
                    var_list,
                    success_message="Gradient ready",
                    error_message="Gradient failed"
                )
                
                if not error and result['success']:
                    st.markdown("**Gradient Vector:**")
                    gradient_components = result['gradient']
                    
                    gradient_str = "∇f = ("
                    gradient_str += ", ".join([f"`{comp}`" for comp in gradient_components])
                    gradient_str += ")"
                    
                    st.markdown(gradient_str)
                    copy_button(", ".join(str(comp) for comp in gradient_components), key="gradient")
                    
                    # Display magnitude
                    if 'magnitude' in result:
                        st.markdown(f"**Magnitude:** `{result['magnitude']}`")
                elif not error:
                    st.error(f"Gradient computation failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"Error computing gradient: {str(e)}")
    
    # Directional derivatives
    st.markdown("---")
    st.markdown("**Directional Derivative**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        point_coords = st.text_input(
            "Point coordinates (comma-separated):",
            placeholder="1,2,3",
            key="direction_point"
        )
    
    with col2:
        direction_vector = st.text_input(
            "Direction vector (comma-separated):",
            placeholder="1,1,1",
            key="direction_vec"
        )
    
    if expression and variables and point_coords and direction_vector:
        if st.button("Compute Directional Derivative", key="compute_directional"):
            try:
                var_list = [v.strip() for v in variables.split(',') if v.strip()]
                point_vals = [float(v.strip()) for v in point_coords.split(',')]
                dir_vals = [float(v.strip()) for v in direction_vector.split(',')]
                
                if len(point_vals) != len(var_list) or len(dir_vals) != len(var_list):
                    st.error("Number of coordinates and direction components must match number of variables")
                    return

                def _compute_directional():
                    gradient_result = calculus_engine.compute_gradient(expression, var_list)
                    if not gradient_result['success']:
                        raise ValueError(gradient_result['error'])

                    gradient = gradient_result['gradient']

                    subs_dict = {sp.Symbol(var_list[j]): point_vals[j] for j in range(len(var_list))}
                    grad_at_point = [float(comp.subs(subs_dict)) for comp in gradient]

                    dir_magnitude = sum(x**2 for x in dir_vals)**0.5
                    if dir_magnitude == 0:
                        raise ValueError("Direction vector magnitude must be non-zero")
                    unit_dir = [x / dir_magnitude for x in dir_vals]

                    dir_deriv = sum(grad_at_point[i] * unit_dir[i] for i in range(len(grad_at_point)))
                    return dir_deriv, grad_at_point, unit_dir

                result, _, error = run_task(
                    "Computing directional derivative...",
                    _compute_directional,
                    success_message="Directional derivative ready",
                    error_message="Directional derivative failed"
                )
                if error:
                    return

                dir_deriv, grad_at_point, unit_dir = result
                st.markdown(f"**Directional Derivative:** `{dir_deriv:.6f}`")
                st.markdown(f"**Gradient at point:** `{grad_at_point}`")
                st.markdown(f"**Unit direction vector:** `{unit_dir}`")
                copy_button(str(dir_deriv), key="directional-derivative")
            except Exception as e:
                st.error(f"Error computing directional derivative: {str(e)}")

def render_multiple_integrals():
    """Render multiple integrals section"""
    st.markdown("**Multiple Integrals**")
    st.info("Double and triple integrals with specified bounds")
    
    # Integral type selection
    integral_type = st.selectbox(
        "Integral Type:",
        ["Double Integral", "Triple Integral"],
        key="multiple_integral_type"
    )
    
    if integral_type == "Double Integral":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            expression = st.text_input(
                "Integrand f(x,y):",
                placeholder="x*y + x**2",
                key="double_integral_expr"
            )
        
        with col2:
            order = st.selectbox(
                "Integration Order:",
                ["dx dy", "dy dx"],
                key="double_order"
            )
        
        # Bounds input
        st.markdown("**Integration Bounds:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_lower = st.text_input("x lower:", value="0", key="x_lower_double")
        with col2:
            x_upper = st.text_input("x upper:", value="1", key="x_upper_double")
        with col3:
            y_lower = st.text_input("y lower:", value="0", key="y_lower_double")
        with col4:
            y_upper = st.text_input("y upper:", value="1", key="y_upper_double")
        
        if st.button("Compute Double Integral", key="compute_double"):
            try:
                expr = expression_parser.parse(expression)
                
                # Parse bounds
                x_l = expression_parser.parse(x_lower) if x_lower else 0
                x_u = expression_parser.parse(x_upper) if x_upper else 1
                y_l = expression_parser.parse(y_lower) if y_lower else 0
                y_u = expression_parser.parse(y_upper) if y_upper else 1
                
                def _compute_double():
                    if order == "dx dy":
                        inner_val = sp.integrate(expr, (sp.Symbol('x'), x_l, x_u))
                        return sp.integrate(inner_val, (sp.Symbol('y'), y_l, y_u))
                    inner_val = sp.integrate(expr, (sp.Symbol('y'), y_l, y_u))
                    return sp.integrate(inner_val, (sp.Symbol('x'), x_l, x_u))

                result, _, error = run_task(
                    "Computing double integral...",
                    _compute_double,
                    success_message="Double integral ready",
                    error_message="Double integral failed"
                )
                if error:
                    return
                st.markdown(f"**Result:** `{result}`")
                copy_button(str(result), key="double-integral")
                
                # Try to evaluate numerically
                try:
                    numeric_result = float(result)
                    st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
                except Exception:
                    st.markdown("*Result contains symbolic expressions*")
                    
            except Exception as e:
                st.error(f"Error computing double integral: {str(e)}")
    
    else:  # Triple Integral
        col1, col2 = st.columns([3, 1])
        
        with col1:
            expression = st.text_input(
                "Integrand f(x,y,z):",
                placeholder="x*y*z",
                key="triple_integral_expr"
            )
        
        with col2:
            order = st.selectbox(
                "Integration Order:",
                ["dx dy dz", "dx dz dy", "dy dx dz", "dy dz dx", "dz dx dy", "dz dy dx"],
                key="triple_order"
            )
        
        # Bounds input
        st.markdown("**Integration Bounds:**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            x_lower = st.text_input("x lower:", value="0", key="x_lower_triple")
        with col2:
            x_upper = st.text_input("x upper:", value="1", key="x_upper_triple")
        with col3:
            y_lower = st.text_input("y lower:", value="0", key="y_lower_triple")
        with col4:
            y_upper = st.text_input("y upper:", value="1", key="y_upper_triple")
        with col5:
            z_lower = st.text_input("z lower:", value="0", key="z_lower_triple")
        with col6:
            z_upper = st.text_input("z upper:", value="1", key="z_upper_triple")
        
        if st.button("Compute Triple Integral", key="compute_triple"):
            try:
                expr = expression_parser.parse(expression)
                
                # Parse bounds
                x_l = expression_parser.parse(x_lower) if x_lower else 0
                x_u = expression_parser.parse(x_upper) if x_upper else 1
                y_l = expression_parser.parse(y_lower) if y_lower else 0
                y_u = expression_parser.parse(y_upper) if y_upper else 1
                z_l = expression_parser.parse(z_lower) if z_lower else 0
                z_u = expression_parser.parse(z_upper) if z_upper else 1
                
                def _compute_triple():
                    order_mapping = {
                        "dx dy dz": [(sp.Symbol('x'), x_l, x_u), (sp.Symbol('y'), y_l, y_u), (sp.Symbol('z'), z_l, z_u)],
                        "dx dz dy": [(sp.Symbol('x'), x_l, x_u), (sp.Symbol('z'), z_l, z_u), (sp.Symbol('y'), y_l, y_u)],
                        "dy dx dz": [(sp.Symbol('y'), y_l, y_u), (sp.Symbol('x'), x_l, x_u), (sp.Symbol('z'), z_l, z_u)],
                        "dy dz dx": [(sp.Symbol('y'), y_l, y_u), (sp.Symbol('z'), z_l, z_u), (sp.Symbol('x'), x_l, x_u)],
                        "dz dx dy": [(sp.Symbol('z'), z_l, z_u), (sp.Symbol('x'), x_l, x_u), (sp.Symbol('y'), y_l, y_u)],
                        "dz dy dx": [(sp.Symbol('z'), z_l, z_u), (sp.Symbol('y'), y_l, y_u), (sp.Symbol('x'), x_l, x_u)]
                    }

                    integration_order = order_mapping[order]

                    result_val = expr
                    for var, lower, upper in integration_order:
                        result_val = sp.integrate(result_val, (var, lower, upper))
                    return result_val

                result, _, error = run_task(
                    "Computing triple integral...",
                    _compute_triple,
                    success_message="Triple integral ready",
                    error_message="Triple integral failed"
                )
                if error:
                    return
                st.markdown(f"**Result:** `{result}`")
                copy_button(str(result), key="triple-integral")
                
                # Try to evaluate numerically
                try:
                    numeric_result = float(result)
                    st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
                except Exception:
                    st.markdown("*Result contains symbolic expressions*")
                    
            except Exception as e:
                st.error(f"Error computing triple integral: {str(e)}")

def render_vector_fields():
    """Render comprehensive vector calculus section"""
    st.markdown("**Vector Calculus**")
    st.markdown("*Comprehensive vector field analysis, line integrals, surface integrals, and flux calculations*")
    
    # Sub-tabs for vector calculus
    vector_tab1, vector_tab2, vector_tab3, vector_tab4 = st.tabs([
        "Vector Fields", "Line Integrals", "Surface Integrals", "Theorems"
    ])
    
    with vector_tab1:
        render_vector_field_analysis()
    
    with vector_tab2:
        render_line_integrals()
    
    with vector_tab3:
        render_surface_integrals()
    
    with vector_tab4:
        render_vector_theorems()

def render_vector_field_analysis():
    """Vector field analysis (divergence, curl, etc.)"""
    st.subheader("Vector Field Analysis")
    
    # Vector field input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        F_x = st.text_input(
            "F_x component:",
            placeholder="y",
            key="vector_fx"
        )
    
    with col2:
        F_y = st.text_input(
            "F_y component:",
            placeholder="-x",
            key="vector_fy"
        )
    
    with col3:
        F_z = st.text_input(
            "F_z component (optional):",
            placeholder="z",
            key="vector_fz"
        )
    
    if F_x and F_y:
        # Divergence and curl calculations
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Compute Divergence", key="compute_divergence"):
                try:
                    def _compute_divergence():
                        fx_expr = expression_parser.parse(F_x)
                        fy_expr = expression_parser.parse(F_y)

                        div_x = sp.diff(fx_expr, sp.Symbol('x'))
                        div_y = sp.diff(fy_expr, sp.Symbol('y'))

                        if F_z:
                            fz_expr = expression_parser.parse(F_z)
                            div_z = sp.diff(fz_expr, sp.Symbol('z'))
                            return div_x + div_y + div_z
                        return div_x + div_y

                    divergence, _, error = run_task(
                        "Computing divergence...",
                        _compute_divergence,
                        success_message="Divergence ready",
                        error_message="Divergence failed"
                    )
                    if error:
                        return

                    st.markdown(f"**Divergence:** `{divergence}`")
                    copy_button(str(divergence), key="divergence")

                    if divergence == 0:
                        st.info("Incompressible flow: no sources or sinks")
                    else:
                        st.info("Compressible flow: sources or sinks present")
                except Exception as e:
                    st.error(f"Error computing divergence: {str(e)}")
        
        with col2:
            if st.button("Compute Curl", key="compute_curl"):
                try:
                    def _compute_curl():
                        fx_expr = expression_parser.parse(F_x)
                        fy_expr = expression_parser.parse(F_y)

                        if F_z:
                            fz_expr = expression_parser.parse(F_z)

                            curl_x = sp.diff(fz_expr, sp.Symbol('y')) - sp.diff(fy_expr, sp.Symbol('z'))
                            curl_y = sp.diff(fx_expr, sp.Symbol('z')) - sp.diff(fz_expr, sp.Symbol('x'))
                            curl_z = sp.diff(fy_expr, sp.Symbol('x')) - sp.diff(fx_expr, sp.Symbol('y'))
                            curl_magnitude = sp.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
                            return {
                                "type": "3d",
                                "curl_x": curl_x,
                                "curl_y": curl_y,
                                "curl_z": curl_z,
                                "magnitude": curl_magnitude
                            }

                        curl_val = sp.diff(fy_expr, sp.Symbol('x')) - sp.diff(fx_expr, sp.Symbol('y'))
                        return {"type": "2d", "curl": curl_val}

                    curl_result, _, error = run_task(
                        "Computing curl...",
                        _compute_curl,
                        success_message="Curl ready",
                        error_message="Curl failed"
                    )
                    if error:
                        return

                    if curl_result["type"] == "3d":
                        st.markdown("**Curl (3D):**")
                        st.markdown(f"curl_x = `{curl_result['curl_x']}`")
                        st.markdown(f"curl_y = `{curl_result['curl_y']}`")
                        st.markdown(f"curl_z = `{curl_result['curl_z']}`")
                        st.markdown(f"**Curl Magnitude:** `{curl_result['magnitude']}`")
                        copy_button(
                            f"({curl_result['curl_x']}, {curl_result['curl_y']}, {curl_result['curl_z']})",
                            key="curl-3d"
                        )
                    else:
                        curl_val = curl_result["curl"]
                        st.markdown(f"**Curl (2D):** `{curl_val}`")
                        copy_button(str(curl_val), key="curl-2d")

                        if curl_val == 0:
                            st.info("Conservative field: no rotation")
                        else:
                            st.info("Non-conservative field: rotation present")
                except Exception as e:
                    st.error(f"Error computing curl: {str(e)}")

def render_line_integrals():
    """Line integrals of scalar and vector fields"""
    st.subheader("Line Integrals")
    
    integral_type = st.selectbox(
        "Line Integral Type:",
        ["Scalar Field", "Vector Field", "Arc Length"],
        key="line_integral_type"
    )
    
    if integral_type == "Scalar Field":
        st.markdown("**Line Integral of Scalar Field:** ∫_C f(x,y,z) ds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scalar_field = st.text_input(
                "Scalar field f(x,y,z):",
                placeholder="x**2 + y**2",
                key="scalar_field_line"
            )
        
        with col2:
            parametric_path = st.text_area(
                "Parametric path (one per line):\nx(t) =\ny(t) =\nz(t) = (optional)",
                value="cos(t)\nsin(t)\n",
                key="parametric_path_scalar"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            t_start = st.number_input("t start:", value=0.0, key="t_start_scalar")
        with col2:
            t_end = st.number_input("t end:", value=6.28, key="t_end_scalar")
        
        if st.button("Compute Scalar Line Integral", key="compute_scalar_line"):
            compute_scalar_line_integral(scalar_field, parametric_path, t_start, t_end)
    
    elif integral_type == "Vector Field":
        st.markdown("**Line Integral of Vector Field:** ∫_C F⃗·dr⃗")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            F_x = st.text_input("F_x:", placeholder="y", key="vector_line_fx")
        with col2:
            F_y = st.text_input("F_y:", placeholder="-x", key="vector_line_fy")
        with col3:
            F_z = st.text_input("F_z:", placeholder="0", key="vector_line_fz")
        
        parametric_path = st.text_area(
            "Parametric path (one per line):\nx(t) =\ny(t) =\nz(t) = (optional)",
            value="cos(t)\nsin(t)\n0",
            key="parametric_path_vector"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            t_start = st.number_input("t start:", value=0.0, key="t_start_vector")
        with col2:
            t_end = st.number_input("t end:", value=6.28, key="t_end_vector")
        
        if st.button("Compute Vector Line Integral", key="compute_vector_line"):
            compute_vector_line_integral(F_x, F_y, F_z, parametric_path, t_start, t_end)

def render_surface_integrals():
    """Surface integrals of scalar and vector fields"""
    st.subheader("Surface Integrals")
    
    integral_type = st.selectbox(
        "Surface Integral Type:",
        ["Scalar Field", "Vector Field (Flux)", "Surface Area"],
        key="surface_integral_type"
    )
    
    if integral_type == "Scalar Field":
        st.markdown("**Surface Integral of Scalar Field:** ∬_S f(x,y,z) dS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scalar_field = st.text_input(
                "Scalar field f(x,y,z):",
                placeholder="x**2 + y**2 + z**2",
                key="scalar_field_surface"
            )
        
        with col2:
            surface_equation = st.text_input(
                "Surface z = g(x,y):",
                placeholder="x**2 + y**2",
                key="surface_equation"
            )
        
        # Domain bounds
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("x min:", value=-1.0, key="x_min_surface")
        with col2:
            x_max = st.number_input("x max:", value=1.0, key="x_max_surface")
        with col3:
            y_min = st.number_input("y min:", value=-1.0, key="y_min_surface")
        with col4:
            y_max = st.number_input("y max:", value=1.0, key="y_max_surface")
        
        if st.button("Compute Scalar Surface Integral", key="compute_scalar_surface"):
            compute_scalar_surface_integral(scalar_field, surface_equation, x_min, x_max, y_min, y_max)
    
    elif integral_type == "Vector Field (Flux)":
        st.markdown("**Flux through Surface:** ∬_S F⃗·n̂ dS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            F_x = st.text_input("F_x:", placeholder="x", key="flux_fx")
        with col2:
            F_y = st.text_input("F_y:", placeholder="y", key="flux_fy")
        with col3:
            F_z = st.text_input("F_z:", placeholder="z", key="flux_fz")
        
        surface_equation = st.text_input(
            "Surface z = g(x,y):",
            placeholder="1 - x**2 - y**2",
            key="flux_surface"
        )
        
        # Domain bounds
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("x min:", value=-1.0, key="x_min_flux")
        with col2:
            x_max = st.number_input("x max:", value=1.0, key="x_max_flux")
        with col3:
            y_min = st.number_input("y min:", value=-1.0, key="y_min_flux")
        with col4:
            y_max = st.number_input("y max:", value=1.0, key="y_max_flux")
        
        if st.button("Compute Flux Integral", key="compute_flux"):
            compute_flux_integral(F_x, F_y, F_z, surface_equation, x_min, x_max, y_min, y_max)

def render_vector_theorems():
    """Vector calculus theorems (Green's, Stokes', Divergence)"""
    st.subheader("Vector Calculus Theorems")
    
    theorem = st.selectbox(
        "Select Theorem:",
        ["Green's Theorem", "Stokes' Theorem", "Divergence Theorem"],
        key="vector_theorem"
    )
    
    if theorem == "Green's Theorem":
        st.markdown("**Green's Theorem:** ∮_C (P dx + Q dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            P_expr = st.text_input("P(x,y):", placeholder="y", key="green_P")
        with col2:
            Q_expr = st.text_input("Q(x,y):", placeholder="-x", key="green_Q")
        
        st.info("**Verification:** Compute both line integral around boundary and double integral over region")
        
        if P_expr and Q_expr:
            if st.button("Verify Green's Theorem", key="verify_green"):
                verify_greens_theorem(P_expr, Q_expr)
    
    elif theorem == "Stokes' Theorem":
        st.markdown("**Stokes' Theorem:** ∮_C F⃗·dr⃗ = ∬_S (∇×F⃗)·n̂ dS")
        st.info("**Relates:** Line integral around boundary to surface integral of curl")
        
    elif theorem == "Divergence Theorem":
        st.markdown("**Divergence Theorem:** ∬_S F⃗·n̂ dS = ∭_V ∇·F⃗ dV")
        st.info("**Relates:** Flux through closed surface to volume integral of divergence")

def compute_scalar_line_integral(scalar_field, parametric_path, t_start, t_end):
    """Compute line integral of scalar field"""
    try:
        def _compute_scalar_line():
            f_expr = expression_parser.parse(scalar_field)

            path_lines = parametric_path.strip().split('\n')
            x_param = expression_parser.parse(path_lines[0]) if len(path_lines) > 0 else sp.Symbol('t')
            y_param = expression_parser.parse(path_lines[1]) if len(path_lines) > 1 else sp.Symbol('t')
            z_param = expression_parser.parse(path_lines[2]) if len(path_lines) > 2 else 0

            t = sp.Symbol('t')

            f_parametric = f_expr.subs([
                (sp.Symbol('x'), x_param),
                (sp.Symbol('y'), y_param),
                (sp.Symbol('z'), z_param)
            ])

            dx_dt = sp.diff(x_param, t)
            dy_dt = sp.diff(y_param, t)
            dz_dt = sp.diff(z_param, t)

            ds_dt = sp.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
            integrand = f_parametric * ds_dt

            result_val = sp.integrate(integrand, (t, t_start, t_end))
            return ds_dt, integrand, result_val

        (ds_dt, integrand, result), _, error = run_task(
            "Computing scalar line integral...",
            _compute_scalar_line,
            success_message="Scalar line integral ready",
            error_message="Scalar line integral failed"
        )
        if error:
            return

        st.markdown(f"**Arc length element:** `ds = {ds_dt} dt`")
        st.markdown(f"**Integrand:** `{integrand}`")
        st.markdown(f"**Result:** `{result}`")
        copy_button(str(result), key="scalar-line-integral")

        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except Exception:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing scalar line integral: {str(e)}")

def compute_vector_line_integral(F_x, F_y, F_z, parametric_path, t_start, t_end):
    """Compute line integral of vector field"""
    try:
        def _compute_vector_line():
            fx_expr = expression_parser.parse(F_x) if F_x else 0
            fy_expr = expression_parser.parse(F_y) if F_y else 0
            fz_expr = expression_parser.parse(F_z) if F_z else 0

            path_lines = parametric_path.strip().split('\n')
            x_param = expression_parser.parse(path_lines[0]) if len(path_lines) > 0 else sp.Symbol('t')
            y_param = expression_parser.parse(path_lines[1]) if len(path_lines) > 1 else sp.Symbol('t')
            z_param = expression_parser.parse(path_lines[2]) if len(path_lines) > 2 else 0

            t = sp.Symbol('t')

            fx_parametric = fx_expr.subs([
                (sp.Symbol('x'), x_param),
                (sp.Symbol('y'), y_param),
                (sp.Symbol('z'), z_param)
            ])
            fy_parametric = fy_expr.subs([
                (sp.Symbol('x'), x_param),
                (sp.Symbol('y'), y_param),
                (sp.Symbol('z'), z_param)
            ])
            fz_parametric = fz_expr.subs([
                (sp.Symbol('x'), x_param),
                (sp.Symbol('y'), y_param),
                (sp.Symbol('z'), z_param)
            ])

            dx_dt = sp.diff(x_param, t)
            dy_dt = sp.diff(y_param, t)
            dz_dt = sp.diff(z_param, t)

            integrand = fx_parametric * dx_dt + fy_parametric * dy_dt + fz_parametric * dz_dt
            result_val = sp.integrate(integrand, (t, t_start, t_end))
            return (dx_dt, dy_dt, dz_dt), integrand, result_val

        (dr_dt, integrand, result), _, error = run_task(
            "Computing vector line integral...",
            _compute_vector_line,
            success_message="Vector line integral ready",
            error_message="Vector line integral failed"
        )
        if error:
            return

        dx_dt, dy_dt, dz_dt = dr_dt
        st.markdown(f"**dr/dt:** `({dx_dt}, {dy_dt}, {dz_dt})`")
        st.markdown(f"**F⃗·dr/dt:** `{integrand}`")
        st.markdown(f"**Result:** `{result}`")
        copy_button(str(result), key="vector-line-integral")

        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except Exception:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing vector line integral: {str(e)}")

def compute_scalar_surface_integral(scalar_field, surface_equation, x_min, x_max, y_min, y_max):
    """Compute surface integral of scalar field"""
    try:
        def _compute_scalar_surface():
            f_expr = expression_parser.parse(scalar_field)
            z_expr = expression_parser.parse(surface_equation)

            x, y, z = sp.symbols('x y z')

            f_surface = f_expr.subs(z, z_expr)

            dz_dx = sp.diff(z_expr, x)
            dz_dy = sp.diff(z_expr, y)

            dS = sp.sqrt(1 + dz_dx**2 + dz_dy**2)
            integrand = f_surface * dS

            result_val = sp.integrate(integrand, (x, x_min, x_max), (y, y_min, y_max))
            return dS, integrand, result_val

        (dS, integrand, result), _, error = run_task(
            "Computing scalar surface integral...",
            _compute_scalar_surface,
            success_message="Scalar surface integral ready",
            error_message="Scalar surface integral failed"
        )
        if error:
            return

        st.markdown(f"**Surface element:** `dS = {dS} dx dy`")
        st.markdown(f"**Integrand:** `{integrand}`")
        st.markdown(f"**Result:** `{result}`")
        copy_button(str(result), key="scalar-surface-integral")

        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except Exception:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing scalar surface integral: {str(e)}")

def compute_flux_integral(F_x, F_y, F_z, surface_equation, x_min, x_max, y_min, y_max):
    """Compute flux integral through surface"""
    try:
        def _compute_flux():
            fx_expr = expression_parser.parse(F_x) if F_x else 0
            fy_expr = expression_parser.parse(F_y) if F_y else 0
            fz_expr = expression_parser.parse(F_z) if F_z else 0
            z_expr = expression_parser.parse(surface_equation)

            x, y, z = sp.symbols('x y z')

            fx_surface = fx_expr.subs(z, z_expr)
            fy_surface = fy_expr.subs(z, z_expr)
            fz_surface = fz_expr.subs(z, z_expr)

            dz_dx = sp.diff(z_expr, x)
            dz_dy = sp.diff(z_expr, y)

            flux_density = -fx_surface * dz_dx - fy_surface * dz_dy + fz_surface
            result_val = sp.integrate(flux_density, (x, x_min, x_max), (y, y_min, y_max))
            return (-dz_dx, -dz_dy, 1), flux_density, result_val

        (normal_vec, flux_density, result), _, error = run_task(
            "Computing flux integral...",
            _compute_flux,
            success_message="Flux integral ready",
            error_message="Flux integral failed"
        )
        if error:
            return

        st.markdown(f"**Normal vector:** `n̂ = {normal_vec}`")
        st.markdown(f"**F⃗·n̂:** `{flux_density}`")
        st.markdown(f"**Result:** `{result}`")
        copy_button(str(result), key="flux-integral")

        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except Exception:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing flux integral: {str(e)}")

def verify_greens_theorem(P_expr, Q_expr):
    """Verify Green's theorem for a simple closed curve"""
    try:
        # Parse expressions
        P = expression_parser.parse(P_expr)
        Q = expression_parser.parse(Q_expr)
        
        x, y = sp.symbols('x y')
        
        # Compute curl (∂Q/∂x - ∂P/∂y)
        curl_2d = sp.diff(Q, x) - sp.diff(P, y)
        
        st.markdown(f"**P(x,y):** `{P}`")
        st.markdown(f"**Q(x,y):** `{Q}`")
        st.markdown(f"**∂Q/∂x - ∂P/∂y:** `{curl_2d}`")
        
        # For unit circle as example
        st.markdown("**Example verification for unit circle:**")
        
        # Double integral over unit disk
        # Convert to polar coordinates: x = r cos θ, y = r sin θ
        r, theta = sp.symbols('r theta', real=True, positive=True)
        curl_polar = curl_2d.subs([(x, r*sp.cos(theta)), (y, r*sp.sin(theta))])
        
        # Jacobian for polar coordinates is r
        double_integral = sp.integrate(curl_polar * r, (r, 0, 1), (theta, 0, 2*sp.pi))
        
        st.markdown(f"**Double integral (unit disk):** `{double_integral}`")
        
        # Line integral around unit circle
        # Parametric: x = cos(t), y = sin(t), t ∈ [0, 2π]
        t = sp.Symbol('t')
        x_param = sp.cos(t)
        y_param = sp.sin(t)
        
        P_param = P.subs([(x, x_param), (y, y_param)])
        Q_param = Q.subs([(x, x_param), (y, y_param)])
        
        dx_dt = sp.diff(x_param, t)
        dy_dt = sp.diff(y_param, t)
        
        line_integrand = P_param * dx_dt + Q_param * dy_dt
        line_integral = sp.integrate(line_integrand, (t, 0, 2*sp.pi))
        
        st.markdown(f"**Line integral (unit circle):** `{line_integral}`")
        
        # Check if they're equal
        if sp.simplify(double_integral - line_integral) == 0:
            st.success("**Green's theorem verified!** Both integrals are equal.")
        else:
            st.info("**Computed both integrals.** Check if they're numerically equal.")
            
    except Exception as e:
        st.error(f"Error verifying Green's theorem: {str(e)}")



def render_differential_equations_section():
    """Render differential equations section"""
    st.subheader("Differential Equations Solver")
    st.markdown("Advanced ODE solver with multiple numerical methods")
    
    # Initialize solver
    ode_solver = AdvancedODESolver()
    
    # Equation type selection
    eq_type = st.selectbox(
        "Equation Type",
        ["First Order ODE", "Second Order ODE", "System of ODEs", "Boundary Value Problem", "Partial Differential Equation"]
    )
    
    if eq_type == "First Order ODE":
        render_first_order_ode(ode_solver)
    elif eq_type == "Second Order ODE":
        render_second_order_ode(ode_solver)
    elif eq_type == "System of ODEs":
        render_system_odes(ode_solver)
    elif eq_type == "Boundary Value Problem":
        render_bvp(ode_solver)
    else:
        render_pde(ode_solver)

def render_first_order_ode(solver):
    """Render first order ODE solver"""
    col1, col2 = st.columns(2)
    
    with col1:
        # ODE input
        st.markdown("**Differential Equation**")
        ode_str = st.text_input(
            "dy/dt = f(t, y)",
            value="-0.5 * y",
            help="Enter the right-hand side of dy/dt = f(t, y)"
        )
        
        # Initial conditions
        st.markdown("**Initial Conditions**")
        y0 = st.number_input("y(t₀)", value=1.0, step=0.1)
        t0 = st.number_input("t₀", value=0.0, step=0.1)
        
    with col2:
        # Solution parameters
        st.markdown("**Solution Parameters**")
        tf = st.number_input("Final time", value=10.0, min_value=t0+0.1, step=0.5)
        
        # Method selection
        method = st.selectbox(
            "Numerical Method",
            [m.value for m in SolverMethod]
        )
        
        num_points = st.slider("Number of points", 100, 5000, 1000)
    
    if st.button("Solve ODE", key="solve_first_order"):
        try:
            # Create ODE function
            def ode_func(t, y):
                # Create local variables for evaluation
                local_dict = {'t': t, 'y': y, 'np': np, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos}
                return eval(ode_str, {"__builtins__": {}}, local_dict)
            
            # Create problem
            problem = ODEProblem(
                equation=ode_func,
                initial_conditions={'y0': y0},
                domain=(t0, tf),
                ode_type=ODEType.FIRST_ORDER
            )
            
            # Solve
            solution = solver.solve_ode(
                problem,
                method=SolverMethod[method],
                num_points=num_points
            )
            
            if solution.success:
                # Display solution
                st.success(f"Solution computed successfully using {method}")
                
                # Plot solution
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=solution.t,
                    y=solution.y,
                    mode='lines',
                    name='y(t)',
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title="Solution of dy/dt = " + ode_str,
                    xaxis_title="t",
                    yaxis_title="y(t)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Display some solution values
                with st.expander("Solution Values"):
                    indices = np.linspace(0, len(solution.t)-1, min(10, len(solution.t)), dtype=int)
                    data = {
                        "t": [solution.t[i] for i in indices],
                        "y(t)": [solution.y[i] for i in indices]
                    }
                    st.table(data)
                
                # Error estimate if available
                if solution.error_estimate is not None:
                    st.info(f"Estimated error: {np.max(np.abs(solution.error_estimate)):.2e}")
            else:
                st.error(f"Failed to solve ODE: {solution.message}")
                
        except Exception as e:
            st.error(f"Error solving ODE: {str(e)}")

def render_second_order_ode(solver):
    """Render second order ODE solver"""
    st.info("Convert second order ODE y'' = f(t, y, y') to system of first order ODEs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ODE input
        st.markdown("**Differential Equation**")
        ode_str = st.text_input(
            "y'' = f(t, y, y')",
            value="-y - 0.1*dy",
            help="Enter expression for y''. Use 'y' for y and 'dy' for y'"
        )
        
        # Initial conditions
        st.markdown("**Initial Conditions**")
        y0 = st.number_input("y(0)", value=1.0, step=0.1)
        dy0 = st.number_input("y'(0)", value=0.0, step=0.1)
        
    with col2:
        # Solution parameters
        st.markdown("**Solution Parameters**")
        tf = st.number_input("Final time", value=20.0, min_value=0.1, step=0.5)
        
        # Method selection
        method = st.selectbox(
            "Numerical Method",
            ["RK45", "DOP853", "BDF", "RADAU"],
            key="method_second"
        )
    
    if st.button("Solve Second Order ODE", key="solve_second_order"):
        try:
            # Convert to system of first order ODEs
            def system(t, state):
                y, dy = state
                # Evaluate second derivative
                local_dict = {'t': t, 'y': y, 'dy': dy, 'np': np, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos}
                d2y = eval(ode_str, {"__builtins__": {}}, local_dict)
                return [dy, d2y]
            
            # Solve system
            solution = solver.solve_system(
                [lambda t, s: s[1], lambda t, s: system(t, s)[1]],
                initial_conditions=np.array([y0, dy0]),
                t_span=(0, tf),
                method=method
            )
            
            if solution.success:
                st.success(f"Solution computed successfully using {method}")
                
                # Plot solution
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=solution.t,
                    y=solution.y[:, 0],
                    mode='lines',
                    name='y(t)',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=solution.t,
                    y=solution.y[:, 1],
                    mode='lines',
                    name="y'(t)",
                    line=dict(width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Solution of y'' = " + ode_str,
                    xaxis_title="t",
                    yaxis_title="Value",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Phase portrait
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=solution.y[:, 0],
                    y=solution.y[:, 1],
                    mode='lines',
                    name='Phase Portrait',
                    line=dict(width=2)
                ))
                
                fig2.update_layout(
                    title="Phase Portrait",
                    xaxis_title="y",
                    yaxis_title="y'",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig2, width='stretch')
            else:
                st.error(f"Failed to solve ODE: {solution.message}")
                
        except Exception as e:
            st.error(f"Error solving ODE: {str(e)}")

def render_system_odes(solver):
    """Render system of ODEs solver"""
    st.markdown("**System of ODEs**")
    
    # Predefined examples
    example = st.selectbox(
        "Select Example",
        ["Custom", "Lorenz System", "Van der Pol Oscillator", "Predator-Prey Model"]
    )
    
    if example == "Lorenz System":
        st.latex(r"\begin{cases} \dot{x} = \sigma(y - x) \\ \dot{y} = x(\rho - z) - y \\ \dot{z} = xy - \beta z \end{cases}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma = st.number_input("σ", value=10.0, step=0.5)
        with col2:
            rho = st.number_input("ρ", value=28.0, step=0.5)
        with col3:
            beta = st.number_input("β", value=8/3, step=0.1)
        
        initial = st.text_input("Initial conditions [x, y, z]", value="[1.0, 1.0, 1.0]")
        
        if st.button("Solve Lorenz System"):
            try:
                from core.advanced_ode_solver import lorenz_system
                
                # Parse initial conditions
                init_vals = eval(initial)
                
                # Create system function
                def system_func(t, state):
                    return lorenz_system(t, state, sigma, rho, beta)
                
                # Solve
                solution = solver.solve_system(
                    [lambda t, s: system_func(t, s)[i] for i in range(3)],
                    initial_conditions=np.array(init_vals),
                    t_span=(0, 50),
                    method='RK45'
                )
                
                if solution.success:
                    st.success("Lorenz system solved successfully!")
                    
                    # 3D trajectory
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=solution.y[:, 0],
                        y=solution.y[:, 1],
                        z=solution.y[:, 2],
                        mode='lines',
                        line=dict(width=2, color=solution.t, colorscale='Viridis'),
                        name='Trajectory'
                    ))
                    
                    fig.update_layout(
                        title="Lorenz Attractor",
                        scene=dict(
                            xaxis_title="x",
                            yaxis_title="y",
                            zaxis_title="z"
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Time series
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 0], name='x(t)'))
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 1], name='y(t)'))
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 2], name='z(t)'))
                    
                    fig2.update_layout(
                        title="Time Series",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, width='stretch')
                    
            except Exception as e:
                st.error(f"Error solving system: {str(e)}")
    
    elif example == "Van der Pol Oscillator":
        st.latex(r"\begin{cases} \dot{x} = y \\ \dot{y} = \mu(1 - x^2)y - x \end{cases}")
        
        mu = st.number_input("μ", value=1.0, min_value=0.0, step=0.1)
        initial = st.text_input("Initial conditions [x, y]", value="[1.0, 0.0]")
        
        if st.button("Solve Van der Pol"):
            try:
                from core.advanced_ode_solver import van_der_pol
                
                # Parse initial conditions
                init_vals = eval(initial)
                
                # Create system function
                def system_func(t, state):
                    return van_der_pol(t, state, mu)
                
                # Solve
                solution = solver.solve_system(
                    [lambda t, s: system_func(t, s)[i] for i in range(2)],
                    initial_conditions=np.array(init_vals),
                    t_span=(0, 50),
                    method='RK45'
                )
                
                if solution.success:
                    st.success("Van der Pol oscillator solved successfully!")
                    
                    # Phase portrait
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=solution.y[:, 0],
                        y=solution.y[:, 1],
                        mode='lines',
                        line=dict(width=2),
                        name='Phase Portrait'
                    ))
                    
                    fig.update_layout(
                        title=f"Van der Pol Phase Portrait (μ = {mu})",
                        xaxis_title="x",
                        yaxis_title="y",
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Time series
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 0], name='x(t)'))
                    
                    fig2.update_layout(
                        title="Position vs Time",
                        xaxis_title="Time",
                        yaxis_title="Position",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, width='stretch')
                    
            except Exception as e:
                st.error(f"Error solving system: {str(e)}")
    
    else:
        st.info("Custom system implementation - enter your system of ODEs")

def render_bvp(solver):
    """Render boundary value problem solver"""
    st.markdown("**Boundary Value Problem**")
    st.info("Solve second order BVP: y'' = f(x, y, y') with boundary conditions")
    
    # Not fully implemented in the basic UI
    st.warning("BVP solver interface coming soon. Use the API directly for now.")

def render_pde(solver):
    """Render partial differential equation solver"""
    st.markdown("**Partial Differential Equations**")
    
    pde_type = st.selectbox(
        "PDE Type",
        ["Heat Equation", "Wave Equation"]
    )
    
    if pde_type == "Heat Equation":
        st.latex(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parameters
            alpha = st.number_input("Thermal diffusivity α", value=1.0, min_value=0.1, step=0.1)
            L = st.number_input("Domain length", value=1.0, min_value=0.1, step=0.1)
            
            # Initial condition
            ic_type = st.selectbox("Initial condition", ["Sine wave", "Step function", "Gaussian"])
            
        with col2:
            # Boundary conditions
            bc_left = st.number_input("Left boundary (u(0,t))", value=0.0, step=0.1)
            bc_right = st.number_input("Right boundary (u(L,t))", value=0.0, step=0.1)
            
            # Time
            t_max = st.number_input("Final time", value=0.5, min_value=0.01, step=0.05)
        
        if st.button("Solve Heat Equation"):
            try:
                # Define initial condition
                if ic_type == "Sine wave":
                    ic_func = lambda x: np.sin(np.pi * x / L)
                elif ic_type == "Step function":
                    ic_func = lambda x: np.where(x < L/2, 1.0, 0.0)
                else:  # Gaussian
                    ic_func = lambda x: np.exp(-50*(x - L/2)**2)
                
                # Solve PDE
                solution = solver.solve_pde_heat_equation(
                    initial_condition=ic_func,
                    boundary_conditions=(bc_left, bc_right),
                    x_domain=(0, L),
                    t_domain=(0, t_max),
                    nx=100,
                    nt=1000,
                    alpha=alpha
                )
                
                st.success("Heat equation solved successfully!")
                
                # Create animation frames
                frames = []
                for i in range(0, len(solution['t']), max(1, len(solution['t'])//50)):
                    frames.append(go.Frame(
                        data=[go.Scatter(
                            x=solution['x'],
                            y=solution['u'][i, :],
                            mode='lines',
                            line=dict(width=2)
                        )],
                        name=f"t={solution['t'][i]:.3f}"
                    ))
                
                # Create figure
                fig = go.Figure(
                    data=[go.Scatter(
                        x=solution['x'],
                        y=solution['u'][0, :],
                        mode='lines',
                        line=dict(width=2)
                    )],
                    frames=frames
                )
                
                # Add play button
                fig.update_layout(
                    title="Heat Equation Solution",
                    xaxis_title="x",
                    yaxis_title="u(x,t)",
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 50}}]},
                            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                        ]
                    }],
                    sliders=[{
                        'active': 0,
                        'steps': [{'args': [[f.name], {'frame': {'duration': 0}, 'mode': 'immediate'}], 
                                  'label': f"t={solution['t'][i]:.3f}", 'method': 'animate'} 
                                 for i, f in enumerate(frames)]
                    }]
                )
                
                st.plotly_chart(fig, width='stretch')
                
                if solution['stability_parameter'] > 0.5:
                    st.warning(f"Warning: Stability parameter r = {solution['stability_parameter']:.3f} > 0.5. Solution may be unstable.")
                
            except Exception as e:
                st.error(f"Error solving PDE: {str(e)}")


def render_transforms_series_section():
    """Render transforms and series analysis section"""
    st.subheader("Transforms & Series Analysis")
    
    st.info("""
    This tool provides comprehensive transform and series analysis including:
    - Laplace and Fourier transforms
    - Transfer function calculation from differential equations
    - Differential equation solving using transforms
    - Solution decomposition (transient/steady-state, zero-input/zero-state)
    - Fourier series expansion
    """)
    
    # Sub-tabs for different operations
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "Laplace Transform", "Fourier Transform", "Transfer Functions", "DE Solving", "Fourier Series"
    ])
    
    with subtab1:
        render_laplace_transform_tab()
    
    with subtab2:
        render_fourier_transform_tab()
    
    with subtab3:
        render_transfer_function_tab()
    
    with subtab4:
        render_de_solving_tab()
    
    with subtab5:
        render_fourier_series_tab()


def render_laplace_transform_tab():
    """Render Laplace transform section"""
    st.markdown("### Laplace Transform")
    
    transform_type = st.radio(
        "Transform Type:",
        ["Forward Transform", "Inverse Transform"],
        key="laplace_type",
        horizontal=True
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    if transform_type == "Forward Transform":
        with col1:
            expression = st.text_input(
                "Time Domain Function f(t):",
                placeholder="sin(t), exp(-t), t**2",
                key="laplace_forward_expr"
            )
        with col2:
            t_var = st.text_input("Time Variable:", value="t", key="laplace_t_var")
        with col3:
            s_var = st.text_input("Laplace Variable:", value="s", key="laplace_s_var")
        
        if expression:
            if st.button("Compute Laplace Transform", key="compute_laplace"):
                with st.spinner("Computing Laplace transform..."):
                    result = transforms_series_engine.laplace_transform(expression, t_var, s_var)
                    
                    if result['success']:
                        st.success("Laplace transform computed successfully")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Time Domain f(t):**")
                            st.latex(sp.latex(result['original_expression']))
                        
                        with col2:
                            st.markdown(f"**Laplace Domain F({s_var}):**")
                            st.latex(sp.latex(result['transform']))
                        
                        st.markdown("**LaTeX Code:**")
                        copy_button(sp.latex(result['transform']), "laplace_result", "Copy LaTeX")
                    else:
                        st.error(f"Transform failed: {result['error']}")
    
    else:  # Inverse Transform
        with col1:
            expression = st.text_input(
                "Laplace Domain Function F(s):",
                placeholder="1/(s+1), s/(s**2+1)",
                key="laplace_inverse_expr"
            )
        with col2:
            s_var = st.text_input("Laplace Variable:", value="s", key="laplace_inv_s_var")
        with col3:
            t_var = st.text_input("Time Variable:", value="t", key="laplace_inv_t_var")
        
        if expression:
            if st.button("Compute Inverse Laplace Transform", key="compute_inv_laplace"):
                with st.spinner("Computing inverse Laplace transform..."):
                    result = transforms_series_engine.inverse_laplace_transform(expression, s_var, t_var)
                    
                    if result['success']:
                        st.success("Inverse Laplace transform computed successfully")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Laplace Domain F({s_var}):**")
                            st.latex(sp.latex(result['original_expression']))
                        
                        with col2:
                            st.markdown("**Time Domain f(t):**")
                            st.latex(sp.latex(result['transform']))
                        
                        st.markdown("**LaTeX Code:**")
                        copy_button(sp.latex(result['transform']), "inv_laplace_result", "Copy LaTeX")
                    else:
                        st.error(f"Inverse transform failed: {result['error']}")


def render_fourier_transform_tab():
    """Render Fourier transform section"""
    st.markdown("### Fourier Transform")
    
    transform_type = st.radio(
        "Transform Type:",
        ["Forward Transform", "Inverse Transform"],
        key="fourier_type",
        horizontal=True
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    if transform_type == "Forward Transform":
        with col1:
            expression = st.text_input(
                "Time Domain Function f(t):",
                placeholder="exp(-abs(t)), exp(-t**2)",
                key="fourier_forward_expr"
            )
        with col2:
            t_var = st.text_input("Time Variable:", value="t", key="fourier_t_var")
        with col3:
            omega_var = st.text_input("Frequency Variable:", value="omega", key="fourier_omega_var")
        
        if expression:
            if st.button("Compute Fourier Transform", key="compute_fourier"):
                with st.spinner("Computing Fourier transform..."):
                    result = transforms_series_engine.fourier_transform(expression, t_var, omega_var)
                    
                    if result['success']:
                        st.success("Fourier transform computed successfully")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Time Domain f(t):**")
                            st.latex(sp.latex(result['original_expression']))
                        
                        with col2:
                            st.markdown(f"**Frequency Domain F({omega_var}):**")
                            st.latex(sp.latex(result['transform']))
                        
                        st.markdown("**LaTeX Code:**")
                        copy_button(sp.latex(result['transform']), "fourier_result", "Copy LaTeX")
                    else:
                        st.error(f"Transform failed: {result['error']}")
    
    else:  # Inverse Transform
        with col1:
            expression = st.text_input(
                "Frequency Domain Function F(omega):",
                placeholder="1/(1+omega**2)",
                key="fourier_inverse_expr"
            )
        with col2:
            omega_var = st.text_input("Frequency Variable:", value="omega", key="fourier_inv_omega_var")
        with col3:
            t_var = st.text_input("Time Variable:", value="t", key="fourier_inv_t_var")
        
        if expression:
            if st.button("Compute Inverse Fourier Transform", key="compute_inv_fourier"):
                with st.spinner("Computing inverse Fourier transform..."):
                    result = transforms_series_engine.inverse_fourier_transform(expression, omega_var, t_var)
                    
                    if result['success']:
                        st.success("Inverse Fourier transform computed successfully")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Frequency Domain F({omega_var}):**")
                            st.latex(sp.latex(result['original_expression']))
                        
                        with col2:
                            st.markdown("**Time Domain f(t):**")
                            st.latex(sp.latex(result['transform']))
                        
                        st.markdown("**LaTeX Code:**")
                        copy_button(sp.latex(result['transform']), "inv_fourier_result", "Copy LaTeX")
                    else:
                        st.error(f"Inverse transform failed: {result['error']}")


def render_transfer_function_tab():
    """Render transfer function analysis section"""
    st.markdown("### Transfer Function Analysis")
    
    st.markdown("**Analyze Transfer Function:**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        transfer_function = st.text_input(
            "Transfer Function H(s):",
            placeholder="1/(s**2 + 2*s + 1), (s+1)/(s**2 + 3*s + 2)",
            key="transfer_func"
        )
    with col2:
        s_var = st.text_input("Laplace Variable:", value="s", key="transfer_s_var")
    
    if transfer_function:
        if st.button("Analyze Transfer Function", key="analyze_tf"):
            with st.spinner("Analyzing transfer function..."):
                result = transforms_series_engine.analyze_frequency_response(transfer_function, s_var)
                
                if result['success']:
                    st.success("Transfer function analyzed successfully")
                    
                    st.markdown("**Transfer Function:**")
                    st.latex(f"H(s) = {sp.latex(result['transfer_function'])}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Poles:**")
                        if result['poles']:
                            for i, pole in enumerate(result['poles']):
                                st.latex(f"p_{i+1} = {sp.latex(pole)}")
                        else:
                            st.write("No poles")
                    
                    with col2:
                        st.markdown("**Zeros:**")
                        if result['zeros']:
                            for i, zero in enumerate(result['zeros']):
                                st.latex(f"z_{i+1} = {sp.latex(zero)}")
                        else:
                            st.write("No zeros")
                    
                    with col3:
                        st.markdown("**Stability:**")
                        if result['stable']:
                            st.success("System is STABLE")
                        else:
                            st.error("System is UNSTABLE")
                    
                    st.markdown("---")
                    st.markdown("**Frequency Response:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Magnitude:**")
                        st.latex(f"|H(j\\omega)| = {sp.latex(result['magnitude'])}")
                    
                    with col2:
                        st.markdown("**Phase:**")
                        st.latex(f"\\angle H(j\\omega) = {sp.latex(result['phase'])}")
                else:
                    st.error(f"Analysis failed: {result['error']}")
    
    st.markdown("---")
    st.markdown("**Compute Transfer Function from Differential Equation:**")
    st.info("Note: This feature provides basic transfer function computation. For complex equations, manual conversion may be needed.")


def render_de_solving_tab():
    """Render differential equation solving with transforms"""
    st.markdown("### Solve Differential Equations Using Transforms")
    
    method = st.radio(
        "Transform Method:",
        ["Laplace Transform", "Fourier Transform"],
        key="de_method",
        horizontal=True
    )
    
    if method == "Laplace Transform":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            de_expr = st.text_input(
                "Differential Equation:",
                placeholder="Derivative(y(t), t, 2) + 2*Derivative(y(t), t) + y(t) - u(t)",
                key="de_laplace_expr",
                help="Enter as equation equal to 0, or just the left side"
            )
        
        with col2:
            t_var = st.text_input("Time Variable:", value="t", key="de_t_var")
        
        st.markdown("**Input Function:**")
        input_func = st.text_input(
            "Input u(t):",
            placeholder="1 (step), sin(t), exp(-t)",
            value="1",
            key="de_input"
        )
        
        st.markdown("**Initial Conditions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            y0 = st.number_input("y(0):", value=0.0, key="de_y0")
        with col2:
            y0_prime = st.number_input("y'(0):", value=0.0, key="de_y0_prime")
        with col3:
            use_higher = st.checkbox("Higher order ICs", key="de_higher_ic")
        
        initial_conditions = {
            'y(0)': y0,
            "y'(0)": y0_prime
        }
        
        if use_higher:
            y0_double_prime = st.number_input("y''(0):", value=0.0, key="de_y0_double_prime")
            initial_conditions["y''(0)"] = y0_double_prime
        
        if de_expr and input_func:
            if st.button("Solve Using Laplace Transform", key="solve_de_laplace"):
                with st.spinner("Solving differential equation..."):
                    result = transforms_series_engine.solve_de_with_laplace(
                        de_expr, initial_conditions, input_func, t_var
                    )
                    
                    if result['success']:
                        st.success("Differential equation solved successfully")
                        
                        if result['solution']:
                            st.markdown("**Complete Solution:**")
                            st.latex(f"y(t) = {sp.latex(result['solution'])}")
                        
                        st.markdown("---")
                        st.markdown("**Solution Decomposition:**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Transient Response:**")
                            if result['transient']:
                                st.latex(f"y_{{tr}}(t) = {sp.latex(result['transient'])}")
                                st.info("Decays to zero as t → ∞")
                            else:
                                st.write("No transient component")
                        
                        with col2:
                            st.markdown("**Steady-State Response:**")
                            if result['steady_state']:
                                st.latex(f"y_{{ss}}(t) = {sp.latex(result['steady_state'])}")
                                st.info("Persists as t → ∞")
                            else:
                                st.write("No steady-state component")
                        
                        if result['zero_input']:
                            st.markdown("---")
                            st.markdown("**Zero-Input Response:**")
                            st.latex(f"y_{{zi}}(t) = {sp.latex(result['zero_input'])}")
                        
                        if result['zero_state']:
                            st.markdown("**Zero-State Response:**")
                            st.latex(f"y_{{zs}}(t) = {sp.latex(result['zero_state'])}")
                    else:
                        st.error(f"Solving failed: {result['error']}")
    
    else:  # Fourier Transform
        st.warning("⚠️ **Fourier Transform DE Solving Not Yet Implemented**")
        st.info("""
        Fourier transform method for differential equations is best suited for spatial problems with boundary conditions.
        
        **Current Status:** This feature is under development and not yet available.
        
        **Recommendation:** Please use the **Laplace Transform** method for time-domain differential equations.
        """)
        
        # Disabled UI elements to show what's planned
        st.markdown("**Planned Features (Coming Soon):**")
        st.text("• Spatial domain differential equations")
        st.text("• Boundary condition handling")
        st.text("• Frequency domain analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.text_input(
                "Differential Equation:",
                placeholder="Feature under development",
                key="de_fourier_expr",
                disabled=True
            )
        
        with col2:
            st.text_input("Spatial Variable:", value="x", key="de_x_var", disabled=True)
        
        st.button("Solve Using Fourier Transform", key="solve_de_fourier", disabled=True)


def render_fourier_series_tab():
    """Render Fourier series expansion section"""
    st.markdown("### Fourier Series Expansion")
    
    st.info("Expand periodic functions as Fourier series with sine and cosine terms.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Periodic Function f(x):",
            placeholder="x, x**2, abs(x)",
            key="fourier_series_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x", key="fourier_series_var")
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.number_input(
            "Period:",
            value=2*np.pi,
            min_value=0.1,
            key="fourier_series_period",
            help="Period of the function (default: 2π)"
        )
    
    with col2:
        n_terms = st.number_input(
            "Number of Terms:",
            value=5,
            min_value=1,
            max_value=20,
            key="fourier_series_terms"
        )
    
    if expression:
        if st.button("Compute Fourier Series", key="compute_fourier_series"):
            with st.spinner("Computing Fourier series..."):
                result = transforms_series_engine.compute_fourier_series(
                    expression, variable, period, n_terms
                )
                
                if result['success']:
                    st.success(f"Fourier series computed with {n_terms} terms")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Function:**")
                        st.latex(f"f({variable}) = {sp.latex(result['original_expression'])}")
                    
                    with col2:
                        st.markdown("**Period:**")
                        st.metric("T", f"{period:.4g}")
                    
                    st.markdown("---")
                    st.markdown("**Fourier Series (Trigonometric Form):**")
                    st.latex(sp.latex(result['fourier_series']))
                    
                    st.markdown("---")
                    st.markdown("**Coefficients:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**a₀ (DC component):**")
                        st.latex(f"a_0 = {sp.latex(result['a0'])}")
                    
                    with col2:
                        st.markdown("**aₙ (cosine coefficients):**")
                        for n, coeff in list(result['an_coefficients'].items())[:5]:
                            st.latex(f"a_{n} = {sp.latex(coeff)}")
                    
                    with col3:
                        st.markdown("**bₙ (sine coefficients):**")
                        for n, coeff in list(result['bn_coefficients'].items())[:5]:
                            st.latex(f"b_{n} = {sp.latex(coeff)}")
                    
                    st.markdown("---")
                    st.markdown("**Complex Exponential Form:**")
                    st.latex(sp.latex(result['complex_form']))
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("**Series Approximation Visualization:**")
                    try:
                        visualize_fourier_series(
                            expression, variable, period, n_terms, result['fourier_series']
                        )
                    except Exception as e:
                        st.warning(f"Visualization not available: {str(e)}")
                else:
                    st.error(f"Fourier series computation failed: {result['error']}")


def visualize_fourier_series(original_expr: str, variable: str, period: float, 
                            n_terms: int, fourier_series):
    """Visualize Fourier series approximation"""
    try:
        # Parse expressions
        original = expression_parser.parse(original_expr)
        var_sym = sp.Symbol(variable, real=True)
        
        # Create numerical functions
        L = period / 2
        x_vals = np.linspace(-L, L, 1000)
        
        # Original function
        try:
            f_original = sp.lambdify(var_sym, original, 'numpy')
            y_original = f_original(x_vals)
        except:
            st.warning("Cannot plot original function")
            return
        
        # Fourier series
        f_series = sp.lambdify(var_sym, fourier_series, 'numpy')
        y_series = f_series(x_vals)
        
        # Create plot
        fig = go.Figure()
        
        # Original function
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_original,
            mode='lines',
            name='Original Function',
            line=dict(color='blue', width=3)
        ))
        
        # Fourier series approximation
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_series,
            mode='lines',
            name=f'Fourier Series ({n_terms} terms)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Fourier Series Approximation with {n_terms} Terms",
            xaxis_title=variable,
            yaxis_title=f"f({variable})",
            hovermode='x unified',
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        error = np.abs(y_original - y_series)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maximum Error", f"{max_error:.6f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mean_error:.6f}")
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")


