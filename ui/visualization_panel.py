import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from core.expression_parser import expression_parser
from core.calculation_engine import calculation_engine
from utils.ui_helpers import run_task, copy_button
import math

def render_visualization_panel():
    """Render the function visualization panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Function Visualization</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Interactive 2D/3D function plotting with advanced visualization features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Visualization Type:",
        [
            "2D Function Plot",
            "3D Surface Plot", 
            "Parametric Plot",
            "Polar Plot",
            "Vector Field",
            "Contour Plot",
            "Animation"
        ]
    )
    
    if viz_type == "2D Function Plot":
        render_2d_function_plot()
    elif viz_type == "3D Surface Plot":
        render_3d_surface_plot()
    elif viz_type == "Parametric Plot":
        render_parametric_plot()
    elif viz_type == "Polar Plot":
        render_polar_plot()
    elif viz_type == "Vector Field":
        render_vector_field()
    elif viz_type == "Contour Plot":
        render_contour_plot()
    elif viz_type == "Animation":
        render_animation_plot()

def render_2d_function_plot():
    """Render 2D function plotting interface"""
    st.subheader("2D Function Plotting")
    
    # Function input section
    st.markdown("**Function Input:**")
    
    # Support multiple functions
    num_functions = st.number_input("Number of functions:", min_value=1, max_value=10, value=1)
    
    functions = []
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i in range(num_functions):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            func = st.text_input(
                f"Function {i+1}:",
                placeholder="sin(x) + cos(2*x)",
                key=f"func_2d_{i}"
            )
            functions.append(func)
        
        with col2:
            line_style = st.selectbox(
                "Style:",
                ["solid", "dash", "dot", "dashdot"],
                key=f"style_2d_{i}"
            )
        
        with col3:
            color = st.selectbox(
                "Color:",
                colors,
                index=i % len(colors),
                key=f"color_2d_{i}"
            )
    
    # Plot parameters
    st.markdown("**Plot Parameters:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_min = st.number_input("X Min:", value=-10.0, key="x_min_2d")
    with col2:
        x_max = st.number_input("X Max:", value=10.0, key="x_max_2d")
    with col3:
        num_points = st.number_input("Points:", min_value=100, max_value=10000, value=1000, key="points_2d")
    with col4:
        variable = st.text_input("Variable:", value="x", key="var_2d")
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            show_grid = st.checkbox("Show Grid", True)
            show_axes = st.checkbox("Show Axes", True)
            show_legend = st.checkbox("Show Legend", True)
        
        with col2:
            log_scale_x = st.checkbox("Log Scale X", False)
            log_scale_y = st.checkbox("Log Scale Y", False)
            equal_aspect = st.checkbox("Equal Aspect Ratio", False)
    
    # Plot functions
    if st.button("Plot Functions", key="plot_2d"):
        plot_2d_functions(functions, variable, x_min, x_max, num_points, 
                         colors, show_grid, show_axes, show_legend, 
                         log_scale_x, log_scale_y, equal_aspect)
    
    # Function analysis
    if functions[0]:  # If at least one function is provided
        st.markdown("---")
        st.subheader("Function Analysis")
        render_function_analysis_viz(functions[0], variable, x_min, x_max)

def render_3d_surface_plot():
    """Render 3D surface plotting interface"""
    st.subheader("3D Surface Plotting")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        function = st.text_input(
            "Function f(x,y):",
            placeholder="sin(sqrt(x**2 + y**2))",
            key="func_3d"
        )
    
    with col2:
        plot_type = st.selectbox(
            "Plot Type:",
            ["Surface", "Wireframe", "Scatter"]
        )
    
    # Range parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_min = st.number_input("X Min:", value=-5.0, key="x_min_3d")
    with col2:
        x_max = st.number_input("X Max:", value=5.0, key="x_max_3d")
    with col3:
        y_min = st.number_input("Y Min:", value=-5.0, key="y_min_3d")
    with col4:
        y_max = st.number_input("Y Max:", value=5.0, key="y_max_3d")
    
    col1, col2 = st.columns(2)
    
    with col1:
        resolution = st.slider("Resolution:", 20, 200, 50)
    with col2:
        colorscale = st.selectbox(
            "Color Scale:",
            ["Viridis", "Plasma", "Inferno", "Magma", "Rainbow", "Turbo"]
        )
    
    if function and st.button("Plot 3D Surface", key="plot_3d"):
        plot_3d_surface(function, x_min, x_max, y_min, y_max, resolution, 
                       plot_type, colorscale)

def render_parametric_plot():
    """Render parametric plotting interface"""
    st.subheader("Parametric Plotting")
    
    plot_dim = st.radio("Dimension:", ["2D", "3D"])
    
    if plot_dim == "2D":
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.text_input("x(t):", placeholder="cos(t)", key="x_param_2d")
        with col2:
            y_param = st.text_input("y(t):", placeholder="sin(t)", key="y_param_2d")
    
    else:  # 3D
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_param = st.text_input("x(t):", placeholder="cos(t)", key="x_param_3d")
        with col2:
            y_param = st.text_input("y(t):", placeholder="sin(t)", key="y_param_3d")
        with col3:
            z_param = st.text_input("z(t):", placeholder="t", key="z_param_3d")
    
    # Parameter range
    col1, col2, col3 = st.columns(3)
    
    with col1:
        t_min = st.number_input("t Min:", value=0.0, key="t_min_param")
    with col2:
        t_max = st.number_input("t Max:", value=2*math.pi, key="t_max_param")
    with col3:
        num_points = st.number_input("Points:", min_value=100, max_value=5000, value=1000, key="points_param")
    
    if plot_dim == "2D":
        if x_param and y_param and st.button("Plot Parametric 2D", key="plot_param_2d"):
            plot_parametric_2d(x_param, y_param, t_min, t_max, num_points)
    else:
        if x_param and y_param and z_param and st.button("Plot Parametric 3D", key="plot_param_3d"):
            plot_parametric_3d(x_param, y_param, z_param, t_min, t_max, num_points)

def render_polar_plot():
    """Render polar plotting interface"""
    st.subheader("Polar Plotting")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        r_function = st.text_input(
            "r(θ):",
            placeholder="1 + cos(theta)",
            key="r_polar"
        )
    
    with col2:
        theta_var = st.text_input("Angle Variable:", value="theta", key="theta_var")
    
    # Range parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        theta_min = st.number_input("θ Min:", value=0.0, key="theta_min")
    with col2:
        theta_max = st.number_input("θ Max:", value=2*math.pi, key="theta_max")
    with col3:
        num_points = st.number_input("Points:", min_value=100, max_value=5000, value=1000, key="points_polar")
    
    if r_function and st.button("Plot Polar", key="plot_polar"):
        plot_polar_function(r_function, theta_var, theta_min, theta_max, num_points)

def render_vector_field():
    """Render vector field plotting interface"""
    st.subheader("Vector Field Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        u_component = st.text_input("U component (dx/dt):", placeholder="y", key="u_vector")
    with col2:
        v_component = st.text_input("V component (dy/dt):", placeholder="-x", key="v_vector")
    
    # Grid parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_min = st.number_input("X Min:", value=-5.0, key="x_min_vector")
    with col2:
        x_max = st.number_input("X Max:", value=5.0, key="x_max_vector")
    with col3:
        y_min = st.number_input("Y Min:", value=-5.0, key="y_min_vector")
    with col4:
        y_max = st.number_input("Y Max:", value=5.0, key="y_max_vector")
    
    col1, col2 = st.columns(2)
    
    with col1:
        grid_density = st.slider("Grid Density:", 5, 50, 20)
    with col2:
        arrow_scale = st.slider("Arrow Scale:", 0.1, 2.0, 1.0, 0.1)
    
    if u_component and v_component and st.button("Plot Vector Field", key="plot_vector"):
        plot_vector_field(u_component, v_component, x_min, x_max, y_min, y_max, 
                         grid_density, arrow_scale)

def render_contour_plot():
    """Render contour plotting interface"""
    st.subheader("Contour Plot")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        function = st.text_input(
            "Function f(x,y):",
            placeholder="x**2 + y**2",
            key="func_contour"
        )
    
    with col2:
        contour_type = st.selectbox(
            "Type:",
            ["Filled", "Lines", "Both"]
        )
    
    # Range parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_min = st.number_input("X Min:", value=-5.0, key="x_min_contour")
    with col2:
        x_max = st.number_input("X Max:", value=5.0, key="x_max_contour")
    with col3:
        y_min = st.number_input("Y Min:", value=-5.0, key="y_min_contour")
    with col4:
        y_max = st.number_input("Y Max:", value=5.0, key="y_max_contour")
    
    col1, col2 = st.columns(2)
    
    with col1:
        resolution = st.slider("Resolution:", 20, 200, 100, key="res_contour")
    with col2:
        num_contours = st.slider("Number of Contours:", 5, 50, 20)
    
    if function and st.button("Plot Contour", key="plot_contour"):
        plot_contour_function(function, x_min, x_max, y_min, y_max, 
                             resolution, num_contours, contour_type)

def render_animation_plot():
    """Render animation plotting interface"""
    st.subheader("Function Animation")
    
    st.info("Create animated plots showing how functions change with a parameter")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        function = st.text_input(
            "Function f(x,t):",
            placeholder="sin(x - t)",
            key="func_anim"
        )
    
    with col2:
        time_var = st.text_input("Time Variable:", value="t", key="time_var")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_min = st.number_input("X Min:", value=-10.0, key="x_min_anim")
        x_max = st.number_input("X Max:", value=10.0, key="x_max_anim")
    
    with col2:
        t_min = st.number_input("t Min:", value=0.0, key="t_min_anim")
        t_max = st.number_input("t Max:", value=4*math.pi, key="t_max_anim")
    
    with col3:
        num_frames = st.number_input("Frames:", min_value=10, max_value=100, value=50)
        x_points = st.number_input("X Points:", min_value=100, max_value=1000, value=500)
    
    if function and st.button("Create Animation", key="plot_anim"):
        create_function_animation(function, time_var, x_min, x_max, t_min, t_max, 
                                num_frames, x_points)

# Plotting functions
def plot_2d_functions(functions, variable, x_min, x_max, num_points, colors, 
                     show_grid, show_axes, show_legend, log_scale_x, log_scale_y, equal_aspect):
    """Plot multiple 2D functions"""
    try:
        fig = go.Figure()
        
        x_vals = np.linspace(x_min, x_max, num_points)
        var_symbol = sp.Symbol(variable)
        
        for i, func_str in enumerate(functions):
            if not func_str:
                continue
                
            try:
                # Parse and evaluate function
                expr = expression_parser.parse(func_str)
                func = sp.lambdify(var_symbol, expr, 'numpy')
                
                # Handle potential division by zero or other issues
                y_vals = []
                x_plot = []
                
                for x in x_vals:
                    try:
                        y = func(x)
                        if np.isfinite(y):
                            y_vals.append(y)
                            x_plot.append(x)
                    except:
                        continue
                
                if x_plot and y_vals:
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_vals,
                        mode='lines',
                        name=f'f{i+1}(x) = {func_str}',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                
            except Exception as e:
                st.error(f"Error plotting function {i+1}: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title="2D Function Plot",
            xaxis_title=variable,
            yaxis_title="f(x)",
            showlegend=show_legend,
            hovermode='x unified'
        )
        
        if show_grid:
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
        
        if log_scale_x:
            fig.update_xaxes(type="log")
        if log_scale_y:
            fig.update_yaxes(type="log")
        
        if equal_aspect:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")

def plot_3d_surface(function, x_min, x_max, y_min, y_max, resolution, plot_type, colorscale):
    """Plot 3D surface"""
    try:
        # Create coordinate arrays
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Parse and evaluate function
        expr = expression_parser.parse(function)
        x_sym, y_sym = sp.symbols('x y')
        func = sp.lambdify((x_sym, y_sym), expr, 'numpy')
        
        # Evaluate function
        Z = func(X, Y)
        
        # Create plot based on type
        if plot_type == "Surface":
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale=colorscale.lower(),
                showscale=True
            )])
        elif plot_type == "Wireframe":
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale=colorscale.lower(),
                showscale=True,
                surfacecolor=Z,
                opacity=0.7,
                contours={"x": {"show": True}, "y": {"show": True}, "z": {"show": True}}
            )])
        else:  # Scatter
            # Sample points for scatter plot
            n_sample = min(1000, resolution**2)
            indices = np.random.choice(resolution**2, n_sample, replace=False)
            X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
            
            fig = go.Figure(data=[go.Scatter3d(
                x=X_flat[indices], y=Y_flat[indices], z=Z_flat[indices],
                mode='markers',
                marker=dict(
                    size=3,
                    color=Z_flat[indices],
                    colorscale=colorscale.lower(),
                    showscale=True
                )
            )])
        
        fig.update_layout(
            title=f"3D Plot: f(x,y) = {function}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="f(x,y)"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Function statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Value", f"{np.nanmin(Z):.4f}")
        with col2:
            st.metric("Max Value", f"{np.nanmax(Z):.4f}")
        with col3:
            st.metric("Mean Value", f"{np.nanmean(Z):.4f}")
        with col4:
            st.metric("Std Dev", f"{np.nanstd(Z):.4f}")
        
    except Exception as e:
        st.error(f"Error creating 3D plot: {str(e)}")

def plot_parametric_2d(x_param, y_param, t_min, t_max, num_points):
    """Plot 2D parametric curve"""
    try:
        t_vals = np.linspace(t_min, t_max, num_points)
        t_sym = sp.Symbol('t')
        
        # Parse parametric equations
        x_expr = expression_parser.parse(x_param)
        y_expr = expression_parser.parse(y_param)
        
        x_func = sp.lambdify(t_sym, x_expr, 'numpy')
        y_func = sp.lambdify(t_sym, y_expr, 'numpy')
        
        # Evaluate parametric equations
        x_vals = x_func(t_vals)
        y_vals = y_func(t_vals)
        
        fig = go.Figure()
        
        # Main curve
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            name='Parametric Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Start and end points
        fig.add_trace(go.Scatter(
            x=[x_vals[0]], y=[y_vals[0]],
            mode='markers',
            name='Start',
            marker=dict(color='green', size=10, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_vals[-1]], y=[y_vals[-1]],
            mode='markers',
            name='End',
            marker=dict(color='red', size=10, symbol='square')
        ))
        
        fig.update_layout(
            title=f"Parametric Plot: x(t) = {x_param}, y(t) = {y_param}",
            xaxis_title="x(t)",
            yaxis_title="y(t)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating parametric plot: {str(e)}")

def plot_parametric_3d(x_param, y_param, z_param, t_min, t_max, num_points):
    """Plot 3D parametric curve"""
    try:
        t_vals = np.linspace(t_min, t_max, num_points)
        t_sym = sp.Symbol('t')
        
        # Parse parametric equations
        x_expr = expression_parser.parse(x_param)
        y_expr = expression_parser.parse(y_param)
        z_expr = expression_parser.parse(z_param)
        
        x_func = sp.lambdify(t_sym, x_expr, 'numpy')
        y_func = sp.lambdify(t_sym, y_expr, 'numpy')
        z_func = sp.lambdify(t_sym, z_expr, 'numpy')
        
        # Evaluate parametric equations
        x_vals = x_func(t_vals)
        y_vals = y_func(t_vals)
        z_vals = z_func(t_vals)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=2)
        )])
        
        # Add start and end points
        fig.add_trace(go.Scatter3d(
            x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
            mode='markers',
            name='Start',
            marker=dict(color='green', size=8, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]],
            mode='markers',
            name='End',
            marker=dict(color='red', size=8, symbol='square')
        ))
        
        fig.update_layout(
            title=f"3D Parametric Plot",
            scene=dict(
                xaxis_title="x(t)",
                yaxis_title="y(t)",
                zaxis_title="z(t)"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating 3D parametric plot: {str(e)}")

def plot_polar_function(r_function, theta_var, theta_min, theta_max, num_points):
    """Plot polar function"""
    try:
        theta_vals = np.linspace(theta_min, theta_max, num_points)
        theta_sym = sp.Symbol(theta_var)
        
        # Parse r function
        r_expr = expression_parser.parse(r_function)
        r_func = sp.lambdify(theta_sym, r_expr, 'numpy')
        
        # Evaluate r function
        r_vals = r_func(theta_vals)
        
        # Convert to Cartesian coordinates
        x_vals = r_vals * np.cos(theta_vals)
        y_vals = r_vals * np.sin(theta_vals)
        
        fig = go.Figure()
        
        # Polar plot in Cartesian coordinates
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            name=f'r = {r_function}',
            line=dict(color='blue', width=2)
        ))
        
        # Add origin
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            name='Origin',
            marker=dict(color='red', size=8, symbol='circle')
        ))
        
        fig.update_layout(
            title=f"Polar Plot: r({theta_var}) = {r_function}",
            xaxis_title="x",
            yaxis_title="y",
            showlegend=True
        )
        
        # Equal aspect ratio for polar plots
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show the actual polar plot
        fig_polar = go.Figure()
        
        fig_polar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=np.degrees(theta_vals),
            mode='lines',
            name=f'r = {r_function}',
            line=dict(color='blue', width=2)
        ))
        
        fig_polar.update_layout(
            title="Polar Coordinate View",
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(visible=True)
            )
        )
        
        st.plotly_chart(fig_polar, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating polar plot: {str(e)}")

def plot_vector_field(u_component, v_component, x_min, x_max, y_min, y_max, grid_density, arrow_scale):
    """Plot vector field"""
    try:
        # Create grid
        x = np.linspace(x_min, x_max, grid_density)
        y = np.linspace(y_min, y_max, grid_density)
        X, Y = np.meshgrid(x, y)
        
        # Parse vector components
        x_sym, y_sym = sp.symbols('x y')
        u_expr = expression_parser.parse(u_component)
        v_expr = expression_parser.parse(v_component)
        
        u_func = sp.lambdify((x_sym, y_sym), u_expr, 'numpy')
        v_func = sp.lambdify((x_sym, y_sym), v_expr, 'numpy')
        
        # Evaluate vector components
        U = u_func(X, Y)
        V = v_func(X, Y)
        
        # Normalize for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitude + 1e-10) * arrow_scale
        V_norm = V / (magnitude + 1e-10) * arrow_scale
        
        fig = go.Figure()
        
        # Create quiver plot using annotations
        for i in range(0, len(x), max(1, len(x)//20)):  # Subsample for clarity
            for j in range(0, len(y), max(1, len(y)//20)):
                fig.add_annotation(
                    x=X[j,i], y=Y[j,i],
                    ax=X[j,i] + U_norm[j,i], ay=Y[j,i] + V_norm[j,i],
                    arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor="blue",
                    showarrow=True, axref="x", ayref="y"
                )
        
        # Add color-coded magnitude as background
        fig.add_trace(go.Contour(
            x=x, y=y, z=magnitude,
            colorscale='Viridis',
            opacity=0.3,
            showscale=True,
            colorbar=dict(title="Magnitude")
        ))
        
        fig.update_layout(
            title=f"Vector Field: U = {u_component}, V = {v_component}",
            xaxis_title="x",
            yaxis_title="y",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Magnitude", f"{np.max(magnitude):.4f}")
        with col2:
            st.metric("Min Magnitude", f"{np.min(magnitude):.4f}")
        with col3:
            st.metric("Mean Magnitude", f"{np.mean(magnitude):.4f}")
        
    except Exception as e:
        st.error(f"Error creating vector field: {str(e)}")

def plot_contour_function(function, x_min, x_max, y_min, y_max, resolution, num_contours, contour_type):
    """Plot contour plot"""
    try:
        # Create coordinate arrays
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Parse and evaluate function
        expr = expression_parser.parse(function)
        x_sym, y_sym = sp.symbols('x y')
        func = sp.lambdify((x_sym, y_sym), expr, 'numpy')
        
        # Evaluate function
        Z = func(X, Y)
        
        fig = go.Figure()
        
        if contour_type in ["Filled", "Both"]:
            fig.add_trace(go.Contour(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                ncontours=num_contours,
                showscale=True,
                colorbar=dict(title="f(x,y)")
            ))
        
        if contour_type in ["Lines", "Both"]:
            fig.add_trace(go.Contour(
                x=x, y=y, z=Z,
                showscale=False,
                contours=dict(
                    showlines=True,
                    coloring='none'
                ),
                line=dict(color='black', width=1),
                ncontours=num_contours
            ))
        
        fig.update_layout(
            title=f"Contour Plot: f(x,y) = {function}",
            xaxis_title="x",
            yaxis_title="y"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating contour plot: {str(e)}")

def create_function_animation(function, time_var, x_min, x_max, t_min, t_max, num_frames, x_points):
    """Create animated function plot"""
    try:
        # Create time and space arrays
        t_vals = np.linspace(t_min, t_max, num_frames)
        x_vals = np.linspace(x_min, x_max, x_points)
        
        # Parse function
        expr = expression_parser.parse(function)
        x_sym = sp.Symbol('x')
        t_sym = sp.Symbol(time_var)
        func = sp.lambdify((x_sym, t_sym), expr, 'numpy')
        
        # Create frames
        frames = []
        for i, t in enumerate(t_vals):
            y_vals = func(x_vals, t)
            
            frame = go.Frame(
                data=[go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f't = {t:.2f}'
                )],
                name=str(i)
            )
            frames.append(frame)
        
        # Initial frame
        y_init = func(x_vals, t_vals[0])
        
        fig = go.Figure(
            data=[go.Scatter(
                x=x_vals, y=y_init,
                mode='lines',
                line=dict(color='blue', width=2)
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=f"Animation: f(x,{time_var}) = {function}",
            xaxis_title="x",
            yaxis_title="f(x,t)",
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True},
                                      "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"{i}"], {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }],
                        "label": f"{t:.2f}",
                        "method": "animate"
                    }
                    for i, t in enumerate(t_vals)
                ]
            }]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Use the play button or slider to animate the function over time.")
        
    except Exception as e:
        st.error(f"Error creating animation: {str(e)}")

def render_function_analysis_viz(function, variable, x_min, x_max):
    """Render function analysis visualization"""
    try:
        from core.calculus_engine import calculus_engine
        
        # Get function analysis
        analysis = calculus_engine.analyze_function(function, variable)
        
        if analysis['success']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Critical Points:**")
                if analysis['critical_points']:
                    for point in analysis['critical_points']:
                        st.write(f"• x = {point}")
                else:
                    st.write("No critical points found")
            
            with col2:
                st.markdown("**Inflection Points:**")
                if analysis['inflection_points']:
                    for point in analysis['inflection_points']:
                        st.write(f"• x = {point}")
                else:
                    st.write("No inflection points found")
            
            # Plot function with critical and inflection points
            x_vals = np.linspace(x_min, x_max, 1000)
            var_symbol = sp.Symbol(variable)
            
            expr = expression_parser.parse(function)
            func = sp.lambdify(var_symbol, expr, 'numpy')
            y_vals = func(x_vals)
            
            fig = go.Figure()
            
            # Main function
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                name='f(x)',
                line=dict(color='blue', width=2)
            ))
            
            # Critical points
            for point_str in analysis['critical_points']:
                try:
                    point = float(point_str)
                    if x_min <= point <= x_max:
                        y_point = func(point)
                        fig.add_trace(go.Scatter(
                            x=[point], y=[y_point],
                            mode='markers',
                            name=f'Critical: x={point:.3f}',
                            marker=dict(color='red', size=10, symbol='circle')
                        ))
                except:
                    pass
            
            # Inflection points
            for point_str in analysis['inflection_points']:
                try:
                    point = float(point_str)
                    if x_min <= point <= x_max:
                        y_point = func(point)
                        fig.add_trace(go.Scatter(
                            x=[point], y=[y_point],
                            mode='markers',
                            name=f'Inflection: x={point:.3f}',
                            marker=dict(color='green', size=10, symbol='square')
                        ))
                except:
                    pass
            
            fig.update_layout(
                title="Function Analysis",
                xaxis_title=variable,
                yaxis_title="f(x)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in function analysis: {str(e)}")
