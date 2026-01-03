import streamlit as st
import numpy as np
import sympy as sp
from core.tensor_calculus_engine import TensorCalculusEngine, Tensor, TensorType, StandardMetrics
from utils.ui_helpers import run_task, copy_button

def render_tensor_calculus_panel():
    """Render the tensor calculus interface"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Tensor Calculus</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Advanced tensor operations and differential geometry</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different operations
    tabs = st.tabs([
        "Metric & Tensors",
        "Christoffel Symbols", 
        "Curvature",
        "Standard Metrics",
        "Geodesics"
    ])
    
    with tabs[0]:
        render_metric_tensors()
    
    with tabs[1]:
        render_christoffel_symbols()
        
    with tabs[2]:
        render_curvature_calculations()
        
    with tabs[3]:
        render_standard_metrics()
        
    with tabs[4]:
        render_geodesics()

def render_metric_tensors():
    """Metric tensor and basic tensor operations"""
    st.subheader("Metric Tensor & Basic Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Define Coordinate System**")
        n_dim = st.number_input("Number of dimensions", min_value=2, max_value=4, value=2)
        
        # Define coordinates
        coord_names = []
        for i in range(n_dim):
            coord = st.text_input(f"Coordinate {i+1}", value=['x', 'y', 'z', 't'][i] if i < 4 else f'x{i}')
            coord_names.append(coord)
        
        # Create symbolic coordinates
        try:
            coords = [sp.Symbol(name) for name in coord_names]
            
            st.write("**Define Metric Tensor**")
            st.write("Enter metric components g_ij:")
            
            # Metric input
            metric = np.zeros((n_dim, n_dim), dtype=object)
            for i in range(n_dim):
                cols = st.columns(n_dim)
                for j in range(n_dim):
                    with cols[j]:
                        if i <= j:  # Only upper triangle for symmetric metric
                            val = st.text_input(f"g_{i}{j}", value="1" if i == j else "0", key=f"metric_{i}_{j}")
                            try:
                                metric[i, j] = sp.sympify(val)
                                metric[j, i] = metric[i, j]  # Symmetric
                            except:
                                st.error(f"Invalid expression for g_{i}{j}")
                                return
        except Exception as e:
            st.error(f"Error setting up coordinates: {str(e)}")
            return
    
    with col2:
        if st.button("Calculate Metric Properties"):
            try:
                def _calc_metric():
                    engine = TensorCalculusEngine()
                    engine.set_metric(metric, coords)
                    det_g = sp.Matrix(metric).det()
                    return engine, det_g
                
                (engine, det_g), elapsed, error = run_task(
                    "Calculating metric properties...",
                    _calc_metric,
                    success_message="Metric properties ready",
                    error_message="Metric calculation failed"
                )
                if error:
                    return
                
                st.write("**Metric Tensor:**")
                st.latex(f"g_{{ij}} = {sp.latex(sp.Matrix(metric))}")
                
                st.write("**Inverse Metric:**")
                st.latex(f"g^{{ij}} = {sp.latex(sp.Matrix(engine.inverse_metric.components))}")
                
                # Calculate determinant
                det_g = sp.Matrix(metric).det()
                st.write("**Metric Determinant:**")
                st.latex(f"\\det(g) = {sp.latex(det_g)}")
                
                # Store in session state
                st.session_state['tensor_engine'] = engine
                st.session_state['tensor_coords'] = coords
                
            except Exception as e:
                st.error(f"Error calculating metric properties: {str(e)}")
    
    # Tensor operations section
    st.divider()
    st.subheader("Tensor Operations")
    
    if 'tensor_engine' in st.session_state:
        engine = st.session_state['tensor_engine']
        coords = st.session_state['tensor_coords']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Define Tensor 1**")
            rank1 = st.number_input("Rank", min_value=1, max_value=3, value=1, key="rank1")
            
            # Components input
            if rank1 == 1:
                components1 = []
                for i in range(len(coords)):
                    comp = st.text_input(f"Component {i}", value=str(i+1), key=f"comp1_{i}")
                    components1.append(sp.sympify(comp))
                components1 = np.array(components1)
                
                variance1 = st.selectbox("Type", ["Contravariant", "Covariant"], key="var1")
                indices1 = [TensorType.CONTRAVARIANT if variance1 == "Contravariant" else TensorType.COVARIANT]
            
        with col2:
            st.write("**Define Tensor 2**")
            rank2 = st.number_input("Rank", min_value=1, max_value=3, value=1, key="rank2")
            
            if rank2 == 1:
                components2 = []
                for i in range(len(coords)):
                    comp = st.text_input(f"Component {i}", value=str(2*i+1), key=f"comp2_{i}")
                    components2.append(sp.sympify(comp))
                components2 = np.array(components2)
                
                variance2 = st.selectbox("Type", ["Contravariant", "Covariant"], key="var2")
                indices2 = [TensorType.CONTRAVARIANT if variance2 == "Contravariant" else TensorType.COVARIANT]
        
        if st.button("Calculate Tensor Product"):
            try:
                tensor1 = Tensor(components1, indices1, coords, "T")
                tensor2 = Tensor(components2, indices2, coords, "S")
                
                result, elapsed, error = run_task(
                    "Calculating tensor product...",
                    engine.tensor_product,
                    tensor1, tensor2,
                    success_message="Tensor product calculated",
                    error_message="Tensor product failed"
                )
                if error:
                    return
                
                st.write(f"**Result (in {elapsed:.2f}s):**")
                st.write(f"Tensor product T x S has rank {result.rank}")
                st.write("Components:")
                st.code(str(result.components))
                copy_button(str(result.components), key="tensor-product")
                
            except Exception as e:
                st.error(f"Error in tensor product: {str(e)}")

def render_christoffel_symbols():
    """Calculate Christoffel symbols"""
    st.subheader("Christoffel Symbols")
    
    if 'tensor_engine' not in st.session_state:
        st.warning("Please define a metric tensor first in the 'Metric & Tensors' tab")
        return
    
    engine = st.session_state['tensor_engine']
    coords = st.session_state['tensor_coords']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Calculate Christoffel Symbols (1st kind)"):
            try:
                gamma_lower, elapsed, error = run_task(
                    "Computing Christoffel symbols (1st kind)...",
                    engine.christoffel_symbols_first_kind,
                    success_message="Christoffel symbols ready",
                    error_message="Calculation failed"
                )
                if error:
                    return
                st.write("**Christoffel Symbols Γ_ijk:**")
                
                # Display non-zero components
                dim = len(coords)
                for i in range(dim):
                    for j in range(dim):
                        for k in range(dim):
                            if gamma_lower[i, j, k] != 0:
                                st.latex(f"\\Gamma_{{{i}{j}{k}}} = {sp.latex(gamma_lower[i, j, k])}")
                
            except Exception as e:
                st.error(f"Error calculating Christoffel symbols: {str(e)}")
    
    with col2:
        if st.button("Calculate Christoffel Symbols (2nd kind)"):
            try:
                gamma_upper, elapsed, error = run_task(
                    "Calculating Christoffel symbols (2nd kind)...",
                    engine.christoffel_symbols_second_kind,
                    success_message="Christoffel symbols calculated",
                    error_message="Christoffel calculation failed"
                )
                if error:
                    return
                    
                st.write(f"**Christoffel Symbols Γ^i_jk (in {elapsed:.2f}s):**")
                
                # Display non-zero components
                dim = len(coords)
                for i in range(dim):
                    for j in range(dim):
                        for k in range(dim):
                            if gamma_upper[i, j, k] != 0:
                                st.latex(f"\\Gamma^{{{i}}}_{{{j}{k}}} = {sp.latex(gamma_upper[i, j, k])}")
                
            except Exception as e:
                st.error(f"Error calculating Christoffel symbols: {str(e)}")

def render_curvature_calculations():
    """Calculate curvature tensors"""
    st.subheader("Curvature Tensors")
    
    if 'tensor_engine' not in st.session_state:
        st.warning("Please define a metric tensor first in the 'Metric & Tensors' tab")
        return
    
    engine = st.session_state['tensor_engine']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Calculate Ricci Tensor"):
            try:
                ricci, elapsed, error = run_task(
                    "Computing Ricci tensor...",
                    engine.ricci_tensor,
                    success_message="Ricci tensor ready",
                    error_message="Calculation failed"
                )
                if error:
                    return
                    
                st.write("**Ricci Tensor R_ij:**")
                st.latex(f"R_{{ij}} = {sp.latex(sp.Matrix(ricci.components))}")
                
            except Exception as e:
                st.error(f"Error calculating Ricci tensor: {str(e)}")
    
    with col2:
        if st.button("Calculate Ricci Scalar"):
            try:
                scalar, elapsed, error = run_task(
                    "Calculating Ricci scalar...",
                    engine.ricci_scalar,
                    success_message="Ricci scalar calculated",
                    error_message="Ricci scalar calculation failed"
                )
                if error:
                    return
                    
                st.write(f"**Ricci Scalar R (in {elapsed:.2f}s):**")
                st.latex(f"R = {sp.latex(scalar)}")
                copy_button(str(scalar), key="ricci-scalar")
                
                # Einstein tensor
                if st.checkbox("Calculate Einstein Tensor"):
                    metric = engine.metric_tensor.components
                    ricci = engine.ricci_tensor().components
                    G = ricci - sp.Rational(1, 2) * scalar * metric
                    
                    st.write("**Einstein Tensor G_ij:**")
                    st.latex(f"G_{{ij}} = R_{{ij}} - \\frac{{1}}{{2}}Rg_{{ij}} = {sp.latex(sp.Matrix(G))}")
                
            except Exception as e:
                st.error(f"Error calculating curvature: {str(e)}")

def render_standard_metrics():
    """Pre-defined standard metrics"""
    st.subheader("Standard Metrics")
    
    metric_type = st.selectbox(
        "Select metric",
        ["2D Euclidean (Cartesian)", "2D Euclidean (Polar)", 
         "2-Sphere", "Schwarzschild", "Minkowski Spacetime"]
    )
    
    if st.button("Load Metric"):
        try:
            engine = TensorCalculusEngine()
            
            if metric_type == "2D Euclidean (Cartesian)":
                metric, coords = StandardMetrics.euclidean_2d()
            elif metric_type == "2D Euclidean (Polar)":
                metric, coords = StandardMetrics.euclidean_polar()
            elif metric_type == "2-Sphere":
                metric, coords = StandardMetrics.sphere_2d()
            elif metric_type == "Schwarzschild":
                metric, coords = StandardMetrics.schwarzschild()
            else:  # Minkowski
                metric, coords = StandardMetrics.minkowski()
            
            engine.set_metric(metric, coords)
            
            st.write("**Metric Tensor:**")
            st.latex(f"g_{{ij}} = {sp.latex(sp.Matrix(metric))}")
            
            st.write("**Coordinates:**")
            st.write(", ".join([str(c) for c in coords]))
            
            # Line element
            st.write("**Line Element:**")
            ds2_terms = []
            for i in range(len(coords)):
                for j in range(len(coords)):
                    if metric[i, j] != 0:
                        if i == j:
                            ds2_terms.append(f"{sp.latex(metric[i, j])} d{coords[i]}^2")
                        elif i < j:
                            ds2_terms.append(f"2 \\cdot {sp.latex(metric[i, j])} d{coords[i]} d{coords[j]}")
            
            st.latex(f"ds^2 = {' + '.join(ds2_terms)}")
            
            # Store in session state
            st.session_state['tensor_engine'] = engine
            st.session_state['tensor_coords'] = coords
            
        except Exception as e:
            st.error(f"Error loading metric: {str(e)}")

def render_geodesics():
    """Calculate geodesic equations"""
    st.subheader("Geodesic Equations")
    
    if 'tensor_engine' not in st.session_state:
        st.warning("Please define a metric tensor first")
        return
    
    engine = st.session_state['tensor_engine']
    coords = st.session_state['tensor_coords']
    
    parameter = st.text_input("Affine parameter", value="s")
    param_symbol = sp.Symbol(parameter)
    
    if st.button("Generate Geodesic Equations"):
        try:
            equations, elapsed, error = run_task(
                "Calculating geodesic equations...",
                engine.geodesic_equation,
                param_symbol,
                success_message="Geodesic equations generated",
                error_message="Geodesic calculation failed"
            )
            if error:
                return
            
            st.write(f"**Geodesic Equations (in {elapsed:.2f}s):**")
            st.write(f"For curve x^i({parameter}), the geodesic equations are:")
            
            for i, eq in enumerate(equations):
                st.latex(f"{sp.latex(eq)}")
            
            # Interpretation
            with st.expander("Interpretation"):
                st.write("""
                The geodesic equations describe the path of a particle moving solely under 
                the influence of gravity (in GR) or the straightest possible path in the geometry.
                
                These are second-order differential equations that need initial conditions:
                - Initial position x^i(0)
                - Initial velocity dx^i/ds(0)
                """)
                
        except Exception as e:
            st.error(f"Error calculating geodesics: {str(e)}")