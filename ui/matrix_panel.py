import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
from sympy import Matrix as SpMatrix

from core.matrix_operations import matrix_operations
from utils.ui_helpers import run_task, copy_button

def render_matrix_panel():
    """Render the matrix operations panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Matrix Operations</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Comprehensive matrix operations and linear algebra tools</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Matrix input method selection
    input_method = st.radio(
        "Matrix Input Method:",
        ["Manual Entry", "Generate Random", "Predefined"]
    )
    
    # Initialize session state for matrices
    if 'matrix_a' not in st.session_state:
        st.session_state.matrix_a = None
    if 'matrix_b' not in st.session_state:
        st.session_state.matrix_b = None
    
    # Matrix input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matrix A")
        get_matrix_input_setup(input_method, "A")
        if st.session_state.matrix_a is not None:
            display_matrix(st.session_state.matrix_a, "Matrix A")
    
    with col2:
        st.subheader("Matrix B")
        get_matrix_input_setup(input_method, "B")
        if st.session_state.matrix_b is not None:
            display_matrix(st.session_state.matrix_b, "Matrix B")
    
    # Single button to generate both matrices
    st.markdown("---")
    if st.button("Generate Matrices", key="generate_both_matrices", type="primary"):
        matrix_a = get_matrix_from_setup(input_method, "A")
        matrix_b = get_matrix_from_setup(input_method, "B")
        
        if matrix_a is not None:
            st.session_state.matrix_a = matrix_a
            st.success("Matrix A generated successfully!")
        if matrix_b is not None:
            st.session_state.matrix_b = matrix_b
            st.success("Matrix B generated successfully!")
        
        if matrix_a is not None or matrix_b is not None:
            st.rerun()
    
    st.markdown("---")
    
    # Operation selection
    st.subheader("Matrix Operations")
    
    # Single matrix operations
    st.markdown("**Single Matrix Operations (using Matrix A):**")
    single_ops_cols = st.columns(4)
    
    single_operations = [
        ("Determinant", "determinant"),
        ("Inverse", "inverse"),
        ("Transpose", "transpose"),
        ("Eigenvalues", "eigenvalues")
    ]
    
    for i, (name, op) in enumerate(single_operations):
        with single_ops_cols[i]:
            if st.button(name, key=f"single_{op}"):
                if st.session_state.matrix_a is not None:
                    perform_single_matrix_operation(st.session_state.matrix_a, op)
                else:
                    st.error("Please input Matrix A first")
    
    # Two matrix operations
    st.markdown("**Two Matrix Operations:**")
    dual_ops_cols = st.columns(3)
    
    dual_operations = [
        ("A + B", "addition"),
        ("A × B", "multiplication"),
        ("Solve Ax = B", "solve_system")
    ]
    
    for i, (name, op) in enumerate(dual_operations):
        with dual_ops_cols[i]:
            if st.button(name, key=f"dual_{op}"):
                if st.session_state.matrix_a is not None and st.session_state.matrix_b is not None:
                    perform_dual_matrix_operation(st.session_state.matrix_a, st.session_state.matrix_b, op)
                else:
                    st.error("Please input both matrices first")
    
    # Advanced operations
    st.markdown("---")
    st.subheader("Advanced Operations")
    
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        if st.button("Matrix Rank", key="rank"):
            if st.session_state.matrix_a is not None:
                perform_single_matrix_operation(st.session_state.matrix_a, "rank")
            else:
                st.error("Please input Matrix A first")
    
    with adv_col2:
        norm_type = st.selectbox("Norm Type:", ["frobenius", "1", "2", "inf"])
        if st.button("Matrix Norm", key="norm"):
            if st.session_state.matrix_a is not None:
                perform_matrix_norm(st.session_state.matrix_a, norm_type)
            else:
                st.error("Please input Matrix A first")
    
    with adv_col3:
        symbolic_ops = st.checkbox("Use Symbolic Computation", value=False)

def get_matrix_input_setup(method, matrix_label):
    """Setup matrix input UI without generating matrix"""
    if method == "Manual Entry":
        manual_matrix_entry_setup(matrix_label)
    elif method == "Generate Random":
        generate_random_matrix_setup(matrix_label)
    elif method == "Predefined":
        get_predefined_matrix_setup(matrix_label)

def get_matrix_from_setup(method, matrix_label):
    """Generate matrix from setup"""
    if method == "Manual Entry":
        return manual_matrix_entry_generate(matrix_label)
    elif method == "Generate Random":
        return generate_random_matrix_generate(matrix_label)
    elif method == "Predefined":
        return get_predefined_matrix_generate(matrix_label)
    return None

def manual_matrix_entry_setup(matrix_label):
    """Setup manual matrix entry UI"""
    col1, col2 = st.columns(2)
    
    with col1:
        rows = st.number_input(f"Rows ({matrix_label}):", min_value=1, max_value=10, value=3, key=f"rows_{matrix_label}")
    with col2:
        cols = st.number_input(f"Columns ({matrix_label}):", min_value=1, max_value=10, value=3, key=f"cols_{matrix_label}")
    
    # Create input grid
    st.write(f"Enter matrix {matrix_label} elements:")
    
    for i in range(rows):
        row_cols = st.columns(cols)
        for j in range(cols):
            with row_cols[j]:
                st.number_input(
                    f"[{i+1},{j+1}]",
                    value=0.0,
                    key=f"matrix_{matrix_label}_{i}_{j}",
                    format="%.6f"
                )

def manual_matrix_entry_generate(matrix_label):
    """Generate matrix from manual entry setup"""
    try:
        if f"rows_{matrix_label}" not in st.session_state or f"cols_{matrix_label}" not in st.session_state:
            return None
            
        rows = st.session_state[f"rows_{matrix_label}"]
        cols = st.session_state[f"cols_{matrix_label}"]
        
        matrix_data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                key = f"matrix_{matrix_label}_{i}_{j}"
                if key in st.session_state:
                    row_data.append(st.session_state[key])
                else:
                    row_data.append(0.0)
            matrix_data.append(row_data)
        
        result = matrix_operations.create_matrix(matrix_data, symbolic=False)
        if result['success']:
            return result['matrix']
        else:
            st.error(f"Error creating matrix {matrix_label}: {result['error']}")
            return None
    except Exception as e:
        st.error(f"Error generating matrix {matrix_label}: {str(e)}")
        return None



def generate_random_matrix_setup(matrix_label):
    """Setup random matrix generation UI"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.number_input(f"Rows ({matrix_label}):", min_value=1, max_value=10, value=3, key=f"rand_rows_{matrix_label}")
    with col2:
        st.number_input(f"Columns ({matrix_label}):", min_value=1, max_value=10, value=3, key=f"rand_cols_{matrix_label}")
    with col3:
        st.number_input(f"Max Value ({matrix_label}):", min_value=1, max_value=100, value=10, key=f"rand_max_{matrix_label}")

def generate_random_matrix_generate(matrix_label):
    """Generate random matrix from setup"""
    try:
        rows = st.session_state.get(f"rand_rows_{matrix_label}", 3)
        cols = st.session_state.get(f"rand_cols_{matrix_label}", 3)
        max_val = st.session_state.get(f"rand_max_{matrix_label}", 10)
        
        matrix_data = np.random.randint(-max_val, max_val + 1, size=(rows, cols))
        result = matrix_operations.create_matrix(matrix_data.tolist(), symbolic=False)
        if result['success']:
            return result['matrix']
        else:
            st.error(f"Error creating matrix {matrix_label}: {result['error']}")
    except Exception as e:
        st.error(f"Error generating matrix {matrix_label}: {str(e)}")
    return None

def get_predefined_matrix_setup(matrix_label):
    """Setup predefined matrix selection UI"""
    predefined_options = {
        "Identity 3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Zero 3x3": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "Ones 3x3": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "Diagonal": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "Symmetric": [[1, 2, 3], [2, 4, 5], [3, 5, 6]],
        "Hilbert 3x3": [[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]]
    }
    
    st.selectbox(
        f"Select predefined matrix for {matrix_label}:",
        list(predefined_options.keys()),
        key=f"predefined_{matrix_label}"
    )

def get_predefined_matrix_generate(matrix_label):
    """Generate predefined matrix from setup"""
    predefined_options = {
        "Identity 3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Zero 3x3": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "Ones 3x3": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "Diagonal": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "Symmetric": [[1, 2, 3], [2, 4, 5], [3, 5, 6]],
        "Hilbert 3x3": [[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]]
    }
    
    selected = st.session_state.get(f"predefined_{matrix_label}", "Identity 3x3")
    try:
        matrix_data = predefined_options[selected]
        result = matrix_operations.create_matrix(matrix_data, symbolic=False)
        if result['success']:
            return result['matrix']
        else:
            st.error(f"Error creating matrix {matrix_label}: {result['error']}")
    except Exception as e:
        st.error(f"Error generating matrix {matrix_label}: {str(e)}")
    return None

def display_matrix(matrix, title):
    """Display matrix in a formatted way"""
    if isinstance(matrix, np.ndarray):
        df = pd.DataFrame(matrix)
        st.write(f"**{title}** (Shape: {matrix.shape})")
        st.dataframe(df, use_container_width=True)
    elif isinstance(matrix, SpMatrix):
        st.write(f"**{title}** (Shape: {matrix.shape})")
        st.latex(sp.latex(matrix))

def perform_single_matrix_operation(matrix, operation):
    """Perform single matrix operations"""
    st.markdown("---")
    st.subheader(f"Result: {operation.title()}")
    
    if operation == "determinant":
        result, _, error = run_task(
            "Computing determinant...",
            matrix_operations.matrix_determinant,
            matrix,
            success_message="Determinant ready",
            error_message="Determinant failed"
        )
        if not error:
            st.success(f"Determinant calculated successfully")
            st.metric("Determinant", f"{result['determinant']:.6g}")
            copy_button(str(result['determinant']), key="determinant-value")
            
            # Additional info
            if abs(result['determinant']) < 1e-10:
                st.warning("Matrix is singular (determinant ≈ 0)")
            else:
                st.info("Matrix is non-singular")
    
    elif operation == "inverse":
        result, _, error = run_task(
            "Computing inverse...",
            matrix_operations.matrix_inverse,
            matrix,
            success_message="Inverse ready",
            error_message="Inverse failed"
        )
        if not error:
            st.success("Inverse calculated successfully")
            display_matrix(result['inverse_matrix'], "Inverse Matrix")
            
            # Verify inverse
            if isinstance(matrix, np.ndarray):
                product = np.dot(matrix, result['inverse_matrix'])
                st.write("**Verification (A × A⁻¹):**")
                display_matrix(product, "A × A⁻¹")
    
    elif operation == "transpose":
        result, _, error = run_task(
            "Computing transpose...",
            matrix_operations.matrix_transpose,
            matrix,
            success_message="Transpose ready",
            error_message="Transpose failed"
        )
        if not error:
            st.success("Transpose calculated successfully")
            display_matrix(result['transpose_matrix'], "Transpose Matrix")
    
    elif operation == "eigenvalues":
        result, _, error = run_task(
            "Computing eigenvalues...",
            matrix_operations.matrix_eigenvalues,
            matrix,
            success_message="Eigenvalues ready",
            error_message="Eigenvalue computation failed"
        )
        if not error:
            st.success("Eigenvalues calculated successfully")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Eigenvalues:**")
                copy_button(", ".join(str(ev) for ev in result['eigenvalues']), key="eigenvalues-copy")
                for i, eigenval in enumerate(result['eigenvalues']):
                    st.write(f"λ_{i+1} = {eigenval}")
            
            with col2:
                st.markdown("**Eigenvectors:**")
                if isinstance(result['eigenvectors'][0], list):
                    for i, eigenvec in enumerate(result['eigenvectors']):
                        st.write(f"v_{i+1} = {eigenvec}")
    
    elif operation == "rank":
        result, _, error = run_task(
            "Computing rank...",
            matrix_operations.matrix_rank,
            matrix,
            success_message="Rank ready",
            error_message="Rank computation failed"
        )
        if not error:
            st.success("Rank calculated successfully")
            st.metric("Matrix Rank", result['rank'])
            copy_button(str(result['rank']), key="rank-copy")
            
            # Additional info
            rows, cols = matrix.shape
            st.write(f"Matrix dimensions: {rows} × {cols}")
            if result['rank'] == min(rows, cols):
                st.info("Matrix has full rank")
            else:
                st.warning("Matrix is rank deficient")

def perform_dual_matrix_operation(matrix_a, matrix_b, operation):
    """Perform operations involving two matrices"""
    st.markdown("---")
    st.subheader(f"Result: {operation.title()}")
    
    if operation == "addition":
        result, _, error = run_task(
            "Computing A + B...",
            matrix_operations.matrix_addition,
            matrix_a,
            matrix_b,
            success_message="Addition ready",
            error_message="Addition failed"
        )
        if not error:
            st.success("Addition completed successfully")
            display_matrix(result['result_matrix'], "A + B")
    
    elif operation == "multiplication":
        result, _, error = run_task(
            "Computing A × B...",
            matrix_operations.matrix_multiplication,
            matrix_a,
            matrix_b,
            success_message="Multiplication ready",
            error_message="Multiplication failed"
        )
        if not error:
            st.success("Multiplication completed successfully")
            display_matrix(result['result_matrix'], "A × B")
            
            # Show dimensions
            st.info(f"Result dimensions: {result['result_matrix'].shape}")
    
    elif operation == "solve_system":
        result, _, error = run_task(
            "Solving Ax = B...",
            matrix_operations.solve_linear_system,
            matrix_a,
            matrix_b,
            success_message="Solution ready",
            error_message="Linear solve failed"
        )
        if not error:
            st.success("Linear system solved successfully")
            
            st.markdown("**Solution (x):**")
            if isinstance(result['solution'], list):
                for i, sol in enumerate(result['solution']):
                    if isinstance(sol, (list, np.ndarray)):
                        st.write(f"x_{i+1} = {float(sol[0]) if hasattr(sol, '__len__') and len(sol) > 0 else float(sol):.6g}")
                    else:
                        st.write(f"x_{i+1} = {float(sol):.6g}")
                copy_button(", ".join(str(sol) for sol in result['solution']), key="solution-copy")
            else:
                display_matrix(result['solution'], "Solution Vector")
                
            # Verification
            if isinstance(matrix_a, np.ndarray):
                verification = np.dot(matrix_a, result['solution'])
                st.write("**Verification (A × x):**")
                display_matrix(verification.reshape(-1, 1), "A × x")

def perform_matrix_norm(matrix, norm_type):
    """Perform matrix norm calculation"""
    st.markdown("---")
    st.subheader(f"Result: {norm_type.title()} Norm")
    
    result, _, error = run_task(
        "Computing norm...",
        matrix_operations.matrix_norm,
        matrix,
        norm_type,
        success_message=f"{norm_type.title()} norm ready",
        error_message="Norm computation failed"
    )
    if not error:
        st.success(f"{norm_type.title()} norm calculated successfully")
        st.metric(f"{norm_type.title()} Norm", f"{result['norm']:.6g}")
        copy_button(str(result['norm']), key="norm-copy")
