import streamlit as st
import sympy as sp
import numpy as np
from core.calculation_engine import calculation_engine
from core.expression_parser import expression_parser
from utils.ui_helpers import run_task, copy_button
from utils.math_helpers import MathHelpers
from utils.exceptions import InvalidInputError, ExpressionParseError

# Example expressions for quick loading
CALCULATOR_EXAMPLES = {
    "Basic Arithmetic": "2 + 3 * 4 - sqrt(16)",
    "Trigonometry": "sin(pi/4) + cos(pi/3) + tan(pi/6)",
    "Logarithms": "ln(e**2) + log(100, 10)",
    "Complex Expression": "sqrt(16) + exp(1) * sin(pi/2) + log(1000)",
    "With Variables": "x**2 + 2*x + 1"
}

def render_calculator_panel():
    """Render the scientific calculator panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Scientific Calculator</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Advanced mathematical expression evaluator with step-by-step solutions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example selector
    st.markdown("**Load Example:**")
    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        example_choice = st.selectbox(
            "Choose an example:",
            [""] + list(CALCULATOR_EXAMPLES.keys()),
            help="Select an example to auto-fill the expression field"
        )
    with col_ex2:
        if st.button("Load Example", disabled=not example_choice):
            if example_choice in CALCULATOR_EXAMPLES:
                st.session_state.main_expression = CALCULATOR_EXAMPLES[example_choice]
                st.rerun()
    
    st.markdown("---")
    
    # Calculator mode selection
    # Initialize session state for expression and handle button updates BEFORE creating widget
    if 'temp_expression_update' in st.session_state:
        # Apply the temp update before widget is created
        st.session_state.main_expression = st.session_state.temp_expression_update
        del st.session_state.temp_expression_update
    
    if 'main_expression' not in st.session_state:
        st.session_state.main_expression = ""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Enter mathematical expression:",
            placeholder="e.g., sin(pi/4) + log(e^2) + sqrt(16)",
            help="Use standard mathematical notation. Supported functions: sin, cos, tan, log, ln, exp, sqrt, etc.",
            key="main_expression"
        )
    
    with col2:
        mode = st.selectbox(
            "Mode:",
            ["Evaluate", "Simplify", "Expand", "Factor"]
        )
    
    # Quick function buttons
    st.markdown("**Quick Functions:**")
    button_cols = st.columns(8)
    
    quick_functions = [
        ("π", "pi"), ("e", "e"), ("√", "sqrt()"), ("ln", "ln()"),
        ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"), ("∫", "integrate()")
    ]
    
    for i, (display, func) in enumerate(quick_functions):
        with button_cols[i]:
            if st.button(display, key=f"btn_{i}"):
                # Store the updated expression in temp variable
                current_value = st.session_state.get('main_expression', '')
                st.session_state.temp_expression_update = current_value + func
                # Mark that we need to position cursor
                if func.endswith(')'):
                    st.session_state.position_cursor = True
                st.rerun()
    
    # JavaScript to position cursor inside parentheses after button click
    if st.session_state.get('position_cursor', False):
        cursor_js = """
        <script>
        setTimeout(function() {
            const input = window.parent.document.querySelector('input[aria-label="Enter mathematical expression:"]');
            if (input && input.value.endsWith(')')) {
                const pos = input.value.length - 1;
                input.focus();
                input.setSelectionRange(pos, pos);
            }
        }, 100);
        </script>
        """
        st.markdown(cursor_js, unsafe_allow_html=True)
        st.session_state.position_cursor = False
    
    # Variable substitution section
    st.markdown("---")
    st.subheader("Variable Substitution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_names = st.text_input(
            "Variables (comma-separated):",
            placeholder="x, y, z",
            help="Enter variable names separated by commas"
        )
    
    with col2:
        var_values = st.text_input(
            "Values (comma-separated):",
            placeholder="1, 2, 3",
            help="Enter corresponding values separated by commas"
        )
    
    # Process calculation
    if expression:
        # Parse variables if provided
        variables = None
        if var_names and var_values:
            try:
                # Strip and filter out empty strings
                names = [name.strip() for name in var_names.split(',') if name.strip()]
                values_str = [val.strip() for val in var_values.split(',') if val.strip()]
                
                # Check if we have at least one variable
                if not names or not values_str:
                    st.error("Please provide at least one variable name and value")
                    return
                
                # Convert values to floats
                values = [float(val) for val in values_str]
                
                # Check lengths match
                if len(names) != len(values):
                    st.error("Number of variable names and values must match!")
                    st.info(f"You provided {len(names)} names and {len(values)} values")
                    return
                    
                variables = dict(zip(names, values))
            except ValueError as e:
                # More specific error handling for conversion errors
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    st.error("Invalid variable values. Please enter numeric values.")
                    st.info(f"Hint: Values must be numbers (e.g., '1, 2.5, -3')")
                else:
                    st.error(f"Error parsing variables: {str(e)}")
                return
            except Exception as e:
                st.error(f"Error parsing variables: {str(e)}")
                return
        
        # Perform calculation based on mode
        if mode == "Evaluate":
            result, _, error = run_task(
                "Evaluating expression...",
                calculation_engine.evaluate_expression,
                expression,
                variables,
                success_message="Evaluation completed",
                error_message="Evaluation failed"
            )
            if not error:
                display_calculation_result(result, "Evaluation")
        
        elif mode == "Simplify":
            result, _, error = run_task(
                "Simplifying expression...",
                calculation_engine.simplify_expression,
                expression,
                success_message="Simplified",
                error_message="Simplification failed"
            )
            if not error:
                display_simplification_result(result)
        
        elif mode == "Expand":
            result, _, error = run_task(
                "Expanding expression...",
                calculation_engine.expand_expression,
                expression,
                success_message="Expanded",
                error_message="Expansion failed"
            )
            if not error:
                display_expansion_result(result)
        
        elif mode == "Factor":
            result, _, error = run_task(
                "Factoring expression...",
                calculation_engine.factor_expression,
                expression,
                success_message="Factored",
                error_message="Factoring failed"
            )
            if not error:
                display_factoring_result(result)
        
        # Expression validation and info
        st.markdown("---")
        st.subheader("Expression Analysis")
        validation = expression_parser.validate_expression(expression)
        display_expression_info(validation)

def display_calculation_result(result, operation_name):
    """Display calculation results"""
    if result['success']:
        st.success(f"{operation_name} completed successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Symbolic Result:**")
            if result['symbolic_result'] is not None:
                st.latex(sp.latex(result['symbolic_result']))
                copy_button(str(result['symbolic_result']), key=f"{operation_name}-symbolic")
            else:
                st.write("No symbolic result")
        
        with col2:
            st.markdown("**Numeric Result:**")
            if result['numeric_result'] is not None:
                numeric_value = result['numeric_result']
                value_str = MathHelpers.format_number(numeric_value, precision=10)
                st.metric("Value", value_str)
                copy_button(value_str, key=f"{operation_name}-numeric")
                
                # Additional numeric info
                if isinstance(numeric_value, (int, float)):
                    st.write(f"Scientific notation: {MathHelpers.format_scientific(numeric_value, precision=6)}")
                    if numeric_value != 0:
                        st.write(f"Magnitude: {MathHelpers.format_scientific(abs(numeric_value), precision=6)}")
            else:
                st.write("No numeric result available")
    
    else:
        st.error(f"{operation_name} failed: {result['error']}")

def display_simplification_result(result):
    """Display simplification results"""
    if result['success']:
        st.success("Expression simplified successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original:**")
            st.latex(sp.latex(result['original']))
        
        with col2:
            st.markdown("**Simplified:**")
            st.latex(sp.latex(result['simplified']))
        
        # Check if simplification made a difference
        if str(result['original']) == str(result['simplified']):
            st.info("Expression is already in its simplest form")
    
    else:
        st.error(f"Simplification failed: {result['error']}")

def display_expansion_result(result):
    """Display expansion results"""
    if result['success']:
        st.success("Expression expanded successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original:**")
            st.latex(sp.latex(result['original']))
        
        with col2:
            st.markdown("**Expanded:**")
            st.latex(sp.latex(result['expanded']))
        
        # Check if expansion made a difference
        if str(result['original']) == str(result['expanded']):
            st.info("Expression is already expanded")
    
    else:
        st.error(f"Expansion failed: {result['error']}")

def display_factoring_result(result):
    """Display factoring results"""
    if result['success']:
        st.success("Expression factored successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original:**")
            st.latex(sp.latex(result['original']))
        
        with col2:
            st.markdown("**Factored:**")
            st.latex(sp.latex(result['factored']))
        
        # Check if factoring made a difference
        if str(result['original']) == str(result['factored']):
            st.info("Expression cannot be factored further")
    
    else:
        st.error(f"Factoring failed: {result['error']}")

def display_expression_info(validation):
    """Display expression validation and information"""
    if validation['valid']:
        st.success("Expression is valid")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Variables:**")
            if validation['variables']:
                for var in validation['variables']:
                    st.write(f"• {var}")
            else:
                st.write("No variables")
        
        with col2:
            st.markdown("**Functions:**")
            if validation['functions']:
                for func in validation['functions']:
                    st.write(f"• {func}")
            else:
                st.write("No functions")
        
        with col3:
            if validation['expression']:
                info = expression_parser.get_expression_info(validation['expression'])
                st.markdown("**Properties:**")
                st.write(f"• Complexity: {info['complexity']}")
                st.write(f"• Polynomial: {info['is_polynomial']}")
                st.write(f"• Rational: {info['is_rational']}")
    
    else:
        st.error(f"Invalid expression: {validation['error']}")
        st.markdown("**Help:**")
        st.markdown("""
        - Use standard mathematical notation
        - Supported functions: sin, cos, tan, asin, acos, atan, ln, log, exp, sqrt, abs
        - Constants: pi, e
        - Use ** for exponentiation (e.g., x**2)
        - Use parentheses for grouping
        """)
