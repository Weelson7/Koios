import streamlit as st
import sys
import os
from pathlib import Path

# Add all module paths
current_dir = Path(__file__).parent
sys.path.extend([
    str(current_dir / "core"),
    str(current_dir / "ui"),
    str(current_dir / "utils")
])

# Import core modules
try:
    from ui.calculator_panel import render_calculator_panel
    from ui.matrix_panel import render_matrix_panel
    from ui.calculus_panel import render_calculus_panel
    from ui.equation_solver_panel import render_equation_solver_panel
    from ui.physics_panel import render_physics_panel
    from ui.visualization_panel import render_visualization_panel
    from ui.engineering_panel import render_engineering_panel
    from ui.complex_analysis_panel import render_complex_analysis_panel
    from ui.tensor_calculus_panel import render_tensor_calculus_panel
    from ui.numerical_methods_panel import render_numerical_methods_panel
    from ui.optimization_panel import render_optimization_panel
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error loading modules: {e}")
    MODULES_LOADED = False

def main():
    """Main application entry point"""
    # Get logo path
    logo_path = str(current_dir / "Koïos_Logo.png")
    
    st.set_page_config(
        page_title="Koios - Advanced Mathematical Toolset",
        page_icon=logo_path,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for cutting-edge UI
    st.markdown("""
    <style>
        /* Modern header styling */
        .main-header {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        /* Enhanced card styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            background-color: #262730;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            box-shadow: 0 4px 16px rgba(30, 136, 229, 0.4);
        }
        
        /* Input field enhancements */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            border-radius: 8px;
            border: 2px solid #3d4148;
            background-color: #1a1d26;
            color: #FAFAFA;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: #1E88E5;
            background-color: #262730;
            box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.2);
        }
        
        /* Selectbox enhancements */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1a1d26;
            border: 2px solid #3d4148;
            color: #FAFAFA !important;
        }
        
        .stSelectbox div[data-baseweb="select"] > div > div {
            color: #FAFAFA !important;
        }
        
        .stSelectbox [data-baseweb="select"] input {
            color: #FAFAFA !important;
        }
        
        .stSelectbox div[data-baseweb="select"]:focus-within > div {
            background-color: #262730;
            border-color: #1E88E5;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu items */
        [data-baseweb="menu"] {
            background-color: #1a1d26;
        }
        
        [data-baseweb="option"] {
            background-color: #1a1d26;
            color: #FAFAFA !important;
        }
        
        [data-baseweb="option"]:hover {
            background-color: #262730;
            color: #FFFFFF !important;
        }
        
        [aria-selected="true"][data-baseweb="option"] {
            background-color: #1E88E5 !important;
            color: #FFFFFF !important;
        }
        
        /* Radio button text */
        .stRadio label {
            color: #FAFAFA !important;
        }
        
        /* Checkbox text */
        .stCheckbox label {
            color: #FAFAFA !important;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            border: none;
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(30, 136, 229, 0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(30, 136, 229, 0.5);
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Info/Success/Warning boxes */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid #1E88E5;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1117 0%, #1a1d26 100%);
        }
        
        /* Headers */
        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        /* Code blocks */
        .stCodeBlock {
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(logo_path, width=80)
    with col2:
        st.markdown("<h1 style='margin-top: 0; margin-bottom: 0.2rem;'>Koios</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #B0BEC5; font-size: 1rem; margin: 0;'>Advanced Mathematical Toolset</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #78909C; font-size: 0.85rem; margin-top: 0.3rem;'>Comprehensive mathematical computation platform with advanced capabilities</p>", unsafe_allow_html=True)
    
    if not MODULES_LOADED:
        st.error("Failed to load core modules. Please check the installation.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Navigation</h2>", unsafe_allow_html=True)
        
        # Tool categories with icons
        tools = {
            "Basic Tools": {
                "Scientific Calculator": "calculator",
                "Matrix Calculator": "matrix"
            },
            "Advanced Mathematics": {
                "Calculus Tools": "calculus",
                "Equation Solver": "equations",
                "Complex Analysis": "complex",
                "Tensor Calculus": "tensor",
                "Numerical Methods": "numerical",
                "Optimization": "optimization"
            },
            "Physics & Engineering": {
                "Physics Simulations": "physics",
                "Engineering Simulations": "engineering"
            },
            "Visualization": {
                "Function Plotting": "visualization"
            }
        }
        
        # Create tool selection with improved styling
        st.markdown("<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        selected_category = st.selectbox(
            "Category",
            options=list(tools.keys()),
            label_visibility="collapsed",
            key="category_selector"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        selected_tool_name = st.selectbox(
            "Tool",
            options=list(tools[selected_category].keys()),
            label_visibility="collapsed",
            key="tool_selector"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        tool_key = tools[selected_category][selected_tool_name]
        
        st.markdown("---")
        
        # Add info panel
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.1) 0%, rgba(21, 101, 192, 0.1) 100%); 
                    padding: 1rem; border-radius: 12px; margin-top: 2rem; border-left: 4px solid #1E88E5;'>
            <h4 style='margin: 0; color: #1E88E5;'>Quick Tips</h4>
            <p style='font-size: 0.85rem; margin-top: 0.5rem; color: #B0BEC5;'>
                Use the category selector above to navigate between different mathematical tools.
                Each tool provides advanced capabilities with real-time computation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    try:
        if tool_key == "calculator":
            render_calculator_panel()
            
        elif tool_key == "matrix":
            render_matrix_panel()
            
        elif tool_key == "calculus":
            render_calculus_panel()
            
        elif tool_key == "equations":
            render_equation_solver_panel()
            
        elif tool_key == "complex":
            render_complex_analysis_panel()
            
        elif tool_key == "physics":
            render_physics_panel()
            
        elif tool_key == "visualization":
            render_visualization_panel()
            
        elif tool_key == "engineering":
            render_engineering_panel()
            
        elif tool_key == "tensor":
            render_tensor_calculus_panel()
            
        elif tool_key == "numerical":
            render_numerical_methods_panel()
            
        elif tool_key == "optimization":
            render_optimization_panel()
            
    except Exception as e:
        st.error(f"Error rendering panel: {str(e)}")
        st.exception(e)
    


if __name__ == "__main__":
    main()