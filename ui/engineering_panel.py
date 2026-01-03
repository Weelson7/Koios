import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import engineering engines
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.ui_helpers import run_task, copy_button
from utils.exceptions import InvalidInputError, NumericalInstabilityError

# Example engineering analyses
ENGINEERING_EXAMPLES = {
    "Cantilever Beam": {"type": "FEA", "geometry": "Cantilever Beam", "material": "Steel"},
    "Pipe Flow": {"type": "CFD", "geometry": "Pipe", "fluid": "Water"},
    "Waveguide": {"type": "EM", "geometry": "Rectangular Waveguide"},
    "Crystal Structure": {"type": "Material", "system": "FCC"},
}

try:
    from core.engineering.fea_engine import FEAEngine, Material as FEAMaterial, ElementType
    from core.engineering.cfd_engine import CFDEngine, CFDMesh, Fluid, BoundaryCondition, BoundaryType
    from core.engineering.electromagnetics_engine import ElectromagneticsEngine, EMaterial
    from core.engineering.material_science_engine import MaterialScienceEngine, Material, CrystalSystem, BravaisLattice
    ENGINES_LOADED = True
except ImportError as e:
    st.error(f"Failed to load engineering engines: {e}")
    ENGINES_LOADED = False

def render_engineering_panel():
    """Render the engineering simulations panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Advanced Engineering Simulations</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Professional-grade engineering analysis tools</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not ENGINES_LOADED:
        st.error("Engineering modules not properly loaded.")
        st.info("Hint: Check that all engineering engine files are in core/engineering/ directory")
        return
    
    # Example selector
    st.markdown("**Load Example:**")
    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        example_choice = st.selectbox(
            "Choose an example:",
            [""] + list(ENGINEERING_EXAMPLES.keys()),
            help="Select an example engineering analysis"
        )
    with col_ex2:
        if st.button("Load Example", disabled=not example_choice):
            if example_choice in ENGINEERING_EXAMPLES:
                st.session_state.engineering_example = ENGINEERING_EXAMPLES[example_choice]
                st.rerun()
    
    st.markdown("---")
    
    # Determine active tab from example
    example_data = st.session_state.get("engineering_example", {})
    example_type = example_data.get("type", "FEA")
    tab_index = {"FEA": 0, "CFD": 1, "EM": 2, "Material": 3}.get(example_type, 0)
    
    # Engineering simulation tabs
    tabs = st.tabs([
        "FEA (Structural)",
        "CFD (Fluids)", 
        "Electromagnetics",
        "Material Science"
    ])
    
    with tabs[0]:
        render_fea_tab()
    
    with tabs[1]:
        render_cfd_tab()
    
    with tabs[2]:
        render_electromagnetics_tab()
    
    with tabs[3]:
        render_material_science_tab()

def render_fea_tab():
    """Render Finite Element Analysis tab"""
    st.subheader("Finite Element Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Analysis Setup")
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Static Linear", "Modal Analysis", "Thermal", "Buckling"]
        )
        
        # Geometry selection
        example_data = st.session_state.get("engineering_example", {})
        default_geometry = example_data.get("geometry", "Cantilever Beam") if example_data.get("type") == "FEA" else "Cantilever Beam"
        geometry = st.selectbox(
            "Example Geometry",
            ["Cantilever Beam", "Simply Supported Beam", "Plate with Hole", "Custom"],
            index=["Cantilever Beam", "Simply Supported Beam", "Plate with Hole", "Custom"].index(default_geometry) if default_geometry in ["Cantilever Beam", "Simply Supported Beam", "Plate with Hole", "Custom"] else 0
        )
        
        # Material selection
        default_material = example_data.get("material", "Steel") if example_data.get("type") == "FEA" else "Steel"
        material = st.selectbox(
            "Material",
            ["Steel", "Aluminum", "Concrete", "Titanium", "Copper", "Inconel", "Carbon Fiber", "Custom"],
            index=["Steel", "Aluminum", "Concrete", "Titanium", "Copper", "Inconel", "Carbon Fiber", "Custom"].index(default_material) if default_material in ["Steel", "Aluminum", "Concrete", "Titanium", "Copper", "Inconel", "Carbon Fiber", "Custom"] else 0
        )
        
        # Mesh parameters
        st.markdown("### Mesh Parameters")
        if geometry == "Cantilever Beam":
            length = st.slider("Length (m)", 0.1, 5.0, 1.0, 0.1)
            height = st.slider("Height (m)", 0.01, 0.5, 0.1, 0.01)
            width = st.slider("Width (m)", 0.01, 0.5, 0.05, 0.01)
            num_elements = st.slider("Number of Elements", 5, 50, 10)
            
            # Loading
            st.markdown("### Loading")
            load_type = st.selectbox("Load Type", ["Point Load", "Distributed Load"])
            load_magnitude = st.number_input("Load Magnitude (N)", 100, 10000, 1000)
    
    with col2:
        st.markdown("### Results")
        
        if st.button("Run FEA Analysis", type="primary"):
            with st.spinner("Running FEA analysis..."):
                try:
                    # Create FEA engine
                    fea = FEAEngine()
                    
                    # Get material properties (use FEA-specific materials in SI units)
                    if material == "Steel":
                        mat = FEAMaterial.steel()
                    elif material == "Aluminum":
                        mat = FEAMaterial.aluminum()
                    elif material == "Titanium":
                        mat = FEAMaterial.titanium()
                    elif material == "Copper":
                        mat = FEAMaterial.copper()
                    elif material == "Inconel":
                        mat = FEAMaterial.inconel()
                    elif material == "Carbon Fiber":
                        mat = FEAMaterial.carbon_fiber()
                    else:
                        mat = FEAMaterial.concrete()
                    
                    # Create mesh
                    area = height * width
                    moment = width * height**3 / 12
                    
                    mesh = fea.generate_beam_mesh(
                        length=length,
                        num_elements=num_elements,
                        material=mat,
                        area=area,
                        moment_of_inertia=moment
                    )
                    
                    # Apply boundary conditions
                    mesh.apply_constraint(1, 'u', True)
                    mesh.apply_constraint(1, 'v', True)
                    mesh.apply_constraint(1, 'w', True)
                    
                    # Apply load
                    last_node = num_elements + 1
                    mesh.apply_load(last_node, 'Fy', -load_magnitude)
                    
                    # Solve
                    fea.assemble_stiffness_matrix()
                    fea.apply_boundary_conditions()
                    fea.assemble_force_vector()
                    fea.solve_static()
                    
                    # Display results
                    st.success("Analysis completed successfully!")
                    
                    # Maximum displacement
                    max_disp = np.max(np.abs(fea.displacement))
                    st.metric("Maximum Displacement", f"{max_disp*1000:.2f} mm")
                    
                    # Theoretical vs numerical
                    theoretical_disp = load_magnitude * length**3 / (3 * mat.youngs_modulus * 1e9 * moment)
                    st.metric("Theoretical Displacement", f"{theoretical_disp*1000:.2f} mm")
                    st.metric("Error", f"{abs((max_disp - theoretical_disp)/theoretical_disp)*100:.1f}%")
                    
                    # Visualize
                    fig = fea.visualize_mesh(show_loads=True, show_constraints=True)
                    st.pyplot(fig)
                    
                    # Deformed shape
                    fig2 = fea.visualize_results('displacement', scale_factor=100)
                    st.pyplot(fig2)
                    
                except Exception as e:
                    st.error(f"FEA analysis failed: {str(e)}")

def render_cfd_tab():
    """Render Computational Fluid Dynamics tab"""
    st.subheader("Computational Fluid Dynamics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Flow Setup")
        
        # Flow type
        flow_case = st.selectbox(
            "Flow Case",
            ["Lid-Driven Cavity", "Channel Flow", "Flow Over Cylinder", "Pipe Flow"]
        )
        
        # Fluid properties
        fluid_type = st.selectbox("Fluid", ["Air", "Water", "Oil", "Custom"])
        
        if fluid_type == "Custom":
            density = st.number_input("Density (kg/m³)", 0.1, 2000.0, 1000.0)
            viscosity = st.number_input("Viscosity (Pa·s)", 1e-6, 1.0, 0.001, format="%.6f")
        
        # Domain parameters
        st.markdown("### Domain")
        if flow_case == "Lid-Driven Cavity":
            cavity_size = st.slider("Cavity Size (m)", 0.01, 1.0, 0.1, 0.01)
            lid_velocity = st.slider("Lid Velocity (m/s)", 0.001, 1.0, 0.1, 0.001)
        
        # Mesh resolution
        mesh_resolution = st.selectbox("Mesh Resolution", ["Coarse (25x25)", "Medium (50x50)", "Fine (100x100)"])
        
        # Solver parameters
        st.markdown("### Solver")
        max_iterations = st.number_input("Max Iterations", 100, 5000, 1000)
        convergence_tol = st.selectbox("Convergence Tolerance", ["1e-4", "1e-5", "1e-6"])
    
    with col2:
        st.markdown("### Results")
        
        if st.button("Run CFD Simulation", type="primary"):
            with st.spinner("Running CFD simulation..."):
                try:
                    # Parse mesh resolution
                    res_map = {"Coarse (25x25)": 25, "Medium (50x50)": 50, "Fine (100x100)": 100}
                    n = res_map[mesh_resolution]
                    
                    # Create mesh
                    mesh = CFDMesh(nx=n, ny=n, lx=cavity_size, ly=cavity_size)
                    
                    # Set boundary conditions
                    mesh.set_boundary('north', BoundaryCondition(
                        BoundaryType.WALL, {'u': lid_velocity, 'v': 0.0}))
                    mesh.set_boundary('south', BoundaryCondition(
                        BoundaryType.WALL, {'u': 0.0, 'v': 0.0}))
                    mesh.set_boundary('east', BoundaryCondition(
                        BoundaryType.WALL, {'u': 0.0, 'v': 0.0}))
                    mesh.set_boundary('west', BoundaryCondition(
                        BoundaryType.WALL, {'u': 0.0, 'v': 0.0}))
                    
                    # Create fluid
                    if fluid_type == "Air":
                        fluid = Fluid.air()
                    elif fluid_type == "Water":
                        fluid = Fluid.water()
                    else:
                        fluid = Fluid("Custom", density, viscosity)
                    
                    # Create solver
                    cfd = CFDEngine(mesh, fluid)
                    cfd.tolerance = float(convergence_tol)
                    
                    # Solve
                    cfd.solve_SIMPLE(n_iterations=max_iterations)
                    
                    st.success("CFD simulation completed!")
                    
                    # Calculate Reynolds number
                    Re = cfd.calculate_reynolds_number(cavity_size, lid_velocity)
                    st.metric("Reynolds Number", f"{Re:.0f}")
                    
                    # Visualizations
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("#### Velocity Field")
                        fig1 = cfd.visualize_flow('velocity', streamlines=True)
                        st.pyplot(fig1)
                    
                    with col_b:
                        st.markdown("#### Vorticity")
                        fig2 = cfd.visualize_flow('vorticity')
                        st.pyplot(fig2)
                    
                    # Stream function
                    st.markdown("#### Stream Function")
                    fig3 = cfd.visualize_flow('stream')
                    st.pyplot(fig3)
                    
                except Exception as e:
                    st.error(f"CFD simulation failed: {str(e)}")

def render_electromagnetics_tab():
    """Render Electromagnetics tab"""
    st.subheader("Electromagnetic Field Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### EM Setup")
        
        # Application type
        em_application = st.selectbox(
            "Application",
            ["Antenna Design", "Waveguide Analysis", "PCB Trace", "Microstrip Patch"]
        )
        
        # Frequency
        freq_ghz = st.number_input("Frequency (GHz)", 0.1, 100.0, 2.4, 0.1)
        frequency = freq_ghz * 1e9
        
        if em_application == "Antenna Design":
            antenna_type = st.selectbox("Antenna Type", ["Dipole", "Monopole", "Patch", "Yagi"])
            antenna_length = st.slider("Length (wavelengths)", 0.1, 2.0, 0.5, 0.05)
        
        elif em_application == "Waveguide Analysis":
            waveguide_type = st.selectbox("Waveguide", ["WR-90 (X-band)", "WR-75 (Ku-band)", "Custom"])
            mode_m = st.number_input("Mode m", 1, 5, 1)
            mode_n = st.number_input("Mode n", 0, 5, 0)
    
    with col2:
        st.markdown("### Results")
        
        if st.button("Run EM Analysis", type="primary"):
            with st.spinner("Running electromagnetic analysis..."):
                try:
                    em = ElectromagneticsEngine(freq=frequency)
                    
                    if em_application == "Waveguide Analysis":
                        # Waveguide dimensions
                        if waveguide_type == "WR-90 (X-band)":
                            a, b = 0.02286, 0.01016
                        elif waveguide_type == "WR-75 (Ku-band)":
                            a, b = 0.01905, 0.00952
                        else:
                            a, b = 0.02, 0.01
                        
                        # Calculate modes
                        results = em.waveguide_modes(a, b, mode_m, mode_n)
                        
                        st.success("Waveguide analysis complete!")
                        
                        # Display results
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Cutoff Frequency", f"{results['cutoff_freq']/1e9:.2f} GHz")
                            if 'propagation_const' in results:
                                st.metric("Propagation Constant", f"{results['propagation_const']:.2f} rad/m")
                                st.metric("Guide Wavelength", f"{results['wavelength_guide']*1000:.1f} mm")
                        
                        with col_b:
                            if 'phase_velocity' in results:
                                st.metric("Phase Velocity", f"{results['phase_velocity']/3e8:.3f} c")
                                st.metric("Group Velocity", f"{results['group_velocity']/3e8:.3f} c")
                            else:
                                st.warning("Mode is below cutoff - evanescent")
                        
                    elif em_application == "Antenna Design":
                        # Simple visualization
                        st.success("Antenna analysis complete!")
                        
                        wavelength = 3e8 / frequency
                        st.metric("Wavelength", f"{wavelength*1000:.1f} mm")
                        st.metric("Antenna Length", f"{antenna_length * wavelength * 1000:.1f} mm")
                        
                        # Theoretical gain for dipole
                        if antenna_type == "Dipole":
                            gain_dbi = 2.15
                            st.metric("Theoretical Gain", f"{gain_dbi:.1f} dBi")
                    
                except Exception as e:
                    st.error(f"EM analysis failed: {str(e)}")

def render_material_science_tab():
    """Render Material Science tab"""
    st.subheader("Material Science Analysis")
    
    # Analysis type selection
    analysis_tabs = st.tabs([
        "Material Properties",
        "Crystal Structure", 
        "Composite Analysis",
        "Fatigue & Fracture",
        "Material Selection"
    ])
    
    engine = MaterialScienceEngine()
    
    with analysis_tabs[0]:
        st.markdown("### Material Property Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            material_select = st.selectbox(
                "Select Material",
                ["Al 6061-T6", "AISI 4140 Steel", "Ti-6Al-4V", "Copper C101", 
                 "Inconel 718", "Carbon Fiber T300", "Silicon Carbide", "Magnesium AZ31", "Custom"]
            )
            
            if material_select == "Custom":
                E = st.number_input("Young's Modulus (GPa)", 1.0, 500.0, 200.0)
                nu = st.number_input("Poisson's Ratio", 0.0, 0.5, 0.3, 0.01)
                yield_str = st.number_input("Yield Strength (MPa)", 10.0, 2000.0, 250.0)
                ult_str = st.number_input("Ultimate Strength (MPa)", 10.0, 3000.0, 400.0)
                density = st.number_input("Density (g/cm³)", 0.1, 20.0, 7.85, 0.01)
            else:
                mat_map = {
                    "Al 6061-T6": Material.aluminum_6061(),
                    "AISI 4140 Steel": Material.steel_aisi_4140(),
                    "Ti-6Al-4V": Material.titanium_grade5(),
                    "Copper C101": Material.copper_c101(),
                    "Inconel 718": Material.inconel_718(),
                    "Carbon Fiber T300": Material.carbon_fiber_t300(),
                    "Silicon Carbide": Material.silicon_carbide(),
                    "Magnesium AZ31": Material.magnesium_az31()
                }
                material = mat_map[material_select]
        
        with col2:
            st.markdown("### Calculated Properties")
            
            if material_select != "Custom":
                # Calculate elastic constants
                constants = engine.calculate_elastic_constants(material)
                
                st.metric("Shear Modulus", f"{constants['shear_modulus']:.1f} GPa")
                st.metric("Bulk Modulus", f"{constants['bulk_modulus']:.1f} GPa")
                st.metric("Specific Stiffness", f"{material.youngs_modulus/material.density:.1f} GPa·cm³/g")
                st.metric("Specific Strength", f"{material.yield_strength/material.density:.1f} MPa·cm³/g")
        
        # Stress-strain curve
        if st.button("Generate Stress-Strain Curve"):
            if material_select != "Custom":
                strain, stress = engine.stress_strain_curve(material)
                
                # Plot
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(strain * 100, stress, 'b-', linewidth=2)
                ax.set_xlabel('Strain [%]')
                ax.set_ylabel('Stress [MPa]')
                ax.set_title(f'Stress-Strain Curve - {material.name}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=material.yield_strength, color='r', linestyle='--', 
                          label=f'Yield: {material.yield_strength} MPa')
                ax.legend()
                st.pyplot(fig)
    
    with analysis_tabs[1]:
        st.markdown("### Crystal Structure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crystal_system = st.selectbox(
                "Crystal System",
                [cs.value for cs in CrystalSystem]
            )
            
            lattice_type = st.selectbox(
                "Bravais Lattice",
                ["Simple Cubic", "BCC", "FCC", "Hexagonal"]
            )
            
            st.markdown("### Lattice Parameters")
            a = st.number_input("a (Å)", 1.0, 10.0, 4.05, 0.01)
            
            if crystal_system == "HEXAGONAL":
                c = st.number_input("c (Å)", 1.0, 10.0, 4.68, 0.01)
            else:
                c = a
        
        with col2:
            if st.button("Analyze Structure"):
                lattice_map = {
                    "Simple Cubic": BravaisLattice.SIMPLE_CUBIC,
                    "BCC": BravaisLattice.BODY_CENTERED_CUBIC,
                    "FCC": BravaisLattice.FACE_CENTERED_CUBIC,
                    "Hexagonal": BravaisLattice.HEXAGONAL
                }
                
                lattice = lattice_map[lattice_type]
                params = {'a': a * 1e-10, 'c': c * 1e-10}
                
                results = engine.crystal_structure_analysis(lattice, params)
                
                st.markdown("### Results")
                st.metric("Unit Cell Volume", f"{results['volume']*1e30:.2f} ų")
                st.metric("Atoms per Cell", results['atoms_per_cell'])
                if results['atomic_packing_factor']:
                    st.metric("Atomic Packing Factor", f"{results['atomic_packing_factor']:.3f}")
                
                # Visualize
                fig = engine.visualize_crystal_structure(lattice, {'a': a, 'b': a, 'c': c})
                st.pyplot(fig)
    
    with analysis_tabs[2]:
        st.markdown("### Composite Laminate Analysis")
        st.info("Classical Laminate Theory (CLT) calculator for composite materials")
        
        # Simplified interface for demonstration
        st.markdown("#### Layup Configuration")
        num_layers = st.number_input("Number of Layers", 2, 16, 8, 2)
        symmetric = st.checkbox("Symmetric Laminate", value=True)
        
        if st.button("Analyze Composite"):
            st.success("Composite analysis would be performed here")
            st.markdown("Example results:")
            st.metric("Ex (effective)", "145.2 GPa")
            st.metric("Ey (effective)", "10.3 GPa") 
            st.metric("Gxy (effective)", "5.1 GPa")
    
    with analysis_tabs[3]:
        st.markdown("### Fatigue & Fracture Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fatigue Life Prediction")
            material_fat = st.selectbox(
                "Material",
                ["Al 6061-T6", "AISI 4140 Steel"],
                key="fatigue_mat"
            )
            
            stress_amp = st.number_input("Stress Amplitude (MPa)", 10.0, 500.0, 150.0)
            mean_stress = st.number_input("Mean Stress (MPa)", 0.0, 300.0, 50.0)
            surface = st.selectbox("Surface Finish", ["polished", "machined", "hot_rolled"])
        
        with col2:
            if st.button("Calculate Fatigue Life"):
                mat = Material.aluminum_6061() if "Al" in material_fat else Material.steel_aisi_4140()
                
                life = engine.fatigue_life_prediction(mat, stress_amp, mean_stress, surface)
                
                st.markdown("### Results")
                st.metric("Cycles to Failure", f"{life:.0e}")
                st.metric("Years at 1 Hz", f"{life / (365 * 24 * 3600):.2f}")
                
                if life > 1e6:
                    st.success("Infinite life expected")
                else:
                    st.warning("Finite life - consider redesign")
    
    with analysis_tabs[4]:
        st.markdown("### Material Selection Tool")
        st.info("Multi-criteria decision analysis for optimal material selection")
        
        # Requirements
        st.markdown("#### Requirements")
        min_yield = st.slider("Minimum Yield Strength (MPa)", 100, 1000, 300)
        max_density = st.slider("Maximum Density (g/cm³)", 1.0, 10.0, 5.0)
        
        # Weights
        st.markdown("#### Importance Weights")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            w_strength = st.slider("Strength", 0.0, 1.0, 0.3)
            w_stiffness = st.slider("Stiffness", 0.0, 1.0, 0.2)
        with col_b:
            w_density = st.slider("Low Density", 0.0, 1.0, 0.3)
            w_cost = st.slider("Cost", 0.0, 1.0, 0.2)
        
        if st.button("Find Best Material"):
            requirements = {
                'min_yield_strength': min_yield,
                'max_density': max_density
            }
            
            weights = {
                'strength': w_strength,
                'stiffness': w_stiffness,
                'density': w_density,
                'cost': w_cost
            }
            
            results = engine.material_selection(requirements, weights)
            
            st.markdown("### Recommended Materials")
            for i, (mat_name, score) in enumerate(results[:3]):
                st.metric(f"{i+1}. {mat_name}", f"Score: {score:.2f}")