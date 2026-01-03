import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.physics_simulator import physics_simulator
from utils.ui_helpers import run_task, copy_button
from utils.exceptions import InvalidInputError, NumericalInstabilityError
import math

# Example physics simulations for quick loading
PHYSICS_EXAMPLES = {
    "Classic Projectile": {"sim": "projectile_motion", "params": {"v0": 50, "angle": 45}},
    "Simple Pendulum": {"sim": "pendulum", "params": {"length": 1.0, "angle0": 30}},
    "Damped Oscillator": {"sim": "damped_oscillator", "params": {"k": 10, "m": 1, "c": 0.5}},
    "RC Circuit": {"sim": "circuit_rc", "params": {"R": 1000, "C": 1e-6}},
}

def render_physics_panel():
    """Render the physics simulations panel"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(30, 136, 229, 0.15) 0%, rgba(21, 101, 192, 0.15) 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #1E88E5;'>
        <h2 style='margin: 0; color: #FAFAFA;'>Physics Simulations</h2>
        <p style='margin: 0.5rem 0 0 0; color: #B0BEC5;'>Interactive physics simulations with real-time parameter adjustment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example selector
    st.markdown("**Load Example:**")
    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        example_choice = st.selectbox(
            "Choose an example:",
            [""] + list(PHYSICS_EXAMPLES.keys()),
            help="Select an example simulation"
        )
    with col_ex2:
        if st.button("Load Example", disabled=not example_choice):
            if example_choice in PHYSICS_EXAMPLES:
                st.session_state.physics_example = PHYSICS_EXAMPLES[example_choice]
                st.rerun()
    
    st.markdown("---")

    # Get available simulations
    available_sims = physics_simulator.get_available_simulations()

    # Simulation selection - use session state example if available
    example_data = st.session_state.get("physics_example", {})
    default_sim = example_data.get("sim", "projectile_motion")
    
    simulation_name = st.selectbox(
        "Select Physics Simulation:",
        [
             "projectile_motion",
             "simple_harmonic_motion",
             "damped_oscillator",
             "pendulum",
             "circuit_rc",
             "circuit_rl",
             "circuit_rlc",
             "electromagnetic_wave",
             "doppler_effect",
             "wave_interference",
             "heat_conduction",
             "orbital_motion",
             "three_body_orbital",
             "four_body_orbital",
             "quantum_harmonic_oscillator",
             "blackbody_radiation",
             "fluid_flow",
             "electromagnetic_field",
             "photoelectric_effect",
             "compton_scattering",
             "relativity_time_dilation",
             "nuclear_decay",
             "particle_accelerator",
             "gravitational_waves",
             "plasma_confinement",
             "superconductivity",
             "quantum_tunneling",
             "laser_physics",
             "solid_state_physics",
             "chaos_theory",
             "quantum_mechanics",
             "crystallography",
             "optics_lenses"
         ],
        index=[
             "projectile_motion",
             "simple_harmonic_motion",
             "damped_oscillator",
             "pendulum",
             "circuit_rc",
             "circuit_rl",
             "circuit_rlc",
             "electromagnetic_wave",
             "doppler_effect",
             "wave_interference",
             "heat_conduction",
             "orbital_motion",
             "three_body_orbital",
             "four_body_orbital",
             "quantum_harmonic_oscillator",
             "blackbody_radiation",
             "fluid_flow",
             "electromagnetic_field",
             "photoelectric_effect",
             "compton_scattering",
             "relativity_time_dilation",
             "nuclear_decay",
             "particle_accelerator",
             "gravitational_waves",
             "plasma_confinement",
             "superconductivity",
             "quantum_tunneling",
             "laser_physics",
             "solid_state_physics",
             "chaos_theory",
             "quantum_mechanics",
             "crystallography",
             "optics_lenses"
         ].index(default_sim) if default_sim in [
             "projectile_motion",
             "simple_harmonic_motion",
             "damped_oscillator",
             "pendulum",
             "circuit_rc",
             "circuit_rl",
             "circuit_rlc",
             "electromagnetic_wave",
             "doppler_effect",
             "wave_interference",
             "heat_conduction",
             "orbital_motion",
             "three_body_orbital",
             "four_body_orbital",
             "quantum_harmonic_oscillator",
             "blackbody_radiation",
             "fluid_flow",
             "electromagnetic_field",
             "photoelectric_effect",
             "compton_scattering",
             "relativity_time_dilation",
             "nuclear_decay",
             "particle_accelerator",
             "gravitational_waves",
             "plasma_confinement",
             "superconductivity",
             "quantum_tunneling",
             "laser_physics",
             "solid_state_physics",
             "chaos_theory",
             "quantum_mechanics",
             "crystallography",
             "optics_lenses"
         ] else 0,
        format_func=lambda x: {
            'projectile_motion': 'Projectile Motion',
            'simple_harmonic_motion': 'Simple Harmonic Motion',
            'damped_oscillator': 'Damped Oscillator',
            'pendulum': 'Simple Pendulum',
            'circuit_rc': 'RC Circuit',
            'circuit_rl': 'RL Circuit',
            'circuit_rlc': 'RLC Circuit',
            'optics_lenses': 'Optical Lenses',
            'electromagnetic_wave': 'Electromagnetic Wave',
            'doppler_effect': 'Doppler Effect',
            'wave_interference': 'Wave Interference',
            'heat_conduction': 'Heat Conduction',
            'orbital_motion': 'Orbital Motion',
            'electromagnetic_field': 'Electromagnetic Field',
            'photoelectric_effect': 'Photoelectric Effect',
            'relativity_time_dilation': 'Relativity Time Dilation',
            'nuclear_decay': 'Nuclear Decay',
            'particle_accelerator': 'Particle Accelerator',
            'fluid_flow': 'Fluid Flow',
            'three_body_orbital': 'Three Body Orbital',
            'four_body_orbital': 'Four Body Orbital'
        }.get(x, x.replace('_', ' ').title())
    )

    # Render specific simulation interface
    if simulation_name == 'projectile_motion':
        render_projectile_motion()
    elif simulation_name == 'simple_harmonic_motion':
        render_simple_harmonic_motion()
    elif simulation_name == 'damped_oscillator':
        render_damped_oscillator()
    elif simulation_name == 'pendulum':
        render_pendulum_simulation()
    elif simulation_name == 'circuit_rc':
        render_rc_circuit()
    elif simulation_name == 'circuit_rl':
        render_rl_circuit()
    elif simulation_name == 'circuit_rlc':
        render_rlc_circuit()
    elif simulation_name == 'optics_lenses':
        render_optics_lenses()
    elif simulation_name == 'electromagnetic_wave':
        render_electromagnetic_wave()
    elif simulation_name == 'doppler_effect':
        render_doppler_effect()
    elif simulation_name == 'wave_interference':
        render_wave_interference()
    elif simulation_name == 'heat_conduction':
        render_heat_conduction()
    elif simulation_name == 'orbital_motion':
        render_orbital_motion()
    elif simulation_name == 'electromagnetic_field':
        render_electromagnetic_field()
    elif simulation_name == 'photoelectric_effect':
        render_photoelectric_effect()
    elif simulation_name == 'relativity_time_dilation':
        render_relativity_time_dilation()
    elif simulation_name == 'nuclear_decay':
        render_nuclear_decay()
    elif simulation_name == 'particle_accelerator':
        render_particle_accelerator()
    elif simulation_name == 'fluid_flow':
        render_fluid_flow()
    elif simulation_name == 'three_body_orbital':
        render_three_body_orbital()
    elif simulation_name == 'four_body_orbital':
        render_four_body_orbital()
    else:
        st.warning(f"UI not implemented for '{simulation_name}'. Select a different simulation.")

def render_projectile_motion():
    """Render projectile motion simulation"""
    st.subheader("Projectile Motion Simulation")
    st.markdown("*Simulate the motion of a projectile under gravity*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    # Get example params if available
    example_data = st.session_state.get("physics_example", {})
    example_params = example_data.get("params", {}) if example_data.get("sim") == "projectile_motion" else {}

    with col1:
        v0 = st.slider("Initial Velocity (m/s):", 1.0, 100.0, float(example_params.get("v0", 20.0)), 1.0)
        angle = st.slider("Launch Angle (degrees):", 0.0, 90.0, float(example_params.get("angle", 45.0)), 1.0)

    with col2:
        g = st.slider("Gravity (m/s²):", 1.0, 20.0, 9.81, 0.1)
        num_points = st.slider("Data Points:", 50, 500, 100, 10)

    with col3:
        show_vectors = st.checkbox("Show Velocity Vectors", False)
        show_trajectory = st.checkbox("Show Trajectory", True)

    # Real-time simulation
    parameters = {
        'v0': v0,
        'angle': angle,
        'g': g,
        'num_points': num_points
    }

    result = physics_simulator.run_simulation('projectile_motion', parameters)

    if result['success']:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Max Height", f"{result['max_height']:.2f} m")
        with col2:
            st.metric("Range", f"{result['range']:.2f} m")
        with col3:
            st.metric("Flight Time", f"{result['flight_time']:.2f} s")
        with col4:
            impact_velocity = math.sqrt(
                result['x_velocity'][-1]**2 + result['y_velocity'][-1]**2
            ) if result['x_velocity'] else 0
            st.metric("Impact Speed", f"{impact_velocity:.2f} m/s")

        # Create visualization
        fig = create_projectile_plot(result, show_vectors, show_trajectory)
        st.plotly_chart(fig, width='stretch')

        # Additional analysis
        render_projectile_analysis(result, parameters)

    else:
        st.error(f"Simulation failed: {result['error']}")

def render_simple_harmonic_motion():
    """Render simple harmonic motion simulation"""
    st.subheader("Simple Harmonic Motion")
    st.markdown("*Simulate oscillatory motion with adjustable parameters*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    # Get example params if available
    example_data = st.session_state.get("physics_example", {})
    example_params = example_data.get("params", {}) if example_data.get("sim") == "simple_harmonic_motion" else {}

    with col1:
        amplitude = st.slider("Amplitude:", 0.1, 5.0, float(example_params.get("amplitude", 1.0)), 0.1)
        frequency = st.slider("Frequency (Hz):", 0.1, 5.0, float(example_params.get("frequency", 1.0)), 0.1)

    with col2:
        phase = st.slider("Phase (radians):", -math.pi, math.pi, 0.0, 0.1)
        time_max = st.slider("Simulation Time (s):", 1.0, 20.0, 4.0, 0.5)

    # Simulation parameters
    parameters = {
        'amplitude': amplitude,
        'frequency': frequency,
        'phase': phase,
        'time_max': time_max,
        'num_points': 400
    }

    result = physics_simulator.run_simulation('simple_harmonic_motion', parameters)

    if result['success']:
        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            period = 1.0 / frequency
            st.metric("Period", f"{period:.2f} s")
        with col2:
            omega = 2 * math.pi * frequency
            st.metric("Angular Frequency", f"{omega:.2f} rad/s")
        with col3:
            max_velocity = amplitude * omega
            st.metric("Max Velocity", f"{max_velocity:.2f} m/s")

        # Create visualization
        fig = create_shm_plot(result)
        st.plotly_chart(fig, width='stretch')

        # Phase space plot
        render_phase_space(result)

    else:
        st.error(f"Simulation failed: {result['error']}")

def render_damped_oscillator():
    """Render damped oscillator simulation"""
    st.subheader("Damped Harmonic Oscillator")
    st.markdown("*Simulate oscillations with energy loss due to damping*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    # Get example params if available
    example_data = st.session_state.get("physics_example", {})
    example_params = example_data.get("params", {}) if example_data.get("sim") == "damped_oscillator" else {}

    with col1:
        amplitude = st.slider("Initial Amplitude:", 0.1, 5.0, float(example_params.get("amplitude", 1.0)), 0.1, key="damp_amp")
        omega0 = st.slider("Natural Frequency:", 0.5, 10.0, float(example_params.get("omega0", 2.0)), 0.1)

    with col2:
        gamma = st.slider("Damping Coefficient:", 0.0, 2.0, 0.1, 0.01)
        time_max = st.slider("Simulation Time:", 1.0, 20.0, 10.0, 0.5, key="damp_time")

    with col3:
        # Damping regime indicators
        discriminant = gamma**2 - omega0**2
        if discriminant < 0:
            damping_type = "Underdamped"
            color = "green"
        elif discriminant > 0:
            damping_type = "Overdamped"
            color = "red"
        else:
            damping_type = "Critically Damped"
            color = "orange"

        st.markdown(f"**Damping Regime:**")
        st.markdown(f"<span style='color:{color}'>{damping_type}</span>", unsafe_allow_html=True)

    # Simulation
    parameters = {
        'amplitude': amplitude,
        'omega0': omega0,
        'gamma': gamma,
        'time_max': time_max,
        'num_points': 500
    }

    result = physics_simulator.run_simulation('damped_oscillator', parameters)

    if result['success']:
        # Create visualization
        fig = create_damped_oscillator_plot(result, parameters)
        st.plotly_chart(fig, width='stretch')

        # Energy analysis
        render_energy_analysis(result, parameters)

    else:
        st.error(f"Simulation failed: {result['error']}")

def render_pendulum_simulation():
    """Render simple pendulum simulation"""
    st.subheader("Simple Pendulum")
    st.markdown("*Simulate pendulum motion (small angle approximation)*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    # Get example params if available
    example_data = st.session_state.get("physics_example", {})
    example_params = example_data.get("params", {}) if example_data.get("sim") == "pendulum" else {}

    with col1:
        length = st.slider("Length (m):", 0.1, 5.0, float(example_params.get("length", 1.0)), 0.1)
        angle0 = st.slider("Initial Angle (degrees):", 1.0, 45.0, float(example_params.get("angle0", 10.0)), 1.0)

    with col2:
        g = st.slider("Gravity (m/s²):", 1.0, 20.0, 9.81, 0.1, key="pend_g")
        time_max = st.slider("Simulation Time:", 1.0, 20.0, 4.0, 0.5, key="pend_time")

    with col3:
        show_trajectory = st.checkbox("Show Pendulum Path", True)
        show_forces = st.checkbox("Show Force Analysis", False)

    # Simulation
    parameters = {
        'length': length,
        'angle0': angle0,
        'g': g,
        'time_max': time_max,
        'num_points': 200
    }

    result = physics_simulator.run_simulation('pendulum', parameters)

    if result['success']:
        # Display metrics
        omega = math.sqrt(g / length)
        period = 2 * math.pi / omega

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Period", f"{period:.2f} s")
        with col2:
            st.metric("Frequency", f"{1/period:.2f} Hz")
        with col3:
            max_speed = abs(angle0 * math.pi/180) * math.sqrt(g * length)
            st.metric("Max Speed", f"{max_speed:.2f} m/s")

        # Create visualization
        fig = create_pendulum_plot(result, parameters, show_trajectory)
        st.plotly_chart(fig, width='stretch')

        if show_forces:
            render_pendulum_forces(result, parameters)

    else:
        st.error(f"Simulation failed: {result['error']}")

def render_rc_circuit():
    """Render RC circuit simulation"""
    st.subheader("RC Circuit")
    st.markdown("*Simulate capacitor charging and discharging in RC circuit*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    # Get example params if available
    example_data = st.session_state.get("physics_example", {})
    example_params = example_data.get("params", {}) if example_data.get("sim") == "circuit_rc" else {}

    with col1:
        R = st.slider("Resistance (Ω):", 100.0, 10000.0, float(example_params.get("R", 1000.0)), 100.0)
        C = st.slider("Capacitance (μF):", 0.1, 100.0, float(example_params.get("C", 1.0)), 0.1)

    with col2:
        V0 = st.slider("Initial Voltage (V):", -10.0, 10.0, 0.0, 0.5)
        Vs = st.slider("Source Voltage (V):", 0.0, 12.0, 5.0, 0.5)

    with col3:
        # Calculate time constant
        C_farads = C * 1e-6  # Convert to Farads
        tau = R * C_farads
        st.metric("Time Constant (τ)", f"{tau*1000:.2f} ms")

        time_max = st.slider("Simulation Time (ms):", 0.1, 50.0, 5*tau*1000, 0.1)

    # Simulation
    parameters = {
        'R': R,
        'C': C_farads,
        'V0': V0,
        'Vs': Vs,
        'time_max': time_max / 1000,  # Convert to seconds
        'num_points': 200
    }

    result = physics_simulator.run_simulation('circuit_rc', parameters)

    if result['success']:
        # Create visualization
        fig = create_rc_circuit_plot(result, parameters, tau)
        st.plotly_chart(fig, width='stretch')

        # Circuit analysis
        render_rc_analysis(result, parameters, tau)

    else:
        st.error(f"Simulation failed: {result['error']}")

def render_rl_circuit():
    """Render RL circuit simulation"""
    st.subheader("RL Circuit")
    st.markdown("*Simulate inductor behavior in RL circuit*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        R = st.slider("Resistance (Ω):", 10.0, 1000.0, 100.0, 10.0, key="rl_R")
        L = st.slider("Inductance (mH):", 0.1, 100.0, 1.0, 0.1, key="rl_L")

    with col2:
        V0 = st.slider("Source Voltage (V):", 0.0, 12.0, 5.0, 0.5, key="rl_V0")
        I0 = st.slider("Initial Current (A):", -1.0, 1.0, 0.0, 0.01, key="rl_I0")

    with col3:
        # Calculate time constant
        L_henries = L * 1e-3  # Convert to Henries
        tau = L_henries / R
        st.metric("Time Constant (τ)", f"{tau*1000:.2f} ms")

        time_max = st.slider("Simulation Time (ms):", 0.1, 20.0, 5*tau*1000, 0.1, key="rl_time")

    # Display circuit characteristics
    col1, col2 = st.columns(2)

    with col1:
        steady_current = V0 / R
        st.metric("Steady-State Current", f"{steady_current:.3f} A")

    with col2:
        initial_voltage = V0 - I0 * R
        st.metric("Initial Inductor Voltage", f"{initial_voltage:.2f} V")

    # Simulation
    parameters = {
        'R': R,
        'L': L_henries,
        'V0': V0,
        'I0': I0,
        'time_max': time_max / 1000,  # Convert to seconds
        'num_points': 200
    }

    result = physics_simulator.run_simulation('circuit_rl', parameters)

    if result['success']:
        # Create visualization
        fig = create_rl_circuit_plot(result, parameters, tau)
        st.plotly_chart(fig, width='stretch')

        # Circuit analysis
        render_rl_analysis(result, parameters, tau)

    else:
        st.error(f"Simulation failed: {result['error']}")

def render_rlc_circuit():
    """Render RLC circuit simulation"""
    st.subheader("RLC Circuit")
    st.markdown("*Simulate transient response of RLC circuit*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        R = st.slider("Resistance (Ω):", 10.0, 1000.0, 100.0, 10.0, key="rlc_R")
        L = st.slider("Inductance (mH):", 0.1, 100.0, 1.0, 0.1)

    with col2:
        C = st.slider("Capacitance (μF):", 0.1, 100.0, 1.0, 0.1, key="rlc_C")
        V0 = st.slider("Initial Voltage (V):", 0.0, 12.0, 5.0, 0.5, key="rlc_V0")

    with col3:
        I0 = st.slider("Initial Current (A):", -1.0, 1.0, 0.0, 0.01)
        time_max = st.slider("Simulation Time (ms):", 0.1, 10.0, 1.0, 0.1, key="rlc_time")

    # Circuit analysis
    L_henries = L * 1e-3  # Convert to Henries
    C_farads = C * 1e-6   # Convert to Farads

    omega0 = 1.0 / math.sqrt(L_henries * C_farads)
    gamma = R / (2 * L_henries)
    discriminant = gamma**2 - omega0**2

    # Display circuit characteristics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Natural Frequency", f"{omega0:.0f} rad/s")

    with col2:
        if discriminant < 0:
            circuit_type = "Underdamped"
            color = "green"
        elif discriminant > 0:
            circuit_type = "Overdamped"
            color = "red"
        else:
            circuit_type = "Critically Damped"
            color = "orange"

        st.markdown(f"**Circuit Type:**")
        st.markdown(f"<span style='color:{color}'>{circuit_type}</span>", unsafe_allow_html=True)

    with col3:
        Q_factor = omega0 / (2 * gamma) if gamma > 0 else float('inf')
        st.metric("Quality Factor (Q)", f"{Q_factor:.2f}")

    # Simulation
    parameters = {
        'R': R,
        'L': L_henries,
        'C': C_farads,
        'V0': V0,
        'I0': I0,
        'time_max': time_max / 1000,  # Convert to seconds
        'num_points': 500
    }

    result = physics_simulator.run_simulation('circuit_rlc', parameters)

    if result['success']:
        # Create visualization
        fig = create_rlc_circuit_plot(result, parameters)
        st.plotly_chart(fig, width='stretch')

        # Power and energy analysis
        render_rlc_analysis(result, parameters)

    else:
        st.error(f"Simulation failed: {result['error']}")

# Plotting functions
def create_projectile_plot(result, show_vectors, show_trajectory):
    """Create projectile motion visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trajectory', 'Position vs Time', 'Velocity vs Time', 'Height vs Range'),
        specs=[[{"colspan": 2}, None],
               [{"secondary_y": True}, {}]]
    )

    time = result['time']
    x_pos = result['x_position']
    y_pos = result['y_position']
    x_vel = result['x_velocity']
    y_vel = result['y_velocity']

    # Trajectory plot
    if show_trajectory:
        fig.add_trace(
            go.Scatter(x=x_pos, y=y_pos, mode='lines', name='Trajectory', 
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )

    # Position vs time
    fig.add_trace(
        go.Scatter(x=time, y=x_pos, mode='lines', name='X Position', 
                  line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=y_pos, mode='lines', name='Y Position', 
                  line=dict(color='green')),
        row=2, col=1, secondary_y=True
    )

    # Velocity vs time
    fig.add_trace(
        go.Scatter(x=time, y=x_vel, mode='lines', name='X Velocity', 
                  line=dict(color='orange')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=time, y=y_vel, mode='lines', name='Y Velocity', 
                  line=dict(color='purple')),
        row=2, col=2
    )

    # Add velocity vectors if requested
    if show_vectors and len(time) > 10:
        step = len(time) // 10
        for i in range(0, len(time), step):
            fig.add_annotation(
                x=x_pos[i], y=y_pos[i],
                ax=x_pos[i] + x_vel[i]*0.1, ay=y_pos[i] + y_vel[i]*0.1,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red",
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        title="Projectile Motion Analysis",
        showlegend=True,
        height=600
    )

    fig.update_xaxes(title_text="Horizontal Distance (m)", row=1, col=1)
    fig.update_yaxes(title_text="Height (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="X Position (m)", row=2, col=1)
    fig.update_yaxes(title_text="Y Position (m)", secondary_y=True, row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)

    return fig

def create_shm_plot(result):
    """Create simple harmonic motion visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Position vs Time', 'Velocity vs Time', 
                       'Acceleration vs Time', 'Phase Space'),
    )

    time = result['time']
    position = result['position']
    velocity = result['velocity']
    acceleration = result['acceleration']

    # Position vs time
    fig.add_trace(
        go.Scatter(x=time, y=position, mode='lines', name='Position',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Velocity vs time
    fig.add_trace(
        go.Scatter(x=time, y=velocity, mode='lines', name='Velocity',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )

    # Acceleration vs time
    fig.add_trace(
        go.Scatter(x=time, y=acceleration, mode='lines', name='Acceleration',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )

    # Phase space (position vs velocity)
    fig.add_trace(
        go.Scatter(x=position, y=velocity, mode='lines', name='Phase Space',
                  line=dict(color='purple', width=2)),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title="Simple Harmonic Motion Analysis",
        showlegend=False,
        height=600
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    fig.update_xaxes(title_text="Position (m)", row=2, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)

    return fig

def create_damped_oscillator_plot(result, parameters):
    """Create damped oscillator visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Damped Oscillation', 'Energy Decay')
    )

    time = result['time']
    position = result['position']
    velocity = result['velocity']
    envelope = result['envelope']

    # Damped oscillation with envelope
    fig.add_trace(
        go.Scatter(x=time, y=position, mode='lines', name='Position',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time, y=envelope, mode='lines', name='Envelope',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time, y=[-x for x in envelope], mode='lines', 
                  name='Envelope (negative)',
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )

    # Energy calculation and plot
    omega0 = parameters['omega0']
    gamma = parameters['gamma']

    # Total energy (approximate)
    kinetic_energy = [0.5 * v**2 for v in velocity]
    potential_energy = [0.5 * omega0**2 * x**2 for x in result['data']['position']]
    total_energy = [k + p for k, p in zip(kinetic_energy, potential_energy)]

    fig.add_trace(
        go.Scatter(x=time, y=total_energy, mode='lines', name='Total Energy',
                  line=dict(color='purple', width=2)),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="Damped Harmonic Oscillator",
        showlegend=True,
        height=400
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Energy", row=1, col=2)

    return fig

def create_pendulum_plot(result, parameters, show_trajectory):
    """Create pendulum visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Pendulum Motion', 'Angular Position vs Time')
    )

    time = result['time']
    angle = result['angle']
    x_pos = result['x_position']
    y_pos = result['y_position']

    # Pendulum visualization
    if show_trajectory:
        fig.add_trace(
            go.Scatter(x=x_pos, y=y_pos, mode='lines', name='Pendulum Path',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

    # Add pendulum bob at current position (use last position)
    fig.add_trace(
        go.Scatter(x=[0, x_pos[-1]], y=[0, y_pos[-1]], 
                  mode='lines+markers', name='Pendulum',
                  line=dict(color='black', width=3),
                  marker=dict(size=[5, 15], color=['black', 'red'])),
        row=1, col=1
    )

    # Angular position vs time
    fig.add_trace(
        go.Scatter(x=time, y=angle, mode='lines', name='Angular Position',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="Simple Pendulum Simulation",
        showlegend=True,
        height=400
    )

    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Angle (degrees)", row=1, col=2)

    # Set equal aspect ratio for pendulum plot
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    return fig

def create_rc_circuit_plot(result, parameters, tau):
    """Create RC circuit visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Voltage vs Time', 'Current vs Time')
    )

    time = np.array(result['time']) * 1000  # Convert to ms
    voltage = result['voltage']
    current = np.array(result['current']) * 1000  # Convert to mA

    # Voltage plot
    fig.add_trace(
        go.Scatter(x=time, y=voltage, mode='lines', name='Capacitor Voltage',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Add time constant markers
    tau_ms = tau * 1000
    for i in range(1, 6):
        if i * tau_ms <= time[-1]:
            fig.add_vline(x=i * tau_ms, line_dash="dash", line_color="red",
                         annotation_text=f"{i}τ", row=1, col=1)

    # Current plot
    fig.add_trace(
        go.Scatter(x=time, y=current, mode='lines', name='Current',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="RC Circuit Response",
        showlegend=True,
        height=400
    )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Current (mA)", row=1, col=2)

    return fig

def create_rlc_circuit_plot(result, parameters):
    """Create RLC circuit visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Voltage vs Time', 'Current vs Time')
    )

    time = np.array(result['time']) * 1000  # Convert to ms
    voltage = result['voltage']
    current = np.array(result['current']) * 1000  # Convert to mA

    # Voltage plot
    fig.add_trace(
        go.Scatter(x=time, y=voltage, mode='lines', name='Capacitor Voltage',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Current plot
    fig.add_trace(
        go.Scatter(x=time, y=current, mode='lines', name='Current',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="RLC Circuit Transient Response",
        showlegend=True,
        height=400
    )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Current (mA)", row=1, col=2)

    return fig

# Analysis functions
def render_projectile_analysis(result, parameters):
    """Render projectile motion analysis"""
    st.markdown("---")
    st.subheader("Projectile Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trajectory Equations:**")
        st.latex(r"x(t) = v_0 \cos(\theta) \cdot t")
        st.latex(r"y(t) = v_0 \sin(\theta) \cdot t - \frac{1}{2}gt^2")

        st.markdown("**Key Formulas:**")
        st.latex(r"R = \frac{v_0^2 \sin(2\theta)}{g}")
        st.latex(r"H = \frac{v_0^2 \sin^2(\theta)}{2g}")

    with col2:
        st.markdown("**Optimal Angle Analysis:**")
        g = parameters['g']
        v0 = parameters['v0']

        # Calculate range for different angles
        angles = np.linspace(0, 90, 91)
        ranges = [(v0**2 * np.sin(2 * np.radians(angle)) / g) for angle in angles]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=angles, y=ranges, mode='lines',
            name='Range vs Angle'
        ))

        # Mark current angle
        current_range = result['range']
        current_angle = parameters['angle']
        fig.add_trace(go.Scatter(
            x=[current_angle], y=[current_range],
            mode='markers', marker=dict(size=10, color='red'),
            name='Current'
        ))

        fig.update_layout(
            title="Range vs Launch Angle",
            xaxis_title="Angle (degrees)",
            yaxis_title="Range (m)",
            height=300
        )

        st.plotly_chart(fig, width='stretch')

def render_phase_space(result):
    """Render phase space plot for SHM"""
    st.markdown("---")
    st.subheader("Phase Space Analysis")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result['position'], y=result['velocity'],
        mode='lines', name='Phase Trajectory',
        line=dict(color='purple', width=2)
    ))

    fig.update_layout(
        title="Phase Space (Position vs Velocity)",
        xaxis_title="Position (m)",
        yaxis_title="Velocity (m/s)",
        height=400
    )

    st.plotly_chart(fig, width='stretch')

    st.info("The phase space plot shows the relationship between position and velocity. For SHM, this creates an elliptical trajectory.")

def render_energy_analysis(result, parameters):
    """Render energy analysis for damped oscillator"""
    st.markdown("---")
    st.subheader("Energy Analysis")

    # Calculate energies
    omega0 = parameters['omega0']
    gamma = parameters['gamma']

    time = result['time']
    # Handle different position key names
    position = result.get('position') or result.get('displacement') or result.get('data', {}).get('position', [])
    velocity = result.get('velocity') or result.get('data', {}).get('velocity', [])

    # Check if data exists
    if not position or not velocity:
        st.error("Position or velocity data not available for energy analysis")
        return

    kinetic_energy = [0.5 * v**2 for v in velocity]
    potential_energy = [0.5 * omega0**2 * x**2 for x in position]
    total_energy = [k + p for k, p in zip(kinetic_energy, potential_energy)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time, y=kinetic_energy, mode='lines',
        name='Kinetic Energy', line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=time, y=potential_energy, mode='lines',
        name='Potential Energy', line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=time, y=total_energy, mode='lines',
        name='Total Energy', line=dict(color='black', width=3)
    ))

    fig.update_layout(
        title="Energy vs Time",
        xaxis_title="Time (s)",
        yaxis_title="Energy",
        height=400
    )

    st.plotly_chart(fig, width='stretch')

def render_pendulum_forces(result, parameters):
    """Render pendulum force analysis"""
    st.markdown("---")
    st.subheader("Force Analysis")

    st.markdown("**Forces acting on the pendulum bob:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Gravitational Force:**")
        st.latex(r"F_g = mg")

        st.markdown("**Tension Force:**")
        st.latex(r"T = mg\cos(\theta) + \frac{mv^2}{L}")

        st.markdown("**Restoring Force:**")
        st.latex(r"F_{restoring} = -mg\sin(\theta)")

    with col2:
        st.markdown("**Small Angle Approximation:**")
        st.latex(r"\sin(\theta) \approx \theta")
        st.latex(r"F_{restoring} \approx -mg\theta")

        st.markdown("**Equation of Motion:**")
        st.latex(r"\frac{d^2\theta}{dt^2} + \frac{g}{L}\theta = 0")

def render_rc_analysis(result, parameters, tau):
    """Render RC circuit analysis"""
    st.markdown("---")
    st.subheader("Circuit Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Circuit Equations:**")
        st.latex(r"V_C(t) = V_s + (V_0 - V_s)e^{-t/\tau}")
        st.latex(r"I(t) = \frac{V_s - V_0}{R}e^{-t/\tau}")
        st.latex(r"\tau = RC")

        st.markdown("**Time Constants:**")
        for i in range(1, 6):
            percentage = (1 - math.exp(-i)) * 100
            st.write(f"{i}τ: {percentage:.1f}% of final value")

    with col2:
        st.markdown("**Energy Analysis:**")

        # Calculate energy stored in capacitor
        C = parameters['C']
        voltage = result['voltage']

        energy = [0.5 * C * v**2 for v in voltage]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.array(result['time']) * 1000, y=energy,
            mode='lines', name='Stored Energy',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title="Energy Stored in Capacitor",
            xaxis_title="Time (ms)",
            yaxis_title="Energy (J)",
            height=300
        )

        st.plotly_chart(fig, width='stretch')

def render_rlc_analysis(result, parameters):
    """Render RLC circuit analysis"""
    st.markdown("---")
    st.subheader("Power and Energy Analysis")

    # Calculate power dissipated in resistor
    R = parameters['R']
    current = result['current']

    power = [R * i**2 for i in current]

    # Calculate energy stored in inductor and capacitor
    L = parameters['L']
    C = parameters['C']
    voltage = result['voltage']

    inductor_energy = [0.5 * L * i**2 for i in current]
    capacitor_energy = [0.5 * C * v**2 for v in voltage]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Power Dissipation', 'Energy Storage')
    )

    time_ms = np.array(result['time']) * 1000

    # Power plot
    fig.add_trace(
        go.Scatter(x=time_ms, y=power, mode='lines', name='Power Dissipated',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Energy plot
    fig.add_trace(
        go.Scatter(x=time_ms, y=inductor_energy, mode='lines', name='Inductor Energy',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=time_ms, y=capacitor_energy, mode='lines', name='Capacitor Energy',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )

    total_stored_energy = [l + c for l, c in zip(inductor_energy, capacitor_energy)]
    fig.add_trace(
        go.Scatter(x=time_ms, y=total_stored_energy, mode='lines', name='Total Stored',
                  line=dict(color='black', width=2, dash='dash')),
        row=1, col=2
    )

    fig.update_layout(
        title="RLC Circuit Power and Energy",
        showlegend=True,
        height=400
    )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Energy (J)", row=1, col=2)

    st.plotly_chart(fig, width='stretch')

def create_rl_circuit_plot(result, parameters, tau):
    """Create RL circuit visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Current vs Time', 'Inductor Voltage vs Time')
    )

    time = np.array(result['time']) * 1000  # Convert to ms
    current = np.array(result['current']) * 1000  # Convert to mA
    voltage_inductor = result['voltage_inductor']

    # Current plot
    fig.add_trace(
        go.Scatter(x=time, y=current, mode='lines', name='Current',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Add time constant markers
    tau_ms = tau * 1000
    for i in range(1, 6):
        if i * tau_ms <= time[-1]:
            fig.add_vline(x=i * tau_ms, line_dash="dash", line_color="red",
                         annotation_text=f"{i}τ", row=1, col=1)

    # Voltage plot
    fig.add_trace(
        go.Scatter(x=time, y=voltage_inductor, mode='lines', name='Inductor Voltage',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="RL Circuit Response",
        showlegend=True,
        height=400
    )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Current (mA)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)

    return fig

def render_rl_analysis(result, parameters, tau):
    """Render RL circuit analysis"""
    st.markdown("---")
    st.subheader("Circuit Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Circuit Equations:**")
        st.latex(r"I(t) = \frac{V_0}{R}(1 - e^{-t/\tau}) + I_0 e^{-t/\tau}")
        st.latex(r"V_L(t) = V_0 - I(t)R")
        st.latex(r"\tau = \frac{L}{R}")

        st.markdown("**Time Constants:**")
        for i in range(1, 6):
            percentage = (1 - math.exp(-i)) * 100
            st.write(f"{i}τ: {percentage:.1f}% of final value")

    with col2:
        st.markdown("**Energy Analysis:**")

        # Calculate energy stored in inductor
        L = parameters['L']
        current = result['current']

        energy = [0.5 * L * i**2 for i in current]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.array(result['time']) * 1000, y=energy,
            mode='lines', name='Stored Energy',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title="Energy Stored in Inductor",
            xaxis_title="Time (ms)",
            yaxis_title="Energy (J)",
            height=300
        )

        st.plotly_chart(fig, width='stretch')

def render_optics_lenses():
    """Render optical lenses simulation"""
    st.subheader("Optical Lenses Ray Tracing")
    st.markdown("*Simulate light ray behavior through different lens types*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        lens_type = st.selectbox("Lens Type:", ['convex', 'concave', 'planoconvex', 'planoconcave'])
        focal_length = st.slider("Focal Length (cm):", 1.0, 50.0, 10.0, 1.0)

    with col2:
        object_distance = st.slider("Object Distance (cm):", 1.0, 100.0, 20.0, 1.0)
        object_height = st.slider("Object Height (cm):", 0.5, 10.0, 5.0, 0.5)

    with col3:
        lens_diameter = st.slider("Lens Diameter (cm):", 1.0, 20.0, 5.0, 0.5)
        num_rays = st.slider("Number of Rays:", 3, 10, 5, 1)

    # Simulation
    parameters = {
        'lens_type': lens_type,
        'focal_length': focal_length,
        'object_distance': object_distance,
        'object_height': object_height,
        'lens_diameter': lens_diameter,
        'num_rays': num_rays
    }

    result = physics_simulator.run_simulation('optics_lenses', parameters)

    if result['success']:
        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            image_distance = result.get('image_distance', 0)
            if not math.isinf(image_distance):
                st.metric("Image Distance", f"{image_distance:.2f} cm")
            else:
                st.metric("Image Distance", "∞ (no image)")

        with col2:
            image_height = result.get('image_height', 0)
            if not math.isinf(image_height):
                st.metric("Image Height", f"{image_height:.2f} cm")
            else:
                st.metric("Image Height", "∞")

        with col3:
            magnification = result.get('magnification', 0)
            if not math.isinf(magnification):
                st.metric("Magnification", f"{magnification:.2f}x")
            else:
                st.metric("Magnification", "∞")

        # Image characteristics
        image_type = result.get('image_type', 'Unknown')
        st.info(f"**Image Type:** {image_type}")

        # Create ray diagram
        fig = create_optics_ray_diagram(result, parameters)
        st.plotly_chart(fig, width='stretch')

        # Lens equation analysis
        render_lens_equation_analysis(result, parameters)

    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def create_optics_ray_diagram(result, parameters):
    """Create optical ray diagram"""
    fig = go.Figure()

    focal_length = parameters['focal_length']
    object_distance = parameters['object_distance']
    object_height = parameters['object_height']
    lens_diameter = parameters['lens_diameter']

    # Lens type affects focal length sign
    if parameters['lens_type'] in ['concave', 'planoconcave']:
        f = -abs(focal_length)
    else:
        f = abs(focal_length)

    # Draw principal axis
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Principal Axis")

    # Draw lens
    lens_x = 0
    lens_top = lens_diameter / 2
    lens_bottom = -lens_diameter / 2

    fig.add_trace(go.Scatter(
        x=[lens_x, lens_x], y=[lens_bottom, lens_top],
        mode='lines', name='Lens',
        line=dict(color='black', width=4)
    ))

    # Draw object
    fig.add_trace(go.Scatter(
        x=[-object_distance, -object_distance], y=[0, object_height],
        mode='lines+markers', name='Object',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    # Draw image if it exists
    image_distance = result.get('image_distance', 0)
    image_height = result.get('image_height', 0)

    if not math.isinf(image_distance) and not math.isinf(image_height):
        # Determine image position sign for virtual images
        img_x = image_distance if result.get('image_type', '').startswith('Real') else -image_distance

        fig.add_trace(go.Scatter(
            x=[img_x, img_x], y=[0, image_height],
            mode='lines+markers', name='Image',
            line=dict(color='red', width=3, dash='dot'),
            marker=dict(size=8)
        ))

    # Draw ray paths
    ray_paths = result.get('ray_paths', [])
    for i, ray in enumerate(ray_paths):
        if 'start' in ray and 'end' in ray:
            start = ray['start']
            end = ray['end']

            fig.add_trace(go.Scatter(
                x=[start[0], lens_x, end[0]], 
                y=[start[1], ray.get('lens_entry', [0, start[1]])[1], end[1]],
                mode='lines', name=f'Ray {i+1}',
                line=dict(width=2)
            ))

    # Mark focal points
    if f > 0:  # Converging lens
        fig.add_trace(go.Scatter(
            x=[f, -f], y=[0, 0],
            mode='markers', name='Focal Points',
            marker=dict(symbol='x', size=10, color='green')
        ))
    else:  # Diverging lens
        fig.add_trace(go.Scatter(
            x=[abs(f), -abs(f)], y=[0, 0],
            mode='markers', name='Virtual Focal Points',
            marker=dict(symbol='x', size=10, color='orange')
        ))

    # Update layout
    fig.update_layout(
        title="Optical Ray Diagram",
        xaxis_title="Distance (cm)",
        yaxis_title="Height (cm)",
        showlegend=True,
        height=500,
        xaxis=dict(zeroline=True, zerolinewidth=2),
        yaxis=dict(zeroline=True, zerolinewidth=2)
    )

    return fig

def render_lens_equation_analysis(result, parameters):
    """Render lens equation analysis"""
    st.markdown("---")
    st.subheader("Lens Equation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Thin Lens Equation:**")
        st.latex(r"\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i}")
        st.latex(r"m = -\frac{d_i}{d_o} = \frac{h_i}{h_o}")

        st.markdown("**Given Values:**")
        f = parameters['focal_length']
        do = parameters['object_distance']
        ho = parameters['object_height']

        if parameters['lens_type'] in ['concave', 'planoconcave']:
            f = -f

        st.write(f"Focal length: {f:.2f} cm")
        st.write(f"Object distance: {do:.2f} cm")
        st.write(f"Object height: {ho:.2f} cm")

    with col2:
        st.markdown("**Calculated Results:**")

        image_distance = result.get('image_distance', 0)
        image_height = result.get('image_height', 0)
        magnification = result.get('magnification', 0)

        if not math.isinf(image_distance):
            st.write(f"Image distance: {image_distance:.2f} cm")
        else:
            st.write("Image distance: ∞ (no image formed)")

        if not math.isinf(image_height):
            st.write(f"Image height: {image_height:.2f} cm")
        else:
            st.write("Image height: ∞")

        if not math.isinf(magnification):
            st.write(f"Magnification: {magnification:.2f}x")
        else:
            st.write("Magnification: ∞")


# Add missing simulation render functions
def render_electromagnetic_wave():
    """Render electromagnetic wave simulation"""
    st.subheader("Electromagnetic Wave Propagation")
    st.markdown("*Simulate the propagation of electromagnetic waves*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        frequency = st.number_input("Frequency (Hz):", min_value=1e3, max_value=1e12, value=1e9, format="%.0e")
        amplitude = st.slider("Amplitude:", 0.1, 5.0, 1.0, 0.1)

    with col2:
        time_max = st.number_input("Simulation Time (s):", min_value=1e-12, max_value=1e-6, value=5e-9, format="%.0e")
        num_points = st.slider("Data Points:", 100, 1000, 500, 50)

    with col3:
        wavelength = 3e8 / frequency  # c = λf
        st.metric("Wavelength", f"{wavelength:.3e} m")
        st.metric("Period", f"{1/frequency:.3e} s")

    # Simulation
    parameters = {
        'frequency': frequency,
        'amplitude': amplitude,
        'time_max': time_max,
        'num_points': num_points
    }

    result = physics_simulator.run_simulation('electromagnetic_wave', parameters)

    if result['success']:
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Electric Field', 'Magnetic Field')
        )

        time = result['time']
        e_field = result.get('data', {}).get('electric_field', result.get('E_field', []))
        b_field = result.get('data', {}).get('magnetic_field', result.get('B_field', []))

        fig.add_trace(
            go.Scatter(x=time, y=e_field, mode='lines', name='E-field',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=time, y=b_field, mode='lines', name='B-field',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )

        fig.update_layout(
            title="Electromagnetic Wave Propagation",
            showlegend=True,
            height=400
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Electric Field (V/m)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Magnetic Field (T)", row=1, col=2)

        st.plotly_chart(fig, width='stretch')
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_doppler_effect():
    """Render Doppler effect simulation"""
    st.subheader("Doppler Effect")
    st.markdown("*Simulate frequency shifts due to relative motion*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        source_frequency = st.slider("Source Frequency (Hz):", 100.0, 1000.0, 440.0, 10.0)
        source_velocity = st.slider("Source Velocity (m/s):", -50.0, 50.0, 10.0, 1.0)

    with col2:
        observer_velocity = st.slider("Observer Velocity (m/s):", -50.0, 50.0, 0.0, 1.0)
        wave_speed = st.slider("Wave Speed (m/s):", 100.0, 500.0, 343.0, 1.0)

    # Simulation
    parameters = {
        'source_frequency': source_frequency,
        'source_velocity': source_velocity,
        'observer_velocity': observer_velocity,
        'wave_speed': wave_speed
    }

    result = physics_simulator.run_simulation('doppler_effect', parameters)

    if result['success']:
        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Original Frequency", f"{source_frequency:.1f} Hz")
        with col2:
            observed_freq = result.get('observed_frequency', 0)
            st.metric("Observed Frequency", f"{observed_freq:.1f} Hz")
        with col3:
            freq_shift = result.get('frequency_shift', 0)
            st.metric("Frequency Shift", f"{freq_shift:.1f} Hz")

        # Visualization
        fig = go.Figure()

        # Show frequency comparison
        frequencies = ['Source', 'Observed']
        values = [source_frequency, observed_freq]
        colors = ['blue', 'red']

        fig.add_trace(go.Bar(
            x=frequencies,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f} Hz" for v in values],
            textposition='auto'
        ))

        fig.update_layout(
            title="Doppler Effect - Frequency Comparison",
            yaxis_title="Frequency (Hz)",
            height=400
        )

        st.plotly_chart(fig, width='stretch')

        # Add explanation
        if source_velocity > 0 and observer_velocity == 0:
            st.info("Source moving towards observer - frequency increases")
        elif source_velocity < 0 and observer_velocity == 0:
            st.info("Source moving away from observer - frequency decreases")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_wave_interference():
    """Render wave interference simulation"""
    st.subheader("Wave Interference")
    st.markdown("*Simulate the interference pattern of two waves*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        freq1 = st.slider("Wave 1 Frequency (Hz):", 1.0, 20.0, 10.0, 0.5)
        amp1 = st.slider("Wave 1 Amplitude:", 0.1, 2.0, 1.0, 0.1)

    with col2:
        freq2 = st.slider("Wave 2 Frequency (Hz):", 1.0, 20.0, 12.0, 0.5)
        amp2 = st.slider("Wave 2 Amplitude:", 0.1, 2.0, 1.0, 0.1)

    with col3:
        phase_diff = st.slider("Phase Difference (rad):", -math.pi, math.pi, 0.0, 0.1)
        time_max = st.slider("Time Duration (s):", 0.5, 5.0, 2.0, 0.5)

    # Simulation
    parameters = {
        'frequency1': freq1,
        'frequency2': freq2,
        'amplitude1': amp1,
        'amplitude2': amp2,
        'phase_diff': phase_diff,
        'time_max': time_max,
        'num_points': 1000
    }

    result = physics_simulator.run_simulation('wave_interference', parameters)

    if result['success']:
        # Create visualization
        fig = go.Figure()

        time = result['time']
        data = result.get('data', {})
        wave1 = data.get('wave1', [])
        wave2 = data.get('wave2', [])
        interference = data.get('interference', [])

        fig.add_trace(go.Scatter(x=time, y=wave1, mode='lines', name='Wave 1',
                                line=dict(color='blue', width=2), opacity=0.7))
        fig.add_trace(go.Scatter(x=time, y=wave2, mode='lines', name='Wave 2',
                                line=dict(color='red', width=2), opacity=0.7))
        fig.add_trace(go.Scatter(x=time, y=interference, mode='lines', name='Interference',
                                line=dict(color='purple', width=3)))

        fig.update_layout(
            title="Wave Interference Pattern",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=500
        )

        st.plotly_chart(fig, width='stretch')

        # Beat frequency information
        beat_freq = abs(freq1 - freq2)
        st.info(f"Beat frequency: {beat_freq:.2f} Hz")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_heat_conduction():
    """Render heat conduction simulation"""
    st.subheader("Heat Conduction")
    st.markdown("*Simulate heat transfer through a material*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        length = st.slider("Rod Length (m):", 0.1, 2.0, 1.0, 0.1)
        thermal_diff = st.number_input("Thermal Diffusivity (m²/s):", 
                                      min_value=1e-6, max_value=1e-3, 
                                      value=1e-4, format="%.2e")

    with col2:
        initial_temp = st.slider("Initial Temperature (°C):", 0.0, 200.0, 100.0, 10.0)
        boundary_temp = st.slider("Boundary Temperature (°C):", 0.0, 100.0, 0.0, 10.0)

    time_max = st.slider("Simulation Time (s):", 100.0, 5000.0, 1000.0, 100.0)

    # Simulation
    parameters = {
        'length': length,
        'thermal_diffusivity': thermal_diff,
        'initial_temp': initial_temp,
        'boundary_temp': boundary_temp,
        'time_max': time_max,
        'num_points': 50
    }

    result = physics_simulator.run_simulation('heat_conduction', parameters)

    if result['success']:
        # Create heatmap visualization
        position = result['position']
        time_points = result['time']
        temperature = result['temperature']

        # Create 2D heatmap
        fig = go.Figure(data=go.Heatmap(
            x=position,
            y=time_points,
            z=temperature,
            colorscale='Hot',
            colorbar=dict(title="Temperature (°C)")
        ))

        fig.update_layout(
            title="Heat Conduction Over Time",
            xaxis_title="Position (m)",
            yaxis_title="Time (s)",
            height=500
        )

        st.plotly_chart(fig, width='stretch')

        # Show temperature profile at specific times
        st.subheader("Temperature Profiles")
        selected_time_idx = st.slider("Select Time Point:", 0, len(time_points)-1, len(time_points)//2)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=position,
            y=temperature[selected_time_idx],
            mode='lines',
            line=dict(color='red', width=3)
        ))

        fig2.update_layout(
            title=f"Temperature Profile at t = {time_points[selected_time_idx]:.1f} s",
            xaxis_title="Position (m)",
            yaxis_title="Temperature (°C)",
            height=300
        )

        st.plotly_chart(fig2, width='stretch')
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_orbital_motion():
    """Render orbital motion simulation"""
    st.subheader("Orbital Motion")
    st.markdown("*Simulate gravitational orbital mechanics*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        mass_central = st.number_input("Central Mass (kg):", 
                                      min_value=1e20, max_value=1e30, 
                                      value=5.972e24, format="%.3e")
        mass_orbiting = st.number_input("Orbiting Mass (kg):", 
                                       min_value=1.0, max_value=1e10, 
                                       value=1000.0, format="%.0f")

    with col2:
        altitude = st.slider("Altitude (km):", 100.0, 2000.0, 400.0, 50.0)
        initial_distance = 6.371e6 + altitude * 1000  # Earth radius + altitude

        # Calculate orbital velocity
        G = 6.674e-11
        orbital_velocity = math.sqrt(G * mass_central / initial_distance)
        st.metric("Orbital Velocity", f"{orbital_velocity:.0f} m/s")

    time_max = st.slider("Simulation Time (hours):", 0.5, 10.0, 1.5, 0.5) * 3600

    # Simulation
    parameters = {
        'mass_central': mass_central,
        'mass_orbiting': mass_orbiting,
        'initial_distance': initial_distance,
        'initial_velocity': orbital_velocity,
        'time_max': time_max,
        'num_points': 1000
    }

    result = physics_simulator.run_simulation('orbital_motion', parameters)

    if result['success']:
        # Create orbital visualization
        fig = go.Figure()

        # Get orbital data
        data = result.get('data', {})
        x_pos = data.get('x_position', result.get('x1', []))
        y_pos = data.get('y_position', result.get('y1', []))

        # Plot orbit
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='lines',
            name='Orbit',
            line=dict(color='blue', width=2)
        ))

        # Plot central body
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            name='Central Body',
            marker=dict(size=20, color='orange')
        ))

        # Plot current position
        if x_pos and y_pos:
            fig.add_trace(go.Scatter(
                x=[x_pos[-1]], y=[y_pos[-1]],
                mode='markers',
                name='Satellite',
                marker=dict(size=10, color='red')
            ))

        fig.update_layout(
            title="Orbital Motion",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=500
        )

        st.plotly_chart(fig, width='stretch')

        # Calculate orbital period
        orbital_period = 2 * math.pi * math.sqrt(initial_distance**3 / (G * mass_central))
        st.info(f"Orbital Period: {orbital_period/3600:.2f} hours")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_electromagnetic_field():
    """Render electromagnetic field simulation"""
    st.subheader("Electromagnetic Field")
    st.markdown("*Visualize electric and magnetic fields from charges*")

    # Parameter inputs
    st.write("Configure charge distribution:")

    num_charges = st.slider("Number of Charges:", 1, 5, 2)

    charges = []
    cols = st.columns(num_charges)

    for i, col in enumerate(cols):
        with col:
            st.write(f"Charge {i+1}")
            charge = st.number_input(f"Charge (μC):", -10.0, 10.0, 
                                   5.0 if i == 0 else -5.0, 0.1, 
                                   key=f"charge_{i}")
            x_pos = st.number_input(f"X position:", -5.0, 5.0, 
                                   -1.0 if i == 0 else 1.0, 0.1, 
                                   key=f"x_{i}")
            y_pos = st.number_input(f"Y position:", -5.0, 5.0, 0.0, 0.1, 
                                   key=f"y_{i}")
            charges.append({'charge': charge, 'x': x_pos, 'y': y_pos})

    # Grid parameters
    grid_size = st.slider("Grid Resolution:", 10, 30, 20)

    # Simulation
    parameters = {
        'charges': charges,
        'grid_size': grid_size,
        'x_range': (-5, 5),
        'y_range': (-5, 5)
    }

    result = physics_simulator.run_simulation('electromagnetic_field', parameters)

    if result['success']:
        # Create field visualization
        fig = go.Figure()

        # Add electric field streamlines
        field_data = result.get('field_lines', {})
        if field_data:
            for line in field_data.get('lines', []):
                fig.add_trace(go.Scatter(
                    x=line['x'], y=line['y'],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ))

        # Add charges
        for charge in charges:
            color = 'red' if charge['charge'] > 0 else 'blue'
            fig.add_trace(go.Scatter(
                x=[charge['x']], y=[charge['y']],
                mode='markers',
                marker=dict(size=abs(charge['charge'])*5+10, color=color),
                name=f"Q={charge['charge']}μC"
            ))

        fig.update_layout(
            title="Electric Field Lines",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=500
        )

        st.plotly_chart(fig, width='stretch')
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_photoelectric_effect():
    """Render photoelectric effect simulation"""
    st.subheader("Photoelectric Effect")
    st.markdown("*Simulate the quantum nature of light-matter interaction*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        frequency = st.number_input("Light Frequency (Hz):", 
                                   min_value=1e14, max_value=1e16, 
                                   value=1e15, format="%.2e")
        work_function = st.slider("Work Function (eV):", 1.0, 10.0, 4.5, 0.1)

    with col2:
        intensity = st.slider("Light Intensity (W/m²):", 100.0, 5000.0, 1000.0, 100.0)

        # Calculate photon energy
        h = 6.626e-34  # Planck constant
        e = 1.602e-19  # Elementary charge
        photon_energy_eV = h * frequency / e
        st.metric("Photon Energy", f"{photon_energy_eV:.2f} eV")

    # Simulation
    parameters = {
        'frequency': frequency,
        'work_function': work_function,
        'intensity': intensity
    }

    result = physics_simulator.run_simulation('photoelectric_effect', parameters)

    if result['success']:
        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            ke = result.get('kinetic_energy', 0)
            st.metric("Max Kinetic Energy", f"{ke:.2f} eV")

        with col2:
            current = result.get('photocurrent', 0)
            st.metric("Photocurrent", f"{current:.2e} A")

        with col3:
            threshold = result.get('threshold_frequency', work_function * e / h)
            st.metric("Threshold Frequency", f"{threshold:.2e} Hz")

        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Kinetic Energy vs Frequency', 'Current vs Intensity')
        )

        # Plot KE vs frequency
        frequencies = np.linspace(threshold, 2e16, 100)
        ke_values = [(h * f / e - work_function) for f in frequencies]
        ke_values = [max(0, ke) for ke in ke_values]

        fig.add_trace(
            go.Scatter(x=frequencies, y=ke_values, mode='lines', name='KE vs Frequency'),
            row=1, col=1
        )

        # Mark current point
        fig.add_trace(
            go.Scatter(x=[frequency], y=[max(0, ke)], 
                      mode='markers', marker=dict(size=10, color='red'),
                      name='Current'),
            row=1, col=1
        )

        # Plot current vs intensity
        intensities = np.linspace(100, 5000, 50)
        if photon_energy_eV > work_function:
            currents = intensities * 1e-10  # Proportional to intensity
        else:
            currents = np.zeros_like(intensities)

        fig.add_trace(
            go.Scatter(x=intensities, y=currents, mode='lines', name='Current vs Intensity'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="Kinetic Energy (eV)", row=1, col=1)
        fig.update_xaxes(title_text="Intensity (W/m²)", row=1, col=2)
        fig.update_yaxes(title_text="Current (A)", row=1, col=2)

        st.plotly_chart(fig, width='stretch')

        # Explanation
        if photon_energy_eV < work_function:
            st.warning("Photon energy is below work function - no photoelectrons emitted!")
        else:
            st.success("Photoelectric emission is occurring!")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_relativity_time_dilation():
    """Render time dilation simulation"""
    st.subheader("Relativistic Time Dilation")
    st.markdown("*Explore time dilation effects at high velocities*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        velocity_fraction = st.slider("Velocity (fraction of c):", 0.0, 0.99, 0.8, 0.01)
        velocity = velocity_fraction * 299792458  # Speed of light
        proper_time = st.slider("Proper Time (s):", 0.1, 10.0, 1.0, 0.1)

    with col2:
        # Calculate Lorentz factor
        gamma = 1 / math.sqrt(1 - velocity_fraction**2)
        st.metric("Lorentz Factor (γ)", f"{gamma:.3f}")
        st.metric("Velocity", f"{velocity_fraction:.2f}c")

    # Simulation
    parameters = {
        'velocity': velocity,
        'proper_time': proper_time
    }

    result = physics_simulator.run_simulation('relativity_time_dilation', parameters)

    if result['success']:
        # Display results
        dilated_time = result.get('dilated_time', proper_time * gamma)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Proper Time", f"{proper_time:.2f} s")
        with col2:
            st.metric("Dilated Time", f"{dilated_time:.2f} s")
        with col3:
            time_diff = dilated_time - proper_time
            st.metric("Time Difference", f"{time_diff:.2f} s")

        # Create visualization
        fig = go.Figure()

        # Plot time dilation vs velocity
        velocities = np.linspace(0, 0.99, 100)
        gammas = 1 / np.sqrt(1 - velocities**2)
        dilated_times = proper_time * gammas

        fig.add_trace(go.Scatter(
            x=velocities, y=dilated_times,
            mode='lines',
            name='Time Dilation',
            line=dict(color='blue', width=3)
        ))

        # Mark current point
        fig.add_trace(go.Scatter(
            x=[velocity_fraction], y=[dilated_time],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Current'
        ))

        fig.update_layout(
            title="Time Dilation vs Velocity",
            xaxis_title="Velocity (fraction of c)",
            yaxis_title="Dilated Time (s)",
            height=400
        )

        st.plotly_chart(fig, width='stretch')

        # Twin paradox example
        st.subheader("Twin Paradox Example")
        journey_years = st.slider("Journey Duration (Earth years):", 1.0, 50.0, 10.0, 1.0)
        earth_time = journey_years * 2  # Round trip
        traveler_time = earth_time / gamma

        st.info(f"If a twin travels at {velocity_fraction:.2f}c for {journey_years:.1f} years (round trip):\n"
                f"- Earth twin ages: {earth_time:.1f} years\n"
                f"- Traveling twin ages: {traveler_time:.1f} years\n"
                f"- Age difference: {earth_time - traveler_time:.1f} years")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_nuclear_decay():
    """Render nuclear decay simulation"""
    st.subheader("Nuclear Decay")
    st.markdown("*Simulate radioactive decay processes*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        half_life = st.number_input("Half-Life (s):", 
                                   min_value=0.1, max_value=1e6, 
                                   value=3600.0, format="%.1f")
        initial_nuclei = st.number_input("Initial Nuclei:", 
                                        min_value=100.0, max_value=1e6, 
                                        value=10000.0, format="%.0f")

    with col2:
        time_max = st.slider("Simulation Time (half-lives):", 
                            0.5, 10.0, 5.0, 0.5) * half_life

        # Calculate decay constant
        decay_constant = math.log(2) / half_life
        st.metric("Decay Constant", f"{decay_constant:.3e} /s")

    # Simulation
    parameters = {
        'half_life': half_life,
        'initial_nuclei': initial_nuclei,
        'time_max': time_max,
        'num_points': 200
    }

    result = physics_simulator.run_simulation('nuclear_decay', parameters)

    if result['success']:
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Nuclei Count vs Time', 'Decay Rate vs Time')
        )

        time = result['time']
        nuclei = result['nuclei_count']
        decay_rate = result['decay_rate']

        # Plot nuclei count
        fig.add_trace(
            go.Scatter(x=time, y=nuclei, mode='lines', name='Nuclei Count',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Add half-life markers
        for i in range(1, int(time_max/half_life) + 1):
            fig.add_vline(x=i*half_life, line_dash="dash", line_color="red",
                         annotation_text=f"{i} t½", row=1, col=1)

        # Plot decay rate
        fig.add_trace(
            go.Scatter(x=time, y=decay_rate, mode='lines', name='Decay Rate',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Nuclei", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Decay Rate (nuclei/s)", row=1, col=2)

        st.plotly_chart(fig, width='stretch')

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            remaining = nuclei[-1] if nuclei else 0
            st.metric("Remaining Nuclei", f"{int(remaining)}")

        with col2:
            decayed = initial_nuclei - remaining
            st.metric("Decayed Nuclei", f"{int(decayed)}")

        with col3:
            percentage = (remaining / initial_nuclei) * 100
            st.metric("Remaining %", f"{percentage:.1f}%")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_particle_accelerator():
    """Render particle accelerator simulation"""
    st.subheader("Particle Accelerator")
    st.markdown("*Simulate charged particle acceleration*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        particle_type = st.selectbox("Particle Type:", 
                                    ["Electron", "Proton", "Alpha"])

        # Set particle properties
        if particle_type == "Electron":
            mass = 9.11e-31
            charge = -1.6e-19
        elif particle_type == "Proton":
            mass = 1.67e-27
            charge = 1.6e-19
        else:  # Alpha
            mass = 6.64e-27
            charge = 3.2e-19

        voltage = st.number_input("Accelerating Voltage (V):", 
                                 min_value=1e3, max_value=1e9, 
                                 value=1e6, format="%.0e")

    with col2:
        magnetic_field = st.slider("Magnetic Field (T):", 0.01, 1.0, 0.1, 0.01)

        st.metric("Particle Mass", f"{mass:.2e} kg")
        st.metric("Particle Charge", f"{abs(charge):.2e} C")

    # Simulation
    parameters = {
        'particle_mass': mass,
        'charge': abs(charge),
        'voltage': voltage,
        'magnetic_field': magnetic_field
    }

    result = physics_simulator.run_simulation('particle_accelerator', parameters)

    if result['success']:
        # Display results
        velocity = result.get('velocity', 0)
        kinetic_energy = result.get('kinetic_energy', 0)
        cyclotron_freq = result.get('cyclotron_frequency', 0)
        radius = velocity / (2 * math.pi * cyclotron_freq) if cyclotron_freq > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Final Velocity", f"{velocity:.2e} m/s")
        with col2:
            st.metric("Kinetic Energy", f"{kinetic_energy:.2e} J")
        with col3:
            st.metric("Cyclotron Frequency", f"{cyclotron_freq:.2e} Hz")
        with col4:
            st.metric("Orbital Radius", f"{radius:.3f} m")

        # Create circular path visualization
        if radius > 0:
            theta = np.linspace(0, 2*math.pi, 100)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            fig = go.Figure()

            # Plot circular path
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Particle Path',
                line=dict(color='blue', width=2)
            ))

            # Add particle position
            fig.add_trace(go.Scatter(
                x=[radius], y=[0],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Particle'
            ))

            # Add magnetic field indication
            fig.add_annotation(
                x=0, y=0,
                text="B (into page)",
                showarrow=False,
                font=dict(size=20)
            )

            fig.update_layout(
                title="Particle in Magnetic Field",
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=400
            )

            st.plotly_chart(fig, width='stretch')

        # Energy comparison
        rest_energy = mass * (3e8)**2
        st.info(f"Kinetic energy is {(kinetic_energy/rest_energy)*100:.3f}% of rest mass energy")
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_fluid_flow():
    """Render fluid flow simulation"""
    st.subheader("Fluid Flow")
    st.markdown("*Simulate fluid dynamics using Bernoulli's equation*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Point 1 (Inlet):**")
        velocity1 = st.slider("Velocity 1 (m/s):", 0.1, 20.0, 5.0, 0.1)
        pressure1 = st.slider("Pressure 1 (kPa):", 50.0, 500.0, 200.0, 10.0)
        height1 = st.slider("Height 1 (m):", 0.0, 10.0, 2.0, 0.1)
        diameter1 = st.slider("Pipe Diameter 1 (cm):", 1.0, 20.0, 10.0, 0.5)

    with col2:
        st.write("**Point 2 (Outlet):**")
        height2 = st.slider("Height 2 (m):", 0.0, 10.0, 1.0, 0.1)
        diameter2 = st.slider("Pipe Diameter 2 (cm):", 1.0, 20.0, 5.0, 0.5)
        density = st.slider("Fluid Density (kg/m³):", 500.0, 1500.0, 1000.0, 50.0)

    # Convert units
    pressure1_pa = pressure1 * 1000  # kPa to Pa
    diameter1_m = diameter1 / 100    # cm to m
    diameter2_m = diameter2 / 100    # cm to m

    # Simulation
    parameters = {
        'velocity1': velocity1,
        'pressure1': pressure1_pa,
        'height1': height1,
        'height2': height2,
        'density': density,
        'pipe_diameter1': diameter1_m,
        'pipe_diameter2': diameter2_m
    }

    result = physics_simulator.run_simulation('fluid_flow', parameters)

    if result['success']:
        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Velocity 2", f"{result.get('velocity2', 0):.2f} m/s")
        with col2:
            st.metric("Pressure 2", f"{result.get('pressure2', 0)/1000:.1f} kPa")
        with col3:
            st.metric("Flow Rate", f"{result.get('flow_rate', 0):.4f} m³/s")

        # Create flow visualization
        fig = go.Figure()

        # Pipe sections
        fig.add_shape(
            type="rect",
            x0=0, x1=5, y0=height1-diameter1_m/2, y1=height1+diameter1_m/2,
            fillcolor="lightblue", opacity=0.5,
            line=dict(color="blue", width=2)
        )

        fig.add_shape(
            type="rect",
            x0=5, x1=10, y0=height2-diameter2_m/2, y1=height2+diameter2_m/2,
            fillcolor="lightgreen", opacity=0.5,
            line=dict(color="green", width=2)
        )

        # Flow arrows
        arrow_positions = np.linspace(0.5, 9.5, 10)
        for pos in arrow_positions:
            if pos < 5:
                arrow_length = velocity1 * 0.2
                y_center = height1
            else:
                arrow_length = result.get('velocity2', velocity1) * 0.2
                y_center = height2

            fig.add_annotation(
                x=pos, y=y_center,
                ax=pos+arrow_length, ay=y_center,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
            )

        fig.update_layout(
            title="Fluid Flow Visualization",
            xaxis_title="Distance (m)",
            yaxis_title="Height (m)",
            height=400,
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, max(height1, height2)+1])
        )

        st.plotly_chart(fig, width='stretch')

        # Show Bernoulli's equation
        st.subheader("Bernoulli's Equation Analysis")
        st.latex(r"P_1 + \frac{1}{2}\rho v_1^2 + \rho g h_1 = P_2 + \frac{1}{2}\rho v_2^2 + \rho g h_2")

        # Energy terms
        kinetic1 = 0.5 * density * velocity1**2
        potential1 = density * 9.81 * height1
        total1 = pressure1_pa + kinetic1 + potential1

        velocity2 = result.get('velocity2', 0)
        kinetic2 = 0.5 * density * velocity2**2
        potential2 = density * 9.81 * height2
        total2 = result.get('pressure2', 0) + kinetic2 + potential2

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Point 1 Energy Terms:**")
            st.write(f"Pressure: {pressure1_pa:.0f} Pa")
            st.write(f"Kinetic: {kinetic1:.0f} Pa")
            st.write(f"Potential: {potential1:.0f} Pa")
            st.write(f"Total: {total1:.0f} Pa")

        with col2:
            st.write("**Point 2 Energy Terms:**")
            st.write(f"Pressure: {result.get('pressure2', 0):.0f} Pa")
            st.write(f"Kinetic: {kinetic2:.0f} Pa")
            st.write(f"Potential: {potential2:.0f} Pa")
            st.write(f"Total: {total2:.0f} Pa")

        # Reynolds number analysis
        reynolds = result.get('reynolds_number', 0)
        if reynolds > 0:
            st.subheader("Flow Characteristics")
            st.metric("Reynolds Number", f"{reynolds:.0f}")

            if reynolds < 2300:
                flow_type = "Laminar"
                color = "green"
            elif reynolds > 4000:
                flow_type = "Turbulent"
                color = "red"
            else:
                flow_type = "Transitional"
                color = "orange"

            st.markdown(f"Flow Type: <span style='color:{color}'>{flow_type}</span>", 
                       unsafe_allow_html=True)
    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_three_body_orbital():
    """Render three-body orbital simulation"""
    st.subheader("Three-Body Orbital Motion")
    st.markdown("*Simulate orbital dynamics with three gravitationally interacting bodies*")

    # Parameter inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Primary Body (Earth):**")
        mass_primary = st.number_input("Mass (kg):", min_value=1e20, max_value=1e30, 
                                      value=5.972e24, format="%.3e", key="m1_3body")
        
        st.write("**Secondary Body (Moon):**")
        mass_secondary = st.number_input("Mass (kg):", min_value=1e20, max_value=1e25, 
                                        value=7.342e22, format="%.3e", key="m2_3body")

    with col2:
        st.write("**Tertiary Body (Satellite):**")
        mass_tertiary = st.number_input("Mass (kg):", min_value=1.0, max_value=1e10, 
                                       value=1000.0, format="%.0f", key="m3_3body")
        
        primary_secondary_distance = st.number_input("Primary-Secondary Distance (km):", 
                                                    min_value=100000.0, max_value=500000.0, 
                                                    value=384400.0, format="%.0f", key="r12_3body") * 1000

    with col3:
        st.write("**Initial Conditions:**")
        initial_distance = st.number_input("Initial Tertiary Distance (km):", 
                                         min_value=200000.0, max_value=600000.0, 
                                         value=400000.0, format="%.0f", key="r13_3body") * 1000
        
        initial_velocity = st.number_input("Initial Tertiary Velocity (m/s):", 
                                         min_value=100.0, max_value=3000.0, 
                                         value=1000.0, format="%.0f", key="v3_3body")
        
        tertiary_angle = st.slider("Tertiary Angle (degrees):", 0.0, 360.0, 30.0, 1.0, key="angle3_3body")

    # Simulation parameters
    time_max = st.slider("Simulation Time (days):", 1.0, 60.0, 28.0, 1.0, key="time_3body") * 24 * 3600
    
    # Calculate orbital periods for reference
    G = 6.674e-11
    primary_period = 2 * math.pi * math.sqrt(primary_secondary_distance**3 / (G * mass_primary))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Primary-Secondary Orbital Period", f"{primary_period/(24*3600):.2f} days")
    with col2:
        approx_tertiary_period = 2 * math.pi * math.sqrt(initial_distance**3 / (G * mass_primary))
        st.metric("Approx. Tertiary Period", f"{approx_tertiary_period/(24*3600):.2f} days")

    # Simulation
    parameters = {
        'mass_primary': mass_primary,
        'mass_secondary': mass_secondary,
        'mass_tertiary': mass_tertiary,
        'primary_secondary_distance': primary_secondary_distance,
        'initial_tertiary_distance': initial_distance,
        'initial_tertiary_velocity': initial_velocity,
        'tertiary_angle': tertiary_angle,
        'time_max': time_max,
        'num_points': 2000
    }

    result = physics_simulator.run_simulation('three_body_orbital', parameters)

    if result['success']:
        # Create orbital visualization
        fig = go.Figure()

        # Get orbital data
        x1 = result['x1']  # Primary
        y1 = result['y1']
        x2 = result['x2']  # Secondary
        y2 = result['y2']
        x3 = result['x3']  # Tertiary
        y3 = result['y3']

        # Plot orbits
        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x1], y=[y/1000 for y in y1],
            mode='markers',
            name='Primary (Earth)',
            marker=dict(size=15, color='blue'),
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x2], y=[y/1000 for y in y2],
            mode='lines',
            name='Secondary (Moon)',
            line=dict(color='gray', width=2),
            opacity=0.7
        ))

        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x3], y=[y/1000 for y in y3],
            mode='lines',
            name='Tertiary (Satellite)',
            line=dict(color='red', width=2)
        ))

        # Add current positions
        fig.add_trace(go.Scatter(
            x=[x2[-1]/1000], y=[y2[-1]/1000],
            mode='markers',
            name='Moon Position',
            marker=dict(size=10, color='gray'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[x3[-1]/1000], y=[y3[-1]/1000],
            mode='markers',
            name='Satellite Position',
            marker=dict(size=8, color='red'),
            showlegend=False
        ))

        fig.update_layout(
            title="Three-Body Orbital Motion",
            xaxis_title="X Position (km)",
            yaxis_title="Y Position (km)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=600,
            showlegend=True
        )

        st.plotly_chart(fig, width='stretch')

        # Display orbital analysis
        st.subheader("Orbital Analysis")
        
        # Calculate distances over time
        distances_13 = [math.sqrt(x**2 + y**2) for x, y in zip(x3, y3)]
        distances_23 = [math.sqrt((x3[i]-x2[i])**2 + (y3[i]-y2[i])**2) for i in range(len(x3))]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Distance to Primary", f"{min(distances_13)/1000:.0f} km")
        with col2:
            st.metric("Max Distance to Primary", f"{max(distances_13)/1000:.0f} km")
        with col3:
            st.metric("Min Distance to Secondary", f"{min(distances_23)/1000:.0f} km")

        # Distance plots
        fig_dist = go.Figure()
        
        time_days = [t/(24*3600) for t in result['time']]
        
        fig_dist.add_trace(go.Scatter(
            x=time_days, y=[d/1000 for d in distances_13],
            mode='lines', name='Distance to Primary',
            line=dict(color='blue', width=2)
        ))
        
        fig_dist.add_trace(go.Scatter(
            x=time_days, y=[d/1000 for d in distances_23],
            mode='lines', name='Distance to Secondary',
            line=dict(color='gray', width=2)
        ))

        fig_dist.update_layout(
            title="Distance Variations Over Time",
            xaxis_title="Time (days)",
            yaxis_title="Distance (km)",
            height=400
        )

        st.plotly_chart(fig_dist, width='stretch')

        # Stability analysis
        initial_energy = (G * mass_primary * mass_tertiary / initial_distance + 
                         0.5 * mass_tertiary * initial_velocity**2)
        
        st.info(f"This simulation shows the complex dynamics of three-body orbital motion. "
                f"The system exhibits chaotic behavior with no closed-form analytical solution.")

    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")

def render_four_body_orbital():
    """Render four-body orbital simulation"""
    st.subheader("Four-Body Orbital Motion")
    st.markdown("*Simulate orbital dynamics with four gravitationally interacting bodies*")

    # Parameter inputs
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Primary Body (Earth):**")
        mass_primary = st.number_input("Mass (kg):", min_value=1e20, max_value=1e30, 
                                      value=5.972e24, format="%.3e", key="m1_4body")
        
        st.write("**Secondary Body (Moon):**")
        mass_secondary = st.number_input("Mass (kg):", min_value=1e20, max_value=1e25, 
                                        value=7.342e22, format="%.3e", key="m2_4body")
        
        primary_secondary_distance = st.number_input("Primary-Secondary Distance (km):", 
                                                    min_value=100000.0, max_value=500000.0, 
                                                    value=384400.0, format="%.0f", key="r12_4body") * 1000

    with col2:
        st.write("**Tertiary Body (Satellite):**")
        mass_tertiary = st.number_input("Mass (kg):", min_value=1.0, max_value=1e10, 
                                       value=1000.0, format="%.0f", key="m3_4body")
        
        st.write("**Quaternary Body (Subsatellite):**")
        mass_quaternary = st.number_input("Mass (kg):", min_value=1.0, max_value=1e6, 
                                         value=100.0, format="%.0f", key="m4_4body")

    # Initial conditions
    st.write("**Initial Conditions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("*Tertiary Body:*")
        initial_tertiary_distance = st.number_input("Distance from Primary (km):", 
                                                   min_value=200000.0, max_value=600000.0, 
                                                   value=400000.0, format="%.0f", key="r13_4body") * 1000
        
        initial_tertiary_velocity = st.number_input("Velocity (m/s):", 
                                                   min_value=100.0, max_value=3000.0, 
                                                   value=1000.0, format="%.0f", key="v3_4body")
        
        tertiary_angle = st.slider("Angle (degrees):", 0.0, 360.0, 30.0, 1.0, key="angle3_4body")

    with col2:
        st.write("*Quaternary Body:*")
        initial_quaternary_distance = st.number_input("Distance from Tertiary (km):", 
                                                     min_value=1.0, max_value=10000.0, 
                                                     value=1000.0, format="%.0f", key="r34_4body")
        
        initial_quaternary_velocity = st.number_input("Velocity relative to Tertiary (m/s):", 
                                                     min_value=10.0, max_value=500.0, 
                                                     value=100.0, format="%.0f", key="v4_4body")
        
        quaternary_angle = st.slider("Angle relative to Tertiary (degrees):", 0.0, 360.0, 45.0, 1.0, key="angle4_4body")

    # Simulation parameters
    time_max = st.slider("Simulation Time (days):", 1.0, 30.0, 7.0, 1.0, key="time_4body") * 24 * 3600
    
    # Display system information
    G = 6.674e-11
    primary_period = 2 * math.pi * math.sqrt(primary_secondary_distance**3 / (G * mass_primary))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Primary-Secondary Period", f"{primary_period/(24*3600):.2f} days")
    with col2:
        hill_sphere = primary_secondary_distance * (mass_secondary / (3 * mass_primary))**(1/3)
        st.metric("Hill Sphere Radius", f"{hill_sphere/1000:.0f} km")

    # Simulation
    parameters = {
        'mass_primary': mass_primary,
        'mass_secondary': mass_secondary,
        'mass_tertiary': mass_tertiary,
        'mass_quaternary': mass_quaternary,
        'primary_secondary_distance': primary_secondary_distance,
        'initial_tertiary_distance': initial_tertiary_distance,
        'initial_tertiary_velocity': initial_tertiary_velocity,
        'tertiary_angle': tertiary_angle,
        'initial_quaternary_distance': initial_quaternary_distance,
        'initial_quaternary_velocity': initial_quaternary_velocity,
        'quaternary_angle': quaternary_angle,
        'time_max': time_max,
        'num_points': 1500
    }

    result = physics_simulator.run_simulation('four_body_orbital', parameters)

    if result['success']:
        # Create orbital visualization
        fig = go.Figure()

        # Get orbital data
        x1 = result['x1']  # Primary
        y1 = result['y1']
        x2 = result['x2']  # Secondary
        y2 = result['y2']
        x3 = result['x3']  # Tertiary
        y3 = result['y3']
        x4 = result['x4']  # Quaternary
        y4 = result['y4']

        # Plot orbits
        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x1], y=[y/1000 for y in y1],
            mode='markers',
            name='Primary (Earth)',
            marker=dict(size=15, color='blue'),
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x2], y=[y/1000 for y in y2],
            mode='lines',
            name='Secondary (Moon)',
            line=dict(color='gray', width=2),
            opacity=0.7
        ))

        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x3], y=[y/1000 for y in y3],
            mode='lines',
            name='Tertiary (Satellite)',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[x/1000 for x in x4], y=[y/1000 for y in y4],
            mode='lines',
            name='Quaternary (Subsatellite)',
            line=dict(color='green', width=2)
        ))

        # Add current positions
        fig.add_trace(go.Scatter(
            x=[x2[-1]/1000], y=[y2[-1]/1000],
            mode='markers',
            name='Moon Position',
            marker=dict(size=10, color='gray'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[x3[-1]/1000], y=[y3[-1]/1000],
            mode='markers',
            name='Satellite Position',
            marker=dict(size=8, color='red'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[x4[-1]/1000], y=[y4[-1]/1000],
            mode='markers',
            name='Subsatellite Position',
            marker=dict(size=6, color='green'),
            showlegend=False
        ))

        fig.update_layout(
            title="Four-Body Orbital Motion",
            xaxis_title="X Position (km)",
            yaxis_title="Y Position (km)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=600,
            showlegend=True
        )

        st.plotly_chart(fig, width='stretch')

        # Display orbital analysis
        st.subheader("Orbital Analysis")
        
        # Calculate distances over time
        distances_34 = [math.sqrt((x4[i]-x3[i])**2 + (y4[i]-y3[i])**2) for i in range(len(x4))]
        distances_14 = [math.sqrt(x4[i]**2 + y4[i]**2) for i in range(len(x4))]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Tertiary-Quaternary Distance", f"{min(distances_34)/1000:.2f} km")
        with col2:
            st.metric("Max Tertiary-Quaternary Distance", f"{max(distances_34)/1000:.2f} km")
        with col3:
            st.metric("Final Primary-Quaternary Distance", f"{distances_14[-1]/1000:.0f} km")

        # Distance plots
        fig_dist = go.Figure()
        
        time_days = [t/(24*3600) for t in result['time']]
        
        fig_dist.add_trace(go.Scatter(
            x=time_days, y=[d/1000 for d in distances_34],
            mode='lines', name='Tertiary-Quaternary Distance',
            line=dict(color='purple', width=2)
        ))
        
        fig_dist.add_trace(go.Scatter(
            x=time_days, y=[d/1000 for d in distances_14],
            mode='lines', name='Primary-Quaternary Distance',
            line=dict(color='orange', width=2)
        ))

        fig_dist.update_layout(
            title="Distance Variations Over Time",
            xaxis_title="Time (days)",
            yaxis_title="Distance (km)",
            height=400
        )

        st.plotly_chart(fig_dist, width='stretch')

        # System stability analysis
        st.subheader("System Stability")
        
        # Check if quaternary is still bound to tertiary
        initial_34_distance = initial_quaternary_distance
        current_34_distance = distances_34[-1]
        
        if current_34_distance > 2 * initial_34_distance:
            stability_status = "Quaternary body appears to be escaping from tertiary"
            stability_color = "orange"
        elif current_34_distance < initial_34_distance / 2:
            stability_status = "Quaternary body appears to be falling into tertiary"
            stability_color = "red"
        else:
            stability_status = "System appears stable"
            stability_color = "green"
        
        st.markdown(f"**System Status:** <span style='color:{stability_color}'>{stability_status}</span>", 
                   unsafe_allow_html=True)

        st.info(f"This simulation demonstrates the highly complex dynamics of four-body orbital motion. "
                f"The system exhibits chaotic behavior with extreme sensitivity to initial conditions.")

    else:
        st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")
