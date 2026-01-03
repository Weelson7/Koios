import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Callable
import math
from utils.exceptions import InvalidInputError

# Module-level constants for hardcoded values
NUMERICAL_TOLERANCE = 1e-10  # Tolerance for floating point comparisons
DAMPING_CRITICAL_THRESHOLD = 1e-10  # Threshold for critically damped system detection
DIVISION_BY_ZERO_TOLERANCE = 1e-10  # Prevent division by zero
PHOTOCURRENT_SCALE = 1e-10  # Photocurrent scaling factor

class PhysicsSimulator:
    """
    Extensible physics simulation framework
    """

    def __init__(self):
        self.simulations = {}
        self._register_default_simulations()

    def _register_default_simulations(self):
        """Register default physics simulations"""
        self.register_simulation('projectile_motion', self.projectile_motion)
        self.register_simulation('simple_harmonic_motion', self.simple_harmonic_motion)
        self.register_simulation('damped_oscillator', self.damped_oscillator)
        self.register_simulation('pendulum', self.pendulum_motion)
        self.register_simulation('circuit_rc', self.rc_circuit)
        self.register_simulation('circuit_rl', self.rl_circuit)
        self.register_simulation('circuit_rlc', self.rlc_circuit)
        self.register_simulation('electromagnetic_wave', self.electromagnetic_wave)
        self.register_simulation('doppler_effect', self.doppler_effect)
        self.register_simulation('wave_interference', self.wave_interference)
        self.register_simulation('heat_conduction', self.heat_conduction)
        self.register_simulation('orbital_motion', self.orbital_motion)
        self.register_simulation('three_body_orbital', self.three_body_orbital)
        self.register_simulation('four_body_orbital', self.four_body_orbital)
        self.register_simulation('fluid_flow', self.fluid_flow)
        self.register_simulation('electromagnetic_field', self.electromagnetic_field)
        self.register_simulation('photoelectric_effect', self.photoelectric_effect)
        self.register_simulation('relativity_time_dilation', self.relativity_time_dilation)
        self.register_simulation('nuclear_decay', self.nuclear_decay)
        self.register_simulation('particle_accelerator', self.particle_accelerator)
        self.register_simulation('optics_lenses', self.optics_lenses)

    def register_simulation(self, name: str, simulation_func: Callable):
        """Register a new physics simulation"""
        self.simulations[name] = simulation_func

    def get_available_simulations(self) -> List[str]:
        """Get list of available simulations"""
        return list(self.simulations.keys())

    def run_simulation(self, simulation_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific physics simulation"""
        if simulation_name not in self.simulations:
            return {
                'success': False,
                'error': f"Simulation '{simulation_name}' not found"
            }

        try:
            return self.simulations[simulation_name](parameters)
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def simulate(self, simulation_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for run_simulation for testing compatibility"""
        return self.run_simulation(simulation_name, parameters)

    def projectile_motion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate projectile motion

        Parameters:
            v0: Initial velocity (m/s)
            angle: Launch angle (degrees)
            g: Gravitational acceleration (m/s^2, default 9.81)
            time_max: Maximum simulation time (s)
            num_points: Number of simulation points
            
        Note:
            Output may contain fewer points than requested if trajectory ends
            (projectile returns to ground level) before time_max.
        """
        result = {
            'success': False,
            'time': None,
            'x_position': None,
            'y_position': None,
            'x_velocity': None,
            'y_velocity': None,
            'max_height': None,
            'range': None,
            'flight_time': None,
            'error': None
        }

        try:
            # Extract parameters
            v0 = parameters.get('v0', 20)
            angle_deg = parameters.get('angle', 45)
            g = parameters.get('g', 9.81)
            time_max = parameters.get('time_max', None)
            num_points = parameters.get('num_points', 100)

            # Validate parameters
            if v0 <= 0:
                raise InvalidInputError("velocity", str(v0), "positive value > 0")
            if not (-90 < angle_deg < 90):
                raise InvalidInputError("angle", str(angle_deg), "value between -90 and 90 degrees")
            if g <= 0:
                raise InvalidInputError("gravitational acceleration", str(g), "positive value")
            if num_points <= 0:
                raise InvalidInputError("num_points", str(num_points), "positive integer")
            if num_points > 10000:
                raise InvalidInputError("num_points", str(num_points), "value <= 10000 to avoid memory issues")
            if time_max is not None and time_max <= 0:
                raise InvalidInputError("time_max", str(time_max), "positive value")
            if math.isinf(v0) or math.isnan(v0):
                result['error'] = "Initial velocity cannot be infinite or NaN"
                return result
            if math.isinf(angle_deg) or math.isnan(angle_deg):
                result['error'] = "Angle cannot be infinite or NaN"
                return result
            if math.isinf(g) or math.isnan(g):
                result['error'] = "Gravitational acceleration cannot be infinite or NaN"
                return result

            # Convert angle to radians
            angle_rad = math.radians(angle_deg)
            v0x = v0 * math.cos(angle_rad)
            v0y = v0 * math.sin(angle_rad)

            # Calculate flight time if not provided
            flight_time = 2 * v0y / g
            if time_max is None:
                time_max = flight_time

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Position equations
            x_position = v0x * time
            y_position = v0y * time - 0.5 * g * time**2

            # Velocity equations
            x_velocity = np.full_like(time, v0x)
            y_velocity = v0y - g * time

            # Calculate key metrics
            max_height = (v0y**2) / (2 * g)
            range_max = (v0**2 * math.sin(2 * angle_rad)) / g

            # Remove negative y values (below ground)
            valid_indices = y_position >= 0
            time = time[valid_indices]
            x_position = x_position[valid_indices]
            y_position = y_position[valid_indices]
            x_velocity = x_velocity[valid_indices]
            y_velocity = y_velocity[valid_indices]
            
            # Store actual number of points generated (may differ from num_points)
            actual_points = len(time)

            result['success'] = True
            result['time'] = time.tolist()
            result['x_position'] = x_position.tolist()
            result['y_position'] = y_position.tolist()
            result['x_velocity'] = x_velocity.tolist()
            result['y_velocity'] = y_velocity.tolist()
            result['data'] = {
                'position': x_position.tolist(),
                'displacement': y_position.tolist(),
                'x_velocity': x_velocity.tolist(),
                'y_velocity': y_velocity.tolist()
            }
            result['max_height'] = max_height
            result['range'] = range_max
            result['flight_time'] = flight_time
            result['actual_points_generated'] = actual_points

        except Exception as e:
            result['error'] = str(e)

        return result

    def simple_harmonic_motion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate simple harmonic motion

        Parameters:
            amplitude: Amplitude of oscillation
            frequency: Frequency of oscillation (Hz)
            phase: Phase shift (radians)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'position': None,
            'velocity': None,
            'acceleration': None,
            'error': None
        }

        try:
            # Extract parameters
            amplitude = parameters.get('amplitude', 1.0)
            frequency = parameters.get('frequency', 1.0)
            phase = parameters.get('phase', 0.0)
            time_max = parameters.get('time_max', 4.0)
            num_points = parameters.get('num_points', 200)

            # Validate parameters
            if frequency <= 0:
                raise InvalidInputError("frequency", str(frequency), "positive value > 0")
            if time_max <= 0:
                raise InvalidInputError("time_max", str(time_max), "positive value")
            if num_points <= 0:
                raise InvalidInputError("num_points", str(num_points), "positive integer")
            if num_points > 10000:
                raise InvalidInputError("num_points", str(num_points), "value <= 10000 to avoid memory issues")
            if math.isinf(amplitude) or math.isnan(amplitude):
                result['error'] = "Amplitude cannot be infinite or NaN"
                return result
            if math.isinf(frequency) or math.isnan(frequency):
                result['error'] = "Frequency cannot be infinite or NaN"
                return result

            # Angular frequency
            omega = 2 * math.pi * frequency

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Simple harmonic motion equations
            position = amplitude * np.cos(omega * time + phase)
            velocity = -amplitude * omega * np.sin(omega * time + phase)
            acceleration = -amplitude * omega**2 * np.cos(omega * time + phase)

            result['success'] = True
            result['time'] = time.tolist()
            result['position'] = position.tolist()  # Changed from 'displacement' to 'position'
            result['velocity'] = velocity.tolist()
            result['acceleration'] = acceleration.tolist()
            result['data'] = {
                'position': position.tolist(),
                'velocity': velocity.tolist(),
                'acceleration': acceleration.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def damped_oscillator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate damped harmonic oscillator

        Parameters:
            amplitude: Initial amplitude
            omega0: Natural frequency
            gamma: Damping coefficient
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'position': None,
            'velocity': None,
            'envelope': None,
            'error': None
        }

        try:
            # Extract parameters
            amplitude = parameters.get('amplitude', 1.0)
            omega0 = parameters.get('omega0', 2.0)
            gamma = parameters.get('gamma', 0.1)
            time_max = parameters.get('time_max', 10.0)
            num_points = parameters.get('num_points', 500)

            # Validate parameters
            if omega0 <= 0:
                raise InvalidInputError("natural frequency (omega0)", str(omega0), "positive value > 0")
            if gamma < 0:
                raise InvalidInputError("damping coefficient (gamma)", str(gamma), "non-negative value >= 0")
            if time_max <= 0:
                raise InvalidInputError("time_max", str(time_max), "positive value")
            if num_points <= 0:
                raise InvalidInputError("num_points", str(num_points), "positive integer")
            if num_points > 10000:
                raise InvalidInputError("num_points", str(num_points), "value <= 10000 to avoid memory issues")
            if math.isinf(amplitude) or math.isnan(amplitude):
                result['error'] = "Amplitude cannot be infinite or NaN"
                return result
            if math.isinf(omega0) or math.isnan(omega0):
                result['error'] = "Natural frequency cannot be infinite or NaN"
                return result
            if math.isinf(gamma) or math.isnan(gamma):
                result['error'] = "Damping coefficient cannot be infinite or NaN"
                return result

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Damped oscillation equations
            discriminant = omega0**2 - gamma**2

            if discriminant > 0:  # Underdamped
                omega_d = np.sqrt(discriminant)
                envelope = amplitude * np.exp(-gamma * time)
                position = envelope * np.cos(omega_d * time)
                velocity = -envelope * (gamma * np.cos(omega_d * time) + omega_d * np.sin(omega_d * time))
            elif abs(discriminant) < DAMPING_CRITICAL_THRESHOLD:  # Critically damped
                envelope = amplitude * np.exp(-gamma * time)
                position = amplitude * (1 + gamma * time) * np.exp(-gamma * time)
                velocity = -amplitude * gamma**2 * time * np.exp(-gamma * time)
            else:  # Overdamped (discriminant < 0)
                lambda1 = -gamma + np.sqrt(-discriminant)
                lambda2 = -gamma - np.sqrt(-discriminant)
                envelope = amplitude * np.exp(-gamma * time)
                position = amplitude * (np.exp(lambda1 * time) + np.exp(lambda2 * time)) / 2
                velocity = amplitude * (lambda1 * np.exp(lambda1 * time) + lambda2 * np.exp(lambda2 * time)) / 2

            result['success'] = True
            result['time'] = time.tolist()
            result['position'] = position.tolist()  # Fixed: changed from displacement to position
            result['velocity'] = velocity.tolist()
            result['envelope'] = envelope.tolist()
            result['data'] = {
                'position': position.tolist(),
                'velocity': velocity.tolist(),
                'envelope': envelope.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def pendulum_motion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate simple pendulum motion (small angle approximation)

        Parameters:
            length: Pendulum length (m)
            angle0: Initial angle (degrees)
            g: Gravitational acceleration (m/s^2)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'angle': None,
            'angular_velocity': None,
            'x_position': None,
            'y_position': None,
            'error': None
        }

        try:
            # Extract parameters
            length = parameters.get('length', 1.0)
            angle0_deg = parameters.get('angle0', 10.0)
            g = parameters.get('g', 9.81)
            time_max = parameters.get('time_max', 4.0)
            num_points = parameters.get('num_points', 200)

            # Convert to radians
            angle0 = math.radians(angle0_deg)

            # Add warning for large angle approximation
            if abs(angle0_deg) > 15:
                result['warning'] = "Large angle approximation may be inaccurate. Consider using numerical integration for angles > 15 degrees."

            # Pendulum frequency
            omega = math.sqrt(g / length)

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Simple pendulum equations (small angle approximation)
            angle = angle0 * np.cos(omega * time)
            angular_velocity = -angle0 * omega * np.sin(omega * time)

            # Cartesian coordinates
            x_position = length * np.sin(angle)
            y_position = -length * np.cos(angle)

            result['success'] = True
            result['time'] = time.tolist()
            result['angle'] = np.degrees(angle).tolist()
            result['angular_velocity'] = angular_velocity.tolist()
            result['x_position'] = x_position.tolist()
            result['y_position'] = y_position.tolist()
            result['data'] = {
                'angle': np.degrees(angle).tolist(),  # Convert back to degrees
                'angular_velocity': angular_velocity.tolist(),
                'x_position': x_position.tolist(),
                'y_position': y_position.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def rc_circuit(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate RC circuit charging/discharging

        Parameters:
            R: Resistance (Ohms)
            C: Capacitance (Farads)
            V0: Initial voltage (V)
            Vs: Source voltage (V)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'voltage': None,
            'current': None,
            'error': None
        }

        try:
            # Extract parameters
            R = parameters.get('R', 1000.0)  # 1k Ohm
            C = parameters.get('C', 1e-6)    # 1 microF
            V0 = parameters.get('V0', 0.0)   # Initial voltage
            Vs = parameters.get('Vs', 5.0)   # Source voltage
            time_max = parameters.get('time_max', None)
            num_points = parameters.get('num_points', 200)

            # Validate parameters
            if R <= 0:
                result['error'] = "Resistance must be positive"
                return result
            if C <= 0:
                result['error'] = "Capacitance must be positive"
                return result
            if math.isinf(R) or math.isnan(R):
                result['error'] = "Resistance cannot be infinite or NaN"
                return result
            if math.isinf(C) or math.isnan(C):
                result['error'] = "Capacitance cannot be infinite or NaN"
                return result

            # Time constant
            tau = R * C
            if time_max is None:
                time_max = 5 * tau  # Default to 5 time constants

            # Time array
            time = np.linspace(0, time_max, num_points)

            # RC circuit equations
            if Vs != V0:  # Charging or discharging
                voltage = Vs + (V0 - Vs) * np.exp(-time / tau)
                current = (Vs - V0) / R * np.exp(-time / tau)
            else:  # No change
                voltage = np.full_like(time, V0)
                current = np.zeros_like(time)

            result['success'] = True
            result['time'] = time.tolist()
            result['voltage'] = voltage.tolist()
            result['current'] = current.tolist()
            result['data'] = {
                'voltage': voltage.tolist(),
                'current': current.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def rl_circuit(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate RL circuit

        Parameters:
            R: Resistance (Ohms)
            L: Inductance (Henries)
            V0: Source voltage (V)
            I0: Initial current (A)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'voltage': None,
            'current': None,
            'error': None
        }

        try:
            # Extract parameters
            R = parameters.get('R', 100.0)    # 100 Ohm
            L = parameters.get('L', 1e-3)     # 1 mH
            V0 = parameters.get('V0', 5.0)    # Source voltage
            I0 = parameters.get('I0', 0.0)    # Initial current
            time_max = parameters.get('time_max', 5 * L / R)  # 5 time constants
            num_points = parameters.get('num_points', 200)

            # Time constant
            tau = L / R

            # Time array
            time = np.linspace(0, time_max, num_points)

            # RL circuit equations
            # For step response: i(t) = (V/R) * (1 - e^(-t/τ)) + I0 * e^(-t/τ)
            steady_current = V0 / R
            current = steady_current * (1 - np.exp(-time / tau)) + I0 * np.exp(-time / tau)

            # Voltage across inductor: V_L = V0 - I*R
            voltage_inductor = V0 - current * R

            # Voltage across resistor
            voltage_resistor = current * R

            result['success'] = True
            result['time'] = time.tolist()
            result['current'] = current.tolist()
            result['voltage_inductor'] = voltage_inductor.tolist()
            result['voltage_resistor'] = voltage_resistor.tolist()
            result['voltage'] = voltage_inductor.tolist()  # For compatibility
            result['data'] = {
                'current': current.tolist(),
                'voltage_inductor': voltage_inductor.tolist(),
                'voltage_resistor': voltage_resistor.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def rlc_circuit(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate RLC circuit

        Parameters:
            R: Resistance (Ohms)
            L: Inductance (Henries)
            C: Capacitance (Farads)
            V0: Initial voltage (V)
            I0: Initial current (A)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'voltage': None,
            'current': None,
            'error': None
        }

        try:
            # Extract parameters
            R = parameters.get('R', 100.0)
            L = parameters.get('L', 1e-3)    # 1 mH
            C = parameters.get('C', 1e-6)    # 1 microF
            V0 = parameters.get('V0', 5.0)
            I0 = parameters.get('I0', 0.0)
            time_max = parameters.get('time_max', 1e-3)  # 1 ms
            num_points = parameters.get('num_points', 500)

            # Circuit parameters
            omega0 = 1.0 / math.sqrt(L * C)  # Natural frequency
            gamma = R / (2 * L)              # Damping coefficient

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Determine circuit behavior
            discriminant = gamma**2 - omega0**2

            if discriminant < 0:  # Underdamped
                omega_d = math.sqrt(omega0**2 - gamma**2)
                exp_term = np.exp(-gamma * time)

                # Constants determined by initial conditions
                A = V0
                B = (I0 / C + gamma * V0) / omega_d

                voltage = exp_term * (A * np.cos(omega_d * time) + B * np.sin(omega_d * time))
                current = -C * exp_term * (
                    (B * omega_d - A * gamma) * np.cos(omega_d * time) - 
                    (A * omega_d + B * gamma) * np.sin(omega_d * time)
                )

            elif discriminant > 1e-20:  # Overdamped (with tolerance)
                r1 = -gamma + math.sqrt(discriminant)
                r2 = -gamma - math.sqrt(discriminant)

                # Check for division by zero (near critically damped)
                if abs(r1 - r2) < 1e-6 * abs(gamma):
                    # Fall back to critically damped if roots are too close
                    A = V0
                    B = I0 / C + gamma * V0
                    voltage = (A + B * time) * np.exp(-gamma * time)
                    current = -C * (B - gamma * (A + B * time)) * np.exp(-gamma * time)
                else:
                    # Constants determined by initial conditions
                    A = (I0 / C - r2 * V0) / (r1 - r2)
                    B = V0 - A

                    voltage = A * np.exp(r1 * time) + B * np.exp(r2 * time)
                    current = -C * (A * r1 * np.exp(r1 * time) + B * r2 * np.exp(r2 * time))

            else:  # Critically damped
                A = V0
                B = I0 / C + gamma * V0

                voltage = (A + B * time) * np.exp(-gamma * time)
                current = -C * (B - gamma * (A + B * time)) * np.exp(-gamma * time)

            result['success'] = True
            result['time'] = time.tolist()
            result['voltage'] = voltage.tolist()
            result['current'] = current.tolist()
            result['data'] = {
                'voltage': voltage.tolist(),
                'current': current.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def electromagnetic_wave(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate electromagnetic wave propagation

        Parameters:
            frequency: Wave frequency (Hz)
            wavelength: Wavelength (m)
            amplitude: Wave amplitude
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'electric_field': None,
            'magnetic_field': None,
            'error': None
        }

        try:
            # Extract parameters
            frequency = parameters.get('frequency', 1e9)  # 1 GHz
            wavelength = parameters.get('wavelength', 3e8 / frequency)  # c = λf
            amplitude = parameters.get('amplitude', 1.0)
            time_max = parameters.get('time_max', 5 / frequency)
            num_points = parameters.get('num_points', 500)

            # Constants
            c = 3e8  # Speed of light
            omega = 2 * math.pi * frequency
            k = 2 * math.pi / wavelength

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Electromagnetic wave equations
            electric_field = amplitude * np.cos(omega * time)
            magnetic_field = (amplitude / c) * np.cos(omega * time)

            result['success'] = True
            result['time'] = time.tolist()
            result['E_field'] = electric_field.tolist()
            result['B_field'] = magnetic_field.tolist()
            result['data'] = {
                'electric_field': electric_field.tolist(),
                'magnetic_field': magnetic_field.tolist()
            }
            result['frequency'] = frequency
            result['wavelength'] = wavelength

        except Exception as e:
            result['error'] = str(e)

        return result

    def doppler_effect(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate Doppler effect

        Parameters:
            source_frequency: Original frequency (Hz)
            source_velocity: Source velocity (m/s)
            observer_velocity: Observer velocity (m/s)
            sound_speed: Speed of sound (m/s, default 343)
        """
        result = {
            'success': False,
            'observed_frequency': None,
            'frequency_shift': None,
            'error': None
        }

        try:
            # Extract parameters
            f0 = parameters.get('source_frequency', 440.0)  # A4 note
            vs = parameters.get('source_velocity', 0.0)
            vo = parameters.get('observer_velocity', 0.0)
            v = parameters.get('sound_speed', 343.0)  # Speed of sound in air

            # Check for division by zero
            if abs(v - vs) < NUMERICAL_TOLERANCE:
                raise InvalidInputError("source_velocity", str(parameters.get('source_velocity')), "less than sound speed")

            # Doppler effect equation
            # Positive vo: observer moving toward source (higher frequency)
            # Positive vs: source moving toward observer (higher frequency)
            observed_frequency = f0 * (v + vo) / (v - vs)
            frequency_shift = observed_frequency - f0

            result['success'] = True
            result['observed_frequency'] = observed_frequency
            result['frequency_shift'] = frequency_shift
            result['original_frequency'] = f0

        except Exception as e:
            result['error'] = str(e)

        return result

    def wave_interference(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate wave interference between two sources

        Parameters:
            frequency1: Frequency of first wave (Hz)
            frequency2: Frequency of second wave (Hz)
            amplitude1: Amplitude of first wave
            amplitude2: Amplitude of second wave
            phase_diff: Phase difference (radians)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'wave1': None,
            'wave2': None,
            'interference': None,
            'error': None
        }

        try:
            # Extract parameters
            f1 = parameters.get('frequency1', 10.0)
            f2 = parameters.get('frequency2', 12.0)
            A1 = parameters.get('amplitude1', 1.0)
            A2 = parameters.get('amplitude2', 1.0)
            phase_diff = parameters.get('phase_diff', 0.0)
            time_max = parameters.get('time_max', 2.0)
            num_points = parameters.get('num_points', 1000)

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Wave equations
            omega1 = 2 * math.pi * f1
            omega2 = 2 * math.pi * f2

            wave1 = A1 * np.cos(omega1 * time)
            wave2 = A2 * np.cos(omega2 * time + phase_diff)
            interference = wave1 + wave2

            result['success'] = True
            result['time'] = time.tolist()
            result['data'] = {
                'wave1': wave1.tolist(),
                'wave2': wave2.tolist(),
                'interference': interference.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def heat_conduction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate 1D heat conduction using finite difference method

        Parameters:
            length: Rod length (m)
            thermal_diffusivity: Thermal diffusivity (m²/s)
            initial_temp: Initial temperature (°C)
            boundary_temp: Boundary temperature (°C)
            time_max: Maximum simulation time
            num_points: Number of spatial points
        """
        result = {
            'success': False,
            'time': None,
            'position': None,
            'temperature': None,
            'error': None
        }

        try:
            # Extract parameters
            L = parameters.get('length', 1.0)
            alpha = parameters.get('thermal_diffusivity', 1e-4)
            T_initial = parameters.get('initial_temp', 100.0)
            T_boundary = parameters.get('boundary_temp', 0.0)
            time_max = parameters.get('time_max', 1000.0)
            nx = parameters.get('num_points', 50)

            # Validate grid parameters
            if nx < 2:
                raise InvalidInputError("nx", str(nx), "at least 2 (recommend 10+)")
            if nx < 10:
                import warnings
                warnings.warn("Low grid resolution. Recommend at least 10 points for meaningful results.")

            # Spatial grid
            dx = L / (nx - 1)
            x = np.linspace(0, L, nx)

            # Time step (stability condition: dt <= 0.5*dx²/α for 1D, 0.25 for 2D)
            dt = 0.25 * dx**2 / alpha  # Safe for both 1D and 2D
            nt = int(time_max / dt)

            # Initialize temperature array
            T = np.full(nx, T_initial)
            T[0] = T_boundary  # Left boundary
            T[-1] = T_boundary  # Right boundary

            # Store temperature evolution
            temperature_history = []
            time_history = []

            # Time evolution
            for n in range(nt):
                T_new = T.copy()
                for i in range(1, nx-1):
                    T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
                T = T_new

                # Store data every 10 time steps
                if n % 10 == 0:
                    temperature_history.append(T.copy())
                    time_history.append(n * dt)

            result['success'] = True
            result['time'] = time_history
            result['position'] = x.tolist()
            result['temperature'] = [temp.tolist() for temp in temperature_history]

        except Exception as e:
            result['error'] = str(e)

        return result

    def orbital_motion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate orbital motion using gravitational force

        Parameters:
            mass_central: Central body mass (kg)
            mass_orbiting: Orbiting body mass (kg)
            initial_distance: Initial distance (m)
            initial_velocity: Initial velocity (m/s)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'x_position': None,
            'y_position': None,
            'velocity_x': None,
            'velocity_y': None,
            'error': None
        }

        try:
            # Extract parameters
            M = parameters.get('mass_central', 5.972e24)  # Earth mass
            m = parameters.get('mass_orbiting', 1000.0)   # Satellite mass
            r0 = parameters.get('initial_distance', 6.371e6 + 400e3)  # 400 km altitude
            v0 = parameters.get('initial_velocity', 7670.0)  # Orbital velocity
            time_max = parameters.get('time_max', 5400.0)  # 1.5 hours
            num_points = parameters.get('num_points', 1000)

            # Constants
            G = 6.674e-11  # Gravitational constant

            # Time array
            dt = time_max / num_points
            time = np.linspace(0, time_max, num_points)

            # Initialize arrays
            x = np.zeros(num_points)
            y = np.zeros(num_points)
            vx = np.zeros(num_points)
            vy = np.zeros(num_points)

            # Initial conditions
            x[0] = r0
            y[0] = 0.0
            vx[0] = 0.0
            vy[0] = v0

            # Numerical integration using RK4 method for better energy conservation
            def accel(x_pos, y_pos):
                """Calculate gravitational acceleration"""
                r = math.sqrt(x_pos**2 + y_pos**2)
                if r < DIVISION_BY_ZERO_TOLERANCE:
                    return 0, 0
                return -G * M * x_pos / r**3, -G * M * y_pos / r**3

            for i in range(1, num_points):
                # RK4 integration - properly coupling position and velocity
                # k1 values
                k1_vx = vx[i-1]
                k1_vy = vy[i-1]
                k1_ax, k1_ay = accel(x[i-1], y[i-1])
                
                # k2 values
                x2 = x[i-1] + k1_vx * dt/2
                y2 = y[i-1] + k1_vy * dt/2
                k2_vx = vx[i-1] + k1_ax * dt/2
                k2_vy = vy[i-1] + k1_ay * dt/2
                k2_ax, k2_ay = accel(x2, y2)
                
                # k3 values
                x3 = x[i-1] + k2_vx * dt/2
                y3 = y[i-1] + k2_vy * dt/2
                k3_vx = vx[i-1] + k2_ax * dt/2
                k3_vy = vy[i-1] + k2_ay * dt/2
                k3_ax, k3_ay = accel(x3, y3)
                
                # k4 values
                x4 = x[i-1] + k3_vx * dt
                y4 = y[i-1] + k3_vy * dt
                k4_vx = vx[i-1] + k3_ax * dt
                k4_vy = vy[i-1] + k3_ay * dt
                k4_ax, k4_ay = accel(x4, y4)
                
                # Update using weighted average
                vx[i] = vx[i-1] + (k1_ax + 2*k2_ax + 2*k3_ax + k4_ax) * dt / 6
                vy[i] = vy[i-1] + (k1_ay + 2*k2_ay + 2*k3_ay + k4_ay) * dt / 6
                x[i] = x[i-1] + (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) * dt / 6
                y[i] = y[i-1] + (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) * dt / 6

            result['success'] = True
            result['time'] = time.tolist()
            result['x1'] = x.tolist()
            result['y1'] = y.tolist()
            result['x2'] = [0] * len(time)  # Central body position
            result['y2'] = [0] * len(time)
            result['data'] = {
                'x_position': x.tolist(),
                'y_position': y.tolist(),
                'velocity_x': vx.tolist(),
                'velocity_y': vy.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def quantum_harmonic_oscillator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate quantum harmonic oscillator energy levels and wavefunctions

        Parameters:
            mass: Particle mass (kg)
            frequency: Oscillator frequency (Hz)
            quantum_number: Quantum number n (0, 1, 2, ...)
            position_range: Position range for wavefunction
            num_points: Number of position```
 points
        """
        result = {
            'success': False,
            'energy_levels': None,
            'position': None,
            'wavefunction': None,
            'probability': None,
            'error': None
        }

        try:
            # Extract parameters
            m = parameters.get('mass', 9.109e-31)  # Electron mass
            omega = 2 * math.pi * parameters.get('frequency', 1e14)  # Angular frequency
            n = parameters.get('quantum_number', 0)
            x_range = parameters.get('position_range', 10e-10)  # 10 nm
            num_points = parameters.get('num_points', 1000)

            # Constants
            hbar = 1.054571817e-34  # Reduced Planck constant

            # Energy levels
            energy_levels = [(n_level + 0.5) * hbar * omega for n_level in range(n + 5)]

            # Position array
            x = np.linspace(-x_range/2, x_range/2, num_points)

            # Characteristic length
            x0 = np.sqrt(hbar / (m * omega))

            # Normalized position
            xi = x / x0

            # Hermite polynomial (approximation for low n)
            if n == 0:
                H_n = np.ones_like(xi)
            elif n == 1:
                H_n = 2 * xi
            elif n == 2:
                H_n = 4 * xi**2 - 2
            elif n == 3:
                H_n = 8 * xi**3 - 12 * xi
            else:
                H_n = np.ones_like(xi)  # Fallback

            # Wavefunction
            normalization = (m * omega / (math.pi * hbar))**(1/4) / np.sqrt(2**n * math.factorial(n))
            wavefunction = normalization * H_n * np.exp(-xi**2 / 2)

            # Probability density
            probability = np.abs(wavefunction)**2

            result['success'] = True
            result['energy_levels'] = energy_levels
            result['position'] = x.tolist()
            result['data'] = {
                'wavefunction': wavefunction.tolist(),
                'probability': probability.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def blackbody_radiation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate blackbody radiation spectrum (Planck distribution)

        Parameters:
            temperature: Temperature (K)
            wavelength_min: Minimum wavelength (m)
            wavelength_max: Maximum wavelength (m)
            num_points: Number of wavelength points
        """
        result = {
            'success': False,
            'wavelength': None,
            'spectral_radiance': None,
            'peak_wavelength': None,
            'total_power': None,
            'error': None
        }

        try:
            # Extract parameters
            T = parameters.get('temperature', 5778)  # Sun's temperature
            lambda_min = parameters.get('wavelength_min', 100e-9)  # 100 nm
            lambda_max = parameters.get('wavelength_max', 3000e-9)  # 3000 nm
            num_points = parameters.get('num_points', 500)

            # Constants
            h = 6.62607015e-34  # Planck constant
            c = 299792458  # Speed of light
            k_B = 1.380649e-23  # Boltzmann constant

            # Wavelength array
            wavelength = np.linspace(lambda_min, lambda_max, num_points)

            # Planck distribution
            spectral_radiance = (2 * h * c**2 / wavelength**5) / (np.exp(h * c / (wavelength * k_B * T)) - 1)

            # Wien's displacement law - peak wavelength
            peak_wavelength = 2.897771955e-3 / T  # Wien's displacement constant / T

            # Stefan-Boltzmann law - total power
            sigma = 5.670374419e-8  # Stefan-Boltzmann constant
            total_power = sigma * T**4

            result['success'] = True
            result['wavelength'] = wavelength.tolist()
            result['data'] = {
                'spectral_radiance': spectral_radiance.tolist()
            }
            result['peak_wavelength'] = peak_wavelength
            result['total_power'] = total_power
            result['temperature'] = T

        except Exception as e:
            result['error'] = str(e)

        return result

    def fluid_flow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate fluid flow using Bernoulli's equation

        Parameters:
            velocity1: Velocity at point 1 (m/s)
            pressure1: Pressure at point 1 (Pa)
            height1: Height at point 1 (m)
            height2: Height at point 2 (m)
            density: Fluid density (kg/m³)
            pipe_diameter1: Pipe diameter at point 1 (m)
            pipe_diameter2: Pipe diameter at point 2 (m)
        """
        result = {
            'success': False,
            'velocity2': None,
            'pressure2': None,
            'flow_rate': None,
            'reynolds_number': None,
            'error': None
        }

        try:
            # Extract parameters
            v1 = parameters.get('velocity1', 2.0)
            p1 = parameters.get('pressure1', 101325)  # Atmospheric pressure
            h1 = parameters.get('height1', 0.0)
            h2 = parameters.get('height2', 1.0)
            rho = parameters.get('density', 1000)  # Water density
            d1 = parameters.get('pipe_diameter1', 0.1)
            d2 = parameters.get('pipe_diameter2', 0.05)

            # Constants
            g = 9.81  # Gravitational acceleration

            # Continuity equation: A1*v1 = A2*v2
            A1 = math.pi * (d1/2)**2
            A2 = math.pi * (d2/2)**2
            v2 = v1 * A1 / A2

            # Bernoulli's equation: p1 + 0.5*rho*v1^2 + rho*g*h1 = p2 + 0.5*rho*v2^2 + rho*g*h2
            p2 = p1 + 0.5 * rho * (v1**2 - v2**2) + rho * g * (h1 - h2)

            # Flow rate
            flow_rate = A1 * v1

            # Reynolds number (assuming dynamic viscosity of water)
            mu = 1e-3  # Dynamic viscosity of water (Pa·s)
            Re = rho * v1 * d1 / mu

            result['success'] = True
            result['velocity2'] = v2
            result['pressure2'] = p2
            result['flow_rate'] = flow_rate
            result['reynolds_number'] = Re
            result['data'] = {
                'velocity1': v1,
                'velocity2': v2,
                'pressure1': p1,
                'pressure2': p2,
                'flow_rate': flow_rate,
                'reynolds_number': Re
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def electromagnetic_field(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate electromagnetic field from multiple charges

        Parameters:
            charges: List of charges with positions [{charge, x, y}]
            grid_size: Grid resolution
            x_range: X range tuple (min, max)
            y_range: Y range tuple (min, max)
        """
        result = {
            'success': False,
            'field_lines': None,
            'electric_field': None,
            'electric_potential': None,
            'error': None
        }

        try:
            # Extract parameters
            charges = parameters.get('charges', [])
            grid_size = parameters.get('grid_size', 20)
            x_range = parameters.get('x_range', (-5, 5))
            y_range = parameters.get('y_range', (-5, 5))

            # Constants
            k_e = 8.9875517923e9  # Coulomb constant

            # Create grid
            x = np.linspace(x_range[0], x_range[1], grid_size)
            y = np.linspace(y_range[0], y_range[1], grid_size)
            X, Y = np.meshgrid(x, y)

            # Initialize field components
            Ex = np.zeros_like(X)
            Ey = np.zeros_like(Y)
            V = np.zeros_like(X)

            # Calculate field from all charges
            for charge_data in charges:
                q = charge_data['charge'] * 1e-6  # Convert from μC to C
                x0 = charge_data['x']
                y0 = charge_data['y']

                # Distance from charge to each grid point
                dx = X - x0
                dy = Y - y0
                r = np.sqrt(dx**2 + dy**2)
                r[r == 0] = DIVISION_BY_ZERO_TOLERANCE  # Avoid division by zero

                # Electric field contributions
                Ex += k_e * q * dx / r**3
                Ey += k_e * q * dy / r**3

                # Electric potential contribution
                V += k_e * q / r

            # Generate field lines
            field_lines = []
            num_lines = 8  # Lines per charge

            for charge_data in charges:
                q = charge_data['charge']
                x0 = charge_data['x']
                y0 = charge_data['y']

                # Starting points around the charge
                for i in range(num_lines):
                    angle = i * 2 * np.pi / num_lines
                    start_x = x0 + 0.1 * np.cos(angle)
                    start_y = y0 + 0.1 * np.sin(angle)

                    # Trace field line
                    line_x = [start_x]
                    line_y = [start_y]

                    current_x = start_x
                    current_y = start_y
                    step_size = 0.1
                    max_steps = 100

                    for step in range(max_steps):
                        # Interpolate field at current position
                        if (x_range[0] <= current_x <= x_range[1] and 
                            y_range[0] <= current_y <= y_range[1]):

                            # Simple field calculation at point
                            Ex_point = 0
                            Ey_point = 0

                            for other_charge in charges:
                                q_other = other_charge['charge'] * 1e-6
                                dx = current_x - other_charge['x']
                                dy = current_y - other_charge['y']
                                r = np.sqrt(dx**2 + dy**2)
                                if r > DIVISION_BY_ZERO_TOLERANCE:
                                    Ex_point += k_e * q_other * dx / r**3
                                    Ey_point += k_e * q_other * dy / r**3

                            # Normalize and step
                            E_mag = np.sqrt(Ex_point**2 + Ey_point**2)
                            if E_mag > 0:
                                direction = 1 if q > 0 else -1
                                current_x += direction * step_size * Ex_point / E_mag
                                current_y += direction * step_size * Ey_point / E_mag
                                line_x.append(current_x)
                                line_y.append(current_y)
                            else:
                                break
                        else:
                            break

                    if len(line_x) > 1:
                        field_lines.append({'x': line_x, 'y': line_y})

            result['success'] = True
            result['field_lines'] = {'lines': field_lines}
            result['data'] = {
                'Ex': Ex.tolist(),
                'Ey': Ey.tolist(),
                'potential': V.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def photoelectric_effect(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate photoelectric effect

        Parameters:
            frequency: Light frequency (Hz)
            work_function: Work function of material (eV)
            intensity: Light intensity (W/m²)
        """
        result = {
            'success': False,
            'kinetic_energy': None,
            'threshold_frequency': None,
            'photocurrent': None,
            'stopping_potential': None,
            'error': None
        }

        try:
            # Extract parameters
            frequency = parameters.get('frequency', 1e15)  # Hz
            phi = parameters.get('work_function', 4.5)  # eV, typical for metals
            intensity = parameters.get('intensity', 1000)  # W/m²

            # Constants
            h = 6.62607015e-34  # Planck constant (J·s)
            h_eV = 4.135667696e-15  # Planck constant in eV·s
            c = 299792458  # Speed of light
            e = 1.602176634e-19  # Elementary charge

            # Calculate photon energy in eV
            E_photon = h * frequency / e  # Convert J to eV

            # Threshold frequency: f_0 = phi/h
            threshold_frequency = phi * e / h  # Convert from eV to Hz

            # Einstein's photoelectric equation: E_k = E_photon - phi
            if E_photon > phi:
                kinetic_energy = E_photon - phi
                photocurrent = intensity * PHOTOCURRENT_SCALE
                stopping_potential = kinetic_energy
            else:
                kinetic_energy = 0
                photocurrent = 0
                stopping_potential = 0

            result['success'] = True
            result['kinetic_energy'] = kinetic_energy
            result['threshold_frequency'] = threshold_frequency
            result['photocurrent'] = photocurrent
            result['stopping_potential'] = stopping_potential
            result['work_function'] = phi
            result['photon_energy'] = E_photon

        except Exception as e:
            result['error'] = str(e)

        return result

    def compton_scattering(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate Compton scattering

        Parameters:
            initial_wavelength: Initial photon wavelength (m)
            scattering_angle: Scattering angle (degrees)
        """
        result = {
            'success': False,
            'final_wavelength': None,
            'wavelength_shift': None,
            'final_photon_energy': None,
            'electron_recoil_energy': None,
            'error': None
        }

        try:
            # Extract parameters
            lambda_0 = parameters.get('initial_wavelength', 1e-12)
            theta_deg = parameters.get('scattering_angle', 90)

            # Constants
            h = 6.62607015e-34  # Planck constant
            c = 299792458  # Speed of light
            m_e = 9.1093837015e-31  # Electron rest mass

            # Convert angle to radians
            theta = math.radians(theta_deg)

            # Compton scattering formula
            compton_wavelength = h / (m_e * c)  # Compton wavelength
            lambda_f = lambda_0 + compton_wavelength * (1 - math.cos(theta))

            # Wavelength shift
            delta_lambda = lambda_f - lambda_0

            # Photon energies
            E_initial = h * c / lambda_0
            E_final = h * c / lambda_f

            # Electron recoil energy
            E_electron = E_initial - E_final

            result['success'] = True
            result['final_wavelength'] = lambda_f
            result['wavelength_shift'] = delta_lambda
            result['initial_photon_energy'] = E_initial
            result['final_photon_energy'] = E_final
            result['electron_recoil_energy'] = E_electron
            result['scattering_angle'] = theta_deg

        except Exception as e:
            result['error'] = str(e)

        return result

    def relativity_time_dilation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate relativistic time dilation

        Parameters:
            velocity: Relative velocity (m/s)
            proper_time: Proper time (s)
            velocities_range: Range of velocities for analysis
        """
        result = {
            'success': False,
            'dilated_time': None,
            'lorentz_factor': None,
            'velocity_analysis': None,
            'error': None
        }

        try:
            # Extract parameters
            v = parameters.get('velocity', 0.8 * 299792458)  # 0.8c
            tau = parameters.get('proper_time', 1.0)  # 1 second
            v_range = parameters.get('velocities_range', [0.1, 0.9])  # fraction of c

            # Constants
            c = 299792458  # Speed of light

            # Lorentz factor: γ = 1/sqrt(1 - v²/c²)
            beta = v / c
            if beta >= 1:
                raise InvalidInputError("velocity", str(parameters.get('velocity')), "less than speed of light")

            gamma = 1 / math.sqrt(1 - beta**2)

            # Time dilation: Δt = γ * Δτ
            dilated_time = gamma * tau

            # Analysis over velocity range
            v_analysis = np.linspace(v_range[0] * c, v_range[1] * c, 100)
            beta_analysis = v_analysis / c
            gamma_analysis = 1 / np.sqrt(1 - beta_analysis**2)
            time_analysis = gamma_analysis * tau

            result['success'] = True
            result['dilated_time'] = dilated_time
            result['lorentz_factor'] = gamma
            result['velocity'] = v
            result['proper_time'] = tau
            result['velocity_analysis'] = {
                'velocities': v_analysis.tolist(),
                'lorentz_factors': gamma_analysis.tolist(),
                'dilated_times': time_analysis.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def nuclear_decay(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate nuclear decay processes
        """
        result = {
            'success': False,
            'time': None,
            'remaining_nuclei': None,
            'decay_constant': None,
            'half_life': None,
            'activity': None,
            'error': None
        }

        try:
            # Use N0 parameter name to match test expectations
            N0 = float(parameters.get('N0', parameters.get('initial_nuclei', 1000)))
            half_life = float(parameters.get('half_life', 3600))  # seconds
            time_max = float(parameters.get('time_max', half_life * 5))
            num_points = int(parameters.get('num_points', 100))

            # Decay constant
            lambda_decay = math.log(2) / half_life

            # Time array
            t = np.linspace(0, time_max, num_points)

            # Exponential decay: N(t) = N0 * exp(-λt)
            N_t = N0 * np.exp(-lambda_decay * t)

            # Activity: A(t) = λ * N(t)
            activity = lambda_decay * N_t

            result['success'] = True
            result['time'] = t.tolist()
            result['nuclei_count'] = N_t.tolist()
            result['decay_rate'] = activity.tolist()
            result['data'] = {
                'position': N_t.tolist(),  # Consistent with other simulations
                'displacement': activity.tolist()
            }
            result['decay_constant'] = float(lambda_decay)
            result['half_life'] = float(half_life)

        except Exception as e:
            result['error'] = str(e)

        return result

    def particle_accelerator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate particle acceleration in electric and magnetic fields

        Parameters:
            particle_charge: Charge of particle (C)
            particle_mass: Mass of particle (kg)
            field_strength: Electric field strength (V/m)
            initial_velocity: Initial velocity (m/s)
            accelerator_length: Length of accelerator (m)
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'velocity': None,
            'position': None,
            'kinetic_energy': None,
            'error': None
        }

        try:
            # Extract parameters
            charge = float(parameters.get('particle_charge', 1.602e-19))  # Elementary charge
            mass = float(parameters.get('particle_mass', 9.109e-31))  # Electron mass
            E_field = float(parameters.get('field_strength', 1e6))  # V/m
            v0 = float(parameters.get('initial_velocity', 0))  # m/s
            length = float(parameters.get('accelerator_length', 0.1))  # m
            num_points = int(parameters.get('num_points', 100))

            # Acceleration: a = qE/m
            acceleration = charge * E_field / mass

            # Time to traverse accelerator
            # Using: x = v0*t + 0.5*a*t^2
            # Solving quadratic: 0.5*a*t^2 + v0*t - length = 0
            if acceleration != 0:
                discriminant = v0**2 + 2 * acceleration * length
                if discriminant >= 0:
                    time_max = (-v0 + math.sqrt(discriminant)) / acceleration
                else:
                    time_max = length / v0 if v0 > 0 else 1e-6
            else:
                time_max = length / v0 if v0 > 0 else 1e-6

            # Time array
            time = np.linspace(0, time_max, num_points)

            # Motion equations
            position = v0 * time + 0.5 * acceleration * time**2
            velocity = v0 + acceleration * time

            # Kinetic energy
            kinetic_energy = 0.5 * mass * velocity**2

            result['success'] = True
            result['time'] = time.tolist()
            result['velocity'] = velocity.tolist()
            result['position'] = position.tolist()
            result['kinetic_energy'] = kinetic_energy.tolist()
            result['data'] = {
                'position': position.tolist(),
                'displacement': velocity.tolist()  # Consistent naming
            }
            result['acceleration'] = float(acceleration)
            result['final_velocity'] = float(velocity[-1])

        except Exception as e:
            result['error'] = str(e)

        return result

    def gravitational_waves(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate gravitational wave propagation and detection
        """
        result = {
            'success': False,
            'strain': None,
            'frequency': None,
            'amplitude': None,
            'chirp_mass': None,
            'error': None
        }

        try:
            # Binary system parameters
            m1 = parameters.get('mass1', 30 * 1.989e30)  # Solar masses in kg
            m2 = parameters.get('mass2', 30 * 1.989e30)
            distance = parameters.get('distance', 1e25)  # meters (roughly 100 Mpc)

            # Time parameters
            time_max = parameters.get('time_max', 1.0)  # seconds
            num_points = parameters.get('num_points', 1000)

            t = np.linspace(0, time_max, num_points)

            # Constants
            G = 6.67430e-11  # Gravitational constant
            c = 299792458    # Speed of light

            M_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

            f0 = parameters.get('initial_frequency', 20)
            frequency = f0 * (1 + t * 100)**2

            h0 = (G * M_chirp)**(5/3) * (math.pi * f0)**(2/3) / (c**4 * distance)
            strain = h0 * np.sin(2 * math.pi * frequency * t) * np.exp(-t * 5)

            result['success'] = True
            result['time'] = t.tolist()
            result['data'] = {
                'strain': strain.tolist(),
                'frequency': frequency.tolist()
            }
            result['chirp_mass'] = M_chirp
            result['initial_amplitude'] = h0

        except Exception as e:
            result['error'] = str(e)

        return result

    def plasma_confinement(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate plasma confinement in magnetic fields (tokamak-like)
        """
        result = {
            'success': False,
            'particle_trajectories': None,
            'confinement_time': None,
            'beta_parameter': None,
            'error': None
        }

        try:
            temperature = parameters.get('temperature', 1e8)  # Kelvin
            density = parameters.get('density', 1e20)  # particles/m³
            B_field = parameters.get('magnetic_field', 5.0)  # Tesla

            # Constants
            k_B = 1.380649e-23  # Boltzmann constant
            m_ion = 1.67262192e-27  # Proton mass

            # Thermal velocity
            v_thermal = math.sqrt(2 * k_B * temperature / m_ion)

            # Larmor radius
            q = 1.602176634e-19  # Elementary charge
            r_larmor = m_ion * v_thermal / (q * B_field)

            # Confinement time (approximate)
            tau_confinement = parameters.get('confinement_time', 1.0)  # seconds

            # Beta parameter (plasma pressure / magnetic pressure)
            p_plasma = density * k_B * temperature
            p_magnetic = B_field**2 / (2 * 4e-7 * math.pi)
            beta = p_plasma / p_magnetic

            # Simple trajectory simulation
            time_max = min(tau_confinement, 0.1)  # seconds
            num_points = parameters.get('num_points', 500)

            t = np.linspace(0, time_max, num_points)
            omega_c = q * B_field / m_ion

            # Helical trajectory
            x = r_larmor * np.cos(omega_c * t)
            y = r_larmor * np.sin(omega_c * t)
            z = v_thermal * 0.1 * t  # Parallel motion

            result['success'] = True
            result['time'] = t.tolist()
            result['data'] = {
                'x_trajectory': x.tolist(),
                'y_trajectory': y.tolist(),
                'z_trajectory': z.tolist()
            }
            result['beta_parameter'] = beta
            result['larmor_radius'] = r_larmor
            result['thermal_velocity'] = v_thermal

        except Exception as e:
            result['error'] = str(e)

        return result

    def superconductivity(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate superconductivity phenomena (BCS theory basics)
        """
        result = {
            'success': False,
            'critical_temperature': None,
            'energy_gap': None,
            'cooper_pair_density': None,
            'critical_field': None,
            'error': None
        }

        try:
            # Material parameters
            Tc = parameters.get('critical_temperature', 9.2)  # Kelvin (Niobium)
            temperature = parameters.get('temperature', 4.2)  # Kelvin

            # Constants
            k_B = 1.380649e-23  # Boltzmann constant

            # BCS energy gap at T=0
            Delta_0 = 1.764 * k_B * Tc  # BCS weak-coupling limit

            # Temperature-dependent energy gap (approximate)
            if temperature < Tc:
                t = temperature / Tc
                Delta_T = Delta_0 * math.sqrt(1 - t**4)  # Simplified temperature dependence
            else:
                Delta_T = 0

            # Critical magnetic field (thermodynamic)
            mu_0 = 4e-7 * math.pi  # Permeability of free space
            n_cooper = parameters.get('cooper_pair_density', 1e28)  # pairs/m³
            Hc = math.sqrt(2 * mu_0 * n_cooper * Delta_T)

            # Temperature array for plotting
            T_array = np.linspace(0, Tc * 1.5, 100)
            gap_array = []

            for T in T_array:
                if T < Tc:
                    t = T / Tc
                    gap = Delta_0 * math.sqrt(1 - t**4) if t < 1 else 0
                else:
                    gap = 0
                gap_array.append(gap)

            result['success'] = True
            result['temperature_array'] = T_array.tolist()
            result['data'] = {
                'energy_gap': gap_array
            }
            result['critical_temperature'] = Tc
            result['energy_gap_at_T'] = Delta_T
            result['critical_field'] = Hc

        except Exception as e:
            result['error'] = str(e)

        return result

    def quantum_tunneling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate quantum tunneling through potential barriers
        """
        result = {
            'success': False,
            'transmission_coefficient': None,
            'reflection_coefficient': None,
            'wavefunction': None,
            'probability_density': None,
            'error': None
        }

        try:
            # Barrier parameters
            barrier_height = parameters.get('barrier_height', 2.0)  # eV
            barrier_width = parameters.get('barrier_width', 1e-9)  # meters
            particle_energy = parameters.get('particle_energy', 1.0)  # eV

            # Constants
            hbar = 1.054571817e-34  # Reduced Planck constant
            m_e = 9.1093837015e-31  # Electron mass
            eV_to_J = 1.602176634e-19  # eV to Joules conversion

            # Convert energies to Joules
            V0 = barrier_height * eV_to_J
            E = particle_energy * eV_to_J

            # Wave vectors
            k1 = math.sqrt(2 * m_e * E) / hbar

            if E > V0:
                # Classical case - particle has enough energy
                k2 = math.sqrt(2 * m_e * (E - V0)) / hbar
                transmission = 1.0  # Simplified
            else:
                # Quantum tunneling case
                kappa = math.sqrt(2 * m_e * (V0 - E)) / hbar
                transmission = 1 / (1 + (V0**2 * math.sinh(kappa * barrier_width)**2) / (4 * E * (V0 - E)))

            reflection = 1 - transmission

            # Position array for wavefunction
            x_max = barrier_width * 3
            num_points = parameters.get('num_points', 1000)
            x = np.linspace(-x_max, x_max, num_points)

            # Reflection/transmission amplitudes (simplified)
            r = math.sqrt(reflection)
            t = math.sqrt(transmission)

            # Wavefunction across three regions
            psi = np.zeros(len(x), dtype=complex)
            for i, xi in enumerate(x):
                if xi < 0:
                    # Region I: incident + reflected
                    psi[i] = np.exp(1j * k1 * xi) + r * np.exp(-1j * k1 * xi)
                elif xi < barrier_width:
                    # Region II: inside barrier
                    if E < V0:
                        psi[i] = t * np.exp(-kappa * xi)
                    else:
                        psi[i] = t * np.exp(1j * k2 * xi)
                else:
                    # Region III: transmitted
                    if E < V0:
                        psi[i] = t * np.exp(-kappa * barrier_width) * np.exp(1j * k1 * (xi - barrier_width))
                    else:
                        psi[i] = t * np.exp(1j * k1 * xi)

            probability = np.abs(psi)**2

            result['success'] = True
            result['position'] = x.tolist()
            result['data'] = {
                'wavefunction_real': np.real(psi).tolist(),
                'wavefunction_imag': np.imag(psi).tolist(),
                'probability_density': probability.tolist()
            }
            result['transmission_coefficient'] = transmission
            result['reflection_coefficient'] = reflection

        except Exception as e:
            result['error'] = str(e)

        return result

    def laser_physics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate laser operation (rate equations and population inversion)
        """
        result = {
            'success': False,
            'population_inversion': None,
            'photon_density': None,
            'output_power': None,
            'threshold_current': None,
            'error': None
        }

        try:
            # Laser parameters
            pump_rate = parameters.get('pump_rate', 1e12)  # s⁻¹
            spontaneous_lifetime = parameters.get('spontaneous_lifetime', 1e-9)  # seconds
            stimulated_cross_section = parameters.get('stimulated_cross_section', 1e-20)  # m²

            # Rate equation parameters
            time_max = parameters.get('time_max', 10 * spontaneous_lifetime)
            num_points = parameters.get('num_points', 1000)

            t = np.linspace(0, time_max, num_points)
            dt = t[1] - t[0]

            # Initial conditions
            N_upper = np.zeros(len(t))  # Upper level population
            N_lower = np.zeros(len(t))  # Lower level population
            photon_density = np.zeros(len(t))

            # Rate equations (simplified two-level system)
            for i in range(1, len(t)):
                # Population inversion
                delta_N = N_upper[i-1] - N_lower[i-1]

                # Stimulated emission rate
                W_stimulated = stimulated_cross_section * photon_density[i-1] * 3e8  # c * sigma * phi

                # Rate equations
                dN_upper_dt = pump_rate - N_upper[i-1] / spontaneous_lifetime - W_stimulated * delta_N
                dN_lower_dt = N_upper[i-1] / spontaneous_lifetime + W_stimulated * delta_N
                dphoton_dt = W_stimulated * delta_N - photon_density[i-1] / (1e-12)  # Cavity loss

                # Update populations
                N_upper[i] = N_upper[i-1] + dN_upper_dt * dt
                N_lower[i] = N_lower[i-1] + dN_lower_dt * dt
                photon_density[i] = max(0, photon_density[i-1] + dphoton_dt * dt)

            # Output power (proportional to photon density)
            hf = 6.626e-34 * 3e14  # Photon energy (assuming visible light)
            output_power = photon_density * hf

            result['success'] = True
            result['time'] = t.tolist()
            result['data'] = {
                'upper_population': N_upper.tolist(),
                'lower_population': N_lower.tolist(),
                'photon_density': photon_density.tolist(),
                'output_power': output_power.tolist()
            }
            result['pump_rate'] = pump_rate

        except Exception as e:
            result['error'] = str(e)

        return result

    def solid_state_physics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate solid state physics phenomena (band structure, DOS)
        """
        result = {
            'success': False,
            'band_structure': None,
            'density_of_states': None,
            'fermi_energy': None,
            'effective_mass': None,
            'error': None
        }

        try:
            # Material parameters
            lattice_constant = parameters.get('lattice_constant', 5.43e-10)  # Silicon
            band_gap = parameters.get('band_gap', 1.12)  # eV (Silicon)
            effective_mass_ratio = parameters.get('effective_mass_ratio', 0.26)  # m*/m_e

            # Constants
            hbar = 1.054571817e-34
            m_e = 9.1093837015e-31
            k_B = 1.380649e-23
            eV_to_J = 1.602176634e-19

            # k-space grid
            k_max = math.pi / lattice_constant
            num_points = parameters.get('num_points', 100)
            k = np.linspace(-k_max, k_max, num_points)

            # Simple parabolic band model
            m_eff = effective_mass_ratio * m_e
            E_conduction = (hbar * k)**2 / (2 * m_eff) + band_gap * eV_to_J / 2
            E_valence = -(hbar * k)**2 / (2 * m_eff) - band_gap * eV_to_J / 2

            # Convert to eV for display
            E_conduction_eV = E_conduction / eV_to_J
            E_valence_eV = E_valence / eV_to_J

            # Density of states (3D parabolic bands)
            E_array = np.linspace(-2, 4, 200)  # eV
            dos = np.zeros_like(E_array)

            for i, E in enumerate(E_array):
                if E > band_gap / 2:  # Conduction band
                    # Energy above conduction band edge
                    E_shifted = (E - band_gap / 2) * eV_to_J
                    if E_shifted > 0:
                        dos[i] = math.sqrt(2 * m_eff**3) * math.sqrt(E_shifted) / (math.pi**2 * hbar**3)
                elif E < -band_gap / 2:  # Valence band
                    # Energy below valence band edge (as positive value)
                    E_shifted = abs(E + band_gap / 2) * eV_to_J
                    if E_shifted > 0:
                        dos[i] = math.sqrt(2 * m_eff**3) * math.sqrt(E_shifted) / (math.pi**2 * hbar**3)

            # Fermi energy (assume intrinsic semiconductor)
            E_fermi = 0  # Mid-gap for intrinsic

            result['success'] = True
            result['k_space'] = k.tolist()
            result['energy_array'] = E_array.tolist()
            result['data'] = {
                'conduction_band': E_conduction_eV.tolist(),
                'valence_band': E_valence_eV.tolist(),
                'density_of_states': dos.tolist()
            }
            result['band_gap'] = band_gap
            result['fermi_energy'] = E_fermi
            result['effective_mass'] = m_eff

        except Exception as e:
            result['error'] = str(e)

        return result

    def three_body_orbital(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate three-body orbital motion (e.g., Earth-Moon-Satellite system)

        Parameters:
            mass_primary: Primary body mass (kg) - e.g., Earth
            mass_secondary: Secondary body mass (kg) - e.g., Moon
            mass_tertiary: Tertiary body mass (kg) - e.g., Satellite
            primary_secondary_distance: Distance between primary and secondary (m)
            initial_tertiary_distance: Initial distance of tertiary from primary (m)
            initial_tertiary_velocity: Initial velocity of tertiary (m/s)
            tertiary_angle: Initial angle of tertiary relative to primary-secondary line (degrees)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'x1': None,  # Primary body
            'y1': None,
            'x2': None,  # Secondary body
            'y2': None,
            'x3': None,  # Tertiary body
            'y3': None,
            'error': None
        }

        try:
            # Extract parameters
            M1 = parameters.get('mass_primary', 5.972e24)  # Earth mass
            M2 = parameters.get('mass_secondary', 7.342e22)  # Moon mass
            M3 = parameters.get('mass_tertiary', 1000.0)   # Satellite mass
            r12 = parameters.get('primary_secondary_distance', 3.844e8)  # Earth-Moon distance
            r13_init = parameters.get('initial_tertiary_distance', 4.0e8)  # Initial satellite distance
            v3_init = parameters.get('initial_tertiary_velocity', 1000.0)  # Initial satellite velocity
            angle_deg = parameters.get('tertiary_angle', 30.0)  # Initial angle
            time_max = parameters.get('time_max', 3600 * 24 * 28)  # 28 days
            num_points = parameters.get('num_points', 5000)

            # Constants
            G = 6.674e-11  # Gravitational constant

            # Time array
            dt = time_max / num_points
            time = np.linspace(0, time_max, num_points)

            # Initialize arrays
            x1 = np.zeros(num_points)  # Primary (fixed at origin for simplicity)
            y1 = np.zeros(num_points)
            x2 = np.zeros(num_points)  # Secondary (Moon)
            y2 = np.zeros(num_points)
            x3 = np.zeros(num_points)  # Tertiary (Satellite)
            y3 = np.zeros(num_points)
            vx2 = np.zeros(num_points)  # Secondary velocity
            vy2 = np.zeros(num_points)
            vx3 = np.zeros(num_points)  # Tertiary velocity
            vy3 = np.zeros(num_points)

            # Initial conditions
            # Primary at origin
            x1[0] = 0.0
            y1[0] = 0.0

            # Secondary in circular orbit around primary
            x2[0] = r12
            y2[0] = 0.0
            orbital_velocity_2 = math.sqrt(G * M1 / r12)
            vx2[0] = 0.0
            vy2[0] = orbital_velocity_2

            # Tertiary at specified distance and angle
            angle_rad = math.radians(angle_deg)
            x3[0] = r13_init * math.cos(angle_rad)
            y3[0] = r13_init * math.sin(angle_rad)
            vx3[0] = -v3_init * math.sin(angle_rad)
            vy3[0] = v3_init * math.cos(angle_rad)

            def accelerations(x1p, y1p, x2p, y2p, x3p, y3p):
                r12 = max(math.sqrt((x2p - x1p)**2 + (y2p - y1p)**2), 1e3)
                r13 = max(math.sqrt((x3p - x1p)**2 + (y3p - y1p)**2), 1e3)
                r23 = max(math.sqrt((x3p - x2p)**2 + (y3p - y2p)**2), 1e3)

                F12x = -G * M1 * M2 * (x2p - x1p) / r12**3
                F12y = -G * M1 * M2 * (y2p - y1p) / r12**3
                F32x = -G * M2 * M3 * (x2p - x3p) / r23**3
                F32y = -G * M2 * M3 * (y2p - y3p) / r23**3

                F13x = -G * M1 * M3 * (x3p - x1p) / r13**3
                F13y = -G * M1 * M3 * (y3p - y1p) / r13**3
                F23x = -G * M2 * M3 * (x3p - x2p) / r23**3
                F23y = -G * M2 * M3 * (y3p - y2p) / r23**3

                ax2 = (F12x + F32x) / M2
                ay2 = (F12y + F32y) / M2
                ax3 = (F13x + F23x) / M3
                ay3 = (F13y + F23y) / M3
                return ax2, ay2, ax3, ay3

            # Velocity-Verlet integration for better energy behavior
            for i in range(1, num_points):
                ax2_prev, ay2_prev, ax3_prev, ay3_prev = accelerations(
                    x1[i-1], y1[i-1], x2[i-1], y2[i-1], x3[i-1], y3[i-1]
                )

                vx2_half = vx2[i-1] + 0.5 * ax2_prev * dt
                vy2_half = vy2[i-1] + 0.5 * ay2_prev * dt
                vx3_half = vx3[i-1] + 0.5 * ax3_prev * dt
                vy3_half = vy3[i-1] + 0.5 * ay3_prev * dt

                x2[i] = x2[i-1] + vx2_half * dt
                y2[i] = y2[i-1] + vy2_half * dt
                x3[i] = x3[i-1] + vx3_half * dt
                y3[i] = y3[i-1] + vy3_half * dt

                x1[i] = 0.0
                y1[i] = 0.0

                ax2_new, ay2_new, ax3_new, ay3_new = accelerations(
                    x1[i], y1[i], x2[i], y2[i], x3[i], y3[i]
                )

                vx2[i] = vx2_half + 0.5 * ax2_new * dt
                vy2[i] = vy2_half + 0.5 * ay2_new * dt
                vx3[i] = vx3_half + 0.5 * ax3_new * dt
                vy3[i] = vy3_half + 0.5 * ay3_new * dt

            result['success'] = True
            result['time'] = time.tolist()
            result['x1'] = x1.tolist()  # Primary body
            result['y1'] = y1.tolist()
            result['x2'] = x2.tolist()  # Secondary body (Moon)
            result['y2'] = y2.tolist()
            result['x3'] = x3.tolist()  # Tertiary body (Satellite)
            result['y3'] = y3.tolist()
            result['data'] = {
                'x_primary': x1.tolist(),
                'y_primary': y1.tolist(),
                'x_secondary': x2.tolist(),
                'y_secondary': y2.tolist(),
                'x_tertiary': x3.tolist(),
                'y_tertiary': y3.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def four_body_orbital(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate four-body orbital motion (e.g., Earth-Moon-Satellite-Subsatellite system)

        Parameters:
            mass_primary: Primary body mass (kg) - e.g., Earth
            mass_secondary: Secondary body mass (kg) - e.g., Moon
            mass_tertiary: Tertiary body mass (kg) - e.g., Satellite
            mass_quaternary: Quaternary body mass (kg) - e.g., Subsatellite
            primary_secondary_distance: Distance between primary and secondary (m)
            initial_tertiary_distance: Initial distance of tertiary from primary (m)
            initial_tertiary_velocity: Initial velocity of tertiary (m/s)
            tertiary_angle: Initial angle of tertiary relative to primary-secondary line (degrees)
            initial_quaternary_distance: Initial distance of quaternary from tertiary (m)
            initial_quaternary_velocity: Initial velocity of quaternary relative to tertiary (m/s)
            quaternary_angle: Initial angle of quaternary relative to tertiary (degrees)
            time_max: Maximum simulation time
            num_points: Number of simulation points
        """
        result = {
            'success': False,
            'time': None,
            'x1': None,  # Primary body
            'y1': None,
            'x2': None,  # Secondary body
            'y2': None,
            'x3': None,  # Tertiary body
            'y3': None,
            'x4': None,  # Quaternary body
            'y4': None,
            'error': None
        }

        try:
            # Extract parameters
            M1 = parameters.get('mass_primary', 5.972e24)  # Earth mass
            M2 = parameters.get('mass_secondary', 7.342e22)  # Moon mass
            M3 = parameters.get('mass_tertiary', 1000.0)   # Satellite mass
            M4 = parameters.get('mass_quaternary', 100.0)   # Subsatellite mass
            r12 = parameters.get('primary_secondary_distance', 3.844e8)  # Earth-Moon distance
            r13_init = parameters.get('initial_tertiary_distance', 4.0e8)  # Initial satellite distance
            v3_init = parameters.get('initial_tertiary_velocity', 1000.0)  # Initial satellite velocity
            angle3_deg = parameters.get('tertiary_angle', 30.0)  # Initial angle
            r34_init = parameters.get('initial_quaternary_distance', 1000.0)  # Initial subsatellite distance
            v4_init = parameters.get('initial_quaternary_velocity', 100.0)  # Initial subsatellite velocity
            angle4_deg = parameters.get('quaternary_angle', 45.0)  # Initial angle
            time_max = parameters.get('time_max', 3600 * 24 * 7)  # 7 days
            num_points = parameters.get('num_points', 3000)

            # Constants
            G = 6.674e-11  # Gravitational constant

            # Time array
            dt = time_max / num_points
            time = np.linspace(0, time_max, num_points)

            # Initialize arrays
            x1 = np.zeros(num_points)  # Primary (fixed at origin)
            y1 = np.zeros(num_points)
            x2 = np.zeros(num_points)  # Secondary (Moon)
            y2 = np.zeros(num_points)
            x3 = np.zeros(num_points)  # Tertiary (Satellite)
            y3 = np.zeros(num_points)
            x4 = np.zeros(num_points)  # Quaternary (Subsatellite)
            y4 = np.zeros(num_points)
            vx2 = np.zeros(num_points)  # Secondary velocity
            vy2 = np.zeros(num_points)
            vx3 = np.zeros(num_points)  # Tertiary velocity
            vy3 = np.zeros(num_points)
            vx4 = np.zeros(num_points)  # Quaternary velocity
            vy4 = np.zeros(num_points)

            # Initial conditions
            # Primary at origin
            x1[0] = 0.0
            y1[0] = 0.0

            # Secondary in circular orbit around primary
            x2[0] = r12
            y2[0] = 0.0
            orbital_velocity_2 = math.sqrt(G * M1 / r12)
            vx2[0] = 0.0
            vy2[0] = orbital_velocity_2

            # Tertiary at specified distance and angle
            angle3_rad = math.radians(angle3_deg)
            x3[0] = r13_init * math.cos(angle3_rad)
            y3[0] = r13_init * math.sin(angle3_rad)
            vx3[0] = -v3_init * math.sin(angle3_rad)
            vy3[0] = v3_init * math.cos(angle3_rad)

            # Quaternary relative to tertiary
            angle4_rad = math.radians(angle4_deg)
            x4[0] = x3[0] + r34_init * math.cos(angle4_rad)
            y4[0] = y3[0] + r34_init * math.sin(angle4_rad)
            vx4[0] = vx3[0] - v4_init * math.sin(angle4_rad)
            vy4[0] = vy3[0] + v4_init * math.cos(angle4_rad)

            def accelerations_four(x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p):
                r12 = max(math.sqrt((x2p - x1p)**2 + (y2p - y1p)**2), 1e3)
                r13 = max(math.sqrt((x3p - x1p)**2 + (y3p - y1p)**2), 1e3)
                r14 = max(math.sqrt((x4p - x1p)**2 + (y4p - y1p)**2), 1e3)
                r23 = max(math.sqrt((x3p - x2p)**2 + (y3p - y2p)**2), 1e3)
                r24 = max(math.sqrt((x4p - x2p)**2 + (y4p - y2p)**2), 1e3)
                r34 = max(math.sqrt((x4p - x3p)**2 + (y4p - y3p)**2), 1e3)

                F12x = -G * M1 * M2 * (x2p - x1p) / r12**3
                F12y = -G * M1 * M2 * (y2p - y1p) / r12**3
                F32x = -G * M2 * M3 * (x2p - x3p) / r23**3
                F32y = -G * M2 * M3 * (y2p - y3p) / r23**3
                F42x = -G * M2 * M4 * (x2p - x4p) / r24**3
                F42y = -G * M2 * M4 * (y2p - y4p) / r24**3

                F13x = -G * M1 * M3 * (x3p - x1p) / r13**3
                F13y = -G * M1 * M3 * (y3p - y1p) / r13**3
                F23x = -G * M2 * M3 * (x3p - x2p) / r23**3
                F23y = -G * M2 * M3 * (y3p - y2p) / r23**3
                F43x = -G * M3 * M4 * (x3p - x4p) / r34**3
                F43y = -G * M3 * M4 * (y3p - y4p) / r34**3

                F14x = -G * M1 * M4 * (x4p - x1p) / r14**3
                F14y = -G * M1 * M4 * (y4p - y1p) / r14**3
                F24x = -G * M2 * M4 * (x4p - x2p) / r24**3
                F24y = -G * M2 * M4 * (y4p - y2p) / r24**3
                F34x = -G * M3 * M4 * (x4p - x3p) / r34**3
                F34y = -G * M3 * M4 * (y4p - y3p) / r34**3

                ax2 = (F12x + F32x + F42x) / M2
                ay2 = (F12y + F32y + F42y) / M2
                ax3 = (F13x + F23x + F43x) / M3
                ay3 = (F13y + F23y + F43y) / M3
                ax4 = (F14x + F24x + F34x) / M4
                ay4 = (F14y + F24y + F34y) / M4
                return ax2, ay2, ax3, ay3, ax4, ay4

            # Velocity-Verlet integration
            for i in range(1, num_points):
                ax2_prev, ay2_prev, ax3_prev, ay3_prev, ax4_prev, ay4_prev = accelerations_four(
                    x1[i-1], y1[i-1], x2[i-1], y2[i-1], x3[i-1], y3[i-1], x4[i-1], y4[i-1]
                )

                vx2_half = vx2[i-1] + 0.5 * ax2_prev * dt
                vy2_half = vy2[i-1] + 0.5 * ay2_prev * dt
                vx3_half = vx3[i-1] + 0.5 * ax3_prev * dt
                vy3_half = vy3[i-1] + 0.5 * ay3_prev * dt
                vx4_half = vx4[i-1] + 0.5 * ax4_prev * dt
                vy4_half = vy4[i-1] + 0.5 * ay4_prev * dt

                x2[i] = x2[i-1] + vx2_half * dt
                y2[i] = y2[i-1] + vy2_half * dt
                x3[i] = x3[i-1] + vx3_half * dt
                y3[i] = y3[i-1] + vy3_half * dt
                x4[i] = x4[i-1] + vx4_half * dt
                y4[i] = y4[i-1] + vy4_half * dt

                x1[i] = 0.0
                y1[i] = 0.0

                ax2_new, ay2_new, ax3_new, ay3_new, ax4_new, ay4_new = accelerations_four(
                    x1[i], y1[i], x2[i], y2[i], x3[i], y3[i], x4[i], y4[i]
                )

                vx2[i] = vx2_half + 0.5 * ax2_new * dt
                vy2[i] = vy2_half + 0.5 * ay2_new * dt
                vx3[i] = vx3_half + 0.5 * ax3_new * dt
                vy3[i] = vy3_half + 0.5 * ay3_new * dt
                vx4[i] = vx4_half + 0.5 * ax4_new * dt
                vy4[i] = vy4_half + 0.5 * ay4_new * dt

            result['success'] = True
            result['time'] = time.tolist()
            result['x1'] = x1.tolist()  # Primary body
            result['y1'] = y1.tolist()
            result['x2'] = x2.tolist()  # Secondary body (Moon)
            result['y2'] = y2.tolist()
            result['x3'] = x3.tolist()  # Tertiary body (Satellite)
            result['y3'] = y3.tolist()
            result['x4'] = x4.tolist()  # Quaternary body (Subsatellite)
            result['y4'] = y4.tolist()
            result['data'] = {
                'x_primary': x1.tolist(),
                'y_primary': y1.tolist(),
                'x_secondary': x2.tolist(),
                'y_secondary': y2.tolist(),
                'x_tertiary': x3.tolist(),
                'y_tertiary': y3.tolist(),
                'x_quaternary': x4.tolist(),
                'y_quaternary': y4.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def finite_element_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finite Element Analysis for engineering structures
        """
        result = {
            'success': False,
            'displacement': None,
            'stress': None,
            'strain': None,
            'safety_factor': None,
            'error': None
        }

        try:
            # Extract FEA parameters
            material_props = params.get('material', {})
            E = material_props.get('elastic_modulus', 200e9)  # Pa
            nu = material_props.get('poisson_ratio', 0.3)
            density = material_props.get('density', 7850)  # kg/m³

            geometry = params.get('geometry', {})
            length = geometry.get('length', 1.0)  # m
            width = geometry.get('width', 0.1)  # m
            height = geometry.get('height', 0.1)  # m

            loading = params.get('loading', {})
            force = loading.get('force', 1000)  # N
            pressure = loading.get('pressure', 0)  # Pa

            # Simplified beam analysis (Euler-Bernoulli theory)
            I = (width * height**3) / 12  # Second moment of area
            A = width * height  # Cross-sectional area

            # Maximum deflection (simply supported beam, point load at center)
            max_deflection = (force * length**3) / (48 * E * I)

            # Maximum stress
            max_moment = force * length / 4
            max_stress = (max_moment * height/2) / I

            # Maximum strain
            max_strain = max_stress / E

            # Safety factor (assuming yield strength of steel)
            yield_strength = material_props.get('yield_strength', 250e6)  # Pa
            safety_factor = yield_strength / max_stress if max_stress > 0 else float('inf')

            # Element mesh data (simplified)
            num_elements = params.get('mesh_density', 10)
            element_length = length / num_elements

            nodes = []
            elements = []
            displacements = []
            stresses = []

            for i in range(num_elements + 1):
                x = i * element_length
                nodes.append({'id': i, 'x': x, 'y': 0, 'z': 0})

                # Deflection at each node (parabolic approximation)
                if i <= num_elements / 2:
                    deflection = (force * x) / (48 * E * I) * (3 * length**2 - 4 * x**2)
                else:
                    deflection = (force * (length - x)) / (48 * E * I) * (3 * length**2 - 4 * (length - x)**2)

                displacements.append({'node': i, 'ux': 0, 'uy': deflection, 'uz': 0})

                if i < num_elements:
                    # Element stress (simplified)
                    element_stress = max_stress * abs(2*i/num_elements - 1)
                    stresses.append({'element': i, 'stress': element_stress})
                    elements.append({'id': i, 'node1': i, 'node2': i+1})

            result['success'] = True
            result['displacement'] = {
                'max_deflection': max_deflection,
                'nodes': nodes,
                'displacements': displacements
            }
            result['stress'] = {
                'max_stress': max_stress,
                'elements': elements,
                'stresses': stresses
            }
            result['strain'] = {'max_strain': max_strain}
            result['safety_factor'] = safety_factor

        except Exception as e:
            result['error'] = str(e)

        return result

    def computational_fluid_dynamics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplified Computational Fluid Dynamics simulation
        """
        result = {
            'success': False,
            'velocity_field': None,
            'pressure_field': None,
            'flow_properties': None,
            'note': 'Simplified channel approximation; not a full Navier-Stokes solution',
            'error': None
        }

        try:
            # Extract CFD parameters
            fluid_props = params.get('fluid', {})
            density = fluid_props.get('density', 1.225)  # kg/m³ (air at STP)
            viscosity = fluid_props.get('viscosity', 1.81e-5)  # Pa⋅s (air)

            geometry = params.get('geometry', {})
            length = geometry.get('length', 1.0)  # m
            width = geometry.get('width', 0.1)  # m

            flow_conditions = params.get('flow', {})
            inlet_velocity = flow_conditions.get('inlet_velocity', 1.0)  # m/s
            inlet_pressure = flow_conditions.get('inlet_pressure', 101325)  # Pa

            # Grid generation (simplified)
            nx = params.get('grid_x', 20)
            ny = params.get('grid_y', 10)

            # Validate grid parameters
            if nx < 2 or ny < 2:
                raise InvalidInputError("grid dimensions (nx, ny)", f"({nx}, {ny})", "at least (2, 2), recommend (10+, 10+)")
            if nx < 10 or ny < 10:
                import warnings
                warnings.warn("Low grid resolution. Recommend at least 10 points per dimension for meaningful results.")

            dx = length / (nx - 1)
            dy = width / (ny - 1)

            # Reynolds number
            Re = (density * inlet_velocity * width) / viscosity

            # Flow regime
            if Re < 2300:
                flow_regime = 'laminar'
            elif Re > 4000:
                flow_regime = 'turbulent'
            else:
                flow_regime = 'transitional'

            # Simplified velocity field (parabolic profile for laminar flow)
            velocity_field = []
            pressure_field = []

            for i in range(nx):
                for j in range(ny):
                    x = i * dx
                    y = j * dy

                    # Parabolic velocity profile (Poiseuille flow approximation)
                    if flow_regime == 'laminar':
                        u_velocity = inlet_velocity * 1.5 * (1 - (2*y/width - 1)**2)
                        v_velocity = 0
                    else:
                        # Power law profile for turbulent flow
                        u_velocity = inlet_velocity * (y/width)**(1/7)
                        v_velocity = 0

                    # Pressure drop (simplified)
                    if flow_regime == 'laminar':
                        pressure_drop = (32 * viscosity * inlet_velocity * x) / width**2
                    else:
                        # Darcy-Weisbach equation approximation
                        f = 0.316 / Re**0.25  # Blasius correlation
                        pressure_drop = (f * density * inlet_velocity**2 * x) / (2 * width)

                    pressure = inlet_pressure - pressure_drop

                    velocity_field.append({
                        'x': x, 'y': y, 'u': u_velocity, 'v': v_velocity
                    })
                    pressure_field.append({
                        'x': x, 'y': y, 'pressure': pressure
                    })

            # Flow properties
            flow_properties = {
                'reynolds_number': Re,
                'flow_regime': flow_regime,
                'pressure_drop_total': pressure_drop,
                'average_velocity': inlet_velocity * 2/3 if flow_regime == 'laminar' else inlet_velocity * 7/8
            }

            result['success'] = True
            result['velocity_field'] = velocity_field
            result['pressure_field'] = pressure_field
            result['flow_properties'] = flow_properties

        except Exception as e:
            result['error'] = str(e)

        return result

    def electromagnetics_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Electromagnetic field simulation
        """
        result = {
            'success': False,
            'electric_field': None,
            'magnetic_field': None,
            'power_density': None,
            'error': None
        }

        try:
            # Extract electromagnetic parameters
            geometry = params.get('geometry', {})
            length = geometry.get('length', 0.1)  # m
            width = geometry.get('width', 0.1)  # m

            source = params.get('source', {})
            frequency = source.get('frequency', 1e9)  # Hz (1 GHz)
            amplitude = source.get('amplitude', 1.0)  # V/m

            # Physical constants
            c = 299792458  # m/s (speed of light)
            mu0 = 4e-7 * np.pi  # H/m (permeability of free space)
            eps0 = 8.854e-12  # F/m (permittivity of free space)

            # Wave properties
            wavelength = c / frequency
            k = 2 * np.pi / wavelength  # wave number
            omega = 2 * np.pi * frequency  # angular frequency

            # Grid generation
            nx = params.get('grid_x', 50)
            ny = params.get('grid_y', 50)

            # Validate grid parameters
            if nx < 2 or ny < 2:
                raise InvalidInputError("grid dimensions (nx, ny)", f"({nx}, {ny})", "at least (2, 2), recommend (10+, 10+)")
            if nx < 10 or ny < 10:
                import warnings
                warnings.warn("Low grid resolution. Recommend at least 10 points per dimension for meaningful results.")

            dx = length / (nx - 1)
            dy = width / (ny - 1)

            # Field calculation (simplified plane wave)
            electric_field = []
            magnetic_field = []
            power_density = []

            for i in range(nx):
                for j in range(ny):
                    x = i * dx
                    y = j * dy

                    # Distance from source (assumed at origin)
                    r = np.sqrt(x**2 + y**2)

                    # Electric field (plane wave propagation)
                    phase = k * r
                    Ex = amplitude * np.cos(phase)
                    Ey = 0
                    Ez = 0

                    # Magnetic field (perpendicular to electric field)
                    Hx = 0
                    Hy = amplitude / np.sqrt(mu0/eps0) * np.cos(phase)
                    Hz = 0

                    # Poynting vector (power density)
                    S = 0.5 * np.sqrt(eps0/mu0) * amplitude**2

                    electric_field.append({
                        'x': x, 'y': y, 'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
                        'magnitude': np.sqrt(Ex**2 + Ey**2 + Ez**2)
                    })

                    magnetic_field.append({
                        'x': x, 'y': y, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
                        'magnitude': np.sqrt(Hx**2 + Hy**2 + Hz**2)
                    })

                    power_density.append({
                        'x': x, 'y': y, 'power_density': S
                    })

            result['success'] = True
            result['electric_field'] = electric_field
            result['magnetic_field'] = magnetic_field
            result['power_density'] = power_density

        except Exception as e:
            result['error'] = str(e)

        return result

    def chaos_theory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate chaotic systems (Lorenz attractor)
        """
        result = {
            'success': False,
            'x': None,
            'y': None,
            'z': None,
            'error': None
        }

        try:
            # Lorenz system parameters
            sigma = parameters.get('sigma', 10.0)
            rho = parameters.get('rho', 28.0)
            beta = parameters.get('beta', 8.0/3.0)

            # Initial conditions
            x0 = parameters.get('x0', 1.0)
            y0 = parameters.get('y0', 1.0)
            z0 = parameters.get('z0', 1.0)

            # Time parameters
            dt = parameters.get('dt', 0.01)
            num_points = parameters.get('num_points', 5000)

            # Arrays for storing results
            x = np.zeros(num_points)
            y = np.zeros(num_points)
            z = np.zeros(num_points)

            # Initial conditions
            x[0], y[0], z[0] = x0, y0, z0

            # Euler integration
            for i in range(1, num_points):
                dx = sigma * (y[i-1] - x[i-1])
                dy = x[i-1] * (rho - z[i-1]) - y[i-1]
                dz = x[i-1] * y[i-1] - beta * z[i-1]

                x[i] = x[i-1] + dx * dt
                y[i] = y[i-1] + dy * dt
                z[i] = z[i-1] + dz * dt

            result['success'] = True
            # Ensure time array matches data length
            result['time'] = (np.arange(num_points) * dt).tolist()
            result['data'] = {
                'x': x.tolist(),
                'y': y.tolist(),
                'z': z.tolist()
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def quantum_mechanics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate quantum mechanical systems (particle in a box)
        """
        result = {
            'success': False,
            'energy_levels': None,
            'wavefunctions': None,
            'error': None
        }

        try:
            # Box parameters
            L = parameters.get('length', 1e-9)  # m (nanometer scale)
            mass = parameters.get('mass', 9.109e-31)  # kg (electron mass)
            n_max = parameters.get('n_max', 5)  # number of energy levels

            # Physical constants
            h_bar = 1.055e-34  # J⋅s

            # Energy levels: E_n = (n²π²ℏ²)/(2mL²)
            energy_levels = []
            wavefunctions = []

            x = np.linspace(0, L, 1000)

            for n in range(1, n_max + 1):
                # Energy level
                energy = (n**2 * np.pi**2 * h_bar**2) / (2 * mass * L**2)
                energy_levels.append(energy)

                # Wavefunction: ψ_n(x) = √(2/L) * sin(nπx/L)
                psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
                wavefunctions.append({
                    'n': n,
                    'x': x.tolist(),
                    'psi': psi.tolist(),
                    'probability_density': (psi**2).tolist()
                })

            result['success'] = True
            result['energy_levels'] = energy_levels
            result['wavefunctions'] = wavefunctions

        except Exception as e:
            result['error'] = str(e)

        return result

    def crystallography(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate crystal structures and X-ray diffraction
        """
        result = {
            'success': False,
            'lattice_parameters': None,
            'diffraction_pattern': None,
            'error': None
        }

        try:
            # Crystal parameters
            crystal_system = parameters.get('crystal_system', 'cubic')
            lattice_constant = parameters.get('lattice_constant', 5e-10)  # m

            # X-ray parameters
            wavelength = parameters.get('wavelength', 1.54e-10)  # m (Cu Kα)
            theta_max = parameters.get('theta_max', 90)  # degrees

            # Generate lattice parameters
            if crystal_system == 'cubic':
                a = b = c = lattice_constant
                alpha = beta = gamma = 90
            elif crystal_system == 'tetragonal':
                a = b = lattice_constant
                c = lattice_constant * 1.2
                alpha = beta = gamma = 90
            else:  # hexagonal
                a = b = lattice_constant
                c = lattice_constant * 1.6
                alpha = beta = 90
                gamma = 120

            # Bragg's law: nλ = 2d sinθ
            # For cubic system: d_hkl = a/√(h² + k² + l²)
            reflections = []
            angles = []
            intensities = []

            for h in range(1, 4):
                for k in range(0, 4):
                    for l in range(0, 4):
                        h_sq_sum = h**2 + k**2 + l**2
                        if h_sq_sum == 0:
                            continue

                        d_spacing = a / np.sqrt(h_sq_sum)
                        sin_theta = wavelength / (2 * d_spacing)

                        if 0 < sin_theta <= 1:  # Valid range for sine
                            theta = np.arcsin(sin_theta) * 180 / np.pi
                            if theta <= theta_max:
                                # Structure factor (simplified)
                                intensity = 100 / h_sq_sum

                                reflections.append({'h': h, 'k': k, 'l': l})
                                angles.append(theta)
                                intensities.append(intensity)

            result['success'] = True
            result['lattice_parameters'] = {
                'a': a, 'b': b, 'c': c,
                'alpha': alpha, 'beta': beta, 'gamma': gamma,
                'system': crystal_system
            }
            result['diffraction_pattern'] = {
                'reflections': reflections,
                'angles': angles,
                'intensities': intensities
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def optics_lenses(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate optical lenses and ray tracing

        Parameters:
            lens_type: Type of lens ('convex', 'concave', 'planoconvex', 'planoconcave')
            focal_length: Focal length of the lens (cm)
            object_distance: Distance of object from lens (cm)
            object_height: Height of object (cm)
            lens_diameter: Diameter of lens (cm)
            num_rays: Number of rays to trace
        """
        result = {
            'success': False,
            'image_distance': None,
            'image_height': None,
            'magnification': None,
            'image_type': None,
            'ray_paths': None,
            'error': None
        }

        try:
            # Extract parameters
            lens_type = parameters.get('lens_type', 'convex')
            f = parameters.get('focal_length', 10.0)  # cm
            do = parameters.get('object_distance', 20.0)  # cm
            ho = parameters.get('object_height', 5.0)  # cm
            lens_diameter = parameters.get('lens_diameter', 5.0)  # cm
            num_rays = parameters.get('num_rays', 5)

            # Adjust focal length based on lens type
            if lens_type in ['concave', 'planoconcave']:
                f = -abs(f)  # Negative focal length for diverging lenses
            else:
                f = abs(f)  # Positive focal length for converging lenses

            # Thin lens equation: 1/f = 1/do + 1/di
            if abs(do - f) < NUMERICAL_TOLERANCE:
                # Object at focal point
                di = float('inf')
                hi = float('inf')
                m = float('inf')
                image_type = 'No image formed (object at focal point)'
            else:
                di = (f * do) / (do - f)
                m = -di / do  # Magnification
                hi = m * ho  # Image height

                # Determine image type
                if di > 0:
                    image_type = 'Real'
                else:
                    image_type = 'Virtual'
                    di = abs(di)  # Report positive distance for virtual images

                if abs(m) > 1:
                    image_type += ', Magnified'
                elif abs(m) < 1:
                    image_type += ', Diminished'

                if m < 0:
                    image_type += ', Inverted'
                else:
                    image_type += ', Upright'

            # Ray tracing
            ray_paths = []

            # Principal rays for ray diagram
            # Ray 1: Parallel to principal axis, passes through focal point after lens
            ray1 = {
                'start': [-do, ho],
                'lens_entry': [0, ho],
                'end': [di, -hi] if di != float('inf') else [2*abs(f), -ho*2*abs(f)/do]
            }
            ray_paths.append(ray1)

            # Ray 2: Through center of lens (undeviated)
            if abs(do) > NUMERICAL_TOLERANCE:
                ray2 = {
                    'start': [-do, ho],
                    'lens_entry': [0, 0],
                    'end': [di, -hi] if di != float('inf') else [2*abs(f), -ho*2*abs(f)/do]
                }
                ray_paths.append(ray2)

            # Ray 3: Through focal point, emerges parallel
            if f > 0 and do > f:  # Only for converging lenses
                if abs(do - f) > NUMERICAL_TOLERANCE:  # Additional safety check
                    y_lens = ho * f / (do - f)
                    if abs(y_lens) <= lens_diameter/2:
                        ray3 = {
                            'start': [-do, ho],
                            'lens_entry': [0, y_lens],
                            'end': [di, -hi] if di != float('inf') else [2*abs(f), y_lens]
                        }
                    ray_paths.append(ray3)

            # Additional rays for visualization
            for i in range(num_rays - 3):
                y_lens = (i - num_rays/2) * lens_diameter / num_rays
                if lens_type in ['convex', 'planoconvex']:
                    # Converging lens behavior
                    if di != float('inf'):
                        ray = {
                            'start': [-do, ho],
                            'lens_entry': [0, y_lens],
                            'end': [di, -hi]
                        }
                    else:
                        ray = {
                            'start': [-do, ho],
                            'lens_entry': [0, y_lens],
                            'end': [2*abs(f), y_lens]
                        }
                else:
                    # Diverging lens behavior
                    ray = {
                        'start': [-do, ho],
                        'lens_entry': [0, y_lens],
                        'end': [abs(di), -abs(hi)]
                    }
                ray_paths.append(ray)

            result['success'] = True
            result['image_distance'] = di if di != float('inf') else None
            result['image_height'] = hi if di != float('inf') else None
            result['magnification'] = m if di != float('inf') else None
            result['image_type'] = image_type
            result['ray_paths'] = ray_paths
            result['lens_properties'] = {
                'type': lens_type,
                'focal_length': f,
                'diameter': lens_diameter
            }

        except Exception as e:
            result['error'] = str(e)

        return result

# Global physics simulator instance
physics_simulator = PhysicsSimulator()