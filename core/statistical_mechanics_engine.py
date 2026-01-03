import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import math
import random
from dataclasses import dataclass
from enum import Enum

class StatisticalMethod(Enum):
    """Available statistical mechanics methods"""
    BOLTZMANN = "BOLTZMANN"
    CANONICAL_ENSEMBLE = "CANONICAL_ENSEMBLE"
    MICROCANONICAL = "MICROCANONICAL"
    GRAND_CANONICAL = "GRAND_CANONICAL"
    MONTE_CARLO = "MONTE_CARLO"
    MOLECULAR_DYNAMICS = "MOLECULAR_DYNAMICS"
    METROPOLIS = "METROPOLIS"
    ISING_MODEL = "ISING_MODEL"

@dataclass
class StatMechResult:
    """Result container for statistical mechanics calculations"""
    success: bool
    method: str
    results: Dict[str, Any]
    error: Optional[str] = None
    computation_time: Optional[float] = None

class StatisticalMechanicsEngine:
    """Engine for statistical mechanics calculations and simulations"""
    
    def __init__(self):
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.N_A = 6.02214076e23  # Avogadro's number
        self.h = 6.62607015e-34   # Planck constant (J⋅s)
        
    def calculate_boltzmann_distribution(self, energies: List[float], temperature: float) -> StatMechResult:
        """
        Calculate Boltzmann distribution for given energy levels
        
        Parameters:
            energies: List of energy levels (in units of kT)
            temperature: Temperature (K)
        """
        try:
            # Convert to numpy array
            E = np.array(energies)
            
            # Calculate Boltzmann factors
            boltzmann_factors = np.exp(-E / (self.k_B * temperature))
            
            # Calculate partition function
            Z = np.sum(boltzmann_factors)
            
            # Calculate probabilities
            probabilities = boltzmann_factors / Z
            
            # Calculate average energy
            avg_energy = np.sum(E * probabilities)
            
            # Calculate entropy
            S = -self.k_B * np.sum(probabilities * np.log(probabilities + 1e-10))
            
            results = {
                'energies': E.tolist(),
                'probabilities': probabilities.tolist(),
                'partition_function': Z,
                'average_energy': avg_energy,
                'entropy': S,
                'temperature': temperature
            }
            
            return StatMechResult(
                success=True,
                method=StatisticalMethod.BOLTZMANN.value,
                results=results
            )
            
        except Exception as e:
            return StatMechResult(
                success=False,
                method=StatisticalMethod.BOLTZMANN.value,
                results={},
                error=str(e)
            )
    
    def canonical_partition_function(self, energy_function: Callable, 
                                   states: List[Any], temperature: float) -> StatMechResult:
        """
        Calculate canonical partition function for a system
        
        Parameters:
            energy_function: Function that returns energy for a given state
            states: List of possible states
            temperature: Temperature (K)
        """
        try:
            beta = 1 / (self.k_B * temperature)
            
            # Calculate partition function
            Z = 0
            state_probabilities = []
            energies = []
            
            for state in states:
                E = energy_function(state)
                energies.append(E)
                exp_factor = np.exp(-beta * E)
                Z += exp_factor
                state_probabilities.append(exp_factor)
            
            # Normalize probabilities
            state_probabilities = np.array(state_probabilities) / Z
            
            # Calculate thermodynamic properties
            avg_energy = np.sum(np.array(energies) * state_probabilities)
            
            # Helmholtz free energy
            F = -self.k_B * temperature * np.log(Z)
            
            # Entropy
            S = (avg_energy - F) / temperature
            
            results = {
                'partition_function': Z,
                'state_probabilities': state_probabilities.tolist(),
                'average_energy': avg_energy,
                'helmholtz_free_energy': F,
                'entropy': S,
                'temperature': temperature
            }
            
            return StatMechResult(
                success=True,
                method=StatisticalMethod.CANONICAL_ENSEMBLE.value,
                results=results
            )
            
        except Exception as e:
            return StatMechResult(
                success=False,
                method=StatisticalMethod.CANONICAL_ENSEMBLE.value,
                results={},
                error=str(e)
            )
    
    def monte_carlo_simulation(self, energy_function: Callable, initial_state: Any,
                             temperature: float, n_steps: int = 10000) -> StatMechResult:
        """
        Perform Monte Carlo simulation using Metropolis algorithm
        
        Parameters:
            energy_function: Function that returns energy for a given state
            initial_state: Initial configuration
            temperature: Temperature (K)
            n_steps: Number of Monte Carlo steps
        """
        try:
            beta = 1 / (self.k_B * temperature)
            
            # Initialize
            current_state = initial_state
            current_energy = energy_function(current_state)
            
            states_history = [current_state]
            energy_history = [current_energy]
            accepted_moves = 0
            
            for step in range(n_steps):
                # Propose new state (simple random perturbation)
                new_state = self._perturb_state(current_state)
                new_energy = energy_function(new_state)
                
                # Metropolis acceptance criterion
                delta_E = new_energy - current_energy
                
                if delta_E < 0 or random.random() < np.exp(-beta * delta_E):
                    current_state = new_state
                    current_energy = new_energy
                    accepted_moves += 1
                
                if step % 100 == 0:  # Record every 100 steps
                    states_history.append(current_state)
                    energy_history.append(current_energy)
            
            # Calculate statistics
            energy_array = np.array(energy_history)
            avg_energy = np.mean(energy_array)
            energy_variance = np.var(energy_array)
            specific_heat = energy_variance * beta**2
            
            results = {
                'average_energy': avg_energy,
                'energy_variance': energy_variance,
                'specific_heat': specific_heat,
                'acceptance_rate': accepted_moves / n_steps,
                'energy_history': energy_history[-100:],  # Last 100 values
                'final_state': current_state,
                'temperature': temperature,
                'n_steps': n_steps
            }
            
            return StatMechResult(
                success=True,
                method=StatisticalMethod.MONTE_CARLO.value,
                results=results
            )
            
        except Exception as e:
            return StatMechResult(
                success=False,
                method=StatisticalMethod.MONTE_CARLO.value,
                results={},
                error=str(e)
            )
    
    def ising_model_2d(self, size: int, temperature: float, 
                      n_steps: int = 10000, h_field: float = 0) -> StatMechResult:
        """
        Simulate 2D Ising model
        
        Parameters:
            size: Grid size (size x size)
            temperature: Temperature (in units of J/k_B)
            n_steps: Number of Monte Carlo steps
            h_field: External magnetic field
        """
        try:
            # Initialize random spin configuration
            spins = np.random.choice([-1, 1], size=(size, size))
            beta = 1 / temperature
            
            magnetization_history = []
            energy_history = []
            
            for step in range(n_steps):
                # Random site selection
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                
                # Calculate energy change for flip
                neighbors_sum = (
                    spins[(i+1)%size, j] + spins[(i-1)%size, j] +
                    spins[i, (j+1)%size] + spins[i, (j-1)%size]
                )
                
                delta_E = 2 * spins[i, j] * (neighbors_sum + h_field)
                
                # Metropolis criterion
                if delta_E < 0 or random.random() < np.exp(-beta * delta_E):
                    spins[i, j] *= -1
                
                # Record observables
                if step % 100 == 0:
                    magnetization = np.mean(spins)
                    energy = self._calculate_ising_energy(spins, h_field)
                    magnetization_history.append(magnetization)
                    energy_history.append(energy)
            
            # Calculate final statistics
            final_magnetization = np.mean(spins)
            final_energy = self._calculate_ising_energy(spins, h_field)
            
            # Phase transition order parameter
            susceptibility = np.var(magnetization_history) * beta * size**2
            
            results = {
                'final_magnetization': final_magnetization,
                'final_energy': final_energy,
                'susceptibility': susceptibility,
                'magnetization_history': magnetization_history[-50:],
                'energy_history': energy_history[-50:],
                'final_configuration': spins.tolist(),
                'temperature': temperature,
                'size': size,
                'external_field': h_field
            }
            
            return StatMechResult(
                success=True,
                method=StatisticalMethod.ISING_MODEL.value,
                results=results
            )
            
        except Exception as e:
            return StatMechResult(
                success=False,
                method=StatisticalMethod.ISING_MODEL.value,
                results={},
                error=str(e)
            )
    
    def molecular_dynamics(self, particles: List[Dict], potential_function: Callable,
                         time_step: float = 1e-15, n_steps: int = 1000,
                         temperature: Optional[float] = None) -> StatMechResult:
        """
        Perform molecular dynamics simulation
        
        Parameters:
            particles: List of particle dictionaries with 'position', 'velocity', 'mass'
            potential_function: Function that returns potential energy and forces
            time_step: Integration time step (seconds)
            n_steps: Number of time steps
            temperature: Optional temperature for velocity rescaling
        """
        try:
            # Initialize particle arrays
            n_particles = len(particles)
            positions = np.array([p['position'] for p in particles])
            velocities = np.array([p['velocity'] for p in particles])
            masses = np.array([p['mass'] for p in particles])
            
            # Storage for trajectory
            position_history = [positions.copy()]
            energy_history = []
            
            for step in range(n_steps):
                # Calculate forces
                potential_energy, forces = potential_function(positions)
                
                # Velocity Verlet integration
                velocities += 0.5 * forces / masses[:, np.newaxis] * time_step
                positions += velocities * time_step
                
                # Recalculate forces
                _, new_forces = potential_function(positions)
                velocities += 0.5 * new_forces / masses[:, np.newaxis] * time_step
                
                # Temperature control (if requested)
                if temperature is not None and step % 100 == 0:
                    velocities = self._rescale_velocities(velocities, masses, temperature)
                
                # Record data
                if step % 10 == 0:
                    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
                    total_energy = kinetic_energy + potential_energy
                    energy_history.append(total_energy)
                    
                if step % 100 == 0:
                    position_history.append(positions.copy())
            
            # Calculate final properties
            final_kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
            final_potential, _ = potential_function(positions)
            avg_energy = np.mean(energy_history)
            
            # Temperature from kinetic energy
            final_temperature = 2 * final_kinetic / (3 * n_particles * self.k_B)
            
            results = {
                'final_positions': positions.tolist(),
                'final_velocities': velocities.tolist(),
                'average_energy': avg_energy,
                'final_kinetic_energy': final_kinetic,
                'final_potential_energy': final_potential,
                'final_temperature': final_temperature,
                'energy_history': energy_history[-100:],
                'n_particles': n_particles,
                'time_step': time_step,
                'total_time': time_step * n_steps
            }
            
            return StatMechResult(
                success=True,
                method=StatisticalMethod.MOLECULAR_DYNAMICS.value,
                results=results
            )
            
        except Exception as e:
            return StatMechResult(
                success=False,
                method=StatisticalMethod.MOLECULAR_DYNAMICS.value,
                results={},
                error=str(e)
            )
    
    def _perturb_state(self, state):
        """Generate small perturbation of current state"""
        if isinstance(state, np.ndarray):
            perturbation = np.random.normal(0, 0.1, state.shape)
            return state + perturbation
        elif isinstance(state, (int, float)):
            return state + random.gauss(0, 0.1)
        else:
            return state  # Default: no perturbation
    
    def _calculate_ising_energy(self, spins: np.ndarray, h_field: float) -> float:
        """Calculate total energy of Ising configuration"""
        # Nearest neighbor interactions
        interaction_energy = -np.sum(
            spins * np.roll(spins, 1, axis=0) +
            spins * np.roll(spins, 1, axis=1)
        )
        
        # External field contribution
        field_energy = -h_field * np.sum(spins)
        
        return interaction_energy + field_energy
    
    def _rescale_velocities(self, velocities: np.ndarray, masses: np.ndarray, 
                          target_temp: float) -> np.ndarray:
        """Rescale velocities to match target temperature"""
        current_kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        current_temp = 2 * current_kinetic / (3 * len(masses) * self.k_B)
        
        if current_temp > 0:
            scale_factor = np.sqrt(target_temp / current_temp)
            return velocities * scale_factor
        return velocities

# Simple test function
def test_energy_function(state):
    """Simple quadratic potential for testing"""
    if isinstance(state, (list, np.ndarray)):
        return np.sum(np.array(state)**2)
    return state**2

# Lennard-Jones potential for molecular dynamics
def lennard_jones_potential(positions, epsilon=1.0, sigma=1.0):
    """Calculate Lennard-Jones potential and forces"""
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    potential = 0.0
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            
            if r < 3 * sigma:  # Cutoff
                r6 = (sigma / r)**6
                r12 = r6**2
                
                # Potential
                potential += 4 * epsilon * (r12 - r6)
                
                # Force
                f_magnitude = 24 * epsilon * (2 * r12 - r6) / r
                f_vec = f_magnitude * r_vec / r
                
                forces[i] -= f_vec
                forces[j] += f_vec
    
    return potential, forces