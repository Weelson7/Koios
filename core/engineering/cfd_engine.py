import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.exceptions import (
    InvalidInputError, ConvergenceError, ConfigurationError, NumericalInstabilityError
)

class FlowType(Enum):
    """Types of fluid flow"""
    LAMINAR = "LAMINAR"
    TURBULENT = "TURBULENT"
    TRANSITIONAL = "TRANSITIONAL"
    COMPRESSIBLE = "COMPRESSIBLE"
    INCOMPRESSIBLE = "INCOMPRESSIBLE"

class BoundaryType(Enum):
    """CFD boundary condition types"""
    WALL = "WALL"
    INLET = "INLET"
    OUTLET = "OUTLET"
    SYMMETRY = "SYMMETRY"
    PERIODIC = "PERIODIC"
    PRESSURE_OUTLET = "PRESSURE_OUTLET"
    VELOCITY_INLET = "VELOCITY_INLET"

class SolverType(Enum):
    """CFD solver types"""
    SIMPLE = "SIMPLE"  # Semi-Implicit Method for Pressure Linked Equations
    PISO = "PISO"      # Pressure-Implicit with Splitting of Operators
    COUPLED = "COUPLED"
    FRACTIONAL_STEP = "FRACTIONAL_STEP"

@dataclass
class Fluid:
    """Fluid properties"""
    name: str
    density: float  # kg/m³
    viscosity: float  # Pa·s (dynamic viscosity)
    specific_heat: Optional[float] = None  # J/(kg·K)
    thermal_conductivity: Optional[float] = None  # W/(m·K)
    
    @property
    def kinematic_viscosity(self) -> float:
        """Calculate kinematic viscosity"""
        return self.viscosity / self.density
    
    @classmethod
    def water(cls, temperature: float = 20.0):
        """Water properties at given temperature (°C)"""
        # Simplified temperature dependence
        density = 1000 - 0.0178 * abs(temperature - 4)**1.7
        viscosity = 0.001 * (1.78 - 0.045 * temperature + 0.00045 * temperature**2)
        
        return cls(
            name=f"Water at {temperature}°C",
            density=density,
            viscosity=viscosity,
            specific_heat=4186,
            thermal_conductivity=0.598
        )
    
    @classmethod
    def air(cls, temperature: float = 20.0, pressure: float = 101325):
        """Air properties at given temperature (°C) and pressure (Pa)"""
        # Ideal gas approximation
        R = 287.0  # Specific gas constant for air
        T_kelvin = temperature + 273.15
        density = pressure / (R * T_kelvin)
        
        # Sutherland's formula for viscosity
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 110.4
        viscosity = mu_ref * (T_kelvin / T_ref)**1.5 * (T_ref + S) / (T_kelvin + S)
        
        return cls(
            name=f"Air at {temperature}°C",
            density=density,
            viscosity=viscosity,
            specific_heat=1005,
            thermal_conductivity=0.0257
        )

@dataclass
class BoundaryCondition:
    """CFD boundary condition"""
    type: BoundaryType
    values: Dict[str, float]  # e.g., {'u': 1.0, 'v': 0.0, 'p': 0.0}
    
class CFDMesh:
    """Structured CFD mesh"""
    def __init__(self, nx: int, ny: int, nz: int = 1, 
                 lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        
        # Grid spacing
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        self.dz = lz / (nz - 1) if nz > 1 else 1.0
        
        # Create coordinate arrays
        self.x = np.linspace(0, lx, nx)
        self.y = np.linspace(0, ly, ny)
        self.z = np.linspace(0, lz, nz) if nz > 1 else np.array([0])
        
        # Mesh grid
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Boundary markers
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        
    def set_boundary(self, location: str, condition: BoundaryCondition):
        """Set boundary condition at location (north, south, east, west)"""
        self.boundary_conditions[location] = condition

class CFDEngine:
    """Main Computational Fluid Dynamics engine"""
    
    def __init__(self, mesh: CFDMesh, fluid: Fluid):
        self.mesh = mesh
        self.fluid = fluid
        
        # Solution fields
        self.u = np.zeros((mesh.ny, mesh.nx))  # x-velocity
        self.v = np.zeros((mesh.ny, mesh.nx))  # y-velocity
        self.p = np.zeros((mesh.ny, mesh.nx))  # pressure
        self.T = np.zeros((mesh.ny, mesh.nx))  # temperature
        
        # Previous time step values
        self.u_old = np.copy(self.u)
        self.v_old = np.copy(self.v)
        
        # Solver parameters
        self.dt = 0.001  # time step
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.solver_type = SolverType.SIMPLE
        
    def initialize_flow(self, u_init: float = 0.0, v_init: float = 0.0, 
                       p_init: float = 0.0, T_init: float = 20.0):
        """Initialize flow field"""
        self.u.fill(u_init)
        self.v.fill(v_init)
        self.p.fill(p_init)
        self.T.fill(T_init)
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions to flow field"""
        nx, ny = self.mesh.nx, self.mesh.ny
        
        for location, bc in self.mesh.boundary_conditions.items():
            bc_type = bc.type
            if location == 'west':  # x = 0
                if bc_type == BoundaryType.WALL:
                    self.u[:, 0] = 0.0
                    self.v[:, 0] = 0.0
                elif bc_type in (BoundaryType.VELOCITY_INLET, BoundaryType.INLET):
                    self.u[:, 0] = bc.values.get('u', self.u[:, 0])
                    self.v[:, 0] = bc.values.get('v', self.v[:, 0])
                elif bc_type in (BoundaryType.OUTLET, BoundaryType.PRESSURE_OUTLET):
                    self.u[:, 0] = self.u[:, 1]
                    self.v[:, 0] = self.v[:, 1]
                    if 'p' in bc.values:
                        self.p[:, 0] = bc.values['p']
                elif bc_type == BoundaryType.SYMMETRY:
                    self.u[:, 0] = self.u[:, 1]
                    self.v[:, 0] = 0.0
                if 'p' in bc.values:
                    self.p[:, 0] = bc.values['p']
                if 'T' in bc.values:
                    self.T[:, 0] = bc.values['T']
                    
            elif location == 'east':  # x = lx
                if bc_type == BoundaryType.WALL:
                    self.u[:, -1] = 0.0
                    self.v[:, -1] = 0.0
                elif bc_type in (BoundaryType.VELOCITY_INLET, BoundaryType.INLET):
                    self.u[:, -1] = bc.values.get('u', self.u[:, -1])
                    self.v[:, -1] = bc.values.get('v', self.v[:, -1])
                elif bc_type in (BoundaryType.OUTLET, BoundaryType.PRESSURE_OUTLET):
                    self.u[:, -1] = self.u[:, -2]
                    self.v[:, -1] = self.v[:, -2]
                    if 'p' in bc.values:
                        self.p[:, -1] = bc.values['p']
                elif bc_type == BoundaryType.SYMMETRY:
                    self.u[:, -1] = self.u[:, -2]
                    self.v[:, -1] = 0.0
                if 'p' in bc.values:
                    self.p[:, -1] = bc.values['p']
                if 'T' in bc.values:
                    self.T[:, -1] = bc.values['T']
                    
            elif location == 'south':  # y = 0
                if bc_type == BoundaryType.WALL:
                    self.u[0, :] = 0.0
                    self.v[0, :] = 0.0
                elif bc_type in (BoundaryType.VELOCITY_INLET, BoundaryType.INLET):
                    self.u[0, :] = bc.values.get('u', self.u[0, :])
                    self.v[0, :] = bc.values.get('v', self.v[0, :])
                elif bc_type in (BoundaryType.OUTLET, BoundaryType.PRESSURE_OUTLET):
                    self.u[0, :] = self.u[1, :]
                    self.v[0, :] = self.v[1, :]
                    if 'p' in bc.values:
                        self.p[0, :] = bc.values['p']
                elif bc_type == BoundaryType.SYMMETRY:
                    self.u[0, :] = self.u[1, :]
                    self.v[0, :] = 0.0
                if 'p' in bc.values:
                    self.p[0, :] = bc.values['p']
                if 'T' in bc.values:
                    self.T[0, :] = bc.values['T']
                    
            elif location == 'north':  # y = ly
                if bc_type == BoundaryType.WALL:
                    self.u[-1, :] = 0.0
                    self.v[-1, :] = 0.0
                elif bc_type in (BoundaryType.VELOCITY_INLET, BoundaryType.INLET):
                    self.u[-1, :] = bc.values.get('u', self.u[-1, :])
                    self.v[-1, :] = bc.values.get('v', self.v[-1, :])
                elif bc_type in (BoundaryType.OUTLET, BoundaryType.PRESSURE_OUTLET):
                    self.u[-1, :] = self.u[-2, :]
                    self.v[-1, :] = self.v[-2, :]
                    if 'p' in bc.values:
                        self.p[-1, :] = bc.values['p']
                elif bc_type == BoundaryType.SYMMETRY:
                    self.u[-1, :] = self.u[-2, :]
                    self.v[-1, :] = 0.0
                if 'p' in bc.values:
                    self.p[-1, :] = bc.values['p']
                if 'T' in bc.values:
                    self.T[-1, :] = bc.values['T']
    
    def calculate_dt_stability(self) -> float:
        """Calculate stable time step based on CFL condition"""
        dx, dy = self.mesh.dx, self.mesh.dy
        nu = self.fluid.kinematic_viscosity
        
        # Maximum velocities
        u_max = np.max(np.abs(self.u))
        v_max = np.max(np.abs(self.v))
        
        # CFL conditions
        dt_conv = 0.5 * min(dx / (u_max + 1e-10), dy / (v_max + 1e-10))
        dt_diff = 0.25 * min(dx**2, dy**2) / nu
        
        return 0.5 * min(dt_conv, dt_diff)
    
    def solve_momentum_equation(self):
        """Solve momentum equations using finite difference"""
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        nu = self.fluid.kinematic_viscosity
        
        # Copy current values
        u_star = np.copy(self.u)
        v_star = np.copy(self.v)
        
        # Interior points only (1:-1, 1:-1)
        # X-momentum equation
        u_star[1:-1, 1:-1] = self.u[1:-1, 1:-1] + dt * (
            # Advection terms (upwind scheme)
            - self.u[1:-1, 1:-1] * (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2*dx)
            - self.v[1:-1, 1:-1] * (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2*dy)
            # Diffusion terms
            + nu * ((self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) / dx**2
                  + (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) / dy**2)
            # Pressure gradient
            - (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2*dx * self.fluid.density)
        )
        
        # Y-momentum equation
        v_star[1:-1, 1:-1] = self.v[1:-1, 1:-1] + dt * (
            # Advection terms
            - self.u[1:-1, 1:-1] * (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2*dx)
            - self.v[1:-1, 1:-1] * (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2*dy)
            # Diffusion terms
            + nu * ((self.v[1:-1, 2:] - 2*self.v[1:-1, 1:-1] + self.v[1:-1, :-2]) / dx**2
                  + (self.v[2:, 1:-1] - 2*self.v[1:-1, 1:-1] + self.v[:-2, 1:-1]) / dy**2)
            # Pressure gradient
            - (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2*dy * self.fluid.density)
        )
        
        return u_star, v_star
    
    def solve_pressure_poisson(self, u_star: np.ndarray, v_star: np.ndarray):
        """Solve pressure Poisson equation"""
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        rho = self.fluid.density
        
        # Divergence of velocity field
        div = np.zeros_like(self.p)
        div[1:-1, 1:-1] = (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2*dx) +
            (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2*dy)
        )
        
        # Pressure Poisson equation: ∇²p = ρ/dt * div(V*)
        # Using iterative solver (Gauss-Seidel)
        p_new = np.copy(self.p)
        
        dx2 = dx * dx
        dy2 = dy * dy
        denom = 2.0 * (dx2 + dy2)

        for _ in range(100):  # Sub-iterations
            p_old_iter = np.copy(p_new)
            
            # Anisotropic discretization of ∇²p = ρ/dt * ∇·V*
            p_new[1:-1, 1:-1] = (
                (p_old_iter[1:-1, 2:] + p_new[1:-1, :-2]) * dy2 +
                (p_old_iter[2:, 1:-1] + p_new[:-2, 1:-1]) * dx2 +
                dx2 * dy2 * rho / dt * div[1:-1, 1:-1]
            ) / denom
            
            # Apply pressure boundary conditions
            # Neumann BC on walls (dp/dn = 0)
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]
            
            # Check convergence
            if np.max(np.abs(p_new - p_old_iter)) < 1e-6:
                break
        
        return p_new
    
    def correct_velocity(self, u_star: np.ndarray, v_star: np.ndarray, 
                        p_new: np.ndarray):
        """Correct velocity field with pressure gradient"""
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        rho = self.fluid.density
        
        # Velocity correction
        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt / rho * (
            (p_new[1:-1, 2:] - p_new[1:-1, :-2]) / (2*dx)
        )
        
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt / rho * (
            (p_new[2:, 1:-1] - p_new[:-2, 1:-1]) / (2*dy)
        )
        
        self.p = p_new
    
    def solve_SIMPLE(self, n_iterations: int = 100):
        """SIMPLE algorithm for incompressible flow"""
        for iteration in range(n_iterations):
            # Store old values
            self.u_old = np.copy(self.u)
            self.v_old = np.copy(self.v)
            
            # Apply boundary conditions
            self.apply_boundary_conditions()
            
            # Solve momentum equations
            u_star, v_star = self.solve_momentum_equation()
            
            # Solve pressure correction equation
            p_new = self.solve_pressure_poisson(u_star, v_star)
            
            # Correct velocities
            self.correct_velocity(u_star, v_star, p_new)
            
            # Check convergence
            u_residual = np.max(np.abs(self.u - self.u_old))
            v_residual = np.max(np.abs(self.v - self.v_old))
            
            if max(u_residual, v_residual) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
    
    def solve_heat_equation(self):
        """Solve energy equation for temperature"""
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        alpha = self.fluid.thermal_conductivity / (self.fluid.density * self.fluid.specific_heat)
        
        # Temperature equation
        self.T[1:-1, 1:-1] = self.T[1:-1, 1:-1] + dt * (
            # Advection
            - self.u[1:-1, 1:-1] * (self.T[1:-1, 2:] - self.T[1:-1, :-2]) / (2*dx)
            - self.v[1:-1, 1:-1] * (self.T[2:, 1:-1] - self.T[:-2, 1:-1]) / (2*dy)
            # Diffusion
            + alpha * ((self.T[1:-1, 2:] - 2*self.T[1:-1, 1:-1] + self.T[1:-1, :-2]) / dx**2
                     + (self.T[2:, 1:-1] - 2*self.T[1:-1, 1:-1] + self.T[:-2, 1:-1]) / dy**2)
        )
    
    def calculate_stream_function(self) -> np.ndarray:
        """Calculate stream function from velocity field"""
        psi = np.zeros_like(self.u)
        
        # Integrate v along x (bottom boundary)
        for i in range(1, self.mesh.nx):
            psi[0, i] = psi[0, i-1] + self.v[0, i] * self.mesh.dx
        
        # Integrate -u along y
        for j in range(1, self.mesh.ny):
            psi[j, :] = psi[j-1, :] - self.u[j, :] * self.mesh.dy
        
        return psi
    
    def calculate_vorticity(self) -> np.ndarray:
        """Calculate vorticity field"""
        dx, dy = self.mesh.dx, self.mesh.dy
        omega = np.zeros_like(self.u)
        
        # ω = ∂v/∂x - ∂u/∂y
        omega[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2*dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2*dy)
        )
        
        return omega
    
    def calculate_reynolds_number(self, characteristic_length: float, 
                                 characteristic_velocity: float) -> float:
        """Calculate Reynolds number"""
        return (self.fluid.density * characteristic_velocity * 
                characteristic_length / self.fluid.viscosity)
    
    def visualize_flow(self, field: str = 'velocity', streamlines: bool = True):
        """Visualize flow field"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if field == 'velocity':
            # Velocity magnitude
            vel_mag = np.sqrt(self.u**2 + self.v**2)
            c = ax.contourf(self.mesh.X, self.mesh.Y, vel_mag, 
                           levels=20, cmap='viridis')
            plt.colorbar(c, ax=ax, label='Velocity magnitude [m/s]')
            
            if streamlines:
                # Add streamlines
                ax.streamplot(self.mesh.X, self.mesh.Y, self.u, self.v, 
                            color='white', linewidth=0.5, density=1.5)
            
        elif field == 'pressure':
            c = ax.contourf(self.mesh.X, self.mesh.Y, self.p, 
                           levels=20, cmap='RdBu_r')
            plt.colorbar(c, ax=ax, label='Pressure [Pa]')
            
        elif field == 'vorticity':
            omega = self.calculate_vorticity()
            c = ax.contourf(self.mesh.X, self.mesh.Y, omega, 
                           levels=20, cmap='seismic')
            plt.colorbar(c, ax=ax, label='Vorticity [1/s]')
            
        elif field == 'stream':
            psi = self.calculate_stream_function()
            c = ax.contour(self.mesh.X, self.mesh.Y, psi, 
                          levels=20, colors='black', linewidths=1)
            ax.clabel(c, inline=True, fontsize=8)
            
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'CFD Results - {field.capitalize()}')
        ax.set_aspect('equal')
        
        return fig
    
    def export_results(self, filename: str):
        """Export results to file"""
        results = {
            'mesh': {
                'x': self.mesh.x.tolist(),
                'y': self.mesh.y.tolist(),
            },
            'solution': {
                'u': self.u.tolist(),
                'v': self.v.tolist(),
                'p': self.p.tolist(),
                'T': self.T.tolist()
            },
            'fluid': {
                'density': self.fluid.density,
                'viscosity': self.fluid.viscosity
            }
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

# Example applications
def lid_driven_cavity_flow():
    """Classic lid-driven cavity flow benchmark"""
    # Create mesh
    mesh = CFDMesh(nx=50, ny=50, lx=1.0, ly=1.0)
    
    # Set boundary conditions
    mesh.set_boundary('north', BoundaryCondition(
        BoundaryType.WALL, {'u': 1.0, 'v': 0.0}))  # Moving lid
    mesh.set_boundary('south', BoundaryCondition(
        BoundaryType.WALL, {'u': 0.0, 'v': 0.0}))
    mesh.set_boundary('east', BoundaryCondition(
        BoundaryType.WALL, {'u': 0.0, 'v': 0.0}))
    mesh.set_boundary('west', BoundaryCondition(
        BoundaryType.WALL, {'u': 0.0, 'v': 0.0}))
    
    # Create solver
    fluid = Fluid(name="Test fluid", density=1.0, viscosity=0.01)
    cfd = CFDEngine(mesh, fluid)
    
    # Solve
    cfd.solve_SIMPLE(n_iterations=1000)
    
    return cfd

def flow_over_cylinder():
    """Flow over a cylinder (simplified 2D)"""
    # Create mesh
    mesh = CFDMesh(nx=200, ny=80, lx=4.0, ly=1.0)
    
    # Set boundary conditions
    mesh.set_boundary('west', BoundaryCondition(
        BoundaryType.VELOCITY_INLET, {'u': 1.0, 'v': 0.0}))
    mesh.set_boundary('east', BoundaryCondition(
        BoundaryType.PRESSURE_OUTLET, {'p': 0.0}))
    mesh.set_boundary('north', BoundaryCondition(
        BoundaryType.SYMMETRY, {}))
    mesh.set_boundary('south', BoundaryCondition(
        BoundaryType.SYMMETRY, {}))
    
    # Create solver
    fluid = Fluid.air(temperature=20.0)
    cfd = CFDEngine(mesh, fluid)
    
    # Add cylinder (simplified as blocked cells)
    cx, cy = 1.0, 0.5  # Cylinder center
    r = 0.1  # Cylinder radius
    
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            x, y = mesh.x[i], mesh.y[j]
            if (x - cx)**2 + (y - cy)**2 <= r**2:
                cfd.u[j, i] = 0.0
                cfd.v[j, i] = 0.0
    
    return cfd

def channel_flow_heat_transfer():
    """Channel flow with heat transfer"""
    # Create mesh
    mesh = CFDMesh(nx=100, ny=40, lx=2.0, ly=0.1)
    
    # Boundary conditions
    mesh.set_boundary('west', BoundaryCondition(
        BoundaryType.VELOCITY_INLET, {'u': 0.1, 'v': 0.0, 'T': 20.0}))
    mesh.set_boundary('east', BoundaryCondition(
        BoundaryType.PRESSURE_OUTLET, {'p': 0.0}))
    mesh.set_boundary('north', BoundaryCondition(
        BoundaryType.WALL, {'u': 0.0, 'v': 0.0, 'T': 80.0}))  # Heated wall
    mesh.set_boundary('south', BoundaryCondition(
        BoundaryType.WALL, {'u': 0.0, 'v': 0.0, 'T': 20.0}))  # Cold wall
    
    # Create solver
    fluid = Fluid.water(temperature=20.0)
    cfd = CFDEngine(mesh, fluid)
    
    # Initialize temperature field
    cfd.T.fill(20.0)
    
    return cfd