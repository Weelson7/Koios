import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.constants as const
from scipy.special import jv, yv, hankel1
import matplotlib.pyplot as plt
from matplotlib import cm

class FieldType(Enum):
    """Types of electromagnetic fields"""
    ELECTRIC = "ELECTRIC"
    MAGNETIC = "MAGNETIC"
    ELECTROMAGNETIC = "ELECTROMAGNETIC"
    ELECTROSTATIC = "ELECTROSTATIC"
    MAGNETOSTATIC = "MAGNETOSTATIC"

class MaterialType(Enum):
    """Types of electromagnetic materials"""
    DIELECTRIC = "DIELECTRIC"
    CONDUCTOR = "CONDUCTOR"
    MAGNETIC = "MAGNETIC"
    ANISOTROPIC = "ANISOTROPIC"

@dataclass
class EMaterial:
    """Electromagnetic material properties"""
    name: str
    permittivity: float  # Relative permittivity (εr)
    permeability: float  # Relative permeability (μr)
    conductivity: float  # Electrical conductivity (S/m)
    loss_tangent: Optional[float] = 0.0
    
    @property
    def impedance(self) -> float:
        """Characteristic impedance of the material"""
        return np.sqrt(self.permeability * const.mu_0 / 
                      (self.permittivity * const.epsilon_0))
    
    @property
    def refractive_index(self) -> float:
        """Refractive index"""
        return np.sqrt(self.permittivity * self.permeability)
    
    @classmethod
    def vacuum(cls):
        """Vacuum/free space"""
        return cls("Vacuum", 1.0, 1.0, 0.0)
    
    @classmethod
    def air(cls):
        """Air at standard conditions"""
        return cls("Air", 1.00058986, 1.00000037, 0.0)
    
    @classmethod
    def copper(cls):
        """Copper conductor"""
        return cls("Copper", 1.0, 0.999994, 5.96e7)
    
    @classmethod
    def fr4(cls):
        """FR-4 PCB substrate"""
        return cls("FR-4", 4.4, 1.0, 0.0, loss_tangent=0.02)
    
    @classmethod
    def silicon(cls):
        """Silicon semiconductor"""
        return cls("Silicon", 11.9, 1.0, 2.3e-3)

@dataclass
class Source:
    """Electromagnetic source"""
    type: str  # 'voltage', 'current', 'plane_wave', 'dipole'
    position: np.ndarray
    magnitude: complex
    frequency: float
    direction: Optional[np.ndarray] = None

class ElectromagneticsEngine:
    """Main electromagnetics simulation engine"""
    
    def __init__(self, freq: float = 1e9):
        self.frequency = freq
        self.omega = 2 * np.pi * freq
        self.wavelength = const.c / freq
        self.k0 = 2 * np.pi / self.wavelength  # Free space wave number
        
        # Field arrays
        self.Ex = None
        self.Ey = None
        self.Ez = None
        self.Hx = None
        self.Hy = None
        self.Hz = None
        
        # Material distribution
        self.materials: Dict[Tuple[int, int, int], EMaterial] = {}
        
    def create_mesh_3d(self, nx: int, ny: int, nz: int, 
                      dx: float, dy: float, dz: float):
        """Create 3D computational mesh"""
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        # Initialize field components
        self.Ex = np.zeros((nx, ny+1, nz+1), dtype=complex)
        self.Ey = np.zeros((nx+1, ny, nz+1), dtype=complex)
        self.Ez = np.zeros((nx+1, ny+1, nz), dtype=complex)
        self.Hx = np.zeros((nx+1, ny, nz), dtype=complex)
        self.Hy = np.zeros((nx, ny+1, nz), dtype=complex)
        self.Hz = np.zeros((nx, ny, nz+1), dtype=complex)
        
        # Coordinate arrays
        self.x = np.arange(nx) * dx
        self.y = np.arange(ny) * dy
        self.z = np.arange(nz) * dz
        
    def set_material_region(self, x_range: Tuple[int, int], 
                           y_range: Tuple[int, int], 
                           z_range: Tuple[int, int], 
                           material: EMaterial):
        """Set material properties in a region"""
        for i in range(x_range[0], x_range[1]):
            for j in range(y_range[0], y_range[1]):
                for k in range(z_range[0], z_range[1]):
                    self.materials[(i, j, k)] = material
    
    def get_material(self, i: int, j: int, k: int) -> EMaterial:
        """Get material at grid point"""
        return self.materials.get((i, j, k), EMaterial.vacuum())
    
    def fdfd_2d_te(self, nx: int, ny: int, dx: float, dy: float):
        """2D Finite Difference Frequency Domain solver (TE mode)"""
        # Create system matrix for Hz field
        N = nx * ny
        A = np.zeros((N, N), dtype=complex)
        b = np.zeros(N, dtype=complex)
        
        # Fill matrix using finite differences
        for i in range(nx):
            for j in range(ny):
                n = i * ny + j
                
                # Get material properties
                mat = self.get_material(i, j, 0)
                eps_r = mat.permittivity
                mu_r = mat.permeability
                sigma = mat.conductivity
                
                # Complex permittivity
                eps_eff = eps_r - 1j * sigma / (self.omega * const.epsilon_0)
                
                # Wave number in material
                k = self.k0 * np.sqrt(eps_eff * mu_r)
                
                # Finite difference coefficients
                if i > 0:
                    A[n, n - ny] = 1 / dx**2
                if i < nx - 1:
                    A[n, n + ny] = 1 / dx**2
                if j > 0:
                    A[n, n - 1] = 1 / dy**2
                if j < ny - 1:
                    A[n, n + 1] = 1 / dy**2
                
                A[n, n] = -2/dx**2 - 2/dy**2 - k**2
        
        return A, b
    
    def plane_wave(self, E0: float, k_vector: np.ndarray, polarization: np.ndarray):
        """Generate plane wave fields"""
        # E = E0 * p * exp(j * k · r)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    r = np.array([i * self.dx, j * self.dy, k * self.dz])
                    phase = np.exp(1j * np.dot(k_vector, r))
                    
                    E_field = E0 * polarization * phase
                    self.Ex[i, j, k] = E_field[0]
                    self.Ey[i, j, k] = E_field[1]
                    self.Ez[i, j, k] = E_field[2]
    
    def dipole_antenna(self, position: np.ndarray, current: float, 
                      length: float, orientation: str = 'z'):
        """Calculate fields from dipole antenna"""
        # Hertzian dipole approximation for short dipoles
        I0 = current
        dl = length
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    r_obs = np.array([i * self.dx, j * self.dy, k * self.dz])
                    r_vec = r_obs - position
                    r = np.linalg.norm(r_vec)
                    
                    if r < 1e-10:
                        continue
                    
                    # Spherical coordinates
                    theta = np.arccos(r_vec[2] / r)
                    phi = np.arctan2(r_vec[1], r_vec[0])
                    
                    # Far field approximation
                    if r > 10 * self.wavelength:
                        E_theta = (1j * self.k0 * I0 * dl * np.sin(theta) * 
                                  np.exp(-1j * self.k0 * r) / (4 * np.pi * r))
                        
                        # Convert to Cartesian
                        self.Ex[i, j, k] = E_theta * np.cos(theta) * np.cos(phi)
                        self.Ey[i, j, k] = E_theta * np.cos(theta) * np.sin(phi)
                        self.Ez[i, j, k] = -E_theta * np.sin(theta)
    
    def calculate_poynting_vector(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate time-averaged Poynting vector S = 0.5 * Re(E × H*) for phasor fields"""
        # Average fields to common grid points
        Ex_avg = 0.5 * (self.Ex[:-1, :-1, :-1] + self.Ex[1:, :-1, :-1])
        Ey_avg = 0.5 * (self.Ey[:-1, :-1, :-1] + self.Ey[:-1, 1:, :-1])
        Ez_avg = 0.5 * (self.Ez[:-1, :-1, :-1] + self.Ez[:-1, :-1, 1:])
        Hx_avg = 0.5 * (self.Hx[:-1, :-1, :-1] + self.Hx[1:, :-1, :-1])
        Hy_avg = 0.5 * (self.Hy[:-1, :-1, :-1] + self.Hy[:-1, 1:, :-1])
        Hz_avg = 0.5 * (self.Hz[:-1, :-1, :-1] + self.Hz[:-1, :-1, 1:])
        
        # Time-averaged Poynting vector for complex phasor fields: S = 0.5 * Re(E × H*)
        Sx = 0.5 * np.real(Ey_avg * np.conj(Hz_avg) - Ez_avg * np.conj(Hy_avg))
        Sy = 0.5 * np.real(Ez_avg * np.conj(Hx_avg) - Ex_avg * np.conj(Hz_avg))
        Sz = 0.5 * np.real(Ex_avg * np.conj(Hy_avg) - Ey_avg * np.conj(Hx_avg))
        
        return Sx, Sy, Sz
    
    def calculate_power_flow(self, surface: str) -> float:
        """Calculate power flow through a surface"""
        Sx, Sy, Sz = self.calculate_poynting_vector()
        
        if surface == 'x_min':
            power = -np.sum(Sx[0, :, :]) * self.dy * self.dz
        elif surface == 'x_max':
            power = np.sum(Sx[-1, :, :]) * self.dy * self.dz
        elif surface == 'y_min':
            power = -np.sum(Sy[:, 0, :]) * self.dx * self.dz
        elif surface == 'y_max':
            power = np.sum(Sy[:, -1, :]) * self.dx * self.dz
        elif surface == 'z_min':
            power = -np.sum(Sz[:, :, 0]) * self.dx * self.dy
        elif surface == 'z_max':
            power = np.sum(Sz[:, :, -1]) * self.dx * self.dy
        else:
            power = 0.0
        
        return power
    
    def antenna_pattern(self, antenna_pos: np.ndarray, r_sphere: float = 100.0,
                       n_theta: int = 180, n_phi: int = 360) -> np.ndarray:
        """Calculate antenna radiation pattern"""
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        pattern = np.zeros((n_theta, n_phi))
        
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                x = antenna_pos[0] + r_sphere * np.sin(th) * np.cos(ph)
                y = antenna_pos[1] + r_sphere * np.sin(th) * np.sin(ph)
                z = antenna_pos[2] + r_sphere * np.cos(th)
                
                ix = int(x / self.dx)
                iy = int(y / self.dy)
                iz = int(z / self.dz)
                
                if (0 <= ix < self.nx-1 and 0 <= iy < self.ny-1 and 
                    0 <= iz < self.nz-1):
                    E_mag = np.abs(self.Ex[ix, iy, iz])**2 + \
                           np.abs(self.Ey[ix, iy, iz])**2 + \
                           np.abs(self.Ez[ix, iy, iz])**2
                    pattern[i, j] = 10 * np.log10(E_mag + 1e-20)
        
        return pattern
    
    def waveguide_modes(self, a: float, b: float, mode_m: int = 1, mode_n: int = 0):
        """Calculate rectangular waveguide TE/TM modes"""
        kc = np.sqrt((mode_m * np.pi / a)**2 + (mode_n * np.pi / b)**2)
        fc = kc * const.c / (2 * np.pi)
        
        if self.frequency > fc:
            beta = np.sqrt(self.k0**2 - kc**2)
            vp = self.omega / beta
            vg = const.c**2 / vp
            
            return {
                'cutoff_freq': fc,
                'propagation_const': beta,
                'phase_velocity': vp,
                'group_velocity': vg,
                'wavelength_guide': 2 * np.pi / beta
            }
        else:
            return {'cutoff_freq': fc, 'mode': 'Evanescent'}
    
    def s_parameters(self, port1_fields: Dict, port2_fields: Dict) -> np.ndarray:
        """Calculate 2-port S-parameters"""
        # Simplified S-parameter calculation
        # In practice, would use mode matching or field integration
        
        # Incident and reflected waves at ports
        a1 = port1_fields.get('incident', 1.0)
        b1 = port1_fields.get('reflected', 0.0)
        a2 = port2_fields.get('incident', 0.0)
        b2 = port2_fields.get('reflected', 0.0)
        
        # S-parameters (assuming reciprocal network)
        S11 = b1 / a1 if a1 != 0 else 0
        S21 = b2 / a1 if a1 != 0 else 0
        S12 = S21  # Reciprocity
        S22 = b2 / a2 if a2 != 0 else 0
        
        return np.array([[S11, S12], [S21, S22]])
    
    def visualize_fields(self, field_component: str = 'Ex', slice_axis: str = 'z',
                        slice_index: int = None):
        """Visualize electromagnetic fields"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select field component
        field_map = {
            'Ex': self.Ex, 'Ey': self.Ey, 'Ez': self.Ez,
            'Hx': self.Hx, 'Hy': self.Hy, 'Hz': self.Hz
        }
        field = field_map.get(field_component, self.Ex)
        
        # Take slice through 3D field
        if slice_axis == 'z':
            idx = slice_index if slice_index else self.nz // 2
            data = np.abs(field[:, :, idx])
            extent = [0, self.nx * self.dx, 0, self.ny * self.dy]
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
        elif slice_axis == 'y':
            idx = slice_index if slice_index else self.ny // 2
            data = np.abs(field[:, idx, :])
            extent = [0, self.nx * self.dx, 0, self.nz * self.dz]
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Z [m]')
        else:  # x
            idx = slice_index if slice_index else self.nx // 2
            data = np.abs(field[idx, :, :])
            extent = [0, self.ny * self.dy, 0, self.nz * self.dz]
            ax.set_xlabel('Y [m]')
            ax.set_ylabel('Z [m]')
        
        # Plot field magnitude
        im = ax.imshow(data.T, origin='lower', extent=extent, 
                      cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, ax=ax, label=f'|{field_component}| [V/m]')
        
        ax.set_title(f'Electromagnetic Field - {field_component} component')
        
        return fig
    
    def visualize_radiation_pattern(self, pattern: np.ndarray):
        """Visualize antenna radiation pattern"""
        fig = plt.figure(figsize=(12, 5))
        
        # Polar plot (E-plane)
        ax1 = fig.add_subplot(121, projection='polar')
        theta = np.linspace(0, np.pi, pattern.shape[0])
        ax1.plot(theta, pattern[:, 0] - np.min(pattern))
        ax1.set_title('E-plane Pattern')
        ax1.set_theta_zero_location('N')
        
        # Polar plot (H-plane)
        ax2 = fig.add_subplot(122, projection='polar')
        phi = np.linspace(0, 2*np.pi, pattern.shape[1])
        ax2.plot(phi, pattern[pattern.shape[0]//2, :] - np.min(pattern))
        ax2.set_title('H-plane Pattern')
        
        return fig

# Example applications
def microstrip_patch_antenna():
    """Microstrip patch antenna simulation"""
    em = ElectromagneticsEngine(freq=2.4e9)  # 2.4 GHz
    
    # Create mesh (simplified 2D for demonstration)
    em.create_mesh_3d(nx=100, ny=100, nz=20, 
                     dx=0.001, dy=0.001, dz=0.001)  # 1mm resolution
    
    # Substrate (FR-4)
    em.set_material_region((0, 100), (0, 100), (0, 2), EMaterial.fr4())
    
    # Ground plane
    em.set_material_region((0, 100), (0, 100), (0, 1), EMaterial.copper())
    
    # Patch
    patch_x = (30, 70)  # 40mm x 30mm patch
    patch_y = (35, 65)
    em.set_material_region(patch_x, patch_y, (2, 3), EMaterial.copper())
    
    # Feed point
    feed_x, feed_y = 50, 50
    em.dipole_antenna(np.array([feed_x * 0.001, feed_y * 0.001, 0.002]), 
                     current=1.0, length=0.001)
    
    return em

def waveguide_filter():
    """Waveguide band-pass filter design"""
    em = ElectromagneticsEngine(freq=10e9)  # X-band
    
    # WR-90 waveguide dimensions
    a = 0.02286  # 22.86 mm
    b = 0.01016  # 10.16 mm
    
    # Calculate modes
    modes = []
    for m in range(1, 4):
        for n in range(0, 3):
            mode_data = em.waveguide_modes(a, b, m, n)
            if mode_data.get('propagation_const'):
                modes.append({
                    'TE': (m, n),
                    'cutoff': mode_data['cutoff_freq'],
                    'beta': mode_data['propagation_const']
                })
    
    return modes

def pcb_trace_analysis():
    """PCB transmission line analysis"""
    em = ElectromagneticsEngine(freq=1e9)  # 1 GHz
    
    # Create 2D mesh for microstrip
    em.create_mesh_3d(nx=50, ny=50, nz=10,
                     dx=0.0001, dy=0.0001, dz=0.0001)  # 0.1mm resolution
    
    # FR-4 substrate
    em.set_material_region((0, 50), (0, 50), (0, 5), EMaterial.fr4())
    
    # Copper trace (50 ohm line)
    trace_width = 10  # Grid points
    trace_y = (20, 20 + trace_width)
    em.set_material_region((0, 50), trace_y, (5, 6), EMaterial.copper())
    
    return em