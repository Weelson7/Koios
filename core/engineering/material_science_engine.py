import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.constants as const
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CrystalSystem(Enum):
    """Seven crystal systems"""
    CUBIC = "CUBIC"
    TETRAGONAL = "TETRAGONAL"
    ORTHORHOMBIC = "ORTHORHOMBIC"
    HEXAGONAL = "HEXAGONAL"
    TRIGONAL = "TRIGONAL"
    MONOCLINIC = "MONOCLINIC"
    TRICLINIC = "TRICLINIC"

class BravaisLattice(Enum):
    """14 Bravais lattices"""
    SIMPLE_CUBIC = "SC"
    BODY_CENTERED_CUBIC = "BCC"
    FACE_CENTERED_CUBIC = "FCC"
    SIMPLE_TETRAGONAL = "ST"
    BODY_CENTERED_TETRAGONAL = "BCT"
    SIMPLE_ORTHORHOMBIC = "SO"
    BASE_CENTERED_ORTHORHOMBIC = "BCO"
    BODY_CENTERED_ORTHORHOMBIC = "BCCOR"
    FACE_CENTERED_ORTHORHOMBIC = "FCO"
    HEXAGONAL = "HEX"
    RHOMBOHEDRAL = "R"
    SIMPLE_MONOCLINIC = "SM"
    BASE_CENTERED_MONOCLINIC = "BCM"
    TRICLINIC = "TRI"

@dataclass
class Material:
    """Comprehensive material properties"""
    name: str
    # Mechanical properties
    youngs_modulus: float  # GPa
    poissons_ratio: float
    yield_strength: float  # MPa
    ultimate_strength: float  # MPa
    hardness: float  # HV
    
    # Physical properties
    density: float  # g/cm³
    melting_point: float  # K
    thermal_expansion: float  # 10^-6/K
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    
    # Optional properties
    fracture_toughness: Optional[float] = None  # MPa·m^0.5
    electrical_resistivity: Optional[float] = None  # Ω·m
    crystal_system: Optional[CrystalSystem] = None
    lattice_parameter: Optional[Dict[str, float]] = None
    
    @classmethod
    def aluminum_6061(cls):
        """Aluminum 6061-T6 alloy"""
        return cls(
            name="Al 6061-T6",
            youngs_modulus=68.9,
            poissons_ratio=0.33,
            yield_strength=276,
            ultimate_strength=310,
            hardness=107,
            fracture_toughness=29,
            density=2.70,
            melting_point=855,
            thermal_expansion=23.6,
            thermal_conductivity=167,
            specific_heat=896,
            electrical_resistivity=3.99e-8,
            crystal_system=CrystalSystem.CUBIC,
            lattice_parameter={'a': 4.05e-10}
        )
    
    @classmethod
    def steel_aisi_4140(cls):
        """AISI 4140 steel"""
        return cls(
            name="AISI 4140 Steel",
            youngs_modulus=205,
            poissons_ratio=0.29,
            yield_strength=655,
            ultimate_strength=1020,
            hardness=302,
            fracture_toughness=80,
            density=7.85,
            melting_point=1698,
            thermal_expansion=12.3,
            thermal_conductivity=42.6,
            specific_heat=473,
            electrical_resistivity=2.2e-7,
            crystal_system=CrystalSystem.CUBIC,
            lattice_parameter={'a': 2.87e-10}
        )
    
    @classmethod
    def titanium_grade5(cls):
        """Ti-6Al-4V (Grade 5) titanium alloy"""
        return cls(
            name="Ti-6Al-4V",
            youngs_modulus=113.8,
            poissons_ratio=0.342,
            yield_strength=880,
            ultimate_strength=950,
            hardness=334,
            fracture_toughness=75,
            density=4.43,
            melting_point=1933,
            thermal_expansion=8.6,
            thermal_conductivity=6.7,
            specific_heat=526,
            electrical_resistivity=1.7e-6,
            crystal_system=CrystalSystem.HEXAGONAL,
            lattice_parameter={'a': 2.95e-10, 'c': 4.68e-10}
        )
    
    @classmethod
    def copper_c101(cls):
        """Oxygen-free electronic copper"""
        return cls(
            name="Copper C101",
            youngs_modulus=110,
            poissons_ratio=0.34,
            yield_strength=69,
            ultimate_strength=220,
            hardness=45,
            fracture_toughness=120,
            density=8.96,
            melting_point=1358,
            thermal_expansion=16.5,
            thermal_conductivity=401,
            specific_heat=385,
            electrical_resistivity=1.7e-8,
            crystal_system=CrystalSystem.CUBIC,
            lattice_parameter={'a': 3.61e-10}
        )
    
    @classmethod
    def inconel_718(cls):
        """Inconel 718 superalloy"""
        return cls(
            name="Inconel 718",
            youngs_modulus=204.9,
            poissons_ratio=0.294,
            yield_strength=1034,
            ultimate_strength=1275,
            hardness=331,
            fracture_toughness=110,
            density=8.19,
            melting_point=1609,
            thermal_expansion=13.0,
            thermal_conductivity=11.4,
            specific_heat=435,
            electrical_resistivity=1.25e-6,
            crystal_system=CrystalSystem.CUBIC,
            lattice_parameter={'a': 3.59e-10}
        )
    
    @classmethod
    def carbon_fiber_t300(cls):
        """T300 carbon fiber composite"""
        return cls(
            name="Carbon Fiber T300",
            youngs_modulus=230,
            poissons_ratio=0.20,
            yield_strength=3530,
            ultimate_strength=3530,
            hardness=0,  # Not applicable
            fracture_toughness=25,
            density=1.76,
            melting_point=3873,  # Sublimation point
            thermal_expansion=-0.5,
            thermal_conductivity=7.0,
            specific_heat=710,
            electrical_resistivity=1.6e-5,
            crystal_system=CrystalSystem.HEXAGONAL,
            lattice_parameter={'a': 2.46e-10, 'c': 6.70e-10}
        )
    
    @classmethod
    def silicon_carbide(cls):
        """Silicon carbide ceramic"""
        return cls(
            name="Silicon Carbide",
            youngs_modulus=410,
            poissons_ratio=0.14,
            yield_strength=0,  # Brittle material
            ultimate_strength=550,
            hardness=2500,
            fracture_toughness=4.6,
            density=3.21,
            melting_point=3003,
            thermal_expansion=4.0,
            thermal_conductivity=120,
            specific_heat=675,
            electrical_resistivity=1e3,  # Variable, can be conductive
            crystal_system=CrystalSystem.HEXAGONAL,
            lattice_parameter={'a': 3.08e-10, 'c': 15.12e-10}
        )
    
    @classmethod
    def magnesium_az31(cls):
        """AZ31 magnesium alloy"""
        return cls(
            name="Magnesium AZ31",
            youngs_modulus=45,
            poissons_ratio=0.35,
            yield_strength=200,
            ultimate_strength=260,
            hardness=49,
            fracture_toughness=15,
            density=1.77,
            melting_point=893,
            thermal_expansion=26.0,
            thermal_conductivity=96,
            specific_heat=1024,
            electrical_resistivity=9.2e-8,
            crystal_system=CrystalSystem.HEXAGONAL,
            lattice_parameter={'a': 3.21e-10, 'c': 5.21e-10}
        )

@dataclass
class CompositeLayer:
    """Layer in a composite material"""
    material: Material
    thickness: float  # m
    angle: float  # degrees (fiber orientation)
    volume_fraction: float  # fiber volume fraction

class MaterialScienceEngine:
    """Main material science computational engine"""
    
    def __init__(self):
        self.materials_database = {
            'Al6061': Material.aluminum_6061(),
            'AISI4140': Material.steel_aisi_4140(),
            'Ti6Al4V': Material.titanium_grade5()
        }
    
    def add_material(self, key: str, material: Material):
        """Add material to database"""
        self.materials_database[key] = material
    
    def calculate_elastic_constants(self, material: Material) -> Dict[str, float]:
        """Calculate all elastic constants from E and ν"""
        E = material.youngs_modulus * 1e9  # Convert to Pa
        nu = material.poissons_ratio
        
        # Shear modulus
        G = E / (2 * (1 + nu))
        
        # Bulk modulus
        K = E / (3 * (1 - 2 * nu))
        
        # Lamé constants
        lambda_lame = nu * E / ((1 + nu) * (1 - 2 * nu))
        mu_lame = G
        
        return {
            'shear_modulus': G / 1e9,  # GPa
            'bulk_modulus': K / 1e9,   # GPa
            'lame_lambda': lambda_lame / 1e9,  # GPa
            'lame_mu': mu_lame / 1e9   # GPa
        }
    
    def stress_strain_curve(self, material: Material, 
                           max_strain: float = 0.1,
                           n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stress-strain curve with Ramberg-Osgood model"""
        E = material.youngs_modulus * 1e3  # Convert to MPa
        sigma_y = material.yield_strength
        
        # Ramberg-Osgood parameters (typical values)
        n = 5  # Strain hardening exponent
        K = sigma_y / (0.002**n)  # Strength coefficient
        
        strain = np.linspace(0, max_strain, n_points)
        stress = np.zeros_like(strain)
        
        for i, eps in enumerate(strain):
            if eps * E <= sigma_y:
                # Elastic region
                stress[i] = E * eps
            else:
                # Plastic region (Ramberg-Osgood)
                # Solve: eps = sigma/E + (sigma/K)^(1/n)
                def equation(sig):
                    return sig/E + (sig/K)**(1/n) - eps
                
                from scipy.optimize import fsolve
                stress[i] = fsolve(equation, sigma_y)[0]
        
        return strain, stress
    
    def crystal_structure_analysis(self, lattice: BravaisLattice, 
                                 lattice_params: Dict[str, float]) -> Dict[str, Any]:
        """Analyze crystal structure properties"""
        a = lattice_params.get('a', 1e-10)
        b = lattice_params.get('b', a)
        c = lattice_params.get('c', a)
        alpha = lattice_params.get('alpha', 90) * np.pi / 180
        beta = lattice_params.get('beta', 90) * np.pi / 180
        gamma = lattice_params.get('gamma', 90) * np.pi / 180
        
        # Calculate unit cell volume
        if lattice in [BravaisLattice.SIMPLE_CUBIC, BravaisLattice.BODY_CENTERED_CUBIC, 
                      BravaisLattice.FACE_CENTERED_CUBIC]:
            volume = a**3
        elif lattice == BravaisLattice.HEXAGONAL:
            volume = np.sqrt(3) / 2 * a**2 * c
        else:
            # General formula for triclinic
            volume = a * b * c * np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) 
                                        - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2)
        
        # Calculate atomic packing factor
        if lattice == BravaisLattice.SIMPLE_CUBIC:
            atoms_per_cell = 1
            apf = np.pi / 6
        elif lattice == BravaisLattice.BODY_CENTERED_CUBIC:
            atoms_per_cell = 2
            apf = np.pi * np.sqrt(3) / 8
        elif lattice == BravaisLattice.FACE_CENTERED_CUBIC:
            atoms_per_cell = 4
            apf = np.pi / (3 * np.sqrt(2))
        elif lattice == BravaisLattice.HEXAGONAL:
            atoms_per_cell = 6
            apf = np.pi / (3 * np.sqrt(2))  # For ideal c/a ratio
        else:
            atoms_per_cell = 1
            apf = None
        
        # Calculate density if atomic mass is known
        # d = n * M / (N_A * V)
        
        return {
            'volume': volume,
            'atoms_per_cell': atoms_per_cell,
            'atomic_packing_factor': apf,
            'lattice_type': lattice.value
        }
    
    def miller_indices_spacing(self, h: int, k: int, l: int, 
                             crystal_system: CrystalSystem,
                             lattice_params: Dict[str, float]) -> float:
        """Calculate d-spacing for Miller indices"""
        a = lattice_params.get('a', 1e-10)
        b = lattice_params.get('b', a)
        c = lattice_params.get('c', a)
        
        if crystal_system == CrystalSystem.CUBIC:
            d = a / np.sqrt(h**2 + k**2 + l**2)
        elif crystal_system == CrystalSystem.TETRAGONAL:
            d = 1 / np.sqrt((h**2 + k**2)/a**2 + l**2/c**2)
        elif crystal_system == CrystalSystem.ORTHORHOMBIC:
            d = 1 / np.sqrt(h**2/a**2 + k**2/b**2 + l**2/c**2)
        elif crystal_system == CrystalSystem.HEXAGONAL:
            d = 1 / np.sqrt(4/3 * (h**2 + h*k + k**2)/a**2 + l**2/c**2)
        else:
            # Simplified for other systems
            d = a / np.sqrt(h**2 + k**2 + l**2)
        
        return d
    
    def composite_laminate_analysis(self, layers: List[CompositeLayer]) -> Dict[str, Any]:
        """Classical Laminate Theory (CLT) analysis"""
        # Initialize ABD matrix
        A = np.zeros((3, 3))  # Extensional stiffness
        B = np.zeros((3, 3))  # Coupling stiffness
        D = np.zeros((3, 3))  # Bending stiffness
        
        # Calculate z-coordinates
        total_thickness = sum(layer.thickness for layer in layers)
        z = [-total_thickness/2]
        for layer in layers:
            z.append(z[-1] + layer.thickness)
        
        # Build ABD matrix
        for i, layer in enumerate(layers):
            # Get material properties
            E1 = layer.material.youngs_modulus * 1e9
            E2 = E1 / 10  # Transverse modulus (typical for composites)
            G12 = E1 / (2 * (1 + layer.material.poissons_ratio))
            nu12 = layer.material.poissons_ratio
            
            # Reduced stiffness matrix
            Q11 = E1 / (1 - nu12**2 * E2/E1)
            Q12 = nu12 * E2 / (1 - nu12**2 * E2/E1)
            Q22 = E2 / (1 - nu12**2 * E2/E1)
            Q66 = G12
            
            # Transformation matrix for angle
            theta = layer.angle * np.pi / 180
            c = np.cos(theta)
            s = np.sin(theta)
            
            # Transformed reduced stiffness
            Q_bar = np.array([
                [Q11*c**4 + Q22*s**4 + 2*(Q12 + 2*Q66)*s**2*c**2,
                 (Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(s**4 + c**4),
                 (Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c],
                [(Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(s**4 + c**4),
                 Q11*s**4 + Q22*c**4 + 2*(Q12 + 2*Q66)*s**2*c**2,
                 (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3],
                [(Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c,
                 (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3,
                 (Q11 + Q22 - 2*Q12 - 2*Q66)*s**2*c**2 + Q66*(s**4 + c**4)]
            ])
            
            # Add to ABD matrix
            z1, z2 = z[i], z[i+1]
            A += Q_bar * (z2 - z1)
            B += Q_bar * (z2**2 - z1**2) / 2
            D += Q_bar * (z2**3 - z1**3) / 3
        
        # Calculate effective properties
        h = total_thickness
        Ex = (A[0,0] * A[1,1] - A[0,1]**2) / (A[1,1] * h)
        Ey = (A[0,0] * A[1,1] - A[0,1]**2) / (A[0,0] * h)
        Gxy = A[2,2] / h
        nuxy = A[0,1] / A[1,1]
        
        return {
            'A_matrix': A,
            'B_matrix': B,
            'D_matrix': D,
            'Ex_effective': Ex / 1e9,  # GPa
            'Ey_effective': Ey / 1e9,  # GPa
            'Gxy_effective': Gxy / 1e9,  # GPa
            'nuxy_effective': nuxy,
            'total_thickness': total_thickness
        }
    
    def fatigue_life_prediction(self, material: Material, stress_amplitude: float,
                              mean_stress: float = 0, surface_finish: str = 'polished') -> float:
        """Predict fatigue life using S-N curve approach"""
        # Estimate endurance limit
        Su = material.ultimate_strength
        
        # Marin equation factors
        # Surface finish factor
        surface_factors = {
            'polished': 1.0,
            'ground': 0.9,
            'machined': 0.8,
            'hot_rolled': 0.7,
            'forged': 0.6
        }
        ka = surface_factors.get(surface_finish, 0.8)
        
        # Size factor (assuming 10mm diameter)
        kb = 0.9
        
        # Loading factor (bending)
        kc = 1.0
        
        # Temperature factor
        kd = 1.0
        
        # Reliability factor (90%)
        ke = 0.897
        
        # Miscellaneous factor
        kf = 1.0
        
        # Modified endurance limit
        Se_prime = 0.5 * Su  # Unmodified endurance limit
        Se = Se_prime * ka * kb * kc * kd * ke * kf
        
        # Goodman relation for mean stress correction
        if mean_stress != 0:
            stress_amplitude_eq = stress_amplitude / (1 - mean_stress / Su)
        else:
            stress_amplitude_eq = stress_amplitude
        
        # S-N curve (simplified Basquin's equation)
        if stress_amplitude_eq <= Se:
            Nf = 1e9  # Infinite life
        else:
            # Fatigue strength coefficient
            sigma_f = 0.9 * Su
            # Fatigue strength exponent
            b = -0.085
            
            Nf = (stress_amplitude_eq / sigma_f) ** (1/b) / 2
        
        return Nf
    
    def fracture_mechanics_analysis(self, material: Material, crack_length: float,
                                  stress: float, geometry: str = 'center_crack') -> Dict[str, float]:
        """Linear Elastic Fracture Mechanics (LEFM) analysis"""
        a = crack_length
        sigma = stress
        
        # Geometry factors for stress intensity factor
        geometry_factors = {
            'center_crack': lambda a, W: np.sqrt(np.pi * a),
            'edge_crack': lambda a, W: 1.12 * np.sqrt(np.pi * a),
            'penny_crack': lambda a, W: (2/np.pi) * np.sqrt(np.pi * a)
        }
        
        # Calculate stress intensity factor
        if geometry in geometry_factors:
            K_I = sigma * geometry_factors[geometry](a, None)
        else:
            K_I = sigma * np.sqrt(np.pi * a)  # Default
        
        # Critical stress intensity (fracture toughness)
        K_IC = material.fracture_toughness if material.fracture_toughness else 50  # MPa·m^0.5
        
        # Factor of safety
        FOS = K_IC / K_I
        
        # Critical crack length
        a_critical = (K_IC / (sigma * np.sqrt(np.pi)))**2
        
        # Crack tip plastic zone size
        r_y = (1 / (2 * np.pi)) * (K_I / material.yield_strength)**2
        
        return {
            'stress_intensity_factor': K_I,
            'critical_SIF': K_IC,
            'factor_of_safety': FOS,
            'critical_crack_length': a_critical,
            'plastic_zone_size': r_y,
            'will_fracture': K_I >= K_IC
        }
    
    def material_selection(self, requirements: Dict[str, Any], 
                         weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """Multi-criteria material selection using weighted scoring"""
        if weights is None:
            weights = {
                'strength': 0.3,
                'stiffness': 0.2,
                'density': 0.2,
                'cost': 0.2,
                'corrosion': 0.1
            }
        
        scores = {}
        
        for name, material in self.materials_database.items():
            score = 0
            
            # Strength score (higher is better)
            if 'min_yield_strength' in requirements:
                if material.yield_strength >= requirements['min_yield_strength']:
                    score += weights.get('strength', 0) * (material.yield_strength / 1000)
            
            # Stiffness score (higher is better)
            if 'min_modulus' in requirements:
                if material.youngs_modulus >= requirements['min_modulus']:
                    score += weights.get('stiffness', 0) * (material.youngs_modulus / 200)
            
            # Density score (lower is better)
            if 'max_density' in requirements:
                if material.density <= requirements['max_density']:
                    score += weights.get('density', 0) * (10 / material.density)
            
            # Cost score (simplified - based on material type)
            cost_factors = {
                'Al': 0.8,
                'Steel': 1.0,
                'Ti': 0.3
            }
            for key, factor in cost_factors.items():
                if key in name:
                    score += weights.get('cost', 0) * factor
            
            scores[name] = score
        
        # Sort by score
        ranked_materials = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_materials
    
    def visualize_crystal_structure(self, lattice: BravaisLattice, 
                                   lattice_params: Dict[str, float]):
        """Visualize crystal structure"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        a = lattice_params.get('a', 1)
        b = lattice_params.get('b', a)
        c = lattice_params.get('c', a)
        
        # Define atom positions based on lattice type
        if lattice == BravaisLattice.SIMPLE_CUBIC:
            positions = [[0, 0, 0]]
        elif lattice == BravaisLattice.BODY_CENTERED_CUBIC:
            positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
        elif lattice == BravaisLattice.FACE_CENTERED_CUBIC:
            positions = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        else:
            positions = [[0, 0, 0]]
        
        # Plot atoms
        for pos in positions:
            ax.scatter(pos[0]*a, pos[1]*b, pos[2]*c, s=300, c='red', alpha=0.8)
        
        # Plot unit cell edges
        edges = [
            [[0, 0, 0], [a, 0, 0]], [[0, 0, 0], [0, b, 0]], [[0, 0, 0], [0, 0, c]],
            [[a, 0, 0], [a, b, 0]], [[a, 0, 0], [a, 0, c]],
            [[0, b, 0], [a, b, 0]], [[0, b, 0], [0, b, c]],
            [[0, 0, c], [a, 0, c]], [[0, 0, c], [0, b, c]],
            [[a, b, 0], [a, b, c]], [[a, 0, c], [a, b, c]], [[0, b, c], [a, b, c]]
        ]
        
        for edge in edges:
            ax.plot([edge[0][0], edge[1][0]], 
                   [edge[0][1], edge[1][1]], 
                   [edge[0][2], edge[1][2]], 'b-', alpha=0.6)
        
        ax.set_xlabel('a [Å]')
        ax.set_ylabel('b [Å]')
        ax.set_zlabel('c [Å]')
        ax.set_title(f'{lattice.value} Crystal Structure')
        
        return fig
    
    def visualize_stress_strain(self, materials: List[Material]):
        """Compare stress-strain curves for multiple materials"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for material in materials:
            strain, stress = self.stress_strain_curve(material, max_strain=0.05)
            ax.plot(strain * 100, stress, label=material.name, linewidth=2)
        
        ax.set_xlabel('Strain [%]')
        ax.set_ylabel('Stress [MPa]')
        ax.set_title('Stress-Strain Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

# Example applications
def aerospace_material_selection():
    """Select materials for aerospace application"""
    engine = MaterialScienceEngine()
    
    requirements = {
        'min_yield_strength': 500,  # MPa
        'min_modulus': 100,  # GPa
        'max_density': 5.0,  # g/cm³
        'temperature_range': (233, 423)  # K
    }
    
    weights = {
        'strength': 0.35,
        'stiffness': 0.25,
        'density': 0.30,
        'cost': 0.10
    }
    
    selected = engine.material_selection(requirements, weights)
    return selected

def composite_wing_analysis():
    """Analyze composite wing structure"""
    engine = MaterialScienceEngine()
    
    # Carbon fiber composite layers
    cf_material = Material(
        name="Carbon Fiber Composite",
        youngs_modulus=230,  # GPa
        poissons_ratio=0.3,
        yield_strength=2000,
        ultimate_strength=2500,
        hardness=500,
        density=1.6,
        melting_point=3900,
        thermal_expansion=0.5,
        thermal_conductivity=5,
        specific_heat=710
    )
    
    # Define laminate layup [0/45/-45/90]s
    layers = [
        CompositeLayer(cf_material, 0.0002, 0, 0.6),
        CompositeLayer(cf_material, 0.0002, 45, 0.6),
        CompositeLayer(cf_material, 0.0002, -45, 0.6),
        CompositeLayer(cf_material, 0.0002, 90, 0.6),
        CompositeLayer(cf_material, 0.0002, 90, 0.6),
        CompositeLayer(cf_material, 0.0002, -45, 0.6),
        CompositeLayer(cf_material, 0.0002, 45, 0.6),
        CompositeLayer(cf_material, 0.0002, 0, 0.6)
    ]
    
    results = engine.composite_laminate_analysis(layers)
    return results

def fatigue_analysis_example():
    """Analyze fatigue life of a component"""
    engine = MaterialScienceEngine()
    material = Material.aluminum_6061()
    
    # Cyclic loading conditions
    stress_amplitude = 150  # MPa
    mean_stress = 50  # MPa
    
    life = engine.fatigue_life_prediction(
        material, stress_amplitude, mean_stress, 'machined'
    )
    
    return {
        'cycles_to_failure': life,
        'years_at_1Hz': life / (365 * 24 * 3600)
    }