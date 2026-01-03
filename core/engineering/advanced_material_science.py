import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from utils.exceptions import (
    InvalidInputError, ConfigurationError, DomainError, NumericalInstabilityError
)

class MaterialType(Enum):
    """Material classification"""
    METAL = "METAL"
    POLYMER = "POLYMER"
    CERAMIC = "CERAMIC"
    COMPOSITE = "COMPOSITE"
    SMART_MATERIAL = "SMART_MATERIAL"

class FailureCriterion(Enum):
    """Failure criteria for materials"""
    VON_MISES = "VON_MISES"
    TRESCA = "TRESCA"
    MOHR_COULOMB = "MOHR_COULOMB"
    TSAI_WU = "TSAI_WU"  # For composites
    MAXIMUM_STRESS = "MAXIMUM_STRESS"

@dataclass
class MaterialProperties:
    """Comprehensive material properties"""
    name: str
    material_type: MaterialType
    # Mechanical properties
    youngs_modulus: float  # Pa
    poissons_ratio: float
    yield_strength: float  # Pa
    ultimate_strength: float  # Pa
    density: float  # kg/m³
    # Thermal properties
    thermal_expansion: float = 0.0  # 1/K
    thermal_conductivity: float = 0.0  # W/(m·K)
    specific_heat: float = 0.0  # J/(kg·K)
    melting_point: float = 0.0  # K
    # Fatigue properties
    fatigue_strength_coefficient: Optional[float] = None  # Pa
    fatigue_strength_exponent: Optional[float] = None
    fatigue_ductility_coefficient: Optional[float] = None
    fatigue_ductility_exponent: Optional[float] = None
    # Fracture properties
    fracture_toughness: Optional[float] = None  # Pa·√m
    critical_stress_intensity: Optional[float] = None  # Pa·√m
    # Creep properties
    creep_activation_energy: Optional[float] = None  # J/mol
    creep_exponent: Optional[float] = None
    
    @classmethod
    def steel_4140(cls):
        """AISI 4140 alloy steel properties"""
        return cls(
            name="AISI 4140 Steel",
            material_type=MaterialType.METAL,
            youngs_modulus=205e9,
            poissons_ratio=0.29,
            yield_strength=415e6,
            ultimate_strength=655e6,
            density=7850,
            thermal_expansion=12.3e-6,
            thermal_conductivity=42.6,
            specific_heat=473,
            melting_point=1698,
            fatigue_strength_coefficient=1.5 * 655e6,
            fatigue_strength_exponent=-0.12,
            fracture_toughness=80e6 * np.sqrt(0.001)
        )
    
    @classmethod
    def aluminum_6061_t6(cls):
        """Aluminum 6061-T6 properties"""
        return cls(
            name="Aluminum 6061-T6",
            material_type=MaterialType.METAL,
            youngs_modulus=68.9e9,
            poissons_ratio=0.33,
            yield_strength=276e6,
            ultimate_strength=310e6,
            density=2700,
            thermal_expansion=23.6e-6,
            thermal_conductivity=167,
            specific_heat=896,
            melting_point=855,
            fatigue_strength_coefficient=500e6,
            fatigue_strength_exponent=-0.106,
            fracture_toughness=29e6 * np.sqrt(0.001)
        )
    
    @classmethod
    def carbon_fiber_epoxy(cls):
        """Carbon fiber/epoxy composite properties (unidirectional)"""
        return cls(
            name="Carbon Fiber/Epoxy Composite",
            material_type=MaterialType.COMPOSITE,
            youngs_modulus=130e9,  # Longitudinal
            poissons_ratio=0.3,
            yield_strength=1500e6,  # Longitudinal
            ultimate_strength=2000e6,  # Longitudinal
            density=1600,
            thermal_expansion=-0.9e-6,  # Longitudinal
            thermal_conductivity=7,
            specific_heat=1200
        )

class AdvancedMaterialScience:
    """Advanced material science analysis engine"""
    
    def __init__(self):
        self.gas_constant = 8.314  # J/(mol·K)
        
    def stress_strain_curve(self, material: MaterialProperties, 
                          max_strain: float = 0.2, 
                          n_points: int = 1000,
                          model: str = 'bilinear') -> Dict[str, np.ndarray]:
        """Generate stress-strain curve for material"""
        strain = np.linspace(0, max_strain, n_points)
        stress = np.zeros_like(strain)
        
        E = material.youngs_modulus
        yield_strain = material.yield_strength / E
        
        if model == 'bilinear':
            # Elastic-perfectly plastic
            for i, eps in enumerate(strain):
                if eps <= yield_strain:
                    stress[i] = E * eps
                else:
                    stress[i] = material.yield_strength
                    
        elif model == 'ramberg_osgood':
            # Ramberg-Osgood model
            n = 0.15  # Hardening exponent
            K = material.yield_strength / (0.002 ** n)
            for i, eps in enumerate(strain):
                # eps = sigma/E + (sigma/K)^(1/n)
                # Solve iteratively
                sigma = E * eps  # Initial guess
                for _ in range(10):
                    f = sigma/E + (sigma/K)**(1/n) - eps
                    df = 1/E + (1/n) * (sigma/K)**(1/n - 1) / K
                    sigma = sigma - f/df
                stress[i] = sigma
                
        elif model == 'johnson_cook':
            # Johnson-Cook model (simplified, without strain rate and temperature)
            A = material.yield_strength
            B = 0.2 * A  # Hardening coefficient
            n = 0.3  # Hardening exponent
            for i, eps in enumerate(strain):
                if eps <= yield_strain:
                    stress[i] = E * eps
                else:
                    plastic_strain = eps - yield_strain
                    stress[i] = A + B * plastic_strain**n
        
        return {
            'strain': strain,
            'stress': stress,
            'elastic_modulus': E,
            'yield_point': (yield_strain, material.yield_strength)
        }
    
    def fatigue_life_prediction(self, material: MaterialProperties,
                              stress_amplitude: float,
                              mean_stress: float,
                              temperature: float = 293.15) -> Dict[str, Any]:
        """Predict fatigue life using various models"""
        # Basquin's equation for high-cycle fatigue
        if material.fatigue_strength_coefficient and material.fatigue_strength_exponent:
            S_f = material.fatigue_strength_coefficient
            b = material.fatigue_strength_exponent
            
            # Goodman mean stress correction
            S_u = material.ultimate_strength
            corrected_amplitude_goodman = stress_amplitude / (1 - mean_stress/S_u)
            
            # Gerber mean stress correction
            corrected_amplitude_gerber = stress_amplitude / (1 - (mean_stress/S_u)**2)
            
            # Soderberg mean stress correction
            S_y = material.yield_strength
            corrected_amplitude_soderberg = stress_amplitude / (1 - mean_stress/S_y)
            
            # Calculate life for each correction
            N_goodman = 0.5 * (corrected_amplitude_goodman / S_f) ** (1/b)
            N_gerber = 0.5 * (corrected_amplitude_gerber / S_f) ** (1/b)
            N_soderberg = 0.5 * (corrected_amplitude_soderberg / S_f) ** (1/b)
            
            # Safety factor
            safety_factor = min(N_goodman, N_gerber, N_soderberg) / 1e6
            
            return {
                'life_cycles_goodman': N_goodman,
                'life_cycles_gerber': N_gerber,
                'life_cycles_soderberg': N_soderberg,
                'stress_amplitude': stress_amplitude,
                'mean_stress': mean_stress,
                'safety_factor': safety_factor,
                'endurance_limit': S_f * (2 * 1e6) ** b  # At 2 million cycles
            }
        else:
            return {
                'error': 'Fatigue properties not available for this material'
            }
    
    def creep_analysis(self, material: MaterialProperties,
                      stress: float,
                      temperature: float,
                      time_hours: float) -> Dict[str, Any]:
        """Analyze creep behavior using power law model"""
        if material.creep_activation_energy and material.creep_exponent:
            Q = material.creep_activation_energy
            n = material.creep_exponent
            R = self.gas_constant
            
            # Norton's power law creep
            # ε̇ = A * σ^n * exp(-Q/RT)
            A = 1e-10  # Material constant (would be determined experimentally)
            
            strain_rate = A * (stress ** n) * np.exp(-Q / (R * temperature))
            total_strain = strain_rate * time_hours * 3600  # Convert to seconds
            
            # Larson-Miller parameter
            C = 20  # Material constant
            LMP = temperature * (C + np.log10(time_hours))
            
            # Time to rupture estimation
            # Simplified model - would use actual material data
            stress_ratio = stress / material.ultimate_strength
            if stress_ratio > 0.5:
                time_to_rupture = 1000 / (stress_ratio ** 3)  # Hours
            else:
                time_to_rupture = float('inf')
            
            return {
                'strain_rate': strain_rate,
                'total_strain': total_strain,
                'larson_miller_parameter': LMP,
                'time_to_rupture_hours': time_to_rupture,
                'temperature': temperature,
                'stress': stress
            }
        else:
            return {
                'error': 'Creep properties not available for this material'
            }
    
    def fracture_mechanics_analysis(self, material: MaterialProperties,
                                  crack_length: float,
                                  applied_stress: float,
                                  geometry_factor: float = np.pi) -> Dict[str, Any]:
        """Analyze fracture mechanics parameters"""
        if material.fracture_toughness:
            # Stress intensity factor
            K_I = geometry_factor * applied_stress * np.sqrt(crack_length)
            
            # Critical crack length
            K_IC = material.fracture_toughness
            a_critical = (K_IC / (geometry_factor * applied_stress)) ** 2
            
            # Factor of safety
            FOS = K_IC / K_I if K_I > 0 else float('inf')
            
            # J-integral (simplified elastic case)
            E = material.youngs_modulus
            nu = material.poissons_ratio
            E_prime = E / (1 - nu**2)  # Plane strain
            J = K_I**2 / E_prime
            
            # Crack growth rate (Paris law)
            # da/dN = C * (ΔK)^m
            C = 1e-11  # Material constant
            m = 3.0    # Material constant
            if K_I > 0:
                crack_growth_rate = C * (K_I ** m)
            else:
                crack_growth_rate = 0
            
            return {
                'stress_intensity_factor': K_I,
                'critical_stress_intensity': K_IC,
                'critical_crack_length': a_critical,
                'factor_of_safety': FOS,
                'j_integral': J,
                'crack_growth_rate': crack_growth_rate,
                'will_fracture': K_I >= K_IC
            }
        else:
            return {
                'error': 'Fracture toughness not available for this material'
            }
    
    def composite_analysis(self, fiber_props: MaterialProperties,
                         matrix_props: MaterialProperties,
                         fiber_volume_fraction: float,
                         loading_angle: float = 0) -> Dict[str, Any]:
        """Analyze composite material properties using rule of mixtures"""
        Vf = fiber_volume_fraction
        Vm = 1 - Vf
        
        # Longitudinal properties (parallel to fibers)
        E_longitudinal = Vf * fiber_props.youngs_modulus + Vm * matrix_props.youngs_modulus
        
        # Transverse properties (perpendicular to fibers)
        E_transverse = 1 / (Vf/fiber_props.youngs_modulus + Vm/matrix_props.youngs_modulus)
        
        # Shear modulus (Halpin-Tsai equations)
        Gf = fiber_props.youngs_modulus / (2 * (1 + fiber_props.poissons_ratio))
        Gm = matrix_props.youngs_modulus / (2 * (1 + matrix_props.poissons_ratio))
        xi = 2  # Reinforcement geometry parameter
        eta = ((Gf/Gm) - 1) / ((Gf/Gm) + xi)
        G12 = Gm * (1 + xi * eta * Vf) / (1 - eta * Vf)
        
        # Strength properties
        tensile_strength_longitudinal = Vf * fiber_props.ultimate_strength + Vm * matrix_props.ultimate_strength
        
        # Transformed properties at angle
        theta = np.radians(loading_angle)
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Transformed modulus
        E_x = 1 / (c**4/E_longitudinal + s**4/E_transverse + c**2*s**2*(1/G12 - 2*0.3/E_longitudinal))
        
        # Tsai-Wu failure criterion coefficients
        F1 = 1/fiber_props.ultimate_strength - 1/abs(fiber_props.ultimate_strength)
        F11 = 1/(fiber_props.ultimate_strength * abs(fiber_props.ultimate_strength))
        
        return {
            'longitudinal_modulus': E_longitudinal,
            'transverse_modulus': E_transverse,
            'shear_modulus': G12,
            'tensile_strength_longitudinal': tensile_strength_longitudinal,
            'modulus_at_angle': E_x,
            'density': Vf * fiber_props.density + Vm * matrix_props.density,
            'fiber_volume_fraction': Vf,
            'specific_modulus': E_longitudinal / (Vf * fiber_props.density + Vm * matrix_props.density),
            'specific_strength': tensile_strength_longitudinal / (Vf * fiber_props.density + Vm * matrix_props.density)
        }
    
    def phase_diagram_binary(self, component_A: str, component_B: str,
                           temperature_range: Tuple[float, float],
                           composition_range: Tuple[float, float] = (0, 1)) -> Dict[str, Any]:
        """Generate binary phase diagram data (simplified model)"""
        # This is a simplified eutectic system model
        # Real phase diagrams would use thermodynamic databases
        
        T_melt_A = 1000  # K
        T_melt_B = 1200  # K
        T_eutectic = 800  # K
        X_eutectic = 0.4  # Composition at eutectic
        
        compositions = np.linspace(composition_range[0], composition_range[1], 100)
        
        # Liquidus line (simplified)
        liquidus = np.zeros_like(compositions)
        for i, X_B in enumerate(compositions):
            if X_B < X_eutectic:
                liquidus[i] = T_melt_A - (T_melt_A - T_eutectic) * X_B / X_eutectic
            else:
                liquidus[i] = T_melt_B - (T_melt_B - T_eutectic) * (1 - X_B) / (1 - X_eutectic)
        
        # Solidus line
        solidus = np.full_like(compositions, T_eutectic)
        
        return {
            'compositions': compositions,
            'liquidus': liquidus,
            'solidus': solidus,
            'eutectic_temperature': T_eutectic,
            'eutectic_composition': X_eutectic,
            'component_A': component_A,
            'component_B': component_B
        }
    
    def hardness_conversion(self, hardness_value: float, 
                          from_scale: str, to_scale: str) -> float:
        """Convert between different hardness scales"""
        # Conversion factors (approximate)
        conversions = {
            ('HB', 'HV'): lambda hb: hb * 1.05,
            ('HV', 'HB'): lambda hv: hv / 1.05,
            ('HB', 'HRC'): lambda hb: 88 * np.tanh(hb/1700),
            ('HRC', 'HB'): lambda hrc: 1700 * np.arctanh(hrc/88),
            ('HV', 'HRC'): lambda hv: 88 * np.tanh(hv/1785),
            ('HRC', 'HV'): lambda hrc: 1785 * np.arctanh(hrc/88),
            ('HB', 'MPa'): lambda hb: hb * 3.45,  # Approximate tensile strength
            ('HV', 'MPa'): lambda hv: hv * 3.2,
            ('HRC', 'MPa'): lambda hrc: 3.45 * 1700 * np.arctanh(hrc/88)
        }
        
        key = (from_scale.upper(), to_scale.upper())
        if key in conversions:
            return conversions[key](hardness_value)
        elif from_scale.upper() == to_scale.upper():
            return hardness_value
        else:
            # Try indirect conversion through HB
            try:
                hb = conversions[(from_scale.upper(), 'HB')](hardness_value)
                return conversions[('HB', to_scale.upper())](hb)
            except:
                raise InvalidInputError(
                    "hardness conversion",
                    f"from {from_scale} to {to_scale}",
                    "supported conversion path"
                )
    
    def material_selection_index(self, requirement: str,
                               materials: List[MaterialProperties]) -> List[Tuple[str, float]]:
        """Calculate material selection indices for different requirements"""
        indices = []
        
        for mat in materials:
            if requirement == 'minimum_weight_tension':
                # Maximize E/ρ for stiffness-limited design
                index = mat.youngs_modulus / mat.density
            elif requirement == 'minimum_weight_bending':
                # Maximize E^(1/2)/ρ for bending stiffness
                index = np.sqrt(mat.youngs_modulus) / mat.density
            elif requirement == 'minimum_cost_stiffness':
                # Simplified - would need cost data
                # Using density as proxy for cost
                index = mat.youngs_modulus / (mat.density ** 2)
            elif requirement == 'maximum_damping':
                # Loss coefficient - simplified estimation
                if mat.youngs_modulus > 0:
                    index = 1 / np.sqrt(mat.youngs_modulus)
                else:
                    index = 0
            elif requirement == 'thermal_shock_resistance':
                # Maximize σf·k/(E·α)
                if mat.thermal_conductivity and mat.thermal_expansion:
                    index = (mat.ultimate_strength * mat.thermal_conductivity) / \
                           (mat.youngs_modulus * mat.thermal_expansion)
                else:
                    index = 0
            else:
                index = 0
            
            indices.append((mat.name, index))
        
        # Sort by index (descending)
        indices.sort(key=lambda x: x[1], reverse=True)
        
        return indices