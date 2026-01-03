"""Engineering-specific computation modules for Koïos."""

from core.engineering.fea_engine import FiniteElementAnalysisEngine
from core.engineering.cfd_engine import ComputationalFluidDynamicsEngine
from core.engineering.electromagnetics_engine import ElectromagneticsEngine
from core.engineering.material_science_engine import MaterialScienceEngine
from core.engineering.advanced_material_science import AdvancedMaterialScienceEngine

__all__ = [
    'FiniteElementAnalysisEngine',
    'ComputationalFluidDynamicsEngine',
    'ElectromagneticsEngine',
    'MaterialScienceEngine',
    'AdvancedMaterialScienceEngine',
]
