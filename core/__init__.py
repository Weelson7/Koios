"""Core modules package for Koïos mathematical and engineering computation."""

# Import commonly used classes for convenient access
from core.calculation_engine import CalculationEngine
from core.calculus_engine import CalculusEngine
from core.expression_parser import ExpressionParser
from core.matrix_operations import MatrixOperations
from core.complex_analysis_engine import ComplexAnalysisEngine
from core.tensor_calculus_engine import TensorCalculusEngine
from core.transforms_series_engine import TransformsSeriesEngine
from core.ode_solver import ODESolver
from core.advanced_ode_solver import AdvancedODESolver
from core.advanced_integration_engine import AdvancedIntegrationEngine
from core.numerical_methods_engine import NumericalMethodsEngine
from core.optimization_algorithms_engine import OptimizationEngine
from core.physics_simulator import PhysicsSimulator
from core.statistical_mechanics_engine import StatisticalMechanicsEngine

__all__ = [
    'CalculationEngine',
    'CalculusEngine',
    'ExpressionParser',
    'MatrixOperations',
    'ComplexAnalysisEngine',
    'TensorCalculusEngine',
    'TransformsSeriesEngine',
    'ODESolver',
    'AdvancedODESolver',
    'AdvancedIntegrationEngine',
    'NumericalMethodsEngine',
    'OptimizationEngine',
    'PhysicsSimulator',
    'StatisticalMechanicsEngine',
]
